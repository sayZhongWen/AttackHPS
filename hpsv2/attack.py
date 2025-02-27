from cProfile import label
import os
import json
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import requests
from clint.textui import progress
import huggingface_hub

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.src.training.train import calc_ImageReward, inversion_score
from hpsv2.src.training.data import ImageRewardDataset, collate_rank, RankingDataset
from hpsv2.utils import root_path, hps_version_map
from hpsv2.torchattack import PGD, FGSM, PIFGSMPP
import matplotlib.pyplot as plt
from hpsv2.evaluation import BenchmarkDataset, collate_eval, initialize_model
from hpsv2.src.open_clip.tokenizer import _tokenizer

from torchvision import transforms
from hpsv2.src.open_clip import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from typing import Optional, Tuple

def inverse_image_transform(
    transformed_image: torch.Tensor,
    mean: Tuple[float, ...] = OPENAI_DATASET_MEAN,
    std: Tuple[float, ...] = OPENAI_DATASET_STD,
    original_size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    inv_tensor = transformed_image.clone()

    # [normalized = (input - mean) / std] => [input = normalized * std + mean]
    for c in range(inv_tensor.shape[0]):
        inv_tensor[c] = inv_tensor[c] * std[c] + mean[c]
    
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(inv_tensor)
    
    if original_size is not None:
        pil_image = pil_image.resize(original_size[::-1], Image.BICUBIC)  # (W, H) -> (H, W)
    
    return pil_image


def attack_benchmark(data_path, img_path, model, batch_size, preprocess_val, tokenizer, device, attacker="PGD"):
    meta_dir = data_path
    style_list = os.listdir(img_path)
    model_id = img_path.split('/')[-1]

    score = {}
    
    score[model_id]={}
    
    if attacker == "PGD":
        atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    elif attacker == "FGSM":
        atk = FGSM(model, eps=4/255)
    elif attacker == "PIFGSMPP":
        atk = PIFGSMPP(model)
    else:
        raise ValueError('Attacker not supported!')
    atk.set_normalization_used(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)

    for style in style_list:
        # score[model_id][style] = [0] * 10
        score[model_id][style] = []
        image_folder = os.path.join(img_path, style)
        meta_file = os.path.join(meta_dir, f'{style}.json')
        dataset = BenchmarkDataset(meta_file, image_folder, preprocess_val, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eval)

        for i, batch in enumerate(tqdm(dataloader, desc=f'{model_id} {style}')):
            images, texts = batch
            
            images = images.to(device=device, non_blocking=True)
            texts = texts.to(device=device, non_blocking=True)

            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = model(images, texts)
                if isinstance(outputs, tuple):
                    logits_per_image, logits_per_text = outputs
                else:
                    image_features, text_features = outputs["image_features"], outputs["text_features"]
                    logits_per_image = image_features @ text_features.T * 100
                
                print("Before", attacker, "attack: ", logits_per_image)
                # Show the first (0, 1) images and corresponding texts in the same figure in the batch before attack
                plt.figure()
                plt.suptitle('Before attack')
                for j in range(2):
                    plt.subplot(2, 1, j+1)
                    plt.imshow(inverse_image_transform(images[j]))
                    text = _tokenizer.decode(texts.cpu().numpy()[j])
                    text = text.split('<end_of_text>')[0].split('<start_of_text>')[-1]
                    text += f' {torch.diagonal(logits_per_image)[j].cpu().item():.2f}'
                    plt.title(text, fontsize=10)
                # plt.show()
                # plt.savefig(f'attacked_imgs/before_{i:05d}.png')
                
                images = atk(images, texts, labels=torch.arange(images.shape[0]))
                with torch.no_grad():
                    outputs_after_attack = model(images, texts)
                if isinstance(outputs_after_attack, tuple):
                    logits_per_image, logits_per_text = outputs_after_attack
                else:
                    image_features, text_features = outputs_after_attack["image_features"], outputs_after_attack["text_features"]
                    logits_per_image = image_features @ text_features.T * 100
                
                print("After attack: ", logits_per_image)
                # Show the first (0, 1) images and corresponding texts in the same figure in the batch after attack
                plt.figure()
                plt.suptitle('After attack')
                for j in range(2):
                    plt.subplot(2, 1, j+1)
                    atk_image = inverse_image_transform(images[j])
                    plt.imshow(atk_image)
                    atk_image.save(f'attacked_imgs/{i:05d}_{j:05d}.png')
                    text = _tokenizer.decode(texts.cpu().numpy()[j])
                    text = text.split('<end_of_text>')[0].split('<start_of_text>')[-1]
                    text += f' {torch.diagonal(logits_per_image)[j].cpu().item():.2f}'
                    plt.title(text, fontsize=10)
                plt.show()
                # plt.savefig(f'attacked_imgs/after_{i:05d}.png')

                
            # score[model_id][style][i] = torch.sum(torch.diagonal(logits_per_image)).cpu().item() / 80
            score[model_id][style].extend(torch.diagonal(logits_per_image).cpu().tolist())
        print(len(score[model_id][style]))
    print('-----------benchmark score ---------------- ')
    for model_id, data in score.items():
        all_score = []
        for style , res in data.items():
            avg_score = [np.mean(res[i:i+80]) for i in range(0, len(res), 80)]
            all_score.extend(res)
            print(model_id, '{:<15}'.format(style), '{:.2f}'.format(np.mean(avg_score)), '\t', '{:.4f}'.format(np.std(avg_score)))
        print(model_id, '{:<15}'.format('Average'), '{:.2f}'.format(np.mean(all_score)), '\t')

model_dict = {}
model_name = "ViT-H-14"
precision = 'amp'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def attack(mode: str, root_dir: str, data_path: str = os.path.join(root_path,'datasets/benchmark'), checkpoint_path: str = None, batch_size: int = 20, hps_version: str = "v2.1") -> None:
    
    # check if the default checkpoint exists
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    if checkpoint_path is None:
        checkpoint_path = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    
    initialize_model()
    model = model_dict['model']
    preprocess_val = model_dict['preprocess_val']

    print('Loading model ...')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    tokenizer = get_tokenizer(model_name)
    model = model.to(device)
    # model.eval()
    print('Loading model successfully!')
    
    
    if mode == 'benchmark':
        attack_benchmark(data_path, root_dir, model, batch_size, preprocess_val, tokenizer, device)
    else:
        raise NotImplementedError

def attack_rank(data_path, image_folder, model, batch_size, preprocess_val, tokenizer, device, attacker="PGD"):
    meta_file = data_path + '/test.json'
    dataset = RankingDataset(meta_file, image_folder, preprocess_val, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_rank)
    
    if attacker == "PGD":
        atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
    elif attacker == "FGSM":
        atk = FGSM(model, eps=8/255)
    elif attacker == "PIFGSMPP":
        atk = PIFGSMPP(model)
    else:
        raise ValueError('Attacker not supported!')
    atk.set_normalization_used(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)

    score = 0
    total = len(dataset)
    all_rankings = []
    attacked_all_rankings = []
    attacked_score = 0
    for batch in tqdm(dataloader):
        images, num_images, labels, texts = batch
        images = images.to(device=device, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        num_images = num_images.to(device=device, non_blocking=True)
        labels = labels.to(device=device, non_blocking=True)
        attack_labels = torch.argmin(labels).unsqueeze(0)
        binary_labels = torch.where(labels == labels.min(), torch.ones_like(labels, dtype=torch.float), torch.zeros_like(labels, dtype=torch.float)).unsqueeze(1)
        with torch.cuda.amp.autocast():
            outputs = model(images, texts)
            if isinstance(outputs, tuple):
                logits_per_image, logits_per_text = outputs
            else:
                image_features, text_features, logit_scale = outputs["image_features"], outputs["text_features"], outputs["logit_scale"]
                logits_per_image = logit_scale * image_features @ text_features.T
            paired_logits_list = [logit[:,i] for i, logit in enumerate(logits_per_image.split(num_images.tolist()))]
            print("Before", attacker, "attack: ", logits_per_image)
            # Show the first (0, 1) images and corresponding texts in the same figure in the batch before attack
            plt.figure(figsize=(30, 10))
            plt.suptitle('Before attack')
            for j in range(images.shape[0]):
                plt.subplot(images.shape[0], 1, j+1)
                plt.imshow(inverse_image_transform(images[j]))
                text = _tokenizer.decode(texts.cpu().numpy()[0])
                text = text.split('<end_of_text>')[0].split('<start_of_text>')[-1]
                text += f' {logits_per_image[j].cpu().item():.2f}'
                # Binary label
                text += f' {binary_labels[j][0].cpu().item()}'
                plt.title(text, fontsize=10)
            plt.show()
            images = atk(images, texts, labels=attack_labels)
            # # Save the attacked images
            # for i in range(images.shape[0]):
            #     img = inverse_image_transform(images[i])
            #     if not os.path.exists('attacked_imgs'):
            #         os.makedirs('attacked_imgs')
            #     img.save(f'attacked_imgs/{i:05d}.png')
            
            with torch.no_grad():
                outputs_after_attack = model(images, texts)
            if isinstance(outputs_after_attack, tuple):
                logits_per_image, logits_per_text = outputs_after_attack
            else:
                image_features, text_features = outputs_after_attack["image_features"], outputs_after_attack["text_features"]
                logits_per_image = image_features @ text_features.T * 100
            attacked_paired_logits_list = [logit[:,i] for i, logit in enumerate(logits_per_image.split(num_images.tolist()))]
            print("Before attack: ", logits_per_image)
            # Show the first (0, 1) images and corresponding texts in the same figure in the batch before attack
            plt.figure(figsize=(30, 10))
            plt.suptitle('After attack')
            for j in range(images.shape[0]):
                plt.subplot(images.shape[0], 1, j+1)
                plt.imshow(inverse_image_transform(images[j]))
                text = _tokenizer.decode(texts.cpu().numpy()[0])
                text = text.split('<end_of_text>')[0].split('<start_of_text>')[-1]
                text += f' {logits_per_image[j].cpu().item():.2f}'
                # Binary label
                text += f' {binary_labels[j][0].cpu().item()}'
                plt.title(text, fontsize=10)
            plt.show()
            
        predicted = [torch.argsort(-k) for k in paired_logits_list]
        hps_ranking = [[predicted[i].tolist().index(j) for j in range(n)] for i,n in enumerate(num_images)]
        attacked_predicted = [torch.argsort(-k) for k in attacked_paired_logits_list]
        attacked_hps_ranking = [[attacked_predicted[i].tolist().index(j) for j in range(n)] for i,n in enumerate(num_images)]
        labels = [label for label in labels.split(num_images.tolist())]
        all_rankings.extend(hps_ranking)
        score += sum([inversion_score(hps_ranking[i], labels[i]) for i in range(len(hps_ranking))])
        attacked_all_rankings.extend(attacked_hps_ranking)
        attacked_score += sum([inversion_score(attacked_hps_ranking[i], labels[i]) for i in range(len(attacked_hps_ranking))])
    print('ranking_acc:', score/total)
    if not os.path.exists('logs'):
        os.makedirs('logs')
    with open('logs/hps_rank.json', 'w') as f:
        json.dump(all_rankings, f)
    
    print('attacked_ranking_acc:', attacked_score/total)
    with open('logs/attacked_hps_rank.json', 'w') as f:
        json.dump(attacked_all_rankings, f)
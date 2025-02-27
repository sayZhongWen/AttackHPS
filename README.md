# Attack HPS
## Introduction
Text-to-image generation models, such as DALL-E, Stable Diffusion, and MidJourney, have made significant progress in generating high-quality images from textual descriptions. However, aligning these models with human preferences remains a critical challenge. Reinforcement Learning from Human Feedback (RLHF) has emerged as a promising approach to address this issue. By training reward models on large-scale human feedback datasets, such as the Human Preference Dataset (HPD), these models can score generated images based on their alignment with human preferences.

Despite the success of RLHF, the robustness of reward models is not well understood. Specifically:

- Adversarial Robustness: Can small, carefully crafted perturbations to an image significantly alter its score in a reward model?

- Transferability of Attacks: Can adversarial perturbations optimized for one reward model (e.g., HPS v2) transfer to another (e.g., HPS v1)?

- Black-Box Attacks: Can reward models deployed as black-box services be fooled by query-based adversarial attacks?

Understanding these questions is crucial for ensuring the reliability and fairness of reward models in real-world applications. This task aims to explore these aspects by analyzing the robustness of reward models, particularly focusing on adversarial attacks and their implications.

## Quick Start
You can download the repo by running
```
git clone https://github.com/sayZhongWen/AttackHPS.git
cd AttackHPSv2
```

In the file `attack.py`, you can modify the data path and attack parameters for more experiments. In addition to the original mode of HPS, we designed `attack_benchmark` for attacking benchmark data and `attack_test` for attacking ranks of test data. Here's the example:
```
hpsv2.evaluate_benchmark('SDXL Refiner 0.9', mode="attack_benchmark", batch_size=2, save_path="./cache_data/", attacker="PGD", hps_version="v2.1")
```

You can also modify the parameters in the attacking function in `hpsv2/attack.py` like:
```
atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
```

## To-do
For future work, it's crucial to encapsulate attack parameters within the evaluate_benchmark function to streamline the attack process and make parameter adjustments and result assessments more straightforward. This oversight in the initial stages of the project led to considerable difficulties when investigating the transferability of attacks, highlighting the need for a more systematic approach to tool development and parameter management from the outset.

## Reference
[1] [Human Preference Score v2: A Solid Benchmark for Evaluating Human Preferences of Text-to-Image Synthesis](https://github.com/tgxs002/HPSv2)

[2] [Human Preference Score: Better Aligning Text-to-Image Models with Human Preference](https://github.com/tgxs002/align_sd)

[3] [torch_attacks](https://github.com/Harry24k/adversarial-attacks-pytorch)
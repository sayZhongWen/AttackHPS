import hpsv2
import hpsv2.attack

print(hpsv2.get_available_models()) # Get models that have access to data
# Available models:
# ['ChilloutMix (Size: 138.86MB)', 'CogView2 (Size: 201.43MB)', 'DALL·E mini (Size: 42.09MB)', 'DALL·E 2 (Size: 132.02MB)', 'DeepFloyd-XL (Size: 231.52MB)', 'Dreamlike Photoreal 2.0 (Size: 306.91MB)', 'Deliberate (Size: 129.43MB)', 'Epic Diffusion (Size: 156.67MB)', 'FuseDream (Size: 138.62MB)', 'Latent Diffusion (Size: 35.31MB)', 'LAFITE (Size: 38.23MB)', 'MajicMix Realistic (Size: 114.97MB)', 'Openjourney (Size: 145.88MB)', 'Realistic Vision (Size: 132.95MB)', 'SDXL Base 0.9 (Size: 381.02MB)', 'SDXL Refiner 0.9 (Size: 372.95MB)', 'Versatile Diffusion (Size: 195.29MB)', 'VQ-Diffusion (Size: 34.94MB)', 'VQGAN + CLIP (Size: 179.41MB)', 'GLIDE (Size: 23.88MB)', 'Stable Diffusion v1.4 (Size: 150.69MB)', 'Stable Diffusion v2.0 (Size: 141.5MB)']  

# Prompt: generate a image of a student studying
# Prompt -> Generative Model A -> Image A
# Prompt -> Generative Model B -> Image B
# Human label: let a human judge which image is better, e.g., judger thinks Image A is better
# HPSv2: Input (Prompt, Image A) -> Score A (Human preference score); (Prompt, Image B) -> Score B (Human preference score)
# If Score A > Score B, then Image A is better than Image B -> Generative Model A is better than Generative Model B

# Test the evaluate function
# hpsv2.evaluate_benchmark('GLIDE', save_path="./cache_data/")
# Result:
# -----------benchmark score ----------------
# glide anime           13.80      0.4514
# glide concept-art     13.36      0.4872
# glide paintings       13.81      0.3987
# glide photo           16.57      0.5736
# glide Average         14.39


# Now let's attack
hpsv2.evaluate_benchmark('SDXL Refiner 0.9', mode="attack_benchmark", batch_size=2, save_path="./cache_data/", attacker="PGD", hps_version="v2.1")
# hpsv2.evaluate_benchmark('SDXL Base 0.9', mode="attack_benchmark", batch_size=2, save_path="./cache_data/", attacker="FGSM")
# hpsv2.evaluate_benchmark('GLIDE', mode="attack_benchmark", batch_size=2, save_path="./cache_data/", attacker="PIFGSMPP")

# from hpsv2 import evaluation as eval
# eval.evaluate(mode="attack_test", data_path=".", root_dir=".\\test", batch_size=1, hps_version="v2.1")


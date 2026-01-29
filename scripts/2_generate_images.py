# ================================================================
# 👩 HERS: Hidden-Pattern Expert Learning for
#          Risk-Specific Vehicle Damage Adaptation
#          in Diffusion Models
#
# Author: Teerapong Panboonyuen
# Affiliation: PBY Laboratory
# Project Page: https://kaopanboonyuen.github.io/HERS
#
# Description:
#   This script is part of the official implementation of HERS,
#   a self-supervised framework for adapting text-to-image diffusion
#   models to generate semantically faithful, risk-aware vehicle
#   damage images for safety-critical domains such as auto insurance.
#
#   HERS trains domain-specific LoRA experts from automatically
#   generated image–text pairs and merges them into a unified
#   diffusion model without manual annotation or inference-time routing.
#
# Disclaimer:
#   This code is released for research, reproducibility, and
#   fraud-awareness purposes only. Any deployment in real-world
#   insurance systems must comply with legal, ethical, and
#   regulatory requirements.
#
# ================================================================

import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

pipe.load_lora_weights("outputs/merged_model")

prompt = "rear bumper dent with cracked paint near taillight"
image = pipe(prompt).images[0]
image.save("outputs/samples/hers_result.png")
import json, os, torch
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

with open("data/prompts/prompts.json") as f:
    prompts = json.load(f)

os.makedirs("data/images", exist_ok=True)

for damage, plist in prompts.items():
    out_dir = f"data/images/{damage}"
    os.makedirs(out_dir, exist_ok=True)
    for i, p in enumerate(tqdm(plist)):
        image = pipe(p, guidance_scale=7.5, num_inference_steps=50).images[0]
        image.save(f"{out_dir}/{i:05d}.png")
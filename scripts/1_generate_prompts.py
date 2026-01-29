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
from hers.prompt_engine import PromptGenerator
from omegaconf import OmegaConf
import json, os

cfg = OmegaConf.load("configs/prompt_generation.yaml")
generator = PromptGenerator(cfg)

prompts = generator.generate_all()

os.makedirs("data/prompts", exist_ok=True)
with open("data/prompts/prompts.json", "w") as f:
    json.dump(prompts, f, indent=2)

print(f"[HERS] Generated {len(prompts)} damage-aware prompts")
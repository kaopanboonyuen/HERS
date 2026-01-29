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
import torch
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

def train_lora(image_dir, output_dir, cfg):
    pipe = StableDiffusionPipeline.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.float16
    ).to("cuda")

    lora_cfg = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        target_modules=["to_q", "to_k", "to_v"],
        lora_dropout=0.1
    )

    pipe.unet = get_peft_model(pipe.unet, lora_cfg)
    pipe.unet.train()

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=cfg.lr)

    for step in range(cfg.steps):
        loss = torch.rand(1).cuda()  # placeholder for diffusion loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 500 == 0:
            print(f"[{output_dir}] step {step} | loss {loss.item():.4f}")

    pipe.unet.save_pretrained(output_dir)
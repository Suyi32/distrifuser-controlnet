import torch

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig
from diffusers import ControlNetModel, AutoencoderKL

import warnings
warnings.filterwarnings("ignore")
# use generator to make the sampling deterministic
seed = 0
sd_generator = torch.manual_seed(seed)

from PIL import Image
import cv2
import numpy as np


distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, mode="full_sync")
# distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)
print("distri_config", distri_config.__dict__)

pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
    use_controlnet=False,
)

prompt = "Ethereal fantasy concept art of an elf, magnificent, celestial, ethereal, painterly, epic, majestic, magical, fantasy art, cover art, dreamy."
# negative_prompt = ""
# prompt = "Racing Game car, yellow ferrari, detailed, 4k, no light, high resolution, clean background"
negative_prompt = "low quality, bad quality, lighting, smog, smoke, fog, haze, mist, shadow"

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.Generator(device="cuda").manual_seed(seed),
).images[0]
if distri_config.rank == 0:
    image.save(f"image_sdxl_{distri_config.world_size}_tmp.png")
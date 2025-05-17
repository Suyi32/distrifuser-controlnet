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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_loras", type=int, default=1, choices=[0,1,2], help="max lora num")
parser.add_argument("--num_controlnets", type=int, default=1, choices=[0,1,2], help="num controlnets")
serve_args = parser.parse_args() 
print("Args", serve_args)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)


# distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, mode="full_sync")
# distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, use_cuda_graph=False)
distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)
print("distri_config", distri_config.__dict__)

pipeline = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
    num_controlnets=serve_args.num_controlnets,
    num_loras=serve_args.num_loras,
    vae=vae,
)

ref_image = cv2.imread("./ref_images/ferrari_ref.png")
ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)

ref_image = ref_image[:, :, None]
ref_image = np.concatenate([ref_image, ref_image, ref_image], axis=2)
ref_image = Image.fromarray(ref_image)


prompt = "Racing Game car, yellow ferrari, detailed, 4k, no light, high resolution, clean background"
prompt = "papercut -subject/scene-" + prompt
negative_prompt = "low quality, bad quality, lighting, smog, smoke, fog, haze, mist, shadow"

pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=ref_image,
    controlnet_conditioning_scale=0.5,
    generator=torch.Generator(device="cuda").manual_seed(seed),
    serve_args=serve_args,
).images[0]
if distri_config.rank == 0:
    image.save(f"image_sdxl_controlnet_{distri_config.world_size}_tmp.png")
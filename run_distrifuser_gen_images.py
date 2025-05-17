import torch

from distrifuser.pipelines import DistriSDXLPipeline
from distrifuser.utils import DistriConfig
from diffusers import ControlNetModel, AutoencoderKL

import warnings
warnings.filterwarnings("ignore")
# use generator to make the sampling deterministic
seed = 0
sd_generator = torch.manual_seed(seed)

from benchmark_utils import read_prompts, process_prompt
from PIL import Image
import cv2
import numpy as np
import argparse
import os
import time

parser = argparse.ArgumentParser()
parser.add_argument("--num_loras", type=int, default=1, choices=[0,1,2], help="max lora num")
parser.add_argument("--num_controlnets", type=int, default=1, choices=[0,1,2,3], help="num controlnets")
serve_args = parser.parse_args() 
print("Args", serve_args)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)


# distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4, mode="full_sync")
distri_config = DistriConfig(height=1024, width=1024, warmup_steps=4)
print("distri_config", distri_config.__dict__)

# create the pipeline
pipeline, load_lora_time = DistriSDXLPipeline.from_pretrained(
    distri_config=distri_config,
    pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16",
    use_safetensors=True,
    num_controlnets=serve_args.num_controlnets,
    num_loras=serve_args.num_loras,
    vae=vae,
)
pipeline.set_progress_bar_config(disable=distri_config.rank != 0)

##################### Inference #####################
output_image_folder = f"/project/infattllm/slida/distrifuser/distrifuser_images/all_images_distrifuser_controlnets_{serve_args.num_controlnets}_loras_{serve_args.num_loras}_wrold_{distri_config.world_size}_mode_{distri_config.mode}"
if not os.path.isdir(output_image_folder):
    os.mkdir(output_image_folder)


prompts = read_prompts(num_prompts=600)
if serve_args.num_loras == 1:
    prompt_prefix = "papercut -subject/scene-"
elif serve_args.num_loras == 2:
    prompt_prefix = "by william eggleston, "
elif serve_args.num_loras == 0:
    prompt_prefix = ""
prompt_suffix = ", 4k, clean background"
negative_prompt = "low quality, bad quality, sketches, numbers, letters"
ref_image_folder = "/home/slida/DF-Serving/PartiPrompts_Detail_eval/images_sdxl_t2i"
assert os.path.isdir(ref_image_folder), "ref image folder not exists"

total_inference_time = 0.0
sd_generator = torch.Generator(device="cuda").manual_seed(seed)
for prompt_id, prompt in enumerate(prompts):
    if prompt_id == 1:
        e2e_serving_start = time.time()
    # Process prompt
    prompt = process_prompt(prompt_prefix, prompt, prompt_suffix)
    print(prompt_id, prompt)

    # create the pipeline
    if prompt_id % 5 == 0:
        del pipeline
        torch.cuda.empty_cache()
        pipeline, load_lora_time = DistriSDXLPipeline.from_pretrained(
            distri_config=distri_config,
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            variant="fp16",
            use_safetensors=True,
            num_controlnets=serve_args.num_controlnets,
            num_loras=serve_args.num_loras,
            vae=vae,
        )
        pipeline.set_progress_bar_config(disable=distri_config.rank != 0)
    
    # Load ref image
    ref_image = cv2.imread(f"{ref_image_folder}/image_{prompt_id}_depth.png")
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    ref_image = ref_image[:, :, None]
    ref_image = np.concatenate([ref_image, ref_image, ref_image], axis=2)
    ref_image = Image.fromarray(ref_image)

    inference_start = time.time()
    ##################### Load LoRAs #####################    

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=ref_image,
        controlnet_conditioning_scale=0.5,
        generator=sd_generator,
        serve_args=serve_args,
    ).images[0]
    
    inference_end = time.time()
    if distri_config.rank == 0:
        print("Load LoRA latency: {:.2f}".format(load_lora_time))
        print("End2End inference latency: {:.2f}".format(inference_end - inference_start + load_lora_time))
        print("==============================")
        image.save(f"{output_image_folder}/image_{prompt_id}.png")
    total_inference_time += inference_end - inference_start + load_lora_time

    # ### Delete the pipeline ###
    # del pipeline
    # torch.cuda.empty_cache()

e2e_serving_end = time.time()
if distri_config.rank == 0:
    # print("End2End inference latency: {:.2f}".format(e2e_serving_end - e2e_serving_start))
    print("Number of prompts: {}".format(len(prompts) - 1))
    print("End2End Throughput: {:.2f}".format( (len(prompts) - 1) / (total_inference_time*distri_config.world_size) ))
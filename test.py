from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler
)
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
from dinov2 import Dinov2ModelwOutput
import os
import requests
from diffusers import DDIMPipeline
import torch
from transformers.models.clip import CLIPTextModel
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from visualization import visualizer
import shutil

from scheduler import DDIMSchedulerWithViT
from pipeline import StableDiffusionPipelineWithViT

class configs:
    def __init__(self):
        self.model_path = 'facebook/dinov2-large'
        self.link = "runwayml/stable-diffusion-v1-5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.link, subfolder="tokenizer")
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.size = [self.processor.crop_size["height"], self.processor.crop_size["width"]]
        self.mean, self.std = self.processor.image_mean, self.processor.image_std
        self.mean = torch.tensor(self.mean, device="cuda")
        self.std = torch.tensor(self.std, device="cuda")
        self.prompt = "a high-quality image"
        self.seed = 20
        self.num_inference_steps = 5
        self.vit = Dinov2ModelwOutput.from_pretrained(self.model_path)
        self.vae = AutoencoderKL.from_pretrained(self.link, subfolder="vae")
        self.text_encoder = CLIPTextModel.from_pretrained(self.link, subfolder="text_encoder")
        self.unet = UNet2DConditionModel.from_pretrained(self.link, subfolder="unet")
        self.scheduler = DDIMSchedulerWithViT.from_pretrained(self.link, subfolder="scheduler")
        self.layeridx = [0]
        self.head = [0]
        self.guidance_strength = [0.0]
        self.guidance_range = [0, 150]
        self.outputdir = "./outputs" 
        self.metricoutputdir = "./metrics"
        self.qkvoutputdir = "./qkv"
        self.trials = 1
        

configs = configs()
visualizer = visualizer()

# clear the result from last experiment
if 'outputs' not in os.listdir("./"):
    os.mkdir(configs.outputdir)
    
#if os.listdir(configs.outputdir) is not None:
#    shutil.rmtree(configs.outputdir)

pipe = StableDiffusionPipelineWithViT(
    vit=configs.vit,
    vae=configs.vae,
    tokenizer = configs.tokenizer,
    feature_extractor = None,
    text_encoder=configs.text_encoder,
    unet=configs.unet,
    scheduler=configs.scheduler,
    safety_checker=None,
).to(device="cuda")

rng = torch.Generator("cuda")

for layer in configs.layeridx:
    for head in configs.head:
        for strength in configs.guidance_strength:
            for counter in range(configs.trials):
                seed = configs.seed + counter
                rng.manual_seed(seed)
                foldername = "layer_" + str(layer) + "_head_" + str(head)
                if foldername not in os.listdir(configs.outputdir):
                    os.mkdir(configs.outputdir + "/" + foldername)
                image, _, attention_mean_per_timestep, entropy_per_timestep, images_per_timestep, allqkv = \
                    pipe(configs.prompt,
                        num_inference_steps=configs.num_inference_steps,
                        generator=rng,
                        vit_input_size=configs.size,
                        vit_input_mean=configs.mean
                        vit_input_std=configs.std,
                        layer_idx=layer,
                        head_idx=head,
                        guidance_strength=strength,
                        return_dict=False)
                image[0].save(configs.outputdir + "/" + foldername + "/" + "trial_" + str(counter+1) + f"_{strength}.png")
                entropy_numpy = [e[0].cpu().data.numpy() for e in entropy_per_timestep]
                visualizer.plot(entropy_numpy, configs.metricoutputdir + "/" + "entropy-layer_" + str(layer) + "_head_" + str(head) + "_trial_" + str(counter+1) + f"_{strength}.png")
                print("all qkv ", len(allqkv))

                #visualizer.plot_qkv(allqkv, iteration, configs.layer_idx, configs.head_idx)
                ##visualizer.plot(attention_mean_per_timestep.cpu().data.numpy() , configs.metricoutputdir + "/" + "attention-layer_" + str(layer) + "_head_" + str(head) + "_trial_" + str(counter+1) + f"_{strength}.png")




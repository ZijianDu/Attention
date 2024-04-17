from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler
)
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
from dinov2 import Dinov2ModelwOutput
import os
import numpy as np
import requests
from diffusers import DDIMPipeline
import torch
from transformers.models.clip import CLIPTextModel
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from visualization import visualizer
import shutil
from processor import processor
from scheduler import DDIMSchedulerWithViT
from pipeline import StableDiffusionPipelineWithViT
from skimage.transform import resize
from visualization import HeatMap
from dataclasses import dataclass
@dataclass
class configs:
    model_path = 'facebook/dinov2-large'
    link = "runwayml/stable-diffusion-v1-5"
    tokenizer = AutoTokenizer.from_pretrained(link, subfolder="tokenizer")
    processor = AutoImageProcessor.from_pretrained(model_path)
    size = [processor.crop_size["height"], processor.crop_size["width"]]
    mean, std = processor.image_mean, processor.image_std
    mean = torch.tensor(mean, device="cuda")
    std = torch.tensor(std, device="cuda")
    prompt = "a high-quality image"
    seed = 20
    num_inference_steps = 5
    vit = Dinov2ModelwOutput.from_pretrained(model_path)
    vae = AutoencoderKL.from_pretrained(link, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(link, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(link, subfolder="unet")
    scheduler = DDIMSchedulerWithViT.from_pretrained(link, subfolder="scheduler")
    layeridx = [0]
    head = [0]
    guidance_strength = [0.0]
    guidance_range = [0, 150]
    num_tokens = 256
    outputdir = "./outputs" 
    metricoutputdir = "./metrics"
    qkvoutputdir = "./qkv"
    trials = 1
    num_pcs = 3
    image_size = 224
        

configs = configs()
visualizer = visualizer()
processor = processor()

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

def run_attention_guided_diffusion():
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
                            vit_input_mean=configs.mean,
                            vit_input_std=configs.std,
                            layer_idx=layer,
                            head_idx=head,
                            guidance_strength=strength,
                            return_dict=False)
                    image[0].save(configs.outputdir + "/" + foldername + "/" + "trial_" + str(counter+1) + f"_{strength}.png")
                    #entropy_numpy = [e[0].cpu().data.numpy() for e in entropy_per_timestep]
                    #visualizer.plot(entropy_numpy, configs.metricoutputdir + "/" + "entropy-layer_" + str(layer) + "_head_" + str(head) + "_trial_" + str(counter+1) + f"_{strength}.png")
                    last_iter_key = allqkv[-1][1].squeeze(dim = 0)
                    processor.factor_analysis(last_iter_key, configs.num_pcs, configs.image_size, configs.qkvoutputdir, '/key_layer{0}_head{1}.png'.format(layer, head))
                    #last_key_pca = processor.factor_analysis(last_iter_key, 3, "pca").mean(axis = 0).reshape((int(np.sqrt(configs.num_tokens)), int(np.sqrt(configs.num_tokens))))
                    ##upsampled_last_key_pca = resize(last_key_pca, [224, 224])
                    #print(upsampled_last_ket_pca)
                #visualizer.plot_qkv(allqkv, iteration, configs.layer_idx, configs.head_idx)
                ##visualizer.plot(attention_mean_per_timestep.cpu().data.numpy() , configs.metricoutputdir + "/" + "attention-layer_" + str(layer) + "_head_" + str(head) + "_trial_" + str(counter+1) + f"_{strength}.png")

if __name__ == "__main__":
    run_attention_guided_diffusion()
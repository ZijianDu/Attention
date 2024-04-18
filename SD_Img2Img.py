import torch
import numpy as np
#import torchvision.utility as tvu
from PIL import Image
import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import inspect
from packaging import version
import requests
import os
from dataclasses import dataclass
from torchvision.utils import save_image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers, DDPMScheduler
from scheduler import myddpmscheduler
from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.pipelines.stable_diffusion.pipeline_output import StableDiffusionPipelineOutput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, StableDiffusionMixin
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from pipeline import StableDiffusionImg2ImgPipelineWithSDEdit
from diffusers.utils import (
    PIL_INTERPOLATION,
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)

from torchvision import transforms
from torchvision.utils import save_image



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
    num_inference_steps = 100
    inputdatadir = "/media/data/leo/style_vector_data/"
    class_label = 0
    all_classes_list = os.listdir(inputdatadir)
    all_images_list = os.listdir(inputdatadir + all_classes_list[class_label])
    vae = AutoencoderKL.from_pretrained(link, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(link, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(link, subfolder="unet")
    scheduler = DDPMScheduler.from_pretrained(link, subfolder="scheduler")
    #scheduler = myddpmscheduler.from_pretrained(link, subfolder = "scheduler")
    layeridx = [0]
    head = [0]
    guidance_strength = [0.0]
    guidance_range = [0, 150]
    num_tokens = 256
    trials = 1
    num_pcs = 3
    image_size = 224
    timestep = 50
        


configs = configs()


preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((configs.image_size, configs.image_size)),
        transforms.Normalize([0.5], [0.5]),
    ]
)


# make sure input image is float and in the range of 0 and 1
sample_image = torch.tensor(np.array(Image.open("church.jpg"))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
#sample_image = preprocess(Image.open("church.jpg"))

prompt = "a high quality image"
strengths = [0.1, 0.2, 0.5, 0.7, 0.9, 1]
steps = [50, 200, 500, 700, 1000]
timesteps = []
for step in steps:
    timesteps.append([i for i in range(step-1, -1, -1)])

pipe = StableDiffusionImg2ImgPipeline(vae=configs.vae, text_encoder=configs.text_encoder, 
                                                tokenizer=configs.tokenizer, unet=configs.unet, scheduler=configs.scheduler, 
                                                safety_checker=None, feature_extractor=None, image_encoder=None, requires_safety_checker=False)

for strength in strengths:
    for i in range(len(steps)):
        output = pipe(prompt = prompt, image = sample_image, strength = strength,  timesteps = timesteps[i],  
                      scheduler = configs.scheduler, return_dict= False)
        img_np = np.array(output[0][0])
        image_arr = torch.tensor(img_np).type(torch.uint8).numpy()
        img_pil = Image.fromarray(image_arr)
        img_pil.save("denoised_strength_" + str(strength) + "_steps_" + str(steps[i]) + ".jpg")

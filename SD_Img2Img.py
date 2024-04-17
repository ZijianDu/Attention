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
    #scheduler = DDPMScheduler.from_pretrained(link, subfolder="scheduler")
    scheduler = myddpmscheduler.from_pretrained(link, subfolder = "scheduler")
    layeridx = [0]
    head = [0]
    guidance_strength = [0.0]
    guidance_range = [0, 150]
    num_tokens = 256
    trials = 1
    num_pcs = 3
    image_size = 224
        


configs = configs()
pipe = StableDiffusionImg2ImgPipelineWithSDEdit(vae=configs.vae, text_encoder=configs.text_encoder, 
                                                tokenizer=configs.tokenizer, unet=configs.unet, scheduler=configs.scheduler, 
                                                safety_checker=None, feature_extractor=None, image_encoder=None, requires_safety_checker=False)


image_file_path = configs.inputdatadir + configs.all_classes_list[configs.class_label] + "/" + configs.all_images_list[0]
original_image = Image.open(image_file_path)
original_image.save("originalimage.png")
sample_image = Image.open("church.jpg")
image = torch.from_numpy(np.array(sample_image)).cuda()

image = torch.from_numpy(np.array(original_image)).cuda()

noisy_image = pipe.create_noisy_image(image, 50)
print(noisy_image.shape)

#test = StableDiffusionImg2ImgPipelineWithSDEdit(configs.vae, configs.text_encoder, configs.tokenizer, configs.unet, configs.scheduler, None, None, None, False)

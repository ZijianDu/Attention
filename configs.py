from PIL import Image
import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import inspect
from packaging import version
import requests
import os
from dataclasses import dataclass
from torchvision.utils import save_image
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, AutoImageProcessor, AutoModel, AutoTokenizer
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin, IPAdapterMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection, UNet2DConditionModel
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers, DDPMScheduler, DDIMScheduler, EulerAncestralDiscreteScheduler
from scheduler import DDPMSchedulerwithGuidance
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
from dinov2 import Dinov2ModelwOutput
import numpy as np
import torch
import torch.nn.functional as F
import shutil
from skimage.transform import resize
import torchvision.transforms as transforms 
from transformers import AutoTokenizer, CLIPTextModelWithProjection
from torchvision.transforms import Resize
from vit_feature_extractor import ViTFeature, ViTPipe, ViTScheduler
from pipeline import StableDiffusionImg2ImgPipelineWithSDEdit, StableDiffusionXLImg2ImgPipelineWithViTGuidance,  StableDiffusionXLPipelineWithViTGuidance
from scheduler import DDPMSchedulerwithGuidance, DDIMSchedulerwithGuidance

@dataclass
class runconfigs:
    ## model related
    prompt = "a high-quality image"
    model_path = 'facebook/dinov2-base'
    link = "stabilityai/sdxl-turbo"

    vae = AutoencoderKL.from_pretrained(link, subfolder="vae").to(device="cuda")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    
    tokenizer = AutoTokenizer.from_pretrained(link, subfolder="tokenizer", torch_dtype=torch.float16)
    tokenizer_2 = AutoTokenizer.from_pretrained(link, subfolder="tokenizer", torch_dtype=torch.float16)
    
    processor = AutoImageProcessor.from_pretrained(model_path, torch_dtype=torch.float16)
    improcessor = AutoImageProcessor.from_pretrained(model_path, torch_dtype=torch.float16)
    
    size = [processor.crop_size["height"], processor.crop_size["width"]]
    mean, std = processor.image_mean, processor.image_std
    mean, std = torch.tensor(mean, device="cuda"), torch.tensor(std, device="cuda")
    
    vit = Dinov2ModelwOutput.from_pretrained(model_path, torch_dtype = torch.float16)
    
    # use overridden schedulers
    #ddpmscheduler = DDPMSchedulerwithGuidance.from_pretrained(link, subfolder="scheduler")
    
    ddimscheduler = DDIMSchedulerwithGuidance.from_pretrained(link, subfolder = "scheduler")

    unet = UNet2DConditionModel.from_pretrained(link, subfolder="unet").to(device="cuda")

    stablediffusionxltext2imgpipewithvitguidance =  StableDiffusionXLPipelineWithViTGuidance(
        vit = vit,
        vae = vae,
        text_encoder = text_encoder,
        text_encoder_2 = text_encoder_2,
        tokenizer = tokenizer,
        tokenizer_2 = tokenizer_2,
        unet = unet, 
        scheduler = ddimscheduler,
        image_encoder = None,
        feature_extractor = None,
        force_zeros_for_empty_prompt = True,
        add_watermarker = None)
    
    stablediffusionxlimg2imgpipewithvitguidance = StableDiffusionXLImg2ImgPipelineWithViTGuidance(
        vit = vit, 
        vae = vae, 
        text_encoder = text_encoder, 
        text_encoder_2 = text_encoder_2, 
        tokenizer=tokenizer, 
        tokenizer_2 = tokenizer_2, 
        unet=unet, 
        scheduler=DDIMSchedulerwithGuidance, 
        image_encoder = None, 
        feature_extractor=None, 
        requires_aesthetics_score= False, 
        force_zeros_for_empty_prompt=True, 
        add_watermarker=None)
    
    # model parameters
    guidance_strength = [1]
    #, 20, 26, 28, 30, 32, 34, 36, 40]

    # select the range of reverse diffusion process when guidance is actually applied
    guidance_range = 0.7
  
    # percentage iterations to add noise before denoising, higher means more noise added
    diffusion_strength = [0.35]
                          #, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34]
    # total 24 layers
    layer_idx = [10]
    all_params = []

    for s in diffusion_strength:
        for g in guidance_strength:
            all_params.append([s, g])
    assert len(all_params) == len(diffusion_strength) * len(guidance_strength) 
    
    num_tokens = 256
    image_size = 224

    # number of total iterations, 1000 is maximum, works when the mode is "running"
    num_steps = 200
    # number of random sampling for sweeping
    sweeping_run_count = 200
    
    
    vitscheduler = ViTScheduler()
    imageH, imageW = 224, 224
    
    # total 12 heads
    num_heads = 12
    all_selected_heads = [[i for i in range(12)]]
    current_selected_heads = all_selected_heads[0]

    # choose which feature to look, q: 0 k: 1 v: 2
    qkv_choice = 1
    # Vit has image patch 16x16
    num_patches = 16
    # 64 total qkv channels
    attention_channels = 64
    batch_size = 1
    
    # I/O related
    base_folder = "/home/leo/Documents/GenAI-Vision/attention/"
    single_image_name = "cat.jpg"
    single_image = improcessor(Image.open(base_folder + single_image_name))["pixel_values"][0]

    # outputs
    outputdir = "./debug/" 
    
    sweepingdir = "./sweeping/"

    metricoutputdir = "./metrics/"

@dataclass
class wandbconfigs:
    mode = "running"

    running_project_name = "test for SDXLtext2img "

    sweeping_project_name = "test sweep"

    #wandb configs for sweepinng parameters
    sweep_config = {'method':'random'}
    metric = {
            'name' : 'dist_vgg',
            'goal' : 'minimize'
            }
    sweep_config['metric'] = metric
    parameters_dict =  {}
    sweep_config['parameters'] = parameters_dict
    
    parameters_dict.update(
        {
            'guidance_strength' : {
                'distribution' : 'normal',
                'mu' : 5,
                'sigma' : 3
            },
            'diffusion_strength' : {
                'distribution' : 'normal',
                'mu' : 0.55,
                'sigma' : 0.03
            },
        })
    
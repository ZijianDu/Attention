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
from diffusers.schedulers import KarrasDiffusionSchedulers, DDPMScheduler
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


@dataclass
class sdimg2imgconfigs:
    ## model related
    model_path = 'facebook/dinov2-base'
    link = "runwayml/stable-diffusion-v1-5"
    tokenizer = AutoTokenizer.from_pretrained(link, subfolder="tokenizer", torch_dtype=torch.float16)
    processor = AutoImageProcessor.from_pretrained(model_path, torch_dtype=torch.float16)
    size = [processor.crop_size["height"], processor.crop_size["width"]]
    mean, std = processor.image_mean, processor.image_std
    mean, std = torch.tensor(mean, device="cuda"), torch.tensor(std, device="cuda")
    vit = Dinov2ModelwOutput.from_pretrained(model_path, torch_dtype = torch.float16)
    vits = [Dinov2ModelwOutput.from_pretrained('facebook/dinov2-base', torch_dtype = torch.float16)]
#            Dinov2ModelwOutput.from_pretrained('facebook/dinov2-small', torch_dtype = torch.float16),
 #            Dinov2ModelwOutput.from_pretrained('facebook/dinov2-large', torch_dtype=torch.float16)]
    
    prompt = "a high-quality image of a bird"
    vae = AutoencoderKL.from_pretrained(link, subfolder="vae").to(device="cuda")
    text_encoder = CLIPTextModel.from_pretrained(link, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(link, subfolder="unet").to(device="cuda")
    ddpmscheduler = DDPMSchedulerwithGuidance.from_pretrained(link, subfolder="scheduler")
    # model parameters
    # coefficient before guidance
    guidance_strength = [30]
    #, 20, 26, 28, 30, 32, 34, 36, 40]

    # select the range of reverse diffusion process when guidance is actually applied
    guidance_range = 0.7
  
    # percentage iterations to add noise before denoising, higher means more noise added
    diffusion_strength = [0.56]
                          #, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34]
    # total 24 layers
    layer_idx = [10]
    all_params = []
    for vitidx in range(len(vits)):
        for s in diffusion_strength:
            for g in guidance_strength:
                all_params.append([vitidx, s, g])
    assert len(all_params) == len(vits) * len(diffusion_strength) * len(guidance_strength) 
    
    num_tokens = 256
    image_size = 224
    improcessor = AutoImageProcessor.from_pretrained(model_path)
    vitscheduler = ViTScheduler()
    imageH, imageW = 224, 224
    
    # total 12 heads
    num_heads = 12
    all_selected_heads = [[i for i in range(12)], [0], [1], [2], [3], [4], [5], [6], 
                          [7], [8], [9], [10], [11]]
    current_selected_heads = all_selected_heads[0]

    # choose which feature to look, q: 0 k: 1 v: 2
    qkv_choice = 1
    # Vit has image patch 16x16
    num_patches = 16
    # 64 total qkv channels
    attention_channels = 64
    batch_size = 1
    
    # data related
    # read single image
    base_folder = "/home/leo/Documents/GenAI-Vision/attention/"
    single_image_name = "bird.jpg"
    single_image = improcessor(Image.open(base_folder + single_image_name))["pixel_values"][0]
    # sweeping perform parameter sweeping through sampling
    # pre specify parameter combo for running mode
    mode = "sweeping"

    running_project_name = "correct implementation single head guided sd test"

    sweeping_project_name = "test sweep runs 4-29"

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
    
    # number of total iterations, 1000 is maximum, works when the mode is "running"
    num_steps = 500
    # number of random sampling for sweeping
    sweeping_run_count = 200
    
    # outputs
    outputdir = "./debug/" 
    
    sweepingdir = "./sweeping/"

    metricoutputdir = "./metrics/"

    # visualization related
    dpi = 300


# configs to be used for SDXLTurboImg2Img pipeline
@dataclass
class sdxlimg2imgconfigs:
    ## model related
    model_path = 'facebook/dinov2-base'
    link = "stabilityai/sdxl-turbo"
    
    tokenizer = AutoTokenizer.from_pretrained(link, subfolder="tokenizer", torch_dtype=torch.float16)
    tokenizer_2 = tokenizer
    
    processor = AutoImageProcessor.from_pretrained(model_path, torch_dtype=torch.float16)
    size = [processor.crop_size["height"], processor.crop_size["width"]]
    mean, std = processor.image_mean, processor.image_std
    mean, std = torch.tensor(mean, device="cuda"), torch.tensor(std, device="cuda")
    
    vit = Dinov2ModelwOutput.from_pretrained(model_path, torch_dtype = torch.float16)
    
    prompt = "a high-quality image"
    
    vae = AutoencoderKL.from_pretrained(link, subfolder="vae").to(device="cuda")
    
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")
    
    unet = UNet2DConditionModel.from_pretrained(link, subfolder="unet").to(device="cuda")
    ddpmscheduler = DDPMSchedulerwithGuidance.from_pretrained(link, subfolder="scheduler")
    # model parameters
    # coefficient before guidance
    guidance_strength = [30]
    #, 20, 26, 28, 30, 32, 34, 36, 40]

    # select the range of reverse diffusion process when guidance is actually applied
    guidance_range = 0.7
  
    # percentage iterations to add noise before denoising, higher means more noise added
    diffusion_strength = [0.56]
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
    improcessor = AutoImageProcessor.from_pretrained(model_path)
    vitscheduler = ViTScheduler()
    imageH, imageW = 224, 224
    
    # total 12 heads
    num_heads = 12
    all_selected_heads = [[i for i in range(12)], [0], [1], [2], [3], [4], [5], [6], 
                          [7], [8], [9], [10], [11]]
    current_selected_heads = all_selected_heads[0]

    # choose which feature to look, q: 0 k: 1 v: 2
    qkv_choice = 1
    # Vit has image patch 16x16
    num_patches = 16
    # 64 total qkv channels
    attention_channels = 64
    batch_size = 1
    
    # data related
    # read single image
    base_folder = "/home/leo/Documents/GenAI-Vision/attention/"
    single_image_name = "bird.jpg"
    single_image = improcessor(Image.open(base_folder + single_image_name))["pixel_values"][0]
    # sweeping perform parameter sweeping through sampling
    # pre specify parameter combo for running mode
    mode = "running"

    running_project_name = "test sdxl run"

    sweeping_project_name = "test sweep runs 4-29"

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
    
    # number of total iterations, 1000 is maximum, works when the mode is "running"
    num_steps = 50
    # number of random sampling for sweeping
    sweeping_run_count = 200
    
    # outputs
    outputdir = "./debug/" 
    
    sweepingdir = "./sweeping/"

    metricoutputdir = "./metrics/"

    # visualization related
    dpi = 300
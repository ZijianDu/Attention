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
from torchvision.utils import save_image
from dinov2 import Dinov2ModelwOutput

from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
from dinov2 import Dinov2ModelwOutput
import os
import numpy as np
import requests
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from visualizer import visualizer
import shutil
from processor import processor
from skimage.transform import resize
from visualizer import HeatMap
import torchvision.transforms as transforms 
from torchvision.transforms import Resize
from dataclasses import dataclass
from vit_visualization import ViTFeature, ViTPipe, ViTScheduler


# needs cleaning
@dataclass
class sdimg2imgconfigs:
    model_path = 'facebook/dinov2-large'
    link = "runwayml/stable-diffusion-v1-5"
    tokenizer = AutoTokenizer.from_pretrained(link, subfolder="tokenizer", torch_dtype=torch.float16)
    processor = AutoImageProcessor.from_pretrained(model_path, torch_dtype=torch.float16)
    size = [processor.crop_size["height"], processor.crop_size["width"]]
    mean, std = processor.image_mean, processor.image_std
    mean, std = torch.tensor(mean, device="cuda"), torch.tensor(std, device="cuda")
    vit = Dinov2ModelwOutput.from_pretrained(model_path, torch_dtype=torch.float16)
    prompt = "a high-quality image"
    vae = AutoencoderKL.from_pretrained(link, subfolder="vae").to(device="cuda")
    text_encoder = CLIPTextModel.from_pretrained(link, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(link, subfolder="unet").to(device="cuda")
    ddpmscheduler = DDPMSchedulerwithGuidance.from_pretrained(link, subfolder="scheduler")
    # coefficient before guidance
    guidance_strength = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0]
    # percentage iterations to add noise before denoising, higher means more noise added
    strengths = [0.1, 0.3, 0.5, 0.8, 1.0, 10]
    # number of total iterations, 1000 is maximum
    num_steps = 1000
    num_tokens = 256
    image_size = 224
    improcessor = AutoImageProcessor.from_pretrained(model_path)
    vitscheduler = ViTScheduler()
    imageH, imageW = 224, 224
    # total 16 heads
    num_heads = 16
    head_idx = [i for i in range(num_heads)]
    # total 24 layers
    layer_idx = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 23]
    # choose which feature to look, q: 0 k: 1 v: 2
    qkv_choice = 1
    # Vit has image patch 16x16
    num_patches = 16
    # 64 total qkv channels
    attention_channels = 64
    batch_size = 1
    ## -1 means ignore no head, all heads are used for guidance
    ignoreheadidx = -1
    seed = 20
    np.random.seed(seed)

    # read single image
    single_image = "cat.jpg"
    ## to read images in batch
    inputdatadir = "/media/data/leo/style_vector_data/"
    random_list = []
    class_label = 0
    all_classes_list = os.listdir(inputdatadir)
    all_images_list = os.listdir(inputdatadir + all_classes_list[class_label])
    while len(random_list) < batch_size:
        randnum = np.random.randint(0, len(all_images_list))
        if randnum not in random_list:
            random_list.append(randnum)
    picked_images_index = random_list
    outputdir = "./outputs/" 
    metricoutputdir = "./metrics/"
    num_images_in_picked_class = len(os.listdir(inputdatadir + all_classes_list[class_label]))
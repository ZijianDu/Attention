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
from configs import sdimg2imgconfigs
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# configs needed for vit and SDimg2img
configs = sdimg2imgconfigs()
visualizer, processor = visualizer(), processor()
# needed to extract image/latent vit features
vitfeature = ViTFeature(configs, processor, visualizer)

def run_sd_img2img(configs):
    # make sure input image is float and in the range of 0 and 1
    sample_image = torch.tensor(np.array(Image.open("cat.png"))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
    prompt = "a high quality image"
    pipe = StableDiffusionImg2ImgPipelineWithSDEdit(vit = configs.vit, vae=configs.vae, text_encoder=configs.text_encoder, 
                                                    tokenizer=configs.tokenizer, unet=configs.unet, scheduler=configs.ddpmscheduler, 
                                                    safety_checker=None, feature_extractor=None, image_encoder=None, requires_safety_checker=False).to(device="cuda")
    for strength in configs.strengths:
        for guidance_strength in configs.guidance_strength:
            output = pipe(vit_input_size=configs.size, vit_input_mean=configs.mean, vit_input_std=configs.std, 
                    layer_idx=configs.layeridx, guidance_strength=guidance_strength, vitfeature = vitfeature, 
                    prompt = prompt, image = sample_image, strength = strength, num_inference_steps=configs.num_steps, 
                    scheduler = configs.ddpmscheduler, return_dict= False)
            img_np = np.array(output[0][0])
            image_arr = torch.tensor(img_np).type(torch.uint8).numpy()
            img_pil = Image.fromarray(image_arr)
            img_pil.save("cat_strength_" + str(strength) + "_numsteps_" + str(configs.num_steps) 
                        + "_guidencestrength_" + str(guidance_strength) + "_layeridx_" + 
                        str(configs.layeridx) + ".jpg")
            print("saved denoised image")

if __name__ == "__main__":
    run_sd_img2img(configs)
    
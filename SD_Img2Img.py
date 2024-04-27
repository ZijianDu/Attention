import torch
import numpy as np
#import torchvision.utility as tvu
from PIL import Image
import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import inspect
from packaging import version
import requests
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
from dinov2 import Dinov2ModelwOutput
import requests
import torch.nn.functional as F
from visualizer import visualizer
import shutil
from processor import processor
from skimage.transform import resize
from visualizer import HeatMap
import torchvision.transforms as transforms 
from torchvision.transforms import Resize
from vit_visualization import ViTFeature, ViTPipe, ViTScheduler
from configs import sdimg2imgconfigs
import os
from evaluate import evaluator
from scheduler import _preprocess_vit_input
from pytorch_ssim import ssim
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")
# configs needed for vit and SDimg2img
sdconfigs = sdimg2imgconfigs()
visualizer, processor = visualizer(), processor()

class runner(sdimg2imgconfigs):
    def __init__(self, sdconfigs):
        self.sdconfigs = sdconfigs
    # function to run experiment through predefined parameters
    def run_sd_img2img(self, wandb, seed, config = None):
        wandb.log({"seed" : seed})
        sample_image = torch.tensor(np.array(Image.open(sdconfigs.single_image_name))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        prompt = "a high quality image"
        pipe = StableDiffusionImg2ImgPipelineWithSDEdit(vit = sdconfigs.vit, vae=sdconfigs.vae, text_encoder=sdconfigs.text_encoder, 
                                                    tokenizer=sdconfigs.tokenizer, unet=sdconfigs.unet, scheduler=sdconfigs.ddpmscheduler,
                                                    safety_checker=None, feature_extractor=None, image_encoder=None, requires_safety_checker=False).to(device="cuda")
        # param: 0-vit 1-diffusion strength 2-guidance strength
        for i, param in enumerate(sdconfigs.all_params):
            wandb.log({"diffusion strength" : param[1], "guidance strength" : param[2]})
            output = pipe(vit_input_size=sdconfigs.size, vit_input_mean=sdconfigs.mean, vit_input_std=sdconfigs.std, 
                guidance_strength=param[2], vitfeature = ViTFeature(sdconfigs, sdconfigs.layer_idx[0], processor), 
                prompt = prompt, image = sample_image, strength = param[1], num_inference_steps=sdconfigs.num_steps, 
                generator = torch.Generator(device="cuda").manual_seed(seed), debugger = wandb, return_dict= False)
            img_np = np.array(output[0][0])
            image_arr = torch.tensor(img_np).type(torch.uint8).numpy()
            wandb.log({"predicted image":wandb.Image(image_arr)})
            img_pil = Image.fromarray(image_arr)
            if "seed" + str(seed) not in os.listdir(sdconfigs.outputdir):
                os.mkdir(sdconfigs.outputdir + "seed" + str(seed))
            image_name = "strength_" + str(param[1]) + "_guidencestrength_" + str(param[2]) + ".jpg"
            img_pil.save(sdconfigs.outputdir + "seed" + str(seed) + "/" + image_name)
            image = sdconfigs.improcessor(Image.open(sdconfigs.outputdir + "seed" + str(seed) + "/" + image_name))["pixel_values"][0]
            curr_ssim = ssim(torch.tensor(image).unsqueeze(0), 
                            torch.tensor(sdconfigs.single_image).unsqueeze(0)).data
            
            wandb.log({"ssim":curr_ssim.cpu().numpy()})

    # function to run sweeping using configs defined by wandb.config dict
    def run_parameter_sweeping(self, config = None):
        sample_image = torch.tensor(np.array(Image.open(sdconfigs.single_image_name))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        prompt = "a high quality image"
        with wandb.init(config = config):
            config = wandb.config
            pipe = StableDiffusionImg2ImgPipelineWithSDEdit(vit = sdconfigs.vit, vae=sdconfigs.vae, text_encoder=sdconfigs.text_encoder, 
                                                    tokenizer=sdconfigs.tokenizer, unet=sdconfigs.unet, scheduler=sdconfigs.ddpmscheduler,
                                                    safety_checker=None, feature_extractor=None, image_encoder=None, requires_safety_checker=False).to(device="cuda")
            output = pipe(vit_input_size=sdconfigs.size, vit_input_mean=sdconfigs.mean, vit_input_std=sdconfigs.std, 
                guidance_strength=config.guidance_strength, vitfeature = ViTFeature(sdconfigs, sdconfigs.layer_idx[0], processor), 
                prompt = prompt, image = sample_image, strength = config.diffusion_strength, num_inference_steps=sdconfigs.num_steps, 
                generator = None, debugger = wandb, return_dict= False)
            img_np = np.array(output[0][0])
            image_arr = torch.tensor(img_np).type(torch.uint8).numpy()
            wandb.log({"predicted image":wandb.Image(image_arr)})
            img_pil = Image.fromarray(image_arr)
            # no seed needed since we sample from a parameter distribution
            image_name = "diffusion_strength_" + str(config.diffusion_strength) + "_guidencestrength_" + str(config.guidance_strength) + ".jpg"
            img_pil.save(sdconfigs.sweepingdir + "/" + image_name)
            image = sdconfigs.improcessor(Image.open(sdconfigs.sweepingdir +  "/" + image_name))["pixel_values"][0]
            curr_ssim = ssim(torch.tensor(image).unsqueeze(0), 
                            torch.tensor(sdconfigs.single_image).unsqueeze(0)).data
            
            wandb.log({"ssim":curr_ssim.cpu().numpy()})

if __name__ == "__main__":
    runner = runner(sdconfigs)
    # perform parameter sweeping procedure
    if sdconfigs.mode == "sweeping":
        wandb.login()
        sweep_id = wandb.sweep(sdconfigs.sweep_config, project = sdconfigs.sweeping_project_name)
        wandb.agent(sweep_id, runner.run_parameter_sweeping, count = sdconfigs.sweeping_run_count)
    
    # perform normal run given preselected parameters
    if sdconfigs.mode == "running":  
        wandb.login()
        for seed in range(5):
            wandb.init(project = sdconfigs.running_project_name, name = "run: " + str(seed))
            runner.run_sd_img2img(wandb, seed)
            wandb.finish()
    
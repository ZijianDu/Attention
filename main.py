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
from pipeline import StableDiffusionImg2ImgPipelineWithSDEdit, StableDiffusionXLImg2ImgPipelineWithViTGuidance
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
from torchvision.transforms.functional import resize
from dinov2 import Dinov2ModelwOutput
import requests
import torch.nn.functional as F
from visualizer import visualizer, HeatMap
import shutil
from processor import processor
from skimage.transform import resize
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from attention.vit_feature_extractor import ViTFeature, ViTPipe, ViTScheduler
from configs import sdimg2imgconfigs
import os
from evaluate import evaluator
from scheduler import _preprocess_vit_input
from pytorch_ssim import ssim
import wandb
import matplotlib.pyplot as plt
import lpips
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

# configs needed for vit and SDimg2img
sdconfigs = sdimg2imgconfigs()
visualizer, processor = visualizer(), processor()

from torchvision.transforms import v2
transforms = v2.Compose([
    v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
    v2.RandomResizedCrop(size=(328, 496), antialias=True)
])

class runner(sdimg2imgconfigs):
    def __init__(self, sdconfigs):
        self.sdconfigs = sdconfigs
    def plot_image_histogram(self, images: List):
        for image in images:
            plt.hist(image.flatten(), alpha = 0.4, bins = 100)
        plt.savefig('overlapped image histo.png')

    # function to run experiment through predefined parameters
    def run_sd_img2img(self, wandb, seed, config = None):
        wandb.log({"seed" : seed})
        # sample image used to prepare for diffusion
        sample_image = torch.tensor(np.array(Image.open(sdconfigs.base_folder + sdconfigs.single_image_name))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        prompt = "a high quality image"
        #pipe = StableDiffusionImg2ImgPipelineWithSDEdit(vit = sdconfigs.vit, vae=sdconfigs.vae, text_encoder=sdconfigs.text_encoder, 
        #                                            tokenizer=sdconfigs.tokenizer, unet=sdconfigs.unet, scheduler=sdconfigs.ddpmscheduler,
        #                                            safety_checker=None, feature_extractor=None, image_encoder=None, requires_safety_checker=False).to(device="cuda")
        
        pipe = StableDiffusionXLImg2ImgPipelineWithViTGuidance(vit = sdconfigs.vit, vae = sdconfigs.vae, text_encoder = sdconfigs.text_encoder, tokenizer=sdconfigs.tokenizer, unet=sdconfigs.unet, scheduler=sdconfigs.ddpmscheduler,
                                  feature_extractor=None, requires_aesthetics_score= False, force_zeros_for_empty_prompt=True, add_watermarker=None)
                                                            
        original_image_vitfeature = ViTFeature(sdconfigs, processor)
        original_image_vitfeature.read_one_image()
        original_image_vitfeature.extract_all_original_vit_features()
        original_vit_features = original_image_vitfeature.get_all_original_vit_features()

        # param: 0-vit 1-diffusion strength 2-guidance strength
        for i, param in enumerate(sdconfigs.all_params):
            #wandb.log({"diffusion strength" : param[1], "guidance strength" : param[2]})
            for selected_heads in sdconfigs.all_selected_heads:
                sdconfigs.current_selected_heads = selected_heads
                wandb.log({"selected heads" : str(selected_heads)})
                latent_vit_features = ViTFeature(sdconfigs, processor)

                #output = pipe(vit_input_size=sdconfigs.size, vit_input_mean=sdconfigs.mean, vit_input_std=sdconfigs.std, 
                #    guidance_strength=param[2], all_original_vit_features = original_vit_features, vitfeature = latent_vit_features,  
                #    configs = sdconfigs, prompt = prompt, image = sample_image, diffusion_strength = param[1], 
                #    num_inference_steps = sdconfigs.num_steps, generator = torch.Generator(device="cuda").manual_seed(seed), debugger = wandb, return_dict= False)
                
                output = pipe(vit_input_size=sdconfigs.size, vit_input_mean=sdconfigs.mean, vit_input_std=sdconfigs.std, 
                        guidance_strength=param[2], all_original_vit_features = original_vit_features, vitfeature = latent_vit_features,  
                        configs = sdconfigs, debugger = wandb, prompt = prompt, image = sample_image, diffusion_strength = param[1], 
                        num_inference_steps = sdconfigs.num_steps, generator = torch.Generator(device="cuda").manual_seed(seed), return_dict= False)
                
                img_predicted = torch.tensor(np.array(output[0][0])).permute(2, 0, 1).unsqueeze(0)
                img_noprocess = Image.open(sdconfigs.base_folder + sdconfigs.single_image_name)
                resized_original_image = transforms(img_noprocess).unsqueeze(0)
                curr_ssim = ssim(img_predicted, resized_original_image).data
                #wandb.log({"ssim" : curr_ssim.cpu().numpy()})
                lpips_normed_predicted_img = img_predicted / 255.0 - 0.5
                lpips_normed_original_img = resized_original_image / 255.0 - 0.5
                loss_fn_alex, loss_fn_vgg = lpips.LPIPS(net = 'alex'), lpips.LPIPS(net = 'vgg')
                dist_alex, dist_vgg = loss_fn_alex(lpips_normed_predicted_img, lpips_normed_original_img), loss_fn_vgg(img_predicted, resized_original_image)
                wandb.log({"dist_alex" : dist_alex.detach().numpy(), "dist_vgg" : dist_vgg.detach().numpy()})
                wandb.log({"predicted image" : wandb.Image(img_predicted)})

    # function to run sweeping using configs defined by wandb.config dict
    def run_parameter_sweeping(self, config = None):
        sample_image = torch.tensor(np.array(Image.open(sdconfigs.single_image_name))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        prompt = "a high quality image"
        with wandb.init(config = config):
            # sample image used to prepare for diffusion
            sample_image = torch.tensor(np.array(Image.open(sdconfigs.base_folder + sdconfigs.single_image_name))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            pipe = StableDiffusionImg2ImgPipelineWithSDEdit(vit = sdconfigs.vit, vae=sdconfigs.vae, text_encoder=sdconfigs.text_encoder, 
                                                        tokenizer=sdconfigs.tokenizer, unet=sdconfigs.unet, scheduler=sdconfigs.ddpmscheduler,
                                                        safety_checker=None, feature_extractor=None, image_encoder=None, requires_safety_checker=False).to(device="cuda")
        
            original_image_vitfeature = ViTFeature(sdconfigs, processor)
            original_image_vitfeature.read_one_image()
            original_image_vitfeature.extract_all_original_vit_features()
            original_vit_features = original_image_vitfeature.get_all_original_vit_features()
            
            # get the parameters we want to sweep from wandb config
            config = wandb.config
            output = pipe(vit_input_size=sdconfigs.size, vit_input_mean=sdconfigs.mean, vit_input_std=sdconfigs.std, 
                guidance_strength=config.guidance_strength, all_original_vit_features = original_vit_features, vitfeature = ViTFeature(sdconfigs, processor), 
                configs = sdconfigs, prompt = prompt, image = sample_image, diffusion_strength = config.diffusion_strength, num_inference_steps=sdconfigs.num_steps, 
                generator = None, debugger = wandb, return_dict= False)
            
            img_predicted = torch.tensor(np.array(output[0][0])).permute(2, 0, 1).unsqueeze(0)
            img_noprocess = Image.open(sdconfigs.base_folder + sdconfigs.single_image_name)
            resized_original_image = transforms(img_noprocess).unsqueeze(0)
            curr_ssim = ssim(img_predicted, resized_original_image).data
            #wandb.log({"ssim" : curr_ssim.cpu().numpy()})
            lpips_normed_predicted_img = img_predicted / 255.0 - 0.5
            lpips_normed_original_img = resized_original_image / 255.0 - 0.5
            loss_fn_alex, loss_fn_vgg = lpips.LPIPS(net = 'alex'), lpips.LPIPS(net = 'vgg')
            dist_alex, dist_vgg = loss_fn_alex(lpips_normed_predicted_img, lpips_normed_original_img), loss_fn_vgg(img_predicted, resized_original_image)
            wandb.log({"dist_alex" : dist_alex.detach().numpy(), "dist_vgg" : dist_vgg.detach().numpy()})


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
        for seed in range(65, 70, 1):
            wandb.init(project = sdconfigs.running_project_name, name = "run: " + str(seed))
            runner.run_sd_img2img(wandb, seed)
            wandb.finish()
    
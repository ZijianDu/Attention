import torch
import numpy as np
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
from pipeline import StableDiffusionImg2ImgPipelineWithSDEdit, StableDiffusionXLImg2ImgPipelineWithViTGuidance,  StableDiffusionXLPipelineWithViTGuidance
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
import shutil
from skimage.transform import resize
import torchvision.transforms as transforms
from torchvision.transforms import Resize
from vit_feature_extractor import ViTFeature, ViTPipe, ViTScheduler
from configs import runconfigs, wandbconfigs
import os
from util import processor
from scheduler import _preprocess_vit_input
from pytorch_ssim import ssim
import wandb
import matplotlib.pyplot as plt
import lpips
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings("ignore")

from configs import runconfigs, wandbconfigs
from torchvision.transforms import v2
runconfigs, wandbconfigs = runconfigs(), wandbconfigs()

class runner:
    def __init__(self, runconfigs, wandbconfigs):
        self.runconfigs = runconfigs
        self.wandbconfigs = wandbconfigs

    def plot_image_histogram(self, images: List):
        for image in images:
            plt.hist(image.flatten(), alpha = 0.4, bins = 100)
        plt.savefig('overlapped image histo.png')

    # function to run experiment through predefined parameters
    def run_sd_img2img(self, wandb, seed, config = None):
        #wandb.log({"seed" : seed})
        # sample image used to prepare for diffusion
        sample_image = torch.tensor(np.array(Image.open(runconfigs.base_folder + runconfigs.single_image_name))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        prompt = "a high quality image"

        pipe = self.runconfigs.stablediffusionxltext2imgpipewithvitguidance                                
        
        original_image_vitfeature = ViTFeature(runconfigs)
        original_image_vitfeature.read_one_image()
        original_image_vitfeature.extract_all_original_vit_features()
        original_vit_features = original_image_vitfeature.get_all_original_vit_features()

        # param: 0-vit 1-diffusion strength 2-guidance strength
        for i, param in enumerate(runconfigs.all_params):
            #wandb.log({"diffusion strength" : param[1], "guidance strength" : param[2]})
            for selected_heads in runconfigs.all_selected_heads:
                runconfigs.current_selected_heads = selected_heads
                wandb.log({"selected heads" : str(selected_heads)})
                latent_vit_features = ViTFeature(runconfigs)

                # parameters for sdxltext2img pipe
                output = pipe(vit_input_size = self.runconfigs.size,
                            vit_input_mean = self.runconfigs.mean,
                            vit_input_std = self.runconfigs.std,
                            guidance_strength = param[1],
                            all_original_vit_features = original_vit_features,
                            vitfeature= latent_vit_features,
                            configs = self.runconfigs, 
                            image = sample_image,
                            diffusion_strength = param[0],
                            debugger = wandb, 
                            prompt = prompt,
                            prompt_2 = prompt,
                            num_inference_steps = self.runconfigs.num_steps,
                            generator = None, 
                            return_dict = False)
                
                img_predicted = torch.tensor(np.array(output[0][0])).permute(2, 0, 1).unsqueeze(0)
                img_noprocess = Image.open(runconfigs.base_folder + runconfigs.single_image_name)
                sample_image_array = np.array(sample_image)
                shape = (sample_image_array[1], sample_image_array[1], sample_image_array[2])
                transforms = v2.Compose([v2.ToImage(),  # Convert to tensor, only needed if you had a PIL image
                                        v2.RandomResizedCrop(size=shape, antialias=True)])
                resized_original_image = transforms(img_noprocess).unsqueeze(0)
                lpips_normed_predicted_img = img_predicted / 255.0 - 0.5
                lpips_normed_original_img = resized_original_image / 255.0 - 0.5
                loss_fn_alex, loss_fn_vgg = lpips.LPIPS(net = 'alex'), lpips.LPIPS(net = 'vgg')
                dist_alex, dist_vgg = loss_fn_alex(lpips_normed_predicted_img, lpips_normed_original_img), loss_fn_vgg(img_predicted, resized_original_image)
                wandb.log({"dist_alex" : dist_alex.detach().numpy(), "dist_vgg" : dist_vgg.detach().numpy()})
                wandb.log({"predicted image" : wandb.Image(img_predicted)})

    # function to run sweeping using configs defined by wandb.config dict
    def run_parameter_sweeping(self, config = None):
        sample_image = torch.tensor(np.array(Image.open(runconfigs.single_image_name))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
        prompt = "a high quality image"
        with wandb.init(config = config):
            # sample image used to prepare for diffusion
            sample_image = torch.tensor(np.array(Image.open(runconfigs.base_folder + runconfigs.single_image_name))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
            pipe = StableDiffusionImg2ImgPipelineWithSDEdit(vit = runconfigs.vit, vae=runconfigs.vae, text_encoder=runconfigs.text_encoder, 
                                                        tokenizer=runconfigs.tokenizer, unet=runconfigs.unet, scheduler=runconfigs.ddpmscheduler,
                                                        safety_checker=None, feature_extractor=None, image_encoder=None, requires_safety_checker=False).to(device="cuda")
        
            original_image_vitfeature = ViTFeature(runconfigs, processor)
            original_image_vitfeature.read_one_image()
            original_image_vitfeature.extract_all_original_vit_features()
            original_vit_features = original_image_vitfeature.get_all_original_vit_features()
            
            # get the parameters we want to sweep from wandb config
            config = wandb.config
            output = pipe(vit_input_size=runconfigs.size, vit_input_mean=runconfigs.mean, vit_input_std=runconfigs.std, 
                guidance_strength=config.guidance_strength, all_original_vit_features = original_vit_features, vitfeature = ViTFeature(runconfigs, processor), 
                configs = runconfigs, prompt = prompt, image = sample_image, diffusion_strength = config.diffusion_strength, num_inference_steps=runconfigs.num_steps, 
                generator = None, debugger = wandb, return_dict= False)
            
            img_predicted = torch.tensor(np.array(output[0][0])).permute(2, 0, 1).unsqueeze(0)
            img_noprocess = Image.open(runconfigs.base_folder + runconfigs.single_image_name)
            resized_original_image = transforms(img_noprocess).unsqueeze(0)
            lpips_normed_predicted_img = img_predicted / 255.0 - 0.5
            lpips_normed_original_img = resized_original_image / 255.0 - 0.5
            loss_fn_alex, loss_fn_vgg = lpips.LPIPS(net = 'alex'), lpips.LPIPS(net = 'vgg')
            dist_alex, dist_vgg = loss_fn_alex(lpips_normed_predicted_img, lpips_normed_original_img), loss_fn_vgg(img_predicted, resized_original_image)
            wandb.log({"dist_alex" : dist_alex.detach().numpy(), "dist_vgg" : dist_vgg.detach().numpy()})


if __name__ == "__main__":
    runner = runner(runconfigs, wandbconfigs)
    # perform parameter sweeping procedure
    if wandbconfigs.mode == "sweeping":
        wandb.login()
        sweep_id = wandb.sweep(runconfigs.sweep_config, project = runconfigs.sweeping_project_name)
        wandb.agent(sweep_id, runner.run_parameter_sweeping, count = runconfigs.sweeping_run_count)
    
    # perform normal run given preselected parameters
    if wandbconfigs.mode == "running":  
        wandb.login()
        for seed in range(1):
            wandb.init(project = wandbconfigs.running_project_name, name = "run: " + str(seed))
            runner.run_sd_img2img(wandb, seed)
            wandb.finish()
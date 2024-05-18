from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.utils.torch_utils import randn_tensor
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torch.distributions import Categorical
import numpy as np
from dataclasses import dataclass
from torch.nn import MSELoss

def _preprocess_vit_input(images: torch.Tensor, size: list[int], mean: torch.Tensor, std: torch.Tensor):
    # step 1: resize
    images = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)
    # step 2: normalize
    normalized  = (images - torch.mean(images)) / torch.std(images)
    return normalized

## modified DDPM scheduler with guidance from ViT
class DDPMSchedulerwithGuidance(DDPMScheduler):
    # modify step function s.t each predicted x0 is guided by Vit feature
    def step(
        self,
        vae, 
        vit,
        debugger,
        vit_input_size,
        vit_input_mean, 
        vit_input_std,
        guidance_strength,
        all_original_vit_features,
        configs, 

        #base model inputs
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        generator=None,
        return_dict = False
    ):
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            vit:
                Vit used as guidance
            vae:
                use decoder to decode latent prediction into image
            model_output/predicted noise (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample/latent/xt (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_ddpm.DDPMSchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        t = timestep
        debugger.log({"timestep": timestep})

        prev_t = self.previous_timestep(t)

        if model_output.shape[1] == sample.shape[1] * 2 and self.variance_type in ["learned", "learned_range"]:
            model_output, predicted_variance = torch.split(model_output, sample.shape[1], dim=1)
        else:
            predicted_variance = None

        # 1. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # define loss
        loss = MSELoss()
        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf

        # add guidance from vit's key enable grad for guidance
        with torch.enable_grad():
            if self.config.prediction_type == "epsilon":
                sample_ = sample.clone().requires_grad_(True)
                
                debugger.log({"xt": debugger.Image(sample_.clone())})

                # pred original sample, original code
                pred_original_sample = (sample_ - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                debugger.log({"x0" : debugger.Image(pred_original_sample.clone())})
                
                # this is needed to calculate vit feature for guidance
                # obtain image space prediction by decoding predicted x0
                images = vae.decode(pred_original_sample / vae.config.scaling_factor).sample / 2 + 0.5  # [0, 1]

                # clamp image range
                torch.clamp(images, min = 0.0, max = 1.0)
            
                vit_input = _preprocess_vit_input(images, vit_input_size, vit_input_mean, vit_input_std)
                debugger.log({"predicted image" : debugger.Image(vit_input.clone())})
                
                latent_vit_features = vit(configs.layer_idx[0], vit_input, output_attentions=False)
                
                debugger.log({"predicted image vit key" : latent_vit_features.clone()})
                
                # select which head we want to use as guidance
                indices = torch.tensor(configs.current_selected_heads).cuda()
                selected_original_vit_features = torch.index_select(all_original_vit_features, 1, indices)   
                selected_latent_vit_features = torch.index_select(latent_vit_features, 1, indices)

                assert selected_original_vit_features.shape[1] == len(configs.current_selected_heads)
                assert selected_latent_vit_features.shape[1] == len(configs.current_selected_heads)

                # MSE between latent vit features and clean image features as guidance
                curr_loss = loss(selected_latent_vit_features, selected_original_vit_features)
                debugger.log({"mse loss": curr_loss.clone().detach().cpu().numpy()})

                # calculate gradient
                gradient = torch.autograd.grad(curr_loss, [sample_])[0]

                actual_guidance = guidance_strength * beta_prod_t ** (0.5) * gradient

                # according to adm, guidance is applied on epislon
                model_output = model_output - actual_guidance

                # use updated epislon to predict original sample
                pred_original_sample = (sample_ - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                
                debugger.log({"x0 with guidance" : debugger.Image(pred_original_sample.clone())})
                #debugger.log({"actual guidance" : actual_guidance.clone().detach().cpu().numpy()})
    
        # 3. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                - self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * current_beta_t) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        # 5. Compute predicted previous sample µ_t
        # See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample_

        # 6. Add noise
        variance = 0
        if t > 0:
            device = model_output.device
            variance_noise = randn_tensor(
                model_output.shape, generator=generator, device=device, dtype=model_output.dtype
            )
            
            if self.variance_type == "fixed_small_log":
                variance = self._get_variance(t, predicted_variance=predicted_variance) * variance_noise
            elif self.variance_type == "learned_range":
                variance = self._get_variance(t, predicted_variance=predicted_variance)
                variance = torch.exp(0.5 * variance) * variance_noise
            else:
                variance = (self._get_variance(t, predicted_variance=predicted_variance) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)


class DDIMSchedulerwithGuidance(DDIMScheduler):
    def step(self, 
             vae,
             vit, 
             debugger,
             vit_input_size, 
             vit_input_mean, 
             vit_input_std,
             guidance_strength, 
             all_original_vit_features, 
             configs, 
             
             # base class arguments
             model_output,  
             timestep,  
             sample,
             eta, 
             use_clipped_model_output,
             generator, 
             variance_noise, 
             return_dict=False
    ):
        """

        Returns:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.DDIMSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )
        # See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
        # Ideally, read DDIM paper in-detail understanding

        # Notation (<variable name> -> <name in paper>
        # - pred_noise_t -> e_theta(x_t, t)
        # - pred_original_sample -> f_theta(x_t, t) or x_0
        # - std_dev_t -> sigma_t
        # - eta -> η
        # - pred_sample_direction -> "direction pointing to x_t"
        # - pred_prev_sample -> "x_t-1"

        # 1. get previous step value (=t-1)
        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

        # 2. compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        
        # define loss function
        loss = MSELoss()

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        with torch.enable_grad():    
            if self.config.prediction_type == "epsilon":

                sample_ = sample.clone().requires_grad_(True)
                debugger.log({"xt" : debugger.Image(sample_.clone())})

                # original code
                pred_original_sample = (sample_ - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                debugger.log({"x0" : debugger.Image(pred_original_sample.clone())})
                
                #images = pred_original_sample.unsqueeze(0)
                images = vae.decode(pred_original_sample / vae.config.scaling_factor).sample / 2 + 0.5

                # clamp image range
                torch.clamp(images, min = 0.0, max = 1.0)

                vit_input = _preprocess_vit_input(images, vit_input_size, vit_input_mean, vit_input_std)
                debugger.log({"predicted image" : debugger.Image(vit_input.clone())})
            
                latent_vit_features = vit(configs.layer_idx[0], vit_input, output_attentions=False)

                debugger.log({"predicted image vit key" : latent_vit_features.clone().detach().cpu().numpy()})

                indices = torch.tensor(configs.current_selected_heads).cuda()
                selected_original_vit_features = torch.index_select(all_original_vit_features, 1, indices)   
                selected_latent_vit_features = torch.index_select(latent_vit_features, 1, indices)

                assert selected_original_vit_features.shape[1] == len(configs.current_selected_heads)
                assert selected_latent_vit_features.shape[1] == len(configs.current_selected_heads)

                # MSE between latent vit features and clean image features as guidance
                curr_loss = loss(selected_original_vit_features, selected_latent_vit_features)
                debugger.log({"mse loss": curr_loss.clone().detach().cpu().numpy()})

                gradient = torch.autograd.grad(curr_loss, [sample_])[0]

                actual_guidance = guidance_strength * beta_prod_t ** (0.5) * gradient
                
                # original code, guidance is added on predicted epsilon
                pred_epsilon = model_output - actual_guidance
                debugger.log({"actual guidance" : debugger.Image(actual_guidance.clone())})

        # 4. Clip or threshold "predicted x_0"
        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * variance ** (0.5)

        if use_clipped_model_output:
            # the pred_epsilon is always re-derived from the clipped x_0 in Glide
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)

        # 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon

        # 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError(
                    "Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
                    " `variance_noise` stays `None`."
                )

            if variance_noise is None:
                variance_noise = randn_tensor(
                    model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
                )
            variance = std_dev_t * variance_noise

            prev_sample = prev_sample + variance

        if not return_dict:

            return (prev_sample,)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
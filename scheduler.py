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

def cal_entropy(p):
    return -1.0 * torch.sum(p * torch.log(p), dim = -1)

def _preprocess_vit_input(images: torch.Tensor, size: list[int], mean: torch.Tensor, std: torch.Tensor):
    # step 1: resize
    images = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)
    # step 2: normalize
    normalized  = (images - torch.mean(images)) / torch.std(images)
    assert abs(torch.mean(normalized).data.cpu().numpy())  < 0.001
    assert abs(torch.std(normalized).detach().cpu().numpy() - 1.0) < 0.001 
    return normalized     


## modified DDPM scheduler with guidance from ViT
class DDPMSchedulerwithGuidance(DDPMScheduler):
    # modify step function s.t each predicted x0 is guided by Vit feature
    def step(
        self,
        vit, 
        vae,
        debugger, 
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        vit_input_size: list[int],
        vit_input_mean: torch.Tensor,
        vit_input_std: torch.Tensor,
        guidance_strength: float, 
        all_original_vit_features, 
        vitfeature,
        configs, 
        generator=None,
        return_dict: bool = True,
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

        #debugger.log({"alpha_prod_t": alpha_prod_t})
        #debugger.log({"beta_prod_t": beta_prod_t})

        # 2. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf

        # add guidance
        if self.config.prediction_type == "epsilon":
            sample_ = sample.clone().requires_grad_(True)
            
            debugger.log({"latent": sample_.detach().cpu().numpy()})
            loss = MSELoss()
            
            # pred original sample ix x0_hat
            pred_original_sample = (sample_ - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            
            # this is needed to calculate vit feature for guidance
            # obtain image space prediction by decoding predicted x0
            images = vae.decode(pred_original_sample / vae.config.scaling_factor).sample / 2 + 0.5  # [0, 1]
            
            # clamp image range
            torch.clamp(images, min = 0.0, max = 1.0)
          
            vit_input = _preprocess_vit_input(images, vit_input_size, vit_input_mean, vit_input_std)
          
            vitfeature.extract_selected_latent_vit_features(vit_input)

            # false if we are not getting original image feature
            latent_vit_features = vitfeature.get_selected_latent_vit_features().to(torch.float16)
            debugger.log({"latent vit feature" : latent_vit_features})

            indices = torch.tensor(configs.current_selected_heads).cuda()
            selected_original_vit_features = torch.index_select(all_original_vit_features, 1, indices)          
            
            assert selected_original_vit_features.shape == (1, len(configs.current_selected_heads),
                                                            configs.num_patches**2, configs.attention_channels)
            assert selected_original_vit_features.shape == latent_vit_features.shape

            # MSE between latent vit features and clean image features as guidance
            curr_loss = loss(latent_vit_features, selected_original_vit_features)
            debugger.log({"mse loss": curr_loss.detach().cpu().numpy()})

            sample_.data = sample_.data.to(curr_loss.dtype)
            # calculate gradient
            gradient = torch.autograd.grad(curr_loss, [sample_])[0]
            #debugger.log({"guidance":guidance.detach().cpu().numpy(), "gradient": gradient.detach().cpu().numpy()})
            # calculate actual guidance and add to xt
            #actual_guidance = torch.tensor(0.0).cuda()
            #timestep <= guidance_range_max:
            actual_guidance = guidance_strength * beta_prod_t ** 0.5 * gradient
            
            sample_ = sample_ - actual_guidance
            debugger.log({"actual guidance" : actual_guidance.detach().cpu().numpy(), "xt with guidance" : sample_.detach().cpu().numpy()})
    
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
        debugger.log({"ddpm noise": variance.cpu().numpy()})
        debugger.log({"xt-1 with noise": pred_prev_sample.detach().cpu().numpy()})

        if not return_dict:
            return (pred_prev_sample,)

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

class DDIMSchedulerWithViT(DDIMScheduler):
    def step(
        self,
        vit,
        vae,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        vit_input_size: list[int],
        vit_input_mean: torch.Tensor,
        vit_input_std: torch.Tensor,
        layer_idx: int,
        head_idx: int,
        guidance_strength: float = 0.0,
        eta: float = 0.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
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

        # 3. compute predicted original sample from predicted noise also called
        # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
        if self.config.prediction_type == "epsilon":
            sample = sample.clone().requires_grad_(True)
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
            # obtain image input to ViT
            images = vae.decode(pred_original_sample / vae.config.scaling_factor).sample / 2 + 0.5  # [0, 1]
            # images: Batchsize x 3 x 512 x 512
            vit_input = _preprocess_vit_input(images, vit_input_size, vit_input_mean, vit_input_std)
            print("one time step, getting attention map from Vit ")
            attentions = vit(layer_idx, head_idx, vit_input, output_attentions=True)

            oneiterationqkv = vit.getqkv()

            attention_scores = attentions.attentions
            score_per_head = attention_scores[layer_idx][:, head_idx, 1:, 1:].squeeze()
            all_token_entropy = cal_entropy(score_per_head)
            mean_entropy = torch.mean(all_token_entropy)
            gradient = torch.autograd.grad(mean_entropy, [sample])[0]
            gradient = 0
            pred_epsilon = model_output + guidance_strength * beta_prod_t ** 0.5 * gradient

        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
                " `v_prediction`"
            )

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
            attention_mean = score_per_head.detach()
            entropy = mean_entropy,
            images = images.detach()
            return (prev_sample, attention_mean, entropy, images, oneiterationqkv)

        return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
        



"""# obselete:
#visualizer.plot_attentionmap(score_per_head, timestep)
#visualizer.plot_attentionmap(attention_scores)
#entropy = Categorical(attention_prob).entropy()
#print("\n")
#print("length of attention scores: ")
#print(len(attention_scores))
#print("\n")
#print("dimension of one attention layer: ")
#print(attention_scores[0].shape)
#print("\n")
#print("value of attention scores for layer 0: ")
#print(attention_scores[0])
#assert layer_idx < len(attentions), f"{len(attentions):d} attentions in ViT, got {layer_idx:d}"
#attention = attentions[layer_idx]
#assert head_idx < attention.shape[1], f"{attention.shape[1]:d} heads in attention, got {head_idx:d}"
#attention = F.softmax([:, :, 0, 1:], dim = 1)
#attention_mean = attention.mean(dim=(-2, -1)).sum()"""
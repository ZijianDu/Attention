from diffusers import DDIMScheduler, DDPMScheduler
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from Attention.visualizer import visualizer
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torch.distributions import Categorical
import numpy as np
visualizer = visualizer()

def cal_entropy(p):
    return -1.0 * torch.sum(p * torch.log(p), dim = -1)

class ViTScheduler():
    def step(self, vit, images, vit_input_size, layer_idx, head_idx):
        print("layer idx %s, head idx %s", layer_idx, head_idx)
        vit(layer_idx, head_idx, images, output_attentions=True)
        print("passed through ViT")
        qkv = vit.getqkv()
        print("returning qkv")
        return qkv

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
            '''
            ## return of modified DinoV2
            class BaseModelOutputWithPoolingwAttentionScores:
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions @after softmax, sum to 1
            attention_scores=encoder_outputs.attention_scores @before softmax, does not sum to one
            '''
            print("one time step, getting attention map from Vit ")
            attentions = vit(layer_idx, head_idx, vit_input, output_attentions=True)

            oneiterationqkv = vit.getqkv()

            attention_scores = attentions.attentions
            score_per_head = attention_scores[layer_idx][:, head_idx, 1:, 1:].squeeze()
            all_token_entropy = cal_entropy(score_per_head)
            mean_entropy = torch.mean(all_token_entropy)
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
            #attention_mean = attention.mean(dim=(-2, -1)).sum()
            # apply guidance
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
        
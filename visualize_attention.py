from diffusers import DDIMScheduler
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from Attention.visualizer import visualizer
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torch.distributions import Categorical
import numpy as np
import os
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler
)

from transformers.models.dinov2.modeling_dinov2 import Dinov2Model
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
from dinov2 import Dinov2ModelwOutput
import os
import requests
from diffusers import DDIMPipeline
import torch
from Attention.visualizer import visualizer
from transformers.models.clip import CLIPTextModel
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from Attention.visualizer import visualizer
import shutil
import unittest

class configs:
    def __init__(self):
        self.model_path = 'facebook/dinov2-large'
        self.link = "runwayml/stable-diffusion-v1-5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.link, subfolder="tokenizer")
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.size = [self.processor.crop_size["height"], self.processor.crop_size["width"]]
        self.mean, self.std = self.processor.image_mean, self.processor.image_std
        self.mean = torch.tensor(self.mean, device="cuda")
        self.std = torch.tensor(self.std, device="cuda")
        self.seed = 20
        self.vit = Dinov2Model.from_pretrained(self.model_path)
        self.outputdir = "./visualize_attentions/"
        self.imagedir = "./imagenet/imagenet_images/"
        self.imageclass = os.listdir(self.imagedir)
        self.heads = [0, 5, 10, 15]
        self.layers = [0, 5, 10, 15]


configs = configs()
visualizer = visualizer()

def _preprocess_vit_input(images: torch.Tensor, size: list[int], mean: torch.Tensor, std: torch.Tensor):
    # step 1: resize
    images = F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)
    # step 2: normalize
    normalized  = (images - torch.mean(images)) / torch.std(images)
    assert abs(torch.mean(normalized).cpu().numpy())  < 0.001
    assert abs(torch.std(normalized).cpu().numpy() - 1.0) < 0.001 
    return normalized     
    
def visualize_attention():
    for head in configs.heads:
        print("processing head: ", str(head))
        for layer in configs.layers:
            print("processing layer: ", str(layer))
            savedir = "head_" + str(head) + "_layer_" + str(layer)
            average_attention = torch.zeros(size = (16, 16))
            for imageclass in os.listdir(configs.imagedir):
                print("processing class: ", imageclass)
                for imagename in os.listdir(configs.imagedir + imageclass + "/"):
                    if imagename[-3:] not in ['png', 'jpg']:
                        continue
                    print("processing image: ", imagename)
                    image = Image.open(configs.imagedir + imageclass + "/" + imagename, 'r')
                    image = torch.tensor(np.array(image), dtype = torch.float32).permute(2, 0, 1).unsqueeze(dim = 0)/ 255.0
                    vit_input = _preprocess_vit_input(image, configs.size, configs.mean, configs.std)
                    attentions = configs.vit(vit_input, output_attentions=True)
                    attention_scores = attentions.attentions
                    score_per_head = attention_scores[layer][:, head, 128, 1:].squeeze()
                    score_per_head_reshaped = score_per_head.reshape((int(np.sqrt(score_per_head.shape[-1])), int(np.sqrt(score_per_head.shape[-1]))))
                    average_attention += score_per_head_reshaped
                    if savedir not in os.listdir(configs.outputdir):
                        os.mkdir(configs.outputdir + savedir + "/")
                    visualizer.plot_attentionmap(score_per_head_reshaped, None, imagename, configs.outputdir + savedir + "/")
                    del image
            visualizer.plot_attentionmap(average_attention, None, "averaged_" + savedir + ".jpg", configs.outputdir + "averaged/")
visualize_attention()
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
from dinov2 import Dinov2ModelwOutput
import os
import numpy as np
import requests
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from visualization import visualizer
import shutil
from processor import processor
from scheduler import DDIMSchedulerWithViT
from pipeline import StableDiffusionPipelineWithViT
from skimage.transform import resize
from visualization import HeatMap

class configs:
    def __init__(self):
        self.model_path = 'facebook/dinov2-large'
        self.tokenizer = AutoTokenizer.from_pretrained(self.link, subfolder="tokenizer")
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.size = [self.processor.crop_size["height"], self.processor.crop_size["width"]]
        self.mean, self.std = self.processor.image_mean, self.processor.image_std
        self.mean = torch.tensor(self.mean, device="cuda")
        self.std = torch.tensor(self.std, device="cuda")
        self.prompt = "a high-quality image"
        self.seed = 20
        self.vit = Dinov2ModelwOutput.from_pretrained(self.model_path)
        self.scheduler = DDIMSchedulerWithViT.from_pretrained(self.link, subfolder="scheduler")
        self.layeridx = [0]
        self.head = [0]
        self.imageH = 224
        self.imageW = 224
        self.outputdir = "./outputs" 
        self.metricoutputdir = "./metrics"
        self.qkvoutputdir = "./qkv"
        self.trials = 1
        

configs = configs()
visualizer = visualizer()
processor = processor()
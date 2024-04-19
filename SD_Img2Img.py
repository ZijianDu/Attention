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
from Attention.visualizer import visualizer
from scheduler import ViTScheduler
import shutil
from processor import processor
from scheduler import ViTScheduler
from pipeline import ViTPipe
from skimage.transform import resize
from Attention.visualizer import HeatMap
import torchvision.transforms as transforms 
from torchvision.transforms import Resize
from dataclasses import dataclass

class configs:
    def __init__(self):
        self.model_path = 'facebook/dinov2-large'
        self.improcessor = AutoImageProcessor.from_pretrained(self.model_path)
        self.size = [self.improcessor.crop_size["height"], self.improcessor.crop_size["width"]]
        self.mean, self.std = self.improcessor.image_mean, self.improcessor.image_std
        self.mean = torch.tensor(self.mean, device="cuda")
        self.std = torch.tensor(self.std, device="cuda")
        self.vit = Dinov2ModelwOutput.from_pretrained(self.model_path)
        self.scheduler = ViTScheduler()
        self.imageH = 224
        self.imageW = 224
        self.outputdir = "./outputs" 
        self.metricoutputdir = "./metrics"
        self.outputdir = ["./qkv/q/", "./qkv/k/", "./qkv/v/"]
        self.class_label = 2
        # total 16 heads
        self.head_idx = [i for i in range(16)]
        # total 24 layers
        self.layer_idx = [23]
        # choose which feature to look, q: 0 k: 1 v: 2
        self.qkv_choice = 2
        self.inputdatadir = "/media/data/leo/style_vector_data/"
        self.all_classes_list = os.listdir(self.inputdatadir)
        self.all_images_list = os.listdir(self.inputdatadir + self.all_classes_list[self.class_label])
        self.num_classes = len(self.all_classes_list)
        assert self.class_label < self.num_classes
        self.num_images_in_picked_class = len(os.listdir(self.inputdatadir + self.all_classes_list[self.class_label]))
        random_list = []
        self.seed = 10
        self.num_patches = 16
        self.attention_channels = 64
        self.batch_size = 1
        self.num_pcs = 3 

        self.all_image_sizes = []
        np.random.seed(self.seed)
        while len(random_list) < self.batch_size:
            randnum = np.random.randint(0, len(self.all_images_list))
            if randnum not in random_list:
                random_list.append(randnum)
        self.picked_images_index = random_list

    def get_original_image_shapes(self):
        for idx in range(len(self.picked_images_index)):
            img_idx = self.picked_images_index[idx]
            image_file_path = self.inputdatadir + self.all_classes_list[self.class_label] + "/" + self.all_images_list[img_idx]
            print("processing image: ", self.all_images_list[img_idx])
            image = np.array(Image.open(image_file_path))
            self.all_image_sizes.append(image.shape[:2])


configs = configs()
configs.get_original_image_shapes()
print(configs.all_image_sizes)
visualizer = visualizer()
processor = processor()

# define pipe with components fixed
class ViTFeature:
    def __init__(self, configs, processor, visualizer):
        self.configs = configs
        self.processor = processor
        self.visualizer = visualizer
        self.pipe = ViTPipe(vit = self.configs.vit, scheduler = self.configs.scheduler, torch_device = 'cuda')
        self.data = torch.empty(configs.batch_size, 3, configs.imageH, configs.imageW).cuda()
        self.original_images = []
        self.all_image_name = []

    def read_all_images(self):
        for idx in range(len(self.configs.picked_images_index)):
            img_idx = self.configs.picked_images_index[idx]
            image_file_path = self.configs.inputdatadir + self.configs.all_classes_list[self.configs.class_label] + "/" + self.configs.all_images_list[img_idx]
            self.all_image_name.append(self.configs.all_images_list[img_idx])
            im = Image.open(image_file_path)
            self.original_images.append(np.array(im))
            image = configs.improcessor(im)["pixel_values"][0]
            self.data[idx] = torch.tensor(image)
        print("finished reading all images and resize ")
        print("all image names: ", self.all_image_name)
        print("all resized images: ", self.data.shape)

    def extract_ViT_features(self):        
        for layer in self.configs.layer_idx:
            for head in self.configs.head_idx:
                # batch of qkv for each layer/head index, differnt for each layer/head combination
                allfeatures = torch.empty(self.configs.batch_size, self.configs.num_patches * self.configs.num_patches, self.configs.attention_channels)
                # send batch of images into vit pipe and get qkv
                qkv = self.pipe(self.data, vit_input_size = self.configs.size, layer_idx = layer, head_idx = head)
                # all keys: 256*Num_images, channels
                allfeatures = qkv[self.configs.qkv_choice].reshape(self.configs.batch_size * self.configs.num_patches * self.configs.num_patches, self.configs.attention_channels)
                filenames = []
                for fileidx in range(self.configs.batch_size):
                    filenames.append("layer_{}_head_{}_class_{}_{}".format(layer, head, self.configs.all_classes_list[self.configs.class_label], self.all_image_name[fileidx]))
                print("shape of all features", allfeatures.shape)
                print("all filenames to be saved", filenames)
                self.processor.factor_analysis(allfeatures, self.original_images, self.configs.num_patches, self.configs.num_pcs, self.configs.all_image_sizes, self.configs.outputdir[configs.qkv_choice], filenames)
                del allfeatures


vitfeature = ViTFeature(configs, processor, visualizer)
vitfeature.read_all_images()
vitfeature.extract_ViT_features()

@dataclass
class configs:
    model_path = 'facebook/dinov2-large'
    link = "runwayml/stable-diffusion-v1-5"
    tokenizer = AutoTokenizer.from_pretrained(link, subfolder="tokenizer")
    processor = AutoImageProcessor.from_pretrained(model_path)
    size = [processor.crop_size["height"], processor.crop_size["width"]]
    mean, std = processor.image_mean, processor.image_std
    mean = torch.tensor(mean, device="cuda")
    std = torch.tensor(std, device="cuda")
    vit = Dinov2ModelwOutput.from_pretrained(model_path)
    prompt = "a high-quality image"
    seed = 20
    num_inference_steps = 100
    inputdatadir = "/media/data/leo/style_vector_data/"
    class_label = 0
    all_classes_list = os.listdir(inputdatadir)
    all_images_list = os.listdir(inputdatadir + all_classes_list[class_label])
    vae = AutoencoderKL.from_pretrained(link, subfolder="vae")
    text_encoder = CLIPTextModel.from_pretrained(link, subfolder="text_encoder")
    unet = UNet2DConditionModel.from_pretrained(link, subfolder="unet")
    scheduler = DDPMSchedulerwithGuidance.from_pretrained(link, subfolder="scheduler")
    #scheduler = myddpmscheduler.from_pretrained(link, subfolder = "scheduler")
    layeridx = 0
    head = [0]
    guidance_strength = 0.0
    guidance_range = [0, 150]
    num_tokens = 256
    trials = 1
    num_pcs = 3
    image_size = 224
    timestep = 50
        


configs = configs()
preprocess = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((configs.image_size, configs.image_size)),
        transforms.Normalize([0.5], [0.5]),
    ]
)

# make sure input image is float and in the range of 0 and 1
sample_image = torch.tensor(np.array(Image.open("church.jpg"))).unsqueeze(0).permute(0, 3, 1, 2) / 255.0
#sample_image = preprocess(Image.open("church.jpg"))

prompt = "a high quality image"
strengths = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
steps = [50, 200, 500, 700, 1000]
timesteps = []
for step in steps:
    timesteps.append([i for i in range(step-1, -1, -1)])

pipe = StableDiffusionImg2ImgPipeline(vit = configs.vit, vae=configs.vae, text_encoder=configs.text_encoder, 
                                                tokenizer=configs.tokenizer, unet=configs.unet, scheduler=configs.scheduler, 
                                                safety_checker=None, feature_extractor=None, image_encoder=None, requires_safety_checker=False)

for strength in strengths:
        output = pipe(vit_input_size=configs.size, vit_input_mean=configs.mean, vit_input_std=configs.std, layer_idx=configs.layeridx,
                      guidance_strength=configs.guidance_strength, prompt = prompt, 
                      image = sample_image, strength = strength,  num_inference_steps= 400, 
                      scheduler = configs.scheduler, return_dict= False)
        img_np = np.array(output[0][0])
        image_arr = torch.tensor(img_np).type(torch.uint8).numpy()
        img_pil = Image.fromarray(image_arr)
        img_pil.save("denoised_strength_" + str(strength) + ".jpg")

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
from scheduler import ViTScheduler
import shutil
from processor import processor
from scheduler import ViTScheduler
from pipeline import ViTPipe
from skimage.transform import resize
from visualization import HeatMap
import torchvision.transforms as transforms 
from torchvision.transforms import Resize

class configs:
    def __init__(self):
        self.model_path = 'facebook/dinov2-large'
        #self.link = "runwayml/stable-diffusion-v1-5"
        #self.tokenizer = AutoTokenizer.from_pretrained(self.link, subfolder="tokenizer")
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.size = [self.processor.crop_size["height"], self.processor.crop_size["width"]]
        self.mean, self.std = self.processor.image_mean, self.processor.image_std
        self.mean = torch.tensor(self.mean, device="cuda")
        self.std = torch.tensor(self.std, device="cuda")
        self.seed = 20
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
        self.layer_idx = [0]
        # choose which feature to look, q: 0 k: 1 v: 2
        self.qkv_choice = 2
        self.inputdatadir = "/media/data/leo/imagenet_images/"
        self.all_classes_list = os.listdir(self.inputdatadir)
        self.all_images_list = os.listdir(self.inputdatadir + self.all_classes_list[self.class_label])
        self.num_classes = len(self.all_classes_list)
        assert self.class_label < self.num_classes
        self.num_images_in_picked_class = len(os.listdir(self.inputdatadir + self.all_classes_list[self.class_label]))
        self.inputimagedir = "./media/"
        random_list = []
        self.seed = 1
        self.num_patches = 16
        self.attention_channels = 64
        self.batch_size = 1
        self.num_pcs = 3 

        self.all_image_sizes = []
        np.random.seed(self.seed)
        while len(random_list) < self.batch_size:
            randnum = np.random.randint(0, 50)
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
resizer = Resize(size = (configs.imageH, configs.imageW), antialias = True)
transform = transforms.Compose([transforms.PILToTensor()])

# define pipe with components fixed
pipe = ViTPipe(vit = configs.vit, scheduler = configs.scheduler, torch_device = 'cuda')

def extract_ViT_features(configs, processor, visualizer):
    data = torch.empty(configs.batch_size, 3, configs.imageH, configs.imageW).cuda()
    original_images = []
    all_image_name = []
    # firstly, read all images in original size and resized to send to vit
    for idx in range(len(configs.picked_images_index)):
        img_idx = configs.picked_images_index[idx]
        image_file_path = configs.inputdatadir + configs.all_classes_list[configs.class_label] + "/" + configs.all_images_list[img_idx]
        all_image_name.append(configs.all_images_list[img_idx])
        im = Image.open(image_file_path)
        original_images.append(np.array(im))
        image = transform(im)
        # resize image of random size into 224x224 as vit input
        image_resized = resizer(image)
        data[idx] = image_resized
    print("finished reading all images and resize ")
    print("all image names: ", all_image_name)
    print("all resized images: ", data.shape)
    
    for layer in configs.layer_idx:
        for head in configs.head_idx:
            # batch of qkv for each layer/head index, differnt for each layer/head combination
            allfeatures = torch.empty(configs.batch_size, configs.num_patches * configs.num_patches, configs.attention_channels)
            # send batch of images into vit pipe and get qkv
            qkv = pipe(data, vit_input_size = configs.size, vit_input_mean = configs.mean, vit_input_std = configs.std, 
            layer_idx = layer, head_idx = head)
            # all keys: 256*Num_images, channels
            allfeatures = qkv[configs.qkv_choice].reshape(configs.batch_size * configs.num_patches * configs.num_patches, configs.attention_channels)
            filenames = []
            for fileidx in range(configs.batch_size):
                filenames.append("layer_{}_head_{}_class_{}_{}".format(layer, head, configs.all_classes_list[configs.class_label], all_image_name[fileidx]))
            print("shape of all features", allfeatures.shape)
            print("all filenames to be saved", filenames)
            processor.factor_analysis(allfeatures, original_images, configs.num_patches, configs.num_pcs, configs.all_image_sizes, configs.outputdir[configs.qkv_choice], filenames)
            del allfeatures

extract_ViT_features(configs, processor, visualizer)

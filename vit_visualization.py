from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
from dinov2 import Dinov2ModelwOutput
import os
import numpy as np
import requests
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from visualizer import visualizer
from scheduler import ViTScheduler
import shutil
from processor import processor
from scheduler import ViTScheduler
from pipeline import ViTPipe
from skimage.transform import resize
from visualizer import HeatMap
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

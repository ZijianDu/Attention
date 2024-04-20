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
import shutil
from processor import processor
from skimage.transform import resize
from visualizer import HeatMap
import torchvision.transforms as transforms 
from torchvision.transforms import Resize
from dataclasses import dataclass
import sys


class ViTPipe():
    # define structural components needed for ViT
    def __init__(self, vit, scheduler, torch_device):
        self.vit = vit.to(torch_device)
        self.scheduler = scheduler
    
    # define variables/parameters needed to run the pipeline
    def __call__(self, image, vit_input_size, layer_idx, head_idx):
        qkv = self.scheduler.step(self.vit, image, vit_input_size, layer_idx, head_idx)
        return qkv
    
class ViTScheduler():
    def step(self, vit, images, vit_input_size, layer_idx, head_idx):
        vit(layer_idx, head_idx, images, output_attentions=True)
        qkv = vit.getqkv()
        return qkv
    
@dataclass
class vitconfigs:
    model_path = 'facebook/dinov2-large'
    improcessor = AutoImageProcessor.from_pretrained(model_path)
    size = [improcessor.crop_size["height"], improcessor.crop_size["width"]]
    mean, std = improcessor.image_mean, improcessor.image_std
    mean = torch.tensor(mean, device="cuda")
    std = torch.tensor(std, device="cuda")
    vit = Dinov2ModelwOutput.from_pretrained(model_path)
    scheduler = ViTScheduler()
    imageH = 224
    imageW = 224
    outputdir = "./outputs" 
    metricoutputdir = "./metrics"
    outputdir = ["./qkv/q/", "./qkv/k/", "./qkv/v/"]
    class_label = 2
    # total 16 heads
    num_heads = 16
    head_idx = [i for i in range(num_heads)]
    # total 24 layers
    layer_idx = [23]
    # choose which feature to look, q: 0 k: 1 v: 2
    qkv_choice = 1
    inputdatadir = "/media/data/leo/style_vector_data/"
    all_classes_list = os.listdir(inputdatadir)
    all_images_list = os.listdir(inputdatadir + all_classes_list[class_label])
    num_classes = len(all_classes_list)
    assert class_label < num_classes
    num_images_in_picked_class = len(os.listdir(inputdatadir + all_classes_list[class_label]))
    random_list = []
    seed = 10
    num_patches = 16
    attention_channels = 64
    batch_size = 1
    num_pcs = 3 
    ## -1 means ignore no head
    ignoreheadidx = -1
    all_image_sizes = []
    np.random.seed(seed)
    while len(random_list) < batch_size:
        randnum = np.random.randint(0, len(all_images_list))
        if randnum not in random_list:
            random_list.append(randnum)
    picked_images_index = random_list

class ViTFeature:
    def __init__(self, configs, processor, visualizer):
        self.configs = configs
        self.processor = processor
        self.visualizer = visualizer
        self.pipe = ViTPipe(vit = self.configs.vit, scheduler = self.configs.scheduler, torch_device = 'cuda')
        self.data = torch.empty(configs.batch_size, 3, configs.imageH, configs.imageW).cuda()
        self.original_images = []
        self.all_image_name = []
        self.single_image = None
        self.tensor_qkv = None

    def _get_feature_qkv(self):
        try:
            assert(self.tensor_qkv != None), "feature has not been extracted, call extract_Vit_features function first"
        except Exception as e:
            print(e)
        return self.tensor_qkv
    
    def read_all_images(self):
        for idx in range(len(self.configs.picked_images_index)):
            img_idx = self.configs.picked_images_index[idx]
            image_file_path = self.configs.inputdatadir + self.configs.all_classes_list[self.configs.class_label] + "/" + self.configs.all_images_list[img_idx]
            self.all_image_name.append(self.configs.all_images_list[img_idx])
            im = Image.open(image_file_path)
            self.original_images.append(np.array(im))
            print("original image shape: ", np.array(im).shape)
            image = self.configs.improcessor(im)["pixel_values"][0]
            self.data[idx] = torch.tensor(image)
            print("image shape after processing: ", self.data[idx].shape)
        print("finished reading all images and resize ")
        print("all image names: ", self.all_image_name)
        print("all resized images: ", self.data.shape)
    
    def read_one_image(self):
        im = Image.open("church.jpg")
        print("image shape: ", np.array(im).shape)
        image = self.configs.improcessor(im)["pixel_values"][0]
        self.data[0]= torch.tensor(image)
        print("finished reading single image and resize, shape: ", self.data[0].shape)
        

    # extract vit features 
    # parameters: qkv_choice to pick q/k/v
    # when ignoreheadidx == None: pick all heads, if int, ignore this head
    def _extract_ViT_features(self, ignoreheadidx):
        if ignoreheadidx == -1:
            tensor_qkv = torch.empty(1, self.configs.num_heads, self.configs.num_patches * self.configs.num_patches, self.configs.attention_channels).cuda()
            for layer in self.configs.layer_idx:
                for head in self.configs.head_idx:
                    # batch of qkv for each layer/head index, differnt for each layer/head combination
                    qkv = self.pipe(self.data, vit_input_size = self.configs.size, layer_idx = layer, head_idx = head)
                    tensor_qkv[:, head, :, :] = qkv[self.configs.qkv_choice]
            assert tensor_qkv.shape[0] == 1 and tensor_qkv.shape[1] == self.configs.num_heads
            assert tensor_qkv.shape[2] == self.configs.num_patches ** 2, tensor_qkv.shape[3] == self.configs.attention_channels
            self.tensor_qkv = tensor_qkv
        else: 
            assert ignoreheadidx >= 0 and ignoreheadidx < 16
            tensor_qkv = torch.zeros(1, 1,  self.configs.num_patches * self.configs.num_patches, self.configs.attention_channels).cuda()
            for layer in self.configs.layer_idx:
                for head in self.configs.head_idx:
                    print(head, ignoreheadidx)
                    # batch of qkv for each layer/head index, different for each layer/head combination
                    if head != ignoreheadidx:
                        qkv = self.pipe(self.data, vit_input_size = self.configs.size, layer_idx = layer, head_idx = head)
                        tensor_qkv = torch.cat((tensor_qkv, qkv[self.configs.qkv_choice].unsqueeze(0)), 1)
                    else:
                        continue
            tensor_qkv = tensor_qkv[:, 1:, :, :]
            assert tensor_qkv.shape[0] == 1 and tensor_qkv.shape[1] == self.configs.num_heads - 1
            assert tensor_qkv.shape[2] == self.configs.num_patches ** 2, tensor_qkv.shape[3] == self.configs.attention_channels
            self.tensor_qkv = tensor_qkv

    def extract_ViT_features(self):
        self.read_one_image()
        self._extract_ViT_features(self.configs.ignoreheadidx)


def get_original_image_shapes(self, configs):
    for idx in range(len(configs.picked_images_index)):
        img_idx = configs.picked_images_index[idx]
        image_file_path = configs.inputdatadir + configs.all_classes_list[configs.class_label] + "/" + configs.all_images_list[img_idx]
        print("processing image: ", configs.all_images_list[img_idx])
        image = np.array(Image.open(image_file_path))
        configs.all_image_sizes.append(image.shape[:2])

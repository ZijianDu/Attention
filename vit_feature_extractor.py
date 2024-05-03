from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
from PIL import Image
from dinov2 import Dinov2ModelwOutput
import os
import numpy as np
import requests
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from util import visualizer, HeatMap
import shutil
from skimage.transform import resize
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
    def __call__(self, image, vit_input_size, layer_idx):
        return self.scheduler.step(self.vit, image, vit_input_size, layer_idx)
    
# scheduler returen q/k/v of vit output of particular layer index
class ViTScheduler():
    def step(self, vit, images, vit_input_size, layer_idx):
        vit(layer_idx, images, output_attentions=False)
        return vit.getkey()
    
class ViTFeature:
    def __init__(self, configs):
        self.configs = configs
        self.pipe = ViTPipe(vit = self.configs.vit, scheduler = self.configs.vitscheduler, torch_device = 'cuda')
        # data is direct input to ViT model, shape: batch, 3, 224, 224, preprocessing needed if 
        # original image is of different shape
        self.data = torch.empty(configs.batch_size, 3, configs.imageH, configs.imageW).cuda()
        self.original_images = []
        self.all_image_name = []
        # write into feature qkv regardless of image or latent as input
        self.vit_features = None
        # write here if input is original image
        self.original_image_features = None
        # write here if input is latent image
        self.latent_image_features = None

        self.all_original_vit_features = None

        self.selected_latent_vit_features = None
    
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
        im = Image.open(self.configs.base_folder + self.configs.single_image_name)
        image = self.configs.improcessor(im)["pixel_values"][0]
        self.data[0]= torch.tensor(image)

    # this function call vit pipeline to get original clean image's features of all heads
    # call this function once in main pipeline
    def extract_all_original_vit_features(self):
        self.read_one_image()
        keys = torch.empty(1, self.configs.num_heads, self.configs.num_patches * self.configs.num_patches, self.configs.attention_channels).cuda()
        # get vit feature of all head of original clean image, cherry pick which head as needed later
        keys = self.pipe(self.data, vit_input_size = self.configs.size, layer_idx = self.configs.layer_idx[0])
        assert keys.shape[0] == 1 and keys.shape[1] == self.configs.num_heads
        assert keys.shape[2] == self.configs.num_patches ** 2, keys.shape[3] == self.configs.attention_channels
        self.all_original_vit_features = keys
        del keys

    def get_all_original_vit_features(self):
        assert self.all_original_vit_features != None
        assert self.all_original_vit_features.shape[0] == self.configs.batch_size
        assert self.all_original_vit_features.shape[1] == self.configs.num_heads
        assert self.all_original_vit_features.shape[2] == self.configs.num_patches * self.configs.num_patches
        assert self.all_original_vit_features.shape[3] == self.configs.attention_channels
        return self.all_original_vit_features

    # extract features of selected head's feature of the predicted x0 from xt, need to call everytime there is a new latent
    def extract_selected_latent_vit_features(self, latent):
        # sanity check
        for headidx in self.configs.current_selected_heads:
            assert headidx >= 0 
            assert headidx < self.configs.num_heads
        keys = torch.zeros(1, 1, self.configs.num_patches * self.configs.num_patches, self.configs.attention_channels).cuda()
        for head in self.configs.current_selected_heads:
            self.data[0] = latent
            qkv = self.pipe(self.data, vit_input_size = self.configs.size, layer_idx = self.configs.layer_idx[0])
            keys = torch.cat((keys, qkv[:, head, :, :].unsqueeze(1)), 1)
            del qkv
        keys = keys[:, 1:, :, :]
        assert keys.shape[0] == 1 and keys.shape[1] == len(self.configs.current_selected_heads)
        assert keys.shape[2] == self.configs.num_patches ** 2 and keys.shape[3] == self.configs.attention_channels
        self.selected_latent_vit_features = keys
        del keys

    def get_selected_latent_vit_features(self):
        assert self.selected_latent_vit_features != None
        assert self.selected_latent_vit_features.shape[0] == self.configs.batch_size
        assert self.selected_latent_vit_features.shape[1] == len(self.configs.current_selected_heads)
        assert self.selected_latent_vit_features.shape[2] == self.configs.num_patches * self.configs.num_patches
        assert self.selected_latent_vit_features.shape[3] == self.configs.attention_channels
        return self.selected_latent_vit_features


def get_original_image_shapes(self, configs):
    for idx in range(len(configs.picked_images_index)):
        img_idx = configs.picked_images_index[idx]
        image_file_path = configs.inputdatadir + configs.all_classes_list[configs.class_label] + "/" + configs.all_images_list[img_idx]
        print("processing image: ", configs.all_images_list[img_idx])
        image = np.array(Image.open(image_file_path))
        configs.all_image_sizes.append(image.shape[:2])

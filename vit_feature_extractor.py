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
    def __init__(self, vit):
        self.vit = vit
    # define variables/parameters needed to run the pipeline
    def __call__(self, image, layer_idx):
        self.vit(layer_idx, image, output_attentions=False)
        return self.vit.getkey()

class ViTFeature(ViTPipe):
    def __init__(self, configs, vit):
        self.configs = configs
        self.pipe = ViTPipe(vit = vit)
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
            image = self.configs.processor(im)["pixel_values"][0]
            self.data[idx] = torch.tensor(image)
            print("image shape after processing: ", self.data[idx].shape)
        print("finished reading all images and resize ")
        print("all image names: ", self.all_image_name)
        print("all resized images: ", self.data.shape)
    
    def read_one_image(self):
        im = Image.open(self.configs.base_folder + self.configs.single_image_name)
        image = self.configs.processor(im)["pixel_values"][0]
        self.data[0]= torch.tensor(image)

    # this function call vit pipeline to get original clean image's features of all heads
    # call this function once in main pipeline
    def extract_all_original_vit_features(self):
        # sanity check
        for headidx in self.configs.current_selected_heads:
            assert headidx >= 0 
            assert headidx < self.configs.num_heads
        self.read_one_image()
        # select the original vit features for specified heads
        keys = self.pipe(self.data, layer_idx = self.configs.layer_idx[0])
        indices = torch.tensor(self.configs.current_selected_heads).cuda()
        selected_original_vit_features = torch.index_select(keys, 1, indices)

        assert selected_original_vit_features.shape[0] == self.configs.batch_size
        assert selected_original_vit_features.shape[1] == len(self.configs.current_selected_heads)
        assert selected_original_vit_features.shape[2] == self.configs.num_patches * self.configs.num_patches
        assert selected_original_vit_features.shape[3] == self.configs.attention_channels
        return selected_original_vit_features

    # extract features of selected head's feature of the predicted x0 from xt, need to call everytime there is a new latent
    def extract_selected_latent_vit_features(self, latent):
        # sanity check
        for headidx in self.configs.current_selected_heads:
            assert headidx >= 0 
            assert headidx < self.configs.num_heads

        # return key in the vit attention 
        keys = self.pipe(latent, layer_idx = self.configs.layer_idx[0])
        # choose the selected heads
        indices = torch.tensor(self.configs.current_selected_heads).cuda()
        selected_latent_vit_features = torch.index_select(keys, 1, indices)

        assert selected_latent_vit_features.shape[0] == self.configs.batch_size
        assert selected_latent_vit_features.shape[1] == len(self.configs.current_selected_heads)
        assert selected_latent_vit_features.shape[2] == self.configs.num_patches * self.configs.num_patches
        assert selected_latent_vit_features.shape[3] == self.configs.attention_channels
        return selected_latent_vit_features


def get_original_image_shapes(self, configs):
    for idx in range(len(configs.picked_images_index)):
        img_idx = configs.picked_images_index[idx]
        image_file_path = configs.inputdatadir + configs.all_classes_list[configs.class_label] + "/" + configs.all_images_list[img_idx]
        print("processing image: ", configs.all_images_list[img_idx])
        image = np.array(Image.open(image_file_path))
        configs.all_image_sizes.append(image.shape[:2])

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
    def __call__(self, image, vit_input_size, layer_idx):
        return self.scheduler.step(self.vit, image, vit_input_size, layer_idx)
    
class ViTScheduler():
    def step(self, vit, images, vit_input_size, layer_idx):
        vit(layer_idx, images, output_attentions=False)
        return vit.getkey()
    
class ViTFeature:
    def __init__(self, configs, layer_idx, processor):
        self.configs = configs
        self.layer_idx = layer_idx
        self.processor = processor
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

    # isoriginal: if return the vit feature of original clean image or predicted x0 through diffusion latent
    def _get_feature_qkv(self, isoriginal = False):
        if isoriginal:
            try:
                assert(self.original_image_features!= None), "original image feature has not been extracted, call extract_Vit_features function first"
            except Exception as e:
                print(e)
            return self.original_image_features
        else:
            try:
                assert(self.latent_image_features != None), "latent image feature has not been extracted, call extract_Vit_features function first"
            except Exception as e:
                print(e)
            return self.latent_image_features
    
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
        im = Image.open(self.configs.single_image_name)
        image = self.configs.improcessor(im)["pixel_values"][0]
        self.data[0]= torch.tensor(image)

    # extract vit features 
    # parameters: qkv_choice to pick q/k/v
    # when ignoreheadidx == None: pick all heads, if int, ignore this head
    def _extract_ViT_features(self, ignoreheadidx):
        if ignoreheadidx == -1:
            keys = torch.empty(1, self.configs.num_heads, self.configs.num_patches * self.configs.num_patches, self.configs.attention_channels).cuda()
            # batch of qkv for each layer/head index, differnt for each layer/head combination
            keys = self.pipe(self.data, vit_input_size = self.configs.size, layer_idx = self.layer_idx)
            assert keys.shape[0] == 1 and keys.shape[1] == self.configs.num_heads
            assert keys.shape[2] == self.configs.num_patches ** 2, keys.shape[3] == self.configs.attention_channels
            self.vit_features = keys
            del keys
        ## needs to be updated
        else: 
            assert ignoreheadidx >= 0 and ignoreheadidx < 16
            tensor_qkv = torch.zeros(1, 1,  self.configs.num_patches * self.configs.num_patches, self.configs.attention_channels).cuda()
            for layer in self.configs.layer_idx:
                for head in self.configs.head_idx:
                    # batch of qkv for each layer/head index, different for each layer/head combination
                    if head != ignoreheadidx:
                        qkv = self.pipe(self.data, vit_input_size = self.configs.size, layer_idx = layer, head_idx = head)
                        tensor_qkv = torch.cat((tensor_qkv, qkv[self.configs.qkv_choice].unsqueeze(0)), 1)
                        del qkv
                    else:
                        continue
            tensor_qkv = tensor_qkv[:, 1:, :, :]
            assert tensor_qkv.shape[0] == 1 and tensor_qkv.shape[1] == self.configs.num_heads - 1
            assert tensor_qkv.shape[2] == self.configs.num_patches ** 2, tensor_qkv.shape[3] == self.configs.attention_channels
            self.feature_qkv = tensor_qkv
            del tensor_qkv

    # extract original image features
    def extract_image_ViT_features(self):
        self.read_one_image()
        self._extract_ViT_features(self.configs.ignoreheadidx)
        self.original_image_features = self.vit_features
        self.vit_features = None
    
    # extract feature of the latent image diffusion
    def extract_latent_ViT_features(self, latent):
        # overrite data by the input latent
        self.data[0] = latent
        self._extract_ViT_features(self.configs.ignoreheadidx)
        self.latent_image_features = self.vit_features
        # set featue to none to prepare for next latent
        self.vit_features = None

def get_original_image_shapes(self, configs):
    for idx in range(len(configs.picked_images_index)):
        img_idx = configs.picked_images_index[idx]
        image_file_path = configs.inputdatadir + configs.all_classes_list[configs.class_label] + "/" + configs.all_images_list[img_idx]
        print("processing image: ", configs.all_images_list[img_idx])
        image = np.array(Image.open(image_file_path))
        configs.all_image_sizes.append(image.shape[:2])

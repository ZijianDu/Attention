from diffusers import DDIMScheduler
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
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

from transformers.models.clip import CLIPTextModel
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

import shutil
import unittest

# class to do specialized processing to data
import numpy as np
import logging

import matplotlib.pyplot as plt
from numpy.random import RandomState
from torchvision import transforms as T
from sklearn import cluster, decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from PIL import Image


import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from PIL import Image
import os
import scipy.ndimage as ndimage


class processor():
    def __init__(self):
        # U, S, V of PCA
        self.U = None 
        self.S = None
        self.V = None
        self.projection = None
        self.coefficients = None

    # factor analysis for q, k, v including pca, ica, etc
    # data dimension: (B*H*W) x channels
    # resize shape is a list of H*W, need to resize key map into original image size and overlay
    # filenames is list of name, one image per original image
    
    def factor_analysis(self, allkeys, allimages, num_patches, num_pc, resize_shape, filepath, filenames):
        allkeys = allkeys.cpu().data.numpy()
        pca = decomposition.PCA(n_components = num_pc)
        print("shape before pca", allkeys.shape)
        # normalization before doing PCA
        scaler = StandardScaler()
        feature_map_normed = scaler.fit_transform(allkeys)
        print("shape after normalization", feature_map_normed.shape)
        # fit pca
        feature_map_pca = pca.fit_transform(feature_map_normed)
        print("after pca", feature_map_pca.shape)

        pca_map_min, pca_map_max = feature_map_pca.min(axis = 0), feature_map_pca.max(axis = 0)
        # normalize pca map
        pca_map_normed = (feature_map_pca - pca_map_min) / (pca_map_max - pca_map_min)

        pca_img_reshaped = (pca_map_normed * 255).astype(np.uint8).reshape((len(resize_shape), num_patches, num_patches, num_pc))
        print("pca after reshape", pca_img_reshaped.shape)

        # resize each pca into input image size and plot overlay

        print("filenames before saving images", filenames)
        print("filepath", filepath)
        for i in range(len(resize_shape)):
            one_pca_img = Image.fromarray(pca_img_reshaped[i])
            one_resized_pc_image = np.array(T.Resize(resize_shape[i], interpolation = T.InterpolationMode.BILINEAR)(one_pca_img))
            one_image = allimages[i]
            assert len(one_image.shape) == len(one_resized_pc_image.shape)
            for shapeidx in range(len(one_image.shape)):
                assert one_image.shape[shapeidx] == one_resized_pc_image.shape[shapeidx]
            
            print("original image shape", one_image.shape)
            print("pc image shape", one_resized_pc_image.shape)
            heatmap = HeatMap(one_image, one_resized_pc_image, 1.0)
            heatmap.save(filenames[i], save_path = filepath)
            del heatmap




class visualizer:
    def __init__(self):
        self.figsize = (15, 15)
        self.dpi = 200


    def plot(self, data, name):
        plt.figure(figsize = self.figsize)
        plt.plot(data)
        plt.savefig(name)
        
    def saveimages(self, images):
        for i, im in enumerate(images):
            img = im[0].transpose(1, 2, 0) * 255 
            img = np.clip(img.astype(np.uint8), 0, 255)
            img = Image.fromarray(img)
            img.save(f"./outputs/{i:02d}.png")
    
    def plot_attentionmap(self, map, timestep, name, folder):
        map = map.cpu().data.numpy() 
        fig, ax = plt.subplots()
        plt.imshow(map, cmap='viridis', interpolation='nearest')
        plt.savefig(folder + name)

    def _plot_ssim(self, ssim):
        assert len(self.all_params) == len(ssim)
        strength = [i[0] for i in self.all_params]
        guidance = [i[1] for i in self.all_params]
        idx = [i[2] for i in self.all_params]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(strength, np.log(guidance), idx, c=np.sqrt(ssim), cmap='plasma')
        ax.set_xlabel("diffusion strength")
        ax.set_ylabel("log(vit guidance)")
        ax.set_zlabel("layer index")
        ax.set_title("SSIM")
        fig.colorbar(scatter, ax=ax)
        plt.savefig(self.single_image_name[:-4] + "_seed_" +str(self.seed)+".jpg", dpi = self.dpi)




class HeatMap:
    def __init__(self, image, heat_map, gaussian_std=10):
        #if image is numpy array
        if isinstance(image,np.ndarray):
            height = image.shape[0]
            width = image.shape[1]
            self.image = image
        else: 
            #PIL open the image path, record the height and width
            image = Image.open(image)
            width, height = image.size
            self.image = image
        
        #Convert numpy heat_map values into image formate for easy upscale
        #Rezie the heat_map to the size of the input image
        #Apply the gausian filter for smoothing
        #Convert back to numpy
        self.heat_map = heat_map
    
    #Plot the figure
    def plot(self,transparency=0.7,color_map='bwr',
             show_axis=False, show_original=False, show_colorbar=False,width_pad=0):
            
        #If show_original is True, then subplot first figure as orginal image
        #Set x,y to let the heatmap plot in the second subfigure, 
        #otherwise heatmap will plot in the first sub figure
    
        
        #Plot the heatmap
        plt.subplot(1, 1, 1)
        if not show_axis:
            plt.axis('off')
        plt.imshow(self.image)
        plt.imshow(self.heat_map/255, alpha=transparency, cmap=color_map)
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout(w_pad=width_pad)
        plt.show()
    
    ###Save the figure
    def save(self, filename, save_path = None,
             transparency=0.6,color_map='viridis',width_pad = -10,
             show_axis=True, show_colorbar=True, **kwargs):
        
        if not show_axis:
            plt.axis('off')
        plt.subplot(1, 1, 1)
        plt.imshow(self.image)
        plt.imshow(self.heat_map/255.0, alpha=transparency, cmap=color_map)
        plt.colorbar()

        print("filename before saving figure: ", save_path + filename)
        plt.savefig(save_path+filename, pad_inches = 0.5)
    



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





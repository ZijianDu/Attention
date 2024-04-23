import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from PIL import Image
import os
import scipy.ndimage as ndimage

class visualizer():
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
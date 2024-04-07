import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from PIL import Image
import os
import matplotlib
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

class visualizer():
    def __init__(self):
        self.figsize = (10, 10)
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

    def plot_qkv(allqkv, iteration, key):
        pass


class HeatMap:
    def __init__(self,image,heat_map,gaussian_std=10):
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
        heatmap_image = Image.fromarray(heat_map*255)
        heatmap_image_resized = heatmap_image.resize((width,height))
        heatmap_image_resized = ndimage.gaussian_filter(heatmap_image_resized, 
                                                        sigma=(gaussian_std, gaussian_std), 
                                                        order=0)
        heatmap_image_resized = np.asarray(heatmap_image_resized)
        self.heat_map = heatmap_image_resized
    
    #Plot the figure
    def plot(self,transparency=0.7,color_map='bwr',
             show_axis=False, show_original=False, show_colorbar=False,width_pad=0):
            
        #If show_original is True, then subplot first figure as orginal image
        #Set x,y to let the heatmap plot in the second subfigure, 
        #otherwise heatmap will plot in the first sub figure
        if show_original:
            plt.subplot(1, 2, 1)
            if not show_axis:
                plt.axis('off')
            plt.imshow(self.image)
            x,y=2,2
        else:
            x,y=1,1
        
        #Plot the heatmap
        plt.subplot(1,x,y)
        if not show_axis:
            plt.axis('off')
        plt.imshow(self.image)
        plt.imshow(self.heat_map/255, alpha=transparency, cmap=color_map)
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout(w_pad=width_pad)
        plt.show()
    
    ###Save the figure
    def save(self,filename,format='png',save_path=os.getcwd(),
             transparency=0.7,color_map='bwr',width_pad = -10,
             show_axis=False, show_original=False, show_colorbar=False, **kwargs):
        if show_original:
            plt.subplot(1, 2, 1)
            if not show_axis:
                plt.axis('off')
            plt.imshow(self.image)
            x,y=2,2
        else:
            x,y=1,1
        
        #Plot the heatmap
        plt.subplot(1,x,y)
        if not show_axis:
            plt.axis('off')
        plt.imshow(self.image)
        plt.imshow(self.heat_map/255, alpha=transparency, cmap=color_map)
        if show_colorbar:
            plt.colorbar()
        plt.tight_layout(w_pad=width_pad)
        plt.savefig(os.path.join(save_path,filename+'.'+format), 
                    format=format, 
                    bbox_inches='tight',
                    pad_inches = 0, **kwargs)
        print('{}.{} has been successfully saved to {}'.format(filename,format,save_path))
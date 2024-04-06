import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
from PIL import Image

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

    def plot_qkv(allqkv, iteration, layeridx, key):
        pass
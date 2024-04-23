import pytorch_ssim
import os
import torch
from configs import sdimg2imgconfigs
from PIL import Image
from collections import defaultdict
import numpy as np
from visualizer import visualizer

class evaluator(sdimg2imgconfigs, visualizer):
    # run this function after each seed
    def get_SSIMs(self):
        # read all generated images
        all_image_names = os.listdir(self.outputdir + "seed" + str(self.seed))
        all_ssim = []
        for img_name in all_image_names:
            image = self.improcessor(
                Image.open(self.outputdir + "seed" + str(self.seed) + "/" + img_name))["pixel_values"][0]
            all_ssim.append(pytorch_ssim.ssim(torch.tensor(image).unsqueeze(0), 
                                                      torch.tensor(self.single_image).unsqueeze(0)).data)
        return all_ssim
    
    def plot_ssim(self, ssim):
        self._plot_ssim(ssim)
        
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
from visualization import HeatMap

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


        


    


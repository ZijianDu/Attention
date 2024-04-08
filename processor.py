# class to do specialized processing to data
import numpy as np
import logging

import matplotlib.pyplot as plt
from numpy.random import RandomState
from torchvision import transforms as T
from sklearn import cluster, decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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
    
    def factor_analysis(self, data, num_pc, resize_shape, filepath, filenames):
        data = data.cpu().data.numpy()
        pca = decomposition.PCA(n_components = num_pc)
        print("shape before pca", data.shape)
        # normalization before doing PCA
        scaler = StandardScaler()
        feature_map_normed = scaler.fit_transform(data)
        print("after norm", feature_map_normed.mean(axis = 1), feature_map_normed.std(axis = 1))
        # fit pca
        feature_map_pca = pca.transform(feature_map_normed)
        print("after pca", feature_map_pca.shape)

        pca_map_min, pca_map_max = feature_map_pca.min(axis = (0, 1)), feature_map_pca.max(axis = (0, 1))
        # normalize pca map
        pca_map_normed = (feature_map_pca - pca_map_min) / (pca_map_max - pca_map_min)

        print("after norm", pca_map_normed.mean(axis = 1), pac_map_normed.std(axis = 1))
        pca_img = Image.fromarray((pca_map_normed * 255).astype(np.uint8))
        # resize to input image size 
        pca_imge = T.Resize(resize_shape, interpolation = T.InterpolationMode.NEAREST)(pca_img)
        pca_img.save(filepath + filename)


        


    


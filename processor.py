# class to do specialized processing to data
import numpy as np
import logging

import matplotlib.pyplot as plt
from numpy.random import RandomState

from sklearn import cluster, decomposition


class processor():
    def __init__(self):
        # U, S, V of PCA
        self.U = None 
        self.S = None
        self.V = None
        self.projection = None
        self.coefficients = None

    # factor analysis for q, k, v including pca, ica, etc
    # data dimension: # of data instances, # of data dimension
    def factor_analysis(self, data, num_pc, type):
        data = data.cpu().data.numpy() 
        rng = RandomState(0)
        n_samples, n_dimensions = data.shape[0], data.shape[1]
        print("number of data: %5d, dimension of data: %5d" % (data.shape[0], data.shape[1]))
        assert num_pc <= data.shape[1]
        # demean
        data_centered = data - data.mean(axis = 0)
        # local demean
        data_centered -= data_centered.mean(axis=1).reshape(n_samples, -1)
        if type == "pca":
            pca_estimator = decomposition.PCA(n_components = num_pc, svd_solver = "randomized", whiten = True)
            pca_estimator.fit(data_centered)
            return pca_estimator.components_[:num_pc]

        


    


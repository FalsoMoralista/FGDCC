import torch
import faiss
import faiss.contrib.torch_utils

import numpy as np
import random
np.random.seed(0)


class SoftKMeans:
    def __init__(self, nb_classes, k_range=[2,3,4,5], d=1280):
        
        self.k_range = k_range
        self.res = faiss.StandardGpuResources()
        self.device = 0       
        self.d = d
        
        self.nb_classes = nb_classes

        self.n_kmeans = []
        for j in range(nb_classes):
            self.n_kmeans.append([faiss.index_cpu_to_gpu(self.res, self.device, faiss.IndexFlatL2(d)) for k in self.k_range])

    '''
        Initialize cluster prototype/centroids by selecting k points at random.  
        This is worse than K-means -- at least initialization-wise -- as although k-means uses the same random initialization strategy
        it then optimizes the centroids until they converge to a certain stable configuration. 
    '''
    def init_centroids(self, feature_bank):
        classes = feature_bank.keys()
        for j in classes:
            feature_list = feature_bank[j] # Gather features from class j
            kmeans_j = self.n_kmeans[j]
            # For each k hypothesis do:
            for k_index, k in enumerate(self.k_range):
                random_centroids = random.sample(feature_list, k) # Select k random images from feature list to init the centroids
                random_centroids = torch.stack(random_centroids)
                kmeans_j[k_index].add(random_centroids) # add each prototype to the index
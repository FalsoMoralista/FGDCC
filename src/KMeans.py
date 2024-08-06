# Author: Luciano Filho

import torch
import torch.distributed as dist

import faiss
import faiss.contrib.torch_utils

import itertools

import torch.nn.functional as F

import numpy as np
np.random.seed(0) # TODO: modify

import time

from src.utils.logging import (
    CSVLogger,
    AverageMeter)

class KMeansModule:

    def __init__(self, nb_classes, dimensionality=256, n_iter=300, tol=1e-4, k_range=[2,3,4,5], resources=None, config=None):
        
        self.resources = resources
        self.config = config
        self.nb_classes = nb_classes

        self.k_range = k_range
        self.d = dimensionality
        self.max_iter = n_iter
        self.tol = tol
        
        # Create the K-means object
        if len(k_range) == 1:
            self.n_kmeans = [faiss.Kmeans(d=dimensionality, k=k_range[0], niter=1, verbose=True, min_points_per_centroid = 1 ) for _ in range(nb_classes)]   
        else:
            self.n_kmeans = []   
            for _ in range(nb_classes):
                self.n_kmeans.append([faiss.Kmeans(d=dimensionality, k=k, niter=1, verbose=False, min_points_per_centroid = 1) for k in k_range])                                                            

    def inter_cluster_separation(self, cls, device):
        def two_by_two_combinations(values):
            return list(itertools.combinations(values, 2))

        S_score = torch.zeros(len(self.k_range), device=device)
        target_K_Means = self.n_kmeans[cls]

        for k_i, k in enumerate(self.k_range):
            centroids = target_K_Means[k_i].centroids
            pairs = torch.tensor(two_by_two_combinations(range(k)), device=device)

            # Extract the centroid pairs in a single operation
            centroid_pairs = centroids[pairs]
            
            # Compute cosine similarity in a vectorized way
            cosine_similarities = F.cosine_similarity(centroid_pairs[:, 0], centroid_pairs[:, 1], dim=1)
            
            # Accumulate the sum of cosine similarities
            S_score[k_i] = cosine_similarities.sum()
        return S_score

    def cosine_cluster_index(self, xb, yb, current_cache, last_epoch_cache, device):

        batch_size = xb.size(0)
        D_batch = []
        best_K_values = torch.zeros(batch_size, dtype=torch.int32, device=device)
        for i in range(batch_size):
            class_id = yb[i].item()
            sample = xb[i].unsqueeze(0)

            image_list = current_cache.get(class_id, last_epoch_cache.get(class_id))
            batch_x = torch.stack(image_list).to(device, dtype=torch.bfloat16)
            batch_x = torch.cat((batch_x, sample), dim=0).detach()

            S_scores = self.inter_cluster_separation(class_id, device=device)
            target_K_Means = self.n_kmeans[class_id]
            
            D_k = [] 
            C_scores = torch.zeros(len(self.k_range), device=device)
            for k_i, k_range in enumerate(self.k_range):

                D, batch_assignments = target_K_Means[k_i].index.search(batch_x, 1)
                batch_assignments = batch_assignments.squeeze(-1)
                D_k.append(D[0])
                centroids = target_K_Means[k_i].centroids
            
                centroid_list = centroids[batch_assignments] 
                # Computes the cosine similarity between every image and the cluster centroid to which is associated to
                C_score = F.cosine_similarity(batch_x, centroid_list)
                C_scores[k_i] = C_score.sum()
            D_batch.append(torch.stack(D_k))
            CCI = S_scores / (C_scores + S_scores)
            best_K_values[i] = CCI.argmax().item()                    
        D_batch = torch.stack(D_batch)
        return D_batch, best_K_values

    def initialize_centroids(self, batch_x, class_id, resources, rank, device, config, cached_features):
        
        def augment(x, n_samples): # Built as a helper function for development
            # Workaround to handle faiss centroid initialization with a single sample.
            # We built upon Mathilde Caron's idea of adding perturbations to the data, but we do it randomly instead.
            augmented_data = x.repeat((n_samples, 1))
            for i in range((n_samples)):
                sign = (torch.randint(0, 3, size=(self.d,)) - 1)
                sign = sign.to(device=device, dtype=torch.float32)
                eps = torch.tensor(1e-7, dtype=torch.float32, device=device)   
                augmented_data[i] += sign * eps                
            return augmented_data 
        
        if cached_features is None:
            print('Augment shouldnt be used, exitting...')
            exit(0)
            batch_x = augment(batch_x, self.k_range[len(self.k_range)-1]) # Create additional synthetic points to meet the minimum requirement for the number of clusters.             
        else:
            image_list = cached_features[class_id] # Otherwise use the features cached from the previous epoch                
            batch_x = torch.stack(image_list)
        for k in range(len(self.k_range)):
            self.n_kmeans[class_id][k].train(batch_x.detach().cpu()) # Then train K-means model for one iteration to initialize centroids

            #gpu_index_flat = faiss.GpuIndexFlatL2(resources, self.d, config)
            #gpu_index_flat = faiss.index_cpu_to_gpu_multiple(resources, devices=[rank],index=index_flat)
            #gpu_index_flat = faiss.GpuIndexFlatL2(resources[rank], self.d, config[rank])
            #gpu_index_flat = self.my_index_cpu_to_gpu_multiple(resources, index_flat, gpu_nos=[0,1,2,3,4,5,6,7])
            
            # Replace the regular index by a gpu one            
            index_flat = self.n_kmeans[class_id][k].index
            gpu_index_flat = faiss.index_cpu_to_gpu(self.resources, rank, index_flat)
            self.n_kmeans[class_id][k].index = gpu_index_flat
        
    def init(self, resources, rank, device, config, cached_features):
        # Initialize the centroids for each class
        for key in cached_features.keys():
            self.initialize_centroids(batch_x=None,
                                      class_id=key,
                                      resources=resources,
                                      rank=rank,
                                      device=device,
                                      config=config,
                                      cached_features=cached_features)             
    
    def batch_assignment(self, xb, y):
        # Assign the vectors to the nearest centroid
        D_k, I_k = [], []
        for k in range(len(self.k_range)):
            D, I = self.n_kmeans[y][k].index.search(xb, 1)
            D_k.append(D[:, 0])
            I_k.append(I[:, 0])            
        D_batch = torch.stack(D_k)
        I_batch = torch.stack(I_k)

        return D_batch, I_batch    
    
    '''
        Assigns a single data point to the set of clusters correspondent to the y target.

        Centroid initialization is going to be a problem in the first epoch because we wouldn't have the features cached yet, after that we can use the previous epoch's
        cache to initialize the centroids appropriately. Despite that, initialization is a problem for the first epochs despite of everything. That's because we are 
        using the reduced dimension features provided by an autoencoder that is going to be simultaneously trained to reduce the dimensionality of the features 
        provided by a ViT encoder that haven't properly learned good features yet. Therefore in the first few epochs, the initialization is expected to be very poor.
        
        Other than that we would still could have residual problems due to this initialization process.
        Because of that, another regularization mechanism that we could do to circumvent this is to reset the K-means
        centroids after every N epochs. (Another ablation parameter).  

    '''
    def assign(self, x, y, resources=None, rank=None, device=None, cached_features=None):
        D_batch = []
        I_batch = []        
        for i in range(len(x)):
            # Expand dims
            batch_x = x[i].unsqueeze(0)
            class_id = y[i].item()
            # Initialize the centroids if it haven't already been initialized
            if self.n_kmeans[y[i]][0].centroids is None:
                # FIXME this won't work inside the training loop because inside it batch_x is not on cpu but since this isn't expected to be called should be safe for now
                self.initialize_centroids(batch_x, class_id, resources, rank, device, cached_features)                
            # Assign the vectors to the nearest centroid
            D_k, I_k = [], []
            for k in range(len(self.k_range)):
                D, I = self.n_kmeans[class_id][k].index.search(batch_x, 1)
                D_k.append(D[0])
                I_k.append(I[0])
            D_batch.append(torch.stack(D_k))
            I_batch.append(torch.stack(I_k))
        D_batch = torch.stack((D_batch))
        I_batch = torch.stack((I_batch))
        return D_batch, I_batch

    def update(self, cached_features, device, empty_clusters_per_epoch):
        means = [[] for k in self.k_range]
        for key in cached_features.keys():

            xb = torch.stack(cached_features[key]) # Form an image batch
            if xb.get_device() == -1:
                xb = xb.to(device, dtype=torch.float32)
            _, batch_k_means_loss = self.iterative_kmeans(xb, key, device, empty_clusters_per_epoch)
        
            # For each "batch" append the losses (average distances) for each K value.
            for k in range(len(self.k_range)):
                means[k].append(batch_k_means_loss[k])
        # Compute the average loss for each value of K in k_range across all data points 
        losses = []
        for k in range(len(self.k_range)):
            stack = torch.stack(means[k])
            losses.append(stack.mean())
        return losses

    def iterative_kmeans(self, xb, class_index, device, empty_clusters_per_epoch):
        empty_clusters = []
        D_per_K_value = torch.zeros(len(self.k_range), device=device)  # Allocate on device
        for k in range(len(self.k_range)):
            K = self.k_range[k]
            previous_inertia = 0
            
            # Initialize tensors to store centroids and counts outside the loop
            new_centroids = torch.zeros((K, self.d), dtype=torch.float32, device=device)
            counts = torch.zeros(K, dtype=torch.int64, device=device)

            for itr in range(self.max_iter - 1):
                # Compute the assignments
                D, I = self.n_kmeans[class_index][k].index.search(xb, 1)
                
                inertia = torch.sum(D)
                if previous_inertia != 0 and abs(previous_inertia - inertia) < self.tol:
                    D_per_K_value[k] = torch.mean(D)
                    break
                previous_inertia = inertia

                # Reset centroids and counts to zero
                new_centroids.zero_()
                counts.zero_()
                
                # Sum up all points in each cluster using vectorized operations
                counts.index_add_(0, I.squeeze(), torch.ones(len(xb), device=device, dtype=torch.int64))
                new_centroids.index_add_(0, I.squeeze(), xb)
                
                # Avoid looping by using boolean masks to handle non-empty clusters
                non_empty_mask = counts > 0
                empty_mask = ~non_empty_mask
                new_centroids[non_empty_mask] /= counts[non_empty_mask].unsqueeze(1).float()
                
                non_empty = torch.nonzero(non_empty_mask).squeeze().tolist()

                # Ensure non_empty is a list
                if isinstance(non_empty, int):
                    non_empty = [non_empty]

                if empty_mask.any():
                    num_empty_clusters = empty_mask.sum().item()
                    eps = torch.full((num_empty_clusters, self.d), 1e-7, device=device)
                    sign = (torch.randint(0, 3, (num_empty_clusters, self.d), device=device) - 1).float()
                    perturbations = sign * eps

                    if len(non_empty) > 0:  # Check if non_empty is actually non-empty
                        non_empty_idx = torch.tensor(non_empty, device=device, dtype=torch.int64)
                        selected_idx = non_empty_idx[torch.randint(0, non_empty_idx.size(0), (num_empty_clusters,), device=device)]
                        new_centroids[empty_mask] = new_centroids[selected_idx] + perturbations
                        empty_clusters.extend([K] * num_empty_clusters)
                    else:
                        selected_idx = torch.randint(0, K, (num_empty_clusters,), device=device)
                        new_centroids[empty_mask] = self.n_kmeans[class_index][k].centroids[selected_idx] + perturbations
                        empty_clusters.extend([K] * num_empty_clusters)
                
                # Update the centroids in the FAISS index
                self.n_kmeans[class_index][k].centroids = new_centroids
                self.n_kmeans[class_index][k].index.reset()
                self.n_kmeans[class_index][k].index.add(new_centroids)
                D_per_K_value[k] = torch.mean(D)
                
        if empty_clusters:
            empty_clusters_per_epoch.update(len(empty_clusters))
        
        return self.n_kmeans, D_per_K_value



import torch


class KMatrices:
    def __init__(self, feature_bank, k_range=[2],d=1024):
        N = len(k_range)
        C = [torch.tensor(d, k) for k in k_range] # centroid matrix

        return 0

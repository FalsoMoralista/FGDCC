import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.tensors import trunc_normal_

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, nb_classes, K_range = [2,3,4,5]):
        self.embed_dim = embed_dim
        self.nb_classes = nb_classes
        self.subclass_classifier = nn.Linear(embed_dim, len(K_range) * nb_classes)        
        
        torch.nn.init.constant_(self.subclass_classifier.bias, 0)
        trunc_normal_(self.subclass_classifier.weight, std=2e-5)

    def forward(self, x):
        return 0
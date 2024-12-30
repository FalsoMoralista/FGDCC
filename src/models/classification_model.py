import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.tensors import trunc_normal_

class ClassificationHead(nn.Module):
    def __init__(self, embed_dim, nb_classes, K_range = [2,3,4,5]):
        super(ClassificationHead, self).__init__()
        self.embed_dim = embed_dim
        self.nb_classes = nb_classes

        num_classes = nb_classes * sum([K for K in K_range])
        self.classifier = nn.Linear(embed_dim, num_classes)        
        
        torch.nn.init.constant_(self.classifier.bias, 0)
        trunc_normal_(self.classifier.weight, std=2e-5)

    def forward(self, x):       
        return self.classifier(x)   

class ClassificationModel(nn.Module):
    def __init__(self, vit_backbone, embed_dim, nb_classes):
        super(ClassificationModel, self).__init__()        
        self.vit_encoder = vit_backbone
        self.classifier = ClassificationHead(embed_dim=embed_dim, nb_classes=nb_classes)
    
    def forward(self, imgs):
        h = self.vit_encoder(imgs)
        h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim 
        classifier_logits = self.classifier(torch.mean(h, dim=1).squeeze(dim=1))
        return 0, classifier_logits, h
    

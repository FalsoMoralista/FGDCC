
# Author: Luciano Filho

import torch
import torch.nn  as nn
import torch.nn.functional as F
from src.models.autoencoder import MaskedAutoEncoder
from src.models.hierarchical_classifiers import JEHierarchicalClassifier
from src.models.joint_embedding_classifier import JointEmbeddingClassifier
from src.models.multi_head_attention_hierarchical_cls import MultiHeadAttentionHierarchicalCls
from src.models.multi_head_attention_classifier import MultiHeadAttentionClassifier
from src.models.paired_multi_head_attention_classifier import PairedCrossAttentionClassifier

class FGDCC(nn.Module):

    def __init__(self, vit_backbone, classifier, backbone_patch_mean=False, raw_features=False):
        super(FGDCC, self).__init__()        
        self.backbone_patch_mean = backbone_patch_mean
        self.vit_encoder = vit_backbone
        self.classifier = classifier
        self.autoencoder = MaskedAutoEncoder()
        self.l2_norm = torch.nn.MSELoss()
        self.raw_features = raw_features
    
    def forward(self, imgs, device):
       if self.raw_features:
           return self.forward_raw_features(imgs)
       return self.forward_autoencoder(imgs, device)

    # Forward with autoencoder
    def forward_autoencoder(self, imgs, device):
        # Step 1. Forward into the encoder
        h = self.vit_encoder(imgs)
        if self.backbone_patch_mean: 
            h = torch.mean(h, dim=1) # Mean over patch-level representation and squeeze
            h = torch.squeeze(h, dim=1) 
        h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim 

        # Step 2. Forward into the hierarchical classifier
        parent_logits, child_logits, child_proj_embed = self.classifier(h) 

        # Detach from the graph
        child_proj_detatched = child_proj_embed.detach()

        # Step 3. Dimensionality Reduction                      
        reconstructed_input, bottleneck_output = self.autoencoder(child_proj_detatched, device)
        
        reconstruction_loss = self.l2_norm(reconstructed_input, child_proj_detatched)
         
        return reconstruction_loss, bottleneck_output, parent_logits, child_logits

    # Forward Function for the FGDCC_V2 (without autoencoder)
    def forward_raw_features(self, imgs):
        # Step 1. Forward into the encoder
        h = self.vit_encoder(imgs)
        if self.backbone_patch_mean: 
            h = torch.mean(h, dim=1) # Mean over patch-level representation and squeeze
            h = torch.squeeze(h, dim=1) 
        h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim 

        # Step 2. Forward into the classifier
        parent_logits, subclass_logits, subclass_proj_embed = self.classifier(h) 
        
        return 0, subclass_proj_embed, parent_logits, subclass_logits

def get_model(embed_dim, drop_path, nb_classes, K_range, proj_embed_dim, pretrained_model, device, raw_features=False):
#    cls = MultiHeadAttentionClassifier(input_dim=embed_dim,
#                                      nb_classes=nb_classes,
#                                      proj_embed_dim=proj_embed_dim,
#                                      drop_path=drop_path,
#                                      num_heads=8,
#                                      nb_subclasses_per_parent=K_range)
    cls = PairedCrossAttentionClassifier(input_dim=embed_dim,
                                      nb_classes=nb_classes,
                                      proj_embed_dim=proj_embed_dim,
                                      drop_path=drop_path,
                                      num_heads=4,
                                      nb_subclasses_per_parent=K_range)

    model = FGDCC(vit_backbone=pretrained_model, classifier=cls, raw_features=raw_features)
    model.to(device)
    return model                 
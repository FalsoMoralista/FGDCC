
# Author: Luciano Filho

import torch
import torch.nn  as nn
import torch.nn.functional as F
from src.models.autoencoder import MaskedAutoEncoder
from src.models.hierarchical_classifiers import JEHierarchicalClassifier
from src.models.joint_embedding_classifier import JointEmbeddingClassifier
from src.models.multi_head_attention_hierarchical_cls import MultiHeadAttentionHierarchicalCls
from src.models.multi_head_attention_classifier import MultiHeadAttentionClassifier

class FGDCC(nn.Module):

    def __init__(self, vit_backbone, classifier, backbone_patch_mean=False):
        super(FGDCC, self).__init__()        
        self.backbone_patch_mean = backbone_patch_mean
        self.vit_encoder = vit_backbone
        self.autoencoder = MaskedAutoEncoder()
        self.classifier = classifier
        self.l2_norm = torch.nn.MSELoss()
    
    def forward(self, imgs, device):
        # Step 1. Forward into the encoder
        h = self.vit_encoder(imgs)
        if self.backbone_patch_mean: 
            h = torch.mean(h, dim=1) # Mean over patch-level representation and squeeze
            h = torch.squeeze(h, dim=1) 
        h = F.layer_norm(h, (h.size(-1),)) # Normalize over feature-dim 

        # Step 2. Forward into the hierarchical classifier
        parent_logits, child_logits, parent_proj_embed, child_proj_embed = self.classifier(h) 

        # Detach from the graph
        child_proj_detatched = child_proj_embed.detach()

        # Step 3. Dimensionality Reduction                      
        reconstructed_input, bottleneck_output = self.autoencoder(child_proj_detatched, device)
        reconstruction_loss = self.l2_norm(reconstructed_input, child_proj_detatched)
         
        return reconstruction_loss, bottleneck_output, parent_logits, child_logits, parent_proj_embed, child_proj_embed

def get_model(embed_dim, drop_path, nb_classes, K_range, proj_embed_dim, pretrained_model ,device):
    
    cls = MultiHeadAttentionClassifier(input_dim=embed_dim,
                                      nb_classes=nb_classes,
                                      proj_embed_dim=proj_embed_dim,
                                      drop_path=drop_path,
                                      num_heads=4,
                                      nb_subclasses_per_parent=K_range)
    
    model = FGDCC(vit_backbone=pretrained_model, classifier=cls, backbone_patch_mean=False)
    model.to(device)
    return model                 

'''
    # -- Consistency Regularization with KL Divergence
    #size = imgs.size(0) # This allows handling cases when the number of points is different from batch size (due to drop_last=False)
    #gathered_child_embed = []
    #for i in range(size):
    #    gathered_child_embed.append(child_proj_embed[best_k_indexes[i]][i])
    #gathered_child_embed = torch.stack(gathered_child_embed)
    #subclass_probs = F.log_softmax(gathered_child_embed, dim=1)

    #subclass_probs = F.log_softmax(child_proj_embed, dim=1)
    #parent_probs = F.log_softmax(parent_proj_embed, dim=1) # Log target perhaps is not necessary
    # TODO: verify divergence direction
    #consistency_loss = F.kl_div(subclass_probs, parent_probs, log_target=True,  reduction='batchmean')

'''

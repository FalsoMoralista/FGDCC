import torch

import sys
 
# setting path
sys.path.append('../')

from src.models.multi_head_attention_hierarchical_cls import MultiHeadAttentionHierarchicalCls

def main():
    batch_size = 64
    input_dim = 1280
    patch_size = 256
    nb_classes = 1000

    x = torch.randn(batch_size, patch_size, input_dim)  # Batch of 64 samples
    print('x Size: ',x.size())
    model = MultiHeadAttentionHierarchicalCls(input_dim=input_dim, proj_embed_dim=1280, nb_classes=nb_classes, drop_path=0.2, nb_subclasses_per_parent=[2,3,4,5], num_heads=4)
    parent_logits, child_logits, parent_proj_embed, subclass_proj_embed = model(x, device=None)
    print(parent_logits.size())
    for j in child_logits:
        print(j.size())


main()






#from src.models.hierarchical_classifiers import (
#
#)
from src.utils.tensors import trunc_normal_
import torch
import torch.nn as nn
import torch.nn.functional as F

class ParentClassifier(nn.Module):
    def __init__(self, input_dim ,nb_parent_classes):
        super(ParentClassifier, self).__init__()

        self.fc = nn.Linear(input_dim, nb_parent_classes)    
        trunc_normal_(self.fc.weight, std=2e-5)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
            
    def forward(self, x):
        x = self.fc(x)
        x = F.layer_norm(x, (x.size(-1),))  # normalize over feature-dim 
        return x

class SubClassClassifier(nn.Module):
    def __init__(self, input_dim, nb_subclasses):
        super(SubClassClassifier, self).__init__()

        self.fc = nn.Linear(input_dim, nb_subclasses)
        
        trunc_normal_(self.fc.weight, std=2e-5)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x):
        x = self.fc(x)        
        x = F.layer_norm(x, (x.size(-1),))  # normalize over feature-dim  TODO: remove
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value):
        B, N, _ = query.shape
        
        Q = self.query(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        attn_output = (attn_weights @ V).transpose(1, 2).contiguous().view(B, N, self.num_heads * self.head_dim)
        output = self.out(attn_output)
        
        return output

class HierarchicalClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_classes_parent, num_classes_subclass):
        super(HierarchicalClassifier, self).__init__()
        self.parent_classifier = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)  # Output features for parent classifier
        )
        
        self.subclass_classifier = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)  # Output features for subclass classifier
        )
        
        self.cross_attention = MultiHeadCrossAttention(embed_dim, num_heads)
        
        # Since we concatenate, we double the input size for the final classifier
        self.parent_final_classifier = nn.Linear(embed_dim * 2, num_classes_parent)
        self.subclass_final_classifier = nn.Linear(embed_dim, num_classes_subclass)
    
    def forward(self, x):
        # Parent classifier to get parent features
        parent_features = self.parent_classifier(x)
        
        # Subclass classifier to get subclass features
        subclass_features = self.subclass_classifier(x)
        
        # Cross-attention to integrate subclass features into parent features
        integrated_features = self.cross_attention(parent_features, subclass_features, subclass_features)
        
        # Concatenate the original parent features with the integrated features
        enhanced_parent_features = torch.cat((parent_features, integrated_features), dim=-1)
        
        # Final classification
        parent_logits = self.parent_final_classifier(enhanced_parent_features)
        subclass_logits = self.subclass_final_classifier(subclass_features)
        
        return parent_logits, subclass_logits

# Example usage
input_dim = 512
embed_dim = 128
num_heads = 4
num_classes_parent = 10
num_classes_subclass = 50

model = HierarchicalClassifier(input_dim, embed_dim, num_heads, num_classes_parent, num_classes_subclass)

x = torch.randn(32, input_dim)  # Example input

parent_logits, subclass_logits = model(x)

print(parent_logits.shape)  # Expected: [32, 10]
print(subclass_logits.shape)  # Expected: [32, 50]



class MultiHeadAttentionHierarchicalCls(nn.Module):

    def __init__(self, input_dim, proj_embed_dim, patch_embed_dim, nb_classes, drop_path, nb_subclasses_per_parent, num_heads):
        super(MultiHeadAttentionHierarchicalCls, self).__init__()
        self.proj_embed_dim = proj_embed_dim
        self.nb_classes = nb_classes
        self.num_subclasses_per_parent = nb_subclasses_per_parent
        self.num_heads = num_heads


        self.cross_attention = MultiHeadCrossAttention(embed_dim, num_heads)

        # -- Classifier Embeddings
        self.parent_proj = nn.Linear((patch_embed_dim, input_dim), (patch_embed_dim, num_heads * proj_embed_dim))
        self.subclass_proj = nn.Linear((patch_embed_dim, input_dim), (patch_embed_dim, proj_embed_dim))


        trunc_normal_(self.parent_proj.weight, std=2e-5)
        trunc_normal_(self.subclass_proj.weight, std=2e-5)

        if self.subclass_proj.bias is not None and self.parent_proj.bias is not None:
            torch.nn.init.constant_(self.parent_proj.bias, 0)
            torch.nn.init.constant_(self.subclass_proj.bias, 0)

        self.head_drop = nn.Dropout(drop_path)

        self.parent_classifier = ParentClassifier(proj_embed_dim, nb_classes) # TODO: sequential
        self.child_classifiers = nn.ModuleList(
            [nn.ModuleList(
                [SubClassClassifier(proj_embed_dim, nb_subclasses=nb_subclasses) for _ in range(nb_classes)]
            ) for nb_subclasses in nb_subclasses_per_parent]    
        )

        self.feature_selection = nn.Sequential(
            nn.Linear((patch_embed_dim, (num_heads + 1) * proj_embed_dim), input_dim),
            nn.Dropout(),
        )


    # TODO check where to apply layer norm[]
    def forward(self, h, device):
        B, N, C = h.size() # [batch_size, num_patches, embed_dim]
        
        parent_proj_embed = self.parent_proj(h) # output shape [B, 256, 5120]
        parent_proj_embed = nn.GELU(parent_proj_embed) 
        
        subclass_proj_embed = self.subclass_proj(h)
        subclass_proj_embed = nn.GELU(subclass_proj_embed)

        # Cross-attention to integrate subclass features into parent features
        integrated_features = self.cross_attention(parent_proj_embed, subclass_proj_embed, subclass_proj_embed)
        
        parent_proj_embed = torch.cat((h, integrated_features), dim=0)
        parent_proj_embed = F.layer_norm(parent_proj_embed, (parent_proj_embed.size(-1),))
        parent_proj_embed = self.feature_selection(parent_proj_embed)

        parent_logits = self.parent_classifier(parent_proj_embed)  # Shape (batch_size, num_parents)
        parent_probs = F.softmax(parent_logits, dim=1)  # Softmax over class dimension

        # The parent class prediction allows to select the index for the correspondent subclass classifier
        y_hat = torch.argmax(parent_probs, dim=1)  # Argmax over class dimension: Shape (batch_size)

        # TODO: vectorize
        # Use the predicted parent class to select the corresponding child classifier
        child_logits = [torch.zeros(x.size(0), num, device=device) for num in self.num_children_per_parent] # Each element within child_logits is associated to a classifier with K outputs.
        for i in range(len(self.nb_subclasses_per_parent)):
            for j in range(x.size(0)): # Iterate over each sample in the batch                   
                # We will make predictions for each value of K belonging to num_children_per_parent (e.g., [2,3,4,5]) 
                logits = self.child_classifiers[i][y_hat[j]](subclass_proj_embed[j]) 
                child_logits[i][j] = logits        

        return parent_logits, child_logits, parent_proj_embed, subclass_proj_embed




import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np


def visualize_masked_input(masked_x, original_x, save_name, item_index=0):
    """
    Visualizes the masked input as a 2D heatmap, highlighting the areas that were masked.

    Args:
        masked_x (torch.Tensor): The input tensor after the mask has been applied, shape (batch_size, num_patches, embed_dim).
        original_x (torch.Tensor): The original input tensor before masking, shape (batch_size, num_patches, embed_dim).
        item_index (int): The index of the item in the batch to visualize.
    """
    # Select the item to visualize
    masked_input = masked_x[item_index].detach().cpu().numpy()
    original_input = original_x[item_index].detach().cpu().numpy()

    # Compute the mean of the embeddings for each patch to create a heatmap-like visualization
    heatmap_masked = np.mean(masked_input, axis=-1)  # Shape: (num_patches,)
    heatmap_original = np.mean(original_input, axis=-1)  # Shape: (num_patches,)

    # Reshape the heatmaps into a 2D grid (e.g., 16x16 for 256 patches)
    num_patches = int(np.sqrt(len(heatmap_masked)))
    heatmap_masked = heatmap_masked.reshape(num_patches, num_patches)
    heatmap_original = heatmap_original.reshape(num_patches, num_patches)

    # Plot the original and masked heatmaps side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original input heatmap
    axes[0].imshow(heatmap_original, cmap='viridis', interpolation='nearest')
    axes[0].set_title("Original Input Heatmap")
    axes[0].axis('off')

    # Masked input heatmap
    axes[1].imshow(heatmap_masked, cmap='viridis', interpolation='nearest')
    axes[1].set_title("Masked Input Heatmap")
    axes[1].axis('off')

    # Show the color bar
    plt.colorbar(axes[1].imshow(heatmap_masked, cmap='viridis'), ax=axes, fraction=0.046, pad=0.04)
    plt.savefig(save_name) # show()


class SpatialMaskedAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1280, 1152),
            nn.GELU(),
            nn.Linear(1152, 1024),
            nn.GELU(),
            nn.Linear(1024 , 768)                        
        )

        self.decoder = torch.nn.Sequential(
            nn.Linear(768, 1024), 
            nn.GELU(),
            nn.Linear(1024, 1152), 
            nn.GELU(),
            nn.Linear(1152, 1280), 
        )

    def generate_patch_mask(self, batch_size, num_patches, embed_dim, device, mask_fraction=0.75):
        """
        Generates a binary mask that masks entire patches of an image's features in a vectorized manner.

        Args:
            batch_size (int): Number of samples in the batch.
            num_patches (int): Number of patches in the sequence (e.g., 256 for a 16x16 grid).
            embed_dim (int): Dimensionality of each patch's feature embedding.
            device (torch.device): Device to create the mask on (e.g., 'cuda' or 'cpu').
            mask_fraction (float): Fraction of patches to mask.

        Returns:
            torch.Tensor: Binary mask of shape (batch_size, num_patches, embed_dim) with `True` for masked patches.
        """
        # Calculate the total number of patches to mask
        num_masked_patches = int(num_patches * mask_fraction)

        # Generate a batch of random indices for masking
        all_indices = torch.arange(num_patches, device=device)
        masked_indices = torch.stack([
            all_indices[torch.randperm(num_patches, device=device)[:num_masked_patches]]
            for _ in range(batch_size)
        ])

        # Create the mask with all `False` values
        mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)

        # Use advanced indexing to set the masked patches to `True`
        mask[torch.arange(batch_size, device=device).unsqueeze(1), masked_indices] = True

        # Expand the mask to match the embedding dimension
        mask = mask.unsqueeze(-1).expand(-1, -1, embed_dim)  # Shape: (batch_size, num_patches, embed_dim)

        return mask

    def forward(self, x, device):
        batch_size, num_patches, embed_dim = x.shape
        # Generate the patch mask
        mask = self.generate_patch_mask(batch_size, num_patches, embed_dim, device)
        
        # Mask the input: Set the masked patches to zero
        masked_input = x * (~mask)  # Use `~mask` to keep unmasked patches

        # Normalize the masked input
        masked_input = F.layer_norm(masked_input, (masked_input.size(-1),))

        # Encode and decode
        bottleneck_output = self.encoder(masked_input)
        reconstructed_input = self.decoder(bottleneck_output)
        reconstructed_input = F.layer_norm(reconstructed_input, (embed_dim,))  # Normalize over feature dimension

        return reconstructed_input, bottleneck_output
    
def transformer_autoencoder():
    return SpatialMaskedAutoEncoder()



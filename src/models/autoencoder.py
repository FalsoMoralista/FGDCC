
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1280, 1024),
            nn.GELU(),
            nn.Linear(1024, 768),
            nn.GELU(),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 384),
            nn.GELU(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(384, 512),
            nn.GELU(),
            torch.nn.Linear(512, 768),
            nn.GELU(),
            torch.nn.Linear(768, 1024), 
            nn.GELU(),
            torch.nn.Linear(1024, 1280), 
        )

    def generate_mask(self, batch_size, feature_dim, device, mask_fraction=0.25):
        # Calculate the number of elements to mask
        total_elements = batch_size * feature_dim
        num_masked_elements = int(total_elements * mask_fraction)
        
        # Create a mask with all zeros
        mask = torch.zeros(total_elements, dtype=torch.bool, device=device)
        
        # Randomly select indices to be masked
        masked_indices = torch.randperm(total_elements, device=device)[:num_masked_elements]
        
        # Set the selected indices to 1 (masked)
        mask[masked_indices] = 1
        
        # Reshape the mask to the original input shape
        mask = mask.view(batch_size, feature_dim)
        
        return mask

    def forward(self, x, device):
        mask = self.generate_mask(x.size(0), 1280, device=device)
        masked_input = x * (~mask) # Mask input    

        bottleneck_output = self.encoder(masked_input)
        bottleneck_output = F.layer_norm(bottleneck_output, (bottleneck_output.size(-1),))  # normalize over feature-dim 

        reconstructed_input = self.decoder(bottleneck_output)
        reconstructed_input = F.layer_norm(reconstructed_input, (reconstructed_input.size(-1),))  # normalize over feature-dim 

        return reconstructed_input, bottleneck_output


def vanilla_autoencoder():
    return MaskedAutoEncoder()

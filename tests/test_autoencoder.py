import torch

import sys
 
# setting path
sys.path.append('../')

from src.models.autoencoder import MaskedAutoEncoder

def main():
    batch_size = 64
    input_dim = 1024
    nb_classes = 1000

    x = torch.randn(batch_size, input_dim)  # Batch of 64 samples
    print('x Size: ',x.size())

    model = MaskedAutoEncoder()
    reconstructed_input, bottleneck_output = model(x)

main()


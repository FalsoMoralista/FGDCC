import torch
import itertools

def all_combinations(tensor):
    # Ensure the input is a 1D tensor
    tensor = tensor.flatten()
    
    # Get the number of elements in the tensor
    n = tensor.size(0)
    
    # Create a meshgrid of indices
    idx1, idx2 = torch.meshgrid(torch.arange(n), torch.arange(n))
    
    # Get the combinations without duplicates and without self-combinations
    mask = idx1 < idx2
    
    # Extract the indices for the valid combinations
    idx1 = idx1[mask]
    idx2 = idx2[mask]
    
    # Use the indices to gather the combinations from the tensor
    combinations = torch.stack((tensor[idx1], tensor[idx2]), dim=1)
    
    return combinations

# Example usage
k_range = [2,3,4,5]
tensor = torch.tensor([1, 2, 3, 4])
combinations = all_combinations(tensor)
print(combinations)

def two_by_two_combinations(values):
    # Get the list of all possible two-by-two permutations
    perms = list(itertools.combinations(values,2))
    # Convert each permutation back to a PyTorch tensor
    return perms
k_range = [2,3,4,5]
combination_list = [two_by_two_combinations(range(k))  for k in k_range]
for c in combination_list:
   print(c)


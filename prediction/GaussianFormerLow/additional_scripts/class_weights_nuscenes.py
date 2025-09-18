import numpy as np
import torch
import torch.nn.functional as F

# Class frequencies from your file (for classes 0-17)
nusc_class_frequencies = np.array([
    944004, 1897170, 152386, 2391677, 16957802, 724139, 189027,
    2074468, 413451, 2384460, 5916653, 175883646, 4275424,
    51393615, 61411620, 105975596, 116424404, 1892500630
])

num_classes = 18


def class_weights_nuscenes(class_freqs):
    class_weights = torch.from_numpy(1 / np.log(class_freqs[:num_classes] + 0.001))
    class_weights = num_classes * F.normalize(class_weights, 1, -1)
    print(class_weights)

def class_weights_effective_number_samples(class_freqs):
    # --- Improvement 1: Isolate valid classes ---
    valid_class_freqs = class_freqs[:17]

    # --- Improvement 2: Use a beta very close to 1 to prevent numerical underflow ---
    beta = 0.99999

    # Calculate weights for VALID classes only
    effective_num = 1.0 - np.power(beta, valid_class_freqs)
    weights = (1.0 - beta) / effective_num

    # Normalize the weights for the valid classes
    weights = weights / np.sum(weights) * 17  # Normalize for 17 classes

    # Manually set the weight for the "empty" class (label 17)
    empty_class_weight = 0.0  # Setting to 0.0 is also common to completely ignore it
    final_weights = np.append(weights, empty_class_weight)

    # Print the final weights for all 18 classes
    print("\nEffective Number of Samples Weights (beta=0.99999):")
    for i, weight in enumerate(final_weights):
        print(f"Class {i}: Weight = {weight:.4f}")

def class_weights_inverse_frequency(class_freqs):
    """
    Calculates class weights using the scaled inverse frequency method.
    """
    # 1. Isolate the 17 valid classes for the balancing calculation
    valid_class_freqs = class_freqs[:17]
    num_valid_classes = len(valid_class_freqs)

    # 2. Calculate the total number of points across all valid classes
    total_samples = valid_class_freqs.sum()

    # 3. Calculate the weight for each valid class
    # Formula: N / (C * N_c)
    weights = total_samples / (num_valid_classes * valid_class_freqs)

    # 4. Manually set the weight for the "empty" or "ignore" class
    empty_class_weight = 0.0  # Typically set to 0.0 or a small number
    final_weights = np.append(weights, empty_class_weight)

    print("\nInverse Frequency Weights:")
    for i, weight in enumerate(final_weights):
        print(f"Class {i}: Weight = {weight:.4f}")

    return final_weights

if __name__ == "__main__":

    class_weights_nuscenes(nusc_class_frequencies)

    class_weights_effective_number_samples(nusc_class_frequencies)

    inverse_weights = class_weights_inverse_frequency(nusc_class_frequencies)




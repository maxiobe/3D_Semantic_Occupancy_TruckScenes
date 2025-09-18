import numpy as np
import torch
import torch.nn.functional as F

# Class frequencies from your file (for classes 0-17)
trsc_class_frequencies = np.array([
    2074, 1773, 1649, 2639, 182634, 1695, 1390,
    12813, 34261, 79555, 113309, 896, 182634,
    3168, 2005
])

trsc_class_frequencies_occupancy = np.array([
    820511, 271429, 292283, 5477952, 139564017, 
    3554278, 275585, 1157896, 1121443, 173378212,
    84734644, 36222, 31427066, 6857852, 53672547,
    7437846035, 852531512028
])

num_classes = 17

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
    weights = weights / np.sum(weights) * num_classes  # Normalize for 17 classes

    # Manually set the weight for the "empty" class (label 17)
    background_class_weight = 0.1
    empty_class_weight = 0.1  # Setting to 0.0 is also common to completely ignore it
    final_weights_intermediate = np.append(weights, empty_class_weight)
    final_weights = np.append(final_weights_intermediate, background_class_weight)

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
    background_class_weight = 0.1
    empty_class_weight = 0.1  # Typically set to 0.0 or a small number
    final_weights_intermediate = np.append(weights, empty_class_weight)
    final_weights = np.append(final_weights_intermediate, background_class_weight)

    print("\nInverse Class Frequency Weights:")
    for i, weight in enumerate(final_weights):
        print(f"Class {i}: Weight = {weight:.4f}")

    return final_weights

def class_weights(class_freqs):
    total_samples = class_freqs.sum()

    weights = 1 - (class_freqs / total_samples)
    background_class_weight = 0.1
    empty_class_weight = 0.1  # Typically set to 0.0 or a small number
    final_weights_intermediate = np.append(weights, empty_class_weight)
    final_weights = np.append(final_weights_intermediate, background_class_weight)

    print("\nClass Weights based on portion of total samples:")
    for i, weight in enumerate(final_weights):
        print(f"Class {i}: Weight = {weight:.4f}")

if __name__ == "__main__":

    class_weights_nuscenes(trsc_class_frequencies)
    class_weights_nuscenes(trsc_class_frequencies_occupancy)
    #class_weights_effective_number_samples(trsc_class_frequencies)

    #inverse_weights = class_weights_inverse_frequency(trsc_class_frequencies)

    #class_weights(trsc_class_frequencies)


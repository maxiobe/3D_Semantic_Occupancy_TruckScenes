import pickle
import numpy as np
import torch


def check_for_non_finite(data, parent_key_path=""):
    """Recursively checks dictionaries, lists, and tensors for NaN/Inf."""
    found_issue = False

    if isinstance(data, dict):
        for key, value in data.items():
            new_path = f"{parent_key_path}['{key}']"
            if check_for_non_finite(value, new_path):
                found_issue = True

    elif isinstance(data, (list, tuple)):
        for i, item in enumerate(data):
            new_path = f"{parent_key_path}[{i}]"
            if check_for_non_finite(item, new_path):
                found_issue = True

    elif isinstance(data, np.ndarray):
        # --- FIX: Only check float arrays for NaN/Inf ---
        if np.issubdtype(data.dtype, np.floating):
            if np.isnan(data).any():
                print(f"🚨 Found NaN values in NumPy array at: {parent_key_path}")
                found_issue = True
            if np.isinf(data).any():
                print(f"🚨 Found Inf values in NumPy array at: {parent_key_path}")
                found_issue = True

    elif isinstance(data, torch.Tensor):
        # --- FIX: Only check float tensors for NaN/Inf ---
        if data.is_floating_point():
            if torch.isnan(data).any():
                print(f"🚨 Found NaN values in PyTorch tensor at: {parent_key_path}")
                found_issue = True
            if torch.isinf(data).any():
                print(f"🚨 Found Inf values in PyTorch tensor at: {parent_key_path}")
                found_issue = True

    return found_issue


# --- Main script ---
file_path = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/occ_infos_temporal_train.pkl'
file_path = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/occ_infos_temporal_val.pkl'
print(f"Loading and checking '{file_path}'...")

try:
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)

    if not check_for_non_finite(data_dict):
        print("\n✅ No NaN or Inf values found in the entire data structure.")
    else:
        print("\nFinished checking. Issues were found at the paths listed above.")

    if 'infos' in data_dict and isinstance(data_dict['infos'], list) and len(data_dict['infos']) > 0:
        print("\nStructure of the first data sample (infos[0]):")
        first_sample_keys = data_dict['infos'][0].keys() if isinstance(data_dict['infos'][0], dict) else "Not a dict"
        print(f"Keys: {list(first_sample_keys)}")

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
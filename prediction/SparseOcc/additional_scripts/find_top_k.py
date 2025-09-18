import numpy as np
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F

from truckscenes import TruckScenes  # Assuming this is in your project


def analyze_occupancy_grid(input_path, max_counts, min_counts, all_counts, device, verbose=False):
    """
    Loads a .npz occupancy grid, counts non-free voxels at 3 resolutions using
    average pooling and a density threshold, and updates the running statistics.
    """
    FREE_LABEL = 16  # The label for free space
    THRESHOLD = 0.4  # The 40% density threshold

    if not os.path.exists(input_path):
        if verbose:
            print(f"Warning: Input file not found at {input_path}")
        return

    with np.load(input_path) as data:
        semantics = data['semantics']

    semantics_tensor = torch.from_numpy(semantics).to(device)

    # Create a mask where non-free = 1.0 and free = 0.0.
    non_free_mask = (semantics_tensor != FREE_LABEL).float()

    # Level 2: Final resolution (original size) - no pooling needed
    count_final = torch.sum(non_free_mask).item()
    max_counts[2] = max(max_counts[2], count_final)
    min_counts[2] = min(min_counts[2], count_final)
    all_counts[2].append(count_final)

    # For pooling, we need a 5D tensor [N, C, D, H, W]
    non_free_mask_5d = non_free_mask.unsqueeze(0).unsqueeze(0)

    # --- MODIFIED LOGIC: Using Average Pooling ---

    # Level 1: Intermediate resolution (downsampled by 2x)
    pooled_density_2x = F.avg_pool3d(non_free_mask_5d, kernel_size=2, stride=2).squeeze()
    occupied_mask_intermediate = (pooled_density_2x >= THRESHOLD)
    count_intermediate = torch.sum(occupied_mask_intermediate).item()
    max_counts[1] = max(max_counts[1], count_intermediate)
    min_counts[1] = min(min_counts[1], count_intermediate)
    all_counts[1].append(count_intermediate)

    # Level 0: Coarsest resolution (downsampled by 4x)
    pooled_density_4x = F.avg_pool3d(non_free_mask_5d, kernel_size=4, stride=4).squeeze()
    occupied_mask_coarsest = (pooled_density_4x >= THRESHOLD)
    count_coarsest = torch.sum(occupied_mask_coarsest).item()
    max_counts[0] = max(max_counts[0], count_coarsest)
    min_counts[0] = min(min_counts[0], count_coarsest)
    all_counts[0].append(count_coarsest)

    if verbose:
        print(f"File: {os.path.basename(input_path)}")
        print(f"  - Counts (Coarse, Mid, Fine): {count_coarsest}, {count_intermediate}, {count_final}")
        print(f"  - Running Max (Coarse, Mid, Fine): {max_counts}")
        print(f"  - Running Min (Coarse, Mid, Fine): {min_counts}")


def main(data_root, version, gts_path):
    trsc = TruckScenes(dataroot=data_root, version=version, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    max_counts = [0, 0, 0]
    min_counts = [float('inf'), float('inf'), float('inf')]
    all_counts = [[], [], []]

    scene_number = 0
    total_scenes = len(trsc.scene)
    for my_scene in tqdm(trsc.scene, total=total_scenes, desc="Processing Scenes"):
        scene_name = my_scene['name']
        current_sample_token = my_scene.get('first_sample_token')

        if not current_sample_token:
            continue

        if scene_number < 100:
            scene_folder = "0-99"
        elif scene_number < 200:
            scene_folder = "100-199"
        elif scene_number < 300:
            scene_folder = "200-299"
        elif scene_number < 400:
            scene_folder = "300-399"
        elif scene_number < 500:
            scene_folder = "400-499"
        else:
            scene_folder = "500-597"

        while current_sample_token:
            my_sample = trsc.get('sample', current_sample_token)
            load_path = os.path.join(gts_path, scene_folder, scene_name, current_sample_token, "labels.npz")

            if os.path.exists(load_path):
                analyze_occupancy_grid(load_path, max_counts, min_counts, all_counts, device)

            current_sample_token = my_sample.get('next', '')

        scene_number += 1

    print("\n--- Analysis Complete ---")

    if not all_counts[0]:
        print("No .npz files were found or processed. Exiting.")
        return

    avg_counts = [int(np.mean(counts)) for counts in all_counts]
    median_counts = [int(np.median(counts)) for counts in all_counts]

    print(f"Statistics for non-free voxels across {len(all_counts[0])} samples (using 40% avg pool threshold):\n")
    print(f"                {'Coarsest':>10} {'Intermediate':>15} {'Final':>10}")
    print(f"----------------------------------------------------")
    print(f"Max Counts:     {max_counts[0]:>10} {max_counts[1]:>15} {max_counts[2]:>10}")
    print(f"Min Counts:     {min_counts[0]:>10} {min_counts[1]:>15} {min_counts[2]:>10}")
    print(f"Average Counts: {avg_counts[0]:>10} {avg_counts[1]:>15} {avg_counts[2]:>10}")
    print(f"Median Counts:  {median_counts[0]:>10} {median_counts[1]:>15} {median_counts[2]:>10}")
    print(f"----------------------------------------------------")

    margin = 1.10
    suggested_topk = [int(np.ceil(c * margin)) for c in max_counts]

    print("\nTo ensure the model doesn't prune real voxels, it's recommended to add a safety margin to the max counts.")
    print(f"Suggested topk values (with 10% margin): {suggested_topk}")
    print("\nYou can use these values for the '_topk_training_' parameter in your config file.")


if __name__ == '__main__':
    data_root = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
    version = 'v1.0-trainval'
    gts_path = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval/gts_64'

    main(data_root, version, gts_path)
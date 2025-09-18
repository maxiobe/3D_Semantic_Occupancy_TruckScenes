import numpy as np
import os
from tqdm import tqdm

from truckscenes import TruckScenes


def crop_occupancy_grid(input_path, output_path, verbose=True):
    """
    Loads a .npz occupancy grid, crops its arrays to 736x736x32, and saves to a new file.
    """
    # --- CROP DEFINITIONS FOR 736x736x32 from 744x744x64 ---
    # X and Y axes: Crop 8 voxels total (744-736), so 4 from each side.
    X_START, X_END = 4, 740  # 4:(744-4) -> shape 736
    Y_START, Y_END = 4, 740  # 4:(744-4) -> shape 736

    # Z axis: Crop 5 from the bottom, resulting in a size of 32.
    Z_START, Z_END = 5, 37  # 5:(5+32) -> shape 32

    if not os.path.exists(input_path):
        if verbose:
            print(f"Error: Input file not found at {input_path}")
        return

    with np.load(input_path) as data:
        semantics = data['semantics']
        mask_lidar = data['mask_lidar']
        mask_camera = data['mask_camera']

        if verbose:
            print(f"Loading data from: {input_path}")
            print(f"Original semantics shape: {semantics.shape}")

        # --- Apply the new 3D slicing ---
        cropped_semantics = semantics[X_START:X_END, Y_START:Y_END, Z_START:Z_END]
        cropped_mask_lidar = mask_lidar[X_START:X_END, Y_START:Y_END, Z_START:Z_END]
        cropped_mask_camera = mask_camera[X_START:X_END, Y_START:Y_END, Z_START:Z_END]

        if verbose:
            print("\n--- Cropping successful ---")
            print(f"New semantics shape: {cropped_semantics.shape}")

        np.savez_compressed(
            output_path,
            semantics=cropped_semantics,
            mask_lidar=cropped_mask_lidar,
            mask_camera=cropped_mask_camera
        )
        if verbose:
            print(f"\nSuccessfully saved cropped grid to: {output_path}")


# --- MAIN FUNCTION (Unchanged, but good practice to update save_path) ---
def main(data_root, version, gts_path, save_path):
    trsc = TruckScenes(dataroot=data_root, version=version, verbose=True)

    scene_number = 0
    for my_scene in tqdm(trsc.scene, desc="Processing Scenes"):
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
            output_path = os.path.join(save_path, scene_folder, scene_name, current_sample_token, "labels.npz")

            if os.path.exists(load_path):
                output_dir = os.path.dirname(output_path)
                os.makedirs(output_dir, exist_ok=True)
                crop_occupancy_grid(load_path, output_path, verbose=False)

            current_sample_token = my_sample.get('next', '')

        scene_number += 1


if __name__ == '__main__':
    data_root = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
    version = 'v1.0-trainval'
    # Source path of your 744x744x64 grids
    gts_path = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval/gts_64'
    # New destination path for the cropped 736x736x32 grids
    save_path = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval/gts_736x32'

    main(data_root, version, gts_path, save_path)
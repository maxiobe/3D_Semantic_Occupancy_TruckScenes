from truckscenes import TruckScenes
import numpy as np
import os
from tqdm import tqdm


def main(data_root, version, gts_path):

    trsc = TruckScenes(dataroot=data_root, version=version, verbose=True)

    total_class_counts = np.zeros(17, dtype=np.int64)

    scene_number = 0
    for my_scene in tqdm(trsc.scene, desc="Processing scene"):
        scene_name = my_scene['name']

        current_sample_token = my_scene.get('first_sample_token')
        sample_count = 0

        if not current_sample_token:
            print(f"Scene '{scene_name}' has no samples.")
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
            sample_count += 1

            load_path = os.path.join(gts_path, scene_folder, scene_name, current_sample_token, "labels.npz")

            if not os.path.exists(load_path):
                print(f"Warning: File not found for sample {current_sample_token}, skipping.")
                current_sample_token = my_sample.get('next', '')
                continue

            data = np.load(load_path)

            semantics = data['semantics']

            classes, counts = np.unique(semantics, return_counts=True)

            total_class_counts[classes] += counts

            current_sample_token = my_sample.get('next', '')

        scene_number += 1

        print(
            f"Processed {sample_count} samples. Reached end of scene '{scene_name}'.")

    print("\n" + "=" * 30)
    print("      Total Class Counts")
    print("=" * 30)
    for i, count in enumerate(total_class_counts):
        print(f"Class {i:2d}: {count}")
    print("=" * 30)
    print("\nRaw numpy array:")
    print(total_class_counts)


if __name__ == '__main__':
    data_root = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
    version = 'v1.0-trainval'
    gts_path = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval/gts_64'

    main(data_root, version, gts_path)

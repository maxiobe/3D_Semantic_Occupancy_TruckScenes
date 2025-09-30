from truckscenes.truckscenes import TruckScenes
import os
import numpy as np
from truckscenes.utils import splits

CLASS_NAMES = [
    'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'animal', 'traffic_sign', 'other_vehicle', 'train', 'background', 'free'
]

def print_class_counts(grid, class_names, title):
    """Counts unique values in a grid and prints them with labels."""
    print(f"\n--- {title} ---")
    unique_labels, counts = np.unique(grid, return_counts=True)
    count_dict = dict(zip(unique_labels, counts))

    for i, class_name in enumerate(class_names):
        count = count_dict.get(i, 0)  # Get count for class i, default to 0 if not present
        print(f"  {i:02d} {class_name:<25}: {count} voxels")
    print("-" * (len(title) + 6))

data_root = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini'
gt_data_root = os.path.join(data_root, 'gts')

trsc = TruckScenes(version='v1.0-mini', dataroot=data_root, verbose=True)

for scene in trsc.scene:

    scene_name = scene['name']

    if scene_name not in splits.mini_val:
        print('\nTrain')
    else:
        print('\nVal')
    sample_token = scene['first_sample_token']
    sample = trsc.get('sample', sample_token)

    sample_num = 0
    while sample_token is not None:
        path_gt = gt_data_root + '/' + scene_name + '/' + sample_token + '/labels.npz'

        gt = np.load(path_gt)
        gt_labels = gt['semantics']

        print(f"\n--- {scene_name} --- sample {sample_num} ---")
        print_class_counts(gt_labels, CLASS_NAMES, "Prediction Class Counts")

        if sample['next'] != '':
            sample_token = sample['next']
            sample = trsc.get('sample', sample_token)
            sample_num += 1
        else:
            break







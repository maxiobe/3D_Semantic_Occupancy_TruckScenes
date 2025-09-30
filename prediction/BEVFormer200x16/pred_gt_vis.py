import numpy as np
import open3d as o3d
import os
from functools import reduce

# --- Configuration ---
# Directory where your .npz files are saved
VIS_DIR = '/home/max/Desktop/Masterarbeit/BEVFormer_Test'
# Index of the sample you want to visualize
SAMPLE_INDEX = 1

# Define a color map for your classes
# Define a color map for your 17 classes
COLOR_MAP = np.array([
    [0, 0, 0, 255],              # 0: noise (Black)
    [112, 128, 144, 255],      # 1: barrier (Slate Gray)
    [220, 20, 60, 255],        # 2: bicycle (Crimson Red)
    [0, 0, 230, 255],          # 3: bus (Blue)
    [0, 128, 0, 255],          # 4: car (Green)
    [255, 127, 80, 255],       # 5: construction_vehicle (Coral)
    [255, 0, 255, 255],        # 6: motorcycle (Magenta)
    [255, 255, 0, 255],        # 7: pedestrian (Yellow)
    [255, 165, 0, 255],        # 8: traffic_cone (Orange)
    [173, 216, 230, 255],      # 9: trailer (Light Blue)
    [75, 0, 130, 255],         # 10: truck (Indigo)
    [139, 69, 19, 255],        # 11: animal (Saddle Brown)
    [0, 255, 255, 255],        # 12: traffic_sign (Cyan)
    [0, 139, 139, 255],        # 13: other_vehicle (Dark Cyan)
    [178, 34, 34, 255],        # 14: train (Firebrick)
    [85, 107, 47, 255],        # 15: background (Dark Olive Green)
    #[211, 211, 211, 255],      # 16: free (Light Gray)
]) / 255.0  # Normalize to 0-1 range for visualization

# Define class names and color map
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

def voxel2points(voxel_grid, voxel_size, point_cloud_range):
    """Converts a voxel grid to a point cloud, ignoring void/free classes."""
    void_classes = [16, 255]  # Ignore 'free' and 'unknown'
    mask = np.logical_not(reduce(np.logical_or, [voxel_grid == void for void in void_classes]))
    occ_indices = np.where(mask)
    semantic_labels = voxel_grid[occ_indices]

    points = np.stack([
        occ_indices[0] * voxel_size[0] + point_cloud_range[0] + voxel_size[0] / 2,
        occ_indices[1] * voxel_size[1] + point_cloud_range[1] + voxel_size[1] / 2,
        occ_indices[2] * voxel_size[2] + point_cloud_range[2] + voxel_size[2] / 2,
    ], axis=1)

    return points, semantic_labels


def main():
    file_path = os.path.join(VIS_DIR, f'{str(SAMPLE_INDEX).zfill(4)}.npz')
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    data = np.load(file_path, allow_pickle=True)
    pred_grid = data['pred']
    # The ground truth is often saved as a dict within the npz
    gt_grid = data['gt']  # Access the 'semantics' array from the dictionary

    print(pred_grid.shape)
    print(gt_grid.shape)

    print_class_counts(pred_grid, CLASS_NAMES, "Prediction Class Counts")
    print_class_counts(gt_grid, CLASS_NAMES, "Ground Truth Class Counts")

    print(f"Visualizing sample {SAMPLE_INDEX}")

    point_cloud_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 5.4]
    voxel_size = [0.4, 0.4, 0.4]

    pred_points, pred_labels = voxel2points(pred_grid, voxel_size, point_cloud_range)
    gt_points, gt_labels = voxel2points(gt_grid, voxel_size, point_cloud_range)

    print(f"Found {len(pred_points)} predicted points (non-empty).")
    print(f"Found {len(gt_points)} ground truth points (non-empty).")

    pcd_pred = o3d.geometry.PointCloud()
    if len(pred_points) > 0:
        pcd_pred.points = o3d.utility.Vector3dVector(pred_points)
        pcd_pred.colors = o3d.utility.Vector3dVector(COLOR_MAP[pred_labels, :3])

    pcd_gt = o3d.geometry.PointCloud()
    if len(gt_points) > 0:
        pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
        pcd_gt.colors = o3d.utility.Vector3dVector(COLOR_MAP[gt_labels, :3])

    pcd_gt.translate((100, 0, 0))  # Move ground truth to the side for comparison

    o3d.visualization.draw_geometries([pcd_pred, pcd_gt], window_name="Prediction (Left) vs. Ground Truth (Right)")


if __name__ == "__main__":
    main()
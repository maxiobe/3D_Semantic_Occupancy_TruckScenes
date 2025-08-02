import open3d as o3d
import os
import numpy as np
import json
from pyquaternion import Quaternion
from truckscenes.utils.data_classes import Box, LidarPointCloud

############################# Flexcloud data saving #############################
def save_pcds_to_directory(pcd_list, timestamps, output_dir):
    """Saves a list of point clouds to individual timestamped files."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (points, ts) in enumerate(zip(pcd_list, timestamps)):
        file_name = f"{int(ts)}_{int((ts % 1) * 1e9)}.pcd"
        file_path = os.path.join(output_dir, file_name)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        o3d.io.write_point_cloud(file_path, pcd)
    print(f"Saved {len(pcd_list)} point clouds to directory {output_dir}")


def save_poses_to_kitti_format(poses_4x4_list, file_path):
    """Saves a list of 4x4 NumPy poses to a KITTI format text file."""
    with open(file_path, 'w') as f:
        for pose in poses_4x4_list:
            # Flatten the 3x4 part of the matrix into a single line
            line = ' '.join(map(str, pose[:3, :].flatten()))
            f.write(line + '\n')
    print(f"Saved {len(poses_4x4_list)} SLAM poses to {file_path}")


def save_gnss_to_directory(gnss_data, timestamps, output_dir):
    """Saves GNSS data into individual timestamped files."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (data, ts) in enumerate(zip(gnss_data, timestamps)):
        # Format: sec_nanosec.txt
        file_name = f"{int(ts)}_{int((ts % 1) * 1e9)}.txt"
        file_path = os.path.join(output_dir, file_name)
        # Format: lat lon ele lat_stddev lon_stddev ele_stddev
        np.savetxt(file_path, data.reshape(1, -1), fmt='%.8f')
    print(f"Saved {len(gnss_data)} GNSS poses to directory {output_dir}")


def save_pointcloud_for_annotation(points_n_features, output_path):
    """
    Saves a point cloud to a .pcd file for annotation tools.

    Args:
        points_n_features (np.ndarray): The point cloud array, shape (N, features).
                                       Assumes XYZ are the first 3 columns.
        output_path (str): The full path to save the .pcd file.
    """
    print(f"Saving point cloud with shape {points_n_features.shape[0]} for annotation to {output_path}")

    if points_n_features.shape[0] == 0:
        print(f"Warning: Attempting to save an empty point cloud to {output_path}. Skipping.")
        return

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_n_features[:, :3])  # Use only XYZ

    # If you have intensity, you can add it as a color or custom attribute if needed,
    # but for bounding box annotation, XYZ is usually sufficient.

    o3d.io.write_point_cloud(output_path, pcd)
    # print(f"Saved point cloud to {output_path}")


def parse_single_annotation_file(json_filepath):
    """
    Loads annotations for a single frame from a JSON file.
    The JSON file is expected to contain a list of objects for one point cloud.

    Args:
        json_filepath (str): Path to the single annotation JSON file.

    Returns:
        tuple: A tuple containing:
               - frame_idx (int): The index of the frame, parsed from the filename.
               - boxes (list): A list of truckscenes.utils.data_classes.Box objects.
               Returns (None, []) if the file cannot be parsed.
    """
    if not os.path.exists(json_filepath):
        # This is not an error, just means no manual annotation for this frame.
        return None, []

    with open(json_filepath, 'r') as f:
        data = json.load(f)

    boxes_for_this_frame = []
    if 'objects' not in data or not isinstance(data['objects'], list):
        return []  # File exists but has no objects

    for label_obj in data['objects']:
        try:
            # --- 3. Access centroid, dimensions, and rotations as dictionaries ---
            center = [label_obj['centroid']['x'], label_obj['centroid']['y'], label_obj['centroid']['z']]

            # Reorder dimensions: exported is (l, w, h), your Box class expects (w, l, h)
            dims = [label_obj['dimensions']['width'], label_obj['dimensions']['length'],
                    label_obj['dimensions']['height']]

            # Get yaw from the 'z' rotation (in radians)
            yaw = label_obj['rotations']['z']

            # Create the Box object
            box = Box(
                center=center,
                size=dims,
                orientation=Quaternion(axis=[0, 0, 1], angle=yaw)
            )
            box.name = label_obj['name']
            boxes_for_this_frame.append(box)

        except KeyError as e:
            print(f"Warning: Skipping an object in {json_filepath} due to missing key: {e}")
            continue

    return boxes_for_this_frame
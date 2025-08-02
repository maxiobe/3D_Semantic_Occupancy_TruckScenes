import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from truckscenes.utils.data_classes import Box, LidarPointCloud
from pyquaternion import Quaternion
from truckscenes.truckscenes import TruckScenes
from copy import deepcopy
import torch
from pytorch3d.ops.iou_box3d import box3d_overlap

def get_boxes(trucksc: TruckScenes, sample: Dict[str, Any]) -> List[Box]:
    """ Retruns the bounding boxes of the given sample.

    Arguments:
        trucksc: TruckScenes dataset instance.
        sample: Reference sample to get the boxes from.

    Returns:
        boxes: List of box instances in the ego vehicle frame at the
            timestamp of the sample.
    """
    # Retrieve all sample annotations
    boxes = list(map(trucksc.get_box, sample['anns']))

    # Get reference ego pose (timestamp of the sample/annotations)
    ref_ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])

    # Transform boxes to ego frame
    for box in boxes:
        box.translate(-np.array(ref_ego_pose['translation']))
        box.rotate(Quaternion(ref_ego_pose['rotation']).inverse)

    return boxes


def transform_boxes_to_ego(boxes, ego_pose_record):
    """
    Transforms a list of boxes from global coordinates into ego vehicle coordinates.

    Args:
        boxes: List of Box instances in global/world frame.
        ego_pose_record: A dictionary with keys 'translation' and 'rotation'
                         describing the ego vehicle pose in global frame.

    Returns:
        A new list of Box instances in the ego vehicle coordinate frame.
    """
    transformed_boxes = []
    ego_translation = np.array(ego_pose_record['translation'])
    ego_rotation_inv = Quaternion(ego_pose_record['rotation']).inverse

    for box in boxes:
        box_copy = deepcopy(box)
        # Translate: global -> ego
        box_copy.translate(-ego_translation)
        # Rotate: global -> ego
        box_copy.rotate(ego_rotation_inv)
        transformed_boxes.append(box_copy)

    return transformed_boxes

# MODIFIED Function to match the [width, length, height] convention
def convert_boxes_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts parameterized 3D boxes to the 8 corner coordinates.
    MODIFIED: Assumes input dimensions are [width, length, height] and maps them
    directly to the box's local X, Y, and Z axes respectively.

    Args:
        boxes (torch.Tensor): A tensor of shape (N, 7) with parameters
                              [cx, cy, cz, w, l, h, yaw_rad].

    Returns:
        torch.Tensor: A tensor of shape (N, 8, 3) representing the box corners.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("Input tensor must be of shape (N, 7).")

    device = boxes.device
    corners_norm = torch.tensor([
        [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5], [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5], [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5],
    ], dtype=torch.float32, device=device)

    centers = boxes[:, 0:3]

    # --- FIX: Use dimensions directly without reordering ---
    # The input tensor's dimensions are [width, length, height].
    # We will use these to scale the x, y, and z axes of the canonical box.
    dims_wlh = boxes[:, 3:6]
    dims_lwh = dims_wlh[:, [1, 0, 2]]

    yaws = boxes[:, 6]

    # Create rotation matrices from yaw
    cos_yaws, sin_yaws = torch.cos(yaws), torch.sin(yaws)
    zeros, ones = torch.zeros_like(cos_yaws), torch.ones_like(cos_yaws)
    rot_matrices = torch.stack([
        cos_yaws, -sin_yaws, zeros,
        sin_yaws, cos_yaws, zeros,
        zeros, zeros, ones
    ], dim=1).reshape(-1, 3, 3)

    scaled_corners = corners_norm.unsqueeze(0) * dims_lwh.unsqueeze(1)

    # Rotate and translate corners
    rotated_corners = torch.bmm(scaled_corners, rot_matrices.transpose(1, 2))
    final_corners = rotated_corners + centers.unsqueeze(1)

    return final_corners


def calculate_3d_iou_pytorch3d(boxes1_params: np.ndarray, boxes2_params: np.ndarray) -> np.ndarray:
    """
    Calculates the exact 3D IoU using PyTorch3D.
    Takes parameterized boxes, converts them to corners, and computes IoU.

    Args:
        boxes1_params (np.ndarray): Box parameters of shape (N, 7) [cx,cy,cz,w,l,h,yaw].
        boxes2_params (np.ndarray): Box parameters of shape (M, 7) [cx,cy,cz,w,l,h,yaw].

    Returns:
        np.ndarray: A matrix of IoU values of shape (N, M).
    """
    # Convert numpy arrays to GPU tensors
    b1_t = torch.from_numpy(boxes1_params).float().cuda()
    b2_t = torch.from_numpy(boxes2_params).float().cuda()

    # Convert parameterized boxes to 8 corner coordinates
    corners1 = convert_boxes_to_corners(b1_t)
    corners2 = convert_boxes_to_corners(b2_t)

    # Calculate IoU using the PyTorch3D CUDA kernel
    # The function returns (intersection_volume, iou_matrix)
    _, iou_matrix_gpu = box3d_overlap(corners1, corners2)

    return iou_matrix_gpu.cpu().numpy()


def calculate_3d_overlap_ratio_pytorch3d(boxes1_params: np.ndarray, boxes2_params: np.ndarray) -> np.ndarray:
    """
    Calculates the overlap ratio (Intersection / min(Volume)) using PyTorch3D.

    Args:
        boxes1_params (np.ndarray): Box parameters of shape (N, 7) [cx,cy,cz,w,l,h,yaw].
        boxes2_params (np.ndarray): Box parameters of shape (M, 7) [cx,cy,cz,w,l,h,yaw].

    Returns:
        np.ndarray: A matrix of overlap ratios of shape (N, M).
    """
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Convert numpy arrays to GPU tensors
    b1_t = torch.from_numpy(boxes1_params).float().to(device)
    b2_t = torch.from_numpy(boxes2_params).float().to(device)

    # Convert parameterized boxes to 8 corner coordinates
    corners1 = convert_boxes_to_corners(b1_t)
    corners2 = convert_boxes_to_corners(b2_t)

    # 1. Calculate Intersection Volume using the PyTorch3D CUDA kernel
    intersection_vol_gpu, _ = box3d_overlap(corners1, corners2)

    # 2. Calculate the volumes of all boxes
    # Dims are [w, l, h] at indices 3, 4, 5
    vol1_gpu = b1_t[:, 3] * b1_t[:, 4] * b1_t[:, 5]  # Shape: (N,)
    vol2_gpu = b2_t[:, 3] * b2_t[:, 4] * b2_t[:, 5]  # Shape: (M,)

    # 3. Create a matrix of the minimum volumes using broadcasting
    # Reshape volumes to (N, 1) and (1, M) to create an (N, M) matrix
    min_volume_matrix = torch.minimum(vol1_gpu.unsqueeze(1), vol2_gpu.unsqueeze(0))

    # 4. Calculate the overlap ratio
    # Add a small epsilon to avoid division by zero
    overlap_ratio_matrix_gpu = intersection_vol_gpu / (min_volume_matrix + 1e-6)

    # Clip the values to a maximum of 1.0 to correct for numerical inaccuracies
    overlap_ratio_matrix_gpu = torch.clamp(overlap_ratio_matrix_gpu, max=1.0)

    return overlap_ratio_matrix_gpu.cpu().numpy()


def get_object_overlap_signature_BATCH(frame_data, target_obj_idx_in_frame, min_threshold=0.01):
    """
        Calculates a RICH overlap signature including overlap ratio and centroid distance.
        """
    all_boxes_in_frame = frame_data['gt_bbox_3d_overlap_enlarged']

    # Isolate the single target box for the batch call
    target_box = all_boxes_in_frame[target_obj_idx_in_frame:target_obj_idx_in_frame + 1]  # Shape: (1, 7)

    overlap_ratio_row = calculate_3d_overlap_ratio_pytorch3d(target_box, all_boxes_in_frame)[0]
    target_obj_bbox_params = all_boxes_in_frame[target_obj_idx_in_frame]
    target_centroid = target_obj_bbox_params[:3]

    overlaps = []
    for other_obj_idx, other_obj_token in enumerate(frame_data['object_tokens']):
        if other_obj_idx == target_obj_idx_in_frame:
            continue

        ratio = overlap_ratio_row[other_obj_idx]

        if ratio > min_threshold:
            other_obj_bbox_params = all_boxes_in_frame[other_obj_idx]
            other_centroid = other_obj_bbox_params[:3]

            centroid_dist = np.linalg.norm(target_centroid - other_centroid)
            relative_yaw = calculate_relative_yaw(target_obj_bbox_params, other_obj_bbox_params)
            other_obj_category = frame_data['converted_object_category'][other_obj_idx]

            overlaps.append((ratio, centroid_dist, other_obj_token, other_obj_category, relative_yaw))

    overlaps.sort(key=lambda x: x[2])
    return overlaps


def compare_signatures(sig1, sig2,
                       iou_strong_threshold=0.05,
                       iou_strong_tolerance=0.01,
                       yaw_tolerance_rad=0.1):
    """
    Compares two overlap signatures with tiered logic for weak and strong overlaps.
    """
    # 1. Check for same number of interacting objects
    if len(sig1) != len(sig2):
        return False

    # 2. Handle the isolated case
    if not sig1:
        return True

    # 3. Iterate and compare each corresponding interaction
    for i in range(len(sig1)):
        iou1, token1, cat1, yaw1 = sig1[i]
        iou2, token2, cat2, yaw2 = sig2[i]

        # 3a. Tokens must match
        if token1 != token2:
            return False

        # 3b. Relative yaws must be similar
        if abs(yaw1 - yaw2) > yaw_tolerance_rad:
            return False

        # ### MODIFICATION: Tiered IoU Comparison Logic ###
        is_strong1 = iou1 > iou_strong_threshold
        is_strong2 = iou2 > iou_strong_threshold

        if is_strong1 and is_strong2:
            # BOTH are strong overlaps. Compare their values with a tolerance.
            if abs(iou1 - iou2) > iou_strong_tolerance:
                return False
        elif not is_strong1 and not is_strong2:
            # BOTH are weak overlaps. We consider them a match without checking the
            # exact IoU value, as they are both just "slight touches".
            pass
        else:
            # One is strong and one is weak. This is a clear mismatch in state.
            return False

    # If all checks passed, the signatures are a match
    return True


def compare_signatures_class_based(sig1, target_cat1, sig2, target_cat2, thresholds_config):
    """
    Compares two overlap signatures using class-based, dynamic thresholds.

    Args:
        sig1 (list): Signature of the first object.
        target_cat1 (int): Class ID of the first target object.
        sig2 (list): Signature of the second object.
        target_cat2 (int): Class ID of the second target object.
        thresholds_config (dict): The configuration dictionary with class-based thresholds.

    Returns:
        bool: True if the signatures are a match based on class-specific rules.
    """
    # 1. Check if the target objects are even the same class
    if target_cat1 != target_cat2:
        return False

    # 2. Check for same number of interacting objects
    if len(sig1) != len(sig2):
        return False

    # 3. Handle the isolated case
    if not sig1:
        return True

    # 4. Iterate and compare each corresponding interaction
    for i in range(len(sig1)):
        iou1, token1, neighbor_cat1, yaw1 = sig1[i]
        iou2, token2, neighbor_cat2, yaw2 = sig2[i]

        # 4a. Tokens and neighbor classes must match
        if token1 != token2 or neighbor_cat1 != neighbor_cat2:
            return False

        # Create the key for the dictionary lookup
        class_pair_key = frozenset({target_cat1, neighbor_cat1})

        # Get the specific thresholds for this pair, falling back to 'default' if not found
        params = thresholds_config.get(class_pair_key, thresholds_config['default'])

        iou_strong_threshold = params['iou_strong_threshold']
        iou_strong_tolerance = params['iou_strong_tolerance']
        yaw_tolerance_rad = params['yaw_tolerance_rad']
        # --- END DYNAMIC LOOKUP ---

        # 4b. Relative yaws must be similar (using the dynamic tolerance)
        if abs(yaw1 - yaw2) > yaw_tolerance_rad:
            return False

        # 4c. Tiered IoU Comparison Logic (using the dynamic thresholds)
        is_strong1 = iou1 > iou_strong_threshold
        is_strong2 = iou2 > iou_strong_threshold

        if is_strong1 and is_strong2:
            # BOTH are strong overlaps. Compare their values with the dynamic tolerance.
            if abs(iou1 - iou2) > iou_strong_tolerance:
                return False
        elif not is_strong1 and not is_strong2:
            # BOTH are weak overlaps. This is a match.
            pass
        else:
            # One is strong and one is weak. This is a clear mismatch in state.
            return False

    # If all checks passed for all interactions, the signatures are a match
    return True


def compare_signatures_class_based_OVERLAP_RATIO(sig1, target_cat1, sig2, target_cat2, thresholds_config):
    """
    Compares two overlap signatures using class-based thresholds for Overlap Ratio.
    """
    if target_cat1 != target_cat2 or len(sig1) != len(sig2):
        return False
    if not sig1:
        return True  # An isolated object is a match with another isolated object

    for i in range(len(sig1)):
        ratio1, dist1, token1, neighbor_cat1, yaw1 = sig1[i]
        ratio2, dist2, token2, neighbor_cat2, yaw2 = sig2[i]

        if token1 != token2 or neighbor_cat1 != neighbor_cat2:
            return False

        class_pair_key = frozenset({target_cat1, neighbor_cat1})
        params = thresholds_config.get(class_pair_key, thresholds_config['default'])

        # --- Check 1: Centroid Distance (Primary, most reliable) ---
        if abs(dist1 - dist2) > params['dist_tolerance_m']:
            return False

        # --- Check 2: Relative Yaw (Secondary, reliable) ---
        if abs(yaw1 - yaw2) > params['yaw_tolerance_rad']:
            return False

        # --- Check 3: Overlap Ratio (Weak sanity check) ---
        # This check only fails if the ratios are wildly different.
        if abs(ratio1 - ratio2) > params['ratio_absolute_tolerance']:
            return False

    # If all interacting pairs pass the checks, the signatures match
    return True


def are_box_sizes_similar(box1_params, box2_params,
                          volume_ratio_tolerance=1.1,
                          dim_ratio_tolerance=1.1):
    """
    Checks if the dimensions of two bounding boxes are similar based on their volume ratio.

    Args:
        box1_params (np.ndarray): Parameters for box 1 [cx, cy, cz, l, w, h, yaw].
        box2_params (np.ndarray): Parameters for box 2 in the same format.
        volume_ratio_tolerance (float): How much larger one volume can be than the other.
                                        E.g., 1.2 means a 20% difference is allowed.

    Returns:
        bool: True if the box sizes are considered similar.
    """
    # Extract dimensions (l, w, h)
    # --- 1. Individual Dimension Check (New) ---
    dims1 = box1_params[3:6]
    dims2 = box2_params[3:6]

    dim_ratios = np.maximum(dims1, dims2) / (np.minimum(dims1, dims2) + 1e-6)

    # Check if ALL individual dimension ratios are within the tolerance
    individual_dims_ok = np.all(dim_ratios <= dim_ratio_tolerance)

    # --- 2. Overall Volume Check (Original Logic) ---
    volume1 = dims1[0] * dims1[1] * dims1[2]
    volume2 = dims2[0] * dims2[1] * dims2[2]

    # Avoid division by zero if a box has no volume
    if volume1 < 1e-6 or volume2 < 1e-6:
        # Handle zero-volume case
        volume_ok = (volume1 < 1e-6) and (volume2 < 1e-6)
    else:
        # Handle non-zero volume case
        ratio = max(volume1, volume2) / min(volume1, volume2)
        volume_ok = ratio <= volume_ratio_tolerance

    # --- 3. Final Result ---
    # Both conditions must be true for the boxes to be considered similar in size and shape.
    return individual_dims_ok and volume_ok


def is_point_centroid_z_similar(points1, points2, category_id,
                                target_category_id, z_tolerance=0.2):
    """
    Checks if the Z-centroid of two point clouds are similar, but only for a specific category.
    This is highly effective for vehicles like forklifts where the point cloud's center of mass
    changes vertically with the load.

    Args:
        points1 (np.ndarray): Point cloud (N,3) for the object in the keyframe.
        points2 (np.ndarray): Point cloud (M,3) for the object in the candidate frame.
        category_id (int): The learning category ID of the object.
        target_category_id (int): The specific category ID to apply this check to.
        z_tolerance (float): The maximum allowed vertical distance (in meters) between point centroids.

    Returns:
        bool: True if the point cloud Z-centroids are considered similar.
    """
    # If the object is not the target category, we don't apply this check.
    if category_id != target_category_id:
        return True

    # Handle cases where one or both point clouds might be empty.
    if points1.shape[0] == 0 and points2.shape[0] == 0:
        return True  # Both empty is a match.
    if points1.shape[0] == 0 or points2.shape[0] == 0:
        return False  # One empty and one not is a mismatch.

    # Calculate the mean of the Z-coordinate (the 3rd column, index 2) for each cloud.
    centroid_z1 = np.mean(points1[:, 2])
    centroid_z2 = np.mean(points2[:, 2])

    # Return True only if the vertical distance is within our tolerance.
    return abs(centroid_z1 - centroid_z2) <= z_tolerance


def calculate_relative_yaw(box1_params, box2_params):
    """
    Calculates the shortest angle difference between the yaws of two boxes.

    Args:
        box1_params (np.ndarray): Parameters for box 1 [..., yaw].
        box2_params (np.ndarray): Parameters for box 2 [..., yaw].

    Returns:
        float: The relative yaw in radians, in the range [-pi, pi].
    """
    yaw1 = box1_params[6]
    yaw2 = box2_params[6]

    # Calculate the difference
    delta_yaw = yaw1 - yaw2

    # Normalize the angle to the range [-pi, pi] to get the shortest path
    relative_yaw = np.arctan2(np.sin(delta_yaw), np.cos(delta_yaw))

    return relative_yaw
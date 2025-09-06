import numpy as np
from scipy.spatial import KDTree
from typing import Any, Dict, List, Optional, Union, Tuple
import torch
from .bbox_utils import calculate_3d_overlap_ratio_pytorch3d, calculate_3d_iou_pytorch3d
import math
from tqdm import tqdm
from pyquaternion import Quaternion
from truckscenes.utils.geometry_utils import transform_matrix
from truckscenes.utils.data_classes import Box
from collections import defaultdict
from mmcv.ops.points_in_boxes import (points_in_boxes_cpu, points_in_boxes_all)
from .visualization import visualize_pointcloud, visualize_pointcloud_bbox

def quaternion_to_matrix_pytorch(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of quaternions to a batch of rotation matrices.
    Args:
        quaternions: (..., 4), tensor of quaternions in (w, x, y, z) format.
    Returns:
        (..., 3, 3), batch of rotation matrices.
    """
    w, x, y, z = torch.unbind(quaternions, -1)
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    mat = torch.stack([
        1.0 - (tyy + tzz), txy - twz, txz + twy,
        txy + twz, 1.0 - (txx + tzz), tyz - twx,
        txz - twy, tyz + twx, 1.0 - (txx + tyy)
    ], dim=-1).view(-1, 3, 3)

    # If the input had batch dimensions, reshape to match
    if quaternions.dim() > 1:
        mat = mat.view(*quaternions.shape[:-1], 3, 3)

    return mat


# Helper to create a (w, x, y, z) quaternion from yaw.
def torch_quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Creates a quaternion from a yaw angle tensor on the same device."""
    half_yaw = yaw / 2.0
    cos_half_yaw = torch.cos(half_yaw)
    sin_half_yaw = torch.sin(half_yaw)
    zero = torch.zeros_like(yaw)
    # Format: (w, x, y, z)
    return torch.stack([cos_half_yaw, zero, zero, sin_half_yaw], dim=-1)


# Helper to build the 4x4 transform matrix using our new function.
def transform_matrix_pytorch(translation: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """Creates a 4x4 transformation matrix on the GPU from translation and quaternion."""
    device = translation.device
    T = torch.eye(4, device=device)
    T[:3, :3] = quaternion_to_matrix_pytorch(quaternion)
    T[:3, 3] = translation
    return T

def resolve_overlap_by_distance_gpu(
        ambiguous_points_xyz: torch.Tensor,
        clean_points_1_xyz: torch.Tensor,
        clean_points_2_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Resolves ambiguity by assigning each point to the class of the nearest "clean" cluster.
    Runs entirely on the GPU.

    Returns:
        A boolean tensor where True means the point is closer to cluster 1.
    """
    # Handle edge cases where one cluster might be empty (e.g., full containment)
    if clean_points_1_xyz.shape[0] == 0 and clean_points_2_xyz.shape[0] == 0:
        # No clean points for either, cannot make a decision. Default to False (closer to 2).
        return torch.zeros(ambiguous_points_xyz.shape[0], dtype=torch.bool, device=ambiguous_points_xyz.device)
    if clean_points_1_xyz.shape[0] == 0:
        # Only cluster 2 has clean points, so all ambiguous points are closer to 2.
        return torch.zeros(ambiguous_points_xyz.shape[0], dtype=torch.bool, device=ambiguous_points_xyz.device)
    if clean_points_2_xyz.shape[0] == 0:
        # Only cluster 1 has clean points.
        return torch.ones(ambiguous_points_xyz.shape[0], dtype=torch.bool, device=ambiguous_points_xyz.device)

    # Calculate all-to-all distances from ambiguous points to clean points
    dist_to_c1 = torch.cdist(ambiguous_points_xyz, clean_points_1_xyz)
    dist_to_c2 = torch.cdist(ambiguous_points_xyz, clean_points_2_xyz)

    # Find the minimum distance from each ambiguous point to each cluster
    min_dist_to_c1 = torch.min(dist_to_c1, dim=1).values
    min_dist_to_c2 = torch.min(dist_to_c2, dim=1).values

    del dist_to_c1, dist_to_c2

    # Return a boolean tensor indicating which cluster was closer for each point
    return min_dist_to_c1 < min_dist_to_c2

def visualize_combined_clusters(ambiguous_pts: torch.Tensor,
                                  clean_pts_1: torch.Tensor,
                                  clean_pts_2: torch.Tensor,
                                  title="Cluster Visualization"):
    """
    Combines and visualizes ambiguous and clean point cloud clusters.

    Args:
        ambiguous_pts (torch.Tensor): The ambiguous points.
        clean_pts_1 (torch.Tensor): The first set of clean points.
        clean_pts_2 (torch.Tensor): The second set of clean points.
        title (str): The window title for the visualization.
    """
    # 1. Convert all tensors to NumPy arrays on the CPU
    ambiguous_np = ambiguous_pts.cpu().numpy()
    clean_1_np = clean_pts_1.cpu().numpy()
    clean_2_np = clean_pts_2.cpu().numpy()

    # 2. Assign a unique label to each cluster (0, 1, 2)
    # This creates a new column for each point cloud.
    ambiguous_labeled = np.hstack([ambiguous_np, np.full((ambiguous_np.shape[0], 1), 0)])
    clean_1_labeled = np.hstack([clean_1_np, np.full((clean_1_np.shape[0], 1), 1)])
    clean_2_labeled = np.hstack([clean_2_np, np.full((clean_2_np.shape[0], 1), 2)])

    # 3. Stack all three labeled arrays into a single point cloud
    combined_cloud = np.vstack([ambiguous_labeled, clean_1_labeled, clean_2_labeled])

    # 4. Call your existing visualizer
    # It will automatically use the 4th column (our new label) for coloring.
    visualize_pointcloud(combined_cloud, title=title)

def resolve_overlap_by_distance_gpu_batched(
        ambiguous_points_xyz: torch.Tensor,
        clean_points_1_xyz: torch.Tensor,
        clean_points_2_xyz: torch.Tensor,
        batch_size: int = 1024 # Process 4096 ambiguous points at a time
) -> torch.Tensor:
    """
    Resolves ambiguity by distance to nearest "clean" cluster, using BATCHING to conserve memory.
    """
    # Handle edge cases where one cluster might be empty
    if clean_points_1_xyz.shape[0] == 0:
        return torch.zeros(ambiguous_points_xyz.shape[0], dtype=torch.bool, device=ambiguous_points_xyz.device)
    if clean_points_2_xyz.shape[0] == 0:
        return torch.ones(ambiguous_points_xyz.shape[0], dtype=torch.bool, device=ambiguous_points_xyz.device)

    print(f"Clean 1: {clean_points_1_xyz.shape}, Clean 2: {clean_points_2_xyz.shape}")
    print(f"Ambigious points: {ambiguous_points_xyz.shape}")

    """visualize_combined_clusters(
        ambiguous_points_xyz,
        clean_points_1_xyz,
        clean_points_2_xyz,
        title="Ambiguous (Red) vs Clean Clusters (Green/Blue)"
    )
    visualize_pointcloud(clean_points_1_xyz.cpu().numpy())
    visualize_pointcloud(clean_points_2_xyz.cpu().numpy())
    visualize_pointcloud(ambiguous_points_xyz.cpu().numpy())"""

    num_ambiguous = ambiguous_points_xyz.shape[0]
    min_dists_c1 = torch.full((num_ambiguous,), float('inf'), device=ambiguous_points_xyz.device)
    min_dists_c2 = torch.full((num_ambiguous,), float('inf'), device=ambiguous_points_xyz.device)

    # Process ambiguous points in batches to prevent OOM errors
    for i in range(0, num_ambiguous, batch_size):
        batch_end = min(i + batch_size, num_ambiguous)
        batch_points = ambiguous_points_xyz[i:batch_end]

        # Calculate distance for the current batch
        dist_to_c1_batch = torch.cdist(batch_points, clean_points_1_xyz)
        dist_to_c2_batch = torch.cdist(batch_points, clean_points_2_xyz)

        # Find the minimum distance for each point in the batch and store it
        min_dists_c1[i:batch_end] = torch.min(dist_to_c1_batch, dim=1).values
        min_dists_c2[i:batch_end] = torch.min(dist_to_c2_batch, dim=1).values

    # Compare the final minimum distances for all points
    return min_dists_c1 < min_dists_c2


def resolve_overlap_by_distance_gpu_batched_robust(
        ambiguous_points_xyz: torch.Tensor,
        clean_points_1_xyz: torch.Tensor,
        clean_points_2_xyz: torch.Tensor,
        batch_size: int = 4096,
        clean_batch_size: int = 16384  # Add a new batch size for the clean points
) -> torch.Tensor:
    # Handle edge cases where one cluster might be empty
    if clean_points_1_xyz.shape[0] == 0:
        return torch.zeros(ambiguous_points_xyz.shape[0], dtype=torch.bool, device=ambiguous_points_xyz.device)
    if clean_points_2_xyz.shape[0] == 0:
        return torch.ones(ambiguous_points_xyz.shape[0], dtype=torch.bool, device=ambiguous_points_xyz.device)

    print(f"Clean 1: {clean_points_1_xyz.shape}, Clean 2: {clean_points_2_xyz.shape}")
    print(f"Ambigious points: {ambiguous_points_xyz.shape}")

    num_ambiguous = ambiguous_points_xyz.shape[0]
    min_dists_c1 = torch.full((num_ambiguous,), float('inf'), device=ambiguous_points_xyz.device)
    min_dists_c2 = torch.full((num_ambiguous,), float('inf'), device=ambiguous_points_xyz.device)

    # Process ambiguous points in batches
    for i in range(0, num_ambiguous, batch_size):
        batch_end = min(i + batch_size, num_ambiguous)
        batch_points = ambiguous_points_xyz[i:batch_end]

        # --- MODIFICATION FOR ROBUSTNESS ---

        # Calculate min distance to cluster 1 by batching through clean_points_1
        batch_min_c1 = torch.full((batch_points.shape[0],), float('inf'), device=batch_points.device)
        for j in range(0, clean_points_1_xyz.shape[0], clean_batch_size):
            clean_batch_1 = clean_points_1_xyz[j:min(j + clean_batch_size, clean_points_1_xyz.shape[0])]
            dist_to_c1_sub_batch = torch.cdist(batch_points, clean_batch_1)
            # Find the minimum distance for this sub-batch and update if it's smaller
            current_min_c1, _ = torch.min(dist_to_c1_sub_batch, dim=1)
            batch_min_c1 = torch.min(batch_min_c1, current_min_c1)
        min_dists_c1[i:batch_end] = batch_min_c1

        # Calculate min distance to cluster 2 by batching through clean_points_2
        batch_min_c2 = torch.full((batch_points.shape[0],), float('inf'), device=batch_points.device)
        for j in range(0, clean_points_2_xyz.shape[0], clean_batch_size):
            clean_batch_2 = clean_points_2_xyz[j:min(j + clean_batch_size, clean_points_2_xyz.shape[0])]
            dist_to_c2_sub_batch = torch.cdist(batch_points, clean_batch_2)
            # Find the minimum distance for this sub-batch and update
            current_min_c2, _ = torch.min(dist_to_c2_sub_batch, dim=1)
            batch_min_c2 = torch.min(batch_min_c2, current_min_c2)
        min_dists_c2[i:batch_end] = batch_min_c2

    # Compare the final minimum distances for all points
    return min_dists_c1 < min_dists_c2

def _resolve_truck_overlap_with_dual_obb(
    points_to_process: torch.Tensor,
    truck_box: torch.Tensor,
    trailer_box: torch.Tensor, # Can be None if no trailer is involved
    hitch_height: float,
    ID_TRUCK: int,
    winning_other_class: int, # Pass in the class that wins if it's not the truck
    device: torch.device
) -> torch.Tensor:
    """
    Contains the core logic for the dual OBB (chassis/cab) test.
    """
    # --- This is your duplicated code, now in one place ---
    cx, cy, cz, w, l, h, yaw = truck_box
    dims_chassis = torch.tensor([l, w, hitch_height], device=device)
    z_bottom_chassis = cz - h / 2.0
    bottom_center_chassis = torch.tensor([cx, cy, z_bottom_chassis], device=device)
    box_chassis_def = torch.cat([bottom_center_chassis, dims_chassis, yaw.unsqueeze(0)]).unsqueeze(0)

    T_world_from_truck = transform_matrix_pytorch(truck_box[:3], torch_quat_from_yaw(yaw))
    T_truck_from_world = torch.linalg.inv(T_world_from_truck)
    boundary_x_local = -l / 2.0

    if trailer_box is not None:
        trailer_center_world_homo = torch.cat([trailer_box[:3], torch.tensor([1.0], device=device)])
        trailer_center_local = (T_truck_from_world @ trailer_center_world_homo)[:3]
        boundary_x_local = trailer_center_local[0] + trailer_box[4] / 2.0

    cab_len = l / 2.0 - boundary_x_local
    cab_height = h - hitch_height
    dims_cab = torch.tensor([cab_len, w, cab_height], device=device)
    center_x_cab_local = boundary_x_local + cab_len / 2.0
    center_y_cab_local = 0.0
    z_bottom_cab_local = (-h / 2.0) + hitch_height
    bottom_center_cab_local_homo = torch.tensor([center_x_cab_local, center_y_cab_local, z_bottom_cab_local, 1.0], device=device)
    bottom_center_cab_world = (T_world_from_truck @ bottom_center_cab_local_homo)[:3]
    box_cab_def = torch.cat([bottom_center_cab_world, dims_cab, yaw.unsqueeze(0)]).unsqueeze(0)

    points_b = points_to_process.unsqueeze(0)
    in_chassis_mask = points_in_boxes_all(points_b, box_chassis_def.unsqueeze(0)).squeeze().bool()
    in_cab_mask = points_in_boxes_all(points_b, box_cab_def.unsqueeze(0)).squeeze().bool()

    is_truck_point = in_chassis_mask | in_cab_mask
    final_labels = torch.where(is_truck_point, ID_TRUCK, winning_other_class)
    return final_labels


def assign_label_by_dual_obb_check(
        overlap_idxs: np.ndarray,
        pt_to_box_map: Dict[int, List[int]],
        points: np.ndarray,
        boxes: np.ndarray,
        boxes_iou: np.ndarray,
        dyn_points_in_boxes: torch.Tensor,
        box_cls_labels: np.ndarray,
        pt_labels: np.ndarray,
        ID_TRAILER: int,
        ID_TRUCK: int,
        ID_FORKLIFT: int,
        ID_PEDESTRIAN: int,
        device: torch.device,
        high_overlap_threshold: float = 0.85,
        hitch_height: float = 1.20,
) -> Tuple[np.ndarray, List[int]]:
    """
    Resolves overlaps by running points_in_boxes twice.
    This version correctly calculates the BOTTOM CENTER for each box,
    as required by the mmcv function.
    """
    print("--- Running Refinement (Corrected Dual OBB Check w/ Bottom Centers) ---")
    original_labels_gpu = torch.from_numpy(pt_labels.flatten()).long().to(device)
    new_labels_gpu = original_labels_gpu.clone()
    points_gpu = torch.from_numpy(points).float().to(device)
    boxes[:, 3] *= 1.10
    boxes_gpu = torch.from_numpy(boxes).float().to(device)

    truck_to_trailer_map = {}
    all_truck_indices = np.where(box_cls_labels == ID_TRUCK)[0]
    all_trailer_indices = np.where(box_cls_labels == ID_TRAILER)[0]

    if all_truck_indices.size > 0 and all_trailer_indices.size > 0:
        # Calculate IoU between all trucks and all trailers
        iou_matrix = calculate_3d_iou_pytorch3d(
            boxes_iou[all_truck_indices],
            boxes_iou[all_trailer_indices]
        )
        # For each truck, find the trailer it overlaps with most
        best_trailer_matches = np.argmax(iou_matrix, axis=1)
        for i, truck_idx in enumerate(all_truck_indices):
            # Only pair if overlap is significant
            if iou_matrix[i, best_trailer_matches[i]] > 0.01:
                truck_to_trailer_map[truck_idx] = all_trailer_indices[best_trailer_matches[i]]

    groups = defaultdict(list)
    for pi in overlap_idxs:
        box_indices = tuple(sorted(pt_to_box_map.get(pi, [])))
        if len(box_indices) > 1:
            groups[box_indices].append(pi)

    print(f"Found {len(groups)} unique overlap groups to process.")

    for box_indices_tuple, point_indices in tqdm(groups.items(), desc="Processing overlap groups"):

        overlapping_classes = {box_cls_labels[b_idx] for b_idx in box_indices_tuple}
        point_indices_gpu = torch.tensor(point_indices, device=device, dtype=torch.long)
        points_to_process_gpu = points_gpu[point_indices_gpu]

        is_truck_trailer_forklift_case = {ID_TRUCK, ID_TRAILER, ID_FORKLIFT}.issubset(overlapping_classes)
        is_truck_trailer_case = {ID_TRUCK, ID_TRAILER}.issubset(overlapping_classes)
        is_truck_forklift_case = {ID_TRUCK, ID_FORKLIFT}.issubset(overlapping_classes)
        is_trailer_forklift_case = {ID_FORKLIFT, ID_TRAILER}.issubset(overlapping_classes)

        if is_truck_trailer_forklift_case:
            print(f">>> Path Taken: 3-Way Truck-Trailer-Forklift Rule")
            truck_idx = next(b_idx for b_idx in box_indices_tuple if box_cls_labels[b_idx] == ID_TRUCK)
            trailer_idx = next(b_idx for b_idx in box_indices_tuple if box_cls_labels[b_idx] == ID_TRAILER)
            forklift_idx = next(b_idx for b_idx in box_indices_tuple if box_cls_labels[b_idx] == ID_FORKLIFT)

            truck_box = boxes_gpu[truck_idx]
            trailer_box = boxes_gpu[trailer_idx]

            cx, cy, cz, w, l, h, yaw = truck_box
            dims_chassis = torch.tensor([l, w, hitch_height], device=device)
            z_bottom_chassis = cz - h / 2.0
            bottom_center_chassis = torch.tensor([cx, cy, z_bottom_chassis], device=device)
            box_chassis_def = torch.cat([bottom_center_chassis, dims_chassis, yaw.unsqueeze(0)]).unsqueeze(0)
            T_world_from_truck = transform_matrix_pytorch(truck_box[:3], torch_quat_from_yaw(yaw))
            T_truck_from_world = torch.linalg.inv(T_world_from_truck)
            trailer_center_world_homo = torch.cat([trailer_box[:3], torch.tensor([1.0], device=device)])
            trailer_center_local = (T_truck_from_world @ trailer_center_world_homo)[:3]
            boundary_x_local = trailer_center_local[0] + trailer_box[4] / 2.0
            cab_len = l / 2.0 - boundary_x_local
            cab_height = h - hitch_height
            dims_cab = torch.tensor([cab_len, w, cab_height], device=device)
            center_x_cab_local = boundary_x_local + cab_len / 2.0
            z_bottom_cab_local = (-h / 2.0) + hitch_height
            bottom_center_cab_local_homo = torch.tensor([center_x_cab_local, 0.0, z_bottom_cab_local, 1.0],
                                                        device=device)
            bottom_center_cab_world = (T_world_from_truck @ bottom_center_cab_local_homo)[:3]
            box_cab_def = torch.cat([bottom_center_cab_world, dims_cab, yaw.unsqueeze(0)]).unsqueeze(0)
            points_b = points_to_process_gpu.unsqueeze(0)
            in_chassis_mask = points_in_boxes_all(points_b, box_chassis_def.unsqueeze(0)).squeeze().bool()
            in_cab_mask = points_in_boxes_all(points_b, box_cab_def.unsqueeze(0)).squeeze().bool()
            is_truck_point_mask = in_chassis_mask | in_cab_mask

            # Label the points that are definitively part of the truck
            truck_point_indices = point_indices_gpu[is_truck_point_mask]
            new_labels_gpu[truck_point_indices] = ID_TRUCK

            # --- STEP 2: Resolve remaining points between trailer and forklift ---

            remaining_mask = ~is_truck_point_mask
            remaining_indices_gpu = point_indices_gpu[remaining_mask]

            if remaining_indices_gpu.shape[0] > 0:
                remaining_points_xyz = points_to_process_gpu[remaining_mask, :3]

                # Get "clean" points for the trailer (in trailer, but not truck or forklift)
                mask_in_trailer = dyn_points_in_boxes[:, trailer_idx].bool()
                mask_in_truck = dyn_points_in_boxes[:, truck_idx].bool()
                mask_in_forklift = dyn_points_in_boxes[:, forklift_idx].bool()
                mask_clean_trailer = mask_in_trailer & ~mask_in_truck & ~mask_in_forklift
                clean_points_trailer_xyz = points_gpu[mask_clean_trailer, :3]

                # Get "clean" points for the forklift (in forklift, but not truck or trailer)
                mask_clean_forklift = mask_in_forklift & ~mask_in_truck & ~mask_in_trailer
                clean_points_forklift_xyz = points_gpu[mask_clean_forklift, :3]

                # Resolve based on distance
                is_closer_to_trailer = resolve_overlap_by_distance_gpu_batched(
                    remaining_points_xyz,
                    clean_points_trailer_xyz,
                    clean_points_forklift_xyz
                )

                winner_labels_remaining = torch.where(is_closer_to_trailer, ID_TRAILER, ID_FORKLIFT)
                new_labels_gpu[remaining_indices_gpu] = winner_labels_remaining


        elif is_truck_trailer_case:
            truck_idx = next(b_idx for b_idx in box_indices_tuple if box_cls_labels[b_idx] == ID_TRUCK)
            trailer_idx = next(b_idx for b_idx in box_indices_tuple if box_cls_labels[b_idx] == ID_TRAILER)

            final_labels_for_group = _resolve_truck_overlap_with_dual_obb(
                points_to_process_gpu, boxes_gpu[truck_idx], boxes_gpu[trailer_idx],
                hitch_height, ID_TRUCK, ID_TRAILER, device
            )

            new_labels_gpu[point_indices_gpu] = final_labels_for_group.long()

            continue

        elif is_truck_forklift_case:
            print(f">>> Path Taken: 2-Way Truck-Forklift Rule")
            truck_idx = next(b_idx for b_idx in box_indices_tuple if box_cls_labels[b_idx] == ID_TRUCK)

            if truck_idx in truck_to_trailer_map:
                trailer_idx = truck_to_trailer_map[truck_idx]

                final_labels_for_group = _resolve_truck_overlap_with_dual_obb(
                    points_to_process_gpu, boxes_gpu[truck_idx], boxes_gpu[trailer_idx],
                    hitch_height, ID_TRUCK, ID_FORKLIFT, device
                )

                new_labels_gpu[point_indices_gpu] = final_labels_for_group.long()

            else:
                # Fallback if no trailer is paired: truck wins ambiguity
                new_labels_gpu[point_indices_gpu] = ID_TRUCK
            continue

        elif is_trailer_forklift_case:
            print(f">>> Path Taken: 2-Way Trailer-Forklift Rule (Forced Distance Logic)")
            # This logic is taken directly from your general "low overlap" case
            b_idx1, b_idx2 = box_indices_tuple
            cls1, cls2 = box_cls_labels[b_idx1], box_cls_labels[b_idx2]

            dyn_mask = dyn_points_in_boxes.bool()

            nb0, nb1 = dyn_mask.shape
            num_points = points_gpu.shape[0]

            if nb0 == num_points:
                # dyn_mask is (num_points, num_boxes)
                mask_in_box1 = dyn_mask[:, b_idx1]
                mask_in_box2 = dyn_mask[:, b_idx2]
            elif nb1 == num_points:
                # dyn_mask is (num_boxes, num_points) --> transpose indexing
                mask_in_box1 = dyn_mask[b_idx1, :]
                mask_in_box2 = dyn_mask[b_idx2, :]
            else:
                raise RuntimeError(
                    f"Unexpected dyn_points_in_boxes shape: {dyn_mask.shape} for {num_points} points")

            total_points = dyn_points_in_boxes.shape[0]
            print(f"Points 'in' box {b_idx1}: {torch.sum(mask_in_box1).item()} / {total_points}")
            print(f"Points 'in' box {b_idx2}: {torch.sum(mask_in_box2).item()} / {total_points}")

            mask_clean_box1 = mask_in_box1 & ~mask_in_box2
            mask_clean_box2 = mask_in_box2 & ~mask_in_box1

            clean_points_1_xyz = points_gpu[mask_clean_box1, :3]
            clean_points_2_xyz = points_gpu[mask_clean_box2, :3]

            is_closer_to_1 = resolve_overlap_by_distance_gpu_batched(
                points_to_process_gpu[:, :3], clean_points_1_xyz, clean_points_2_xyz
            )
            winner_labels = torch.where(is_closer_to_1, cls1, cls2)
            new_labels_gpu[point_indices_gpu] = winner_labels

            continue

        else:
            point_indices_gpu = torch.tensor(point_indices, device=device, dtype=torch.long)
            points_to_process_gpu = points_gpu[point_indices_gpu, :3]  # We only need XYZ for this logic

            is_pedestrian_case = ID_PEDESTRIAN in overlapping_classes and len(box_indices_tuple) == 2

            if is_pedestrian_case:
                # Get indices and classes for the pedestrian and the other object
                try:
                    ped_g_idx = next(b_idx for b_idx in box_indices_tuple if box_cls_labels[b_idx] == ID_PEDESTRIAN)
                    other_g_idx = next(b_idx for b_idx in box_indices_tuple if b_idx != ped_g_idx)
                except StopIteration:
                    # This should not happen if is_pedestrian_case is true, but as a safeguard:
                    continue

                other_cls = box_cls_labels[other_g_idx]

                # Check the overlap ratio first
                ped_box_np = boxes_iou[ped_g_idx:ped_g_idx + 1]
                other_box_np = boxes_iou[other_g_idx:other_g_idx + 1]
                overlap_ratio = calculate_3d_overlap_ratio_pytorch3d(ped_box_np, other_box_np)[0, 0]

                if overlap_ratio > high_overlap_threshold:
                    print(f">>> Path Taken: Advanced Pedestrian Rule for boxes {ped_g_idx} and {other_g_idx}")

                    # 1. Create a "core" pedestrian box with 40% width/length
                    ped_box_params = boxes_gpu[ped_g_idx]
                    cx, cy, cz, w, l, h, yaw = ped_box_params
                    core_w, core_l = w * 0.4, l * 0.4

                    # Create the core box definition (using bottom center for points_in_boxes_all)
                    z_bottom_core = cz - h / 2.0
                    bottom_center_core = torch.tensor([cx, cy, z_bottom_core], device=device)
                    dims_core = torch.tensor([core_l, core_w, h], device=device)  # l, w, h format
                    core_ped_box_def = torch.cat([bottom_center_core, dims_core, yaw.unsqueeze(0)]).unsqueeze(0)

                    # 2. Check which ambiguous points fall inside this core box
                    points_b = points_to_process_gpu.unsqueeze(0)
                    in_core_mask = points_in_boxes_all(points_b, core_ped_box_def.unsqueeze(0)).squeeze().bool()

                    # 3. Label points INSIDE the core box as pedestrian
                    inside_indices = point_indices_gpu[in_core_mask]
                    new_labels_gpu[inside_indices] = ID_PEDESTRIAN

                    # 4. Handle points OUTSIDE the core box with a distance check
                    outside_mask = ~in_core_mask
                    outside_indices = point_indices_gpu[outside_mask]

                    # Proceed only if there are points left to process
                    if outside_indices.shape[0] > 0:
                        outside_points_xyz = points_to_process_gpu[outside_mask, :3]

                        # Cluster 1: Points inside the pedestrian core
                        inside_points_xyz = points_to_process_gpu[in_core_mask, :3]

                        # Cluster 2: "Clean" points of the other object
                        mask_in_other = dyn_points_in_boxes[:, other_g_idx].bool()
                        mask_in_ped = dyn_points_in_boxes[:, ped_g_idx].bool()
                        mask_clean_other = mask_in_other & ~mask_in_ped
                        clean_points_other_xyz = points_gpu[mask_clean_other, :3]

                        # Resolve based on distance to the two new clusters
                        is_closer_to_ped_core = resolve_overlap_by_distance_gpu_batched(
                            outside_points_xyz,
                            inside_points_xyz,
                            clean_points_other_xyz
                        )
                        winner_labels = torch.where(is_closer_to_ped_core, ID_PEDESTRIAN, other_cls)
                        new_labels_gpu[outside_indices] = winner_labels

                    # Skip to the next group as this one is fully handled
                    continue


                    # --- Logic for 2-way overlaps ---
            if len(box_indices_tuple) == 2:
                b_idx1, b_idx2 = box_indices_tuple
                cls1, cls2 = box_cls_labels[b_idx1], box_cls_labels[b_idx2]

                #print(f"\n--- Debugging Group: Boxes {b_idx1}(cls {cls1}) vs {b_idx2}(cls {cls2}) ---")

                box1_params_np = boxes_iou[b_idx1:b_idx1 + 1]
                box2_params_np = boxes_iou[b_idx2:b_idx2 + 1]

                # Call the function with the expected NumPy array inputs.
                overlap_ratio = calculate_3d_overlap_ratio_pytorch3d(box1_params_np, box2_params_np)[0, 0]
                print(f"Calculated Overlap Ratio: {overlap_ratio:.4f} | Threshold: {high_overlap_threshold}")

                # --- STAGE 2.1: High Overlap -> Assign to smaller volume box ---
                if overlap_ratio > high_overlap_threshold:
                    print(">>> Path Taken: High Overlap (Smaller Volume Rule)")
                    box1 = boxes_gpu[b_idx1]
                    box2 = boxes_gpu[b_idx2]
                    vol1 = torch.prod(box1[3:6])
                    vol2 = torch.prod(box2[3:6])
                    print(f"Box {b_idx1} Volume: {vol1.item():.2f} | Box {b_idx2} Volume: {vol2.item():.2f}")
                    winner_cls = cls1 if vol1 < vol2 else cls2
                    new_labels_gpu[point_indices_gpu] = winner_cls  # Assign all points to the same winner

                # --- STAGE 2.2: Low Overlap -> Assign based on distance to clean clusters ---
                else:
                    print(">>> Path Taken: Low Overlap (Distance to Clean Clusters Rule)")
                    # Find "clean" points using boolean masks on the GPU
                    dyn_mask = dyn_points_in_boxes.bool()

                    #print(dyn_mask.shape)

                    nb0, nb1 = dyn_mask.shape
                    num_points = points_gpu.shape[0]

                    if nb0 == num_points:
                        # dyn_mask is (num_points, num_boxes)
                        mask_in_box1 = dyn_mask[:, b_idx1]
                        mask_in_box2 = dyn_mask[:, b_idx2]
                    elif nb1 == num_points:
                        # dyn_mask is (num_boxes, num_points) --> transpose indexing
                        mask_in_box1 = dyn_mask[b_idx1, :]
                        mask_in_box2 = dyn_mask[b_idx2, :]
                    else:
                        raise RuntimeError(
                            f"Unexpected dyn_points_in_boxes shape: {dyn_mask.shape} for {num_points} points")

                    total_points = dyn_points_in_boxes.shape[0]
                    print(f"Points 'in' box {b_idx1}: {torch.sum(mask_in_box1).item()} / {total_points}")
                    print(f"Points 'in' box {b_idx2}: {torch.sum(mask_in_box2).item()} / {total_points}")

                    mask_clean_box1 = mask_in_box1 & ~mask_in_box2
                    mask_clean_box2 = mask_in_box2 & ~mask_in_box1

                    clean_points_1_xyz = points_gpu[mask_clean_box1, :3]
                    clean_points_2_xyz = points_gpu[mask_clean_box2, :3]

                    """print(f"Shape of clean_points_1_xyz: {clean_points_1_xyz.shape}")
                    if clean_points_1_xyz.shape[0] > 0:
                        print("First 5 points of clean_points_1_xyz:")
                        # Print the first 5 rows. The output will include tensor metadata.
                        print(clean_points_1_xyz[:5])
                    else:
                        print("clean_points_1_xyz is empty.")

                    # Inspect the second cloud
                    print(f"\nShape of clean_points_2_xyz: {clean_points_2_xyz.shape}")
                    if clean_points_2_xyz.shape[0] > 0:
                        print("First 5 points of clean_points_2_xyz:")
                        print(clean_points_2_xyz[:5])
                    else:
                        print("clean_points_2_xyz is empty.")"""

                    # Use the new GPU helper function
                    is_closer_to_1 = resolve_overlap_by_distance_gpu_batched(
                        points_to_process_gpu,
                        clean_points_1_xyz,
                        clean_points_2_xyz
                    )

                    # Assign a label for each point individually based on the result
                    winner_labels_for_group = torch.where(is_closer_to_1, cls1, cls2)
                    new_labels_gpu[point_indices_gpu] = winner_labels_for_group

            # --- Fallback for rare 3+ general overlaps ---
            elif len(box_indices_tuple) > 2:
                # Gather volumes for all overlapping boxes
                dims = boxes_gpu[list(box_indices_tuple), 3:6]
                volumes = torch.prod(dims, dim=1)

                # Find the index of the smallest volume within the local group
                smallest_vol_idx_local = torch.argmin(volumes)

                # Get the global box index and class label of the winner
                winner_g_idx = box_indices_tuple[smallest_vol_idx_local]
                winner_cls = box_cls_labels[winner_g_idx]

                new_labels_gpu[point_indices_gpu] = winner_cls

            """pts_np = points_to_process_gpu.cpu().numpy()
            final_labels_for_group_gpu = new_labels_gpu[point_indices_gpu]
            labels_np = final_labels_for_group_gpu.cpu().numpy().reshape(-1, 1)
            pts_classed = np.hstack((pts_np, labels_np))

            viz_boxes = []
            for b_idx in box_indices_tuple:
                # Get the box parameters from the GPU tensor used in the logic
                box_params = boxes_gpu[b_idx].cpu().numpy()
                center = box_params[:3]
                dims = box_params[3:6]  # Note: This is [w, l, h] and reflects the 1.1x width enlargement
                yaw = box_params[6]

                # Create a Quaternion for the truckscenes Box object
                orientation = Quaternion(axis=[0, 0, 1], angle=yaw)

                # Instantiate the Box object your visualizer expects
                ts_box = Box(
                    center=center,
                    size=dims,
                    orientation=orientation
                )
                viz_boxes.append(ts_box)

            # 3. Call the visualization function
            print(f"Visualizing group of {len(pts_classed)} points with assigned labels and {len(viz_boxes)} boxes.")
            # This function will now show the points colored by their new labels.
            visualize_pointcloud_bbox(pts_classed, boxes=viz_boxes)"""


    new_labels_cpu = new_labels_gpu.cpu().numpy()
    reassigned_indices = torch.where(original_labels_gpu != new_labels_gpu)[0].cpu().numpy().tolist()

    print(f"Refinement complete. {len(reassigned_indices)} points were reassigned using the Dual OBB method.")
    return new_labels_cpu.reshape(-1, 1), reassigned_indices
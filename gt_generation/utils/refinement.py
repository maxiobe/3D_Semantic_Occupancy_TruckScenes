import numpy as np
from scipy.spatial import KDTree
from typing import Any, Dict, List, Optional, Union, Tuple
from shapely.geometry import Polygon, Point, MultiPolygon
import torch
from .bbox_utils import calculate_3d_overlap_ratio_pytorch3d, calculate_3d_iou_pytorch3d
import math
from tqdm import tqdm
from pyquaternion import Quaternion
from truckscenes.utils.geometry_utils import transform_matrix, points_in_box

def resolve_overlap_by_distance(
        ambiguous_points_xyz: np.ndarray,
        clean_points_1_xyz: np.ndarray,
        clean_points_2_xyz: np.ndarray
) -> np.ndarray:
    """
    Classifies ambiguous points based on their nearest neighbor in two clean point sets.

    Args:
        ambiguous_points_xyz: (N, 3) points to be classified.
        clean_points_1_xyz: (M, 3) "clean" points belonging to the first class.
        clean_points_2_xyz: (K, 3) "clean" points belonging to the second class.

    Returns:
        A boolean array of shape (N,) where True means the point belongs to class 1.
    """
    # Handle edge cases where one of the clean sets might be empty
    if clean_points_1_xyz.shape[0] == 0:
        return np.zeros(ambiguous_points_xyz.shape[0], dtype=bool)  # All points go to class 2
    if clean_points_2_xyz.shape[0] == 0:
        return np.ones(ambiguous_points_xyz.shape[0], dtype=bool)  # All points go to class 1

    # Build KD-Trees for efficient nearest neighbor search
    kdtree_1 = KDTree(clean_points_1_xyz)
    kdtree_2 = KDTree(clean_points_2_xyz)

    # Query the trees to find the distance to the nearest neighbor for all ambiguous points
    dist_to_1, _ = kdtree_1.query(ambiguous_points_xyz, k=1)
    dist_to_2, _ = kdtree_2.query(ambiguous_points_xyz, k=1)

    # Return a boolean mask: True if the point is closer to class 1
    is_closer_to_1 = dist_to_1 < dist_to_2
    return is_closer_to_1


def create_side_L_shapes(box_tr: Tuple, box_tl: Tuple, h_hitch: float = 1.5) -> Tuple[Polygon, float, float]:
    """
    Given two LOCAL-FRAME box tuples, returns ONLY the L-shape for the
    primary vehicle (box_tr), and its z-bounds.
    """
    cxt, _, szt, Lt, _, Ht, _ = box_tr
    cxl, _, _, Ll, _, _, _ = box_tl

    # Key Z-planes for the truck
    z_ground_tr = szt - Ht / 2.0
    z_roof_tr = szt + Ht / 2.0
    z_hitch = z_ground_tr + h_hitch

    # Key X-planes for the shape definition
    x_front_tr = cxt + Lt / 2.0
    x_back_tr = cxt - Lt / 2.0
    x_front_tl = cxl + Ll / 2.0

    tractor_pts = [
        (x_front_tr, z_ground_tr),
        (x_front_tr, z_roof_tr),
        (x_front_tl, z_roof_tr),
        (x_front_tl, z_hitch),
        (x_back_tr, z_hitch),
        (x_back_tr, z_ground_tr),
    ]
    tractor_L2D = Polygon(tractor_pts).buffer(0)

    # Only return the truck's information
    return tractor_L2D, z_ground_tr, z_roof_tr


def assign_label_by_L_shape(
        overlap_idxs: np.ndarray,
        pt_to_box_map: Dict[int, List[int]],
        points: np.ndarray,
        boxes: np.ndarray,
        boxes_iou: np.ndarray,
        box_cls_labels: np.ndarray,
        pt_labels: np.ndarray,
        all_object_points: List[np.ndarray],
        dyn_points_in_boxes: torch.Tensor,
        ID_TRAILER,
        ID_TRUCK,
        ID_FORKLIFT,
        high_overlap_threshold: float = 0.85,
) -> Tuple[np.ndarray, List[int]]:
    """
    Handles complex multi-class overlaps with a priority system:
    1. Resolves 3-way (Truck, Trailer, Forklift) overlaps.
    2. Resolves 2-way overlaps for all new combinations.
    """
    new_labels = pt_labels.flatten().copy()
    reassigned_point_indices = []

    # --- Step 1: Pre-compute Truck-to-Trailer Pairings via IoU ---
    truck_to_trailer_map = {}
    all_truck_indices = np.where(box_cls_labels == ID_TRUCK)[0]
    all_trailer_indices = np.where(box_cls_labels == ID_TRAILER)[0]
    if all_truck_indices.size > 0 and all_trailer_indices.size > 0:
        truck_boxes = boxes_iou[all_truck_indices]
        trailer_boxes = boxes_iou[all_trailer_indices]
        iou_matrix = calculate_3d_iou_pytorch3d(truck_boxes, trailer_boxes)
        # For each truck, find the trailer with the highest overlap
        for i, truck_idx in enumerate(all_truck_indices):
            if iou_matrix.shape[1] > 0:
                best_trailer_match_idx = np.argmax(iou_matrix[i, :])
                # Only create a pair if the overlap is significant
                if iou_matrix[i, best_trailer_match_idx] > 0.01:
                    # Map the global truck index to the global trailer index
                    truck_to_trailer_map[truck_idx] = all_trailer_indices[best_trailer_match_idx]

    truly_ambiguous_idxs = []
    for pi in tqdm(overlap_idxs, desc="Calculate truly ambiguous points", leave=False):
        overlapping_box_indices = pt_to_box_map.get(pi, [])
        unique_classes = set(box_cls_labels[b_idx] for b_idx in overlapping_box_indices)
        if len(unique_classes) > 1:
            truly_ambiguous_idxs.append(pi)

    print(f"Found {len(overlap_idxs)} points in overlapping zones. "
          f"Found {len(truly_ambiguous_idxs)} points with true inter-class ambiguity to resolve.")

    # --- Step 2: Iterate through ambiguous points and apply rules ---
    print(f"Refining {len(truly_ambiguous_idxs)} points with true inter-class ambiguity...")
    changed_ids = 0

    for pi in tqdm(truly_ambiguous_idxs, desc="Resolving point overlaps", leave=False):
        original_label = new_labels[pi]

        # Get all necessary info for this point on-the-fly
        overlapping_box_indices = pt_to_box_map.get(pi, [])
        overlapping_classes = {box_cls_labels[b_idx] for b_idx in overlapping_box_indices}
        point_to_class_map = {box_cls_labels[b_idx]: b_idx for b_idx in overlapping_box_indices}

        is_special_case = {ID_TRUCK, ID_TRAILER}.issubset(overlapping_classes) or \
                          {ID_TRUCK, ID_FORKLIFT}.issubset(overlapping_classes)

        if is_special_case:
            # --- Rule 1: Handle 3-way Truck, Trailer, Forklift overlaps (Highest Priority) ---
            if {ID_TRUCK, ID_TRAILER, ID_FORKLIFT}.issubset(overlapping_classes):
                idx_tr = point_to_class_map[ID_TRUCK]

                # Perform the L-shape check for the truck first
                # (Code to create L_tr and check point is condensed here for clarity)
                cx_tr, cy_tr, cz_tr, w_tr, l_tr, h_tr, yaw_tr, _ = boxes[idx_tr]
                c, s = math.cos(-yaw_tr), math.sin(-yaw_tr)
                trailer_center_in_truck_frame = transform_matrix([cx_tr, cy_tr, cz_tr],
                                                                 Quaternion(axis=[0, 0, 1], angle=yaw_tr),
                                                                 inverse=True) @ [
                                                    boxes[point_to_class_map[ID_TRAILER]][0],
                                                    boxes[point_to_class_map[ID_TRAILER]][1],
                                                    boxes[point_to_class_map[ID_TRAILER]][2], 1]
                L_tr, z_ground, z_roof = create_side_L_shapes(
                    (0, 0, 0, l_tr, w_tr, h_tr, 0),
                    (trailer_center_in_truck_frame[0], trailer_center_in_truck_frame[1],
                     trailer_center_in_truck_frame[2],
                     boxes[point_to_class_map[ID_TRAILER]][4], boxes[point_to_class_map[ID_TRAILER]][3],
                     boxes[point_to_class_map[ID_TRAILER]][5], 0)
                )
                xg, yg, zg = points[pi]
                dx_pt = (xg - cx_tr) * c - (yg - cy_tr) * s
                dz_pt = zg - cz_tr

                if L_tr.contains(Point(dx_pt, dz_pt)):
                    new_labels[pi] = ID_TRUCK
                    changed_ids += 1
                else:
                    # If not in truck's L-shape, your next priority is trailer
                    new_labels[pi] = ID_TRAILER
                    changed_ids += 1

            # --- Rule 2: Handle original Truck & Trailer overlaps ---
            elif {ID_TRUCK, ID_TRAILER}.issubset(overlapping_classes):
                idx_tr = point_to_class_map[ID_TRUCK]

                # Same L-shape logic as your preferred method
                cx_tr, cy_tr, cz_tr, w_tr, l_tr, h_tr, yaw_tr, _ = boxes[idx_tr]
                c, s = math.cos(-yaw_tr), math.sin(-yaw_tr)
                trailer_center_in_truck_frame = transform_matrix([cx_tr, cy_tr, cz_tr],
                                                                 Quaternion(axis=[0, 0, 1], angle=yaw_tr),
                                                                 inverse=True) @ [
                                                    boxes[point_to_class_map[ID_TRAILER]][0],
                                                    boxes[point_to_class_map[ID_TRAILER]][1],
                                                    boxes[point_to_class_map[ID_TRAILER]][2], 1]
                L_tr, _, _ = create_side_L_shapes(
                    (0, 0, 0, l_tr, w_tr, h_tr, 0),
                    (trailer_center_in_truck_frame[0], trailer_center_in_truck_frame[1],
                     trailer_center_in_truck_frame[2],
                     boxes[point_to_class_map[ID_TRAILER]][4], boxes[point_to_class_map[ID_TRAILER]][3],
                     boxes[point_to_class_map[ID_TRAILER]][5], 0)
                )
                xg, yg, zg = points[pi]
                dx_pt = (xg - cx_tr) * c - (yg - cy_tr) * s
                dz_pt = zg - cz_tr

                if L_tr.contains(Point(dx_pt, dz_pt)):
                    new_labels[pi] = ID_TRUCK
                    changed_ids += 1
                else:
                    new_labels[pi] = ID_TRAILER
                    changed_ids += 1

            # --- Rule 3: Handle 2-way Trailer & Forklift overlaps ---
            elif {ID_TRAILER, ID_FORKLIFT}.issubset(overlapping_classes):
                # Your logic: Trailer always wins
                new_labels[pi] = ID_TRAILER
                changed_ids += 1

            # --- Rule 4: Handle 2-way Truck & Forklift overlaps ---
            elif {ID_TRUCK, ID_FORKLIFT}.issubset(overlapping_classes):
                idx_tr = point_to_class_map[ID_TRUCK]

                if idx_tr in truck_to_trailer_map:
                    idx_tl = truck_to_trailer_map[idx_tr]

                    # Same L-shape logic as above
                    cx_tr, cy_tr, cz_tr, w_tr, l_tr, h_tr, yaw_tr, _ = boxes[idx_tr]
                    c, s = math.cos(-yaw_tr), math.sin(-yaw_tr)

                    # --- Use the paired trailer to create the precise L-shape ---
                    cx_tr, cy_tr, cz_tr, w_tr, l_tr, h_tr, yaw_tr, _ = boxes[idx_tr]
                    cx_tl, cy_tl, cz_tl, w_tl, l_tl, h_tl, yaw_tl, _ = boxes[idx_tl]

                    T_truck_from_world = transform_matrix([cx_tr, cy_tr, cz_tr],
                                                          Quaternion(axis=[0, 0, 1], angle=yaw_tr),
                                                          inverse=True)
                    trailer_center_in_world = np.array([cx_tl, cy_tl, cz_tl, 1])
                    trailer_center_in_truck_frame = T_truck_from_world @ trailer_center_in_world

                    L_tr, z_ground, z_roof = create_side_L_shapes(
                        (0, 0, 0, l_tr, w_tr, h_tr, 0),
                        (trailer_center_in_truck_frame[0], trailer_center_in_truck_frame[1],
                         trailer_center_in_truck_frame[2],
                         l_tl, w_tl, h_tl, 0)
                    )
                    xg, yg, zg = points[pi]
                    dx_pt = (xg - cx_tr) * c - (yg - cy_tr) * s
                    dz_pt = zg - cz_tr

                    if L_tr.contains(Point(dx_pt, dz_pt)):
                        new_labels[pi] = ID_TRUCK
                        changed_ids += 1
                    else:
                        new_labels[pi] = ID_FORKLIFT
                        changed_ids += 1

        """# --- Priority 2: Your New Two-Stage General Logic ---
        elif len(overlapping_box_indices) == 2:
            box1_idx, box2_idx = overlapping_box_indices
            box1_params = boxes_iou[box1_idx:box1_idx + 1]
            box2_params = boxes_iou[box2_idx:box2_idx + 1]
            overlap_ratio = calculate_3d_overlap_ratio_pytorch3d(box1_params, box2_params)[0, 0]

            # --- STAGE 1: High Overlap / Containment ---
            if overlap_ratio > high_overlap_threshold:
                # Assign to the class of the box with the smaller volume
                dims1 = boxes[box1_idx, 3:6]
                dims2 = boxes[box2_idx, 3:6]
                vol1 = dims1[0] * dims1[1] * dims1[2]
                vol2 = dims2[0] * dims2[1] * dims2[2]

                winner_idx = box1_idx if vol1 < vol2 else box2_idx
                new_labels[pi] = box_cls_labels[winner_idx]

            # --- STAGE 2: Low Overlap / Partial ---
            else:
                all_indices_in_box1 = torch.where(dyn_points_in_boxes[:, box1_idx])[0]
                all_indices_in_box2 = torch.where(dyn_points_in_boxes[:, box2_idx])[0]

                # "Clean" points for box 1 are those in box 1 BUT NOT in box 2.
                clean_indices_box1 = np.setdiff1d(all_indices_in_box1.cpu().numpy(), all_indices_in_box2.cpu().numpy())

                # "Clean" points for box 2 are those in box 2 BUT NOT in box 1.
                clean_indices_box2 = np.setdiff1d(all_indices_in_box2.cpu().numpy(), all_indices_in_box1.cpu().numpy())

                # Get the actual XYZ coordinates of these clean points from the main dynamic cloud
                clean_points_1_xyz = points[clean_indices_box1, :3]
                clean_points_2_xyz = points[clean_indices_box2, :3]

                # Using your existing function to resolve based on distance to clean clusters
                is_closer_to_1 = resolve_overlap_by_distance(
                    ambiguous_points_xyz=points[pi:pi + 1, :3],
                    clean_points_1_xyz=clean_points_1_xyz,
                    clean_points_2_xyz=clean_points_2_xyz
                )
                winner_idx = box1_idx if is_closer_to_1[0] else box2_idx
                new_labels[pi] = box_cls_labels[winner_idx]

        # Fallback for rare 3+ general overlaps: assign to smallest volume box
        elif len(overlapping_box_indices) > 2:
            smallest_vol = float('inf')
            winner_idx = -1
            for box_idx in overlapping_box_indices:
                dims = boxes[box_idx, 3:6]
                vol = dims[0] * dims[1] * dims[2]
                if vol < smallest_vol:
                    smallest_vol = vol
                    winner_idx = box_idx
            if winner_idx != -1:
                new_labels[pi] = box_cls_labels[winner_idx]"""

        # Record if the final label is different from the initial one
        if new_labels[pi] != original_label:
            reassigned_point_indices.append(pi)

    print(f"Assigned {changed_ids} overlap points")

    return new_labels.reshape(-1, 1), reassigned_point_indices
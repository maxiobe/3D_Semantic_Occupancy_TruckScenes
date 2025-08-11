import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Union, Tuple
from .constants import *
from pathlib import Path

def visualize_pointcloud(points, colors=None, title="Point Cloud"):
    """
    Visualize a point cloud using Open3D.
    Args:
        points: Nx3 or Nx4 numpy array of XYZ[+label/feature].
        colors: Optional Nx3 RGB array or a string-based colormap (e.g., "label").
        title: Optional window title.
    """
    if points.shape[1] < 3:
        print("Invalid point cloud shape:", points.shape)
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    if colors is not None:
        if isinstance(colors, str) and colors == "label" and points.shape[1] > 3:
            labels = points[:, 3].astype(int)
            max_label = labels.max() + 1
            cmap = plt.get_cmap("tab20", max_label)
            rgb = cmap(labels)[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(rgb)
        elif isinstance(colors, np.ndarray) and colors.shape == points[:, :3].shape:
            pcd.colors = o3d.utility.Vector3dVector(colors)
    elif points.shape[1] > 3:
        # Use label to colorize by default
        labels = points[:, 3].astype(int)
        max_label = labels.max() + 1
        cmap = plt.get_cmap("tab20", max_label)
        rgb = cmap(labels)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(rgb)

    print(f"Visualizing point cloud with {np.asarray(pcd.points).shape[0]} points")
    o3d.visualization.draw_geometries([pcd], window_name=title)


def visualize_pointcloud_bbox(points: np.ndarray,
                              boxes: Optional[List] = None,  # Use List[Box] if Box class is imported
                              colors: Optional[Union[np.ndarray, str]] = None,
                              title: str = "Point Cloud with BBoxes",
                              self_vehicle_range: Optional[List[float]] = None,  # New parameter
                              vis_self_vehicle: bool = False):
    """
    Visualize a point cloud and optional bounding boxes using Open3D.

    Args:
        points: Nx3 or Nx(>3) numpy array of XYZ[+label/feature].
        boxes: List of Box objects (e.g., from truckscenes) in the same coordinate frame as points.
               Assumes Box objects have .center, .wlh, and .orientation (pyquaternion.Quaternion) attributes.
        colors: Optional Nx3 RGB array or a string-based colormap (e.g., "label").
                If "label", assumes the 4th column of `points` contains integer labels.
        title: Optional window title.
        self_vehicle_range: Optional list [x_min, y_min, z_min, x_max, y_max, z_max] for the ego vehicle box.
        vis_self_vehicle: If True and self_vehicle_range is provided, draws the ego vehicle box.
    """

    geometries = []

    # --- Point cloud ---
    if points.ndim != 2 or points.shape[1] < 3:
        print(f"Error: Invalid point cloud shape: {points.shape}. Needs Nx(>=3).")
        return
    if points.shape[0] == 0:
        print("Warning: Point cloud is empty.")
        # Continue to potentially draw boxes

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    # --- Point Cloud Coloring ---
    use_label_coloring = False
    if colors is not None:
        if isinstance(colors, str) and colors.lower() == "label":
            if points.shape[1] > 3:
                use_label_coloring = True
            else:
                print("Warning: 'colors' set to 'label' but points array has < 4 columns.")
        elif isinstance(colors, np.ndarray) and colors.shape == points[:, :3].shape:
            # Ensure colors are float64 and in range [0, 1] for Open3D
            colors_float = colors.astype(np.float64)
            if np.max(colors_float) > 1.0:  # Basic check if maybe 0-255 range
                colors_float /= 255.0
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_float, 0.0, 1.0))
        else:
            print(
                f"Warning: Invalid 'colors' argument. Type: {type(colors)}, Value/Shape: {colors if isinstance(colors, str) else colors.shape}. Using default colors.")
    elif points.shape[1] > 3:  # Default to label coloring if 4th column exists and colors=None
        print("Info: No 'colors' provided, attempting to color by 4th column (label).")
        use_label_coloring = True

    if use_label_coloring:
        try:
            labels = points[:, 3].astype(int)
            unique_labels = np.unique(labels)
            if unique_labels.size > 0:
                # Map labels to colors
                min_label = unique_labels.min()
                max_label = unique_labels.max()
                label_range = max_label - min_label + 1
                # Use a colormap suitable for categorical data
                cmap = plt.get_cmap("tab20", label_range)
                # Normalize labels to 0..label_range-1 for colormap indexing
                normalized_labels = labels - min_label
                rgb = cmap(normalized_labels)[:, :3]  # Get RGB, ignore alpha
                pcd.colors = o3d.utility.Vector3dVector(rgb)
            else:
                print("Warning: Found label column, but no unique labels detected.")
        except Exception as e:
            print(f"Error applying label coloring: {e}. Using default colors.")
            use_label_coloring = False  # Revert if error occurred

    geometries.append(pcd)

    # --- Ego Vehicle Bounding Box ---
    if vis_self_vehicle and self_vehicle_range is not None:
        if len(self_vehicle_range) == 6:
            x_min_s, y_min_s, z_min_s, x_max_s, y_max_s, z_max_s = self_vehicle_range
            center_s = np.array([(x_min_s + x_max_s) / 2.0,
                                 (y_min_s + y_max_s) / 2.0,
                                 (z_min_s + z_max_s) / 2.0])
            # Open3D extent is [length(x), width(y), height(z)]
            extent_s = np.array([x_max_s - x_min_s,
                                 y_max_s - y_min_s,
                                 z_max_s - z_min_s])
            R_s = np.eye(3)  # Ego vehicle box is axis-aligned in its own coordinate frame
            ego_obb = o3d.geometry.OrientedBoundingBox(center_s, R_s, extent_s)
            ego_obb.color = (0.0, 0.8, 0.2)  # Green color for ego vehicle
            geometries.append(ego_obb)
        else:
            print(
                f"Warning: self_vehicle_range provided for ego vehicle but not of length 6. Got: {self_vehicle_range}")

    # --- Other Bounding boxes (Annotations) ---
    num_boxes_drawn = 0
    if boxes is not None:
        for i, box in enumerate(boxes):
            try:
                # Create Open3D OrientedBoundingBox from truckscenes Box properties
                center = box.center  # Should be numpy array (3,)
                # truckscenes Box.wlh = [width(y), length(x), height(z)]
                # o3d OrientedBoundingBox extent = [length(x), width(y), height(z)]
                extent = np.array([box.wlh[1], box.wlh[0], box.wlh[2]])
                # Get rotation matrix from pyquaternion.Quaternion
                R = box.orientation.rotation_matrix  # Should be 3x3 numpy array

                obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
                obb.color = (1.0, 0.0, 0.0)  # Set color to red
                geometries.append(obb)
                num_boxes_drawn += 1

            except AttributeError as e:
                print(
                    f"Error processing box {i} (Token: {getattr(box, 'token', 'N/A')}): Missing attribute {e}. Skipping box.")
            except Exception as e:
                print(f"Error processing box {i} (Token: {getattr(box, 'token', 'N/A')}): {e}. Skipping box.")

    # --- Visualize ---
    if not geometries:
        print("No geometries (point cloud or boxes) to visualize.")
        return

    point_count = np.asarray(pcd.points).shape[0]
    num_ego_box = 1 if (vis_self_vehicle and self_vehicle_range is not None and len(self_vehicle_range) == 6) else 0
    print(
        f"Visualizing point cloud with {point_count} points, {num_boxes_drawn} annotation boxes, and {num_ego_box} ego vehicle box.")
    o3d.visualization.draw_geometries(geometries, window_name=title)


def visualize_occupancy_o3d(voxel_state, voxel_label, pc_range, voxel_size,
                            class_color_map, default_color,
                            show_semantics=False, show_free=False, show_unobserved=False):
    """
    Visualizes occupancy grid using Open3D.

    Args:
        voxel_state (np.ndarray): 3D array, STATE_OCCUPIED (2), STATE_FREE (1), STATE_UNOBSERVED (0).
        voxel_label (np.ndarray): 3D array of same shape, per-voxel semantic label.
        pc_range (list or np.ndarray): [xmin, ymin, zmin, xmax, ymax, zmax].
        voxel_size (list or np.ndarray): [vx, vy, vz].
        class_color_map (dict): Mapping from semantic label index to RGB color.
        default_color (list): Default RGB color for labels not in class_color_map.
        show_semantics (bool): If True, color occupied voxels by their semantic label.
                               Otherwise, occupied voxels are red.
        show_free (bool): If True, visualize free voxels (colored light blue).
        show_unobserved (bool): If True, visualize unobserved voxels (colored gray).
    """
    geometries = []

    # --- Process Occupied Voxels ---
    occ_indices = np.where(voxel_state == STATE_OCCUPIED)
    if len(occ_indices[0]) > 0:
        # Convert voxel indices to world-coords of their centers
        xs_occ = (occ_indices[0].astype(float) + 0.5) * voxel_size[0] + pc_range[0]
        ys_occ = (occ_indices[1].astype(float) + 0.5) * voxel_size[1] + pc_range[1]
        zs_occ = (occ_indices[2].astype(float) + 0.5) * voxel_size[2] + pc_range[2]

        occupied_points_world = np.vstack((xs_occ, ys_occ, zs_occ)).T

        pcd_occupied = o3d.geometry.PointCloud()
        pcd_occupied.points = o3d.utility.Vector3dVector(occupied_points_world)

        if show_semantics:
            labels_occ = voxel_label[occ_indices]
            colors_occ = np.array([class_color_map.get(int(label), default_color) for label in labels_occ])
            pcd_occupied.colors = o3d.utility.Vector3dVector(colors_occ)
        else:
            pcd_occupied.paint_uniform_color([1.0, 0.0, 0.0])  # Red for occupied
        geometries.append(pcd_occupied)
    else:
        print("No occupied voxels to show.")

    # --- Process Free Voxels (Optional) ---
    if show_free:
        free_indices = np.where(voxel_state == STATE_FREE)
        if len(free_indices[0]) > 0:
            xs_free = (free_indices[0].astype(float) + 0.5) * voxel_size[0] + pc_range[0]
            ys_free = (free_indices[1].astype(float) + 0.5) * voxel_size[1] + pc_range[1]
            zs_free = (free_indices[2].astype(float) + 0.5) * voxel_size[2] + pc_range[2]
            free_points_world = np.vstack((xs_free, ys_free, zs_free)).T

            pcd_free = o3d.geometry.PointCloud()
            pcd_free.points = o3d.utility.Vector3dVector(free_points_world)
            pcd_free.paint_uniform_color([0.5, 0.7, 1.0])  # Light blue for free
            geometries.append(pcd_free)
        else:
            print("No free voxels to show (or show_free=False).")

    # --- Process Unobserved Voxels (Optional) ---
    if show_unobserved:
        unobserved_indices = np.where(voxel_state == STATE_UNOBSERVED)
        if len(unobserved_indices[0]) > 0:
            xs_unobs = (unobserved_indices[0].astype(float) + 0.5) * voxel_size[0] + pc_range[0]
            ys_unobs = (unobserved_indices[1].astype(float) + 0.5) * voxel_size[1] + pc_range[1]
            zs_unobs = (unobserved_indices[2].astype(float) + 0.5) * voxel_size[2] + pc_range[2]
            unobserved_points_world = np.vstack((xs_unobs, ys_unobs, zs_unobs)).T

            pcd_unobserved = o3d.geometry.PointCloud()
            pcd_unobserved.points = o3d.utility.Vector3dVector(unobserved_points_world)
            pcd_unobserved.paint_uniform_color([0.8, 0.8, 0.8])  # Light gray for unobserved
            geometries.append(pcd_unobserved)
        else:
            print("No unobserved voxels to show (or show_unobserved=False).")

    if not geometries:
        print("Nothing to visualize.")
        return

    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Occupancy Grid (Open3D)')
    for geom in geometries:
        vis.add_geometry(geom)

    vis.run()
    vis.destroy_window()


def visualize_mapmos_predictions(points_xyz: np.ndarray,
                                 predicted_labels: np.ndarray,
                                 scan_index: int,
                                 window_title_prefix: str = "MapMOS Prediction"):
    """
    Visualizes a point cloud, coloring points based on MapMOS predicted labels.

    Args:
        points_xyz (np.ndarray): Point cloud array of shape (N, 3) for XYZ coordinates.
        predicted_labels (np.ndarray): NumPy array of shape (N,) containing predicted labels.
                                      Assumes: 0=static, 1=dynamic, -1=not classified.
        scan_index (int): The index of the scan for the window title.
        window_title_prefix (str): Prefix for the visualization window title.
    """
    if points_xyz.shape[0] == 0:
        print(f"Scan {scan_index}: No points to visualize.")
        return
    if points_xyz.shape[0] != predicted_labels.shape[0]:
        print(
            f"Scan {scan_index}: Mismatch between number of points ({points_xyz.shape[0]}) and labels ({predicted_labels.shape[0]}). Skipping visualization.")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz[:, :3])

    # Define colors:
    # Static (label 0) = Blue
    # Dynamic (label 1) = Red
    # Not Classified (label -1) = Gray

    colors = np.zeros_like(points_xyz)  # Initialize with black

    static_mask = (predicted_labels == 0)
    dynamic_mask = (predicted_labels == 1)
    unclassified_mask = (predicted_labels == -1)

    colors[static_mask] = [0.0, 0.0, 1.0]  # Blue for static
    colors[dynamic_mask] = [1.0, 0.0, 0.0]  # Red for dynamic
    colors[unclassified_mask] = [0.5, 0.5, 0.5]  # Gray for not classified/out of range

    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"Visualizing Scan Index: {scan_index}")
    print(f"  Static points (blue): {np.sum(static_mask)}")
    print(f"  Dynamic points (red): {np.sum(dynamic_mask)}")
    print(f"  Unclassified points (gray): {np.sum(unclassified_mask)}")

    o3d.visualization.draw_geometries([pcd], window_name=f"{window_title_prefix} - Scan {scan_index}")


def calculate_and_plot_pose_errors(poses_estimated, poses_reference, title_prefix, scene_name, save_dir,
                                   show_plot=False):
    """
    Calculates and plots the translational and rotational errors between two sets of poses.

    Args:
        poses_estimated (np.ndarray): The set of estimated poses (N, 4, 4).
        poses_reference (np.ndarray): The set of reference/ground truth poses (N, 4, 4).
        title_prefix (str): A prefix for the plot title, e.g., "FlexCloud vs. GT".
        scene_name (str): The name of the scene for saving.
        save_dir (str or Path): The directory to save the plot in.
        show_plot (bool): Whether to display the plot interactively.
    """
    if poses_estimated.shape != poses_reference.shape:
        print(
            f"Warning: Pose array shapes do not match. Estimated: {poses_estimated.shape}, Reference: {poses_reference.shape}. Cannot compare.")
        return

    print(f"\n--- Comparing {title_prefix} Poses ---")
    trans_errors, rot_errors_rad = [], []
    trans_errors_x, trans_errors_y, trans_errors_z = [], [], []

    for k_idx in range(poses_estimated.shape[0]):
        pose_est = poses_estimated[k_idx]
        pose_ref = poses_reference[k_idx]

        # Translational error
        t_est = pose_est[:3, 3]
        t_ref = pose_ref[:3, 3]
        t_error_vec = t_ref - t_est
        trans_errors.append(np.linalg.norm(t_error_vec))
        trans_errors_x.append(t_error_vec[0])
        trans_errors_y.append(t_error_vec[1])
        trans_errors_z.append(t_error_vec[2])

        # Rotational error
        R_est = pose_est[:3, :3]
        R_ref = pose_ref[:3, :3]
        R_error = R_est.T @ R_ref
        trace_val = np.trace(R_error)
        clipped_arg = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        rot_errors_rad.append(np.arccos(clipped_arg))

    # --- Print Statistics ---
    print(f"  Average Translational Error : {np.mean(trans_errors):.4f} m")
    print(f"  Median Translational Error  : {np.median(trans_errors):.4f} m")
    print(f"  Average Rotational Error    : {np.mean(np.degrees(rot_errors_rad)):.4f} degrees")
    print(f"  Median Rotational Error     : {np.median(np.degrees(rot_errors_rad)):.4f} degrees")

    # --- Plotting ---
    fig, axs = plt.subplots(2, 2, figsize=(17, 10))
    fig.suptitle(f'Scene {scene_name}: {title_prefix} Pose Errors', fontsize=16)

    # (Plotting code is identical to your original, just using generic variable names)
    axs[0, 0].plot(trans_errors, label="Overall Trans. Error")
    axs[0, 0].set_title('Overall Translational Error')
    axs[0, 0].set_ylabel('Error (m)')
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(np.degrees(rot_errors_rad), label="Overall Rot. Error")
    axs[0, 1].set_title('Overall Rotational Error')
    axs[0, 1].set_ylabel('Error (degrees)')
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 0].plot(trans_errors_x, label="X Error (Ref - Est)", alpha=0.9)
    axs[1, 0].plot(trans_errors_y, label="Y Error (Ref - Est)", alpha=0.9)
    axs[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.7)
    axs[1, 0].set_title('X & Y Translational Errors')
    axs[1, 0].set_ylabel('Error (m)')
    axs[1, 0].grid(True)
    axs[1, 0].legend()

    axs[1, 1].plot(trans_errors_z, label="Z Error (Ref - Est)")
    axs[1, 1].axhline(0, color='black', linestyle='--', linewidth=0.7)
    axs[1, 1].set_title('Z Translational Error')
    axs[1, 1].set_ylabel('Error (m)')
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_save_dir = Path(save_dir) / scene_name
    plot_save_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = plot_save_dir / f"errors_{title_prefix.replace(' ', '_')}.png"
    plt.savefig(plot_filename)
    print(f"Saved pose error plot to {plot_filename}")

    if show_plot:
        plt.show()
    plt.close(fig)


def visualize_point_cloud_comparison(
        point_cloud_list: List[np.ndarray],
        frame_indices: List[int],
        colors: Optional[List[List[float]]] = None,
        scene_name: str = "Scene",
        window_size: Tuple[int, int] = (1280, 720)
) -> None:
    """
    Visualizes a selection of point clouds from a list, each with a different color.

    Args:
        point_cloud_list (List[np.ndarray]): The complete list of point clouds, where each
                                             element is a NumPy array of shape (N, 3) or (N, D>3).
        frame_indices (List[int]): A list of 0-based indices specifying which frames from
                                   point_cloud_list to visualize.
        colors (Optional[List[List[float]]]): A list of RGB colors to apply to each
                                               point cloud specified by frame_indices. If None,
                                               a default color palette is used.
        scene_name (str): A name for the scene, used in the visualization window title.
        window_size (Tuple[int, int]): The (width, height) of the visualization window.
    """
    # Define a default, visually distinct color palette
    if colors is None:
        default_colors = [
            [1.0, 0.0, 0.0],  # Red
            [0.0, 0.0, 1.0],  # Blue
            [0.0, 1.0, 0.0],  # Green
            [1.0, 1.0, 0.0],  # Yellow
            [0.0, 1.0, 1.0],  # Cyan
            [1.0, 0.0, 1.0],  # Magenta
        ]
    else:
        default_colors = colors

    geometries_to_draw = []
    valid_frame_indices = []

    for i, frame_idx in enumerate(frame_indices):
        if not (0 <= frame_idx < len(point_cloud_list)):
            print(
                f"⚠️ Warning: Index {frame_idx} is out of bounds for point_cloud_list (size: {len(point_cloud_list)}). Skipping.")
            continue

        pc_np = point_cloud_list[frame_idx]

        if pc_np is None or pc_np.size == 0:
            print(f"⚠️ Warning: Point cloud for frame index {frame_idx} is empty. Skipping.")
            continue

        if pc_np.ndim != 2 or pc_np.shape[1] < 3:
            print(
                f"⚠️ Warning: Point cloud for frame index {frame_idx} has invalid shape {pc_np.shape}. Expected (N, >=3). Skipping.")
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np[:, :3])

        color = default_colors[i % len(default_colors)]
        pcd.paint_uniform_color(color)

        geometries_to_draw.append(pcd)
        valid_frame_indices.append(frame_idx)

    # --- Visualization ---
    if geometries_to_draw:
        frame_labels = ", ".join(str(idx + 1) for idx in valid_frame_indices)  # Use 1-based for display
        window_title = f"{scene_name} - Frames: {frame_labels}"

        print(f"Visualizing frames: {frame_labels}")
        o3d.visualization.draw_geometries(
            geometries_to_draw,
            window_name=window_title,
            width=window_size[0],
            height=window_size[1]
        )
    else:
        print("❌ Error: No valid point clouds found for the selected frames. Nothing to visualize.")
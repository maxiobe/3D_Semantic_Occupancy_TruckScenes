import open3d as o3d
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from copy import deepcopy
from scipy.spatial.transform import Rotation

# Function to perform poisson surface reconstruction on a given point cloud and returns a mesh representation of the point cloud, along with vertex info
# Inputs pcd: input point cloud,
# depth: parameter to control resolution of mesh. Higher depth results in a more detailed mesh but requires more computation
# n_threads: Number of threads for parallel processing
# min_density: threshold for removing low-density vertices from generated mesh. Used to clean up noisy or sparse areas
def run_poisson(pcd, depth, n_threads, min_density=None):
    # creates triangular mesh form pcd using poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, n_threads=n_threads
    )
    # returns mesh and densities: list of density values corresponding to each vertex in the mesh. Density indicates how well a vertex is supported by underlying points

    # Post-process the mesh
    # Purpose: to clean up the mesh by removing low-density vertices (e.g. noise or poorly supported areas)
    if min_density:
        vertices_to_remove = densities < np.quantile(densities, min_density)  # min_density should be between 0 and 1
        mesh.remove_vertices_by_mask(vertices_to_remove)  # removes vertices where density is below threshold
    mesh.compute_vertex_normals()  # computes the normals of the vertices

    return mesh, densities


# Function that creates a 3D mesh from a given point cloud or a buffer of points
# Inputs buffer: a list of point clouds that are combined if no original is given
# depth: resolution for poisson surface reconstruction
# n_threads: Number of threads for parallel processing
# min_density: Optional threshold for removing low-density vertices
# point_cloud_original: provides a preprocessed point cloud, if given buffer is ignored
def create_mesh_from_map(buffer, depth, n_threads, min_density=None, point_cloud_original=None):
    if point_cloud_original is None:
        pcd = buffer_to_pointcloud(
            buffer)  # Calls buffer_to_pointcloud(buffer) to create a combined point cloud from the list of smaller point clouds
    else:
        pcd = point_cloud_original  # Uses the given point cloud directly

    return run_poisson(pcd, depth, n_threads, min_density)  # calls run_poisson function to generate mesh


# Function to combine multiple point clouds from a list (buffer) into a single point cloud and optionally estimating normals in the resulting point cloud
# Input: buffer: a list of individual point clouds (each being an instance of open3d.geometry.PointCloud)
# compute_normals: boolean flag that if set to True estimates normals of the final point cloud
def buffer_to_pointcloud(buffer, compute_normals=False):
    pcd = o3d.geometry.PointCloud()  # Initialize empty point cloud object using Open3d
    for cloud in buffer:
        pcd += cloud  # combine each point cloud with current point cloud object
    if compute_normals:
        pcd.estimate_normals()  # estimate normals for each point in the combined point cloud

    return pcd


# Function to preprocess a given point cloud by estimating and orienting the normals of each point
# Input: pcd: input point cloud
# max_nn: maximum number of nearest neighbors to use when estimating normals with default 20
# normals: boolean flag whether to estimate normals
def preprocess_cloud(
        pcd,
        max_nn=20,
        normals=None,
):
    cloud = deepcopy(
        pcd)  # create a deep copy of the input point cloud to ensure that original point cloud is not modified
    if normals:
        params = o3d.geometry.KDTreeSearchParamKNN(max_nn)
        cloud.estimate_normals(params)  # estimate normals based on nearest max_nn points
        cloud.orient_normals_towards_camera_location()  # orients all computed normals to point towards the camera location to get consistent normal direction for visualization and mesh generation

    return cloud


# Wrapper function that calls preprocess_cloud function with parameters derived from a config file
# Input: pcd: input point cloud
# config: Dictionary containing all configuration parameters loaded from a YAML file
def preprocess(pcd, config):
    return preprocess_cloud(
        pcd,
        config['max_nn'],
        normals=True
    )


def denoise_pointcloud(pcd: o3d.geometry.PointCloud, filter_mode: str, config: dict,
                       location_msg: str = "point cloud") -> o3d.geometry.PointCloud:
    """
    Applies noise filtering to the given point cloud using the specified method.

    Args:
        pcd: Open3D point cloud.
        filter_mode: One of 'none', 'sor', 'ror', 'both'.
        config: Dictionary from config.yaml with noise filtering parameters.

    Returns:
        Filtered point cloud.
    """

    initial_pcd = pcd  # Keep reference to original
    initial_count = np.asarray(initial_pcd.points).shape[0]
    kept_indices = np.arange(initial_count)  # Start with all indices
    filtered_pcd = initial_pcd

    if initial_count == 0:  # Handle empty input cloud
        print(f"Skipping filtering at '{location_msg}' on empty input.")
        return filtered_pcd, kept_indices

    if filter_mode == 'none':  # Explicitly handle 'none' case
        return filtered_pcd, kept_indices

    try:  # Add error handling for filtering operations
        if filter_mode == 'sor':
            filtered_pcd, ind = initial_pcd.remove_statistical_outlier(
                nb_neighbors=config['sor_nb_neighbors'],
                std_ratio=config['sor_std_ratio']
            )
            kept_indices = np.array(ind)
        elif filter_mode == 'ror':
            filtered_pcd, ind = initial_pcd.remove_radius_outlier(
                nb_points=config['ror_nb_points'],
                radius=config['ror_radius']
            )
            kept_indices = np.array(ind)
        elif filter_mode == 'both':
            sor_filtered_pcd, sor_ind = initial_pcd.remove_statistical_outlier(
                nb_neighbors=config['sor_nb_neighbors'],
                std_ratio=config['sor_std_ratio']
            )
            if np.asarray(sor_filtered_pcd.points).shape[0] > 0:
                filtered_pcd, ror_ind = sor_filtered_pcd.remove_radius_outlier(
                    nb_points=config['ror_nb_points'],
                    radius=config['ror_radius']
                )
                kept_indices = np.array(sor_ind)[ror_ind]
            else:
                filtered_pcd = sor_filtered_pcd
                kept_indices = np.array([], dtype=int)

        final_count = np.asarray(filtered_pcd.points).shape[0]

        print(
            f"Filtering {location_msg} with filter mode {filter_mode}. Reduced from {initial_count} to {final_count} points.")

    except Exception as e:
        print(f"Error during filtering ({filter_mode}) at {location_msg}: {e}. Returning original.")
        filtered_pcd = initial_pcd
        kept_indices = np.arange(initial_count)

    return filtered_pcd, kept_indices


def denoise_pointcloud_advanced(
        pcd: o3d.geometry.PointCloud,
        filter_mode: str,
        config: dict,
        location_msg: str = "point cloud",
        preserve_ground: bool = True
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Applies advanced noise filtering with optional ground preservation and
    correctly implemented distance-based logic.
    - For SOR: Global statistics are used with local thresholds.
    - For ROR: The cloud is split to use different radii.
    """
    initial_count = np.asarray(pcd.points).shape[0]
    if initial_count == 0 or filter_mode == 'none':
        return pcd, np.arange(initial_count)

    original_points_np = np.asarray(pcd.points)
    kept_filter_indices = np.arange(initial_count)

    final_keep_mask = np.ones(initial_count, dtype=bool)

    if config.get('distance_split', False):
        print(f"Applying distance-based filtering (mode: '{filter_mode}') at '{location_msg}'...")

        # --- 1. Create distance masks to define near and far points ---
        distances = np.linalg.norm(original_points_np, axis=1)
        near_mask = distances <= config['filter_distance_threshold']

        if filter_mode in ['sor', 'both']:
            print("Applying distance-based sor filtering...")
            # Pass 1: Aggressive settings for NEAR points
            _, ind_near = pcd.remove_statistical_outlier(
                nb_neighbors=config['sor_nb_neighbors'], std_ratio=config['sor_std_ratio']
            )
            keep_mask_near_sor = np.zeros(initial_count, dtype=bool)
            keep_mask_near_sor[ind_near] = True

            # Pass 2: Lenient settings for FAR points
            _, ind_far = pcd.remove_statistical_outlier(
                nb_neighbors=config['sor_nb_neighbors'], std_ratio=config['sor_std_ratio_far']
            )
            keep_mask_far_sor = np.zeros(initial_count, dtype=bool)
            keep_mask_far_sor[ind_far] = True

            # Combine results: Use aggressive mask for near points, lenient for far points
            final_keep_mask = np.where(near_mask, keep_mask_near_sor, keep_mask_far_sor)

        if filter_mode in ['ror', 'both']:
            print("Applying ror filtering...")
            # If 'both', filter the subset of points already kept by SOR
            # Otherwise, filter the original point cloud
            pcd_for_ror = pcd.select_by_index(np.where(final_keep_mask)[0])
            original_indices_for_ror = np.where(final_keep_mask)[0]

            distances_for_ror = distances[original_indices_for_ror]
            near_mask_for_ror = distances_for_ror <= config['filter_distance_threshold']

            # Get indices relative to the pcd_for_ror subset
            original_indices_near_ror = np.where(near_mask_for_ror)[0]
            original_indices_far_ror = np.where(~near_mask_for_ror)[0]

            # Filter subsets
            pcd_near = pcd_for_ror.select_by_index(original_indices_near_ror)
            _, kept_rel_near = pcd_near.remove_radius_outlier(nb_points=config['ror_nb_points'],
                                                              radius=config['ror_radius'])

            pcd_far = pcd_for_ror.select_by_index(original_indices_far_ror)
            _, kept_rel_far = pcd_far.remove_radius_outlier(nb_points=config['ror_nb_points'],
                                                            radius=config['ror_radius_far'])

            # Map relative indices back to the original_indices_for_ror
            kept_abs_indices_near = original_indices_for_ror[original_indices_near_ror[kept_rel_near]]
            kept_abs_indices_far = original_indices_for_ror[original_indices_far_ror[kept_rel_far]]

            # Create a new mask for the ROR pass and update the final mask
            ror_keep_mask = np.zeros(initial_count, dtype=bool)
            ror_keep_mask[kept_abs_indices_near] = True
            ror_keep_mask[kept_abs_indices_far] = True
            final_keep_mask &= ror_keep_mask

        kept_filter_indices = np.where(final_keep_mask)[0]

    else:  # Fallback to a single global filter
        # (Your original global filter logic would go here)
        print(f"Applying global filter '{filter_mode}' at '{location_msg}'...")
        if filter_mode == 'sor':
            _, kept_filter_indices = pcd.remove_statistical_outlier(nb_neighbors=config['sor_nb_neighbors'],
                                                                    std_ratio=config['sor_std_ratio'])
        elif filter_mode == 'ror':
            _, kept_filter_indices = pcd.remove_radius_outlier(nb_points=config['ror_nb_points'],
                                                               radius=config['ror_radius'])

        elif filter_mode == 'both':
            sor_filtered_pcd, sor_ind = pcd.remove_statistical_outlier(
                nb_neighbors=config['sor_nb_neighbors'],
                std_ratio=config['sor_std_ratio']
            )
            if np.asarray(sor_filtered_pcd.points).shape[0] > 0:
                filtered_pcd, ror_ind = sor_filtered_pcd.remove_radius_outlier(
                    nb_points=config['ror_nb_points'],
                    radius=config['ror_radius']
                )
                kept_filter_indices = np.array(sor_ind)[ror_ind]
            else:
                kept_filter_indices = np.array([], dtype=int)

    # --- Ground Plane Preservation ---
    if preserve_ground:
        ground_mask = get_ground_plane_mask(
            original_points_np,
            config['ground_z_min_threshold'],
            config['ground_z_max_threshold']
        )
        ground_indices = np.where(ground_mask)[0]

        num_ground_saved = np.setdiff1d(ground_indices, kept_filter_indices).size
        if num_ground_saved > 0:
            print(f"  -> Preserved {num_ground_saved} ground points that would have been filtered.")

        final_kept_indices = np.union1d(kept_filter_indices, ground_indices)
    else:
        final_kept_indices = kept_filter_indices

    # Ensure indices are integers for Open3D
    final_kept_indices = final_kept_indices.astype(int)

    filtered_pcd = pcd.select_by_index(final_kept_indices)
    final_count = np.asarray(filtered_pcd.points).shape[0]
    print(f"Filtering {location_msg}. Reduced from {initial_count} to {final_count} points.")

    return filtered_pcd, final_kept_indices


def in_range_mask(points, pc_range):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    return (
            (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
            (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
            (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )

def get_ground_plane_mask(
        point_cloud: np.ndarray,
        ground_z_min: float,
        ground_z_max: float
) -> np.ndarray:
    """
    Identifies points that are likely on the ground plane based on their Z-coordinate.

    Args:
        point_cloud (np.ndarray): The input point cloud (N, D) in the ego frame.
        ground_z_min (float): The minimum Z-value for a point to be considered ground.
                              (e.g., -2.5 meters below the sensor).
        ground_z_max (float): The maximum Z-value for a point to be considered ground.
                              (e.g., -1.0 meters below the sensor).

    Returns:
        np.ndarray: A boolean mask of shape (N,) where True indicates a likely ground point.
    """
    if point_cloud.shape[0] == 0:
        return np.array([], dtype=bool)

    z_coords = point_cloud[:, 2]
    ground_mask = (z_coords >= ground_z_min) & (z_coords <= ground_z_max)
    return ground_mask


def get_weather_intensity_filter_mask(
        point_cloud: np.ndarray,
        weather_condition: str,
        intensity_thresh: float,
        distance_thresh: float,
        keep_ground_points: bool = True,
        ground_z_min: float = -2.5,
        ground_z_max: float = -1.0
) -> np.ndarray:
    """
    Calculates a boolean mask for lidar intensity filtering in bad weather,
    with an option to always preserve likely ground points.

    Args:
        point_cloud (np.ndarray): The input point cloud (N, D), where D>=4.
        weather_condition (str): The weather condition, e.g., 'snow', 'rain'.
        intensity_thresh (float): The minimum intensity to keep for non-ground points.
        distance_thresh (float): The distance from ego to apply the intensity filter.
        keep_ground_points (bool): If True, points within the ground Z-range are always kept.
        ground_z_min (float): The minimum Z-value to consider a point as ground.
        ground_z_max (float): The maximum Z-value to consider a point as ground.

    Returns:
        np.ndarray: A boolean mask of shape (N,) where True indicates points to keep.
    """
    if weather_condition not in ['snow', 'rain', 'fog']:
        return np.ones(point_cloud.shape[0], dtype=bool)

    if point_cloud.shape[0] == 0:
        print("Empty point cloud. Returning empty mask.")
        return np.array([], dtype=bool)

    print(f"Calculating weather intensity filter for '{weather_condition}'...")

    # --- Calculations for filter logic and reporting ---
    distances_to_ego = np.linalg.norm(point_cloud[:, :3], axis=1)
    pc_lidar_intensities = point_cloud[:, 3]

    # --- Main intensity filter logic ---
    # A point is kept by the intensity filter if it's far away OR if it's close and has high intensity.
    intensity_keep_mask = (distances_to_ego > distance_thresh) | \
                          ((distances_to_ego <= distance_thresh) & (pc_lidar_intensities > intensity_thresh))

    # Find points that are far away but have low intensity. These are the ones "saved" by the distance rule.
    low_intensity_mask = pc_lidar_intensities <= intensity_thresh
    far_distance_mask = distances_to_ego > distance_thresh
    num_kept_by_distance = np.sum(low_intensity_mask & far_distance_mask)
    print(f"  - Preserved {num_kept_by_distance} low-intensity points due to being beyond the distance threshold.")

    # --- Optional ground point preservation ---
    if keep_ground_points:
        print("  - Ground point preservation is ON.")
        is_ground_mask = get_ground_plane_mask(point_cloud, ground_z_min, ground_z_max)

        # ground_points = point_cloud[is_ground_mask]
        # visualize_pointcloud(ground_points, title=f"Ground points")

        # Combine the masks: A point is kept if it passes the intensity filter OR it's a ground point.
        final_keep_mask = intensity_keep_mask | is_ground_mask

        # Report how many ground points were "saved" by this rule.
        num_ground_kept = np.sum(is_ground_mask & ~intensity_keep_mask)
        print(f"  - Preserved {num_ground_kept} low-intensity ground points.")
    else:
        print("  - Ground point preservation is OFF.")
        final_keep_mask = intensity_keep_mask

    return final_keep_mask

def integrate_imu_for_relative_motion(imu_data_prev, imu_data_curr, dt_sec):
    if dt_sec <= 0:  # Prevent division by zero or backward time
        return np.eye(4)

    avg_vx = (imu_data_prev['vx'] + imu_data_curr['vx']) / 2.0
    avg_vy = (imu_data_prev['vy'] + imu_data_curr['vy']) / 2.0
    avg_vz = (imu_data_prev['vz'] + imu_data_curr['vz']) / 2.0

    translation_vec = np.array([avg_vx * dt_sec, avg_vy * dt_sec, avg_vz * dt_sec])

    avg_roll_rate = (imu_data_prev['roll_rate'] + imu_data_curr['roll_rate']) / 2.0
    avg_pitch_rate = (imu_data_prev['pitch_rate'] + imu_data_curr['pitch_rate']) / 2.0
    avg_yaw_rate = (imu_data_prev['yaw_rate'] + imu_data_curr['yaw_rate']) / 2.0

    d_roll = avg_roll_rate * dt_sec
    d_pitch = avg_pitch_rate * dt_sec
    d_yaw = avg_yaw_rate * dt_sec

    delta_rotation_matrix = Rotation.from_euler('zyx', [d_yaw, d_pitch, d_roll]).as_matrix()

    relative_motion_matrix = np.eye(4)
    relative_motion_matrix[:3, :3] = delta_rotation_matrix
    relative_motion_matrix[:3, 3] = translation_vec

    return relative_motion_matrix
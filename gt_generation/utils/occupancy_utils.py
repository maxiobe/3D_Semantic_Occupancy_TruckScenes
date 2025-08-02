import numpy as np
from .constants import *
from tqdm import tqdm
from numba import njit, prange, cuda
import math
from truckscenes.truckscenes import TruckScenes
from pyquaternion import Quaternion
from truckscenes.utils.geometry_utils import transform_matrix, points_in_box
from typing import Any, Dict, List, Optional, Union, Tuple

def ray_casting(ray_start, ray_end, pc_range, voxel_size, spatial_shape, EPS=1e-9, DISTANCE=0.5):
    """
    3-D DDA / Amanatides–Woo ray casting.
    Returns a list of integer 3-tuples (i,j,k) of all voxels traversed by the ray.
    """
    # shift into voxel grid coords
    new_start = ray_start[:3] - pc_range[:3]
    new_end = ray_end[:3] - pc_range[:3]

    ray = new_end - new_start
    step = np.sign(ray).astype(int)
    tDelta = np.empty(3, float)
    cur_voxel = np.empty(3, int)
    last_voxel = np.empty(3, int)
    tMax = np.empty(3, float)

    # init
    for k in range(3):
        if ray[k] != 0:
            tDelta[k] = (step[k] * voxel_size[k]) / ray[k]
        else:
            tDelta[k] = np.finfo(float).max

        # nudge start/end inside to avoid boundary cases
        new_start[k] += step[k] * voxel_size[k] * EPS
        new_end[k] -= step[k] * voxel_size[k] * EPS

        cur_voxel[k] = int(np.floor(new_start[k] / voxel_size[k]))
        last_voxel[k] = int(np.floor(new_end[k] / voxel_size[k]))

    # compute initial tMax
    for k in range(3):
        if ray[k] != 0:
            # boundary coordinate
            coord = cur_voxel[k] * voxel_size[k]
            if step[k] < 0 and coord < new_start[k]:
                boundary = coord
            else:
                boundary = coord + step[k] * voxel_size[k]
            tMax[k] = (boundary - new_start[k]) / ray[k]
        else:
            tMax[k] = np.finfo(float).max

    visited = []
    # traverse until we've gone past last_voxel in any dimension
    while np.all(step * (cur_voxel - last_voxel) < DISTANCE):
        # record
        visited.append(tuple(cur_voxel.copy()))
        # step to next voxel
        # pick axis with smallest tMax
        m = np.argmin(tMax)
        cur_voxel[m] += step[m]
        if not (0 <= cur_voxel[m] < spatial_shape[m]):
            break
        tMax[m] += tDelta[m]
    return visited


def calculate_lidar_visibility(points, points_origin, points_label,
                               pc_range, voxel_size, spatial_shape, occupancy_grid, FREE_LEARNING_INDEX,
                               points_sensor_indices,
                               sensor_max_ranges):
    """
    points:        (N,3) array of LiDAR hits
    points_origin:(N,3) corresponding sensor origins
    points_label:  (N,) integer semantic labels per point
    Returns:
      voxel_state: (H,W,Z) 0=NOT_OBS,1=FREE,2=OCC
      voxel_label: (H,W,Z) semantic label (FREE_LABEL if no hit)
    """
    NOT_OBS, FREE, OCC = STATE_UNOBSERVED, STATE_FREE, STATE_OCCUPIED

    voxel_occ_count = np.zeros(spatial_shape, int)
    voxel_free_count = np.zeros(spatial_shape, int)
    voxel_label = np.full(spatial_shape, FREE_LEARNING_INDEX, int)

    # for each LiDAR point
    for i in tqdm(range(points.shape[0]), desc='Processing lidar points...'):
        start = points_origin[i]
        end = points[i]
        # direct hit voxel
        actual_hit_voxel_indices = ((end - pc_range[:3]) / voxel_size).astype(int)
        if np.all((0 <= actual_hit_voxel_indices) & (actual_hit_voxel_indices < spatial_shape)):
            voxel_occ_count[tuple(actual_hit_voxel_indices)] += 1
            voxel_label[tuple(actual_hit_voxel_indices)] = int(points_label[i])
        # walk the ray up to the point
        sensor_idx_of_point = points_sensor_indices[i]
        max_range_for_this_sensor = sensor_max_ranges[sensor_idx_of_point]
        current_distance_to_hit = np.linalg.norm(end - start)

        if current_distance_to_hit <= max_range_for_this_sensor:
            for vox_tuple in ray_casting(start, end, pc_range, voxel_size, spatial_shape):
                if np.array_equal(np.array(vox_tuple), actual_hit_voxel_indices):
                    continue

                occupancy_grid_value = occupancy_grid[vox_tuple]
                if occupancy_grid_value != FREE_LEARNING_INDEX:
                    break
                else:
                    voxel_free_count[vox_tuple] += 1

    # build state mask
    voxel_state = np.full(spatial_shape, NOT_OBS, int)
    voxel_state[voxel_free_count > 0] = FREE
    voxel_state[voxel_occ_count > 0] = OCC
    return voxel_state, voxel_label


# --- Numba CUDA Device Function for Ray Casting Steps ---
@cuda.jit(device=True)
def _ray_casting_gpu_step_logic(
        # Inputs for one ray
        ray_start_x, ray_start_y, ray_start_z,  # Origin of this ray (sensor)
        ray_end_x, ray_end_y, ray_end_z,  # Target of this ray (LiDAR hit)
        # Grid parameters
        pc_range_min_x, pc_range_min_y, pc_range_min_z,
        voxel_sx, voxel_sy, voxel_sz,  # Voxel sizes
        grid_dx, grid_dy, grid_dz,  # Grid dimensions in voxels
        # Pre-computed occupancy for early exit
        occupancy_grid_gpu,  # Read-only, shows where aggregated matter is
        FREE_LEARNING_INDEX_CONST,  # Make sure this is the correct constant name
        # Output array to update
        voxel_free_count_gpu,  # This will be updated atomically
        EPS, DISTANCE  # Constants from your ray_casting
):
    # --- Inline DDA logic from your ray_casting function ---

    new_start_x = ray_start_x - pc_range_min_x
    new_start_y = ray_start_y - pc_range_min_y
    new_start_z = ray_start_z - pc_range_min_z

    new_end_x = ray_end_x - pc_range_min_x
    new_end_y = ray_end_y - pc_range_min_y
    new_end_z = ray_end_z - pc_range_min_z

    ray_vx = new_end_x - new_start_x
    ray_vy = new_end_y - new_start_y
    ray_vz = new_end_z - new_start_z

    step_ix, step_iy, step_iz = 0, 0, 0
    if ray_vx > 0:
        step_ix = 1
    elif ray_vx < 0:
        step_ix = -1
    if ray_vy > 0:
        step_iy = 1
    elif ray_vy < 0:
        step_iy = -1
    if ray_vz > 0:
        step_iz = 1
    elif ray_vz < 0:
        step_iz = -1

    t_delta_x = float('inf')
    if ray_vx != 0: t_delta_x = (step_ix * voxel_sx) / ray_vx
    t_delta_y = float('inf')
    if ray_vy != 0: t_delta_y = (step_iy * voxel_sy) / ray_vy
    t_delta_z = float('inf')
    if ray_vz != 0: t_delta_z = (step_iz * voxel_sz) / ray_vz

    # Nudge
    adj_start_x = new_start_x + step_ix * voxel_sx * EPS
    adj_start_y = new_start_y + step_iy * voxel_sy * EPS
    adj_start_z = new_start_z + step_iz * voxel_sz * EPS

    adj_end_x = new_end_x - step_ix * voxel_sx * EPS
    adj_end_y = new_end_y - step_iy * voxel_sy * EPS
    adj_end_z = new_end_z - step_iz * voxel_sz * EPS

    cur_vox_ix = int(math.floor(adj_start_x / voxel_sx))
    cur_vox_iy = int(math.floor(adj_start_y / voxel_sy))
    cur_vox_iz = int(math.floor(adj_start_z / voxel_sz))

    last_vox_ix = int(math.floor(adj_end_x / voxel_sx))
    last_vox_iy = int(math.floor(adj_end_y / voxel_sy))
    last_vox_iz = int(math.floor(adj_end_z / voxel_sz))

    t_max_x = float('inf')
    if ray_vx != 0:
        coord_x = float(cur_vox_ix * voxel_sx)
        boundary_x = coord_x + step_ix * voxel_sx if not (step_ix < 0 and coord_x < adj_start_x) else coord_x
        t_max_x = (boundary_x - adj_start_x) / ray_vx

    t_max_y = float('inf')
    if ray_vy != 0:
        coord_y = float(cur_vox_iy * voxel_sy)
        boundary_y = coord_y + step_iy * voxel_sy if not (step_iy < 0 and coord_y < adj_start_y) else coord_y
        t_max_y = (boundary_y - adj_start_y) / ray_vy

    t_max_z = float('inf')
    if ray_vz != 0:
        coord_z = float(cur_vox_iz * voxel_sz)
        boundary_z = coord_z + step_iz * voxel_sz if not (step_iz < 0 and coord_z < adj_start_z) else coord_z
        t_max_z = (boundary_z - adj_start_z) / ray_vz

    max_iterations = grid_dx + grid_dy + grid_dz + 3  # Max iterations for safety

    for _ in range(max_iterations):
        term_x = True if step_ix == 0 else (step_ix * (cur_vox_ix - last_vox_ix) >= DISTANCE)
        term_y = True if step_iy == 0 else (step_iy * (cur_vox_iy - last_vox_iy) >= DISTANCE)
        term_z = True if step_iz == 0 else (step_iz * (cur_vox_iz - last_vox_iz) >= DISTANCE)
        if term_x and term_y and term_z:
            break

        actual_hit_vx = int(math.floor((ray_end_x - pc_range_min_x) / voxel_sx))
        actual_hit_vy = int(math.floor((ray_end_y - pc_range_min_y) / voxel_sy))
        actual_hit_vz = int(math.floor((ray_end_z - pc_range_min_z) / voxel_sz))

        is_current_voxel_the_actual_hit = (cur_vox_ix == actual_hit_vx and \
                                           cur_vox_iy == actual_hit_vy and \
                                           cur_vox_iz == actual_hit_vz)

        if not is_current_voxel_the_actual_hit:
            if (0 <= cur_vox_ix < grid_dx and
                    0 <= cur_vox_iy < grid_dy and
                    0 <= cur_vox_iz < grid_dz):

                if occupancy_grid_gpu[cur_vox_ix, cur_vox_iy, cur_vox_iz] != FREE_LEARNING_INDEX_CONST:
                    return  # Ray hit an obstruction
                else:
                    cuda.atomic.add(voxel_free_count_gpu, (cur_vox_ix, cur_vox_iy, cur_vox_iz), 1)
            else:  # Current voxel out of bounds
                return

        # Step to next voxel
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                cur_vox_ix += step_ix
                if not (0 <= cur_vox_ix < grid_dx): return
                t_max_x += t_delta_x
            else:
                cur_vox_iz += step_iz
                if not (0 <= cur_vox_iz < grid_dz): return
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                cur_vox_iy += step_iy
                if not (0 <= cur_vox_iy < grid_dy): return
                t_max_y += t_delta_y
            else:
                cur_vox_iz += step_iz
                if not (0 <= cur_vox_iz < grid_dz): return
                t_max_z += t_delta_z


# --- Main Numba CUDA Kernel ---
@cuda.jit
def visibility_kernel(
        points_gpu,  # (N,3) LiDAR hits
        points_origin_gpu,  # (N,3) Sensor origins
        points_label_gpu,  # (N,) Semantic labels for hits
        pc_range_min_gpu,  # (3,) [xmin, ymin, zmin] - expecting np.array
        voxel_size_gpu,  # (3,) [vx, vy, vz] - expecting np.array
        spatial_shape_gpu,  # (3,) [Dx, Dy, Dz] - expecting np.array (int32)
        occupancy_grid_gpu,  # (Dx,Dy,Dz) Pre-computed, read-only (uint8)
        FREE_LEARNING_INDEX_CONST_UINT8,  # Scalar (uint8)
        voxel_occ_count_out_gpu,  # (Dx,Dy,Dz) for writing (int32)
        voxel_free_count_out_gpu,  # (Dx,Dy,Dz) for writing (int32)
        voxel_label_out_gpu,  # (Dx,Dy,Dz) for writing (int32)
        FREE_LABEL_CONST_FOR_INIT_INT32,  # Scalar (int32, e.g. -1)
        EPS_CONST, DISTANCE_CONST,  # Scalars (float64)
        points_sensor_indices_gpu,
        sensor_max_ranges_gpu
):
    i = cuda.grid(1)
    if i >= points_gpu.shape[0]:
        return

    ray_start_x = points_origin_gpu[i, 0]
    ray_start_y = points_origin_gpu[i, 1]
    ray_start_z = points_origin_gpu[i, 2]

    ray_end_x = points_gpu[i, 0]
    ray_end_y = points_gpu[i, 1]
    ray_end_z = points_gpu[i, 2]

    point_label = points_label_gpu[i]  # This is int32

    # Grid parameters from input arrays
    pc_min_x, pc_min_y, pc_min_z = pc_range_min_gpu[0], pc_range_min_gpu[1], pc_range_min_gpu[2]
    voxel_sx, voxel_sy, voxel_sz = voxel_size_gpu[0], voxel_size_gpu[1], voxel_size_gpu[2]
    grid_dx, grid_dy, grid_dz = spatial_shape_gpu[0], spatial_shape_gpu[1], spatial_shape_gpu[2]

    # --- 1. Mark Occupied Voxel (for the actual LiDAR hit 'ray_end') ---
    actual_hit_vx = int(math.floor((ray_end_x - pc_min_x) / voxel_sx))
    actual_hit_vy = int(math.floor((ray_end_y - pc_min_y) / voxel_sy))
    actual_hit_vz = int(math.floor((ray_end_z - pc_min_z) / voxel_sz))

    if (0 <= actual_hit_vx < grid_dx and
            0 <= actual_hit_vy < grid_dy and
            0 <= actual_hit_vz < grid_dz):
        cuda.atomic.add(voxel_occ_count_out_gpu, (actual_hit_vx, actual_hit_vy, actual_hit_vz), 1)
        voxel_label_out_gpu[actual_hit_vx, actual_hit_vy, actual_hit_vz] = point_label  # Direct write (last wins)

    sensor_idx_of_point = points_sensor_indices_gpu[i]
    max_range_for_this_sensor = sensor_max_ranges_gpu[sensor_idx_of_point]

    # Calculate squared distance to avoid sqrt in kernel if possible, or just use norm if math.sqrt is acceptable
    dx = ray_end_x - ray_start_x
    dy = ray_end_y - ray_start_y
    dz = ray_end_z - ray_start_z
    distance_sq_to_hit = dx * dx + dy * dy + dz * dz

    # --- 2. Perform Ray Casting ---
    if distance_sq_to_hit <= max_range_for_this_sensor * max_range_for_this_sensor:
        _ray_casting_gpu_step_logic(
            ray_start_x, ray_start_y, ray_start_z,
            ray_end_x, ray_end_y, ray_end_z,
            pc_min_x, pc_min_y, pc_min_z,
            voxel_sx, voxel_sy, voxel_sz,
            grid_dx, grid_dy, grid_dz,
            occupancy_grid_gpu,
            FREE_LEARNING_INDEX_CONST_UINT8,  # Pass the constant for comparison
            voxel_free_count_out_gpu,
            EPS_CONST, DISTANCE_CONST
        )


# --- Host Function to Manage GPU Execution ---
def calculate_lidar_visibility_gpu_host(
        points_cpu, points_origin_cpu, points_label_cpu,
        pc_range_cpu_list,  # Original list [xmin,ymin,zmin,xmax,ymax,zmax]
        voxel_size_cpu_scalar,  # Original scalar voxel size
        spatial_shape_cpu_list,  # Original list [Dx,Dy,Dz]
        occupancy_grid_cpu,  # (Dx,Dy,Dz) np.uint8
        FREE_LEARNING_INDEX_cpu,  # scalar int/uint8 for free space semantic label
        FREE_LABEL_placeholder_cpu,  # scalar int (e.g., -1 for internal init)
        points_sensor_indices_cpu: np.ndarray,
        sensor_max_ranges_cpu: np.ndarray
):
    num_points = points_cpu.shape[0]
    if num_points == 0:
        voxel_state = np.full(tuple(spatial_shape_cpu_list), STATE_UNOBSERVED, dtype=np.uint8)
        voxel_label = np.full(tuple(spatial_shape_cpu_list), FREE_LEARNING_INDEX_cpu, dtype=np.uint8)
        return voxel_state, voxel_label

    # Prepare data for GPU (ensure contiguous and correct types)
    points_gpu_data = cuda.to_device(np.ascontiguousarray(points_cpu, dtype=np.float64))
    points_origin_gpu_data = cuda.to_device(np.ascontiguousarray(points_origin_cpu, dtype=np.float64))
    points_label_gpu_data = cuda.to_device(np.ascontiguousarray(points_label_cpu, dtype=np.int32))

    pc_range_min_gpu_data = cuda.to_device(np.ascontiguousarray(pc_range_cpu_list[:3], dtype=np.float64))
    voxel_size_gpu_data = cuda.to_device(np.array([voxel_size_cpu_scalar] * 3, dtype=np.float64))
    spatial_shape_gpu_data = cuda.to_device(np.array(spatial_shape_cpu_list, dtype=np.int32))

    occupancy_grid_gpu_data = cuda.to_device(np.ascontiguousarray(occupancy_grid_cpu, dtype=np.uint8))

    points_sensor_indices_gpu_data = cuda.to_device(
        np.ascontiguousarray(points_sensor_indices_cpu, dtype=np.int32))
    sensor_max_ranges_gpu_data = cuda.to_device(np.ascontiguousarray(sensor_max_ranges_cpu, dtype=np.float32))

    # Output arrays on GPU
    voxel_occ_count_gpu = cuda.to_device(np.zeros(tuple(spatial_shape_cpu_list), dtype=np.int32))
    voxel_free_count_gpu = cuda.to_device(np.zeros(tuple(spatial_shape_cpu_list), dtype=np.int32))
    voxel_label_out_gpu = cuda.to_device(
        np.full(tuple(spatial_shape_cpu_list), np.int32(FREE_LABEL_placeholder_cpu), dtype=np.int32))

    # Kernel launch configuration
    threads_per_block = 256
    blocks_per_grid = (num_points + (threads_per_block - 1)) // threads_per_block

    EPS_CONST_val = 1e-9  # Standard DDA constant
    DISTANCE_CONST_val = 0.5  # Standard DDA constant

    print(f"Launching GPU kernel: {blocks_per_grid} blocks, {threads_per_block} threads/block for {num_points} points.")
    visibility_kernel[blocks_per_grid, threads_per_block](
        points_gpu_data, points_origin_gpu_data, points_label_gpu_data,
        pc_range_min_gpu_data, voxel_size_gpu_data, spatial_shape_gpu_data,
        occupancy_grid_gpu_data,
        np.uint8(FREE_LEARNING_INDEX_cpu),  # Pass as uint8 for comparison with occupancy_grid_gpu
        voxel_occ_count_gpu, voxel_free_count_gpu, voxel_label_out_gpu,
        np.int32(FREE_LABEL_placeholder_cpu),  # For initializing voxel_label_out_gpu
        EPS_CONST_val, DISTANCE_CONST_val,
        points_sensor_indices_gpu_data,
        sensor_max_ranges_gpu_data
    )
    cuda.synchronize()

    # Copy results back to CPU
    voxel_occ_count_cpu = voxel_occ_count_gpu.copy_to_host()
    voxel_free_count_cpu = voxel_free_count_gpu.copy_to_host()
    voxel_label_from_gpu_cpu = voxel_label_out_gpu.copy_to_host()  # This is int32

    # Final state assignment (on CPU)
    final_voxel_states = np.full(tuple(spatial_shape_cpu_list), STATE_UNOBSERVED, dtype=np.uint8)
    final_voxel_states[voxel_free_count_cpu > 0] = STATE_FREE
    final_voxel_states[voxel_occ_count_cpu > 0] = STATE_OCCUPIED

    # Create final semantic labels grid
    final_semantic_labels = np.full(tuple(spatial_shape_cpu_list), FREE_LEARNING_INDEX_cpu, dtype=np.uint8)
    # Populate labels for occupied voxels
    occupied_mask = (final_voxel_states == STATE_OCCUPIED)
    # voxel_label_from_gpu_cpu contains actual semantic labels for occupied cells,
    # and FREE_LABEL_placeholder_cpu for others.
    final_semantic_labels[occupied_mask] = voxel_label_from_gpu_cpu[occupied_mask].astype(np.uint8)

    print("GPU visibility calculation finished.")
    return final_voxel_states, final_semantic_labels


# --- Helper function to get camera parameters ---
def get_camera_parameters(trucksc: TruckScenes, sample_data_token: str, ego_pose_timestamp: int):
    """
    Retrieves and transforms camera parameters to the current ego vehicle frame.
    """
    sd = trucksc.get('sample_data', sample_data_token)
    cs = trucksc.get('calibrated_sensor', sd['calibrated_sensor_token'])

    # Camera intrinsics (K)
    cam_intrinsics = np.array(cs['camera_intrinsic'])

    # Transformation: camera frame -> ego vehicle frame AT THE TIMESTAMP OF THE CAMERA IMAGE
    # This is P_cam2ego in the paper's notation, specific to this camera's capture time.
    cam_extrinsic_translation = np.array(cs['translation'])
    cam_extrinsic_rotation = Quaternion(cs['rotation'])
    T_ego_at_cam_timestamp_from_cam = transform_matrix(
        cam_extrinsic_translation, cam_extrinsic_rotation, inverse=False
    )

    # If the camera's timestamp differs from the reference ego_pose_timestamp,
    # we need to bring the camera pose into the *current* ego frame.
    # Current ego pose (at ego_pose_timestamp, e.g., LiDAR keyframe time)
    current_ego_pose_rec = trucksc.getclosest('ego_pose', ego_pose_timestamp)
    T_global_from_current_ego = transform_matrix(
        current_ego_pose_rec['translation'], Quaternion(current_ego_pose_rec['rotation']), inverse=False
    )
    T_current_ego_from_global = np.linalg.inv(T_global_from_current_ego)

    # Ego pose at the camera's capture time
    cam_timestamp_ego_pose_rec = trucksc.getclosest('ego_pose', sd['timestamp'])
    T_global_from_ego_at_cam_timestamp = transform_matrix(
        cam_timestamp_ego_pose_rec['translation'], Quaternion(cam_timestamp_ego_pose_rec['rotation']), inverse=False
    )

    # Final transformation: camera frame -> current_ego_frame
    # P_current_ego_from_cam = P_current_ego_from_global @ P_global_from_ego_at_cam_timestamp @ P_ego_at_cam_timestamp_from_cam
    T_current_ego_from_cam = T_current_ego_from_global @ T_global_from_ego_at_cam_timestamp @ T_ego_at_cam_timestamp_from_cam

    # Camera origin in the current ego frame
    cam_origin_in_current_ego = T_current_ego_from_cam[:3, 3]

    # Rotation part for transforming ray directions
    R_current_ego_from_cam = T_current_ego_from_cam[:3, :3]

    return {
        'intrinsics': cam_intrinsics,  # 3x3 K matrix
        'T_current_ego_from_cam': T_current_ego_from_cam,  # 4x4 matrix
        'origin_in_current_ego': cam_origin_in_current_ego,  # (3,)
        'R_current_ego_from_cam': R_current_ego_from_cam,  # (3,3)
        'width': sd['width'],
        'height': sd['height']
    }


# --- CPU Function for Camera Visibility (Algorithm 3 from Occ3D) ---
def calculate_camera_visibility_cpu(
        # Inputs based on Algorithm 3 and practical needs
        trucksc: TruckScenes,
        current_sample_token: str,  # To get camera data for the current keyframe
        lidar_voxel_state: np.ndarray,  # (Dx,Dy,Dz) - output from LiDAR visibility (0=UNOBS, 1=FREE, 2=OCC)
        pc_range_params: list,  # [xmin,ymin,zmin,xmax,ymax,zmax]
        voxel_size_params: np.ndarray,  # [vx,vy,vz]
        spatial_shape_params: np.ndarray,  # [Dx,Dy,Dz]
        camera_names: List[str],  # List of camera sensor names to use
        DEPTH_MAX: float = 100.0
):
    print("Calculating Camera Visibility (CPU)...")

    # Output camera visibility mask: 1 if observed by any camera (and LiDAR), 0 otherwise
    # Initialize to 0 (unobserved by camera)
    camera_visibility_mask = np.zeros(spatial_shape_params, dtype=np.uint8)

    # Get timestamp of the current sample (e.g., the keyframe for which we do this)
    current_sample_rec = trucksc.get('sample', current_sample_token)
    current_ego_pose_ts = current_sample_rec['timestamp']

    # Iterate over each camera specified
    for cam_name in tqdm(camera_names, desc="Processing Cameras"):
        if cam_name not in current_sample_rec['data']:
            print(f"Warning: Camera {cam_name} not found in sample data for token {current_sample_token}. Skipping.")
            continue

        cam_sample_data_token = current_sample_rec['data'][cam_name]
        cam_params = get_camera_parameters(trucksc, cam_sample_data_token, current_ego_pose_ts)

        K = cam_params['intrinsics']  # 3x3
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        cam_origin_ego = cam_params['origin_in_current_ego']  # (3,)
        R_ego_from_cam = cam_params['R_current_ego_from_cam']  # (3,3)

        # Create a meshgrid of pixel coordinates
        u_coords = np.arange(cam_params['width'])
        v_coords = np.arange(cam_params['height'])
        uu, vv = np.meshgrid(u_coords, v_coords)  # vv: (H,W), uu: (H,W)

        # Unproject pixels to get ray directions in camera frame (at Z_cam=1)
        x_cam_norm = (uu - cx) / fx
        y_cam_norm = (vv - cy) / fy
        # Ray directions in camera frame (Z_cam=1 plane)
        # Shape: (H, W, 3)
        ray_dirs_cam = np.stack([x_cam_norm, y_cam_norm, np.ones_like(x_cam_norm)], axis=-1)

        # Transform ray directions to ego frame
        # Reshape for batch matrix multiplication: (H*W, 3)
        ray_dirs_cam_flat = ray_dirs_cam.reshape(-1, 3)
        ray_dirs_ego_flat = (R_ego_from_cam @ ray_dirs_cam_flat.T).T  # (H*W, 3)

        # Normalize ray directions in ego frame
        ray_dirs_ego_flat_norm = ray_dirs_ego_flat / (np.linalg.norm(ray_dirs_ego_flat, axis=1, keepdims=True) + 1e-9)

        # Define ray endpoints (far points)
        far_points_ego = cam_origin_ego + ray_dirs_ego_flat_norm * DEPTH_MAX

        # Iterate through each pixel ray for this camera
        num_pixel_rays = far_points_ego.shape[0]
        for ray_idx in tqdm(range(num_pixel_rays), desc=f" Rays for {cam_name}", leave=False):

            ray_start_ego = cam_origin_ego
            ray_end_ego = far_points_ego[ray_idx]

            # Use your existing ray_casting function
            # It expects (ray_hit_point, ray_sensor_origin, ...)
            # Here, ray_start_ego is the sensor origin, ray_end_ego is the far point
            for vox_tuple in ray_casting(
                    ray_start=ray_start_ego,  # Physical start of ray
                    ray_end=ray_end_ego,  # Physical end of ray (far point)
                    pc_range=pc_range_params,
                    voxel_size=voxel_size_params,
                    spatial_shape=spatial_shape_params
            ):
                # vox_tuple is (vx, vy, vz)
                # Check bounds (ray_casting should handle this, but an extra check is safe)
                if not (0 <= vox_tuple[0] < spatial_shape_params[0] and
                        0 <= vox_tuple[1] < spatial_shape_params[1] and
                        0 <= vox_tuple[2] < spatial_shape_params[2]):
                    continue

                lidar_state_at_vox = lidar_voxel_state[vox_tuple]

                if lidar_state_at_vox == STATE_OCCUPIED:
                    camera_visibility_mask[vox_tuple] = STATE_OCCUPIED  # Observed by camera, was occupied by LiDAR
                    break  # Ray is blocked by a LiDAR-occupied voxel
                elif lidar_state_at_vox == STATE_FREE:
                    camera_visibility_mask[vox_tuple] = STATE_FREE  # Observed by camera, was free by LiDAR
                    # Ray continues through free space
                else:
                    camera_visibility_mask[vox_tuple] = STATE_UNOBSERVED

    print("Finished Camera Visibility (CPU).")
    return camera_visibility_mask


# --- Numba CUDA Device Function for Camera Ray Traversal & Mask Update ---
@cuda.jit(device=True)
def _camera_ray_trace_and_update_mask_device(
        # Ray properties
        ray_start_x, ray_start_y, ray_start_z,  # Camera origin in ego frame
        ray_end_x, ray_end_y, ray_end_z,  # Far point for this pixel ray in ego frame
        # Grid parameters
        pc_range_min_x, pc_range_min_y, pc_range_min_z,
        voxel_sx, voxel_sy, voxel_sz,
        grid_dx, grid_dy, grid_dz,
        # Input LiDAR visibility state
        lidar_voxel_state_gpu,  # (Dx,Dy,Dz) uint8, read-only
        # Output camera visibility mask to update
        camera_visibility_mask_gpu,  # (Dx,Dy,Dz) uint8, for writing
        # Constants
        STATE_OCCUPIED_CONST, STATE_FREE_CONST, STATE_UNOBSERVED_CONST,
        EPS, DISTANCE
):
    # --- Inline DDA logic (adapted from your ray_casting) ---
    new_start_x = ray_start_x - pc_range_min_x
    new_start_y = ray_start_y - pc_range_min_y
    new_start_z = ray_start_z - pc_range_min_z

    new_end_x = ray_end_x - pc_range_min_x
    new_end_y = ray_end_y - pc_range_min_y
    new_end_z = ray_end_z - pc_range_min_z

    ray_vx = new_end_x - new_start_x
    ray_vy = new_end_y - new_start_y
    ray_vz = new_end_z - new_start_z

    step_ix, step_iy, step_iz = 0, 0, 0
    if ray_vx > 0:
        step_ix = 1
    elif ray_vx < 0:
        step_ix = -1
    if ray_vy > 0:
        step_iy = 1
    elif ray_vy < 0:
        step_iy = -1
    if ray_vz > 0:
        step_iz = 1
    elif ray_vz < 0:
        step_iz = -1

    t_delta_x = float('inf')
    if ray_vx != 0: t_delta_x = (step_ix * voxel_sx) / ray_vx
    t_delta_y = float('inf')
    if ray_vy != 0: t_delta_y = (step_iy * voxel_sy) / ray_vy
    t_delta_z = float('inf')
    if ray_vz != 0: t_delta_z = (step_iz * voxel_sz) / ray_vz

    adj_start_x = new_start_x + step_ix * voxel_sx * EPS
    adj_start_y = new_start_y + step_iy * voxel_sy * EPS
    adj_start_z = new_start_z + step_iz * voxel_sz * EPS

    # For camera rays, the 'last_voxel' is effectively the one at DEPTH_MAX
    # The loop should continue as long as we are within bounds and haven't hit an occluder
    cur_vox_ix = int(math.floor(adj_start_x / voxel_sx))
    cur_vox_iy = int(math.floor(adj_start_y / voxel_sy))
    cur_vox_iz = int(math.floor(adj_start_z / voxel_sz))

    # No explicit last_voxel needed for termination if we check bounds and DEPTH_MAX (implicitly by ray_end)
    # The DDA termination based on DISTANCE to last_voxel is less relevant here;
    # we trace until occlusion or max depth (implicitly handled by ray_end_x/y/z) or out of bounds.

    t_max_x = float('inf')
    if ray_vx != 0:
        coord_x = float(cur_vox_ix * voxel_sx)
        boundary_x = coord_x + step_ix * voxel_sx if not (step_ix < 0 and coord_x < adj_start_x) else coord_x
        t_max_x = (boundary_x - adj_start_x) / ray_vx

    t_max_y = float('inf')
    if ray_vy != 0:
        coord_y = float(cur_vox_iy * voxel_sy)
        boundary_y = coord_y + step_iy * voxel_sy if not (step_iy < 0 and coord_y < adj_start_y) else coord_y
        t_max_y = (boundary_y - adj_start_y) / ray_vy

    t_max_z = float('inf')
    if ray_vz != 0:
        coord_z = float(cur_vox_iz * voxel_sz)
        boundary_z = coord_z + step_iz * voxel_sz if not (step_iz < 0 and coord_z < adj_start_z) else coord_z
        t_max_z = (boundary_z - adj_start_z) / ray_vz

    max_iterations = grid_dx + grid_dy + grid_dz + 3  # Safety break

    for _ in range(max_iterations):
        # Check if current voxel is within grid bounds
        if not (0 <= cur_vox_ix < grid_dx and \
                0 <= cur_vox_iy < grid_dy and \
                0 <= cur_vox_iz < grid_dz):
            return  # Ray went out of bounds

        # Check LiDAR state at this voxel
        lidar_state_at_vox = lidar_voxel_state_gpu[cur_vox_ix, cur_vox_iy, cur_vox_iz]

        if lidar_state_at_vox == STATE_OCCUPIED_CONST:
            # Write STATE_OCCUPIED to the camera mask
            camera_visibility_mask_gpu[cur_vox_ix, cur_vox_iy, cur_vox_iz] = STATE_OCCUPIED_CONST
            return  # Ray is blocked
        elif lidar_state_at_vox == STATE_FREE_CONST:
            camera_visibility_mask_gpu[cur_vox_ix, cur_vox_iy, cur_vox_iz] = STATE_FREE_CONST
            # Ray continues through this free voxel
        elif lidar_state_at_vox == STATE_UNOBSERVED_CONST:
            # Voxel is unobserved by LiDAR. For Occ3D compatibility, this means
            # it's not considered "observed" in the joint camera-LiDAR sense.
            # The camera ray itself might physically continue, but we stop marking
            # voxels as camera-visible along this path if it enters LiDAR-unobserved space.
            camera_visibility_mask_gpu[cur_vox_ix, cur_vox_iy, cur_vox_iz] = STATE_UNOBSERVED_CONST
            return  # Stop considering this ray for camera visibility updates

        # Termination condition: check if we've effectively reached the ray_end
        # This is a bit tricky with DDA. The loop usually stops when out of bounds
        # or after a certain number of steps if ray_end is far.
        # The DISTANCE check from original ray_casting might be adapted.
        # For simplicity, we rely on max_iterations or going out of bounds if DEPTH_MAX is large.
        # Or, if cur_voxel is the voxel containing ray_end_x,y,z, we can stop.
        end_vox_ix = int(math.floor((ray_end_x - pc_range_min_x) / voxel_sx))
        end_vox_iy = int(math.floor((ray_end_y - pc_range_min_y) / voxel_sy))
        end_vox_iz = int(math.floor((ray_end_z - pc_range_min_z) / voxel_sz))
        if cur_vox_ix == end_vox_ix and cur_vox_iy == end_vox_iy and cur_vox_iz == end_vox_iz:
            return  # Reached the far point of the ray

        # Step to next voxel
        if t_max_x < t_max_y:
            if t_max_x < t_max_z:
                cur_vox_ix += step_ix
                t_max_x += t_delta_x
            else:
                cur_vox_iz += step_iz
                t_max_z += t_delta_z
        else:
            if t_max_y < t_max_z:
                cur_vox_iy += step_iy
                t_max_y += t_delta_y
            else:
                cur_vox_iz += step_iz
                t_max_z += t_delta_z


# --- CUDA Kernel for a single camera's visibility ---
@cuda.jit
def camera_visibility_kernel_per_cam(
        # Ray origins and directions for this camera
        cam_origin_ego_gpu,  # (3,) XYZ of camera in ego frame
        pixel_ray_dirs_ego_gpu,  # (Num_pixels, 3) Normalized ray directions in ego frame
        # LiDAR visibility and grid parameters (read-only)
        lidar_voxel_state_gpu,  # (Dx,Dy,Dz) uint8
        pc_range_min_gpu,  # (3,)
        voxel_size_gpu,  # (3,)
        spatial_shape_gpu,  # (3,) int32
        # Output
        camera_visibility_mask_gpu,  # (Dx,Dy,Dz) uint8, for writing
        # Constants
        DEPTH_MAX_CONST,
        STATE_OCCUPIED_CONST, STATE_FREE_CONST, STATE_UNOBSERVED_CONST,
        EPS_CONST, DISTANCE_CONST
):
    pixel_idx = cuda.grid(1)  # Global index for the current pixel ray
    if pixel_idx >= pixel_ray_dirs_ego_gpu.shape[0]:
        return

    # Ray start is the camera origin (same for all threads in this launch)
    ray_start_x = cam_origin_ego_gpu[0]
    ray_start_y = cam_origin_ego_gpu[1]
    ray_start_z = cam_origin_ego_gpu[2]

    # Ray direction for this specific pixel
    dir_x = pixel_ray_dirs_ego_gpu[pixel_idx, 0]
    dir_y = pixel_ray_dirs_ego_gpu[pixel_idx, 1]
    dir_z = pixel_ray_dirs_ego_gpu[pixel_idx, 2]

    # Calculate far end-point of the ray
    ray_end_x = ray_start_x + dir_x * DEPTH_MAX_CONST
    ray_end_y = ray_start_y + dir_y * DEPTH_MAX_CONST
    ray_end_z = ray_start_z + dir_z * DEPTH_MAX_CONST

    # Grid parameters for device function
    pc_min_x, pc_min_y, pc_min_z = pc_range_min_gpu[0], pc_range_min_gpu[1], pc_range_min_gpu[2]
    voxel_sx, voxel_sy, voxel_sz = voxel_size_gpu[0], voxel_size_gpu[1], voxel_size_gpu[2]
    grid_dx, grid_dy, grid_dz = spatial_shape_gpu[0], spatial_shape_gpu[1], spatial_shape_gpu[2]

    _camera_ray_trace_and_update_mask_device(
        ray_start_x, ray_start_y, ray_start_z,
        ray_end_x, ray_end_y, ray_end_z,
        pc_min_x, pc_min_y, pc_min_z,
        voxel_sx, voxel_sy, voxel_sz,
        grid_dx, grid_dy, grid_dz,
        lidar_voxel_state_gpu,
        camera_visibility_mask_gpu,  # This is where updates happen
        STATE_OCCUPIED_CONST, STATE_FREE_CONST, STATE_UNOBSERVED_CONST,  # Pass constants
        EPS_CONST, DISTANCE_CONST
    )


# --- Host Function to Manage Camera Visibility GPU Execution ---
def calculate_camera_visibility_gpu_host(
        trucksc: TruckScenes,
        current_sample_token: str,
        lidar_voxel_state_cpu: np.ndarray,  # (Dx,Dy,Dz) uint8
        pc_range_cpu_list: list,
        voxel_size_cpu_scalar: float,
        spatial_shape_cpu_list: list,  # [Dx,Dy,Dz]
        camera_names: List[str],
        DEPTH_MAX_val: float = 100.0
):
    print("Calculating Camera Visibility (GPU)...")

    spatial_shape_tuple = tuple(spatial_shape_cpu_list)

    # Transfer common data to GPU once
    lidar_voxel_state_gpu = cuda.to_device(np.ascontiguousarray(lidar_voxel_state_cpu, dtype=np.uint8))
    pc_range_min_gpu = cuda.to_device(np.ascontiguousarray(pc_range_cpu_list[:3], dtype=np.float64))
    voxel_size_gpu = cuda.to_device(np.array([voxel_size_cpu_scalar] * 3, dtype=np.float64))
    spatial_shape_gpu_dims = cuda.to_device(np.array(spatial_shape_cpu_list, dtype=np.int32))

    # Output mask on GPU, initialized to 0
    camera_visibility_mask_gpu = cuda.to_device(
        np.full(spatial_shape_tuple, STATE_UNOBSERVED, dtype=np.uint8)  # Explicit initialization
    )

    current_sample_rec = trucksc.get('sample', current_sample_token)
    current_ego_pose_ts = current_sample_rec['timestamp']

    EPS_CONST_val = 1e-9
    DISTANCE_CONST_val = 0.5  # From your CPU ray_casting, might not be strictly needed for camera version's termination

    for cam_name in tqdm(camera_names, desc="GPU Processing Cameras"):
        if cam_name not in current_sample_rec['data']:
            print(f"Warning: Camera {cam_name} not found in sample data for token {current_sample_token}. Skipping.")
            continue

        cam_sample_data_token = current_sample_rec['data'][cam_name]
        cam_params = get_camera_parameters(trucksc, cam_sample_data_token, current_ego_pose_ts)

        K = cam_params['intrinsics']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        cam_origin_ego_cpu = np.ascontiguousarray(cam_params['origin_in_current_ego'], dtype=np.float64)
        R_ego_from_cam_cpu = np.ascontiguousarray(cam_params['R_current_ego_from_cam'], dtype=np.float64)

        u_coords = np.arange(cam_params['width'])
        v_coords = np.arange(cam_params['height'])
        uu, vv = np.meshgrid(u_coords, v_coords)

        x_cam_norm = (uu.astype(np.float64) - cx) / fx
        y_cam_norm = (vv.astype(np.float64) - cy) / fy
        ray_dirs_cam = np.stack([x_cam_norm, y_cam_norm, np.ones_like(x_cam_norm)], axis=-1)

        ray_dirs_cam_flat = np.ascontiguousarray(ray_dirs_cam.reshape(-1, 3))
        ray_dirs_ego_flat = (R_ego_from_cam_cpu @ ray_dirs_cam_flat.T).T

        ray_dirs_ego_flat_norm = ray_dirs_ego_flat / (np.linalg.norm(ray_dirs_ego_flat, axis=1, keepdims=True) + 1e-9)
        ray_dirs_ego_flat_norm = np.ascontiguousarray(ray_dirs_ego_flat_norm, dtype=np.float64)

        # Transfer per-camera data
        cam_origin_ego_gpu_current = cuda.to_device(cam_origin_ego_cpu)
        pixel_ray_dirs_ego_gpu_current = cuda.to_device(ray_dirs_ego_flat_norm)

        num_pixel_rays = pixel_ray_dirs_ego_gpu_current.shape[0]
        if num_pixel_rays == 0:
            continue

        threads_per_block_cam = 256
        blocks_per_grid_cam = (num_pixel_rays + (threads_per_block_cam - 1)) // threads_per_block_cam

        # print(f"  Launching Camera Kernel for {cam_name}: {blocks_per_grid_cam} blocks, {threads_per_block_cam} threads")
        camera_visibility_kernel_per_cam[blocks_per_grid_cam, threads_per_block_cam](
            cam_origin_ego_gpu_current,
            pixel_ray_dirs_ego_gpu_current,
            lidar_voxel_state_gpu,  # Already on GPU
            pc_range_min_gpu,  # Already on GPU
            voxel_size_gpu,  # Already on GPU
            spatial_shape_gpu_dims,  # Already on GPU
            camera_visibility_mask_gpu,  # Output, already on GPU
            DEPTH_MAX_val,
            STATE_OCCUPIED, STATE_FREE, STATE_UNOBSERVED,  # Pass constants
            EPS_CONST_val, DISTANCE_CONST_val
        )
        cuda.synchronize()  # Wait for this camera's kernel to finish

    # Copy final camera visibility mask back to CPU
    camera_visibility_mask_cpu = camera_visibility_mask_gpu.copy_to_host()

    print("Finished Camera Visibility (GPU).")
    return camera_visibility_mask_cpu
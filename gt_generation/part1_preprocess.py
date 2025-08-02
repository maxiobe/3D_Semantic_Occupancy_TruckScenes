import numpy as np
import os
import torch
import time
import yaml
import os.path as osp
from truckscenes.utils import splits
from truckscenes.truckscenes import TruckScenes
from utils.pointcloud_loading import load_lidar_entries, group_entries, get_rigid_fused_pointcloud, get_pointwise_fused_pointcloud
from utils.bbox_utils import transform_boxes_to_ego
import open3d as o3d
from mmcv.ops.points_in_boxes import (points_in_boxes_cpu)
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from truckscenes.utils.geometry_utils import transform_matrix
from utils.data_utils import parse_single_annotation_file
from utils.visualization import visualize_pointcloud_bbox, visualize_pointcloud
from utils.pointcloud_processing import denoise_pointcloud, denoise_pointcloud_advanced, get_weather_intensity_filter_mask, integrate_imu_for_relative_motion
from utils.geometry_utils import transform_points
from utils.constants import *
from kiss_icp.pipeline import OdometryPipeline
from mapmos.pipeline import MapMOSPipeline
from utils.custom_datasets import InMemoryDataset, InMemoryDatasetMapMOS
from utils.data_utils import save_gnss_to_directory, save_poses_to_kitti_format, save_pcds_to_directory

def main(trucksc, indice, truckscenesyaml, args, config):
    ########################## Extract parameters from config and args #############################################
    save_path = args.save_path  # Directory where processed data will be saved
    data_root = args.dataroot  # Root directory of dataset
    learning_map = truckscenesyaml['learning_map']  # dictionary that maps raw semantic labels to learning labels
    voxel_size = config['voxel_size']  # Size of each voxel in the occupancy grid
    pc_range = config['pc_range']  # Range of point cloud coordinates to consider
    occ_size = config['occ_size']  # Dimensions of the output occupancy grid
    load_mode = args.load_mode  # Load mode of point clouds with or wihtout deskewing
    self_range = config[
        'self_range']  # Range threshold for the vehicle's own points
    x_min_self, y_min_self, z_min_self, x_max_self, y_max_self, z_max_self = self_range

    intensity_threshold = config['intensity_threshold']  # Threshold for lidar intensity filtering
    distance_intensity_threshold = config['distance_intensity_threshold']  # Distance for lidar intensity filtering
    ground_z_min_threshold = config['ground_z_min_threshold']
    ground_z_max_threshold = config['ground_z_max_threshold']

    max_time_diff = config[
        'max_time_diff']  # allowed time difference between different sensors for aggregating over sensors

    ################################ Load config for used sensors for aggregation #################################
    sensors = config['sensors']
    print(f"Lidar sensors: {sensors}")

    ########################### Generate list for sensor range (needed for visibility masks) later ###################
    sensors_max_range_list = []
    for sensor in sensors:
        if sensor in ['LIDAR_LEFT', 'LIDAR_RIGHT']:
            sensors_max_range_list.append(200)
        if sensor in ['LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR']:
            sensors_max_range_list.append(35)
    print(f"Lidar sensors max range: {sensors_max_range_list}")
    sensor_max_ranges_arr = np.array(sensors_max_range_list, dtype=np.float64)

    ############################## Load config for cameras for camera visibility mask ###############################
    cameras = config['cameras']
    print(f"Cameras: {cameras}")

    ############################# Start processing scene #########################################
    my_scene = trucksc.scene[indice]
    scene_name = my_scene['name']
    print(f"Processing scene: {scene_name}")
    scene_description = my_scene['description']
    print(f"Scene description: {scene_description}")
    first_sample_token = my_scene[
        'first_sample_token']
    my_sample = trucksc.get('sample',
                            first_sample_token)

    ############################# Find weather in scene description for lidar intensity filtering ####################
    # find the part that starts with "weather."
    weather_tag = next(tag for tag in scene_description.split(';') if tag.startswith('weather.'))
    # split on the dot and take the second piece
    weather = weather_tag.split('.', 1)[1]

    ############################## Data split handling #############################################
    if args.split == 'train':
        if scene_name not in splits.train:
            print(f"--- Scene '{scene_name}' is not in split '{args.split}', skipping. ---")
            return
    elif args.split == 'val':
        if scene_name not in splits.val:
            print(f"--- Scene '{scene_name}' is not in split '{args.split}', skipping. ---")
            return
    elif args.split == 'test':
        if scene_name not in splits.test:
            print(f"--- Scene '{scene_name}' is not in split '{args.split}', skipping. ---")
            return
    elif args.split == 'mini_train':
        if scene_name not in splits.mini_train:
            print(f"--- Scene '{scene_name}' is not in split '{args.split}', skipping. ---")
            return
    elif args.split == 'mini_val':
        if scene_name not in splits.mini_val:
            print(f"--- Scene '{scene_name}' is not in split '{args.split}', skipping. ---")
            return
    elif args.split in ['all', 'mini']:
        pass
    else:
        raise NotImplementedError(f"The split '{args.split}' is not a valid or recognized split.")

    ############################ Define learning indexes ##########################################
    # Define the numeric index for the 'Background' class
    background_label_key = 36
    BACKGROUND_LEARNING_INDEX = learning_map.get(background_label_key, 13)

    free_label_key = 37
    FREE_LEARNING_INDEX = learning_map.get(free_label_key, 14)

    ############################ Mapping from Category name to learning id  and vice versa ############################
    # Create a mapping from category name (e.g., "car") to learning ID (e.g., 4)
    category_name_to_learning_id = {}
    for label_key, label_name_in_yaml in truckscenesyaml['labels'].items():
        mapped_label_index = learning_map.get(label_key)
        if mapped_label_index is not None:
            category_name_to_learning_id[label_name_in_yaml] = mapped_label_index

    # Create the reverse mapping from learning ID (e.g., 10) to name (e.g., "truck")
    learning_id_to_name = {int(k): v for k, v in truckscenesyaml['labels_16'].items()}

    ######################### Get all lidar entries for the current scene and given scenes ###########################
    lidar_entries = load_lidar_entries(trucksc=trucksc, sample=my_sample, lidar_sensors=sensors)
    print(f"Number of lidar entries: {len(lidar_entries)}")

    ########################## Generate groups of lidar scans from different sensors with max time diff ##############
    groups = group_entries(entries=lidar_entries, lidar_sensors=sensors, max_time_diff=max_time_diff)
    print(f"\n✅ Total groups found: {len(groups)}")

    ######################## Looping over the generated groups and save data in a dict_list ############################
    dict_list = []

    reference_ego_pose = None
    ref_ego_from_global = None

    for i, group in enumerate(groups):
        print(f"Processing group {i}, timestamps:")
        for sensor in sensors:
            print(f"  {sensor}: {group[sensor]['timestamp']} | keyframe: {group[sensor]['keyframe']}")

        ref_sensor = sensors[0]

        ################### Generate a sample dict to load the lidar point clouds and calculate timestamp #########
        sample_data_dict = {sensor: group[sensor]['token'] for sensor in sensors}
        sample = {
            'timestamp': np.mean([group[s]['timestamp'] for s in sensors]),
            'data': sample_data_dict,
            'sample_data_token': sample_data_dict[ref_sensor],
            'is_key_frame': group[ref_sensor]['keyframe'],
        }

        ##################################### Load point cloud #########################################
        if load_mode == 'pointwise':
            sensor_fused_pc, sensor_ids_points = get_pointwise_fused_pointcloud(trucksc, sample,
                                                                                allowed_sensors=sensors)
        elif load_mode == 'rigid':
            sensor_fused_pc, sensor_ids_points = get_rigid_fused_pointcloud(trucksc, sample, allowed_sensors=sensors)
        else:
            raise ValueError(f'Fusion mode {load_mode} is not supported')

        if sensor_fused_pc.timestamps is not None:
            print(
                f"The fused sensor pc at frame {i} has the shape: {sensor_fused_pc.points.shape} with timestamps: {sensor_fused_pc.timestamps.shape}")
        else:
            print(f"The fused sensor pc at frame {i} has the shape: {sensor_fused_pc.points.shape} with no timestamps.")

        ############################# Specify list for scenes with manual annotations given ####################
        if args.version == 'v1.0-trainval':
            scene_terminal_list = [4, 68, 69, 70, 203, 205, 206, 241, 272, 273, 423, 492, 597]
        elif args.version == 'v1.0-test':
            scene_terminal_list = [3, 114, 115, 116]
        elif args.version == 'v1.0-mini':
            scene_terminal_list = [5]

        ################################# Uncomment if you need to save pointclouds for annotation ###########################

        """if indice in scene_terminal_list:
            annotation_base = os.path.join(data_root, 'annotation')
            # Format the filename with leading zeros to maintain order (e.g., 000000.pcd, 000001.pcd)

            if sample['is_key_frame']:
                output_pcd_filename = f"{i:06d}_keyframe.pcd"
                annotation_data_save_path = os.path.join(annotation_base, scene_name, 'pointclouds', output_pcd_filename)
            else:
                output_pcd_filename = f"{i:06d}_nonkeyframe.pcd"
                annotation_data_save_path = os.path.join(annotation_base, scene_name, 'pointclouds', 'nonkeyframes', output_pcd_filename)

            # Save the raw fused point cloud before any filtering
            # We use .points.T to get the (N, features) shape
            save_pointcloud_for_annotation(sensor_fused_pc.points.T, annotation_data_save_path)"""

        ########################################### Get boxes given in dataset ####################################
        # Get original boxes from the dataset
        boxes_global = trucksc.get_boxes(sample['sample_data_token'])

        # Get ego pose for transformation
        pose_record = trucksc.getclosest('ego_pose',
                                         trucksc.get('sample_data', sample['sample_data_token'])['timestamp'])

        # Transform boxes to ego frame
        boxes_ego = transform_boxes_to_ego(
            boxes=boxes_global,
            ego_pose_record=pose_record
        )

        original_boxes_token = [box.token for box in boxes_ego]
        original_object_category_names = [truckscenes.get('sample_annotation', box_token)['category_name'] for box_token
                                          in
                                          original_boxes_token]

        # Convert original category names to numeric learning IDs
        converted_object_category = []
        for category_name in original_object_category_names:
            if category_name in category_name_to_learning_id:
                numeric_label = category_name_to_learning_id[category_name]
            else:
                print(f"⚠️ Warning: Category '{category_name}' not found in mapping. Assigning as Background.")
                numeric_label = BACKGROUND_LEARNING_INDEX
            converted_object_category.append(numeric_label)

        ###################################### get manual boxes for terminal scenes ###################################
        if indice in scene_terminal_list:
            annotation_base = os.path.join(data_root, 'annotation')
            if sample['is_key_frame']:
                load_pcd_annotation_filename = f"{i:06d}_keyframe.json"
                load_annotation_ending = os.path.join('keyframes', load_pcd_annotation_filename)
            else:
                load_pcd_annotation_filename = f"{i:06d}_nonkeyframe.json"
                load_annotation_ending = os.path.join('nonkeyframes', load_pcd_annotation_filename)

            annotation_data_load_path = os.path.join(annotation_base, scene_name, 'annotations',
                                                     load_annotation_ending)
            manual_boxes = parse_single_annotation_file(annotation_data_load_path)

            if manual_boxes:
                print(f"Frame {i}: Loaded {len(manual_boxes)} manual annotations. Augmenting GT.")
                for box in manual_boxes:
                    boxes_ego.append(box)
                    manual_category_name = box.name
                    numeric_label = category_name_to_learning_id.get(manual_category_name, BACKGROUND_LEARNING_INDEX)
                    converted_object_category.append(numeric_label)

        ####################### Assign object tokens for manual boxes #####################################
        object_tokens = []
        for box in boxes_ego:
            if box.token is not None:
                # This is an original box from the dataset, get its instance token
                instance_token = truckscenes.get('sample_annotation', box.token)['instance_token']
                object_tokens.append(instance_token)
            else:
                # This is a manually added box, it has no token in the dataset
                manual_token = f"manual_{box.name}"
                object_tokens.append(manual_token)

        ############################# get bbox attributes ##########################
        locs = np.array([b.center for b in boxes_ego]).reshape(-1,
                                                               3)
        dims = np.array([b.wlh for b in boxes_ego]).reshape(-1,
                                                            3)
        rots = np.array([b.orientation.yaw_pitch_roll[0]
                         for b in boxes_ego]).reshape(-1, 1)

        ########################## Bounding boxes without any modifications ########################

        gt_bbox_3d_unmodified = np.concatenate([locs, dims, rots], axis=1).astype(
            np.float32)

        ############################### Enlarge bounding boxes to extract dynamic points #################
        percentage_factor = 1.10
        max_absolute_increase_m = 0.5  # The maximum total increase for any dimension (e.g., 25cm per side)

        # 1. Calculate the dimensions if extended by the percentage factor
        dims_extended_by_percentage = dims * percentage_factor

        # 2. Calculate the maximum allowed dimensions based on the absolute cap
        dims_with_max_absolute_increase = dims + max_absolute_increase_m

        # 3. For each dimension, take the SMALLER of the two options
        new_dims = np.minimum(dims_extended_by_percentage, dims_with_max_absolute_increase)

        gt_bbox_3d_points_in_boxes_cpu_enlarged = gt_bbox_3d_unmodified.copy()

        gt_bbox_3d_points_in_boxes_cpu_enlarged[:,
        6] += np.pi / 2.  # adjust yaw angles by 90 degrees for points_in_boxes_cpu()
        gt_bbox_3d_points_in_boxes_cpu_enlarged[:, 2] -= dims[:,
                                                         2] / 2.  # needed as points_in_boxes_cpu() expects bottom center of box
        # gt_bbox_3d_points_in_boxes_cpu_enlarged[:, 2] = gt_bbox_3d_points_in_boxes_cpu_enlarged[:, 2] - 0.1
        gt_bbox_3d_points_in_boxes_cpu_enlarged[:, 3:6] = new_dims

        ########################################### Boxes for calculating IoU later ###################################
        gt_bbox_3d_overlap_enlarged = gt_bbox_3d_unmodified.copy()
        gt_bbox_3d_overlap_enlarged[:, 3:6] = new_dims

        ################################################ Bounding box just for filtering dynamic map ###################
        gt_bbox_3d_points_in_boxes_cpu_max_enlarged = gt_bbox_3d_unmodified.copy()

        percentage_factor_max = 1.15
        fixed_increase_m = 0.3

        increase_from_percentage = dims * (percentage_factor_max - 1.0)

        final_increase = np.maximum(increase_from_percentage, fixed_increase_m)

        dims_filter = dims + final_increase

        # enlarge width of cars and trucks as mirrors often not included in bounding boxes --> avoid artifacts
        width_scale_car = 1.20
        width_scale_truck = 1.25
        for index, cat in enumerate(original_object_category_names):
            if cat == 'vehicle.car':
                special_width = dims[index, 0] * width_scale_car
                dims_filter[index, 0] = np.maximum(dims_filter[index, 0], special_width)
            elif cat == 'vehicle.truck':
                special_width = dims[index, 0] * width_scale_truck
                dims_filter[index, 0] = np.maximum(dims_filter[index, 0], special_width)

        gt_bbox_3d_points_in_boxes_cpu_max_enlarged[:,
        6] += np.pi / 2.  # adjust yaw angles by 90 degrees for points_in_boxes_cpu()
        gt_bbox_3d_points_in_boxes_cpu_max_enlarged[:, 2] -= dims[:,
                                                             2] / 2.  # needed as points_in_boxes_cpu() expects bottom center of box
        # gt_bbox_3d_points_in_boxes_cpu_max_enlarged[:, 2] = gt_bbox_3d_points_in_boxes_cpu_max_enlarged[:, 2] - 0.1
        gt_bbox_3d_points_in_boxes_cpu_max_enlarged[:, 3:6] = dims_filter

        ############################### Visualize if specified in arguments ###########################################
        if args.vis_raw_pc and i % 5 == 0:
            visualize_pointcloud_bbox(sensor_fused_pc.points.T,
                                      boxes=boxes_ego,
                                      title=f"Fused raw sensor PC + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)

        ############################## Filter raw pc #################################################################
        if args.filter_raw_pc and args.filter_mode != 'none':
            # 1) prepare
            raw_pts = sensor_fused_pc.points.T
            raw_sids = sensor_ids_points.copy()
            pcd_raw = o3d.geometry.PointCloud()
            pcd_raw.points = o3d.utility.Vector3dVector(raw_pts[:, :3])

            # 2) filter
            filtered_raw_pcd, kept_raw_idx = denoise_pointcloud(pcd_raw, args.filter_mode, config,
                                                                location_msg=f"raw pc at frame {i}")
            # 3) re‐assemble arrays
            raw_pts = np.asarray(filtered_raw_pcd.points)
            raw_sids = raw_sids[kept_raw_idx]

            # 4) re‐append extra features:
            if sensor_fused_pc.points.shape[0] > 3:
                raw_pts = np.hstack([raw_pts, sensor_fused_pc.points.T[kept_raw_idx, 3:]])

            # Now overwrite fused_pc & sensor_ids:
            sensor_fused_pc.points = raw_pts.T
            sensor_fused_pc.timestamps = sensor_fused_pc.timestamps[:, kept_raw_idx]
            sensor_ids_points = raw_sids.copy()

        assert sensor_fused_pc.points.shape[1] == sensor_ids_points.shape[0], \
            f"point count {sensor_fused_pc.points.shape[1]} vs sensor_ids {sensor_ids_points.shape[0]}"
        ##############################################################################################################

        ############################### Visualize if specified in arguments ###########################################
        if args.vis_raw_pc and args.filter_raw_pc and i % 5 == 0:
            visualize_pointcloud_bbox(sensor_fused_pc.points.T,
                                      boxes=boxes_ego,
                                      title=f"Fused filtered raw sensor PC (filter mode {args.filter_mode}) + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)

        ############################# cut out movable object points and masks ##########################
        points_to_check_np = sensor_fused_pc.points.T[:, :3][np.newaxis, :, :]
        points_to_check_tensor = torch.from_numpy(points_to_check_np).float()

        boxes_to_check_np = gt_bbox_3d_points_in_boxes_cpu_enlarged[np.newaxis, :]
        boxes_to_check_tensor = torch.from_numpy(boxes_to_check_np).float()

        """start_time_gpu = time.perf_counter()

        points_to_check_tensor_gpu = points_to_check_tensor.cuda()
        boxes_to_check_tensor_gpu = boxes_to_check_tensor.cuda()

        points_in_boxes = points_in_boxes_all(points_to_check_tensor_gpu,
                                              boxes_to_check_tensor_gpu)  # use function to identify which points belong to which bounding box

        end_time_gpu = time.perf_counter()
        gpu_time = end_time_gpu - start_time_gpu
        print(f"GPU time for one iteration: {gpu_time:.6f} seconds")"""

        start_time_cpu = time.perf_counter()

        points_in_boxes = points_in_boxes_cpu(
            points_to_check_tensor,
            boxes_to_check_tensor)  # use function to identify which points belong to which bounding box

        end_time_cpu = time.perf_counter()
        cpu_time = end_time_cpu - start_time_cpu
        print(f"CPU time for one iteration: {cpu_time:.6f} seconds")

        #################################### Mask for the ego vehicle itself #######################################
        points_xyz = sensor_fused_pc.points.T[:, :3]

        # Create a mask for points outside the ego vehicle bounding box
        inside_x = torch.from_numpy(points_xyz[:, 0] >= x_min_self) & torch.from_numpy(points_xyz[:, 0] <= x_max_self)
        inside_y = torch.from_numpy(points_xyz[:, 1] >= y_min_self) & torch.from_numpy(points_xyz[:, 1] <= y_max_self)
        inside_z = torch.from_numpy(points_xyz[:, 2] >= z_min_self) & torch.from_numpy(points_xyz[:, 2] <= z_max_self)

        inside_ego_mask = inside_x & inside_y & inside_z

        ############################ Prepare point clouds for kiss-icp #################################
        VELOCITY_THRESHOLD_M_S = 0.2

        is_box_moving_mask = np.zeros(len(boxes_ego), dtype=bool)
        annotated_boxes_indices = [idx for idx, box in enumerate(boxes_ego) if box.token is not None]
        box_annotation_tokens = [boxes_ego[i].token for i in annotated_boxes_indices]

        if box_annotation_tokens:
            box_velocities = np.array([trucksc.box_velocity(token) for token in box_annotation_tokens])
            box_speeds = np.linalg.norm(np.nan_to_num(box_velocities, nan=0.0), axis=1)
            are_annotated_boxes_moving = box_speeds > VELOCITY_THRESHOLD_M_S
            np.put(is_box_moving_mask, annotated_boxes_indices, are_annotated_boxes_moving)

        points_in_moving_boxes_mask_torch = torch.zeros(sensor_fused_pc.points.shape[1], dtype=torch.bool)
        if np.any(is_box_moving_mask):
            points_in_moving_boxes_mask_torch = points_in_boxes[0][:, is_box_moving_mask].any(dim=1)

        points_to_remove_mask_torch = inside_ego_mask | points_in_moving_boxes_mask_torch

        initial_keep_for_icp_mask_np = ~points_to_remove_mask_torch.numpy()

        pc_for_icp = sensor_fused_pc.points.T[initial_keep_for_icp_mask_np]

        if args.filter_lidar_intensity and weather in ['snow', 'rain', 'fog']:
            print(f"Applying special conditional intensity filter to ICP data for weather: '{weather}'...")

            # Identify which points in our current `pc_for_icp` were inside ANY original bounding box.
            points_in_any_box_mask_np = points_in_boxes[0].any(dim=1).numpy()
            is_point_in_box_for_icp_pc = points_in_any_box_mask_np[initial_keep_for_icp_mask_np]

            # The points we need to *evaluate* with the filter are those NOT in a box (i.e., background points).
            background_points_mask = ~is_point_in_box_for_icp_pc
            background_points_to_filter = pc_for_icp[background_points_mask]

            if background_points_to_filter.shape[0] > 0:
                # Get the intensity filter mask ONLY for the background points.
                intensity_keep_mask_for_bg = get_weather_intensity_filter_mask(
                    point_cloud=background_points_to_filter,
                    weather_condition=weather,
                    intensity_thresh=intensity_threshold,
                    distance_thresh=distance_intensity_threshold,
                    keep_ground_points=True,
                    ground_z_min=ground_z_min_threshold,
                    ground_z_max=ground_z_max_threshold
                )

                # Start with a final mask that keeps all points in `pc_for_icp`.
                final_keep_mask = np.ones(pc_for_icp.shape[0], dtype=bool)

                # Points in boxes remain True (kept) because they are not part of `background_points_mask`.
                final_keep_mask[background_points_mask] = intensity_keep_mask_for_bg

                num_before = pc_for_icp.shape[0]
                pc_for_icp = pc_for_icp[final_keep_mask]
                num_after = pc_for_icp.shape[0]
                print(
                    f"  Conditional intensity filter removed {num_before - num_after} background points from ICP data.")

        print(
            f"Original points for frame: {sensor_fused_pc.points.shape[1]}. Final points for ICP: {pc_for_icp.shape[0]}")

        ################################ Prepare points for static and dynamic maps ################################
        num_points = sensor_fused_pc.points.shape[1]
        points_label = np.full((num_points, 1), BACKGROUND_LEARNING_INDEX, dtype=np.uint8)

        # Assign object labels to points inside bounding boxes
        for box_idx in range(gt_bbox_3d_points_in_boxes_cpu_enlarged.shape[0]):
            # Get the mask for points in the current box
            object_points_mask = points_in_boxes[0][:, box_idx].bool()
            # Get the semantic label for this object type
            object_label = converted_object_category[box_idx]
            # Assign the object label to the corresponding points in the points_label
            points_label[object_points_mask] = object_label

        pc_with_semantic = np.concatenate([sensor_fused_pc.points.T[:, :3], points_label], axis=1)

        object_points_list = []  # creates an empty list to store points associated with each object
        objects_points_list_sensor_ids = []
        j = 0
        # Iterate through each bounding box along the last dimension
        while j < points_in_boxes.shape[-1]:
            # Create a boolean mask indicating whether each point belongs to the current bounding box.
            object_points_mask = points_in_boxes[0][:, j].bool()
            # Extract points using mask to filter points
            object_points = sensor_fused_pc.points.T[object_points_mask]
            object_points_sensor_ids = sensor_ids_points.copy().T[object_points_mask]
            # Store the filtered points, Result is a list of arrays, where each element contains the points belonging to a particular object
            object_points_list.append(object_points)
            objects_points_list_sensor_ids.append(object_points_sensor_ids)
            j = j + 1

        point_box_mask = points_in_boxes[0]  # Remove batch dim: shape (Npoints, Nboxes)

        # Point is dynamic if it falls inside *any* box
        dynamic_mask = point_box_mask.any(dim=-1)  # shape: (Npoints,)

        num_dynamic_points = dynamic_mask.sum().item()
        print(f"Number of dynamic points: {num_dynamic_points}")

        # Get static mask (inverse)
        static_mask = ~dynamic_mask

        num_ego_points = inside_ego_mask.sum().item()
        print(f"Number of points on ego vehicle: {num_ego_points}")

        ################################# Dynamic point filtering of static map with max enlarged boxes##################

        points_in_boxes_max_enlarged = points_in_boxes_cpu(
            torch.from_numpy(sensor_fused_pc.points.T[:, :3][np.newaxis, :, :]),
            torch.from_numpy(gt_bbox_3d_points_in_boxes_cpu_max_enlarged[np.newaxis, :])
        )

        dynamic_mask_max_enlarged = points_in_boxes_max_enlarged[0].any(dim=-1)

        only_in_max_enlarged_mask = dynamic_mask_max_enlarged & ~dynamic_mask
        num_points_max_enlarged = only_in_max_enlarged_mask.sum().item()

        print(f"Number of points captured only by the max enlarged box: {num_points_max_enlarged}")

        ############################### Combine masks ##############################
        ego_filter_mask = ~inside_ego_mask

        initial_static_points_mask = static_mask & ego_filter_mask
        final_static_map_mask = initial_static_points_mask & ~dynamic_mask_max_enlarged

        pc_ego_unfiltered = sensor_fused_pc.points.T[final_static_map_mask]
        pc_ego_unfiltered_sensors = sensor_ids_points.copy().T[final_static_map_mask]
        print(
            f"Number of static points extracted: {pc_ego_unfiltered.shape} with sensor_ids {pc_ego_unfiltered_sensors.shape}")

        pc_with_semantic_ego_unfiltered = pc_with_semantic[final_static_map_mask]
        pc_with_semantic_ego_unfiltered_sensors = sensor_ids_points.copy().T[final_static_map_mask]

        print(
            f"Number of semantic static points extracted: {pc_with_semantic_ego_unfiltered.shape} with sensor_ids {pc_with_semantic_ego_unfiltered_sensors.shape}")

        ###################################### Generate labels for mapmos as gt ####################################
        dynamic_mask_mapmos = dynamic_mask | inside_ego_mask | dynamic_mask_max_enlarged

        current_frame_mapmos_labels = dynamic_mask_mapmos.cpu().numpy().astype(np.int32).reshape(-1, 1)
        assert current_frame_mapmos_labels.shape[0] == sensor_fused_pc.points.T.shape[0], \
            "Mismatch between number of points in scan and generated labels"

        total_mapmos_dynamic_labels = dynamic_mask_mapmos.sum().item()
        print(
            f"Total points labeled as dynamic for MapMOS input (annotated dynamic + ego): {total_mapmos_dynamic_labels}")

        ############################### Visualize if specified in arguments ###########################################
        if args.vis_static_pc:
            visualize_pointcloud_bbox(pc_with_semantic_ego_unfiltered,
                                      boxes=boxes_ego,
                                      title=f"Fused static sensor PC + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)
        ############################################################################################################

        pc_ego = pc_ego_unfiltered.copy()

        ########################################## Lidar intensity filtering #######################################
        if args.filter_lidar_intensity:
            print(f'Shape of pc_ego before weather filtering: {pc_ego.shape}')
            intensity_keep_mask_ego = get_weather_intensity_filter_mask(
                point_cloud=pc_ego,
                weather_condition=weather,
                intensity_thresh=intensity_threshold,
                distance_thresh=distance_intensity_threshold,
                keep_ground_points=True,
                ground_z_min=ground_z_min_threshold,
                ground_z_max=ground_z_max_threshold
            )

            # Apply the mask to both the points and their corresponding sensor IDs
            pc_ego = pc_ego[intensity_keep_mask_ego]
            pc_ego_unfiltered_sensors = pc_ego_unfiltered_sensors[intensity_keep_mask_ego]
            print(f"Shape of pc_ego after weather filtering: {pc_ego.shape}")

            ################################ Visualize if specified in arguments ##################################
            if args.vis_lidar_intensity_filtered:
                visualize_pointcloud_bbox(pc_ego,
                                          boxes=boxes_ego,
                                          title=f"Fused filtered static sensor PC after lidar intensity filtering + BBoxes + Ego BBox - Frame {i}",
                                          self_vehicle_range=self_range,
                                          vis_self_vehicle=True)
            #######################################################################################################
        else:
            print(f"No lidar intensity filtering according to arguments")

        ############################# Apply filtering to static points in ego frame #################################
        if args.filter_static_pc and args.filter_mode != 'none':
            print(f"Applying geometric filter ('{args.filter_mode}') to the aggregated static map...")
            pcd_static = o3d.geometry.PointCloud()
            pcd_static.points = o3d.utility.Vector3dVector(pc_ego[:, :3])
            """filtered_pcd_static, kept_indices = denoise_pointcloud(
                pcd_static, args.filter_mode, config, location_msg=f"static ego points at frame {i}"
            )"""
            filtered_pcd_static, kept_indices = denoise_pointcloud_advanced(
                pcd=pcd_static,
                filter_mode=args.filter_mode,
                config=config,
                location_msg="aggregated static points"
            )

            pc_ego = np.asarray(filtered_pcd_static.points)
            pc_ego_unfiltered_sensors = pc_ego_unfiltered_sensors[kept_indices]

        assert pc_ego.shape[0] == pc_ego_unfiltered_sensors.shape[0], (
            f"static points ({pc_ego.shape[0]}) != sensor_ids ({pc_ego_unfiltered_sensors.shape[0]})"
        )

        ############################ Visualization #############################################################
        if args.vis_static_pc and args.filter_static_pc:
            visualize_pointcloud_bbox(pc_ego,
                                      boxes=boxes_ego,
                                      title=f"Fused filtered static sensor PC (filter mode {args.filter_mode}) + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)
        #######################################################################################################

        ################################### Transform points in different coordinate frames #######################
        ego_pose_i = trucksc.getclosest('ego_pose', sample['timestamp'])

        global_from_ego_i = transform_matrix(ego_pose_i['translation'], Quaternion(ego_pose_i['rotation']),
                                             inverse=False)

        if reference_ego_pose is None:
            reference_ego_pose = ego_pose_i
            ref_ego_from_global = transform_matrix(reference_ego_pose['translation'],
                                                   Quaternion(reference_ego_pose['rotation']), inverse=True)
            ego_ref_from_ego_i = np.eye(4)
            print(f"Frame {i}: Set as reference frame.")
        else:
            # ego_ref <- global <- ego_i
            ego_ref_from_ego_i = ref_ego_from_global @ global_from_ego_i
            print(f"Frame {i}: Calculated transform to reference frame.")

        points_in_ref_frame = transform_points(pc_ego, ego_ref_from_ego_i)
        print(f"Frame {i}: Transformed static points to ref ego. Shape: {points_in_ref_frame.shape}")

        points_in_global_frame = transform_points(pc_ego, global_from_ego_i)
        print(f"Frame {i}: Transformed static points to global. Shape: {points_in_global_frame.shape}")

        if args.vis_static_pc_global:
            visualize_pointcloud(points_in_ref_frame, title=f"Fused sensor PC in world coordinates - Frame {i}")

        pc_ego_i_save = pc_ego.copy()
        print(f"Frame {i}: Static points in ego frame shape: {pc_ego_i_save.shape}")
        pc_ego_ref_save = points_in_ref_frame.copy()
        pc_global_save = points_in_global_frame.copy()

        ################## Save all information into a dict  ########################
        ref_sd = trucksc.get('sample_data', sample['sample_data_token'])

        frame_dict = {
            "sample_timestamp": sample['timestamp'],
            "scene_name": scene_name,
            "sample_token": trucksc.get('sample', ref_sd['sample_token'])['token'],
            "is_key_frame": sample['is_key_frame'],
            "converted_object_category": converted_object_category,
            "boxes_ego": boxes_ego,
            "gt_bbox_3d_points_in_boxes_cpu_max_enlarged": gt_bbox_3d_points_in_boxes_cpu_max_enlarged,
            "gt_bbox_3d_points_in_boxes_cpu_enlarged": gt_bbox_3d_points_in_boxes_cpu_enlarged,
            "gt_bbox_3d_overlap_enlarged": gt_bbox_3d_overlap_enlarged,
            "gt_bbox_3d_unmodified": gt_bbox_3d_unmodified,
            "object_tokens": object_tokens,
            "object_points_list": object_points_list,
            "object_points_list_sensor_ids": objects_points_list_sensor_ids,
            "raw_lidar_ego": sensor_fused_pc.points.T,
            "raw_lidar_ego_sensor_ids": sensor_ids_points.T,
            "mapmos_per_point_labels": current_frame_mapmos_labels,
            "lidar_pc_ego_i": pc_ego_i_save,
            "lidar_pc_ego_sensor_ids": pc_ego_unfiltered_sensors,
            "lidar_pc_ego_ref": pc_ego_ref_save,
            "lidar_pc_global": pc_global_save,
            "ego_pose": ego_pose_i,  # Current frame's ego pose dictionary
            "ego_ref_from_ego_i": ego_ref_from_ego_i,
            "global_from_ego_i": global_from_ego_i,
            "lidar_pc_for_icp_ego_i": pc_for_icp
        }

        ########################## Save dictionary into a list ###############################
        dict_list.append(frame_dict)

    ################# Prepare Lists for Static Scene Points (in Reference Ego Frame) ########################.

    print("Extracting static points previously transformed to reference ego frame...")

    ################## static point cloud in ego i frame ################################
    unrefined_pc_ego_list = [frame_dict['lidar_pc_ego_i'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_pc_ego_list)} static point clouds (in ego i frame).")
    unrefined_pc_ego_list_sensor_ids = [frame_dict['lidar_pc_ego_sensor_ids'] for frame_dict in dict_list]
    pc_ego_combined_draw = np.concatenate(unrefined_pc_ego_list, axis=0)
    print(f"Pc ego i combined shape: {pc_ego_combined_draw.shape}")

    ######################## Visualization #################################################
    if args.vis_aggregated_static_ego_i_pc:
        pc_ego_to_draw = o3d.geometry.PointCloud()
        pc_coordinates = pc_ego_combined_draw[:, :3]
        pc_ego_to_draw.points = o3d.utility.Vector3dVector(pc_coordinates)
        o3d.visualization.draw_geometries([pc_ego_to_draw], window_name="Combined static point clouds (in ego i frame)")

    #################### static point cloud in ego ref frame #############################
    unrefined_pc_ego_ref_list = [frame_dict['lidar_pc_ego_ref'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_pc_ego_ref_list)} static point clouds (in ego ref frame).")
    unrefined_pc_ego_ref_list_sensor_ids = [frame_dict['lidar_pc_ego_sensor_ids'] for frame_dict in dict_list]
    pc_ego_ref_combined_draw = np.concatenate(unrefined_pc_ego_ref_list, axis=0)
    print(f"Pc ego ref combined shape: {pc_ego_ref_combined_draw.shape}")

    ###################### Visualization ##################################################
    if args.vis_aggregated_static_ego_ref_pc:
        pc_ego_ref_to_draw = o3d.geometry.PointCloud()
        pc_ego_ref_coordinates = pc_ego_ref_combined_draw[:, :3]
        pc_ego_ref_to_draw.points = o3d.utility.Vector3dVector(pc_ego_ref_coordinates)
        o3d.visualization.draw_geometries([pc_ego_ref_to_draw],
                                          window_name="Combined static point clouds (in ego ref frame)")

    ######################### static point cloud in world frame #############################
    unrefined_pc_global_list = [frame_dict['lidar_pc_global'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_pc_global_list)} static point clouds (in world frame).")
    pc_global_combined_draw = np.concatenate(unrefined_pc_global_list, axis=0)
    print(f"Pc global shape: {pc_global_combined_draw.shape}")

    ####################### Visualization ################################################
    if args.vis_aggregated_static_global_pc:
        pc_global_to_draw = o3d.geometry.PointCloud()
        pc_global_coordinates = pc_global_combined_draw[:, :3]
        pc_global_to_draw.points = o3d.utility.Vector3dVector(pc_global_coordinates)
        o3d.visualization.draw_geometries([pc_global_to_draw],
                                          window_name="Combined static point clouds (in global frame)")

    ######################### point cloud with all static and dynamic objects #################
    raw_pc_list = [frame_dict['raw_lidar_ego'] for frame_dict in dict_list]
    print(f"Extracted {len(raw_pc_list)} static and dynamic point clouds (in ego i frame).")
    raw_pc_list_sensor_ids = [frame_dict['raw_lidar_ego_sensor_ids'] for frame_dict in dict_list]
    print(f"Extracted {len(raw_pc_list_sensor_ids)} sensor ids for static and dynamic point clouds (in ego i frame).")
    raw_pc_draw = np.concatenate(raw_pc_list, axis=0)
    print(f"Raw Pc with static and dynamic points shape: {raw_pc_draw.shape}")

    ########################## Visualization #############################################
    if args.vis_aggregated_raw_pc_ego_i:
        raw_pc_to_draw = o3d.geometry.PointCloud()
        raw_pc_coordinates = raw_pc_draw[:, :3]
        raw_pc_to_draw.points = o3d.utility.Vector3dVector(raw_pc_coordinates)
        o3d.visualization.draw_geometries([raw_pc_to_draw],
                                          window_name="Combined static and dynamic point clouds (in ego i frame)")

    ######################################################################################
    ##################### Prepare lidar timestamps for Kiss-ICP ##########################
    # Extract timestamps associated with each frame's original ego pose
    try:
        lidar_timestamps = [frame_dict['sample_timestamp'] for frame_dict in dict_list]
        print(f"Extracted {len(lidar_timestamps)} timestamps.")
    except KeyError:
        print("Timestamp key not found in ego_pose, setting lidar_timestamps to None.")
        lidar_timestamps = None  # Fallback

    print(f"Lidar timestamps: {lidar_timestamps}")

    ######################### Process ego_ref_from_ego_i for kissicp #############################
    if not dict_list:
        print("dict_list is empty. Cannot proceed with pose comparison.")
    else:
        gt_relative_poses_list = [fd['ego_ref_from_ego_i'] for fd in dict_list]
        gt_relative_poses_arr = np.array(gt_relative_poses_list)  # Shape: (num_frames, 4, 4)
        print(f"Collected {gt_relative_poses_arr.shape[0]} GT relative poses for comparison.")

    ##############################################################################################
    ##################################### MapMOS #################################################

    if args.run_mapmos:

        print("Running mapmos to refine static map.")

        # mos_config = MOSConfig(voxel_size_belief=0.25, delay_mos=10, ...)
        # odometry_config = OdometryConfig(voxel_size=0.5, ...)
        # data_config = DataConfig(max_range=100.0, ...)
        # pipeline_config = { "mos": mos_config, "odom": odometry_config, "data": data_config }

        in_memory_dataset_mapmos = None
        mapmos_pipeline = None
        estimated_poses_kiss = None
        log_dir_mapmos = osp.join(save_path, scene_name, "mapmos_logs")

        mapmos_labels_per_scan = [frame_dict['mapmos_per_point_labels'] for frame_dict in dict_list]

        try:
            print("Initializing InMemoryDatasetMapMOS...")

            in_memory_dataset_mapmos = InMemoryDatasetMapMOS(
                lidar_scans=raw_pc_list,
                scan_timestamps=lidar_timestamps,
                labels_per_scan=mapmos_labels_per_scan,
                gt_global_poses=gt_relative_poses_arr,  # Optional
                sequence_id=f"{scene_name}_mapmos_run"
            )
            print(f"InMemoryDatasetMapMOS initialized with {len(in_memory_dataset_mapmos)} scans.")
        except Exception as e:
            print(f"Error creating InMemoryDatasetMapMos: {e}. Skipping MapMOS.")

        config_path_mapmos = None
        if in_memory_dataset_mapmos:
            try:
                print(f"Initializing MapMOSPipeline with weights: {weights_path_mapmos}")

                if not weights_path_mapmos.is_file():
                    raise FileNotFoundError(f"MapMOS weights not found at: {weights_path_mapmos}")

                mapmos_pipeline = MapMOSPipeline(
                    dataset=in_memory_dataset_mapmos,
                    weights=weights_path_mapmos,
                    config=config_path_mapmos,
                    visualize=False,
                    save_ply=True,
                    save_kitti=False,
                    n_scans=-1,
                    jump=0
                )

                print("MapMOS pipeline initialized.")

                mapmos_start_time = time.time()
                print("Running MapMOS pipeline...")
                run_output, all_frame_predictions = mapmos_pipeline.run()
                mapmos_end_time = time.time()
                print(f"MapMOS pipeline finished in {mapmos_end_time - mapmos_start_time:.2f} seconds.")

                # Process or print results
                if hasattr(run_output, 'print') and callable(getattr(run_output, 'print')):
                    print("Printing MapMOS results:")
                    run_output.print()
                else:
                    print(
                        "MapMOS run completed. Inspect 'mapmos_pipeline_instance' or 'run_output' for results if available.")

                """for frame_data in all_frame_predictions:
                    scan_idx = frame_data["scan_index"]
                    points = frame_data["points"]
                    predicted_labels = frame_data["predicted_labels"]  # These are your belief_labels_query
                    gt_labels = frame_data["gt_labels"]  # These are your query_labels

                    print(
                        f"Scan Index: {scan_idx}, Points shape: {points.shape}, Predicted Labels shape: {predicted_labels.shape}, GT Labels shape: {gt_labels.shape}")

                    visualize_mapmos_predictions(
                        points_xyz=points,
                        predicted_labels=predicted_labels,
                        scan_index=scan_idx,
                        window_title_prefix=f"MapMOS Output ({scene_name})"  # Using your scene_name
                    )

                    # Example: Filter dynamic points based on MapMOS prediction
                    # Assuming 1 means dynamic, 0 means static in predicted_labels
                    dynamic_points = points[predicted_labels == 1]
                    static_points = points[predicted_labels == 0]
                    print(f"  - Found {dynamic_points.shape[0]} dynamic points and {static_points.shape[0]} static points.")"""

            except Exception as e:
                print(f"Error during MapMOS pipeline execution: {e}")

        static_points_refined = []
        static_points_refined_sensor_ids = []

        if all_frame_predictions:
            print(f"\n--- Filtering static points for all {len(all_frame_predictions)} frames ---")
            for frame_idx, frame_data in enumerate(all_frame_predictions):
                original_points_in_frame = raw_pc_list[frame_idx]
                original_points_in_frame_sensor_ids = raw_pc_list_sensor_ids[frame_idx]

                # Condition 1: Point was considered STATIC in your input GT to MapMOS
                input_gt_is_static_mask = (mapmos_labels_per_scan[frame_idx].flatten() == 0)
                # Condition 2: Point was NOT predicted as DYNAMIC by MapMOS
                mapmos_predicted_labels = frame_data["predicted_labels"]
                mapmos_not_predicted_dynamic_mask = (
                        mapmos_predicted_labels != 1)  # True if MapMOS predicted 0 (static) or -1 (unclassified)

                # Combine both conditions with logical AND
                final_static_mask_for_frame = input_gt_is_static_mask & mapmos_not_predicted_dynamic_mask

                # Apply the mask to get the static points for the current frame
                static_points_this_frame = original_points_in_frame[final_static_mask_for_frame]
                static_points_refined.append(static_points_this_frame)
                static_points_this_frame_sensor_ids = original_points_in_frame_sensor_ids[final_static_mask_for_frame]
                static_points_refined_sensor_ids.append(static_points_this_frame_sensor_ids)

                assert static_points_this_frame.shape[0] == static_points_this_frame_sensor_ids.shape[0]

            print(
                f"Static points list mapmos: {len(static_points_refined)} frames with sensor ids: {len(static_points_refined_sensor_ids)}")

        """raw_pc_0 = raw_pc_list[0]

        # Condition 1: Point is NOT dynamic in your GT input to MapMOS
        input_gt_is_static_mask = (mapmos_labels_per_scan[0].flatten() == 0)

        # Condition 2: Point is NOT classified as dynamic by MapMOS prediction
        mapmos_predicted_labels = all_frame_predictions[0]["predicted_labels"]
        mapmos_not_predicted_dynamic_mask = (
                    mapmos_predicted_labels != 1)  # True if MapMOS predicted 0 (static) or -1 (unclassified)

        # Combine both conditions: A point is truly static if both are true
        final_static_mask = input_gt_is_static_mask & mapmos_not_predicted_dynamic_mask

        # Filter raw_pc_0 using this final static mask
        static_points_from_raw_pc_0 = raw_pc_0[final_static_mask]

        print(f"--- Filtering for Frame 0 ---")
        print(f"Original number of points in raw_pc_0: {raw_pc_0.shape[0]}")
        print(f"Number of points static according to your input GT: {np.sum(input_gt_is_static_mask)}")
        print(f"Number of points NOT predicted as dynamic by MapMOS: {np.sum(mapmos_not_predicted_dynamic_mask)}")
        print(f"Number of points satisfying BOTH static conditions: {static_points_from_raw_pc_0.shape[0]}")


        pcd_static_vis = o3d.geometry.PointCloud()
        pcd_static_vis.points = o3d.utility.Vector3dVector(static_points_from_raw_pc_0[:, :3])
        pcd_static_vis.paint_uniform_color([0.0, 0.0, 1.0])  # Blue for these static points
        o3d.visualization.draw_geometries([pcd_static_vis],
                                          window_name=f"Final Static Points (Frame 0)")"""

    else:
        print("Proceeding without running MapMOS pipeline.")
        # Define default lists that will be used if MapMOS is skipped
        static_points_refined = unrefined_pc_ego_list  # Default to the initial static points
        static_points_refined_sensor_ids = unrefined_pc_ego_list_sensor_ids

    ###############################################################################################
    ################# Refinement using KISS-ICP ###################################################
    poses_kiss_icp = None
    if args.icp_refinement and len(dict_list) > 1:
        print(f"--- Performing KISS-ICP refinement on static global point clouds for scene {scene_name} ---")

        # --- 1. Prepare Dataset and Pipeline ---
        in_memory_dataset = None
        pipeline = None
        estimated_poses_kiss = None
        log_dir_kiss = osp.join(save_path, scene_name, "kiss_icp_logs")

        initial_relative_motions = []

        if args.initial_guess_mode == 'ego_pose':
            print("Using 'ego_pose' for initial guesses.")

            if not dict_list:
                print("Warning: dict_list is empty. Cannot generate 'ego_pose' initial guesses.")
            else:
                initial_relative_motions.append(np.eye(4))
                print(f"  Frame 0 (batch index 0): Initial guess is Identity (ego_pose mode anchor).")

                for k_dict_idx in range(1, len(dict_list)):  # k_dict_idx is the index in dict_list
                    # Global pose of previous frame (k-1) from dataset
                    ego_pose_prev_dict = dict_list[k_dict_idx - 1]['ego_pose']
                    P_dataset_prev_global = transform_matrix(
                        ego_pose_prev_dict['translation'],
                        Quaternion(ego_pose_prev_dict['rotation']),
                        inverse=False
                    )

                    # Global pose of current frame (k) from dataset
                    ego_pose_curr_dict = dict_list[k_dict_idx]['ego_pose']
                    P_dataset_curr_global = transform_matrix(
                        ego_pose_curr_dict['translation'],
                        Quaternion(ego_pose_curr_dict['rotation']),
                        inverse=False
                    )

                    try:
                        P_dataset_prev_global_inv = np.linalg.inv(P_dataset_prev_global)
                        relative_motion_from_dataset = P_dataset_prev_global_inv @ P_dataset_curr_global
                        initial_relative_motions.append(relative_motion_from_dataset)
                    except np.linalg.LinAlgError:
                        print(
                            f"  Warning: Singular matrix for ego_pose at dict_list index {k_dict_idx - 1}. Appending Identity for relative motion.")
                        initial_relative_motions.append(np.eye(4))

            print(f"  Generated {len(initial_relative_motions)} initial guesses from 'ego_pose'.")

        elif args.initial_guess_mode == 'imu':
            print("Using 'IMU' for initial guesses.")

            if not dict_list:
                print("Warning: dict_list is empty. Cannot generate 'IMU' initial guesses.")
            else:
                initial_relative_motions.append(np.eye(4))
                print(f"  Frame 0 (batch index 0): Initial guess is Identity (IMU mode anchor).")

                # For subsequent frames k > 0
                for k in range(1, len(dict_list)):
                    frame_data_prev = dict_list[k - 1]
                    frame_data_curr = dict_list[k]

                    imu_data_prev = frame_data_prev.get(
                        'ego_motion_chassis')
                    imu_data_curr = frame_data_curr.get('ego_motion_chassis')
                    ts_prev = frame_data_prev['sample_timestamp']
                    ts_curr = frame_data_curr['sample_timestamp']

                    if imu_data_prev and imu_data_curr:
                        dt_sec = (ts_curr - ts_prev) / 1e6
                        if dt_sec > 1e-6:  # Check for valid positive time difference
                            relative_motion_imu = integrate_imu_for_relative_motion(imu_data_prev, imu_data_curr,
                                                                                    dt_sec)
                            initial_relative_motions.append(relative_motion_imu)
                        else:
                            print(f"  Warning: dt_sec is {dt_sec} for frame {k}. Appending Identity for IMU guess.")
                            initial_relative_motions.append(np.eye(4))
                    else:
                        print(
                            f"  Warning: Missing IMU data for frame index {k - 1} or {k} in dict_list. Appending Identity.")
                        initial_relative_motions.append(np.eye(4))  # Fallback

                    print(f"  Generated {len(initial_relative_motions)} initial guesses from 'IMU'.")
        else:
            print("Initial guess mode is 'none' or invalid. KISS-ICP will use its default constant velocity model.")
            pass

        try:
            icp_input_pc_list = [frame_dict['lidar_pc_for_icp_ego_i'] for frame_dict in dict_list]
            print(f"Preparing InMemoryDataset for KISS-ICP with {len(icp_input_pc_list)} pre-filtered scans.")

            in_memory_dataset = InMemoryDataset(
                lidar_scans=icp_input_pc_list,  # raw_pc_list,
                gt_relative_poses=gt_relative_poses_arr,
                timestamps=lidar_timestamps,
                sequence_id=f"{scene_name}_icp_run",
                log_dir=log_dir_kiss
            )
            print(f"Created InMemoryDataset with {len(in_memory_dataset)} scans.")

        except Exception as e:
            print(f"Error creating InMemoryDataset: {e}. Skipping refinement.")
            args.icp_refinement = False

        if args.icp_refinement and in_memory_dataset:
            try:
                kiss_config_path = Path('kiss_config.yaml')
                pipeline = OdometryPipeline(dataset=in_memory_dataset, config=kiss_config_path,
                                            initial_guesses_relative=initial_relative_motions)
                print("KISS-ICP pipeline initialized.")
            except Exception as e:
                print(f"Error initializing KISS-ICP: {e}. Skipping refinement.")
                args.icp_refinement = False

        # --- 2. Run KISS-ICP Pipeline ---
        if args.icp_refinement and pipeline is not None:
            kiss_start_time = time.time()
            print("Running KISS-ICP pipeline...")
            try:
                results = pipeline.run()
                # ----------------------------------------------------
                kiss_end_time = time.time()
                print(f"KISS-ICP pipeline finished. Time: {kiss_end_time - kiss_start_time:.2f} sec.")
                print("Pipeline Results:", results)

                # --- Get the calculated poses AFTER run() completes ---
                estimated_poses_kiss = pipeline.poses
                poses_kiss_icp = pipeline.poses

                if not isinstance(estimated_poses_kiss, np.ndarray) or \
                        estimated_poses_kiss.shape != (len(raw_pc_list), 4, 4):
                    print(f"Error: Unexpected pose results shape {estimated_poses_kiss.shape}. "
                          f"Expected ({len(raw_pc_list)}, 4, 4). Skipping refinement application.")
                    args.icp_refinement = False

            except Exception as e:
                print(f"Error during KISS-ICP pipeline run: {e}. Skipping refinement application.")
                args.icp_refinement = False
                import traceback
                traceback.print_exc()

        # --- 3. Apply Refined Poses to Original Point Clouds (Only if ICP succeeded) ---
        if args.icp_refinement and estimated_poses_kiss is not None:
            print("Applying refined poses from KISS-ICP...")
            refined_lidar_pc_list = []

            for idx, points_ego in enumerate(
                    static_points_refined):  # (static_points_mapmos): # (unrefined_pc_ego_list):
                pose = estimated_poses_kiss[idx]
                print(f"Applying refined pose {idx}: {pose}")

                print(f"Points ego shape: {points_ego.shape}")

                points_xyz = points_ego[:, :3]
                points_homo = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
                points_transformed = (pose @ points_homo.T)[:3, :].T

                if points_ego.shape[1] > 3:
                    other_features = points_ego[:, 3:]
                    points_transformed = np.hstack((points_transformed, other_features))

                refined_lidar_pc_list.append(points_transformed)

            """for idx, points_semantic_ego in enumerate(unrefined_sem_pc_ego_list):
                pose = estimated_poses_kiss[idx]
                print(f"Applying refined pose {idx}: {pose}")

                print(f"Points semantic ego shape: {points_semantic_ego.shape}")
                points_xyz = points_semantic_ego[:, :3]
                points_homo = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
                points_transformed = (pose @ points_homo.T)[:3, :].T

                if points_semantic_ego.shape[0] > 3:
                    other_features = points_semantic_ego[:, 3:]
                    points_transformed = np.hstack((points_transformed, other_features))

                refined_lidar_pc_with_semantic_list.append(points_transformed)"""

        # --- 4. Compare KISS-ICP Poses with Ground Truth ---
        if 'gt_relative_poses_arr' in locals() and gt_relative_poses_arr.shape[0] > 0:
            if poses_kiss_icp.shape[0] == gt_relative_poses_arr.shape[0]:
                print("\n--- Comparing KISS-ICP Poses with Ground Truth Poses ---")
                trans_errors = []
                rot_errors_rad = []
                trans_errors_x = []
                trans_errors_y = []
                trans_errors_z = []

                kiss_relative_poses_arr = poses_kiss_icp

                for k_idx in range(kiss_relative_poses_arr.shape[0]):
                    pose_kiss_k = kiss_relative_poses_arr[k_idx]  # Pose of frame k in KISS-ICP's frame 0 system
                    pose_gt_k = gt_relative_poses_arr[k_idx]  # Pose of frame k in dataset's frame 0 system

                    # Translational error
                    t_kiss = pose_kiss_k[:3, 3]
                    t_gt = pose_gt_k[:3, 3]

                    # Translational error vector (GT - Estimated)
                    t_error_vec = t_gt - t_kiss
                    trans_errors_x.append(t_error_vec[0])
                    trans_errors_y.append(t_error_vec[1])
                    trans_errors_z.append(t_error_vec[2])

                    # Overall translational error
                    trans_error = np.linalg.norm(t_error_vec)  # Same as np.linalg.norm(t_gt - t_kiss)
                    trans_errors.append(trans_error)

                    # Rotational error
                    R_kiss = pose_kiss_k[:3, :3]
                    R_gt = pose_gt_k[:3, :3]

                    # Relative rotation: R_error = inv(R_kiss) @ R_gt
                    R_error = R_kiss.T @ R_gt

                    # Angle from rotation matrix trace
                    trace_val = np.trace(R_error)
                    clipped_arg = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
                    rot_error_rad = np.arccos(clipped_arg)
                    rot_errors_rad.append(rot_error_rad)

                avg_trans_error = np.mean(trans_errors)
                median_trans_error = np.median(trans_errors)
                avg_rot_error_deg = np.mean(np.degrees(rot_errors_rad))
                median_rot_error_deg = np.median(np.degrees(rot_errors_rad))

                # Calculate statistics for component-wise errors
                mae_trans_error_x = np.mean(np.abs(trans_errors_x))
                mae_trans_error_y = np.mean(np.abs(trans_errors_y))
                mae_trans_error_z = np.mean(np.abs(trans_errors_z))

                mean_trans_error_x = np.mean(trans_errors_x)
                mean_trans_error_y = np.mean(trans_errors_y)
                mean_trans_error_z = np.mean(trans_errors_z)

                print(f"Sequence: {scene_name}")
                print(f"  Average Translational Error : {avg_trans_error:.4f} m")
                print(f"  Median Translational Error  : {median_trans_error:.4f} m")
                print(f"  Average Rotational Error    : {avg_rot_error_deg:.4f} degrees")
                print(f"  Median Rotational Error     : {median_rot_error_deg:.4f} degrees")
                print(f"  MAE X: {mae_trans_error_x:.4f} m (Mean X Bias: {mean_trans_error_x:+.4f} m)")
                print(f"  MAE Y: {mae_trans_error_y:.4f} m (Mean Y Bias: {mean_trans_error_y:+.4f} m)")
                print(f"  MAE Z: {mae_trans_error_z:.4f} m (Mean Z Bias: {mean_trans_error_z:+.4f} m)")

                fig, axs = plt.subplots(2, 2, figsize=(17, 10))  # Adjusted figsize for 2x2
                fig.suptitle(f'Scene {scene_name}: KISS-ICP vs GT Relative Pose Errors', fontsize=16)

                # Top-left: Overall Translational Error
                axs[0, 0].plot(trans_errors, label="Overall Trans. Error")
                axs[0, 0].set_title('Overall Translational Error')
                axs[0, 0].set_ylabel('Error (m)')
                axs[0, 0].grid(True)
                axs[0, 0].legend()
                axs[0, 0].set_xlabel('Frame Index')

                # Top-right: Overall Rotational Error
                axs[0, 1].plot(np.degrees(rot_errors_rad), label="Overall Rot. Error")
                axs[0, 1].set_title('Overall Rotational Error')
                axs[0, 1].set_ylabel('Error (degrees)')
                axs[0, 1].grid(True)
                axs[0, 1].legend()
                axs[0, 1].set_xlabel('Frame Index')

                # Bottom-left: X and Y Translational Errors
                axs[1, 0].plot(trans_errors_x, label="X Error (GT - Est)", alpha=0.9)
                axs[1, 0].plot(trans_errors_y, label="Y Error (GT - Est)", alpha=0.9)
                axs[1, 0].axhline(0, color='black', linestyle='--', linewidth=0.7, label="Zero Error")
                axs[1, 0].set_title('X & Y Translational Component Errors')
                axs[1, 0].set_xlabel('Frame Index')
                axs[1, 0].set_ylabel('Error (m)')
                axs[1, 0].grid(True)
                axs[1, 0].legend()

                # Bottom-right: Z Translational Error
                axs[1, 1].plot(trans_errors_z, label="Z Error (GT - Est)")
                axs[1, 1].axhline(0, color='black', linestyle='--', linewidth=0.7, label="Zero Error")
                axs[1, 1].set_title('Z Translational Component Error')
                axs[1, 1].set_xlabel('Frame Index')
                axs[1, 1].set_ylabel('Error (m)')
                axs[1, 1].grid(True)
                axs[1, 1].legend()

                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for subtitle

                plot_save_dir = Path(args.save_path) / scene_name / "kiss_icp_logs"
                plot_save_dir.mkdir(parents=True, exist_ok=True)
                plot_filename = plot_save_dir / f"errors_scene_{scene_name}.png"
                plt.savefig(plot_filename)
                print(f"Saved pose error plot to {plot_filename}")

                if args.pose_error_plot:
                    plt.show()
                plt.close(fig)

            else:
                print(f"Warning: Number of KISS-ICP poses ({poses_kiss_icp.shape[0]}) "
                      f"does not match GT relative poses ({gt_relative_poses_arr.shape[0]}). Cannot compare.")

        elif not ('gt_relative_poses_arr' in locals() and gt_relative_poses_arr.shape[0] > 0):
            print("GT relative poses not available for comparison.")

    if not args.icp_refinement:
        print("ICP refinement is OFF. Using unrefined points (in reference ego frame) for aggregation.")
        source_pc_list_all_frames = unrefined_pc_ego_ref_list
        source_pc_sids_list_all_frames = unrefined_pc_ego_ref_list_sensor_ids
    else:
        print("ICP refinement is ON. Using KISS-ICP refined points for aggregation.")
        source_pc_list_all_frames = refined_lidar_pc_list
        source_pc_sids_list_all_frames = static_points_refined_sensor_ids

    #################################### FlexCloud ###################################################################
    lidar_timestamps_flexcloud = [frame_dict['sample_timestamp'] / 1e6 for frame_dict in dict_list]

    # 1. Get your data from KISS-ICP and the dataset
    kiss_icp_poses_save = poses_kiss_icp
    point_clouds_save = source_pc_list_all_frames
    timestamps_save = lidar_timestamps_flexcloud
    timestamps_save_original = timestamps_save.copy()

    gnss_data_save = []
    # Define a constant, low standard deviation for all points
    std_dev_placeholder = np.array([0.1, 0.1, 0.1])
    for frame_dict in dict_list:
        # 1. Get the full 4x4 transformation matrix
        transform_matrix_global_from_ego = frame_dict['global_from_ego_i']

        # 2. Extract just the position (x, y, z) from the last column
        position = transform_matrix_global_from_ego[:3, 3]

        # 3. Combine the position and standard deviation into a single 6-element array
        gnss_entry = np.hstack((position, std_dev_placeholder))

        # 4. Append the correct 6-element array to the list
        gnss_data_save.append(gnss_entry)

    if len(gnss_data_save) > 1:
        print("Padding data lists to ensure interpolation coverage...")

        # --- Pad the beginning ---
        first_timestamp_sec = timestamps_save[0]
        last_pose_data = gnss_data_save[-1]

        # Add two points before the start
        for i in range(1, 3): # Creates points for T-0.2s and T-0.1s
            padded_timestamp = first_timestamp_sec - (i * 0.1)

            # Get the closest real pose from the dataset
            ego_pose_rec = trucksc.getclosest("ego_pose", padded_timestamp * 1e6)
            position = np.array(ego_pose_rec['translation'])
            padded_pose_data = np.hstack((position, std_dev_placeholder))

            # Insert into all three lists
            gnss_data_save.insert(0, padded_pose_data)
            timestamps_save.insert(0, padded_timestamp)

        # --- Pad the end ---
        last_timestamp_sec = timestamps_save[-1]

        # Add two points after the end
        for i in range(1, 4):  # Creates points for T+0.1s and T+0.2s
            padded_timestamp = last_timestamp_sec + (i * 0.1)

            # Get the closest real pose from the dataset
            #ego_pose_rec = trucksc.getclosest("ego_pose", padded_timestamp * 1e6)
            #position = np.array(ego_pose_rec['translation'])
            #padded_pose_data = np.hstack((position, std_dev_placeholder))

            # Append to all three lists
            #gnss_data_save.append(padded_pose_data)
            gnss_data_save.append(last_pose_data)
            timestamps_save.append(padded_timestamp)

        print(f"Padded all lists. New total items: {len(timestamps_save)}")

    config_path_select_keyframes = "/flexcloud/config/select_keyframes.yaml"
    config_path_pcd_georef = "/flexcloud/config/pcd_georef.yaml"

    # 2. Define temporary directories for I/O
    io_dir = args.scene_io_dir
    save_dir_flexcloud = os.path.join(io_dir, "flexcloud_io")
    pos_dir = os.path.join(save_dir_flexcloud, "gnss_poses")
    odom_path = os.path.join(save_dir_flexcloud, "odom/slam_poses.txt")
    pcd_dir = os.path.join(save_dir_flexcloud, "point_clouds")
    output_dir = os.path.join(save_dir_flexcloud, "output_keyframes")
    pcd_dir_transformed = os.path.join(save_dir_flexcloud, "pcd_transformed")

    os.makedirs(pos_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pcd_dir_transformed, exist_ok=True)
    os.makedirs(os.path.dirname(odom_path), exist_ok=True)

    # 3. Save your in-memory data to the temporary files
    save_poses_to_kitti_format(kiss_icp_poses_save, odom_path)
    save_gnss_to_directory(gnss_data_save, timestamps_save, pos_dir)
    save_pcds_to_directory(point_clouds_save, timestamps_save_original, pcd_dir)

    ######################### Save all data ##########################
    context_file_path = os.path.join(args.scene_io_dir, "preprocessed_data.npz")
    print(f"Saving data to {context_file_path}")

    np.savez_compressed(
        context_file_path,
        # --- Essential Data ---
        source_pc_list_all_frames=np.array(source_pc_list_all_frames, dtype=object),
        # Use dtype=object for lists of arrays
        source_pc_sids_list_all_frames=np.array(source_pc_sids_list_all_frames, dtype=object),
        dict_list=np.array(dict_list, dtype=object),
        poses_kiss_icp=poses_kiss_icp,  # Can be None if ICP was skipped
        # --- Config/Args needed by Part 3 ---
        config=np.array(config, dtype=object),
        truckscenesyaml=np.array(truckscenesyaml, dtype=object),
        learning_map=np.array(learning_map, dtype=object),
        learning_id_to_name=np.array(learning_id_to_name, dtype=object),
        category_name_to_learning_id=np.array(category_name_to_learning_id, dtype=object),
        pc_range=np.array(pc_range, dtype=object),
        voxel_size=voxel_size,
        occ_size=np.array(occ_size, dtype=object),
        save_path=save_path,
        scene_name=scene_name,
        args_dict=np.array(vars(args), dtype=object),  # Save args as a dictionary
        # --- Constants ---
        FREE_LEARNING_INDEX=FREE_LEARNING_INDEX,
        BACKGROUND_LEARNING_INDEX=BACKGROUND_LEARNING_INDEX,
        sensors=np.array(sensors, dtype=object),
        cameras=np.array(cameras, dtype=object),
        sensor_max_ranges_arr=sensor_max_ranges_arr,
        self_range=np.array(self_range, dtype=object)
    )
    print(f"\n--- Part 1 Complete ---")
    print(f"Saved pipeline context to {context_file_path}")
    print(f"FlexCloud inputs are ready in: {save_dir_flexcloud}")

    print("Creating success flag for Part 1...")
    flag_file_path = os.path.join(args.scene_io_dir, "part1_success.flag")
    with open(flag_file_path, 'w') as f:
        f.write('success')



# Main entry point of the script
if __name__ == '__main__':
    # argument parsing allows users to customize parameters when running the script
    from argparse import ArgumentParser

    parse = ArgumentParser()

    ############################## Define Command-Line Arguments #########################################################################
    parse.add_argument('--dataset', type=str, default='truckscenes')  # Dataset selection with default: "truckScenes"
    parse.add_argument('--config_path', type=str,
                       default='config_truckscenes.yaml')  # Configuration file path with default: "config.yaml"
    parse.add_argument('--split', type=str,
                       default='train')  # data split, default: "train", options: "train", "val", "all"
    parse.add_argument('--save_path', type=str,
                       default='./data/GT_occupancy/')  # save path, default: "./data/GT/GT_occupancy"
    parse.add_argument('--scene_io_dir', type=str, default='./data/scene_io/')
    parse.add_argument('--idx', type=int,
                       default=0)  # start indice, default: 0, determines range of sequences to process
    parse.add_argument('--dataroot', type=str,
                       default='./data/truckscenes/')  # data root path, default: "./data/truckScenes
    parse.add_argument('--version', type=str, default='v1.0-trainval')
    parse.add_argument('--label_mapping', type=str,
                       default='truckscenes.yaml')  # YAML file containing label mappings, default: "truckscenes.yaml"

    parse.add_argument('--load_mode', type=str, default='pointwise')  # pointwise or rigid

    #parse.add_argument('--static_map_keyframes_only', action='store_true',
              #         help='Build the final static map using only keyframes (after ICP, if enabled, ran on all frames).')
    #parse.add_argument('--dynamic_map_keyframes_only', action='store_true',
                      # help='Aggregate dynamic object points using only segments from keyframes..')

    ####################### Kiss-ICP refinement ##########################################
    parse.add_argument('--icp_refinement', action='store_true', help='Enable ICP refinement')
    parse.add_argument('--initial_guess_mode', type=str, default='none', choices=['none', 'ego_pose', 'imu'])
    parse.add_argument('--pose_error_plot', action='store_true', help='Plot pose error')

    ####################### MapMOS ########################################################
    parse.add_argument('--run_mapmos', action='store_true', help='Enable the MapMOS pipeline to refine the static map.')

    ######################## Filtering ####################################################
    parse.add_argument('--filter_mode', type=str, default='none', choices=['none', 'sor', 'ror', 'both'],
                       help='Noise filtering method to apply before meshing')

    parse.add_argument('--filter_lidar_intensity', action='store_true', help='Enable lidar intensity filtering')

    parse.add_argument('--filter_raw_pc', action='store_true', help='Enable raw pc filtering')
    parse.add_argument('--filter_static_pc', action='store_true', help='Enable static pc filtering')
    #parse.add_argument('--filter_aggregated_static_pc', action='store_true',
     #                  help='Enable aggregated static pc filtering')
    #parse.add_argument('--filter_combined_static_dynamic_pc', action='store_true',
     #                  help='Enable combined static and dynamic pc filtering')

    ########################## Visualization ################################################
    parse.add_argument('--vis_raw_pc', action='store_true', help='Enable raw pc visualization')
    parse.add_argument('--vis_static_pc', action='store_true', help='Enable static pc visualization')
    parse.add_argument('--vis_static_pc_global', action='store_true', help='Enable static pc global visualization')
    parse.add_argument('--vis_lidar_intensity_filtered', action='store_true',
                       help='Enable lidar intensity filtered visualization')

    parse.add_argument('--vis_aggregated_static_ego_i_pc', action='store_true',
                       help='Enable aggregated static ego i pc visualization')
    parse.add_argument('--vis_aggregated_static_ego_ref_pc', action='store_true',
                       help='Enable aggregated static ego ref pc visualization')
    parse.add_argument('--vis_aggregated_static_global_pc', action='store_true',
                       help='Enable aggregated static global pc visualization')
    parse.add_argument('--vis_aggregated_raw_pc_ego_i', action='store_true',
                       help='Enable aggregated raw pc ego i visualization')
    #parse.add_argument('--vis_static_frame_comparison_kiss_refined', action='store_true',
     #                  help='Enable static frame comparison kiss refinement')
    #parse.add_argument('--vis_aggregated_static_kiss_refined', action='store_true',
     #                  help='Enable aggregated static kiss refinement')


    args = parse.parse_args()

    if args.dataset == 'truckscenes':  # check dataset type
        # Load the truckScenes dataset
        truckscenes = TruckScenes(version=args.version,
                                  dataroot=args.dataroot,
                                  verbose=True)
    else:
        raise NotImplementedError

    # load config with hyperparameters and settings
    with open(args.config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    # load learning map to map raw semantic labels to learning labels
    label_mapping = args.label_mapping
    with open(label_mapping, 'r') as stream:
        truckscenesyaml = yaml.safe_load(stream)

    print('processing sequence:', args.idx)
    main(truckscenes, indice=args.idx,
         truckscenesyaml=truckscenesyaml, args=args, config=config)
import numpy as np
import os
from tqdm import tqdm
import torch
import sys
import time
import yaml
import json
import shutil
import os.path as osp
from pathlib import Path
from truckscenes.utils import splits
from truckscenes.truckscenes import TruckScenes
from utils.pointcloud_loading import load_lidar_entries, group_entries, get_rigid_fused_pointcloud, get_pointwise_fused_pointcloud
from utils.bbox_utils import transform_boxes_to_ego
import open3d as o3d
from mmcv.ops.points_in_boxes import (points_in_boxes_cpu, points_in_boxes_all)
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
from truckscenes.utils.geometry_utils import transform_matrix
from utils.visualization import visualize_pointcloud_bbox, visualize_pointcloud, visualize_point_cloud_comparison, calculate_and_plot_pose_errors
from utils.pointcloud_processing import denoise_pointcloud, denoise_pointcloud_advanced, get_weather_intensity_filter_mask, integrate_imu_for_relative_motion
from utils.geometry_utils import transform_points, transform_imu_to_ego
from utils.constants import *
from kiss_icp.pipeline import OdometryPipeline
from mapmos.pipeline import MapMOSPipeline
from utils.custom_datasets import InMemoryDataset, InMemoryDatasetMapMOS
from utils.data_utils import save_gnss_to_directory, save_poses_to_kitti_format, save_pcds_to_directory, save_gnss_to_single_file, parse_single_annotation_file, save_pointcloud_for_annotation

def main(trucksc, indice, truckscenesyaml, args, config):
    ################ Set device ################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ########################## Extract parameters from config and args #############################################
    save_path = args.save_path  # Directory where processed data will be saved
    data_root = args.dataroot  # Root directory of dataset
    learning_map = truckscenesyaml['learning_map']  # dictionary that maps raw semantic labels to learning labels
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

    # Enlarged bboxes
    percentage_factor = config['bbox_enlarge_percentage_factor']
    max_absolute_increase_m = config['bbox_enlarge_max_increase_m']  # The maximum total increase for any dimension (e.g., 25cm per side)

    # Max enlarged bboxes
    percentage_factor_max = config['bbox_max_enlarge_percentage_factor']
    fixed_increase_m = config['bbox_max_enlarge_fixed_increase_m']
    special_scales_max_enlarged_bboxes = config['bbox_special_width_scales']

    # ICP input pc velocity threshold bboxes to keep
    VELOCITY_THRESHOLD_M_S = config['moving_object_velocity_threshold_ms']

    # Weight path for mapmos
    weights_path_mapmos = Path(config['weights_path_mapmos'])
    if not weights_path_mapmos:
        raise ValueError("MapMOS weights path is not specified in the config file.")

    # FlexCloud
    std_dev_placeholder = np.array(config['flexcloud_gnss_std_dev_placeholder'])
    imu_freq = config['rosbag_imu_frequency_hz']


    ################################ Load config for used sensors for aggregation #################################
    sensors = config['sensors']
    print(f"Lidar sensors: {sensors}")

    ########################### Generate list for sensor range ###################
    sensor_ranges_map = config['lidar_sensor_max_ranges']
    sensors_max_range_list = [sensor_ranges_map[s] for s in sensors]
    print(f"Lidar sensors max range: {sensors_max_range_list}")
    sensor_max_ranges_arr = np.array(sensors_max_range_list, dtype=np.float64)

    print(f"Lidar sensors max range: {sensor_max_ranges_arr}")

    ############################## Load config for cameras for camera visibility mask ###############################
    cameras = config['cameras']
    print(f"Cameras: {cameras}")

    ############################## Get IMU calibration data ######################################
    sensor_token_imu = truckscenes.field2token('sensor', 'channel', 'XSENSE_CHASSIS')[0]

    calibrated_sensor_record_imu = None
    for cs_record in truckscenes.calibrated_sensor:
        if cs_record['sensor_token'] == sensor_token_imu:
            calibrated_sensor_record_imu = cs_record
            break

    print(f"Calibrated imu chassis sensor record: {calibrated_sensor_record_imu}")

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
    dict_list_for_saving = []
    dict_list_rosbag = []

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
        scene_terminal_list = config['manual_annotation_scenes'].get(args.version, [])

        ################################# Uncomment if you need to save pointclouds for annotation ###########################

        if args.save_sensor_fused_pc:
            if indice in scene_terminal_list:
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
                save_pointcloud_for_annotation(sensor_fused_pc.points.T, annotation_data_save_path)

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

        increase_from_percentage = dims * (percentage_factor_max - 1.0)

        final_increase = np.maximum(increase_from_percentage, fixed_increase_m)

        dims_filter = dims + final_increase

        # enlarge width of cars and trucks as mirrors often not included in bounding boxes --> avoid artifacts

        for index, cat in enumerate(original_object_category_names):
            if cat == 'vehicle.car':
                scale = special_scales_max_enlarged_bboxes[cat]
                special_width = dims[index, 0] * scale
                dims_filter[index, 0] = np.maximum(dims_filter[index, 0], special_width)
            elif cat == 'vehicle.truck':
                scale = special_scales_max_enlarged_bboxes[cat]
                special_width = dims[index, 0] * scale
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
        points_to_check_tensor_gpu = torch.from_numpy(points_to_check_np).float().to(device)

        boxes_to_check_np = gt_bbox_3d_points_in_boxes_cpu_enlarged[np.newaxis, :]
        boxes_to_check_tensor_gpu = torch.from_numpy(boxes_to_check_np).float().to(device)

        points_in_boxes_gpu = points_in_boxes_all(points_to_check_tensor_gpu,
                                              boxes_to_check_tensor_gpu)  # use function to identify which points belong to which bounding box

        points_in_boxes = points_in_boxes_gpu.cpu()

        #################################### Mask for the ego vehicle itself #######################################
        points_xyz = sensor_fused_pc.points.T[:, :3]

        # Create a mask for points outside the ego vehicle bounding box
        inside_x = torch.from_numpy(points_xyz[:, 0] >= x_min_self) & torch.from_numpy(points_xyz[:, 0] <= x_max_self)
        inside_y = torch.from_numpy(points_xyz[:, 1] >= y_min_self) & torch.from_numpy(points_xyz[:, 1] <= y_max_self)
        inside_z = torch.from_numpy(points_xyz[:, 2] >= z_min_self) & torch.from_numpy(points_xyz[:, 2] <= z_max_self)

        inside_ego_mask = (inside_x & inside_y & inside_z).cpu().numpy()

        ############################ Prepare point clouds for kiss-icp #################################

        is_box_moving_mask = np.zeros(len(boxes_ego), dtype=bool)
        annotated_boxes_indices = [idx for idx, box in enumerate(boxes_ego) if box.token is not None]
        box_annotation_tokens = [boxes_ego[i].token for i in annotated_boxes_indices]

        if box_annotation_tokens:
            box_velocities = np.array([trucksc.box_velocity(token) for token in box_annotation_tokens])
            box_speeds = np.linalg.norm(np.nan_to_num(box_velocities, nan=0.0), axis=1)
            are_annotated_boxes_moving = box_speeds > VELOCITY_THRESHOLD_M_S
            np.put(is_box_moving_mask, annotated_boxes_indices, are_annotated_boxes_moving)

        points_in_moving_boxes_mask_np = torch.zeros(sensor_fused_pc.points.shape[1], dtype=torch.bool).numpy()
        if np.any(is_box_moving_mask):
            points_in_moving_boxes_mask_torch = points_in_boxes[0][:, is_box_moving_mask].any(dim=1).cpu().numpy()

        points_to_remove_mask_np = inside_ego_mask | points_in_moving_boxes_mask_np

        initial_keep_for_icp_mask_np = ~points_to_remove_mask_np

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
        dynamic_mask = point_box_mask.any(dim=-1).cpu().numpy()  # shape: (Npoints,)

        num_dynamic_points = dynamic_mask.sum().item()
        print(f"Number of dynamic points: {num_dynamic_points}")

        # Get static mask (inverse)
        static_mask = ~dynamic_mask

        num_ego_points = inside_ego_mask.sum().item()
        print(f"Number of points on ego vehicle: {num_ego_points}")

        ################################# Dynamic point filtering of static map with max enlarged boxes##################
        boxes_to_check_max_enlarged_tensor_gpu = torch.from_numpy(gt_bbox_3d_points_in_boxes_cpu_max_enlarged[np.newaxis, :]).to(device).float()

        points_in_boxes_max_enlarged_gpu = points_in_boxes_all(
            points_to_check_tensor_gpu,
            boxes_to_check_max_enlarged_tensor_gpu
        )

        points_in_boxes_max_enlarged = points_in_boxes_max_enlarged_gpu.cpu()

        dynamic_mask_max_enlarged = points_in_boxes_max_enlarged[0].any(dim=-1).cpu().numpy()

        only_in_max_enlarged_mask = dynamic_mask_max_enlarged & ~dynamic_mask
        num_points_max_enlarged = only_in_max_enlarged_mask.sum().item()

        print(f"Number of points captured only by the max enlarged box: {num_points_max_enlarged}")

        pc_ego_unfiltered = sensor_fused_pc.points.T
        pc_ego_unfiltered_sensors = sensor_ids_points.copy().T
        ########################################## Lidar intensity filtering #######################################
        if args.filter_lidar_intensity:
            print(f'Shape of pc_ego before weather filtering: {pc_ego_unfiltered_sensors.shape}')
            intensity_keep_mask_ego = get_weather_intensity_filter_mask(
                point_cloud=pc_ego_unfiltered,
                weather_condition=weather,
                intensity_thresh=intensity_threshold,
                distance_thresh=distance_intensity_threshold,
                keep_ground_points=True,
                ground_z_min=ground_z_min_threshold,
                ground_z_max=ground_z_max_threshold
            )

            ################################# Visualize if specified in arguments ##################################
            if args.vis_lidar_intensity_filtered:
                pc_ego_intensity_filtered = pc_ego_unfiltered[intensity_keep_mask_ego]
                print(f"Shape of pc_ego after weather filtering: {pc_ego_intensity_filtered.shape}")
                visualize_pointcloud_bbox(pc_ego_intensity_filtered,
                                          boxes=boxes_ego,
                                          title=f"Fused filtered sensor PC after lidar intensity filtering + BBoxes + Ego BBox - Frame {i}",
                                          self_vehicle_range=self_range,
                                          vis_self_vehicle=True)
            #######################################################################################################
        else:
            print(f"No lidar intensity filtering according to arguments")
            intensity_keep_mask_ego = np.ones(sensor_fused_pc.points.shape[1], dtype=bool)

        ###################################### MapMOS inut ################################################
        mapmos_pc_mask = intensity_keep_mask_ego | dynamic_mask | dynamic_mask_max_enlarged
        pc_for_mapmos = pc_ego_unfiltered[mapmos_pc_mask]
        pc_for_mapmos_sensors = pc_ego_unfiltered_sensors[mapmos_pc_mask]

        ###################################### Generate labels for mapmos as gt ####################################
        dynamic_mask_mapmos = dynamic_mask | inside_ego_mask | dynamic_mask_max_enlarged

        dynamic_labels_for_mapmos = dynamic_mask_mapmos[mapmos_pc_mask]

        labels_for_mapmos = dynamic_labels_for_mapmos.astype(np.int32).reshape(-1, 1)
        assert pc_for_mapmos.shape[0] == labels_for_mapmos.shape[0], \
            "Mismatch between the number of points for MapMOS and the number of labels."

        print(f"Created {pc_for_mapmos.shape[0]} points for MapMOS input with {np.sum(labels_for_mapmos)} dynamic labels.")

        ############################### Combine masks #########################################################
        pre_denoising_static_mask = (
                ~dynamic_mask &
                ~inside_ego_mask &
                ~dynamic_mask_max_enlarged &
                intensity_keep_mask_ego
        )
        pc_for_denoising = pc_ego_unfiltered[pre_denoising_static_mask]
        sensors_for_denoising = pc_ego_unfiltered_sensors[pre_denoising_static_mask]

        kept_indices = np.arange(pc_for_denoising.shape[0])  # Default is to keep all
        ############################# Apply filtering to static points in ego frame #################################
        if args.filter_static_pc and args.filter_mode != 'none':
            print(f"Applying geometric filter ('{args.filter_mode}') to {pc_for_denoising.shape[0]} static candidates...")
            pcd_static = o3d.geometry.PointCloud()
            pcd_static.points = o3d.utility.Vector3dVector(pc_for_denoising[:, :3])

            _, kept_indices = denoise_pointcloud_advanced(
                pcd=pcd_static,
                filter_mode=args.filter_mode,
                config=config,
                location_msg="final static points"
            )

        final_static_pc = pc_for_denoising[kept_indices]
        final_static_pc_sensors = sensors_for_denoising[kept_indices]
        print(f"Final static map has {final_static_pc.shape[0]} points after all filters.")

        final_static_points_keep_mask = np.zeros(sensor_fused_pc.points.shape[1], dtype=bool)
        absolute_indices_into_denoiser = np.where(pre_denoising_static_mask)[0]
        absolute_indices_kept = absolute_indices_into_denoiser[kept_indices]
        final_static_points_keep_mask[absolute_indices_kept] = True

        final_static_map_mask = final_static_points_keep_mask[mapmos_pc_mask]

        assert final_static_map_mask.shape[0] == pc_for_mapmos.shape[0], \
            f"Shape mismatch! `pc_for_mapmos` has {pc_for_mapmos.shape[0]} points, but `final_static_map_mask` has {final_static_map_mask.shape[0]}."

        ############################ Visualization #############################################################
        if args.vis_static_pc and args.filter_static_pc and i % 5 == 0:
            visualize_pointcloud_bbox(final_static_pc,
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

        points_in_ref_frame = transform_points(final_static_pc, ego_ref_from_ego_i)
        print(f"Frame {i}: Transformed static points to ref ego. Shape: {points_in_ref_frame.shape}")

        points_in_global_frame = transform_points(final_static_pc, global_from_ego_i)
        print(f"Frame {i}: Transformed static points to global. Shape: {points_in_global_frame.shape}")

        if args.vis_static_pc_global:
            visualize_pointcloud(points_in_ref_frame, title=f"Fused sensor PC in world coordinates - Frame {i}")

        pc_ego_i_save = final_static_pc.copy()
        print(f"Frame {i}: Static points in ego frame shape: {pc_ego_i_save.shape}")
        pc_ego_ref_save = points_in_ref_frame.copy()
        pc_global_save = points_in_global_frame.copy()

        ################## Save all information into a dict  ########################
        ref_sd = trucksc.get('sample_data', sample['sample_data_token'])

        imu_record_raw = trucksc.getclosest('ego_motion_chassis', sample['timestamp'])

        imu_record_transformed = transform_imu_to_ego(imu_record_raw, calibrated_sensor_record_imu)

        frame_dict = {
            "sample_timestamp": sample['timestamp'],
            "scene_name": scene_name,
            "sample_token": trucksc.get('sample', ref_sd['sample_token'])['token'],
            "is_key_frame": sample['is_key_frame'],
            "converted_object_category": converted_object_category,
            "boxes_ego": boxes_ego,
            "ego_motion_chassis": imu_record_transformed,
            "gt_bbox_3d_points_in_boxes_cpu_max_enlarged": gt_bbox_3d_points_in_boxes_cpu_max_enlarged,
            "gt_bbox_3d_points_in_boxes_cpu_enlarged": gt_bbox_3d_points_in_boxes_cpu_enlarged,
            "gt_bbox_3d_overlap_enlarged": gt_bbox_3d_overlap_enlarged,
            "gt_bbox_3d_unmodified": gt_bbox_3d_unmodified,
            "object_tokens": object_tokens,
            "object_points_list": object_points_list,
            "object_points_list_sensor_ids": objects_points_list_sensor_ids,
            #"raw_lidar_ego": sensor_fused_pc.points.T,
            #"raw_lidar_ego_sensor_ids": sensor_ids_points.T,
            "mapmos_per_point_labels": labels_for_mapmos,
            "mapmos_pc": pc_for_mapmos,
            "mapmos_pc_sensors": pc_for_mapmos_sensors,
            "final_static_mask": final_static_map_mask,
            "lidar_pc_ego_i": pc_ego_i_save,
            "lidar_pc_ego_sensor_ids": pc_ego_unfiltered_sensors,
            "lidar_pc_ego_ref": pc_ego_ref_save,
            #"lidar_pc_global": pc_global_save,
            "ego_pose": ego_pose_i,  # Current frame's ego pose dictionary
            "ego_ref_from_ego_i": ego_ref_from_ego_i,
            "global_from_ego_i": global_from_ego_i,
            "lidar_pc_for_icp_ego_i": pc_for_icp
        }

        frame_dict_for_saving = {
            "sample_timestamp": frame_dict["sample_timestamp"],
            "scene_name": frame_dict["scene_name"],
            "sample_token": frame_dict["sample_token"],
            "is_key_frame": frame_dict["is_key_frame"],
            "converted_object_category": frame_dict["converted_object_category"],
            "boxes_ego": frame_dict["boxes_ego"],
            "object_tokens": frame_dict["object_tokens"],
            "object_points_list": frame_dict["object_points_list"],
            "object_points_list_sensor_ids": frame_dict["object_points_list_sensor_ids"],
            "ego_ref_from_ego_i": frame_dict["ego_ref_from_ego_i"],
            # Ground truth boxes needed by part3
            "gt_bbox_3d_unmodified": frame_dict["gt_bbox_3d_unmodified"],
            "gt_bbox_3d_points_in_boxes_cpu_enlarged": frame_dict["gt_bbox_3d_points_in_boxes_cpu_enlarged"],
            "gt_bbox_3d_overlap_enlarged": frame_dict["gt_bbox_3d_overlap_enlarged"],
            # Object points are also very large, so we exclude them.
            # Part 3 re-aggregates them anyway.
        }

        ########################## Save dictionary into a list ###############################
        dict_list.append(frame_dict)
        dict_list_for_saving.append(frame_dict_for_saving)

        if args.save_rosbag_data:
            frame_dict_rosbag = {
                "sample_timestamp": sample['timestamp'],
                "lidar_pc_for_icp_ego_i": pc_for_icp,
                "imu_record": imu_record_transformed,
                "ego_pose": ego_pose_i,
            }

            dict_list_rosbag.append(frame_dict_rosbag)

    ################# Prepare Lists for Static Scene Points (in Reference Ego Frame) ########################.

    print("Extracting static points previously transformed to reference ego frame...")

    ################## static point cloud in ego i frame ################################
    unrefined_pc_ego_list = [frame_dict['lidar_pc_ego_i'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_pc_ego_list)} static point clouds (in ego i frame).")
    unrefined_pc_ego_list_sensor_ids = [frame_dict['lidar_pc_ego_sensor_ids'] for frame_dict in dict_list]


    ######################## Visualization #################################################
    if args.vis_aggregated_static_ego_i_pc:
        pc_ego_combined_draw = np.concatenate(unrefined_pc_ego_list, axis=0)
        print(f"Pc ego i combined shape: {pc_ego_combined_draw.shape}")
        pc_ego_to_draw = o3d.geometry.PointCloud()
        pc_coordinates = pc_ego_combined_draw[:, :3]
        pc_ego_to_draw.points = o3d.utility.Vector3dVector(pc_coordinates)
        o3d.visualization.draw_geometries([pc_ego_to_draw], window_name="Combined static point clouds (in ego i frame)")

    #################### static point cloud in ego ref frame #############################
    unrefined_pc_ego_ref_list = [frame_dict['lidar_pc_ego_ref'] for frame_dict in dict_list]
    print(f"Extracted {len(unrefined_pc_ego_ref_list)} static point clouds (in ego ref frame).")
    unrefined_pc_ego_ref_list_sensor_ids = [frame_dict['lidar_pc_ego_sensor_ids'] for frame_dict in dict_list]

    ###################### Visualization ##################################################
    if args.vis_aggregated_static_ego_ref_pc:
        pc_ego_ref_combined_draw = np.concatenate(unrefined_pc_ego_ref_list, axis=0)
        print(f"Pc ego ref combined shape: {pc_ego_ref_combined_draw.shape}")
        pc_ego_ref_to_draw = o3d.geometry.PointCloud()
        pc_ego_ref_coordinates = pc_ego_ref_combined_draw[:, :3]
        pc_ego_ref_to_draw.points = o3d.utility.Vector3dVector(pc_ego_ref_coordinates)
        o3d.visualization.draw_geometries([pc_ego_ref_to_draw],
                                          window_name="Combined static point clouds (in ego ref frame)")

    """######################### static point cloud in world frame #############################
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
                                          window_name="Combined static and dynamic point clouds (in ego i frame)")"""

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

    ###############################################################################################
    ######################## Calculate initial guesses ###########################################
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
        log_dir_mapmos = osp.join(save_path, scene_name, "mapmos_results")

        mapmos_labels_per_scan = [frame_dict['mapmos_per_point_labels'] for frame_dict in dict_list]
        final_static_keep_mask = [frame_dict['final_static_mask'] for frame_dict in dict_list]

        mapmos_pc_list = [frame_dict['mapmos_pc'] for frame_dict in dict_list]
        mapmos_pc_sensor_list = [frame_dict['mapmos_pc_sensors'] for frame_dict in dict_list]

        try:
            print("Initializing InMemoryDatasetMapMOS...")

            in_memory_dataset_mapmos = InMemoryDatasetMapMOS(
                lidar_scans=mapmos_pc_list,
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
                    log_dir=log_dir_mapmos,
                    visualize=False,
                    save_ply=args.save_mapmos_pc,
                    save_kitti=False,
                    n_scans=-1,
                    jump=0,
                    initial_guesses_relative=initial_relative_motions
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
                original_points_in_frame = mapmos_pc_list[frame_idx]
                original_points_in_frame_sensor_ids = mapmos_pc_sensor_list[frame_idx]

                # Condition 1: Point was considered STATIC in your input GT to MapMOS
                input_gt_is_static_mask = final_static_keep_mask[frame_idx]
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
        log_dir_kiss = osp.join(save_path, scene_name, "kiss_icp_results")

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
                kiss_config_path = Path(config['kiss_icp_config_path'])
                pipeline = OdometryPipeline(dataset=in_memory_dataset, config=kiss_config_path, log_dir=log_dir_kiss,
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
                        estimated_poses_kiss.shape != (len(icp_input_pc_list), 4, 4):
                    print(f"Error: Unexpected pose results shape {estimated_poses_kiss.shape}. "
                          f"Expected ({len(icp_input_pc_list)}, 4, 4). Skipping refinement application.")
                    args.icp_refinement = False

            except Exception as e:
                print(f"Error during KISS-ICP pipeline run: {e}. Skipping refinement application.")
                args.icp_refinement = False
                import traceback
                traceback.print_exc()

    ################################################################################################################

    #################################### Saving data for FlexCloud ################################################
    if args.use_flexcloud:
        if not args.icp_refinement:
            print("Data for flexcloud cannot be saved as no ICP refinement is enabled.")
            sys.exit()
        print("Saving data for flexcloud....")

        if args.icp_refinement and estimated_poses_kiss is not None:
            print("Applying refined poses from KISS-ICP...")
            refined_lidar_pc_list = []

            for idx, points_ego in enumerate(
                    static_points_refined):
                pose = estimated_poses_kiss[idx]
                # print(f"Applying refined pose {idx}: {pose}")

                # print(f"Points ego shape: {points_ego.shape}")

                points_xyz = points_ego[:, :3]
                points_homo = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1))))
                points_transformed = (pose @ points_homo.T)[:3, :].T

                if points_ego.shape[1] > 3:
                    other_features = points_ego[:, 3:]
                    points_transformed = np.hstack((points_transformed, other_features))

                refined_lidar_pc_list.append(points_transformed)

        if not args.icp_refinement:
            print(
                "ICP refinement is OFF. Using mapmos points (if enabled) or unrefined points (if mapmos unabled) (in reference ego frame) for aggregation.")
            source_pc_list_all_frames_to_transform = static_points_refined
            source_pc_list_all_frames = []
            for index_transform, pc_to_transform in enumerate(source_pc_list_all_frames_to_transform):
                pose = gt_relative_poses_list[index_transform]
                transformed_points_ego_ref = transform_points(pc_to_transform, pose)
                source_pc_list_all_frames.append(transformed_points_ego_ref)

            source_pc_sids_list_all_frames = static_points_refined_sensor_ids
        else:
            print("ICP refinement is ON. Using KISS-ICP refined points for aggregation.")
            source_pc_list_all_frames = refined_lidar_pc_list
            source_pc_sids_list_all_frames = static_points_refined_sensor_ids

        #################################### Filtering based on if only keyframes should be used ###########################
        print(f"Static map aggregation: --static_map_keyframes_only is {args.static_map_keyframes_only}")

        lidar_pc_list_for_concat = []
        lidar_pc_sids_list_for_concat = []

        for idx, frame_info in enumerate(dict_list):
            is_key = frame_info['is_key_frame']
            include_in_static_map = True
            if args.static_map_keyframes_only and not is_key:
                include_in_static_map = False

            if include_in_static_map:
                print(f"  Including frame {idx} (Keyframe: {is_key}) in static map aggregation.")
                if idx < len(source_pc_list_all_frames) and source_pc_list_all_frames[idx].shape[0] > 0:
                    lidar_pc_list_for_concat.append(source_pc_list_all_frames[idx])
                    if idx < len(source_pc_sids_list_all_frames) and source_pc_sids_list_all_frames[idx].shape[0] > 0:
                        lidar_pc_sids_list_for_concat.append(source_pc_sids_list_all_frames[idx])
                    elif source_pc_list_all_frames[idx].shape[
                        0] > 0:
                        print(
                            f"Warning: Frame {idx} has {source_pc_list_all_frames[idx].shape[0]} static points but missing/empty SIDs.")
            else:
                print(
                    f"  Skipping frame {idx} (Keyframe: {is_key}) for static map aggregation due to --static_map_keyframes_only.")

        ################################### Concatenating the static point list ##############################################

        if lidar_pc_list_for_concat:
            print(f"Concatenating pc from {len(lidar_pc_list_for_concat)} frames")
            lidar_pc_final_global = np.concatenate(lidar_pc_list_for_concat, axis=0)
            print(f"Concatenated refined static global points. Shape: {lidar_pc_final_global.shape}")
        else:
            sys.exit()

        if lidar_pc_sids_list_for_concat:
            print(f"Concatenating pc sensor ids from {len(lidar_pc_sids_list_for_concat)} frames")
            lidar_pc_final_global_sensor_ids = np.concatenate(lidar_pc_sids_list_for_concat, axis=0)
            print(
                f"Concatenated refined static global point sensor ids. Shape: {lidar_pc_final_global_sensor_ids.shape}")
        else:
            sys.exit()

        assert lidar_pc_final_global.shape[0] == lidar_pc_final_global_sensor_ids.shape[0]

        io_dir = args.scene_io_dir
        save_dir_flexcloud = os.path.join(args.scene_io_dir, "flexcloud_io")
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

        final_pc_path = os.path.join(save_dir_flexcloud, "aggregated_cloud.pcd")
        print(f"Shape of input pc for flexcloud is: {lidar_pc_final_global.shape}")

        pcd_agg = o3d.geometry.PointCloud()
        pcd_agg.points = o3d.utility.Vector3dVector(lidar_pc_final_global[:, :3])
        o3d.io.write_point_cloud(final_pc_path, pcd_agg)

        visualize_pointcloud(lidar_pc_final_global, title=f"Aggregated Refined Point Cloud")

        lidar_timestamps_flexcloud = [frame_dict['sample_timestamp'] / 1e6 for frame_dict in dict_list]

        # 1. Get your data from KISS-ICP and the dataset
        kiss_icp_poses_save = poses_kiss_icp
        point_clouds_save = source_pc_list_all_frames
        timestamps_save = lidar_timestamps_flexcloud
        timestamps_save_original = timestamps_save.copy()

        gnss_data_save = []
        # Define a constant, low standard deviation for all points
        for dict_idx, frame_dict in enumerate(dict_list):
            # 1. Get the full 4x4 transformation matrix
            #transform_matrix_global_from_ego = frame_dict['global_from_ego_i']

            transform_matrix_global_from_ego = frame_dict['ego_ref_from_ego_i']

            # 2. Extract just the position (x, y, z) from the last column
            position = transform_matrix_global_from_ego[:3, 3]

            # 3. Combine the position and standard deviation into a single 6-element array
            gnss_entry = np.hstack((position, std_dev_placeholder))

            # 4. Append the correct 6-element array to the list
            gnss_data_save.append(gnss_entry)

        file_path_gnss_all = os.path.join(args.scene_io_dir, 'flexcloud_io/all_gnss_data_poses.txt')
        save_gnss_to_single_file(gnss_data_save, file_path_gnss_all)

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

        # 3. Save your in-memory data to the temporary files
        save_poses_to_kitti_format(kiss_icp_poses_save, odom_path)
        save_gnss_to_directory(gnss_data_save, timestamps_save, pos_dir)
        save_pcds_to_directory(point_clouds_save, timestamps_save_original, pcd_dir)
        print(f"FlexCloud inputs are ready in: {save_dir_flexcloud}")
    else:
        print("No flexcloud data saved as diabled by command line arguments.")

    ################################################################################################################
    ################################################ Saving data needed for rosbag file generation #################
    if args.save_rosbag_data:
        print("Saving rosbag data...")
        # Get the first and last lidar timestamps
        first_lidar_timestamp = dict_list[0]['sample_timestamp']
        last_lidar_timestamp = dict_list[-1]['sample_timestamp']

        # Calculate the start and end timestamps for interpolation
        start_timestamp_imu = first_lidar_timestamp
        end_timestamp_imu = last_lidar_timestamp

        # Define the desired IMU frequency (e.g., 100 Hz)
        time_step = int(1e6 / imu_freq)

        high_freq_timestamps = np.arange(start_timestamp_imu, end_timestamp_imu, time_step)

        # Use tqdm to show progress for the list comprehension
        #imu_data = [transform_imu_to_ego(truckscenes.getclosest("ego_motion_chassis", timest), calibrated_sensor_record_imu)
         #           for timest in tqdm(high_freq_timestamps, desc="Fetching IMU Data")]

        imu_data = [truckscenes.getclosest("ego_motion_chassis", timest)
                   for timest in tqdm(high_freq_timestamps, desc="Fetching IMU Data")]

        print(f"Filtering {len(imu_data)} raw IMU records to remove duplicates...")

        unique_imu_data = []
        if imu_data:  # Make sure the list is not empty
            # Add the first record to start
            unique_imu_data.append(imu_data[0])

            # Iterate through the rest of the records
            for i in range(1, len(imu_data)):
                # Compare the token of the current record to the token of the last added unique record
                if imu_data[i]['token'] != unique_imu_data[-1]['token']:
                    unique_imu_data.append(imu_data[i])

        imu_data = unique_imu_data
        print(f"Finished filtering. Found {len(imu_data)} unique IMU records.")
        #print(imu_data)

        print("Finished fetching IMU Data")

        print(calibrated_sensor_record_imu)

        print("Saving data to create rosbag...")

        context_file_path_rosbag = os.path.join(args.scene_io_dir, "preprocessed_data_rosbag.npz")
        print(f"Saving data to {context_file_path_rosbag}")
        np.savez_compressed(
            context_file_path_rosbag,
            dict_list=np.array(dict_list_rosbag, dtype=object),
            poses_kiss_icp=poses_kiss_icp,
            scene_name=scene_name,
            save_path=save_path,
            IMU_data=np.array(imu_data, dtype=object),
            imu_calib=np.array(calibrated_sensor_record_imu, dtype=object),
        )
    else:
        print("Rosbag data not saved as disabled by command line.")

    ##########################################################################################################
    ######################### Save all data for part 3 of pipeline ###########################################
    context_file_path = os.path.join(args.scene_io_dir, "preprocessed_data.npz")
    print(f"Saving general data dict to {context_file_path}")

    np.savez_compressed(
        context_file_path,

        # --- Core Data Arrays Needed by Part 3 ---
        # The list of lean dictionaries (metadata only)
        dict_list=np.array(dict_list_for_saving, dtype=object),
        # The two lists of cleaned, per-frame static points and their sensor IDs
        static_points_refined=np.array(static_points_refined, dtype=object),
        static_points_refined_sensor_ids=np.array(static_points_refined_sensor_ids, dtype=object),

        # --- Pose Information ---
        poses_kiss_icp=poses_kiss_icp,  # Can be None
        gt_relative_poses_arr=gt_relative_poses_arr,

        # --- Config & Metadata (small, essential) ---
        config=np.array(config, dtype=object),
        category_name_to_learning_id=np.array(category_name_to_learning_id, dtype=object),
        learning_id_to_name=np.array(learning_id_to_name, dtype=object),
        save_path=save_path,
        scene_name=scene_name,
        FREE_LEARNING_INDEX=FREE_LEARNING_INDEX,
        BACKGROUND_LEARNING_INDEX=BACKGROUND_LEARNING_INDEX,
        sensor_max_ranges_arr=sensor_max_ranges_arr,
    )

    print("Finished saving general data dict")
    print(f"Saved pipeline context to {context_file_path}")

    ###################################### Saving a copy of the config file to the target folder ######################

    print(f"\nSaving a copy of the configuration file...")
    source_config_path = args.config_path
    save_path_config = os.path.join(save_path, scene_name)
    os.makedirs(save_path_config, exist_ok=True)
    destination_config_path = os.path.join(save_path_config, os.path.basename(source_config_path))
    # Copy the file (shutil.copy2 also preserves metadata like timestamps)
    shutil.copy2(source_config_path, destination_config_path)
    print(f"✅ Configuration file saved to: {destination_config_path}")

    ################################## Saving args for script 1 ###########################################
    print(f"Saving runtime arguments...")
    args_file_path = os.path.join(save_path_config, 'runtime_args_part1.json')
    # Convert the argparse Namespace to a dictionary
    args_dict = vars(args)

    # Write the dictionary to a JSON file
    with open(args_file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)

    print(f"✅ Runtime arguments saved to: {args_file_path}")

    ############################## Create success flag for checking if scene was in split ############################

    print("Creating success flag for Part 1...")
    flag_file_path = os.path.join(args.scene_io_dir, "part1_success.flag")
    with open(flag_file_path, 'w') as f:
        f.write('success')

    print(f"\n--- Part 1 Complete ---")



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

    ####################### Kiss-ICP refinement ##########################################
    parse.add_argument('--icp_refinement', action='store_true', help='Enable ICP refinement')
    parse.add_argument('--initial_guess_mode', type=str, default='none', choices=['none', 'ego_pose', 'imu'])
    parse.add_argument('--pose_error_plot', action='store_true', help='Plot pose error')

    ####################### MapMOS ########################################################
    parse.add_argument('--run_mapmos', action='store_true', help='Enable the MapMOS pipeline to refine the static map.')
    parse.add_argument('--save_mapmos_pc', action='store_true', help='Save labeled point clouds from MapMOS.')

    ######################## Filtering ####################################################
    parse.add_argument('--filter_mode', type=str, default='none', choices=['none', 'sor', 'ror', 'both'],
                       help='Noise filtering method to apply before meshing')

    parse.add_argument('--filter_lidar_intensity', action='store_true', help='Enable lidar intensity filtering')

    parse.add_argument('--filter_raw_pc', action='store_true', help='Enable raw pc filtering')
    parse.add_argument('--filter_static_pc', action='store_true', help='Enable static pc filtering')

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

    parse.add_argument('--vis_static_frame_comparison_kiss_refined', action='store_true',
                       help='Enable static frame comparison kiss refinement')
    parse.add_argument('--vis_aggregated_static_kiss_refined', action='store_true',
                       help='Enable aggregated static kiss refinement')

    ###################### Decide if you want to use keyframes only for static map ############################################
    parse.add_argument('--static_map_keyframes_only', action='store_true',
                       help='Build the final static map using only keyframes (after ICP, if enabled, ran on all frames).')

    ##################### Process input for Rosbag file creation and save input #########################################
    parse.add_argument('--save_rosbag_data', action='store_true', help='Save rosbag data')
    parse.add_argument(
        '--use_flexcloud',
        type=int,
        default=0,
        choices=[0, 1],
        help="Flag to indicate whether to use FlexCloud processing (1=True, 0=False)."
    )
    parse.add_argument('--save_pointcloud_flexcloud', action='store_true', help='Save aggregated pointcloud for flexcloud')
    parse.add_argument('--save_sensor_fused_pc', action='store_true', help='Save sensor fused pc')


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
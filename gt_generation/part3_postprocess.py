import os
import sys
import numpy as np
import open3d as o3d
import torch
import time
from tqdm import tqdm
from truckscenes import TruckScenes
from scipy.spatial.transform import Rotation
from collections import defaultdict
from utils.visualization import visualize_pointcloud, visualize_pointcloud_bbox, visualize_occupancy_o3d, calculate_and_plot_pose_errors
from utils.pointcloud_processing import denoise_pointcloud
from truckscenes.utils.geometry_utils import transform_matrix
from mmcv.ops.points_in_boxes import (points_in_boxes_cpu, points_in_boxes_all)
# import chamfer
from pyquaternion import Quaternion
from utils.occupancy_utils import calculate_camera_visibility_gpu_host, calculate_lidar_visibility_gpu_host, calculate_camera_visibility_cpu, calculate_lidar_visibility
from utils.pointcloud_processing import in_range_mask, preprocess_cloud, create_mesh_from_map, preprocess
from utils.constants import CLASS_COLOR_MAP, STATE_FREE, STATE_UNOBSERVED, STATE_OCCUPIED, DEFAULT_COLOR
from utils.refinement import assign_label_by_L_shape, assign_label_by_L_shape_optimized
from utils.bbox_utils import is_point_centroid_z_similar, are_box_sizes_similar, get_object_overlap_signature_BATCH, compare_signatures_class_based_OVERLAP_RATIO
from utils.data_utils import load_kitti_poses

def main(args):
    print("--- Running Part 3: Post-FlexCloud Processing ---")

    ################ Set device ################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir_flexcloud = args.scene_io_dir
    context_file_path = os.path.join(save_dir_flexcloud, "preprocessed_data.npz")

    # --- Load the saved context from Part 1 ---
    print(f"Loading pipeline context from {context_file_path}...")
    context = np.load(context_file_path, allow_pickle=True)
    #source_pc_list_all_frames = context['source_pc_list_all_frames']
    #source_pc_sids_list_all_frames = context['source_pc_sids_list_all_frames']
    dict_list = context['dict_list']
    poses_kiss_icp = context['poses_kiss_icp']  # This might be None
    config = context['config'].item()  # .item() to get the dictionary back
    #truckscenesyaml = context['truckscenesyaml'].item()
    #learning_map = context['learning_map'].item()
    category_name_to_learning_id = context['category_name_to_learning_id'].item()
    learning_id_to_name = context['learning_id_to_name'].item()
    pc_range = context['pc_range']
    voxel_size = context['voxel_size'].item()
    occ_size = context['occ_size']
    save_path = str(context['save_path'])
    scene_name = str(context['scene_name'])
    # args_dict = context['args_dict'].item()
    FREE_LEARNING_INDEX = context['FREE_LEARNING_INDEX'].item()
    BACKGROUND_LEARNING_INDEX = context['BACKGROUND_LEARNING_INDEX'].item()
    sensors = context['sensors']
    cameras = context['cameras']
    sensor_max_ranges_arr = context['sensor_max_ranges_arr']
    self_range = context['self_range']
    lidar_pc_final_global=context['lidar_pc_final_global']
    lidar_pc_final_global_sensor_ids=context['lidar_pc_final_global_sensor_ids']


    # Load the corrected poses from FlexCloud's output ---
    flexcloud_output_dir = os.path.join(save_dir_flexcloud, "flexcloud_io/pcd_transformed")
    flexcloud_xyz_poses_path = os.path.join(flexcloud_output_dir, "traj_matching/target_rs.txt")

    print(f"Loading corrected poses from {flexcloud_xyz_poses_path}...")

    if not os.path.exists(flexcloud_xyz_poses_path):
        print(f"File not found: {flexcloud_xyz_poses_path}")
        sys.exit(1)

    try:
        # Use np.loadtxt to load the xyz values directly into a 2D array
        flexcloud_corrected_poses_xyz = np.loadtxt(flexcloud_xyz_poses_path)
        print("Successfully loaded xyz poses.")
    except Exception as e:
        print(f"Could not load corrected poses. Error: {e}")
        sys.exit(1)

    if flexcloud_corrected_poses_xyz.shape[0] == 0 or flexcloud_corrected_poses_xyz.shape[1] != 3:
        print("Corrected poses data is empty or has an incorrect format. Exiting.")
        sys.exit(1)

    num_frames = poses_kiss_icp.shape[0]
    if flexcloud_corrected_poses_xyz.shape[0] != num_frames:
        print(
            f"Mismatch in number of frames. KISS-ICP has {num_frames}, FlexCloud has {flexcloud_corrected_poses_xyz.shape[0]}.")
        sys.exit(1)

    print(f"Number of frames: {flexcloud_corrected_poses_xyz.shape}")

    flexcloud_corrected_poses = poses_kiss_icp.copy()

    for i in range(num_frames):
        # This line now correctly accesses the i-th pose matrix and its translation vector
        flexcloud_corrected_poses[i, :3, 3] = flexcloud_corrected_poses_xyz[i, :]

    print(f"Successfully loaded {flexcloud_corrected_poses.shape[0]} corrected poses.")

    gt_poses = np.array([frame['ego_ref_from_ego_i'] for frame in dict_list])
    kiss_poses = context['poses_kiss_icp']

    # Comparison 1: KISS-ICP vs Ground Truth
    if kiss_poses is not None:
        calculate_and_plot_pose_errors(
            poses_estimated=kiss_poses,
            poses_reference=gt_poses,
            title_prefix="KISS-ICP vs GT",
            scene_name=scene_name,
            save_dir=args.save_path,
            show_plot=args.pose_error_plot
        )

    # Comparison 2: FlexCloud vs Ground Truth
    calculate_and_plot_pose_errors(
        poses_estimated=flexcloud_corrected_poses,
        poses_reference=gt_poses,
        title_prefix="FlexCloud vs GT",
        scene_name=scene_name,
        save_dir=args.save_path,
        show_plot=args.pose_error_plot
    )

    # Comparison 3: FlexCloud vs KISS-ICP (to see the correction amount)
    if kiss_poses is not None:
        calculate_and_plot_pose_errors(
            poses_estimated=flexcloud_corrected_poses,
            poses_reference=kiss_poses,
            title_prefix="FlexCloud vs KISS-ICP",
            scene_name=scene_name,
            save_dir=args.save_path,
            show_plot=args.pose_error_plot
        )


    # Re-initialize the TruckScenes object
    trucksc = TruckScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    TRUCK_ID = category_name_to_learning_id.get('vehicle.truck', 10)
    TRAILER_ID = category_name_to_learning_id.get('vehicle.trailer', 9)
    CAR_ID = category_name_to_learning_id.get('vehicle.car', 4)
    OTHER_VEHICLE_ID = category_name_to_learning_id.get('vehicle.other', 13)

    print(f"TRUCK_ID: {TRUCK_ID}, TRAILER_ID: {TRAILER_ID}, CAR_ID: {CAR_ID}, OTHER_VEHICLE_ID: {OTHER_VEHICLE_ID}")

    ########################## Specify thresholds for later aggregation based on overlap #########################

    CLASS_BASED_THRESHOLDS = {
        frozenset({TRUCK_ID, TRAILER_ID}): {
            'dist_tolerance_m': 1.0,  # Centroid distance must be within 50cm
            'yaw_tolerance_rad': 0.1,  # Relative yaw must be within ~6 degrees
            'ratio_absolute_tolerance': 0.2  # Overlap ratio
        },
        frozenset({CAR_ID, CAR_ID}): {
            'dist_tolerance_m': 0.5,
            'yaw_tolerance_rad': 0.1,
            'ratio_absolute_tolerance': 0.2
        },
        frozenset({TRUCK_ID, OTHER_VEHICLE_ID}): {
            'dist_tolerance_m': 0.5,
            'yaw_tolerance_rad': 0.1,
            'ratio_absolute_tolerance': 0.05
        },
        frozenset({TRAILER_ID, OTHER_VEHICLE_ID}): {
            'dist_tolerance_m': 0.5,
            'yaw_tolerance_rad': 0.2,
            'ratio_absolute_tolerance': 0.05
        },
        'default': {
            'dist_tolerance_m': 0.5,
            'yaw_tolerance_rad': 0.2,
            'ratio_absolute_tolerance': 0.1
        }
    }

    point_cloud_flexcloud_path = os.path.join(flexcloud_output_dir, "refined_map.pcd")
    point_cloud_flexcloud = o3d.io.read_point_cloud(point_cloud_flexcloud_path)
    point_cloud_flexcloud_np = np.array(point_cloud_flexcloud.points)

    visualize_pointcloud(point_cloud_flexcloud_np)

    unrefined_pc_ego_ref_list = []

    for pose_index, frame_dict in enumerate(dict_list):
        if frame_dict['is_key_frame']:
            pc_to_transform = frame_dict['lidar_pc_ego_i']
            xyz_points = pc_to_transform[:, :3]
            label_col = pc_to_transform[:, 3:]

            pose_to_transform = flexcloud_corrected_poses[pose_index]

            ones_col = np.ones((xyz_points.shape[0], 1))
            homogeneous_pc = np.hstack([xyz_points, ones_col])

            transformed_homogeneous = pose_to_transform @ homogeneous_pc.T
            tranformed_pc_ego_ref_xyz = transformed_homogeneous.T[:, :3]

            tranformed_pc_ego_ref = np.hstack([tranformed_pc_ego_ref_xyz, label_col])

            unrefined_pc_ego_ref_list.append(tranformed_pc_ego_ref)

    unrefined_pc = np.concatenate(unrefined_pc_ego_ref_list, axis=0)

    visualize_pointcloud(unrefined_pc)




    lidar_pc = lidar_pc_final_global.T
    lidar_pc_sensor_ids = lidar_pc_final_global_sensor_ids

    ################################## Filtering of static aggregated point cloud #####################################
    if args.filter_aggregated_static_pc and args.filter_mode != 'none':
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(lidar_pc[:3, :].T)
        filtered_pcd_o3d, kept_indices = denoise_pointcloud(pcd_o3d, args.filter_mode, config,
                                                            location_msg="aggregated static points")
        lidar_pc = lidar_pc[:, kept_indices]
        lidar_pc_sensor_ids = lidar_pc_sensor_ids[kept_indices]

    assert lidar_pc.shape[1] == lidar_pc_sensor_ids.shape[0], (
        f"point count ({lidar_pc.shape[1]}) != sensor_ids count ({lidar_pc_sensor_ids.shape[0]})"
    )

    ############################## Visualization ######################################################
    if args.vis_filtered_aggregated_static:
        visualize_pointcloud(lidar_pc.T, title=f"Aggregated Refined Static PC (Global) - Scene {scene_name}")

    ############################### adapt dict_list to only use keyframes #################################
    source_dict_list_for_objects = dict_list

    if args.dynamic_map_keyframes_only:
        print("Dynamic object aggregation will use ONLY KEYFRAMES.")
        source_dict_list_for_objects = [fd for fd in dict_list if fd['is_key_frame']]
        if not source_dict_list_for_objects:
            print(
                "Warning: --dynamic_map_keyframes_only is set, but no keyframes found in dict_list. Object data will be empty.")
    else:
        print("Dynamic object aggregation will use ALL FRAMES.")

    print(f"Dynamic object aggregation will use {len(source_dict_list_for_objects)} frames.")

    ###################################### Loop to save occupancy data per frame ######################################
    tranforms_ref_to_k = []
    i = 0
    while int(i) < 1000:
        print(f"Processing frame {i} for saving")
        if i >= len(dict_list):
            print('finish scene!')
            return
        frame_dict = dict_list[i]
        # Only processes key frames since non-key frames do not have ground truth annotations
        is_key_frame = frame_dict['is_key_frame']
        if not is_key_frame:
            i = i + 1
            continue

        point_cloud = lidar_pc[:3, :]
        point_cloud_sensor_ids = lidar_pc_sensor_ids
        print(f"Original point_cloud: {point_cloud.shape} with sensor ids: {point_cloud_sensor_ids.shape}")

        ############################## Transform points into common reference frame i ############################
        sample = trucksc.get('sample', frame_dict['sample_token'])
        ego_pose = trucksc.getclosest('ego_pose', sample['timestamp'])
        ego_from_global = transform_matrix(ego_pose['translation'], Quaternion(ego_pose['rotation']), inverse=True)

        sensor_origins = []
        for idx, sensor in enumerate(sensors):
            sd = trucksc.get('sample_data', sample['data'][sensor])
            cs = trucksc.get('calibrated_sensor', sd['calibrated_sensor_token'])

            T_s_to_ego = transform_matrix(cs['translation'], Quaternion(cs['rotation']), inverse=False)
            sensor_origins.append(T_s_to_ego[:3, 3])
        sensor_origins = np.stack(sensor_origins, axis=0)  # shape (S,3)

        if args.icp_refinement:
            T_ref_from_k = poses_kiss_icp[i]
            T_k_from_ref = np.linalg.inv(T_ref_from_k)
            tranforms_ref_to_k.append(T_k_from_ref)
        else:
            T_ref_from_k = frame_dict['ego_ref_from_ego_i']
            T_k_from_ref = np.linalg.inv(T_ref_from_k)
            tranforms_ref_to_k.append(T_ref_from_k)

        point_cloud_global_xyz = lidar_pc[:3, :].copy()
        print(f"Global point_cloud shape: {point_cloud_global_xyz.shape}")

        if point_cloud_global_xyz.shape[1] > 0:
            points_homo = np.hstack((point_cloud_global_xyz.T, np.ones((point_cloud_global_xyz.shape[1], 1))))
            points_k_ego = (T_k_from_ref @ points_homo.T).T[:, :3]
            point_cloud = points_k_ego.T
        else:
            point_cloud = np.zeros((3, 0), dtype=point_cloud_global_xyz.dtype)

        print(f"Ego point_cloud shape: {point_cloud.shape} with sensor ids: {point_cloud_sensor_ids.shape}")

        assert point_cloud.shape[1] == point_cloud_sensor_ids.shape[0]

        ################ load boxes of target frame ############################
        gt_bbox_3d_unmodified = frame_dict['gt_bbox_3d_unmodified']
        boxes = frame_dict['boxes_ego']

        object_tokens = frame_dict['object_tokens']
        converted_object_category = frame_dict['converted_object_category']

        gt_bbox_3d_points_in_boxes_cpu_enlarged_filter = frame_dict['gt_bbox_3d_points_in_boxes_cpu_enlarged']

        ################## Aggregation of object points for the current frame #####################
        current_keyframe_object_tokens = frame_dict['object_tokens']
        current_keyframe_gt_bboxes = frame_dict['gt_bbox_3d_overlap_enlarged']
        current_keyframe_object_categories = frame_dict['converted_object_category']

        dynamic_object_points_list = []  # Final transformed object points for scene
        dynamic_object_points_sids_list = []
        dynamic_object_points_semantic_list = []  # Final semantic-labeled points for scene
        dynamic_object_points_semantic_sids_list = []

        for keyframe_obj_idx, current_object_token in enumerate(current_keyframe_object_tokens):
            class_id = current_keyframe_object_categories[keyframe_obj_idx]
            class_name = learning_id_to_name.get(class_id, 'background')  # Safely get the name
            short_token = current_object_token.split('-')[0]

            print(f"  Processing dynamic object: {class_name} ('{short_token}...') for keyframe {i}...")

            # --- 1. Get the interaction signature for the object in THIS keyframe ---
            signature_at_keyframe_i = get_object_overlap_signature_BATCH(frame_dict, keyframe_obj_idx)

            if signature_at_keyframe_i:
                token_to_name_map = {
                    tok: learning_id_to_name.get(cat_id, 'background')
                    for tok, cat_id in zip(current_keyframe_object_tokens, current_keyframe_object_categories)
                }
                sig_str = ", ".join(
                    [f"(ratio={ratio:.4f}, dist={dist:.2f}m, {learning_id_to_name.get(cat_id, '?')}, {yaw:.4f} rad)"
                     for ratio, dist, tok, cat_id, yaw in signature_at_keyframe_i])
                print(f"    Signature for '{class_name}': [{sig_str}]")

            contextual_canonical_segments = []
            contextual_canonical_sids_segments = []

            # --- 2. Find all frames with a similar context and collect their points ---
            for candidate_frame_data in source_dict_list_for_objects:
                if current_object_token in candidate_frame_data['object_tokens']:
                    candidate_obj_idx = candidate_frame_data['object_tokens'].index(current_object_token)

                    # Get the category IDs for the target object in both the keyframe and candidate frame
                    category_id_keyframe = current_keyframe_object_categories[keyframe_obj_idx]
                    category_id_candidate = candidate_frame_data['converted_object_category'][candidate_obj_idx]

                    # Get the signatures for both
                    signature_at_candidate_frame = get_object_overlap_signature_BATCH(candidate_frame_data,
                                                                                      candidate_obj_idx)

                    # Get the bounding box parameters for both the keyframe and candidate frame object
                    bbox_params_keyframe = current_keyframe_gt_bboxes[keyframe_obj_idx]
                    bbox_params_candidate = candidate_frame_data['gt_bbox_3d_overlap_enlarged'][candidate_obj_idx]
                    points_keyframe = frame_dict['object_points_list'][keyframe_obj_idx]
                    points_candidate = candidate_frame_data['object_points_list'][candidate_obj_idx]

                    # Compare the keyframe's signature with the candidate's signature
                    if (compare_signatures_class_based_OVERLAP_RATIO(
                            signature_at_keyframe_i, category_id_keyframe,
                            signature_at_candidate_frame, category_id_candidate,
                            CLASS_BASED_THRESHOLDS
                    ) and
                            are_box_sizes_similar(bbox_params_keyframe, bbox_params_candidate,
                                                  volume_ratio_tolerance=1.05,
                                                  dim_ratio_tolerance=1.05) and
                            is_point_centroid_z_similar(points_keyframe, points_candidate,
                                                        category_id_keyframe, OTHER_VEHICLE_ID,
                                                        z_tolerance=0.3)):
                        # Contexts match! Get the points from this candidate frame.
                        obj_points_in_candidate = candidate_frame_data['object_points_list'][candidate_obj_idx]
                        obj_sids_in_candidate = candidate_frame_data['object_points_list_sensor_ids'][candidate_obj_idx]

                        if obj_points_in_candidate is not None and obj_points_in_candidate.shape[0] > 0:
                            # Canonicalize the points using the candidate frame's bbox parameters
                            bbox_params_cand = candidate_frame_data['gt_bbox_3d_overlap_enlarged'][candidate_obj_idx]
                            center_cand, yaw_cand = bbox_params_cand[:3], bbox_params_cand[6]

                            translated_pts = obj_points_in_candidate[:, :3] - center_cand
                            Rot_can = Rotation.from_euler('z', -yaw_cand, degrees=False)
                            canonical_pts = Rot_can.apply(translated_pts)

                            contextual_canonical_segments.append(canonical_pts)
                            contextual_canonical_sids_segments.append(obj_sids_in_candidate)

            num_aggregated_frames = len(contextual_canonical_segments)
            print(
                f"      -> For object '{class_name} ({short_token})', found {num_aggregated_frames} similar context frame(s).")

            # --- 3. Aggregate the collected points and place them in the scene ---
            aggregated_canonical_points = np.zeros((0, 3), dtype=float)
            aggregated_canonical_sids = np.zeros((0,), dtype=int)

            if contextual_canonical_segments:
                aggregated_canonical_points = np.concatenate(contextual_canonical_segments, axis=0)
                aggregated_canonical_sids = np.concatenate(contextual_canonical_sids_segments, axis=0)
                print(f"      -> Aggregated {aggregated_canonical_points.shape[0]} points for this context.")
            else:
                print(f"      -> No similar contexts found. Using points from keyframe {i} only.")
                obj_points_in_keyframe = frame_dict['object_points_list'][keyframe_obj_idx]
                if obj_points_in_keyframe is not None and obj_points_in_keyframe.shape[0] > 0:
                    # Canonicalize points from the current keyframe itself
                    bbox_params_keyframe = current_keyframe_gt_bboxes[keyframe_obj_idx]
                    center_key, yaw_key = bbox_params_keyframe[:3], bbox_params_keyframe[6]
                    translated = obj_points_in_keyframe[:, :3] - center_key
                    Rot_key = Rotation.from_euler('z', -yaw_key, degrees=False)
                    aggregated_canonical_points = Rot_key.apply(translated)
                    aggregated_canonical_sids = frame_dict['object_points_list_sensor_ids'][keyframe_obj_idx]

            # --- 4. De-canonicalize and store for final scene assembly ---
            if aggregated_canonical_points.shape[0] > 0:
                # De-canonicalize using the pose from the CURRENT keyframe `i`
                bbox_params_decanon = current_keyframe_gt_bboxes[keyframe_obj_idx]
                decanon_center, decanon_yaw = bbox_params_decanon[:3], bbox_params_decanon[6]

                Rot_decan = Rotation.from_euler('z', decanon_yaw, degrees=False)
                points_in_scene_xyz = Rot_decan.apply(aggregated_canonical_points) + decanon_center

                # Keep track of the sensor IDs for these points
                sids_for_points = aggregated_canonical_sids

                bbox_for_filtering = gt_bbox_3d_points_in_boxes_cpu_enlarged_filter[
                                     keyframe_obj_idx: keyframe_obj_idx + 1]

                original_count = points_in_scene_xyz.shape[0]

                # Only filter if there are points to filter
                if original_count > 0:
                    # Prepare tensors for the operation
                    points_tensor_gpu = torch.from_numpy(points_in_scene_xyz[np.newaxis, :, :]).float().to(device)
                    box_tensor_gpu = torch.from_numpy(bbox_for_filtering[np.newaxis, :, :]).float().to(device)

                    # Get the box indices for each point
                    box_indices_tensor = points_in_boxes_all(points_tensor_gpu, box_tensor_gpu)

                    points_in_box_mask = (box_indices_tensor[0, :, 0] >= 0)

                    points_in_box_mask_numpy = points_in_box_mask.cpu().numpy()

                    # Apply the corrected and safe NumPy mask
                    points_in_scene_xyz = points_in_scene_xyz[points_in_box_mask_numpy]
                    sids_for_points = sids_for_points[points_in_box_mask_numpy]

                    filtered_count = points_in_scene_xyz.shape[0]
                    if original_count > filtered_count:
                        print(
                            f"      [Final Clip] Object {class_name}: Clipped aggregated points from {original_count} to {filtered_count}.")

                # Add to lists for final concatenation
                dynamic_object_points_list.append(points_in_scene_xyz)
                dynamic_object_points_sids_list.append(sids_for_points)

                # Create the semantic version
                semantic_label = current_keyframe_object_categories[keyframe_obj_idx]
                semantic_column = np.full((points_in_scene_xyz.shape[0], 1), semantic_label,
                                          dtype=points_in_scene_xyz.dtype)
                points_with_semantics = np.concatenate([points_in_scene_xyz, semantic_column], axis=1)
                dynamic_object_points_semantic_list.append(points_with_semantics)
                dynamic_object_points_semantic_sids_list.append(sids_for_points)

        # --- Final Scene Assembly for frame i ---
        print("\n  Assembling final scene for saving...")

        if args.vis_static_before_combined_dynamic:
            visualize_pointcloud_bbox(point_cloud.T,
                                      boxes=boxes,
                                      title=f"Fused static PC before combining with dynamic points + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)

        ######################################## Cleanup of overlapping box labels #######################################
        # --- Step 1: Aggregate all dynamic points and their initial labels ---
        dyn_points = np.concatenate(dynamic_object_points_list)
        dyn_points_sids = np.concatenate(dynamic_object_points_sids_list)

        dyn_points_semantic = np.concatenate(dynamic_object_points_semantic_list)
        dyn_points_semantic_sids = np.concatenate(dynamic_object_points_semantic_sids_list)

        if args.vis_dyn_before_reassignment:
            visualize_pointcloud_bbox(dyn_points_semantic,
                                      boxes=boxes,
                                      title=f"Fused dynamic and static PC + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)

        print("--- Performing L-Shape Refinement on Aggregated Scene Points ---")

        # --- Step 2: Find which of these dynamic points lie in overlapping zones ---
        boxes_for_lshape_logic = frame_dict['gt_bbox_3d_unmodified']
        box_categories = np.array(frame_dict["converted_object_category"])

        # Use slightly enlarged, mmcv-compatible boxes to find all potential overlaps
        boxes_for_mmcv_check = frame_dict['gt_bbox_3d_points_in_boxes_cpu_enlarged']
        boxes_for_overlap = frame_dict['gt_bbox_3d_overlap_enlarged']

        all_points_per_box = dynamic_object_points_list

        points_in_boxes = points_in_boxes_cpu(
            torch.from_numpy(dyn_points[np.newaxis, :, :]), #.float().to(device),
            torch.from_numpy(boxes_for_mmcv_check[np.newaxis, :, :]) #.float().to(device),
        )

        point_counts = points_in_boxes[0].sum(axis=1).numpy()
        overlap_idxs = np.where(point_counts > 1)[0]

        if len(overlap_idxs) > 0:
            if args.vis_dyn_ambigious_points:
                viz_cloud_ambiguous = dyn_points_semantic.copy()
                viz_cloud_ambiguous[overlap_idxs, 3] = 20
                visualize_pointcloud_bbox(viz_cloud_ambiguous,
                                          boxes=boxes,
                                          title=f"All Ambiguous Points Highlighted - Frame {i}",
                                          self_vehicle_range=self_range,
                                          vis_self_vehicle=True)

            print(f"Found {len(overlap_idxs)} ambiguous points in the final aggregated cloud.")
            pt_to_box_map = defaultdict(list)
            for pi in overlap_idxs:
                box_indices = np.where(points_in_boxes[0][pi, :].bool())[0]
                pt_to_box_map[pi] = box_indices.tolist()

            # Prepare the boxes with labels for the function
            boxes_with_labels = np.concatenate([boxes_for_lshape_logic, box_categories.reshape(-1, 1)], axis=1)

            labels_before_refinement = dyn_points_semantic[:, 3:4]

            # --- Step 3: Call the refinement function on the dynamic subset ---
            final_dyn_labels, reassigned_indices = assign_label_by_L_shape(
                overlap_idxs=overlap_idxs,
                pt_to_box_map=pt_to_box_map,
                points=dyn_points,  # Pass only the dynamic points
                boxes=boxes_with_labels,
                boxes_iou=boxes_for_overlap,
                box_cls_labels=box_categories,
                pt_labels=labels_before_refinement,
                all_object_points=all_points_per_box,
                dyn_points_in_boxes=points_in_boxes[0],
                ID_TRAILER=TRAILER_ID,
                ID_TRUCK=TRUCK_ID,
                ID_FORKLIFT=OTHER_VEHICLE_ID,
                high_overlap_threshold=0.85,
            )

            """final_dyn_labels, reassigned_indices = assign_label_by_L_shape_optimized(
                overlap_idxs=truly_ambiguous_idxs,  # Pass only points with true inter-class ambiguity
                pt_to_box_map=pt_to_box_map,
                points=dyn_points,
                boxes=boxes_for_lshape_logic,
                box_cls_labels=box_categories,
                pt_labels=labels_before_refinement,
                ID_TRAILER=TRAILER_ID,
                ID_TRUCK=TRUCK_ID,
                ID_FORKLIFT=OTHER_VEHICLE_ID,
                device=device
            )"""

            # --- Step 4: Update the labels within our semantic dynamic points array ---
            dyn_points_semantic[:, 3] = final_dyn_labels.flatten()

            viz_cloud_unreassigned = dyn_points_semantic.copy()
            unreassigned_indices = np.setdiff1d(overlap_idxs, reassigned_indices)
            viz_cloud_unreassigned[unreassigned_indices, 3] = 20

            if args.vis_dyn_unreassigned_points:
                visualize_pointcloud_bbox(viz_cloud_unreassigned,
                                          boxes=boxes,
                                          title=f"Unreassigned Ambiguous Points Highlighted - Frame {i}",
                                          self_vehicle_range=self_range,
                                          vis_self_vehicle=True)

            print(f"Refinement complete. {len(reassigned_indices)} dynamic points were reassigned.")
        else:
            print("No ambiguous overlaps found within the dynamic points.")

        # --- Step 5: Assemble the final scene with the refined dynamic points ---
        print("Assembling final scene with refined dynamic points...")

        # Prepare the static background points and their labels
        static_points = point_cloud.T
        static_labels = np.full((static_points.shape[0], 1), BACKGROUND_LEARNING_INDEX, dtype=np.uint8)
        static_points_with_labels = np.hstack([static_points, static_labels])
        static_points_sids = point_cloud_sensor_ids

        scene_points = np.concatenate([static_points, dyn_points])
        scene_points_sids = np.concatenate([static_points_sids, dyn_points_sids])

        # Combine the clean static background with the now-refined dynamic points
        scene_semantic_points = np.concatenate([static_points_with_labels, dyn_points_semantic])
        scene_semantic_points_sids = np.concatenate([static_points_sids, dyn_points_semantic_sids])

        print(f"Final assembled scene has {scene_semantic_points.shape[0]} points.")

        assert scene_points_sids.shape == scene_semantic_points_sids.shape

        print(f"Scene points before applying range filtering: {scene_points.shape}")
        mask = in_range_mask(scene_points, pc_range)

        scene_points = scene_points[mask]

        print(f"Scene points sids applying range filtering: {scene_points_sids.shape}")
        scene_points_sids = scene_points_sids[mask]

        print(f"Scene points after applying range filtering: {scene_points.shape}")

        ################################## Visualize #####################################################
        if args.vis_combined_static_dynamic_pc:
            visualize_pointcloud_bbox(scene_semantic_points,
                                      boxes=boxes,
                                      title=f"Fused dynamic and static PC + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)

            visualize_pointcloud_bbox(scene_points,
                                      boxes=boxes,
                                      title=f"Fused dynamic and static PC + BBoxes + Ego BBox - Frame {i}",
                                      self_vehicle_range=self_range,
                                      vis_self_vehicle=True)
        ################################################################################################

        ################## Range filtering for semantic points  ##############
        print(f"Scene semantic points before applying range filtering: {scene_semantic_points.shape}")
        mask = in_range_mask(scene_semantic_points, pc_range)

        scene_semantic_points = scene_semantic_points[mask]
        scene_semantic_points_sids = scene_semantic_points_sids[mask]
        print(f"Scene semantic points after applying range filtering: {scene_semantic_points.shape}")

        ################################## Filtering #################################################
        if args.filter_combined_static_dynamic_pc and args.filter_mode != 'none':
            print(f"Filtering {scene_points.shape}")
            pcd_to_filter = o3d.geometry.PointCloud()
            pcd_to_filter.points = o3d.utility.Vector3dVector(scene_points[:, :3])

            filtered_pcd, kept_indices = denoise_pointcloud(pcd_to_filter, args.filter_mode, config,
                                                            location_msg="final scene points")
            scene_points = np.asarray(filtered_pcd.points)
            scene_points_sids = scene_points_sids[kept_indices]
            scene_semantic_points = scene_semantic_points[kept_indices]
            scene_semantic_points_sids = scene_semantic_points_sids[kept_indices]
        ############################################################################################

        assert scene_points.shape[0] == scene_points_sids.shape[0], (
            f"scene_points count ({scene_points.shape[0]}) != scene_points_sids count ({scene_points_sids.shape[0]})"
        )

        assert scene_semantic_points.shape[0] == scene_semantic_points_sids.shape[0], (
            f"scene_semantic_points count ({scene_semantic_points.shape[0]}) != "
            f"scene_semantic_points_sids count ({scene_semantic_points_sids.shape[0]})"
        )

        ############################ Meshing #######################################################
        if args.meshing:
            print("--- Starting Meshing and Voxelization---")
            print(f"Shape of scene points before meshing: {scene_points.shape}")

            ################## get mesh via Possion Surface Reconstruction ##############
            point_cloud_original = o3d.geometry.PointCloud()  # Initialize point cloud object
            with_normal2 = o3d.geometry.PointCloud()  # Initialize point cloud object
            point_cloud_original.points = o3d.utility.Vector3dVector(
                scene_points[:, :3])  # converts scene_points array into open3D point cloud format
            with_normal = preprocess(point_cloud_original, config)  # Uses the preprocess function to compute normals
            with_normal2.points = with_normal.points  # copies the processed points and normals to another point clouds
            with_normal2.normals = with_normal.normals  # copies the processed points and normals to another point clouds

            # Generate mesh from point cloud using Poisson Surface Reconstruction
            mesh, _ = create_mesh_from_map(None, config['depth'], config['n_threads'],
                                           config['min_density'], with_normal2)
            scene_points = np.asarray(mesh.vertices, dtype=float)

            print(f"Shape of scene points after meshing: {scene_points.shape}")

            ################## remain points with a spatial range ##############
            mask_meshing = in_range_mask(scene_points, pc_range)

            scene_points = scene_points[mask_meshing]  # Filter points within a spatial range

            pcd_meshed_vox_indices = scene_points.copy()

            pcd_meshed_vox_indices[:, 0] = (pcd_meshed_vox_indices[:, 0] - pc_range[0]) / voxel_size
            pcd_meshed_vox_indices[:, 1] = (pcd_meshed_vox_indices[:, 1] - pc_range[1]) / voxel_size
            pcd_meshed_vox_indices[:, 2] = (pcd_meshed_vox_indices[:, 2] - pc_range[2]) / voxel_size
            pcd_meshed_vox_indices = np.floor(pcd_meshed_vox_indices).astype(int)

            pcd_meshed_vox_indices[:, 0] = np.clip(pcd_meshed_vox_indices[:, 0], 0, occ_size[0] - 1)
            pcd_meshed_vox_indices[:, 1] = np.clip(pcd_meshed_vox_indices[:, 1], 0, occ_size[1] - 1)
            pcd_meshed_vox_indices[:, 2] = np.clip(pcd_meshed_vox_indices[:, 2], 0, occ_size[2] - 1)

            binary_voxel_grid = np.zeros(occ_size, dtype=bool)
            unique_meshed_voxel_indices = np.unique(pcd_meshed_vox_indices[:, :3], axis=0)
            if unique_meshed_voxel_indices.shape[0] > 0:
                binary_voxel_grid[
                    unique_meshed_voxel_indices[:, 0], unique_meshed_voxel_indices[:, 1], unique_meshed_voxel_indices[:,
                                                                                          2]] = True
            # Get voxel indices (vx,vy,vz) of occupied cells
            x_occ_idx, y_occ_idx, z_occ_idx = np.where(binary_voxel_grid)
            fov_voxel_indices = np.stack([x_occ_idx, y_occ_idx, z_occ_idx], axis=-1)

            if fov_voxel_indices.shape[0] == 0:
                print(
                    "No fov_voxels (occupied centers) found after voxelizing mesh. Occupancy grid will be empty/free.")
            else:
                # Convert these occupied voxel indices to world coordinates (centers of voxels)
                fov_voxels_world_xyz = fov_voxel_indices.astype(float)
                fov_voxels_world_xyz[:, :3] = (fov_voxels_world_xyz[:, :3] + 0.5) * voxel_size
                fov_voxels_world_xyz[:, 0] += pc_range[0]
                fov_voxels_world_xyz[:, 1] += pc_range[1]
                fov_voxels_world_xyz[:, 2] += pc_range[2]

                # Assign semantics using Chamfer distance
                dense_voxels_needing_labels_world = fov_voxels_world_xyz

                if scene_semantic_points.shape[0] == 0:
                    print(
                        "WARNING: scene_semantic_points is empty for Chamfer. Labeled occupancy will be based on FREE_LEARNING_INDEX.")
                    # occupancy_grid is already initialized to FREE
                else:
                    print(
                        f"Chamfer - Dense (target needing labels) world shape: {dense_voxels_needing_labels_world.shape}")
                    print(f"Chamfer - Sparse (source with labels) world shape: {scene_semantic_points.shape}")

                    x_chamfer = torch.from_numpy(dense_voxels_needing_labels_world).cuda().unsqueeze(0).float()
                    y_chamfer = torch.from_numpy(scene_semantic_points[:, :3]).cuda().unsqueeze(
                        0).float()  # Use XYZ for distance

                    _, _, idx1_chamfer, _ = chamfer.forward(x_chamfer, y_chamfer)
                    indices_from_chamfer = idx1_chamfer[0].cpu().numpy()

                    assigned_labels_for_dense = scene_semantic_points[
                        indices_from_chamfer, 3]  # Assuming label is 4th col (index 3)

                    # Combine world coords of dense voxels with their new labels
                    dense_voxels_world_with_semantic = np.concatenate(
                        [dense_voxels_needing_labels_world, assigned_labels_for_dense[:, np.newaxis]], axis=1
                    )

                    # Convert these world points (now with labels) to final voxel coordinates [vx,vy,vz,label]
                    temp_voxel_coords = dense_voxels_world_with_semantic.copy()
                    temp_voxel_coords[:, 0] = (temp_voxel_coords[:, 0] - pc_range[0]) / voxel_size
                    temp_voxel_coords[:, 1] = (temp_voxel_coords[:, 1] - pc_range[1]) / voxel_size
                    temp_voxel_coords[:, 2] = (temp_voxel_coords[:, 2] - pc_range[2]) / voxel_size

                    # Store integer voxel coordinates and labels
                    dense_voxels_with_semantic_voxelcoords = np.zeros_like(temp_voxel_coords, dtype=int)
                    dense_voxels_with_semantic_voxelcoords[:, :3] = np.floor(temp_voxel_coords[:, :3]).astype(int)
                    dense_voxels_with_semantic_voxelcoords[:, 3] = temp_voxel_coords[:, 3].astype(int)  # Labels column

                    # Clip to ensure within bounds
                    dense_voxels_with_semantic_voxelcoords[:, 0] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 0],
                                                                           0, occ_size[0] - 1)
                    dense_voxels_with_semantic_voxelcoords[:, 1] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 1],
                                                                           0, occ_size[1] - 1)
                    dense_voxels_with_semantic_voxelcoords[:, 2] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 2],
                                                                           0, occ_size[2] - 1)

                    unique_final_vox_indices, unique_idx_map = np.unique(dense_voxels_with_semantic_voxelcoords[:, :3],
                                                                         axis=0, return_index=True)
                    labels_for_unique_final_voxels = dense_voxels_with_semantic_voxelcoords[unique_idx_map, 3]

                    occupancy_grid = np.full(occ_size, FREE_LEARNING_INDEX, dtype=np.uint8)

                    if unique_final_vox_indices.shape[0] > 0:
                        occupancy_grid[
                            unique_final_vox_indices[:, 0],
                            unique_final_vox_indices[:, 1],
                            unique_final_vox_indices[:, 2]
                        ] = labels_for_unique_final_voxels

                    occupied_mask = occupancy_grid != FREE_LEARNING_INDEX
                    total_occupied_voxels = np.sum(occupied_mask)

                    if np.any(occupied_mask):  # Check if there are any occupied voxels
                        vx, vy, vz = np.where(occupied_mask)  # Get the indices (vx, vy, vz) of all occupied voxels
                        labels_at_occupied = occupancy_grid[vx, vy, vz]  # Get the labels at these occupied locations

                        # Stack them into the [vx, vy, vz, label] format
                        dense_voxels_with_semantic_voxelcoords_save = np.stack([vx, vy, vz, labels_at_occupied],
                                                                               axis=-1).astype(int)
                    else:
                        # If no voxels are occupied (e.g., the entire grid is FREE_LEARNING_INDEX)
                        dense_voxels_with_semantic_voxelcoords_save = np.zeros((0, 4), dtype=int)
        ######################## No meshing ############################################
        else:
            print("--- Starting Voxelization without Meshing ---")
            if scene_semantic_points.shape[0] == 0:
                print("No semantic points available. Occupancy grid will be empty/free.")
            else:
                ############################### Calculating visibility masks ###########################
                print("Creating Lidar visibility masks")

                points_to_voxelize = scene_semantic_points.copy()

                labels = points_to_voxelize[:, 3].astype(int)

                voxel_indices_float = np.zeros_like(points_to_voxelize[:, :3])
                voxel_indices_float[:, 0] = (points_to_voxelize[:, 0] - pc_range[0]) / voxel_size
                voxel_indices_float[:, 1] = (points_to_voxelize[:, 1] - pc_range[1]) / voxel_size
                voxel_indices_float[:, 2] = (points_to_voxelize[:, 2] - pc_range[2]) / voxel_size

                voxel_indices_int = np.floor(voxel_indices_float).astype(int)

                dense_voxels_with_semantic_voxelcoords = np.concatenate(
                    [voxel_indices_int, labels[:, np.newaxis]], axis=1
                )

                # Clip to ensure within bounds
                dense_voxels_with_semantic_voxelcoords[:, 0] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 0], 0,
                                                                       occ_size[0] - 1)
                dense_voxels_with_semantic_voxelcoords[:, 1] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 1], 0,
                                                                       occ_size[1] - 1)
                dense_voxels_with_semantic_voxelcoords[:, 2] = np.clip(dense_voxels_with_semantic_voxelcoords[:, 2], 0,
                                                                       occ_size[2] - 1)

                occupancy_grid = np.full(occ_size, FREE_LEARNING_INDEX, dtype=np.uint8)
                if dense_voxels_with_semantic_voxelcoords.shape[0] > 0:
                    occupancy_grid[
                        dense_voxels_with_semantic_voxelcoords[:, 0],
                        dense_voxels_with_semantic_voxelcoords[:, 1],
                        dense_voxels_with_semantic_voxelcoords[:, 2]
                    ] = dense_voxels_with_semantic_voxelcoords[:, 3]

                points_origin = sensor_origins[scene_semantic_points_sids]
                print(f"Points origin shape: {points_origin.shape}")
                points_label = scene_semantic_points[:, 3].astype(int)
                print(f"Points label shape: {points_label.shape}")
                points = scene_semantic_points[:, :3]
                print(f"Points shape: {points.shape}")

                ############################### Lidar visibility mask calculation gpu ###############################
                print("Creating Lidar visibility masks using GPU...")
                print("\nTiming GPU Lidar visibility calculation...")
                start_time_gpu = time.perf_counter()

                voxel_state_gpu, voxel_label_gpu = calculate_lidar_visibility_gpu_host(
                    points_cpu=points,  # (N,3) hits
                    points_origin_cpu=points_origin,  # (N,3) original sensor origins
                    points_label_cpu=points_label,  # (N,) semantic labels (ensure int32)
                    pc_range_cpu_list=pc_range,  # [xmin,ymin,zmin,xmax,ymax,zmax] list
                    voxel_size_cpu_scalar=voxel_size,  # scalar voxel_size from config
                    spatial_shape_cpu_list=occ_size,  # [Dx,Dy,Dz] list from config
                    occupancy_grid_cpu=occupancy_grid,  # pre-computed (Dx,Dy,Dz) aggregated occupancy (uint8)
                    FREE_LEARNING_INDEX_cpu=FREE_LEARNING_INDEX,  # semantic index for free space
                    FREE_LABEL_placeholder_cpu=-1,  # internal placeholder for initializing labels on GPU
                    points_sensor_indices_cpu=scene_semantic_points_sids.astype(np.int32),
                    sensor_max_ranges_cpu=sensor_max_ranges_arr
                )

                end_time_gpu = time.perf_counter()
                print(f"GPU Lidar visibility calculation took: {end_time_gpu - start_time_gpu:.4f} seconds")

                print(f"GPU Voxel state shape: {voxel_state_gpu.shape}")
                print(f"GPU Voxel label shape: {voxel_label_gpu.shape}")
                print("Finished Lidar visibility masks (GPU).")

                ######################## Visualization of lidar visibility mask #############################
                if args.vis_lidar_visibility:
                    voxel_size_for_viz = np.array([voxel_size] * 3)
                    visualize_occupancy_o3d(
                        voxel_state=voxel_state_gpu,
                        voxel_label=voxel_label_gpu,
                        pc_range=pc_range,
                        voxel_size=voxel_size_for_viz,
                        class_color_map=CLASS_COLOR_MAP,
                        default_color=DEFAULT_COLOR,
                        show_semantics=True,
                        show_free=True,
                        show_unobserved=False
                    )
                #############################################################################################

                ########################## Lidar visibility mask cpu #######################################
                run_cpu_comparison_lidar = False
                if run_cpu_comparison_lidar:
                    if isinstance(voxel_size, (int, float)):
                        voxel_size_masks = [voxel_size, voxel_size, voxel_size]

                    # --- Time the CPU execution ---
                    print("\nTiming CPU Lidar visibility calculation...")
                    start_time_cpu = time.perf_counter()

                    voxel_state, voxel_label = calculate_lidar_visibility(
                        points=scene_semantic_points[:, :3],  # (N,3) hits in ego–i
                        points_origin=points_origin,  # (N,3) ray‐starts in ego–i
                        points_label=points_label,  # (N,) semantic of each hit
                        pc_range=pc_range,  # [xmin,ymin,zmin,xmax,ymax,zmax]
                        voxel_size=voxel_size_masks,  # [vx,vy,vz]
                        spatial_shape=occ_size,  # (H,W,Z)
                        occupancy_grid=occupancy_grid,
                        FREE_LEARNING_INDEX=FREE_LEARNING_INDEX,
                        points_sensor_indices=scene_semantic_points_sids,
                        sensor_max_ranges=sensor_max_ranges_arr
                    )

                    end_time_cpu = time.perf_counter()
                    print(f"CPU Lidar visibility calculation took: {end_time_cpu - start_time_cpu:.4f} seconds")

                    print(voxel_state.shape)
                    print(voxel_label.shape)

                    print("Finished Lidar visibility masks cpu")

                    ######################## Visualization of lidar visibility mask #############################
                    if args.vis_lidar_visibility:
                        print("Visualizing with Semantics and Free")
                        visualize_occupancy_o3d(
                            voxel_state=voxel_state,
                            voxel_label=voxel_label,
                            pc_range=pc_range,
                            voxel_size=voxel_size_masks,  # Pass as array
                            class_color_map=CLASS_COLOR_MAP,
                            default_color=DEFAULT_COLOR,
                            show_semantics=True,
                            show_free=True,
                            show_unobserved=False
                        )
                    ###########################################################################################

                ############################# Prepare semantics for saving ###################################

                occupied_mask = occupancy_grid != FREE_LEARNING_INDEX
                total_occupied_voxels = np.sum(occupied_mask)

                if np.any(occupied_mask):  # Check if there are any occupied voxels
                    vx, vy, vz = np.where(occupied_mask)  # Get the indices (vx, vy, vz) of all occupied voxels
                    labels_at_occupied = occupancy_grid[vx, vy, vz]  # Get the labels at these occupied locations

                    # Stack them into the [vx, vy, vz, label] format
                    dense_voxels_with_semantic_voxelcoords_save = np.stack([vx, vy, vz, labels_at_occupied],
                                                                           axis=-1).astype(int)
                else:
                    # If no voxels are occupied (e.g., the entire grid is FREE_LEARNING_INDEX)
                    dense_voxels_with_semantic_voxelcoords_save = np.zeros((0, 4), dtype=int)

                # final_voxel_state_to_save will be the grid with 0 (UNOBS), 1 (FREE), 2 (OCC)
                final_voxel_state_to_save = voxel_state_gpu
                # final_voxel_label_to_save will be the grid with semantic labels for OCC,
                # and FREE_LEARNING_INDEX for FREE and UNOBS. This is the 'semantics' array for Occ3D.
                final_voxel_label_to_save = voxel_label_gpu

                ####################################### Calculate camera visibility mask GPU ######################
                print(f"Calculating camera visibility for cameras (GPU): {cameras}")
                start_time_cam_vis_gpu = time.perf_counter()

                mask_camera = calculate_camera_visibility_gpu_host(  # Call the GPU host function
                    trucksc=trucksc,
                    current_sample_token=frame_dict['sample_token'],  # Pass current sample token
                    lidar_voxel_state_cpu=voxel_state_gpu,  # Output from LiDAR visibility
                    pc_range_cpu_list=pc_range,
                    voxel_size_cpu_scalar=voxel_size,
                    spatial_shape_cpu_list=occ_size,
                    camera_names=cameras,
                    DEPTH_MAX_val=config.get('camera_ray_depth_max', 100.0)  # e.g., 70 meters
                )

                print(f"Camera visibility mask cpu has the shape: {mask_camera.shape}")

                end_time_cam_vis_gpu = time.perf_counter()
                print(
                    f"GPU Camera visibility calculation took: {end_time_cam_vis_gpu - start_time_cam_vis_gpu:.4f} seconds")

                mask_camera_binary = np.zeros_like(mask_camera, dtype=np.uint8)
                mask_camera_binary[mask_camera == STATE_OCCUPIED] = 1
                mask_camera_binary[mask_camera == STATE_FREE] = 1

                ######################### Visualization of camera visibility mask #################################
                if args.vis_camera_visibility:
                    print("Visualizing GPU Camera Visibility Mask Results...")

                    temp_voxel_state_for_cam_viz = mask_camera.copy()

                    voxel_size_arr_viz = np.array([voxel_size] * 3) if isinstance(voxel_size,
                                                                                  (int, float)) else np.array(
                        voxel_size)

                    visualize_occupancy_o3d(
                        voxel_state=temp_voxel_state_for_cam_viz,
                        voxel_label=voxel_label_gpu,
                        pc_range=pc_range,
                        voxel_size=voxel_size_arr_viz,
                        class_color_map=CLASS_COLOR_MAP,
                        default_color=DEFAULT_COLOR,
                        show_semantics=True,  # Show semantics of camera-visible regions
                        show_free=True,  # Not showing LiDAR-free for this specific mask viz
                        show_unobserved=False  # Shows what's NOT camera visible as unobserved
                    )
                ##################################################################################################

                ######################## Calculate camera visibility mask CPU #####################################
                run_cpu_comparison_camera = False
                if run_cpu_comparison_camera:
                    print(f"Calculating camera visibility for cameras (CPU): {cameras}")

                    voxel_size_arr = np.array([voxel_size] * 3) if isinstance(voxel_size, (int, float)) else np.array(
                        voxel_size)
                    occ_size_arr = np.array(occ_size)

                    start_time_cam_vis = time.perf_counter()
                    mask_camera = calculate_camera_visibility_cpu(
                        trucksc=trucksc,
                        current_sample_token=frame_dict['sample_token'],  # Pass current sample token
                        lidar_voxel_state=final_voxel_state_to_save,  # Output from LiDAR visibility
                        pc_range_params=pc_range,
                        voxel_size_params=voxel_size_arr,
                        spatial_shape_params=occ_size_arr,
                        camera_names=cameras,
                        DEPTH_MAX=config.get('camera_ray_depth_max', 100.0)  # Make it configurable
                    )
                    print(f"Camera visibility mask cpu has the shape: {mask_camera.shape}")

                    end_time_cam_vis = time.perf_counter()
                    print(
                        f"CPU Camera visibility calculation took: {end_time_cam_vis - start_time_cam_vis:.4f} seconds")

                    mask_camera_binary = np.zeros_like(mask_camera, dtype=np.uint8)
                    mask_camera_binary[mask_camera == STATE_OCCUPIED] = 1
                    mask_camera_binary[mask_camera == STATE_FREE] = 1

                    ######################### Visualization of camera visibility mask #################################
                    if args.vis_camera_visibility:
                        print("Visualizing Camera Visibility Mask Results...")
                        temp_voxel_state_for_cam_viz = mask_camera.copy()

                        voxel_size_arr_viz = np.array([voxel_size] * 3) if isinstance(voxel_size,
                                                                                      (int, float)) else np.array(
                            voxel_size)

                        visualize_occupancy_o3d(
                            voxel_state=temp_voxel_state_for_cam_viz,
                            voxel_label=voxel_label_gpu,
                            pc_range=pc_range,
                            voxel_size=voxel_size_arr_viz,
                            class_color_map=CLASS_COLOR_MAP,
                            default_color=DEFAULT_COLOR,
                            show_semantics=True,  # Show semantics of camera-visible regions
                            show_free=True,  # Not showing LiDAR-free for this specific mask viz
                            show_unobserved=False  # Shows what's NOT camera visible as unobserved
                        )
                        ###############################################################################################
        #################################### Saving the semantics, lidar mask, camera mask ##########################
        print(
            f"Shape of dense_voxels_with_semantic_voxelcoords for saving: {dense_voxels_with_semantic_voxelcoords_save.shape}")
        print(
            f"Occupancy shape: Occsize: {occupancy_grid.shape}, Total number voxels: {occupancy_grid.shape[0] * occupancy_grid.shape[1] * occupancy_grid.shape[2]}, Occupied: {total_occupied_voxels}")

        print(f"\nPreparing data for saving (using GPU results by default)...")

        # Create the binary mask_lidar (0 for unobserved, 1 for observed)
        # Observed means either FREE or OCCUPIED.
        mask_lidar_to_save = (final_voxel_state_to_save != STATE_UNOBSERVED).astype(np.uint8)
        mask_camera_to_save = mask_camera_binary

        ########################################## Save like Occ3D #######################################
        dirs = os.path.join(save_path, scene_name, frame_dict['sample_token'])
        if not os.path.exists(dirs):
            os.makedirs(dirs)

        output_filepath_npz = os.path.join(dirs, 'labels.npz')
        print(f"Saving semantic occupancy and LiDAR visibility mask to {output_filepath_npz}...")
        np.savez_compressed(
            output_filepath_npz,
            semantics=final_voxel_label_to_save,  # This is (Dx,Dy,Dz) semantic grid
            mask_lidar=mask_lidar_to_save,  # This is (Dx,Dy,Dz) 0-1 LiDAR visibility mask
            mask_camera=mask_camera_to_save,  # This is (Dx,Dy,Dz) 0-1 camera visibility mask
        )
        print(f"  Saved 'semantics' shape: {final_voxel_label_to_save.shape}")
        print(f"  Saved 'mask_lidar' shape: {mask_lidar_to_save.shape} (0=unobserved, 1=observed)")
        print(
            f"  Saved 'mask_camera' shape: {mask_camera_to_save.shape} (0=unobserved, 1=observed)")

        i = i + 1
        continue


    ################################################################################################################
    ################################################################################################################

    print("--- Part 3 Complete ---")
    print("Pipeline finished successfully!")


if __name__ == '__main__':
    from argparse import ArgumentParser

    parse = ArgumentParser()

    ######################################## General settings ##################################################
    parse.add_argument('--config_path', type=str,
                       default='config_truckscenes.yaml')  # Configuration file path with default: "config.yaml"
    parse.add_argument('--save_path', type=str,
                       default='./data/GT_occupancy/')  # save path, default: "./data/GT/GT_occupancy"
    parse.add_argument('--dataroot', type=str,
                       default='./data/truckscenes/')  # data root path, default: "./data/truckScenes
    parse.add_argument('--version', type=str, default='v1.0-trainval')
    parse.add_argument('--label_mapping', type=str,
                       default='truckscenes.yaml')  # YAML file containing label mappings, default: "truckscenes.yaml"

    ####################################### icp settings for final alignment ###################################
    parse.add_argument('--icp_refinement', action='store_true', help='Enable ICP refinement')
    parse.add_argument('--pose_error_plot', action='store_true', help='Plot pose error')

    ##################################### Use flexcloud transformed point cloud and refined poses ################
    parse.add_argument('--use_flexcloud', type=int, default=0,
                        help='A flag indicating whether FlexCloud processing was run (1) or not (0).')

    ######################################## Filter settings ##########################################
    parse.add_argument('--filter_mode', type=str, default='none', choices=['none', 'sor', 'ror', 'both'],
                       help='Noise filtering method to apply before meshing')

    parse.add_argument('--filter_aggregated_static_pc', action='store_true',
                       help='Enable aggregated static pc filtering')
    parse.add_argument('--filter_combined_static_dynamic_pc', action='store_true',
                       help='Enable combined static and dynamic pc filtering')

    ####################### Meshing #####################################################
    parse.add_argument('--meshing', action='store_true', help='Enable meshing')

    ######################### Visualization #################################################
    parse.add_argument('--vis_filtered_aggregated_static', action='store_true',
                           help='Enable filtered aggregated static kiss refinement')
    parse.add_argument('--vis_static_before_combined_dynamic', action='store_true',
                       help='Enable static pc visualization before combined with dynamic points')
    parse.add_argument('--vis_dyn_before_reassignment', action='store_true',
                       help='Enable dynamic points visualization before reassignment')
    parse.add_argument('--vis_dyn_ambigious_points', action='store_true',
                       help='Enable dynamic ambigious points visualization')
    parse.add_argument('--vis_dyn_unreassigned_points', action='store_true',
                       help='Enable dynamic unreassigned points visualization')
    parse.add_argument('--vis_combined_static_dynamic_pc', action='store_true',
                       help='Enable combined static and dynamic pc visualization')


    parse.add_argument('--vis_lidar_visibility', action='store_true', help='Enable lidar visibility visualization')
    parse.add_argument('--vis_camera_visibility', action='store_true', help='Enable camera visibility visualization')

    ####################### Input data #####################################
    parse.add_argument('--scene_io_dir', type=str, required=True, help="Path to the intermediate I/O directory for the scene.")

    ######################## Use only keyframes for static and dynamic map ##########################################
    parse.add_argument('--dynamic_map_keyframes_only', action='store_true',
                       help='Aggregate dynamic object points using only segments from keyframes..')

    args = parse.parse_args()

    main(args)
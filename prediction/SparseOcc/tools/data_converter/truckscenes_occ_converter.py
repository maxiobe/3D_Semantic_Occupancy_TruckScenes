import mmcv
import numpy as np
import os
from os import path as osp
from pyquaternion import Quaternion
import simplejson as json

from mmdet3d.datasets import NuScenesDataset
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils import splits as truckscenes_splits


def get_sensor_info(trsc, sensor_token):
    """
    Gets basic sensor data, including the sensor-to-ego transform.
    """
    sd_rec = trsc.get('sample_data', sensor_token)
    cs_record = trsc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_record = trsc.get('ego_pose', sd_rec['ego_pose_token'])

    return {
        'data_path': str(trsc.get_sample_data_path(sd_rec['token'])),
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp'],
    }


def fill_occ_infos(trsc,
                   occ_anno,
                   token2name,
                   train_scenes,
                   val_scenes,
                   test=False,
                   max_sweeps=10):
    """
    Generates the train/val infos from the raw data for SparseOcc.
    """
    train_trsc_infos = []
    val_trsc_infos = []
    scene_infos = occ_anno.get('scene_infos', {})

    # 1. Define your target classes and the mapping from raw names
    TARGET_CLASSES = {
        'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
        'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
        'animal', 'traffic_sign', 'other_vehicle', 'train'
        # Note: 'background' and 'free' are for voxel labels, not 3D boxes
    }

    CLASS_MAPPING = {
        # noise
        'human.pedestrian.personal_mobility': 'noise',
        'human.pedestrian.stroller': 'noise',
        'human.pedestrian.wheelchair': 'noise',
        'movable_object.debris': 'noise',
        'movable_object.pushable_pullable': 'noise',
        'static_object.bicycle_rack': 'noise',
        'vehicle.emergency.ambulance': 'noise',
        'vehicle.emergency.police': 'noise',
        # barrier
        'movable_object.barrier': 'barrier',
        # bicycle
        'vehicle.bicycle': 'bicycle',
        # bus
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        # car
        'vehicle.car': 'car',
        # construction_vehicle
        'vehicle.construction': 'construction_vehicle',
        # motorcycle
        'vehicle.motorcycle': 'motorcycle',
        # pedestrian
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        # traffic_cone
        'movable_object.trafficcone': 'traffic_cone',
        # trailer
        'vehicle.trailer': 'trailer',
        'vehicle.ego_trailer': 'trailer',
        # truck
        'vehicle.truck': 'truck',
        # animal
        'animal': 'animal',
        # traffic_sign
        'static_object.traffic_sign': 'traffic_sign',
        # other_vehicle
        'vehicle.other': 'other_vehicle',
        # train
        'vehicle.train': 'train',
    }

    for sample in mmcv.track_iter_progress(trsc.sample):
        scene_token = sample['scene_token']
        scene_name = token2name[sample['scene_token']]

        # Check if the sample belongs to a scene we want to process
        if scene_token not in train_scenes and scene_token not in val_scenes:
            print(f"{scene_token} not in train and val scenes. Skipping.")
            continue

        primary_lidar_channel = 'LIDAR_LEFT'
        primary_lidar_token = sample['data'].get(primary_lidar_channel)

        if not primary_lidar_token:
            print(f"Primary lidar token {primary_lidar_token} not found. Skipping.")
            continue

        sd_rec = trsc.get('sample_data', primary_lidar_token)
        sample_token = sd_rec['sample_token']

        if sample_token not in scene_infos.get(scene_name, {}):
            print(f"{sample_token} not in scene_infos. Skipping.")
            continue

        occ_sample = scene_infos[scene_name][sample_token]

        pose_record = trsc.get('ego_pose', sd_rec['ego_pose_token'])
        ego2global_trans = np.array(pose_record['translation'])
        ego2global_rot_list = pose_record['rotation']

        cs_record = trsc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        lidar2ego_trans = np.array(cs_record['translation'])
        lidar2ego_rot_list = cs_record['rotation']

        ego2global_rot_quat = Quaternion(ego2global_rot_list)
        lidar2ego_rot_quat = Quaternion(lidar2ego_rot_list)

        primary_lidar_path, boxes, _ = trsc.get_sample_data(primary_lidar_token)

        info = {
            'lidar_path': primary_lidar_path,
            'token': sample['token'],
            'sweeps': [],
            'cams': {},
            'lidar2ego_translation': lidar2ego_trans,
            'lidar2ego_rotation': lidar2ego_rot_list,
            'ego2global_translation': ego2global_trans,
            'ego2global_rotation': ego2global_rot_list,
            'timestamp': sample['timestamp'],
        }

        # Add occupancy ground truth path
        if 'gt_path' in occ_sample:
            info['occ_gt_path'] = occ_sample['gt_path']

        """# Collect camera data
        camera_types = ['CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_BACK']
        for cam_channel in camera_types:
            if cam_channel in sample['data']:
                cam_token = sample['data'][cam_channel]
                cam_info = get_sensor_info(trsc, cam_token)
                _, _, cam_intrinsic = trsc.get_sample_data(cam_token)
                cam_info['cam_intrinsic'] = cam_intrinsic
                info['cams'][cam_channel] = cam_info"""

        # Collect and transform camera data
        camera_types = ['CAMERA_LEFT_FRONT', 'CAMERA_RIGHT_FRONT', 'CAMERA_LEFT_BACK', 'CAMERA_RIGHT_BACK']
        for cam_channel in camera_types:
            if cam_channel in sample['data']:
                cam_token = sample['data'][cam_channel]

                cam_basic_info = get_sensor_info(trsc, cam_token)
                # Convert this camera's rotation to a Quaternion object as well
                sensor2ego_rot_quat = Quaternion(cam_basic_info['sensor2ego_rotation'])
                sensor2ego_trans = np.array(cam_basic_info['sensor2ego_translation'])

                # 3. Perform calculations using the Quaternion objects
                sensor2global_rot = ego2global_rot_quat * sensor2ego_rot_quat
                sensor2global_trans = ego2global_rot_quat.rotate(sensor2ego_trans) + ego2global_trans
                sensor2lidar_rot = lidar2ego_rot_quat.inverse * sensor2ego_rot_quat
                sensor2lidar_trans = lidar2ego_rot_quat.inverse.rotate(sensor2ego_trans - lidar2ego_trans)

                _, _, cam_intrinsic = trsc.get_sample_data(cam_token)

                # Assemble the final dictionary with the correct keys and formats
                cam_info = {
                    'data_path': cam_basic_info['data_path'],
                    'type': cam_channel,
                    'timestamp': cam_basic_info['timestamp'],
                    'sensor2lidar_rotation': sensor2lidar_rot.rotation_matrix,  # Convert to 3x3 matrix
                    'sensor2lidar_translation': sensor2lidar_trans,
                    'cam_intrinsic': cam_intrinsic,
                    'sensor2global_rotation': sensor2global_rot.rotation_matrix,  # Convert to 3x3 matrix
                    'sensor2global_translation': sensor2global_trans
                }
                info['cams'][cam_channel] = cam_info

        """# Collect sweeps for the primary lidar
        sweeps = []
        current_sweep_token = primary_lidar_token
        while len(sweeps) < max_sweeps:
            current_sd_rec = trsc.get('sample_data', current_sweep_token)
            if not current_sd_rec['prev']:
                break
            prev_sweep_token = current_sd_rec['prev']
            sweep_info = get_sensor_info(trsc, prev_sweep_token)
            sweeps.append(sweep_info)
            current_sweep_token = prev_sweep_token
        info['sweeps'] = sweeps"""

        if not test:
            sample_annotation_tokens = sample['anns']
            annotations = [trsc.get('sample_annotation', token) for token in sample_annotation_tokens]
            _, boxes, _ = trsc.get_sample_data(primary_lidar_token, box_vis_level=1)

            # --- Start of Changed Section ---
            # 2. Create temporary lists to hold filtered annotations
            final_gt_boxes = []
            final_gt_names = []
            final_velocity = []
            final_num_lidar_pts = []
            final_num_radar_pts = []
            final_valid_flags = []

            raw_names = [b.name for b in boxes]
            raw_velocities = np.array([trsc.box_velocity(token)[:2] for token in sample_annotation_tokens])
            raw_velocities = np.nan_to_num(raw_velocities)  # Clean NaN values

            # 3. Loop through each box and apply the mapping
            for i, raw_name in enumerate(raw_names):
                # Apply mapping if the raw_name is in our map, otherwise use the raw_name
                mapped_name = CLASS_MAPPING.get(raw_name, raw_name)

                # Check if the final name is in our list of target classes
                if mapped_name in TARGET_CLASSES:
                    # If it's a valid class, keep this box and its data
                    final_gt_names.append(mapped_name)

                    # Get box data
                    loc = boxes[i].center
                    dim = boxes[i].wlh
                    rot = boxes[i].orientation.yaw_pitch_roll[0]
                    gt_box = np.concatenate([loc, dim, [-rot - np.pi / 2]])
                    final_gt_boxes.append(gt_box)

                    # Transform and keep velocity
                    velo = np.array([*raw_velocities[i], 0.0])
                    velo = velo @ np.linalg.inv(ego2global_rot_quat.rotation_matrix).T @ np.linalg.inv(
                        lidar2ego_rot_quat.rotation_matrix).T
                    final_velocity.append(velo[:2])

                    # Keep other metadata
                    final_num_lidar_pts.append(annotations[i]['num_lidar_pts'])
                    final_num_radar_pts.append(annotations[i]['num_radar_pts'])
                    final_valid_flags.append((annotations[i]['num_lidar_pts'] + annotations[i]['num_radar_pts']) > 0)

            # 4. Convert the filtered lists to NumPy arrays and save them
            if final_gt_boxes:  # Only add if there are any valid boxes left
                info['gt_boxes'] = np.array(final_gt_boxes, dtype=np.float32)
                info['gt_names'] = np.array(final_gt_names)
                info['gt_velocity'] = np.array(final_velocity, dtype=np.float32)
                info['num_lidar_pts'] = np.array(final_num_lidar_pts)
                info['num_radar_pts'] = np.array(final_num_radar_pts)
                info['valid_flag'] = np.array(final_valid_flags, dtype=bool)
            else:  # If all boxes were filtered out, save empty arrays
                print("Warning: no valid ground truth boxes")
                info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
                info['gt_names'] = np.array([])
                info['gt_velocity'] = np.zeros((0, 2), dtype=np.float32)
                info['num_lidar_pts'] = np.array([])
                info['num_radar_pts'] = np.array([])
                info['valid_flag'] = np.array([], dtype=bool)

        if sample['scene_token'] in train_scenes:
            train_trsc_infos.append(info)
        else:
            val_trsc_infos.append(info)

    return train_trsc_infos, val_trsc_infos


def create_truckscenes_occ_infos(root_path,
                                 annotation_path,
                                 out_dir,
                                 info_prefix,
                                 version='v1.0-trainval',
                                 max_sweeps=10):
    """
    Main function to create the info file.
    """
    trsc = TruckScenes(version=version, dataroot=root_path, verbose=True)

    if version == 'v1.0-trainval':
        train_scene_names, val_scene_names = truckscenes_splits.train, truckscenes_splits.val
    elif version == 'v1.0-mini':
        train_scene_names, val_scene_names = truckscenes_splits.mini_train, truckscenes_splits.mini_val
    elif version == 'v1.0-test':
        train_scene_names, val_scene_names = truckscenes_splits.test, []
    else:
        raise ValueError(f"Version '{version}' is not supported.")

    with open(os.path.join(annotation_path, 'annotations.json'), 'r') as f:
        occ_anno = json.load(f)

    token2name = {scene['token']: scene['name'] for scene in trsc.scene}
    name2token = {name: token for token, name in token2name.items()}
    train_scenes = {name2token[name] for name in train_scene_names if name in name2token}
    val_scenes = {name2token[name] for name in val_scene_names if name in name2token}

    test = 'test' in version
    print(f'Processing {len(train_scenes)} train scenes and {len(val_scenes)} val scenes.')

    train_infos, val_infos = fill_occ_infos(
        trsc, occ_anno, token2name, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    print(f'Created {len(train_infos)} training samples and {len(val_infos)} validation samples.')

    if test:
        train_data = dict(infos=train_infos, metadata=metadata)
        info_path_test = osp.join(out_dir, f'{info_prefix}_infos_test.pkl')
        mmcv.dump(train_data, info_path_test)
        print(f"Saved test info to {info_path_test}")
    else:
        train_data = dict(infos=train_infos, metadata=metadata)
        info_path_train = osp.join(out_dir, f'{info_prefix}_infos_train_sweep.pkl')
        mmcv.dump(train_data, info_path_train)
        print(f"Saved training info to {info_path_train}")

        val_data = dict(infos=val_infos, metadata=metadata)
        info_path_val = osp.join(out_dir, f'{info_prefix}_infos_val_sweep.pkl')
        mmcv.dump(val_data, info_path_val)
        print(f"Saved validation info to {info_path_val}")
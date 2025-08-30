import os
import pickle
from tqdm import tqdm
from truckscenes.truckscenes import TruckScenes
from truckscenes.utils import splits
import argparse


def create_truckscenes_infos(data_root, version, save_path, gt_path):
    """
    Creates the info .pkl file for the TruckScenes dataset.

    Args:
        data_root (str): Path to the root of the TruckScenes dataset.
        version (str): Dataset version, e.g., 'v1.0-trainval'.
        out_path (str): Path to save the output .pkl file.
    """
    print(f"Loading TruckScenes {version} from {data_root}...")
    # 1. Initialize the devkit's main class
    trsc = TruckScenes(version=version, dataroot=data_root, verbose=True)

    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError(f'Version {version} is not supported.')

    print("Splits created....")

    if version == 'v1.0-test':
        print('Creating test scenes: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    # These will be the main dictionaries in our final .pkl file
    all_infos_train, all_metadata_train = {}, []
    all_infos_val, all_metadata_val = {}, []
    # For the test set, which has no val split
    all_infos_test, all_metadata_test = {}, []

    print("Processing scenes...")
    for scene in tqdm(trsc.scene):
        scene_token = scene['token']
        scene_name = scene['name']

        # This list will hold all frames for the current scene
        current_scene_frames = []
        current_scene_metadata = []

        # 3. Get the first sample (frame) and walk the linked list
        current_sample_token = scene['first_sample_token']
        frame_index = 0

        while current_sample_token:
            sample = trsc.get('sample', current_sample_token)

            data_info = {}

            for channel, sensor_token in sample['data'].items():
                sample_data = trsc.get('sample_data', sensor_token)
                calib_sensor = trsc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                sensor = trsc.get('sensor', calib_sensor['sensor_token'])
                ego_pose = trsc.get('ego_pose', sample_data['ego_pose_token'])

                # Assemble the sensor-specific dictionary
                sensor_dict = {
                    'token': sample_data['token'],
                    'sample_token': sample_data['sample_token'],
                    'ego_pose_token': sample_data['ego_pose_token'],
                    'calibrated_sensor_token': sample_data['calibrated_sensor_token'],
                    'timestamp': sample_data['timestamp'],
                    'fileformat': sample_data['fileformat'],
                    'is_key_frame': sample_data['is_key_frame'],
                    'height': sample_data['height'],
                    'width': sample_data['width'],
                    'filename': sample_data['filename'],
                    'prev': sample_data['prev'],
                    'next': sample_data['next'],
                    'sensor_modality': sensor['modality'],
                    'channel': sensor['channel'],
                    'pose': {
                        'token': ego_pose['token'],
                        'timestamp': ego_pose['timestamp'],
                        'rotation': ego_pose['rotation'],
                        'translation': ego_pose['translation']
                    },
                    'calib': {
                        'token': calib_sensor['token'],
                        'sensor_token': calib_sensor['sensor_token'],
                        'translation': calib_sensor['translation'],
                        'rotation': calib_sensor['rotation'],
                        # Add camera intrinsics only if it's a camera
                        'camera_intrinsic': calib_sensor['camera_intrinsic']
                    }
                }
                data_info[channel] = sensor_dict

            #occ_path = os.path.join(data_root, "gts", scene['name'], sample['token'], "labels.npz") #local
            occ_path = os.path.join(gt_path, "gts", scene['name'], sample['token'], "labels.npz")
            is_key_frame = trsc.get('sample_data', sample['data']['LIDAR_LEFT'])['is_key_frame']
            # --- Assemble the complete dictionary for this frame ---
            frame_info = {
                'token': sample['token'],
                'timestamp': sample['timestamp'],
                'prev': sample['prev'],
                'next': sample['next'],
                'scene_token': sample['scene_token'],
                'data': data_info,
                'anns': [trsc.get('sample_annotation', ann_token) for ann_token in sample['anns']],
                'is_key_frame': is_key_frame,
                'occ_path': occ_path
            }

            current_scene_frames.append(frame_info)

            # If it's a keyframe, add it to our metadata index
            if is_key_frame:
                current_scene_metadata.append((scene_token, frame_index))

            # Move to the next frame in the scene
            current_sample_token = sample['next']
            frame_index += 1

        if scene_name in train_scenes:
            all_infos_train[scene_token] = current_scene_frames
            all_metadata_train.extend(current_scene_metadata)
        elif scene_name in val_scenes:
            all_infos_val[scene_token] = current_scene_frames
            all_metadata_val.extend(current_scene_metadata)

    if version == "v1.0-test":
        test_data = {'infos': all_infos_train, 'metadata': all_metadata_train}
        # output_path_test = os.path.join(data_root, "truckscenes_infos_test_sweeps_occ.pkl") # local
        output_path_test = os.path.join(save_path, "truckscenes_infos_test_sweeps_occ.pkl")
        print(
            f"\nSaving test set with {len(all_infos_train)} scenes and {len(all_metadata_train)} keyframes to {output_path_test}")
        with open(output_path_test, 'wb') as f:
            pickle.dump(test_data, f)
    else:
        # Save the training data
        train_data = {'infos': all_infos_train, 'metadata': all_metadata_train}
        #output_path_train = os.path.join(data_root, f"truckscenes_infos_train_sweeps_occ.pkl")
        output_path_train = os.path.join(save_path, f"truckscenes_infos_train_sweeps_occ.pkl")
        print(
            f"\nSaving train set with {len(all_infos_train)} scenes and {len(all_metadata_train)} keyframes to {output_path_train}")
        with open(output_path_train, 'wb') as f:
            pickle.dump(train_data, f)

        # Save the validation data
        val_data = {'infos': all_infos_val, 'metadata': all_metadata_val}
        #output_path_val = os.path.join(data_root, f"truckscenes_infos_val_sweeps_occ.pkl")
        output_path_val = os.path.join(save_path, f"truckscenes_infos_val_sweeps_occ.pkl")
        print(
            f"Saving val set with {len(all_infos_val)} scenes and {len(all_metadata_val)} keyframes to {output_path_val}")
        with open(output_path_val, 'wb') as f:
            pickle.dump(val_data, f)

    print("Done!")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TruckScenes info files.')
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to the root of the TruckScenes dataset (e.g., /truckscenes/v1.0-trainval).')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory where the output .pkl files will be saved.')
    parser.add_argument('--version', type=str, required=True,
                        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                        help='The dataset version to process.')
    parser.add_argument('--gt-dir', type=str, required=True, help='Path to the occupancy ground truth directory.')
    args = parser.parse_args()

    # Create the save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"Starting data generation for version: {args.version}")

    """save_path_base = '/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2'
    #data_root = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini'
    data_root = '/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes'
    version = 'v1.0-mini'
    #data_root = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
    # data_root = '/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes'
    #version = 'v1.0-trainval'
    #data_root = '/home/max/ssd/Masterarbeit/TruckScenes/test/v1.0-test'
    # data_root = '/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes'
    #version = 'v1.0-test'
    save_path = os.path.join(save_path_base, version)"""


    create_truckscenes_infos(
        data_root=args.data_root,
        version=args.version,
        save_path=args.save_dir,
        gt_path=args.gt_dir
    )
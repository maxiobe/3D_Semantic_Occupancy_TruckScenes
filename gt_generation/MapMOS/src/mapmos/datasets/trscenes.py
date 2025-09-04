import importlib
import os
import sys
from pathlib import Path
from typing import List

import numpy as np
from truckscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

class TruckScenesDataset:
    def __init__(self, data_dir: Path, sequence: int, *_, **__):
        try:
            importlib.import_module("truckscenes")
        except ModuleNotFoundError:
            print("truckscenes-devkit is not installed on your system")
            print('run "pip install truckscenes-devkit"')
            sys.exit(1)

        trsc_version: str = "v1.0-trainval"
        split: str = "train"
        self.lidar_name: str = "LIDAR_LEFT"

        # Lazy loading
        from truckscenes.truckscenes import TruckScenes
        from truckscenes.utils.splits import create_splits_logs

        self.sequence_id = str(sequence).zfill(4)
        self.trsc = TruckScenes(version=trsc_version, dataroot=str(data_dir), verbose=True)
        self.scene_name = f"scene-{self.sequence_id}"

        # Check available scenes
        available = [s["name"] for s in self.trsc.scene]
        if self.scene_name not in available:
            print(f'[ERROR] Sequence "{self.sequence_id}" not available scenes')
            print("\nAvailable scenes:")
            for name in available:
                print(f"  {name}")
            sys.exit(1)

        # Load point cloud reader
        from truckscenes.utils.data_classes import LidarPointCloud
        self.load_point_cloud = LidarPointCloud.from_file

        scene_token = next(s["token"] for s in self.trsc.scene if s["name"] == self.scene_name)

        # Use only the samples from the current split.
        self.lidar_tokens = self._get_lidar_tokens(scene_token)
        self.gt_poses = self._load_poses()

        # Define which attributes are considered moving
        self.moving_attributes = ["vehicles.moving", "cycle.with_rider", "pedestrian.moving"]

    def __len__(self):
        return len(self.lidar_tokens)

    def __getitem__(self, idx):
        points = self.read_point_cloud(self.lidar_tokens[idx])
        timestamps = np.zeros(len(points))
        labels = self.read_labels(points, idx)
        return points, timestamps, labels

    def read_labels(self, points, idx):
        lidar_token = self.lidar_tokens[idx]
        is_key_frame = self.trsc.get("sample_data", lidar_token)["is_key_frame"]

        sample_token = self.trsc.get("sample_data", lidar_token)["sample_token"]
        annotation_tokens = self.trsc.get("sample", sample_token)["anns"]

        # Annotations are only available for keyframes
        if not is_key_frame or len(annotation_tokens) == 0:
            return np.full((len(points), 1), -1, dtype=np.int32)

        points_hom = np.hstack((points, np.ones((len(points), 1))))
        global_points_hom = (self.global_pose(idx) @ points_hom.T).T
        labels = np.zeros((len(points), 1), dtype=np.int32)

        for annotation_token in annotation_tokens:
            annotation = self.trsc.get("sample_annotation", annotation_token)
            attribute_token = annotation["attribute_tokens"]
            if (
                    len(attribute_token) > 0
                    and self.trsc.get("attribute", attribute_token[0])["name"] in self.moving_attributes
            ):
                box_center = self.trsc.get_box(annotation_token).center
                box_rotation = self.trsc.get_box(annotation_token).rotation_matrix
                box_wlh = self.trsc.get_box(annotation_token).wlh
                box_pose = np.vstack(
                    [
                        np.hstack([box_rotation, box_center.reshape(-1, 1)]),
                        np.array([[0, 0, 0, 1]]),
                    ]
                )
                local_points = (np.linalg.inv(box_pose) @ global_points_hom.T).T[:, :3]
                abs_local_points = np.abs(local_points)
                mask = abs_local_points[:, 0] < box_wlh[1] / 2
                mask = np.logical_and(mask, abs_local_points[:, 1] <= box_wlh[0] / 2)
                mask = np.logical_and(mask, abs_local_points[:, 2] <= box_wlh[2] / 2)
                labels[mask] = 1.0

        return labels

    @staticmethod
    def get_timestamps(points):
        x = points[:, 0]
        y = points[:, 1]
        yaw = -np.arctan2(y, x)
        timestamps = 0.5 * (yaw / np.pi + 1.0)
        return timestamps

    def global_pose(self, idx):
        sd_record_lid = self.trsc.get("sample_data", self.lidar_tokens[idx])
        cs_record_lid = self.trsc.get("calibrated_sensor", sd_record_lid["calibrated_sensor_token"])
        ep_record_lid = self.trsc.get("ego_pose", sd_record_lid["ego_pose_token"])

        car_to_velo = transform_matrix(
            cs_record_lid["translation"],
            Quaternion(cs_record_lid["rotation"]),
        )
        pose_car = transform_matrix(
            ep_record_lid["translation"],
            Quaternion(ep_record_lid["rotation"]),
        )
        return pose_car @ car_to_velo

    def read_point_cloud(self, token: str):
        filename = self.trsc.get("sample_data", token)["filename"]
        pcl = self.load_point_cloud(os.path.join(self.trsc.dataroot, filename))
        points = pcl.points.T[:, :3]
        return points.astype(np.float64)

    def _load_poses(self) -> np.ndarray:
        from truckscenes.utils.geometry_utils import transform_matrix
        from pyquaternion import Quaternion

        poses = np.empty((len(self), 4, 4), dtype=np.float32)
        for i, lidar_token in enumerate(self.lidar_tokens):
            sd_record_lid = self.trsc.get("sample_data", lidar_token)
            cs_record_lid = self.trsc.get(
                "calibrated_sensor", sd_record_lid["calibrated_sensor_token"]
            )
            ep_record_lid = self.trsc.get("ego_pose", sd_record_lid["ego_pose_token"])

            car_to_velo = transform_matrix(
                cs_record_lid["translation"],
                Quaternion(cs_record_lid["rotation"]),
            )
            pose_car = transform_matrix(
                ep_record_lid["translation"],
                Quaternion(ep_record_lid["rotation"]),
            )

            poses[i:, :] = pose_car @ car_to_velo

        # Convert from global coordinate poses to local poses
        first_pose = poses[0, :, :]
        poses = np.linalg.inv(first_pose) @ poses
        return poses

    def _get_scene_token(self, split_logs: List[str]) -> str:
        """
        Convenience function to get the samples in a particular split.
        :param split_logs: A list of the log names in this split.
        :return: The list of samples.
        """
        scene_tokens = [s["token"] for s in self.trsc.scene if s["name"] == self.scene_name][0]
        scene = self.trsc.get("scene", scene_tokens)
        log = self.trsc.get("log", scene["log_token"])
        return scene["token"] if log["logfile"] in split_logs else ""

    def _get_lidar_tokens(self, scene_token: str) -> List[str]:
        # Get records from DB.
        scene_rec = self.trsc.get("scene", scene_token)
        start_sample_rec = self.trsc.get("sample", scene_rec["first_sample_token"])
        sd_rec = self.trsc.get("sample_data", start_sample_rec["data"][self.lidar_name])

        # Make list of frames
        cur_sd_rec = sd_rec
        sd_tokens = [cur_sd_rec["token"]]
        while cur_sd_rec["next"] != "":
            cur_sd_rec = self.trsc.get("sample_data", cur_sd_rec["next"])
            sd_tokens.append(cur_sd_rec["token"])
        return sd_tokens
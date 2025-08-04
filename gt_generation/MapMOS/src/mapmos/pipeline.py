# MIT License
#
# Copyright (c) 2023 Benedikt Mersch, Tiziano Guadagnino, Ignacio Vizzo, Cyrill Stachniss
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import time
from collections import deque
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from kiss_icp.pipeline import OdometryPipeline
from tqdm.auto import trange

from mapmos.config import load_config
from mapmos.mapmos_net import MapMOSNet
from mapmos.mapping import VoxelHashMap
from mapmos.metrics import get_confusion_matrix
from mapmos.odometry import Odometry
from mapmos.utils.pipeline_results import MOSPipelineResults
from mapmos.utils.save import KITTIWriter, PlyWriter, StubWriter
from mapmos.utils.visualizer import MapMOSVisualizer, StubVisualizer


class MapMOSPipeline(OdometryPipeline):
    def __init__(
        self,
        dataset,
        weights: Path,
        config: Optional[Path] = None,
        visualize: bool = False,
        save_ply: bool = False,
        save_kitti: bool = False,
        n_scans: int = -1,
        jump: int = 0,
        initial_guesses_relative: Optional[List[np.ndarray]] = None
    ):
        self._dataset = dataset
        self._n_scans = (
            len(self._dataset) - jump if n_scans == -1 else min(len(self._dataset) - jump, n_scans)
        )
        self._first = jump
        self._last = self._first + self._n_scans

        self._initial_guesses_relative = initial_guesses_relative

        # Config and output dir
        self.config = load_config(config)
        self.results_dir = None

        # Pipeline
        state_dict = {
            k.replace("mos.", ""): v for k, v in torch.load(weights)["state_dict"].items()
        }
        self.model = MapMOSNet(self.config.mos.voxel_size_mos)

        self.model.load_state_dict(state_dict)
        self.model.cuda().eval().freeze()

        self.odometry = Odometry(self.config.data, self.config.odometry)
        self.belief = VoxelHashMap(
            voxel_size=self.config.mos.voxel_size_belief,
            max_distance=self.config.mos.max_range_belief,
        )
        self.buffer = deque(maxlen=self.config.mos.delay_mos)

        # Results
        self.results = MOSPipelineResults()
        self.poses = np.zeros((self._n_scans, 4, 4))
        self.has_gt = hasattr(self._dataset, "gt_poses")
        self.gt_poses = self._dataset.gt_poses[self._first : self._last] if self.has_gt else None
        self.dataset_name = self._dataset.__class__.__name__
        self.dataset_sequence = (
            self._dataset.sequence_id
            if hasattr(self._dataset, "sequence_id")
            else os.path.basename(self._dataset.data_dir)
        )
        self.times_mos = []
        self.times_belief = []
        self.confusion_matrix_belief = torch.zeros(2, 2)

        self.frame_predictions = []

        # Visualizer
        self.visualize = visualize
        self.visualizer = MapMOSVisualizer() if visualize else StubVisualizer()
        self.visualizer.set_voxel_size(self.config.mos.voxel_size_belief)
        self.ply_writer = PlyWriter() if save_ply else StubWriter()
        self.kitti_writer = KITTIWriter() if save_kitti else StubWriter()

    # Public interface  ------
    def run(self):
        self._create_output_dir()
        with torch.no_grad():
            self._run_pipeline()
        self._run_evaluation()
        self._write_result_poses()
        self._write_gt_poses()
        self._write_cfg()
        self._write_log()
        return self.results, self.frame_predictions

    def _preprocess(self, points, min_range, max_range):
        ranges = np.linalg.norm(points - self.odometry.current_location(), axis=1)
        mask = ranges <= max_range if max_range > 0 else np.ones_like(ranges, dtype=bool)
        mask = np.logical_and(mask, ranges >= min_range)
        return mask

    # Private interface  ------
    def _run_pipeline(self):
        pbar = trange(self._first, self._last, unit=" frames", dynamic_ncols=True)
        for scan_index in pbar:
            i = scan_index - self._first
            current_user_guess = None
            if self._initial_guesses_relative is not None and i < len(self._initial_guesses_relative):
                current_user_guess = self._initial_guesses_relative[i]

            local_scan, timestamps, gt_labels_unfiltered = self._next(scan_index)
            map_points_unfiltered_from_odom, map_indices_unfiltered_from_odom = self.odometry.get_map_points()

            # scan_points_unfiltered is all N_orig points from local_scan, in global frame
            scan_points_unfiltered = self.odometry.register_points(local_scan, timestamps, scan_index, user_provided_initial_guess=current_user_guess)

            self.poses[scan_index - self._first] = self.odometry.last_pose

            min_range_mos = self.config.mos.min_range_mos
            max_range_mos = self.config.mos.max_range_mos

            # scan_mask applies to scan_points_unfiltered
            scan_mask = self._preprocess(scan_points_unfiltered, min_range_mos, max_range_mos)

            # These are the M points (M <= N_orig) actually processed by the model
            scan_points_filtered_cuda = torch.tensor(scan_points_unfiltered[scan_mask], dtype=torch.float32, device="cuda")
            # Corresponding GT labels for the M filtered points
            gt_labels_filtered = gt_labels_unfiltered[scan_mask]

            map_mask = self._preprocess(map_points_unfiltered_from_odom, min_range_mos, max_range_mos)
            map_points_filtered_cuda = torch.tensor(map_points_unfiltered_from_odom[map_mask], dtype=torch.float32, device="cuda")
            map_indices_filtered_cuda = torch.tensor(map_indices_unfiltered_from_odom[map_mask], dtype=torch.float32, device="cuda")

            start_time = time.perf_counter_ns()
            pred_logits_scan_filtered, pred_logits_map_filtered = self.model.predict(
                scan_points_filtered_cuda,
                map_points_filtered_cuda,
                scan_index * torch.ones(len(scan_points_filtered_cuda)).type_as(scan_points_filtered_cuda),
                map_indices_filtered_cuda,
            )
            self.times_mos.append(time.perf_counter_ns() - start_time)

            # Detach, move to CPU
            pred_logits_scan_filtered_cpu = pred_logits_scan_filtered.detach().cpu().numpy().astype(np.float64)
            pred_logits_map_filtered_cpu = pred_logits_map_filtered.detach().cpu().numpy().astype(np.float64)
            scan_points_filtered_cpu = scan_points_filtered_cuda.cpu().numpy().astype(np.float64)
            map_points_filtered_cpu = map_points_filtered_cuda.cpu().numpy().astype(np.float64)
            torch.cuda.empty_cache()

            pred_labels_scan_filtered = self.model.to_label(pred_logits_scan_filtered_cpu)
            pred_labels_map_filtered = self.model.to_label(pred_logits_map_filtered_cpu)

            # Probabilistic Volumetric Fusion of predictions within the belief range
            # This section uses the *filtered* scan and map points
            map_logits_positive_mask = pred_logits_map_filtered_cpu > 0
            map_points_for_belief_mask = np.logical_and(
                map_logits_positive_mask, self._preprocess(map_points_filtered_cpu, 0.0, self.config.mos.max_range_belief)
            )
            scan_points_for_belief_mask = self._preprocess(scan_points_filtered_cpu, 0.0, self.config.mos.max_range_belief)

            # Ensure arrays are not empty before vstack
            belief_update_scan_points = scan_points_filtered_cpu[scan_points_for_belief_mask]
            belief_update_map_points = map_points_filtered_cpu[map_points_for_belief_mask]

            belief_update_scan_logits = pred_logits_scan_filtered_cpu[scan_points_for_belief_mask].reshape(-1, 1)
            belief_update_map_logits = pred_logits_map_filtered_cpu[map_points_for_belief_mask].reshape(-1, 1)

            points_stacked = np.vstack([belief_update_scan_points, belief_update_map_points])
            logits_stacked = np.vstack(
                [
                    belief_update_scan_logits,
                    belief_update_map_logits,
                ]
            ).reshape(-1)

            start_time = time.perf_counter_ns()
            self.belief.update_belief(points_stacked, logits_stacked)
            self.belief.get_belief(scan_points_filtered_cpu)
            self.times_belief.append(time.perf_counter_ns() - start_time)

            self.visualizer.update(
                scan_points_filtered_cpu,
                map_points_filtered_cpu,
                pred_labels_scan_filtered,
                pred_labels_map_filtered,
                self.belief,
                self.odometry.last_pose,
            )

            # Store all necessary data in the buffer
            self.buffer.append([
                scan_index,  # 0
                scan_points_unfiltered,  # 1: All N_orig points, global frame (np.ndarray)
                gt_labels_unfiltered,  # 2: GT labels for N_orig points (np.ndarray)
                scan_mask,  # 3: Boolean mask of length N_orig (np.ndarray)
                scan_points_filtered_cpu  # 4: The M points processed by model (np.ndarray)
            ])

            if len(self.buffer) == self.buffer.maxlen:
                # Unpack all 5 items correctly
                idx, s_unfilt, gt_unfilt, mask, s_filt_cpu = self.buffer.popleft()
                self.process_final_prediction(idx, s_unfilt, gt_unfilt, mask, s_filt_cpu)

            # Clean up
            self.belief.remove_voxels_far_from_location(self.odometry.current_location())

        # Clear buffer at the end
        while len(self.buffer) != 0:
            # Unpack all 5 items correctly
            idx, s_unfilt, gt_unfilt, mask, s_filt_cpu = self.buffer.popleft()
            self.process_final_prediction(idx, s_unfilt, gt_unfilt, mask, s_filt_cpu)

    def process_final_prediction(self,
                                 query_index,
                                 query_points_unfiltered_global,  # All N_orig points from scan_points_unfiltered
                                 query_gt_labels_unfiltered,  # GT for N_orig points
                                 query_scan_mask,  # The mask applied to get filtered points
                                 query_points_filtered  # The M points that were processed by model & belief
                                 ):

        # Get belief predictions ONLY for the points that were actually processed
        belief_logits_for_filtered_points = self.belief.get_belief(query_points_filtered)
        predicted_labels_for_filtered_points = self.model.to_label(belief_logits_for_filtered_points)  # Shape (M,)

        # Create a full-sized predicted label array for ALL original points for this scan
        num_original_points = query_points_unfiltered_global.shape[0]

        # Initialize with a default "unclassified" or "out-of-model-range" label, e.g., -1
        # (0 for static, 1 for dynamic, -1 for not processed by MOS model)
        full_predicted_labels = np.full(num_original_points, -1, dtype=np.int32)

        # Use the scan_mask to place the actual predictions into the full array
        # query_scan_mask is a boolean array of length num_original_points
        # predicted_labels_for_filtered_points has length M (number of True in query_scan_mask)
        full_predicted_labels[query_scan_mask] = predicted_labels_for_filtered_points

        # For confusion matrix, use only the points that have actual predictions and corresponding GT
        gt_labels_for_filtered_points = query_gt_labels_unfiltered[query_scan_mask]

        self.confusion_matrix_belief += get_confusion_matrix(
            torch.tensor(predicted_labels_for_filtered_points, dtype=torch.int32),
            torch.tensor(gt_labels_for_filtered_points, dtype=torch.int32),
        )

        current_frame_data = {
            "scan_index": query_index,
            "points": query_points_unfiltered_global,  # All original N_orig points (global frame)
            "predicted_labels": full_predicted_labels,  # Labels for all N_orig points (-1 for unprocessed by model)
            "gt_labels": query_gt_labels_unfiltered  # GT labels for all N_orig points
        }
        self.frame_predictions.append(current_frame_data)

        self.ply_writer.write(
            query_points_unfiltered_global,  # Save all points
            full_predicted_labels,  # Save labels for all points (-1 where not predicted)
            query_gt_labels_unfiltered,  # Save GT for all points
            filename=f"{self.results_dir}/ply/{query_index:06}.ply",
        )
        self.kitti_writer.write(
            full_predicted_labels,  # This provides labels for all original points
            filename=f"{self.results_dir}/bin/sequences/{self.dataset_sequence}/predictions/{query_index:06}.label",
        )

    def _next(self, idx):
        dataframe = self._dataset[idx]
        try:
            local_scan, timestamps, gt_labels = dataframe
        except ValueError:
            try:
                local_scan, timestamps = dataframe
                gt_labels = -1 * np.ones(local_scan.shape[0])
            except ValueError:
                local_scan = dataframe
                gt_labels = -1 * np.ones(local_scan.shape[0])
                timestamps = np.zeros(local_scan.shape[0])
        return local_scan.reshape(-1, 3), timestamps.reshape(-1), gt_labels.reshape(-1)

    def _run_evaluation(self):
        if self.has_gt:
            self.results.eval_odometry(self.poses, self.gt_poses)
        self.results.eval_mos(self.confusion_matrix_belief, desc="\nBelief")
        self.results.eval_fps(self.times_mos, desc="Average Frequency MOS")
        self.results.eval_fps(self.times_belief, desc="Average Frequency Belief")

import numpy as np
from typing import List, Optional

#################################### Dataset for Kiss-ICP Lidar Odometry ##############################################
class InMemoryDataset:
    """
    A custom dataset class for KISS-ICP using data loaded into memory
    """
    def __init__(self,
                 lidar_scans: List[np.ndarray],
                 gt_relative_poses: List[np.ndarray],
                 timestamps: Optional[List[float]] = None,
                 initial_guess: bool = True,
                 sequence_id: str = "in_memory_seq",
                 log_dir: str = "."):

        """Initializes and validates the dataset."""

        # Input validation
        if not isinstance(lidar_scans, list) or not lidar_scans:
            raise ValueError("lidar_scans must be a non-empty list.")

        if gt_relative_poses is not None and len(lidar_scans) != len(gt_relative_poses):
            raise ValueError("If gt_global_poses is provided, it must match the length of lidar_scans.")

        if timestamps is not None and len(lidar_scans) != len(timestamps):
            raise ValueError("If timestamps are provided, the length must match lidar_scans.")

        # Assign attributes
        self.scans = lidar_scans
        self.sequence_id = sequence_id
        self.log_dir = log_dir
        self.gt_poses = gt_relative_poses
        self.initial_guess = initial_guess

        # Handle timestamps
        if timestamps is None:
            self.timestamps = [float(i) * 0.1 for i in range(len(self.scans))]
            print("Warning:No timestamps provided. Generating dummy timestamps (0.0, 0.1, ...).")
        else:
            self.timestamps = [float(t) for t in timestamps]

    def __len__(self):
        """Returns the number of scans in the dataset."""
        return len(self.scans)

    def __getitem__(self, index):
        """Returns the scan and its timestamp array"""

        # Input validation
        if not 0 <= index < len(self.scans):
            raise IndexError(f"Index {index} is out of range for {len(self.scans)} scans.")

        scan = self.scans[index]
        timestamp = self.timestamps[index]

        if not isinstance(scan, np.ndarray) or scan.ndim != 2:
            print(f"Warning: Scan at index {index} is not a 2D numpy array. Returning empty frame.")
            empty_frame = np.empty((0, 3), dtype=np.float64)
            timestamp_array = np.array([timestamp], dtype=np.float64)
            return empty_frame, timestamp_array

        if scan.shape[0] == 0 or scan.shape[1] < 3:
            print(f"Scan at index {index} is empty or has < 3 features. Returning empty frame.")
            empty_frame = np.empty((0, 3), dtype=np.float64)
            timestamp_array = np.array([timestamp], dtype=np.float64)
            return empty_frame, timestamp_array

        processed_frame = scan[:, :3].astype(np.float64)

        # Ensure C-Contiguity for pybind11 binding
        if not processed_frame.flags['C_CONTIGUOUS']:
            processed_frame = np.ascontiguousarray(processed_frame)

        timestamp_array = np.array([timestamp], dtype=np.float64)

        return processed_frame, timestamp_array

    def get_frames_timestamps(self):
        """Returns all timestamps e.g. for saving poses in TUM format"""

        return np.array(self.timestamps, dtype=np.float64)


############################ MapMOS dataset for cleaning static map from moving objects ################################
class InMemoryDatasetMapMOS:
    """
    A custom dataset class for MapMOS that uses lidar scans, timestamps,
    and labels already loaded into memory.
    """

    def __init__(self,
                 lidar_scans: List[np.ndarray],
                 scan_timestamps: List[float],
                 labels_per_scan: List[np.ndarray],
                 sequence_id: str = "in_memory_mapmos_seq",
                 gt_global_poses: Optional[List[np.ndarray]] = None
                 ):

        """Initializes and validates the dataset."""

        # Input validation
        if not isinstance(lidar_scans, list) or not lidar_scans:
            raise ValueError("lidar_scans must be a non-empty list.")

        if not (len(lidar_scans) == len(scan_timestamps) == len(labels_per_scan)):
            raise ValueError(
                "Input lists (lidar_scans, scan_timestamps, labels_per_scan) must have the same length.")

        if gt_global_poses is not None and len(lidar_scans) != len(gt_global_poses):
            raise ValueError("If gt_global_poses is provided, it must match the length of lidar_scans.")

        self.scans = lidar_scans
        self.scan_timestamps_data = [float(t) for t in scan_timestamps]
        self.labels_data = labels_per_scan
        self.gt_poses = gt_global_poses
        self.sequence_id = sequence_id

    def __len__(self):
        """Returns the number of scans in the dataset."""
        return len(self.scans)

    def __getitem__(self, index):
        """Returns the points, per-point timestamps, and labels for the given index."""

        if not 0 <= index < len(self.scans):
            raise IndexError(f"Index {index} is out of range for {len(self.scans)} scans.")

        points_raw = self.scans[index]
        labels_raw = self.labels_data[index]

        if not isinstance(points_raw, np.ndarray) or points_raw.ndim != 2:
            raise TypeError(f"Scan at index {index} must be a 2D NumPy array.")

        if points_raw.shape[0] < 3 and points_raw.shape[1] >= 3:
            points_raw = points_raw.T
            print(f"Warning: Scan at index {index} is transposed. Corrected shape to {points_raw.shape}.")

        if points_raw.shape[1] < 3:
            raise ValueError(f"Scan at index {index} must have at least 3 features. Shape is {points_raw.shape}.")

        if not isinstance(labels_raw, np.ndarray) or labels_raw.ndim != 2 or labels_raw.shape[1] != 1:
            raise ValueError(
                f"Labels at index {index} (shape {labels_raw.shape}) must be a 2D NumPy array with shape (N, 1).")

        if labels_raw.shape[0] != points_raw.shape[0]:
            raise ValueError(
                f"Mismatch in number of points ({points_raw.shape[0]}) and labels ({labels_raw.shape[0]}) at index {index}.")

        points = points_raw[:, :3].astype(np.float64)

        timestamps_for_points = np.zeros(len(points), dtype=np.float64)

        labels = labels_raw.astype(np.int32)

        return points, timestamps_for_points, labels

    @property
    def timestamps(self) -> List[float]:
        """Returns the list of per-scan timestamps."""

        return self.scan_timestamps_data
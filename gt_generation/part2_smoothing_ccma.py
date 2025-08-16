import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from ccma import CCMA


def smooth_poses_hybrid(pose_matrices: np.ndarray, timestamps: list):
    """
    Smooths a trajectory of 4x4 pose matrices using a hybrid approach
    and variable timestamps for higher accuracy.

    Args:
        pose_matrices: A numpy array of shape (N, 4, 4).
        timestamps: A list or array of N timestamps in microseconds.

    Returns:
        A numpy array of smoothed 4x4 pose matrices of shape (N, 4, 4).
    """

    num_poses = pose_matrices.shape[0]
    if num_poses < 15:
        print("Warning: Not enough poses to smooth effectively. Returning original poses.")
        return pose_matrices

    # --- 1. Extract Positions and Orientations ---
    positions = pose_matrices[:, :3, 3]
    rotations = [Rotation.from_matrix(m[:3, :3]) for m in pose_matrices]

    # --- 2. Calculate time deltas (dt) for the Kalman Filter ---
    # Convert timestamps from microseconds to seconds and find differences
    timestamps_sec = np.array(timestamps) / 1e6
    dts = np.diff(timestamps_sec)
    # Handle the first dt, a common practice is to use the first calculated dt
    dts = np.insert(dts, 0, dts[0])

    # --- 3. Smooth Positions with the official CCMA library ---
    print("Smoothing positions with CCMA library...")
    # Instantiate the smoother with your desired window sizes
    ccma_smoother = CCMA(w_ma=5, w_cc=3)
    # Filter the (N, 3) position array in a single call
    smoothed_positions = ccma_smoother.filter(positions)

    # --- 4. Smooth Orientations with Kalman/RTS Smoother using variable dt ---
    def smooth_orientations_rts(rotations_list, dts_array):
        print("Smoothing orientations with variable timestamps...")
        rot_vectors = np.array([r.as_rotvec() for r in rotations_list])

        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.H = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 0]])

        # Tuning Parameters (These may need adjustment)
        kf.R *= 0.1 ** 2  # Measurement noise

        # Generate a list of state transition matrices F, one for each dt
        Fs = np.zeros((len(dts_array), 6, 6))
        for i, dt in enumerate(dts_array):
            Fs[i] = np.array(
                [[1, dt, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, dt, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, dt],
                 [0, 0, 0, 0, 0, 1]])

        # Generate process noise matrices Q for each dt
        # Note: a constant var is used here, but it could also be variable
        var = 0.05
        Qs = np.array([Q_discrete_white_noise(dim=2, dt=dt, var=var, block_size=3) for dt in dts_array])

        # Run filter and smoother with variable F and Q matrices
        (mu, cov, _, _) = kf.batch_filter(rot_vectors, Fs=Fs, Qs=Qs)
        (xs, _, _, _) = kf.rts_smoother(mu, cov, Fs=Fs, Qs=Qs)

        smoothed_rot_vectors = xs[:, [0, 2, 4], 0]
        return Rotation.from_rotvec(smoothed_rot_vectors)

    smoothed_rotations = smooth_orientations_rts(rotations, dts)

    # --- 5. Reconstruct Smoothed Pose Matrices ---
    print("Reconstructing final pose matrices...")
    smoothed_poses_arr = np.zeros_like(pose_matrices)
    for i in range(num_poses):
        smoothed_poses_arr[i, :3, :3] = smoothed_rotations[i].as_matrix()
        smoothed_poses_arr[i, :3, 3] = smoothed_positions[i]
        smoothed_poses_arr[i, 3, 3] = 1.0

    return smoothed_poses_arr


def calculate_pose_differences(original_poses, smoothed_poses):
    """
    Calculates the translational and rotational differences between two trajectories.

    Args:
        original_poses (np.ndarray): Array of original poses, shape (N, 4, 4).
        smoothed_poses (np.ndarray): Array of smoothed poses, shape (N, 4, 4).

    Returns:
        A tuple containing:
        - translational_errors (np.ndarray): The Euclidean distance (in meters) for each pose.
        - rotational_errors (np.ndarray): The rotational difference (in degrees) for each pose.
    """
    translational_errors = []
    rotational_errors = []

    for i in range(len(original_poses)):
        # Translational Error (Euclidean distance)
        pos_orig = original_poses[i, :3, 3]
        pos_smooth = smoothed_poses[i, :3, 3]
        trans_err = np.linalg.norm(pos_orig - pos_smooth)
        translational_errors.append(trans_err)

        # Rotational Error (Angle of the difference rotation)
        rot_orig = Rotation.from_matrix(original_poses[i, :3, :3])
        rot_smooth = Rotation.from_matrix(smoothed_poses[i, :3, :3])

        # Calculate the relative rotation
        error_rotation = rot_smooth * rot_orig.inv()

        # The magnitude of the rotation vector is the angle in radians
        rot_err_rad = error_rotation.magnitude()
        rotational_errors.append(np.rad2deg(rot_err_rad))  # Convert to degrees

    return np.array(translational_errors), np.array(rotational_errors)

if __name__ == '__main__':
    # Define the directory where the data was saved by your main script.
    # Adjust this path to match the --scene_io_dir argument you use.
    scene_io_dir = '/home/max/Desktop/Masterarbeit/Python/3D_Semantic_Occupancy_TruckScenes/gt_generation/pipeline_io/scene_60'

    # Define the input and output file paths.
    input_npz_path = os.path.join(scene_io_dir, "preprocessed_data.npz")
    output_npz_path = os.path.join(scene_io_dir, "smoothed_poses.npz")

    # --- 1. Load the Data from the .npz file ---
    print(f"Loading data from {input_npz_path}...")
    if not os.path.exists(input_npz_path):
        print(f"Error: Input file not found at {input_npz_path}")
        print("Please run your main data processing script first to generate this file.")
    else:
        # allow_pickle=True is required because the .npz contains object arrays
        data = np.load(input_npz_path, allow_pickle=True)

        # Extract the poses and the dictionary list
        gt_relative_poses_arr = data['gt_relative_poses_arr']
        dict_list = data['dict_list']

        # Extract timestamps for more accurate smoothing
        timestamps = [d['sample_timestamp'] for d in dict_list]

        print(f"✅ Loaded {gt_relative_poses_arr.shape[0]} poses and timestamps.")

        # --- 2. Run the Smoothing ---
        smoothed_gt_poses_arr = smooth_poses_hybrid(gt_relative_poses_arr, timestamps)

        # --- 3. Calculate and Print Differences ---
        trans_errors, rot_errors = calculate_pose_differences(gt_relative_poses_arr, smoothed_gt_poses_arr)

        print("\n--- Smoothing Difference Statistics ---")
        print(f"Translational Difference (meters):")
        print(f"  Mean:   {np.mean(trans_errors):.4f}")
        print(f"  Max:    {np.max(trans_errors):.4f}")
        print(f"  Median: {np.median(trans_errors):.4f}")

        print(f"Rotational Difference (degrees):")
        print(f"  Mean:   {np.mean(rot_errors):.4f}")
        print(f"  Max:    {np.max(rot_errors):.4f}")
        print(f"  Median: {np.median(rot_errors):.4f}")
        print("-------------------------------------\n")

        # --- 3. Save the Smoothed Data ---
        print(f"Saving smoothed poses to {output_npz_path}...")
        # Save both original and smoothed for easy comparison later
        np.savez_compressed(
            output_npz_path,
            original_poses=gt_relative_poses_arr,
            smoothed_poses=smoothed_gt_poses_arr
        )
        print("✅ Smoothed poses saved successfully.")

        # --- 4. Visualize the Results ---
        print("Generating detailed comparison plots...")
        original_positions = gt_relative_poses_arr[:, :3, 3]
        smoothed_positions = smoothed_gt_poses_arr[:, :3, 3]
        num_poses = len(original_positions)
        frame_indices = np.arange(num_poses)

        # --- Data Preparation for Rotation Plots ---
        # Convert original rotation matrices to Euler angles (roll, pitch, yaw)
        original_rotations = Rotation.from_matrix(gt_relative_poses_arr[:, :3, :3])
        original_euler = original_rotations.as_euler('xyz', degrees=True)

        # Convert smoothed rotation matrices to Euler angles
        smoothed_rotations = Rotation.from_matrix(smoothed_gt_poses_arr[:, :3, :3])
        smoothed_euler = smoothed_rotations.as_euler('xyz', degrees=True)

        # --- Create the Plot ---
        # A figure with a 2x3 grid of subplots
        fig, axes = plt.subplots(2, 3, figsize=(22, 10))
        fig.suptitle("Original vs. Smoothed Pose Components Over Time", fontsize=18)

        # --- Top Row: Position ---
        # X Position
        axes[0, 0].plot(frame_indices, original_positions[:, 0], 'r-', label='Original', alpha=0.7)
        axes[0, 0].plot(frame_indices, smoothed_positions[:, 0], 'b-', label='Smoothed')
        axes[0, 0].set_title("X Position")
        axes[0, 0].set_ylabel("X (m)")
        axes[0, 0].grid(True)
        axes[0, 0].legend()

        # Y Position
        axes[0, 1].plot(frame_indices, original_positions[:, 1], 'r-', label='Original', alpha=0.7)
        axes[0, 1].plot(frame_indices, smoothed_positions[:, 1], 'b-', label='Smoothed')
        axes[0, 1].set_title("Y Position")
        axes[0, 1].set_ylabel("Y (m)")
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        # Z Position
        axes[0, 2].plot(frame_indices, original_positions[:, 2], 'r-', label='Original', alpha=0.7)
        axes[0, 2].plot(frame_indices, smoothed_positions[:, 2], 'b-', label='Smoothed')
        axes[0, 2].set_title("Z Position (Height)")
        axes[0, 2].set_ylabel("Z (m)")
        axes[0, 2].grid(True)
        axes[0, 2].legend()

        # --- Bottom Row: Rotation (Euler Angles) ---
        # Roll (X-axis rotation)
        axes[1, 0].plot(frame_indices, original_euler[:, 0], 'r-', label='Original', alpha=0.7)
        axes[1, 0].plot(frame_indices, smoothed_euler[:, 0], 'b-', label='Smoothed')
        axes[1, 0].set_title("Roll (X-axis Rotation)")
        axes[1, 0].set_ylabel("Degrees")
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        # Pitch (Y-axis rotation)
        axes[1, 1].plot(frame_indices, original_euler[:, 1], 'r-', label='Original', alpha=0.7)
        axes[1, 1].plot(frame_indices, smoothed_euler[:, 1], 'b-', label='Smoothed')
        axes[1, 1].set_title("Pitch (Y-axis Rotation)")
        axes[1, 1].set_xlabel("Frame Index")
        axes[1, 1].set_ylabel("Degrees")
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        # Yaw (Z-axis rotation)
        axes[1, 2].plot(frame_indices, original_euler[:, 2], 'r-', label='Original', alpha=0.7)
        axes[1, 2].plot(frame_indices, smoothed_euler[:, 2], 'b-', label='Smoothed')
        axes[1, 2].set_title("Yaw (Z-axis Rotation)")
        axes[1, 2].set_ylabel("Degrees")
        axes[1, 2].grid(True)
        axes[1, 2].legend()

        # Adjust layout and display the plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # --- 5. Generate a FOCUSED plot for positional differences ---
        print("Generating detailed difference plot for position axes...")

        # Calculate the difference (error) for each axis individually
        x_diff = smoothed_positions[:, 0] - original_positions[:, 0]
        y_diff = smoothed_positions[:, 1] - original_positions[:, 1]
        z_diff = smoothed_positions[:, 2] - original_positions[:, 2]

        # Create a new figure with 3 rows, 1 column, sharing the x-axis
        fig_diff, axes_diff = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig_diff.suptitle("Smoother's Positional Adjustments per Frame", fontsize=16)

        # X Difference
        axes_diff[0].plot(frame_indices, x_diff, 'g-')
        axes_diff[0].set_title("X Position Difference")
        axes_diff[0].set_ylabel("Difference (m)")
        axes_diff[0].grid(True)

        # Y Difference
        axes_diff[1].plot(frame_indices, y_diff, 'g-')
        axes_diff[1].set_title("Y Position Difference")
        axes_diff[1].set_ylabel("Difference (m)")
        axes_diff[1].grid(True)

        # Z Difference
        axes_diff[2].plot(frame_indices, z_diff, 'g-')
        axes_diff[2].set_title("Z Position Difference")
        axes_diff[2].set_xlabel("Frame Index")
        axes_diff[2].set_ylabel("Difference (m)")
        axes_diff[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        # --- 6. Generate a FOCUSED plot for rotational differences ---
        print("Generating detailed difference plot for rotation axes...")

        # Calculate the difference (error) for each Euler angle
        roll_diff = smoothed_euler[:, 0] - original_euler[:, 0]
        pitch_diff = smoothed_euler[:, 1] - original_euler[:, 1]
        yaw_diff = smoothed_euler[:, 2] - original_euler[:, 2]

        # --- IMPORTANT: Correct for angle wraparound (+180/-180 degrees) ---
        # If the difference is > 180, it's shorter to go the other way around the circle
        roll_diff[roll_diff > 180] -= 360
        roll_diff[roll_diff < -180] += 360
        pitch_diff[pitch_diff > 180] -= 360
        pitch_diff[pitch_diff < -180] += 360
        yaw_diff[yaw_diff > 180] -= 360
        yaw_diff[yaw_diff < -180] += 360

        # Create a new figure with 3 rows, 1 column, sharing the x-axis
        fig_rot, axes_rot = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        fig_rot.suptitle("Smoother's Rotational Adjustments per Frame", fontsize=16)

        # Roll Difference
        axes_rot[0].plot(frame_indices, roll_diff, 'm-')
        axes_rot[0].set_title("Roll (X-axis) Difference")
        axes_rot[0].set_ylabel("Difference (degrees)")
        axes_rot[0].grid(True)

        # Pitch Difference
        axes_rot[1].plot(frame_indices, pitch_diff, 'm-')
        axes_rot[1].set_title("Pitch (Y-axis) Difference")
        axes_rot[1].set_ylabel("Difference (degrees)")
        axes_rot[1].grid(True)

        # Yaw Difference
        axes_rot[2].plot(frame_indices, yaw_diff, 'm-')
        axes_rot[2].set_title("Yaw (Z-axis) Difference")
        axes_rot[2].set_xlabel("Frame Index")
        axes_rot[2].set_ylabel("Difference (degrees)")
        axes_rot[2].grid(True)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

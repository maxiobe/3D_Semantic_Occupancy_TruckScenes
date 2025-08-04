import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
from pyquaternion import Quaternion

def transform_points(points_n_features, transform_4x4):
    """
    Transforms points (N, features) using a 4x4 matrix.
    Assumes input points_n_features has shape (N, features), where features >= 3 (x, y, z, ...).
    Outputs transformed points in the same (N, features) format.
    """
    # Debug prints (optional, remove after verification)
    # print(f"-- Inside transform_points --")
    # print(f"Input points shape: {points_n_features.shape}")
    # print(f"Transform matrix shape: {transform_4x4.shape}")

    # Check if there are any points
    if points_n_features.shape[0] == 0:
        # print("Input points array is empty, returning empty array.")
        return points_n_features  # Return empty array if no points

    # --- Core Logic ---
    # Extract XYZ coordinates (N, 3) - Select first 3 COLUMNS
    points_xyz_n3 = points_n_features[:, :3]
    # print(f"Extracted XYZ shape: {points_xyz_n3.shape}")

    # Convert to homogeneous coordinates (N, 4)
    points_homo_n4 = np.hstack((points_xyz_n3, np.ones((points_xyz_n3.shape[0], 1))))
    # print(f"Homogeneous points shape: {points_homo_n4.shape}")

    # Apply transformation: (4, 4) @ (4, N) -> (4, N)
    # Note the transpose of points_homo_n4 before multiplication
    transformed_homo_4n = transform_4x4 @ points_homo_n4.T
    # print(f"Transformed homogeneous shape (before T): {transformed_homo_4n.shape}")

    # Transpose back and extract XYZ: (N, 4) -> (N, 3)
    transformed_xyz_n3 = transformed_homo_4n.T[:, :3]
    # print(f"Transformed XYZ shape: {transformed_xyz_n3.shape}")

    # Combine transformed XYZ with original extra features (if any)
    if points_n_features.shape[1] > 3:  # Check if there were features beyond XYZ (columns > 3)
        # Get the remaining features from the original input
        extra_features = points_n_features[:, 3:]
        # print(f"Extra features shape: {extra_features.shape}")
        # Stack horizontally to keep the (N, features) shape
        transformed_n_features = np.hstack((transformed_xyz_n3, extra_features))
    else:
        # If only XYZ, the transformed XYZ is the result
        transformed_n_features = transformed_xyz_n3

    # print(f"Output transformed features shape: {transformed_n_features.shape}")
    # print(f"-- Exiting transform_points --")
    return transformed_n_features


def transform_matrix_interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """
    Arguments:
        x: (n,)
        xp: (m,)
        fp: (4, 4, m)

    Returns:
        y: (4, 4, n)
    """
    # Initialize interpolated transformation matrices
    y = np.repeat(np.eye(4, dtype=fp.dtype)[..., None], x.size, axis=-1)

    # Split homogeneous transformation matrices in rotational and translational part
    rot = fp[:3, :3, :]
    trans = fp[:3, 3, :]

    # Get interpolated rotation matrices
    slerp = Slerp(xp, Rotation.from_matrix(np.moveaxis(rot, -1, 0)))
    y[:3, :3, :] = np.moveaxis(slerp(x).as_matrix(), 0, -1)

    # Get interpolated translation vectors
    y[:3, 3, :] = np.vstack((
        interp1d(xp, trans[0, :])(x),
        interp1d(xp, trans[1, :])(x),
        interp1d(xp, trans[2, :])(x),
    ))

    return y


def transform_pointwise(points: np.ndarray, transforms: np.ndarray) -> np.ndarray:
    """Retruns a transformed point cloud

    Point cloud transformation with a transformation matrix for each point.

    Arguments:
        points: Point cloud with dimensions (3, n).
        transforms: Homogeneous transformation matrices with dimesnion (4, 4, n).

    Retruns:
        points: Transformed point cloud with dimension (3, n).
    """
    # Add extra dimesnion to points (3, n) -> (4, n)
    points = np.vstack((points[:3, :], np.ones(points.shape[1], dtype=points.dtype)))

    # Point cloud transformation as 3D dot product
    # T@P^T with dimensions (n, 4, 4) x (n, 1, 4) -> (n, 1, 4)
    points = np.einsum('nij,nkj->nki', np.moveaxis(transforms, -1, 0), points.T[:, None, :])

    # Remove extra dimensions (n, 1, 4) -> (n, 3)
    points = np.squeeze(points)[:, :3]

    return points.T


def transform_imu_to_ego(imu_record, imu_calibration):
    """
    Transforms all relevant IMU data from the sensor's frame to the ego vehicle's frame.

    Args:
        imu_record (dict): A single ego_motion_chassis record from TruckScenes.
        imu_calibration (dict): The calibrated_sensor record for the IMU.

    Returns:
        dict: A new dictionary with all IMU data correctly represented in the ego vehicle frame.
    """
    # 1. Get the calibration rotation from the IMU's frame to the ego frame
    q_ego_from_imu = Quaternion(imu_calibration['rotation'])

    # 2. Transform MEASUREMENT VECTORS (acceleration, velocity, angular rate)
    # These vectors are measured in the IMU's frame and need to be rotated to the ego frame.

    # Linear Acceleration
    vec_accel_imu = np.array([imu_record['ax'], imu_record['ay'], imu_record['az']])
    vec_accel_ego = q_ego_from_imu.rotate(vec_accel_imu)

    # Linear Velocity
    vec_vel_imu = np.array([imu_record['vx'], imu_record['vy'], imu_record['vz']])
    vec_vel_ego = q_ego_from_imu.rotate(vec_vel_imu)

    # Angular Velocity
    vec_rate_imu = np.array([imu_record['roll_rate'], imu_record['pitch_rate'], imu_record['yaw_rate']])
    vec_rate_ego = q_ego_from_imu.rotate(vec_rate_imu)

    # 3. Transform ABSOLUTE ORIENTATION (yaw, pitch, roll)
    # This represents the IMU's orientation in the global frame. We must combine it
    # with the calibration to find the EGO's orientation in the global frame.
    q_yaw = Quaternion(axis=[0, 0, 1], angle=imu_record['yaw'])
    q_pitch = Quaternion(axis=[0, 1, 0], angle=imu_record['pitch'])
    q_roll = Quaternion(axis=[1, 0, 0], angle=imu_record['roll'])

    # Orientation of the IMU in the global frame
    q_imu_in_global = q_yaw * q_pitch * q_roll

    # To get ego's orientation, we post-multiply by the inverse of the calibration rotation
    # Formula: q_ego_in_global = q_imu_in_global * (q_ego_from_imu)^-1
    q_ego_in_global = q_imu_in_global * q_ego_from_imu.inverse

    # Extract the new yaw, pitch, and roll angles for the ego vehicle
    ego_yaw, ego_pitch, ego_roll = q_ego_in_global.yaw_pitch_roll

    # 4. Assemble the final dictionary with all data in the ego frame
    transformed_data = {
        # Transformed Linear Acceleration
        'ax': vec_accel_ego[0],
        'ay': vec_accel_ego[1],
        'az': vec_accel_ego[2],

        # Transformed Linear Velocity
        'vx': vec_vel_ego[0],
        'vy': vec_vel_ego[1],
        'vz': vec_vel_ego[2],

        # Transformed Angular Velocity
        'roll_rate': vec_rate_ego[0],
        'pitch_rate': vec_rate_ego[1],
        'yaw_rate': vec_rate_ego[2],

        # Transformed Absolute Orientation
        'roll': ego_roll,
        'pitch': ego_pitch,
        'yaw': ego_yaw,

        # Carry over metadata
        'timestamp': imu_record['timestamp'],
        'token': imu_record['token']
    }

    return transformed_data
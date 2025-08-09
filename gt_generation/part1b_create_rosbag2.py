import os
import shutil
import numpy as np
from argparse import ArgumentParser

# ROS 2 imports
import rclpy.serialization
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata
from sensor_msgs.msg import PointCloud2, PointField, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion as RosQuaternion, Vector3
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

import math

def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
  Input: Angles in radians.
  """
  qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
  qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
  qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
  qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
  return RosQuaternion(x=qx, y=qy, z=qz, w=qw)


def create_ros2_bag_from_npz(scene_io_dir):
    """
    Loads preprocessed data and creates a chronologically correct ROS 2 bag
    with high-frequency IMU data.
    """

    npz_path = os.path.join(scene_io_dir, "preprocessed_data_rosbag.npz")
    if not os.path.exists(npz_path):
        print(f"ERROR: Could not find preprocessed data at {npz_path}")
        return

    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    imu_data_list = data['IMU_data']
    dict_list = data['dict_list']
    scene_name = str(data['scene_name'].item())

    print("Preparing and merging message streams...")
    all_messages_to_write = []
    for frame_dict in dict_list:
        ts = int(frame_dict['sample_timestamp'])
        all_messages_to_write.append((ts, '/os_cloud_node/points', frame_dict))
        all_messages_to_write.append((ts, '/gnss', frame_dict['ego_pose']))
    for imu_record in imu_data_list:
        ts = int(imu_record['timestamp'])
        all_messages_to_write.append((ts, '/os_cloud_node/imu', imu_record))

    print("Sorting all messages by timestamp...")
    all_messages_to_write.sort(key=lambda item: item[0])

    bag_dir = os.path.join(scene_io_dir, "ros2_bag")
    print(f"--- Creating ROS 2 bag at: {bag_dir} ---")
    if os.path.exists(bag_dir):
        print(f"Found existing directory, removing: {bag_dir}")
        shutil.rmtree(bag_dir)

    writer = None
    try:
        storage_options = StorageOptions(uri=bag_dir, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        writer = SequentialWriter()
        writer.open(storage_options, converter_options)

        # Create topics
        topic_info_points = TopicMetadata(name='/os_cloud_node/points', type='sensor_msgs/msg/PointCloud2',
                                          serialization_format='cdr')
        topic_info_imu = TopicMetadata(name='/os_cloud_node/imu', type='sensor_msgs/msg/Imu', serialization_format='cdr')
        topic_info_tf = TopicMetadata(name='/tf_static', type='tf2_msgs/msg/TFMessage', serialization_format='cdr')
        topic_info_gnss = TopicMetadata(name='/gnss', type='geometry_msgs/msg/PoseWithCovarianceStamped', serialization_format='cdr')

        writer.create_topic(topic_info_points)
        writer.create_topic(topic_info_imu)
        writer.create_topic(topic_info_tf)
        writer.create_topic(topic_info_gnss)

        print("Writing static transform to /tf_static topic...")
        tf_msg = TFMessage()
        static_transform = TransformStamped()

        # Get the first timestamp from your data to use for the transform header
        first_timestamp_ns = all_messages_to_write[0][0] * 1000
        static_transform.header.stamp.sec = first_timestamp_ns // 1_000_000_000
        static_transform.header.stamp.nanosec = first_timestamp_ns % 1_000_000_000

        # This defines the relationship: os_imu -> os_sensor
        static_transform.header.frame_id = 'os_imu'
        static_transform.child_frame_id = 'os_sensor'

        imu_calib_dict = data['imu_calib'].item()

        translation = Vector3(x=imu_calib_dict['translation'][0], y=imu_calib_dict['translation'][1],
                              z=imu_calib_dict['translation'][2])
        q_calib = imu_calib_dict['rotation']
        rotation = RosQuaternion(w=float(q_calib[0]), x=float(q_calib[1]), y=float(q_calib[2]), z=float(q_calib[3]))

        static_transform.transform.translation = translation
        static_transform.transform.rotation = rotation

        tf_msg.transforms.append(static_transform)
        writer.write('/tf_static', rclpy.serialization.serialize_message(tf_msg), first_timestamp_ns)

        #ts_nanoseconds_previous = None
        print(f"Writing {len(all_messages_to_write)} messages to bag...")
        for ts_microseconds, topic_name, data_payload in all_messages_to_write:
            ts_nanoseconds = ts_microseconds * 1000
            header = Header()
            header.stamp.sec = ts_nanoseconds // 1_000_000_000
            header.stamp.nanosec = ts_nanoseconds % 1_000_000_000

            if topic_name == '/os_cloud_node/points':
                header.frame_id = 'os_sensor'
                points_np = data_payload['lidar_pc_for_icp_ego_i']

                #if ts_nanoseconds_previous is not None:
                 #   print(abs(ts_nanoseconds - ts_nanoseconds_previous))

                #ts_nanoseconds_previous = ts_nanoseconds

                fields = [PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1) for i, n in
                          enumerate(['x', 'y', 'z', 't'])]

                points_np[:, 3] = float(0.0)

                msg = pc2.create_cloud(header, fields, points_np[:, :4])

            elif topic_name == '/os_cloud_node/imu':
                header.frame_id = 'os_imu'
                msg = Imu(header=header)

                msg.angular_velocity = Vector3(x=float(data_payload['roll_rate']), y=float(data_payload['pitch_rate']),
                                               z=float(data_payload['yaw_rate']))
                msg.angular_velocity_covariance[0] = 0.0006
                msg.angular_velocity_covariance[4] = 0.0006
                msg.angular_velocity_covariance[8] = 0.0006

                msg.linear_acceleration = Vector3(x=float(data_payload['ax']), y=float(data_payload['ay']),
                                                  z=float(data_payload['az']))
                msg.linear_acceleration_covariance[0] = 0.01
                msg.linear_acceleration_covariance[4] = 0.01
                msg.linear_acceleration_covariance[8] = 0.01

                roll = float(data_payload['roll'])
                pitch = float(data_payload['pitch'])
                yaw = float(data_payload['yaw'])

                msg.orientation = get_quaternion_from_euler(roll, pitch, yaw)
                msg.orientation_covariance[0] = 0.01  # Roll variance
                msg.orientation_covariance[4] = 0.01  # Pitch variance
                msg.orientation_covariance[8] = 0.01  # Yaw variance

            elif topic_name == '/gnss':
                header.frame_id = 'utm'
                msg = PoseWithCovarianceStamped(header=header)

                p_ego = data_payload['translation']
                q_ego = data_payload['rotation']
                msg.pose.pose = Pose(
                    position=Point(x=p_ego[0], y=p_ego[1], z=p_ego[2]),
                    orientation=RosQuaternion(w=q_ego[0], x=q_ego[1], y=q_ego[2], z=q_ego[3])  # Corrected w,x,y,z order
                )

            writer.write(topic_name, rclpy.serialization.serialize_message(msg), ts_nanoseconds)

    finally:
        if writer:
            writer.close()
            print(f"--- ROS 2 bag for scene '{scene_name}' created successfully. ---")


if __name__ == '__main__':
    parser = ArgumentParser(description="Create a ROS 2 bag from preprocessed data.")
    parser.add_argument('--scene_io_dir', type=str, required=True,
                        help='Path to the scene I/O directory containing preprocessed_data_rosbag.npz')
    args = parser.parse_args()
    create_ros2_bag_from_npz(args.scene_io_dir)
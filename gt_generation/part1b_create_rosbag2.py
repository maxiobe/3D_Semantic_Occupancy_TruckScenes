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


def create_ros2_bag_from_npz(scene_io_dir):
    """
    Loads preprocessed data and creates a chronologically correct ROS 2 bag
    with high-frequency IMU data.
    """
    # 1. Load the data
    npz_path = os.path.join(scene_io_dir, "preprocessed_data_rosbag.npz")
    if not os.path.exists(npz_path):
        print(f"ERROR: Could not find preprocessed data at {npz_path}")
        return

    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path, allow_pickle=True)

    imu_data_list = data['IMU_data']
    dict_list = data['dict_list']
    scene_name = str(data['scene_name'].item())

    # 2. Prepare and merge message streams
    print("Preparing and merging message streams...")
    all_messages_to_write = []
    for frame_dict in dict_list:
        ts = int(frame_dict['sample_timestamp'])
        all_messages_to_write.append((ts, '/points_raw', frame_dict))
        all_messages_to_write.append((ts, '/odometry/gps', frame_dict['ego_pose']))
    for imu_record in imu_data_list:
        ts = int(imu_record['timestamp'])
        all_messages_to_write.append((ts, '/imu_raw', imu_record))

    print("Sorting all messages by timestamp...")
    all_messages_to_write.sort(key=lambda item: item[0])

    # 3. Setup and create the ROS 2 Bag
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
        topic_info_points = TopicMetadata(name='/points_raw', type='sensor_msgs/msg/PointCloud2',
                                          serialization_format='cdr')
        topic_info_imu = TopicMetadata(name='/imu_raw', type='sensor_msgs/msg/Imu', serialization_format='cdr')
        topic_info_odom = TopicMetadata(name='/odometry/gps', type='nav_msgs/msg/Odometry', serialization_format='cdr')
        writer.create_topic(topic_info_points)
        writer.create_topic(topic_info_imu)
        writer.create_topic(topic_info_odom)

        # 5. Write the sorted messages to the bag
        print(f"Writing {len(all_messages_to_write)} messages to bag...")
        for ts_microseconds, topic_name, data_payload in all_messages_to_write:
            ts_nanoseconds = ts_microseconds * 1000
            header = Header()
            header.stamp.sec = ts_nanoseconds // 1_000_000_000
            header.stamp.nanosec = ts_nanoseconds % 1_000_000_000

            if topic_name == '/points_raw':
                header.frame_id = 'base_link'
                points_np = data_payload['lidar_pc_for_icp_ego_i']

                # --- THIS IS THE CORRECTED LINE ---
                fields = [PointField(name=n, offset=i * 4, datatype=PointField.FLOAT32, count=1) for i, n in
                          enumerate(['x', 'y', 'z', 'intensity'])]

                msg = pc2.create_cloud(header, fields, points_np[:, :4])

            elif topic_name == '/imu_raw':
                header.frame_id = 'imu_link'
                msg = Imu(header=header)
                msg.angular_velocity = Vector3(x=float(data_payload['roll_rate']), y=float(data_payload['pitch_rate']),
                                               z=float(data_payload['yaw_rate']))
                msg.linear_acceleration = Vector3(x=float(data_payload['ax']), y=float(data_payload['ay']),
                                                  z=float(data_payload['az']))
                msg.orientation = RosQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            elif topic_name == '/odometry/gps':
                header.frame_id = 'odom'
                msg = Odometry(header=header)
                msg.child_frame_id = 'base_link'
                p_ego = data_payload['translation']
                q_ego = data_payload['rotation']
                msg.pose.pose = Pose(
                    position=Point(x=p_ego[0], y=p_ego[1], z=p_ego[2]),
                    orientation=RosQuaternion(x=q_ego[1], y=q_ego[2], z=q_ego[3], w=q_ego[0])
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
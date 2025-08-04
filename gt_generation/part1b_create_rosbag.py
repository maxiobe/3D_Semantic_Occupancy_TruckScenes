import os
import numpy as np
from argparse import ArgumentParser

# ROS 1 imports
import rospy
import rosbag
from sensor_msgs.msg import PointCloud2, PointField, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Quaternion as RosQuaternion, Vector3
from std_msgs.msg import Header
import sensor_msgs.point_cloud2 as pc2


def create_ros_bag_from_npz(scene_io_dir):
    """
    Loads preprocessed data from an .npz file and creates a chronologically correct
    ROS 1 bag with high-frequency IMU data.
    """
    # 1. Load the data from the .npz file
    npz_path = os.path.join(scene_io_dir, "preprocessed_data_rosbag.npz")
    if not os.path.exists(npz_path):
        print("ERROR: Could not find preprocessed data at {}".format(npz_path))
        return

    print("Loading data from {}...".format(npz_path))
    data = np.load(npz_path, allow_pickle=True)

    # Extract both LiDAR and IMU data streams
    dict_list = data['dict_list']
    imu_data_list = data['IMU_data']  # Correctly load the array of IMU data
    scene_name = data['scene_name'].item()

    # 2. Prepare all messages for chronological sorting
    print("Preparing and merging message streams...")
    all_messages_to_write = []

    # Add LiDAR and Odometry messages from the sparse dict_list
    for frame_dict in dict_list:
        ts = frame_dict['sample_timestamp']
        # Add a tuple: (timestamp, topic_name, data_payload)
        all_messages_to_write.append((ts, '/points_raw', frame_dict))
        all_messages_to_write.append((ts, '/odometry/gps', frame_dict))

    # Add high-frequency IMU messages from the dense imu_data_list
    for imu_record in imu_data_list:
        # NOTE: Verify the timestamp key in your IMU data dictionary.
        # This assumes the key is 'timestamp'.
        ts = imu_record['timestamp']
        all_messages_to_write.append((ts, '/imu_raw', imu_record))

    # 3. Sort all messages chronologically by timestamp
    print("Sorting all messages by timestamp...")
    all_messages_to_write.sort(key=lambda item: item[0])

    # 4. Setup and create the ROS Bag
    bag_filename = os.path.join(scene_io_dir, "truckscenes.bag")
    os.makedirs(os.path.dirname(bag_filename), exist_ok=True)
    print("--- Creating ROS bag file at: {} ---".format(bag_filename))

    try:
        with rosbag.Bag(bag_filename, 'w') as bag:
            print("Writing {} messages to bag...".format(len(all_messages_to_write)))

            # 5. Write the sorted messages to the bag
            for ts_microseconds, topic_name, data_payload in all_messages_to_write:
                ts_ros = rospy.Time.from_sec(ts_microseconds / 1e6)

                # Create the correct message based on the topic name
                if topic_name == '/points_raw':
                    points_np = data_payload['lidar_pc_for_icp_ego_i']
                    header_pc = Header(stamp=ts_ros, frame_id='base_link')
                    fields = [
                        PointField('x', 0, PointField.FLOAT32, 1),
                        PointField('y', 4, PointField.FLOAT32, 1),
                        PointField('z', 8, PointField.FLOAT32, 1),
                        PointField('intensity', 12, PointField.FLOAT32, 1)
                    ]
                    # This logic for handling point shapes seems fine to keep
                    if points_np.shape[1] < 4:
                        placeholders = np.zeros((points_np.shape[0], 4 - points_np.shape[1]), dtype=np.float32)
                        points_for_msg = np.hstack([points_np[:, :3], placeholders])
                    else:
                        points_for_msg = points_np
                    msg = pc2.create_cloud(header_pc, fields, points_for_msg[:, :4])

                elif topic_name == '/imu_raw':
                    imu_record = data_payload
                    msg = Imu()
                    msg.header = Header(stamp=ts_ros, frame_id='imu_link')
                    msg.angular_velocity = Vector3(x=imu_record['roll_rate'], y=imu_record['pitch_rate'],
                                                   z=imu_record['yaw_rate'])
                    msg.linear_acceleration = Vector3(x=imu_record['ax'], y=imu_record['ay'], z=imu_record['az'])
                    # Leaving orientation as zero unless it's available in the imu_record
                    msg.orientation = RosQuaternion(x=0.0, y=0.0, z=0.0, w=1.0)

                elif topic_name == '/odometry/gps':
                    frame_dict = data_payload
                    msg = Odometry()
                    msg.header = Header(stamp=ts_ros, frame_id='odom')
                    msg.child_frame_id = 'base_link'
                    p_ego = frame_dict['ego_pose']['translation']
                    q_ego = frame_dict['ego_pose']['rotation']
                    msg.pose.pose = Pose(
                        position=Point(x=p_ego[0], y=p_ego[1], z=p_ego[2]),
                        orientation=RosQuaternion(x=q_ego[1], y=q_ego[2], z=q_ego[3], w=q_ego[0])
                    )

                # Write the message to the bag with its correct timestamp
                bag.write(topic_name, msg, ts_ros)

    finally:
        # The 'with' statement handles closing the bag automatically
        print("--- ROS bag for scene '{}' created successfully. ---".format(scene_name))


if __name__ == '__main__':
    parser = ArgumentParser(description="Create a ROS bag from preprocessed LIO-SAM data.")
    parser.add_argument('--scene_io_dir', type=str, required=True,
                        help='Path to the scene I/O directory containing preprocessed_data.npz')
    args = parser.parse_args()
    create_ros_bag_from_npz(args.scene_io_dir)
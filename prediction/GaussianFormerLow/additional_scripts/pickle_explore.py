import pickle

# The path to your .pkl file
file_path = '/home/max/Desktop/Masterarbeit/Python/GaussianFormer/data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl'

try:
    with open(file_path, 'rb') as file:
        main_data = pickle.load(file)

    infos_dict = main_data.get('infos', {})
    if not infos_dict:
        print("Could not find 'infos' dictionary or it is empty.")
    else:
        # Get the first scene token
        first_scene_token = list(infos_dict.keys())[0]
        # Get the list of frames for that scene
        frames_list = infos_dict[first_scene_token]

        if not frames_list:
            print(f"The frame list for scene {first_scene_token} is empty.")
        else:
            # Get the first frame's dictionary
            first_frame_dict = frames_list[1]
            # Get the 'data' dictionary which contains sensor info
            sensor_data = first_frame_dict.get('data', {})

            print(f"--- Sensor data for first frame of scene {first_scene_token} ---")
            if not sensor_data:
                print("No 'data' key found in the frame dictionary.")
            else:
                for sensor_name, sensor_info in sensor_data.items():
                    print(f"\nSensor: {sensor_name}")
                    if isinstance(sensor_info, dict):
                         # Print the path and other info
                         print(f"  -> Info: {sensor_info}")
                    else:
                         print(f"  -> Value: {sensor_info}")


except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
import pickle

# The path to your .pkl file
file_path = '/home/max/Desktop/Masterarbeit/Python/GaussianFormer/data/nuscenes_cam/nuscenes_infos_train_sweeps_occ.pkl'

file_path = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini/truckscenes_infos_train_sweeps_occ.pkl'


def inspect_sample(data_dict, sample_token):
    """A helper function to inspect a specific sample."""
    if sample_token not in data_dict:
        print(f"Error: Sample token '{sample_token}' not found.")
        return

    print(f"\n--- Inspecting Sample Token: {sample_token} ---")
    sample_data = data_dict[sample_token]
    print(f"Data type for this token is: {type(sample_data)}")

    # NEW: Check if the data is a list
    if isinstance(sample_data, list):
        if not sample_data:
            print("  The list is empty.")
            return

        print(f"  The list contains {len(sample_data)} element(s).")
        print("  Inspecting the FIRST element of the list...")

        # Get the first item from the list
        first_item = sample_data[1]
        print(first_item)

        # Check if this item is a dictionary and print its keys
        if isinstance(first_item, dict):
            print(f"    --> Keys found: {list(first_item.keys())}")
        else:
            print(f"    The first item is of type {type(first_item)}, not a dictionary.")


try:
    with open(file_path, 'rb') as file:
        main_data = pickle.load(file)

        if 'infos' in main_data and isinstance(main_data['infos'], dict):
            infos_dict = main_data['infos']
            if infos_dict:
                first_sample_token = list(infos_dict.keys())[0]
                inspect_sample(infos_dict, first_sample_token)
            else:
                print("The 'infos' dictionary is empty.")

except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
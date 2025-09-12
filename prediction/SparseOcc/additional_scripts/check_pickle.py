import pickle

# Replace with the path to your .pkl file
file_path = '/home/max/Desktop/Masterarbeit/Python/SparseOcc/data/nuscenes/nuscenes_infos_train_sweep.pkl'

file_path = '/home/max/Desktop/Masterarbeit/Python/3D_Semantic_Occupancy_TruckScenes/prediction/SparseOcc/data_info/occ_infos_train.pkl'


def inspect_nested_keys(data, parent_key_name='root'):
    """A helper function to print keys of nested objects."""
    print(f"\nInspecting keys for: '{parent_key_name}' (type: {type(data)})")

    # Case 1: The data is a dictionary
    if isinstance(data, dict):
        print(f"  Keys found: {list(data.keys())}")
        print(f"  Key number found: {len(list(data.keys()))}")
        # Optional: you can recursively inspect further
        # for key, value in data.items():
        #     inspect_nested_keys(value, key)

    # Case 2: The data is a list
    elif isinstance(data, list) and data:  # Check if list is not empty
        print(f"  Item is a list with {len(data)} elements.")
        # Check the type of the first element in the list
        first_item = data[0]
        print(f"  Inspecting the first element (type: {type(first_item)}):")
        # If the first element is a dictionary, show its keys
        if isinstance(first_item, dict):
            print(f"    Keys of first element: {list(first_item.keys())}")
        else:
            print(f"    First element content (preview): {str(first_item)[:100]}")

    # Case 3: Other data types
    else:
        print(f"  No dictionary or list keys to show. Content preview: {str(data)[:100]}")


try:
    with open(file_path, 'rb') as file:
        main_data = pickle.load(file)

        print("--- Top-Level Inspection ---")
        if isinstance(main_data, dict):
            print(f"Top-level keys: {list(main_data.keys())}")

            # Inspect each of the top-level keys
            for key, value in main_data.items():
                inspect_nested_keys(value, parent_key_name=key)
        else:
            print(f"Loaded object is not a dictionary. It's a {type(main_data)}")

except FileNotFoundError:
    print(f"Error: The file at {file_path} was not found.")
except Exception as e:
    print(f"An error occurred: {e}")
import pickle

# Path to the original nuScenes info file
nuscenes_info_path = '/home/max/Desktop/Masterarbeit/Python/SparseOcc/data/nuscenes/nuscenes_infos_train_sweep.pkl'

nuscenes_info_path = '/home/max/Desktop/Masterarbeit/Python/3D_Semantic_Occupancy_TruckScenes/prediction/SparseOcc/data_info/occ_infos_train.pkl'

# Load the data from the pickle file
with open(nuscenes_info_path, 'rb') as f:
    nuscenes_data = pickle.load(f)

# Let's see what's inside
print(f"Metadata: {nuscenes_data['metadata']}")
print(f"Total number of samples: {len(nuscenes_data['infos'])}")

# Inspect the first sample in detail
first_sample_info = nuscenes_data['infos'][0]
for key, value in first_sample_info.items():
    print(f"\nKey: {key}")
    print(value)
from truckscenes import TruckScenes



version = 'v1.0-trainval'
datapath = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'

truckscenes = TruckScenes(version=version,
                          dataroot=datapath,
                          verbose=True)

my_scene = truckscenes.scene[335]

scene_name = my_scene['name']

current_sample_token = my_scene.get('first_sample_token')
sample_count = 0

if not current_sample_token:
    print(f"Scene '{scene_name}' has no samples.")

while current_sample_token:
    my_sample = truckscenes.get('sample', current_sample_token)
    timestamp = my_sample['timestamp']
    sample_count += 1

    # Move to the next sample
    current_sample_token = my_sample.get('next', '')
    if not current_sample_token:
        print(
            f"Processed {sample_count} samples. Reached end of scene '{scene_name}'.")  # Optional: end of scene message
        break  # Explicit break

print(f'Sample count: {sample_count}')
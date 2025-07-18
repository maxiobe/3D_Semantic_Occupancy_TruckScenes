from truckscenes.truckscenes import TruckScenes
import numpy as np

def load_lidar_entries(trucksc, sample, lidar_sensors):
    entries = []
    for sensor in lidar_sensors:
        token = sample['data'][sensor]
        while token:
            sd = trucksc.get('sample_data', token)
            entries.append({
                'sensor': sensor,
                'timestamp': sd['timestamp'],
                'token': token,
                'keyframe': sd['is_key_frame']
            })
            token = sd['next']
    entries.sort(key=lambda x: x['timestamp'])
    return entries


def group_entries(entries, lidar_sensors, max_time_diff):
    used_tokens = set()
    groups = []

    for i, ref_entry in enumerate(entries):
        if ref_entry['token'] in used_tokens:
            continue

        ref_keyframe_flag = ref_entry['keyframe']
        group = {ref_entry['sensor']: ref_entry}
        group_tokens = {ref_entry['token']}

        for j in range(i + 1, len(entries)):
            cand = entries[j]
            if cand['keyframe'] != ref_keyframe_flag:
                continue
            if cand['token'] in used_tokens or cand['sensor'] in group:
                continue
            # Check that the new candidate is close to ALL current group timestamps
            if any(abs(cand['timestamp'] - e['timestamp']) > max_time_diff for e in group.values()):
                continue
            group[cand['sensor']] = cand
            group_tokens.add(cand['token'])

        if len(group) == len(lidar_sensors):
            groups.append(group)
            used_tokens.update(group_tokens)

    return groups


def main(trucksc):
    sensors = ['LIDAR_LEFT', 'LIDAR_RIGHT', 'LIDAR_TOP_FRONT', 'LIDAR_TOP_LEFT', 'LIDAR_TOP_RIGHT', 'LIDAR_REAR']
    max_time_diff = 90000

    my_scene = trucksc.scene[0]
    scene_name = my_scene['name']
    print(f"Processing scene: {scene_name}")
    scene_description = my_scene['description']
    print(f"Scene description: {scene_description}")
    first_sample_token = my_scene[
        'first_sample_token']
    my_sample = trucksc.get('sample',
                            first_sample_token)

    ######################### Get all lidar entries for the current scene and given scenes ###########################
    lidar_entries = load_lidar_entries(trucksc=trucksc, sample=my_sample, lidar_sensors=sensors)
    print(f"Number of lidar entries: {len(lidar_entries)}")

    ########################## Generate groups of lidar scans from different sensors with max time diff ##############
    groups = group_entries(entries=lidar_entries, lidar_sensors=sensors, max_time_diff=max_time_diff)
    print(f"\n✅ Total groups found: {len(groups)}")

    timestamps_group = []

    for i, group in enumerate(groups):
        ref_sensor = sensors[0]

        ################### Generate a sample dict to load the lidar point clouds and calculate timestamp #########
        sample_data_dict = {sensor: group[sensor]['token'] for sensor in sensors}
        sample = {
            'timestamp': np.mean([group[s]['timestamp'] for s in sensors]),
            'data': sample_data_dict,
            'sample_data_token': sample_data_dict[ref_sensor],
            'is_key_frame': group[ref_sensor]['keyframe'],
        }

        if sample['is_key_frame']:
            print(f"Group {i/5}: {sample['timestamp']}")
            timestamps_group.append(sample['timestamp'])

    sample_idx = 0
    timestamp_samples = []
    timestamp_samples_calculated = []
    while True:
        print(f"Sample {sample_idx}: {my_sample['timestamp']}")
        timestamp_samples.append(my_sample['timestamp'])


        sample_data = my_sample['data']
        sensor_timestamps = []
        for sensor in sensors:
            sample_data_sensor_token = sample_data[sensor]
            sample_data_sensor = trucksc.get('sample_data', sample_data_sensor_token)
            sensor_timestamps.append(sample_data_sensor['timestamp'])

        timestamp_samples_calculated.append(np.mean(sensor_timestamps))

        next_sample_token = my_sample['next']
        if next_sample_token != '':
            my_sample = trucksc.get('sample', next_sample_token)
            sample_idx += 1
        else:
            break

    for index in range(40):
        print(f"Sample {index}:")
        print(f"Group timestamp: {timestamps_group[index]}")
        print(f"Sample timestamp: {timestamp_samples[index]}")
        print(f"Sample timestamps mean sensors: {timestamp_samples_calculated[index]}")

        difference = abs(timestamp_samples[index] - timestamps_group[index])
        print(f"Difference between group and sample timestamp: {difference/(10**6)} s\n")


if __name__ == '__main__':
    version='v1.0-trainval'
    dataroot='/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
    truckscenes = TruckScenes(version=version,
                              dataroot=dataroot,
                              verbose=True)

    main(truckscenes)
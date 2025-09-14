from Nuscenes import NuScenes
from collections import defaultdict
from tqdm import tqdm

def main(datroot, version):

    nusc = NuScenes(dataroot=datroot, version=version, verbose=True)

    weather_counts = defaultdict(int)
    area_counts = defaultdict(int)
    daytime_counts = defaultdict(int)
    seasonal_counts = defaultdict(int)
    lightning_counts = defaultdict(int)
    structure_counts = defaultdict(int)
    construction_counts = defaultdict(int)

    for scene in tqdm(nusc.scene, desc="Processing scenes"):
        scene_name = scene['name']
        print("Scene name: {}".format(scene_name))

        scene_description = scene['description']
        print(scene_description)

        weather_string = scene_description.split(';')[0]
        area_string = scene_description.split(';')[1]
        daytime_string = scene_description.split(';')[2]
        seasonal_string = scene_description.split(';')[3]
        lightning_string = scene_description.split(';')[4]
        structure_string = scene_description.split(';')[5]
        construction_string = scene_description.split(';')[6]

        weather_condition = weather_string.split('.')[1]
        area_condition = area_string.split('.')[1]
        daytime_condition = daytime_string.split('.')[1]
        seasonal_condition = seasonal_string.split('.')[1]
        lightning_condition = lightning_string.split('.')[1]
        structure_condition = structure_string.split('.')[1]
        construction_condition = construction_string.split('.')[1]

        weather_counts[weather_condition] += 1
        area_counts[area_condition] += 1
        daytime_counts[daytime_condition] += 1
        seasonal_counts[seasonal_condition] += 1
        lightning_counts[lightning_condition] += 1
        structure_counts[structure_condition] += 1
        construction_counts[construction_condition] += 1

    print("\n--- Weather Scenario Counts ---")
    print(dict(weather_counts))
    print("\n--- Area Scenario Counts ---")
    print(dict(area_counts))
    print("\n--- Daytime Scenario Counts ---")
    print(dict(daytime_counts))
    print("\n--- Seasonal Scenario Counts ---")
    print(dict(seasonal_counts))
    print("\n--- Lightning Scenario Counts ---")
    print(dict(lightning_counts))
    print("\n--- Structure Scenario Counts ---")
    print(dict(structure_counts))
    print("\n--- Construction Scenario Counts ---")
    print(dict(construction_counts))

if __name__ == "__main__":

    data_root = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
    version = 'v1.0-trainval'
    main(data_root, version)
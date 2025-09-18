from nuscenes.nuscenes import NuScenes
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


if __name__ == "__main__":
    data_root = '/home/max/Desktop/Masterarbeit/Data/nuScenes/trainval/v1.0-trainval'
    version = 'v1.0-trainval'
    print("Processing scenes for trainval:")
    main(data_root, version)

    data_root = '/home/max/Desktop/Masterarbeit/Data/nuScenes/test/v1.0-test'
    version = 'v1.0-test'
    print("Processing scenes for test:")
    main(data_root, version)

    data_root = '/home/max/Desktop/Masterarbeit/Data/nuScenes/mini/v1.0-mini'
    version = 'v1.0-mini'
    print("Processing scenes for test:")
    main(data_root, version)
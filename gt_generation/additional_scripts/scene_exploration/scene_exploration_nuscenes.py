from nuscenes import NuScenes

if __name__ == '__main__':

    version = 'v1.0-trainval'
    version = 'v1.0-mini'
    datapath = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
    datapath = '/home/max/Desktop/Masterarbeit/Data/Occupancy3D/data/occ3d-nus'
    nuscenes = NuScenes(version=version,
                              dataroot=datapath,
                              verbose=True)

    # truckscenes.list_scenes()

    print("Trainval scenes:")
    print()

    i = 0
    for scene in nuscenes.scene:
        print(f"Scene {i}:")
        print(f"Scene name: {scene['name']}")
        print(f"Description: {scene['description']}")
        print()
        i += 1

    i = 0
    for scene in nuscenes.scene:
        print(f"Rendering scene {i}...")
        #if i < 75:
         #   i += 1
          #  continue
        my_scene_token = scene['token']

        nuscenes.render_scene(my_scene_token)
        i += 1


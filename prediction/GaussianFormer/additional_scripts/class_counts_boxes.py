from truckscenes import TruckScenes

version_trainval = 'v1.0-trainval'
version_mini = 'v1.0-mini'

data_root_trainval = '/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval'
data_root_mini = '/home/max/ssd/Masterarbeit/TruckScenes/mini/v1.0-mini'


truckscenes_trainval = TruckScenes(version=version_trainval, dataroot=data_root_trainval, verbose=True)
truckscenes_mini = TruckScenes(version=version_mini, dataroot=data_root_mini, verbose=True)

truckscenes_trainval.list_categories()
print()
truckscenes_mini.list_categories()


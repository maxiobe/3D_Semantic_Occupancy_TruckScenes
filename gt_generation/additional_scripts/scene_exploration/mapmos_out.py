import open3d as o3d

import numpy as np
from pathlib import Path

saved_ply_path = Path('/home/max/Desktop/Masterarbeit/Python/3D_Semantic_Occupancy_TruckScenes/gt_generation/results/2025-07-19_11-46-40/ply/000045.ply')


if saved_ply_path.is_file():
    print(f"Loading PLY file: {saved_ply_path}")
    pcd = o3d.io.read_point_cloud(str(saved_ply_path))

    if not pcd.has_points():
        print("Loaded PLY is empty.")
    else:
        print(f"Loaded point cloud with {len(pcd.points)} points.")
        if pcd.has_colors():
            print("Point cloud has colors. These might represent static/moving.")
        else:
            print("Point cloud does not have colors by default.")

    o3d.visualization.draw_geometries([pcd], window_name="MapMOS Output PLY")
else:
    print(f"PLY file not found at: {saved_ply_path}")
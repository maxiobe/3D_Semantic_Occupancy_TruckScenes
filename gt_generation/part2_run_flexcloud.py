import flexcloud_bindings
import os
import sys

def main():
    print("--- Running Part 2: FlexCloud Processing ---")

    if len(sys.argv) < 2:
        print("Error: Please provide the path to the FlexCloud I/O directory.")
        sys.exit(1)

    scene_io_dir = sys.argv[1]
    flexcloud_input_dir = os.path.join(scene_io_dir, "flexcloud_io")
    # Define paths based on the I/O directory
    config_path_select_keyframes = "/home/max/Desktop/Masterarbeit/Python/3D_Semantic_Occupancy_TruckScenes/gt_generation/flexcloud/config/select_keyframes.yaml"
    config_path_pcd_georef = "/home/max/Desktop/Masterarbeit/Python/3D_Semantic_Occupancy_TruckScenes/gt_generation/flexcloud/config/pcd_georef.yaml"
    pos_dir = os.path.join(flexcloud_input_dir, "gnss_poses")
    odom_path = os.path.join(flexcloud_input_dir, "odom/slam_poses.txt")
    pcd_dir = os.path.join(flexcloud_input_dir, "point_clouds")
    output_dir_keyframes = os.path.join(flexcloud_input_dir, "output_keyframes")
    pcd_dir_transformed = os.path.join(flexcloud_input_dir, "pcd_transformed")

    # Initialize the FlexCloud processor
    processor = flexcloud_bindings.FlexCloud()

    # Call the C++ functions
    print("Running Keyframe Interpolation...")
    processor.run_keyframe_interpolation_from_files(
        config_path=config_path_select_keyframes,
        pos_dir_path=pos_dir,
        kitti_odom_path=odom_path,
        pcd_dir_path=pcd_dir,
        dst_dir_path=output_dir_keyframes
    )
    print("Finished running Keyframe Interpolation...")

    print("Running Georeferencing...")
    # NOTE: You will need to adapt the pcd_path for this function.
    # It might need to point to the `output_dir_keyframes` from the previous step.
    # This is a placeholder for the logic you need.
    # For now, we assume it processes the same input PCDs.
    ref_path_in = os.path.join(output_dir_keyframes, "poseData.txt")
    slam_path_in = os.path.join(output_dir_keyframes, "kitti_poses.txt")

    pcd_path_in = os.path.join(output_dir_keyframes, "000000/cloud.pcd")
    output_pcd_filepath = os.path.join(pcd_dir_transformed, "refined_map.pcd")

    processor.run_georeferencing_from_files(
        config_path=config_path_pcd_georef,
        ref_path=ref_path_in,      # Assuming this is correct
        slam_path=slam_path_in,   # Assuming this is correct
        pcd_path=pcd_path_in,       # *** CHECK THIS ***
        pcd_out_path=output_pcd_filepath,
    )
    print("Finished running Georeferencing...")

    print("--- Part 2 Complete ---")
    print(f"FlexCloud processing finished. Transformed PCDs should be in {pcd_dir_transformed}")

if __name__ == '__main__':
    main()
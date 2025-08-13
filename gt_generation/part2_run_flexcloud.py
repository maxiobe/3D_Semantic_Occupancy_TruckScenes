import flexcloud_bindings
import os
import sys

def main():
    print("--- Running Part 2: FlexCloud Processing ---")

    if len(sys.argv) < 2:
        print("Error: Please provide the path to the FlexCloud I/O directory.")
        sys.exit(1)

    ######################### Path reference ###############################
    scene_io_dir = sys.argv[1]
    flexcloud_input_dir = os.path.join(scene_io_dir, "flexcloud_io")


    ########################## Keyframe selection ###########################

    ########################## Config #######################################
    config_path_select_keyframes = "/home/max/Desktop/Masterarbeit/Python/3D_Semantic_Occupancy_TruckScenes/gt_generation/flexcloud/config/select_keyframes.yaml"

    ################ Path to input files generated in part 1 ################
    pos_dir = os.path.join(flexcloud_input_dir, "gnss_poses")
    odom_path = os.path.join(flexcloud_input_dir, "odom/slam_poses.txt")
    pcd_dir = os.path.join(flexcloud_input_dir, "point_clouds")
    output_dir_keyframes = os.path.join(flexcloud_input_dir, "output_keyframes")


    # Initialize the FlexCloud processor
    processor = flexcloud_bindings.FlexCloud()

    # Call the C++ functions
    #print("Running Keyframe Interpolation...")
    processor.run_keyframe_interpolation_from_files(
        config_path=config_path_select_keyframes,
        pos_dir_path=pos_dir,
        kitti_odom_path=odom_path,
        pcd_dir_path=pcd_dir,
        dst_dir_path=output_dir_keyframes
    )
    print("Finished running Keyframe Interpolation...")


    ######################################################################################
    ########################### Georeferencing ###########################################

    print("Running Georeferencing...")

    ########################### Config ################################
    config_path_pcd_georef = "/home/max/Desktop/Masterarbeit/Python/3D_Semantic_Occupancy_TruckScenes/gt_generation/flexcloud/config/pcd_georef.yaml"

    # ref_path_in = os.path.join(output_dir_keyframes, "poseData.txt") # Output from keyframe interpolation
    ref_path_in = os.path.join(flexcloud_input_dir, "all_gnss_data_poses.txt")
    # slam_path_in = os.path.join(output_dir_keyframes, "kitti_poses.txt") # Output from step keyframe interpolation
    slam_path_in = odom_path

    ###################### Input and output of point cloud map ####################
    pcd_dir_transformed = os.path.join(flexcloud_input_dir, "pcd_transformed")
    pcd_path_in =os.path.join(flexcloud_input_dir, "aggregated_cloud.pcd")
    output_pcd_filepath = os.path.join(pcd_dir_transformed, "refined_map.pcd")

    processor.run_georeferencing_from_files(
        config_path=config_path_pcd_georef,
        ref_path=ref_path_in,
        slam_path=slam_path_in,
        pcd_path=pcd_path_in,
        pcd_out_path=output_pcd_filepath,
    )
    print("Finished running Georeferencing...")

    print("--- Part 2 Complete ---")
    print(f"FlexCloud processing finished. Transformed PCDs should be in {pcd_dir_transformed}")

if __name__ == '__main__':
    main()
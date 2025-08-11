#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Define the names of your Conda environments
ENV_PREPROCESS="occ_kiss_p3d"
ENV_ROSBAG="ros_tools"
ENV_FLEXCLOUD="flexcloud-stable"
ENV_POSTPROCESS="occ_kiss_p3d"

# For step 1
CONFIG_PATH="config_truckscenes.yaml"
LABEL_MAPPING="truckscenes.yaml"
SAVE_PATH_GT="/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval/gt/all_sensors_icp_validation"
DATA_ROOT="/home/max/ssd/Masterarbeit/TruckScenes/trainval/v1.0-trainval"
VERSION="v1.0-trainval"
SPLIT="train"
START=2
END=3
LOAD_MODE="pointwise"

USE_FLEXCLOUD=0

# Define the root directory for your Python scripts
PIPELINE_DIR="/home/max/Desktop/Masterarbeit/Python/3D_Semantic_Occupancy_TruckScenes/gt_generation"

# Define a single base directory for all temporary I/O
IO_BASE_DIR="${PIPELINE_DIR}/pipeline_io"

# --- Main Setup ---
echo "--- Pipeline Setup ---"
# Source Conda functions to make 'conda' command available in the script
source ~/miniforge3/etc/profile.d/conda.sh
# Create the base I/O directory
mkdir -p "$IO_BASE_DIR"
echo "Base I/O directory is at: $IO_BASE_DIR"

# --- Main Processing Loop ---
for (( i=$START; i<$END; i++ ))
do
    # Create a specific directory for the current scene's intermediate files
    SCENE_IO_DIR="${IO_BASE_DIR}/scene_${i}"
    echo -e "\n--------------------------------------------------"
    echo "--- Processing Scene Index: $i ---"
    echo "--- Using temporary directory: $SCENE_IO_DIR ---"
    echo "--------------------------------------------------"
    mkdir -p "$SCENE_IO_DIR"

    # --- Step 1: Pre-processing with Open3D ---
    echo "Activating environment: $ENV_PREPROCESS for Part 1..."
    conda activate "$ENV_PREPROCESS"
    python "${PIPELINE_DIR}/part1_preprocess.py" \
        --dataroot "$DATA_ROOT" \
        --version "$VERSION" \
        --config_path "$CONFIG_PATH" \
        --save_path "$SAVE_PATH_GT" \
        --label_mapping "$LABEL_MAPPING" \
        --split "$SPLIT" \
        --idx "$i" \
        --load_mode "$LOAD_MODE" \
        --scene_io_dir "$SCENE_IO_DIR" \
        --icp_refinement \
        --initial_guess_mode ego_pose \
        --pose_error_plot \
        --filter_lidar_intensity \
        --filter_mode both \
        --filter_static_pc \
        --run_mapmos \
        --vis_aggregated_static_ego_ref_pc \
        --static_map_keyframes_only \
        --use_flexcloud "$USE_FLEXCLOUD" \
        --vis_static_frame_comparison_kiss_refined \
        #--vis_raw_pc \
        #--vis_static_pc \
        #--vis_static_pc_global \
        #--vis_lidar_intensity_filtered \
        #--filter_raw_pc \
        #--vis_aggregated_static_ego_i_pc \
        #--vis_aggregated_static_global_pc \
        #--vis_aggregated_raw_pc_ego_i \
    conda deactivate

    # --- Step 1b: Create ROS Bag ---
    #echo "--- Preparing to create ROS Bag ---"
    # Note: This step assumes you have sourced your main ROS environment
    # (e.g., /opt/ros/melodic/setup.bash) in your ~/.bashrc or you can
    # uncomment the line below to source it explicitly.
    #source /opt/ros/melodic/setup.bash

    #echo "--- Creating ROS Bag inside the liosam-ros1 Docker container ---"
    #docker run --rm \
     #   -v "${PIPELINE_DIR}:/scripts" \
      #  -v "${SCENE_IO_DIR}:/scene_io" \
       # rosbag-creator \
        #bash -c "source /opt/ros/melodic/setup.bash && python3 /scripts/part1b_create_rosbag.py --scene_io_dir /scene_io"


    #echo "Finished preparing rosbag files"

    # --- Step 1c: Create ROS 2 Bag (NEW STEP) ---
    #echo "--- Creating ROS 2 Bag inside a ROS 2 Docker container ---"
    #docker run --rm \
     #   -v "${PIPELINE_DIR}:/scripts" \
      #  -v "${SCENE_IO_DIR}:/scene_io" \
       # koide3/glim_ros2:humble \
       # bash -c "source /opt/ros/humble/setup.bash && python3 /scripts/part1b_create_rosbag2.py --scene_io_dir /scene_io"

    # --- Check for success flag before continuing ---
    FLAG_FILE="${SCENE_IO_DIR}/part1_success.flag"
    if [ -f "$FLAG_FILE" ]; then
        echo "Part 1 successful, proceeding to Part 2 and 3..."

        if [ "$USE_FLEXCLOUD" -eq 1 ]; then
            # --- Step 2: FlexCloud Processing ---
            echo "Activating environment: $ENV_FLEXCLOUD for Part 2..."
            conda activate "$ENV_FLEXCLOUD"
            python "${PIPELINE_DIR}/part2_run_flexcloud.py" "$SCENE_IO_DIR"
            conda deactivate
        else
            echo "Skipping FlexCloud processing as USE_FLEXCLOUD is set to 0."
        fi

        # --- Step 3: Post-processing with Open3D ---
        echo "Activating environment: $ENV_POSTPROCESS for Part 3..."
        conda activate "$ENV_POSTPROCESS"

        python "${PIPELINE_DIR}/part3_postprocess.py" \
            --dataroot "$DATA_ROOT" \
            --version "$VERSION" \
            --config_path "$CONFIG_PATH" \
            --save_path "$SAVE_PATH_GT" \
            --label_mapping "$LABEL_MAPPING" \
            --scene_io_dir "$SCENE_IO_DIR" \
            --icp_refinement \
            --pose_error_plot \
            --dynamic_map_keyframes_only \
            --static_map_keyframes_only \
            --use_flexcloud "$USE_FLEXCLOUD" \
            --vis_dyn_ambigious_points \
            --vis_combined_static_dynamic_pc \
            --vis_dyn_unreassigned_points \
            --vis_static_frame_comparison

        conda deactivate
    else
        echo "Skipping Part 2 and 3 for scene $i as it was not in the desired split."
    fi
    # --- (Optional) Clean up the intermediate files for this scene ---
    #echo "--- Cleaning up intermediate files for scene $i ---"
    #rm -rf "$SCENE_IO_DIR"

done

echo -e "\n✅✅✅ Pipeline finished successfully! ✅✅✅"
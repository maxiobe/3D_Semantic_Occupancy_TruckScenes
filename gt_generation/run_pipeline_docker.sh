#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration (UPDATED FOR DOCKER) ---
# Define the names of your Conda environments
ENV_PREPROCESS="occ_kiss_p3d"
ENV_ROSBAG="ros_tools"
ENV_FLEXCLOUD="flexcloud" #flexcloud-stable
ENV_POSTPROCESS="occ_kiss_p3d"

# For step 1
CONFIG_PATH="config_truckscenes.yaml"
LABEL_MAPPING="truckscenes.yaml"
# UPDATED: Paths now point to the /data mount point inside the container
#SAVE_PATH_GT="/truckscenes/trainval/v1.0-trainval/gts_64"
SAVE_PATH_GT="/gts"
#DATA_ROOT="/truckscenes/trainval/v1.0-trainval"
DATA_ROOT="/truckscenes"


VERSION="v1.0-trainval"
SPLIT="all"
START=234
END=235
LOAD_MODE="pointwise"

USE_FLEXCLOUD=0
USE_LOCAL_STATIC_MAP=0
# UPDATED: The pipeline directory is now /app inside the container
PIPELINE_DIR="/code"

# Define a single base directory for all temporary I/O
IO_BASE_DIR="${PIPELINE_DIR}/pipeline_io"

# --- Main Setup ---
echo "--- Pipeline Setup ---"
# UPDATED: Correct path to conda inside the container
source /opt/conda/etc/profile.d/conda.sh

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
        --filter_mode both \
        --filter_static_pc \
        --initial_guess_mode ego_pose \
        --run_mapmos \
        --static_map_keyframes_only \
        --use_flexcloud "$USE_FLEXCLOUD" \
        --filter_lidar_intensity \
        --icp_refinement \
        #--vis_static_frame_comparison_kiss_refined \
        #--pose_error_plot \
        #--vis_aggregated_static_ego_ref_pc \
        #--vis_static_pc \
        #--vis_raw_pc \
        #--vis_static_pc_global \
        #--vis_lidar_intensity_filtered \
        #--filter_raw_pc \
        #--vis_aggregated_static_ego_i_pc \
        #--vis_aggregated_static_global_pc \
        #--vis_aggregated_raw_pc_ego_i \
    conda deactivate

    # Note: The ROS bag creation steps are commented out as they require
    # a more complex Docker-in-Docker setup.

    # --- Check for success flag before continuing ---
    FLAG_FILE="${SCENE_IO_DIR}/part1_success.flag"
    if [ -f "$FLAG_FILE" ]; then
        echo "Part 1 successful, proceeding to Part 2 and 3..."

        if [ "$USE_FLEXCLOUD" -eq 1 ]; then
            # --- Step 2: FlexCloud Processing ---
            echo "Activating environment: $ENV_FLEXCLOUD for Part 2..."
            conda activate "$ENV_FLEXCLOUD"
            python "${PIPELINE_DIR}/part2_run_flexcloud.py" "$SCENE_IO_DIR" "$CONFIG_PATH"
            conda deactivate
        else
            echo "Skipping FlexCloud processing as USE_FLEXCLOUD is set to 0."
        fi

        # --- Step 3: Post-processing with Open3D ---
        echo "Activating environment: $ENV_POSTPROCESS for Part 3..."
        conda activate "$ENV_POSTPROCESS"

        if [ "$USE_LOCAL_STATIC_MAP" -eq 1 ]; then
            echo "--- Running Part 3 with LOCAL (sliding-window) static map ---"
            python "${PIPELINE_DIR}/part3_postprocess_reduced_static.py" \
                --dataroot "$DATA_ROOT" \
                --version "$VERSION" \
                --config_path "$CONFIG_PATH" \
                --save_path "$SAVE_PATH_GT" \
                --label_mapping "$LABEL_MAPPING" \
                --scene_io_dir "$SCENE_IO_DIR" \
                --dynamic_map_keyframes_only \
                --use_flexcloud "$USE_FLEXCLOUD" \
                --filter_aggregated_static_map \
                --filter_static_pc_list \
                --static_map_keyframes_only \
                --icp_refinement
        else
            echo "--- Running Part 3 with GLOBAL (all frames) static map ---"
            python "${PIPELINE_DIR}/part3_postprocess.py" \
                --dataroot "$DATA_ROOT" \
                --version "$VERSION" \
                --config_path "$CONFIG_PATH" \
                --save_path "$SAVE_PATH_GT" \
                --label_mapping "$LABEL_MAPPING" \
                --scene_io_dir "$SCENE_IO_DIR" \
                --dynamic_map_keyframes_only \
                --use_flexcloud "$USE_FLEXCLOUD" \
                --filter_aggregated_static_map \
                --filter_static_pc_list\
                --icp_refinement \
                --static_map_keyframes_only
        fi
        conda deactivate
    else
        echo "Skipping Part 2 and 3 for scene $i as it was not in the desired split."
    fi
    # --- (Optional) Clean up the intermediate files for this scene ---
    echo "--- Cleaning up intermediate files for scene $i ---"
    rm -rf "$SCENE_IO_DIR"

done

echo -e "\n✅✅✅ Pipeline finished successfully! ✅✅✅"

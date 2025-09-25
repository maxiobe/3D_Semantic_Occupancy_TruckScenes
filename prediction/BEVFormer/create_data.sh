#!/bin/bash
#
#SBATCH --partition=lrz-v100x2,lrz-hpe-p100x4,lrz-dgx-1-p100x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=bevformer-create-data
#SBATCH --time=2-00:00:00
#SBATCH --mem=64G

# Define the base log directory
LOG_BASE_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/BEVFormer/logs"
# Create the directory if it doesn't exist
mkdir -p "$LOG_BASE_DIR"
# Set the specific output and error file paths
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --workdir=$LOG_BASE_DIR


# --- Define paths for clarity ---
BASE_PROJECT_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes"
DATA_DIR="/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes"
IMAGE_PATH="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/images/maxiobe123+bevformer+base-cuda111.sqsh"


# --- Construct container mounts ---
MOUNTS="${BASE_PROJECT_DIR}:/code,${DATA_DIR}:/truckscenes:ro"


# --- Execute the data creation command ---
echo "Starting data creation for BEVFormer (v1.0-trainval)..."

srun --mpi=pmi2 --unbuffered \
    --container-image="${IMAGE_PATH}" \
    --container-mounts="${MOUNTS}" \
    bash -c '
      # Activate conda environment and run the python script
      conda run -n bev-former python /code/prediction/BEVFormer/tools/create_data.py occ \
        --version v1.0-trainval \
        --data-root-path /truckscenes \
        --annotation-path /code/prediction/BEVFormer/data_info/trainval \
        --out-dir /code/prediction/BEVFormer/data_info/trainval \
        --extra-tag occ
    '

echo "Data creation script finished."

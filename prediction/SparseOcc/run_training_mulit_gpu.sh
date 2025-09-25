#!/bin/bash
#
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --job-name=sparseocc-multigpu   # Changed for clarity
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                    # Requesting 4 GPUs
#SBATCH --cpus-per-task=32              # Increased CPUs (e.g., 8 per GPU)
#SBATCH --mem=256G                      # Increased RAM (e.g., 64G per GPU)
#SBATCH --time=1-00:00:00
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/SparseOcc/outputs/SparseOcc/sbatch_multi_gpu_logs/slurm_multigpu_%j.out
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/SparseOcc/outputs/SparseOcc/sbatch_multi_gpu_logs/slurm_multigpu_%j.err


BASE_PROJECT_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes"
DATA_DIR="/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes"
GTS_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/trainval/gts"
IMAGE_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/images"
OUTPUT_DIR="${BASE_PROJECT_DIR}/prediction/SparseOcc"


# It's good practice to use a different run name for multi-GPU experiments
RUN_NAME_LOGS="sbatch_multi_gpu_logs"
RUN_NAME_TRAINING="sbatch_multi_gpu_training"
WORK_DIR="${OUTPUT_DIR}/outputs/SparseOcc/${RUN_NAME_LOGS}"

mkdir -p "$WORK_DIR"
rm -rf "${OUTPUT_DIR}/outputs/SparseOcc/${RUN_NAME_TRAINING}"

# Define container and mount paths for clarity
CONTAINER_IMAGE="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/images/maxiobe123+sparseocc+final-base-cuda-118.sqsh"
CONTAINER_MOUNTS="${BASE_PROJECT_DIR}:/code,${DATA_DIR}:/truckscenes:ro,${GTS_DIR}:/gts:ro,${OUTPUT_DIR}:/output"

srun --mpi=pmi2 --unbuffered \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    bash -c '
    # Use && to ensure the script stops if a command fails
    set -e

    echo "--- Setting up environment and compiling custom ops ---"

    # Navigate to the project directory
    cd /code/prediction/SparseOcc && \

    # Install dependencies
    python3.8 -m pip install ninja && \

    # Set CUDA architectures for A100 (8.0) and H100 (9.0)
    export TORCH_CUDA_ARCH_LIST="8.0 9.0" && \

    # Compile custom CUDA operations
    cd models/csrc && \
    python3.8 setup.py build_ext --inplace && \

    # Go back to the project root before training
    cd ../.. && \

    echo "--- Starting training on 4 GPUs ---"

    # Run the training script using torchrun
    torchrun --nproc_per_node=4 train.py \
        --config configs/r50_nuimg_704x256_8f.py \
        --run_name "'"$RUN_NAME_TRAINING"'"
'

echo "--- Job finished ---"

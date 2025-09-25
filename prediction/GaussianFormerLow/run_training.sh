#!/bin/bash
#
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=gaussian-former-1-low
#SBATCH --time=0-05:00:00
#SBATCH --mem=64G
# Pfade für die Slurm-Logdateien 
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/GaussianFormer/out/nuscenes_gs25600_solid/mini_run_final_sbatch_5/slurm_%j.out
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/GaussianFormer/out/nuscenes_gs25600_solid/mini_run_final_sbatch_5/slurm_%j.err


BASE_PROJECT_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes"
DATA_DIR="/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes"
GTS_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/mini/gts"
IMAGE_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/images"
OUTPUT_DIR="${BASE_PROJECT_DIR}/prediction/GaussianFormer/out"

RUN_NAME="mini_run_final_sbatch_5"
WORK_DIR="${OUTPUT_DIR}/nuscenes_gs25600_solid/${RUN_NAME}"

mkdir -p "$WORK_DIR"

MOUNTS="${BASE_PROJECT_DIR}:/code,${DATA_DIR}:/truckscenes:ro,${GTS_DIR}:/gts:ro,${OUTPUT_DIR}:/output"

srun --mpi=pmi2 --unbuffered \
     --container-image="${IMAGE_DIR}/maxiobe123+gaussianformer+latest.sqsh" \
     --container-mounts="$MOUNTS" \
     bash -c '
        # --- On-the-fly Kompilierung ---
        echo "Compiling custom CUDA ops for target architecture: $GPU_ARCH"

        # Die exportierte Variable wird hier verwendet
        export TORCH_CUDA_ARCH_LIST="8.0"

        # Führe die Installation/Kompilierung aus
        conda run -n gaussianformer pip install -e /code/prediction/GaussianFormer/model/encoder/gaussian_encoder/ops
        conda run -n gaussianformer pip install -e /code/prediction/GaussianFormer/model/head/localagg

        echo "Compilation finished."

        # --- Training starten ---
        cd /code/prediction/GaussianFormer && \
        echo "Starting training..."
        conda run -n gaussianformer python -u train.py \
            --py-config config/nuscenes_gs25600_solid.py \
            --work-dir /output/nuscenes_gs25600_solid/'"$RUN_NAME"'
     '

#!/bin/bash
#
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=bevformer
#SBATCH --time=0-0:30:00
#SBATCH --mem=80G
# Pfade für die Slurm-Logdateien 
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/BEVFormer/work_dirs/bevformer_small_occ/mini_run_final_sbatch_reduced_1/slurm_%j.out
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/BEVFormer/work_dirs/bevformer_small_occ/mini_run_final_sbatch_reduced_1/slurm_%j.err


BASE_PROJECT_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes"
DATA_DIR="/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes"
GTS_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/trainval/gts"
IMAGE_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/images"
OUTPUT_DIR="${BASE_PROJECT_DIR}/prediction/BEVFormer/work_dirs"

RUN_NAME="mini_run_final_sbatch_reduced_1"
WORK_DIR="${OUTPUT_DIR}/bevformer_small_occ/${RUN_NAME}"

mkdir -p "$WORK_DIR"

MOUNTS="${BASE_PROJECT_DIR}:/code,${DATA_DIR}:/truckscenes:ro,${GTS_DIR}:/gts:ro,${OUTPUT_DIR}:/output"

srun --mpi=pmi2 --unbuffered \
     --container-image="${IMAGE_DIR}/maxiobe123+bevformer+base-cuda111.sqsh" \
     --container-mounts="$MOUNTS" \
     bash -c '
       # --- Training starten ---
       cd /code/prediction/BEVFormer && \
       echo "Starting training..." && \
       /opt/conda/bin/conda run -n bev-former ./tools/dist_train.sh projects/configs/bevformer/bevformer_small_occ.py 1
     '

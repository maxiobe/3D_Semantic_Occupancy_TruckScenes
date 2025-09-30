#!/bin/bash
#
#SBATCH --partition=lrz-dgx-a100-80x8,lrz-hgx-a100-80x4
#SBATCH --job-name=bevformer-multigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/BEVFormer/work_dirs/bevformer_small_occ/multigpu_run_short/slurm_%j.out
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/BEVFormer/work_dirs/bevformer_small_occ/multigpu_run_short/slurm_%j.err


BASE_PROJECT_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes"
DATA_DIR="/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes"
GTS_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/trainval/gts"
IMAGE_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/images"
OUTPUT_DIR="${BASE_PROJECT_DIR}/prediction/BEVFormer/work_dirs"

# Updated run name for the multi-GPU experiment
RUN_NAME="multigpu_run_short"
WORK_DIR="${OUTPUT_DIR}/bevformer_small_occ/${RUN_NAME}"

mkdir -p "$WORK_DIR"

# --- Stable env ---
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1

# Rendezvous + NIC
export PORT=$((15000 + SLURM_JOB_ID % 20000))
export MASTER_PORT=$PORT
export MASTER_ADDR=127.0.0.1
if command -v ip >/dev/null 2>&1; then
  export NCCL_SOCKET_IFNAME=$(ip -o -4 route show to default | awk '{print $5}')
else
  export NCCL_SOCKET_IFNAME=eth0
fi


MOUNTS="${BASE_PROJECT_DIR}:/code,${DATA_DIR}:/truckscenes:ro,${GTS_DIR}:/gts:ro,${OUTPUT_DIR}:/output"

srun --ntasks=1 --gres=gpu:4 --cpus-per-task=${SLURM_CPUS_PER_TASK} \
     --container-image="${IMAGE_DIR}/maxiobe123+bevformer+base-cuda111.sqsh" \
     --container-mounts="$MOUNTS" \
     bash -lc '
       source /opt/conda/bin/activate bev-former
       cd /code/prediction/BEVFormer
       echo "Starting training on 4 GPUs (PORT=${PORT}, IFACE=${NCCL_SOCKET_IFNAME})..."
       ./tools/dist_train.sh projects/configs/bevformer/bevformer_small_occ.py 4 \
         --work-dir /output/bevformer_small_occ/'"$RUN_NAME"'\
	 --resume-from /output/bevformer_small_occ/multigpu_run/epoch_2.pth

     '

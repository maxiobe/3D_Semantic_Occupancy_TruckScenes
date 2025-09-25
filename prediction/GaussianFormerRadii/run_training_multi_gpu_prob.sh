#!/bin/bash
#
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=gaussian-former-2-multi-gpu
#SBATCH --time=2-0:00:00
#SBATCH --mem=512G
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# ---------------- Paths (edit if needed) ----------------
BASE_PROJECT_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes"
DATA_DIR="/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes"
GTS_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/trainval/gts"
IMAGE_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/images"
OUTPUT_DIR="${BASE_PROJECT_DIR}/prediction/GaussianFormerRadii/out"

RUN_NAME="run_4GPU_sbatch_a100_1_radii"
WORK_DIR="${OUTPUT_DIR}/prob/nuscenes_gs6400/${RUN_NAME}"

mkdir -p "${WORK_DIR}"

MOUNTS="${BASE_PROJECT_DIR}:/code,${DATA_DIR}:/truckscenes:ro,${GTS_DIR}:/gts:ro,${OUTPUT_DIR}:/output"

# ---------------- Launch container with your start pattern ----------------
srun --mpi=pmi2 --unbuffered \
     --container-image="${IMAGE_DIR}/maxiobe123+gaussianformer+base-cuda118-h100.sqsh" \
     --container-mounts="${MOUNTS}" \
     --gpu-bind=closest \
     bash -lc '
        set -euo pipefail

        echo "SLURM assigned GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
        nvidia-smi || true
        nvidia-smi topo -m || true

        # --- NCCL/CUDA env for single-node multi-GPU ---
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
        export NCCL_DEBUG=INFO
        export NCCL_IB_DISABLE=1


        # Avoid port collisions if multiple jobs run on same node
        export MASTER_ADDR=127.0.0.1
        export MASTER_PORT=$((20507 + (${SLURM_JOB_ID:-1} % 2000)))

        # Target archs for A100/H100
        export TORCH_CUDA_ARCH_LIST="8.0"

        # Make sure PyTorch libs are on LD_LIBRARY_PATH (use conda python)
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(conda run -n gaussianformer python - <<PY
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"

        echo "Python sees GPUs: $(conda run -n gaussianformer python - <<PY
import torch; print(torch.cuda.device_count())
PY
)"

        # --- One-time editable builds (idempotent) ---
        echo "[Build] Installing custom CUDA ops (GaussianFormer)"
        conda run -n gaussianformer pip install -e /code/prediction/GaussianFormer/model/encoder/gaussian_encoder/ops
        conda run -n gaussianformer pip install -e /code/prediction/GaussianFormer/model/head/localagg
        conda run -n gaussianformer pip install -e /code/prediction/GaussianFormer/model/head/localagg_prob
        conda run -n gaussianformer pip install -e /code/prediction/GaussianFormer/model/head/localagg_prob_fast

        # pointops (clone if missing)
        if [ ! -d /code/prediction/GaussianFormerRadii/installation/pointops ]; then
          echo "[Build] Cloning pointops"
          mkdir -p /code/prediction/GaussianFormerRadii/installation
          git clone https://github.com/xieyuser/pointops.git /code/prediction/GaussianFormerRadii/installation/pointops
        fi
        echo "[Build] Installing pointops (editable)"
        conda run -n gaussianformer pip install -e /code/prediction/GaussianFormerRadii/installation/pointops

	mkdir -p /output/prob/nuscenes_gs6400/'"$RUN_NAME"'

        # --- Start training (your exact command; spawns per-GPU inside) ---
        cd /code/prediction/GaussianFormerRadii
        echo "Starting training into /output/prob/nuscenes_gs6400/'"$RUN_NAME"' ..."
        conda run -n gaussianformer python -u train.py \
          --py-config config/prob/nuscenes_gs6400.py \
          --work-dir /output/prob/nuscenes_gs6400/'"$RUN_NAME"'
     '

echo "Job finished."

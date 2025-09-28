#!/bin/bash
#
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=gaussian-former-eval-1gpu
#SBATCH --time=0-06:00:00
#SBATCH --mem=128G
# Slurm logs
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/GaussianFormer/out/nuscenes_gs25600_solid/eval_1GPU_sbatch_1/slurm_%j.out
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/prediction/GaussianFormer/out/nuscenes_gs25600_solid/eval_1GPU_sbatch_1/slurm_%j.err

set -euo pipefail

# ---------------- Paths (adjust if needed) ----------------
BASE_PROJECT_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes"
DATA_DIR="/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes"
GTS_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/trainval/gts"
IMAGE_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/images"

# Outputs live under the repo's out/ dir (as in your training script)
OUTPUT_DIR="${BASE_PROJECT_DIR}/prediction/GaussianFormer/out"

# ---- Names/dirs ----
CONF_REL="config/nuscenes_gs25600_solid.py"         # config to eval with
TRAIN_RUN_NAME="run_2GPU_sbatch_1"                  # set to the training run to load from
EVAL_RUN_NAME="eval_1GPU_sbatch_1"                  # new eval run folder

WORK_DIR="${OUTPUT_DIR}/nuscenes_gs25600_solid/${EVAL_RUN_NAME}"
RESUME_FROM="${OUTPUT_DIR}/nuscenes_gs25600_solid/${TRAIN_RUN_NAME}/state_dict.pth"

mkdir -p "${WORK_DIR}"

# ---- Container & mounts ----
CONTAINER_IMAGE="${IMAGE_DIR}/maxiobe123+gaussianformer+base-cuda118-h100.sqsh"
MOUNTS="${BASE_PROJECT_DIR}:/code,${DATA_DIR}:/truckscenes:ro,${GTS_DIR}:/gts:ro,${OUTPUT_DIR}:/output"

# ---------------- Launch ----------------
srun --mpi=pmi2 --unbuffered \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${MOUNTS}" \
     --gpu-bind=closest \
     bash -lc '
        set -euo pipefail

        echo "SLURM GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
        nvidia-smi || true

        # --- Minimal env for single-GPU ---
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
        export TORCH_CUDA_ARCH_LIST="8.0 9.0"

        # Ensure torch libs are discoverable
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$(conda run -n gaussianformer python - <<PY
import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)"

        echo "Python sees GPUs: $(conda run -n gaussianformer python - <<PY
import torch; print(torch.cuda.device_count())
PY
)"

        # --- One-time editable builds needed for GaussianFormer v1 (no localagg_prob) ---
        echo "[Build] Installing CUDA ops + localagg (no *_prob*)"
        conda run -n gaussianformer pip install -e /code/prediction/GaussianFormer/model/encoder/gaussian_encoder/ops
        conda run -n gaussianformer pip install -e /code/prediction/GaussianFormer/model/head/localagg

        # --- Evaluation ---
        cd /code/prediction/GaussianFormer
        echo "Starting evaluation with:"
        echo "  --py-config '"${CONF_REL}"'"
        echo "  --work-dir  /output/nuscenes_gs25600_solid/'"${EVAL_RUN_NAME}"'"
        echo "  --resume-from /output/nuscenes_gs25600_solid/'"${TRAIN_RUN_NAME}"'/state_dict.pth"

        conda run -n gaussianformer python -u eval.py \
          --py-config '"${CONF_REL}"' \
          --work-dir /output/nuscenes_gs25600_solid/'"${EVAL_RUN_NAME}"' \
          --resume-from /output/nuscenes_gs25600_solid/'"${TRAIN_RUN_NAME}"'/state_dict.pth
     '

echo "Evaluation job submitted."

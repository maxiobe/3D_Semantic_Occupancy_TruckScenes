  GNU nano 6.2                                                                                                                                                                                                 run_test_prob.sh                                                                                                                                                                                                          
#!/bin/bash
#
#SBATCH --partition=lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-hgx-h100-94x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=gaussianformer_eval_1gpu
#SBATCH --time=0-02:00:00
#SBATCH --mem=512G
#SBATCH --output=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/GaussianFormer/out/prob/nuscenes_gs6400/run_4GPU_sbatch_a100_1/eval_vis/%x_%j.out
#SBATCH --error=/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes/GaussianFormer/out/prob/nuscenes_gs6400/run_4GPU_sbatch_a100_1/eval_vis/%x_%j.err

set -euo pipefail

# ---------------- Paths (adjust if needed) ----------------
BASE_PROJECT_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/3D_Semantic_Occupancy_TruckScenes"
DATA_DIR="/dss/dssfs04/pn69za/pn69za-dss-0004/datasets/man-truckscenes"
GTS_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/trainval/gts"
IMAGE_DIR="/dss/dssfs02/lwp-dss-0001/t7441/t7441-dss-0000/ge84von2/images"

# Persist outputs under the repo's ./out so that
# --work-dir out/xxxx works exactly as requested.
REPO_OUT_HOST="${BASE_PROJECT_DIR}/prediction/GaussianFormer/out"

# Ensure host out/ exists to receive results
mkdir -p "${REPO_OUT_HOST}/prob/nuscenes_gs6400/run_4GPU_sbatch_a100_1/eval_vis"

# Container & mounts
CONTAINER_IMAGE="${IMAGE_DIR}/maxiobe123+gaussianformer+base-cuda118-h100.sqsh"
MOUNTS="${BASE_PROJECT_DIR}:/code,${DATA_DIR}:/truckscenes:ro,${GTS_DIR}:/gts:ro,${REPO_OUT_HOST}:/code/prediction/GaussianFormer/out"

# ---------------- Launch ----------------
srun --mpi=pmi2 --unbuffered \
     --container-image="${CONTAINER_IMAGE}" \
     --container-mounts="${MOUNTS}" \
     --gpu-bind=closest \
     bash -lc '
        set -euo pipefail

        echo "SLURM GPUs: ${CUDA_VISIBLE_DEVICES:-unset}"
        nvidia-smi || true
        nvidia-smi topo -m || true

        # --- Env for single-GPU run ---
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
        export NCCL_DEBUG=INFO
        export NCCL_IB_DISABLE=1
        export NCCL_P2P_LEVEL=SYS
        export NCCL_NVLS_ENABLE=0

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
        if [ ! -d /code/prediction/GaussianFormer/installation/pointops ]; then
          echo "[Build] Cloning pointops"
          mkdir -p /code/prediction/GaussianFormer/installation
          git clone https://github.com/xieyuser/pointops.git /code/prediction/GaussianFormer/installation/pointops
        fi
        echo "[Build] Installing pointops (editable)"
        conda run -n gaussianformer pip install -e /code/prediction/GaussianFormer/installation/pointops

        # --- Evaluation ---
        cd /code/prediction/GaussianFormer
        echo "Starting evaluation..."
        echo "  --py-config config/prob/nuscenes_gs6400.py"
        echo "  --work-dir  out/prob/nuscenes_gs6400"
        echo "  --resume-from out/prob/nuscenes_gs6400/run_4GPU_sbatch_a100_1/latest.pth"

        conda run -n gaussianformer python -u eval_vis.py \
          --py-config config/prob/nuscenes_gs6400.py \
          --work-dir out/prob/nuscenes_gs6400/run_4GPU_sbatch_a100_1/eval_vis/ \
          --resume-from out/prob/nuscenes_gs6400/run_4GPU_sbatch_a100_1/latest.pth \
	  --vis-occ
     '

echo "Evaluation job finished."

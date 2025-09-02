# Stage 1: Use a modern CUDA base image that supports H100
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    ca-certificates \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set comprehensive environment variables for the CUDA toolkit
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}"
ENV FORCE_CUDA="1"

# === THE CRITICAL FIX: Include all target architectures ===
# 8.0 for A100, 8.6 for local RTX 30-series, 9.0 for H100
ENV TORCH_CUDA_ARCH_LIST="7.0 8.0 8.6 9.0"

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Accept Anaconda Terms of Service
RUN conda config --set auto_update_conda false && conda tos accept

# Create the conda environment
RUN conda create -n bev-former python=3.8 -y

# Set the SHELL to use the conda environment
SHELL ["conda", "run", "-n", "bev-former", "/bin/bash", "-c"]

# --- Install Updated Python Dependencies for CUDA 12.1 ---
# PyTorch and MMCV versions must match the new CUDA version
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
RUN pip install mmdet==3.2.0 mmsegmentation==1.2.2

# mmdetection3d v0.17.1 is old; a newer version is needed for mmcv 2.x
# We will install a compatible version directly.
RUN pip install mmdet3d==1.1.0

# Copy and install the remaining packages from requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set a final working directory
WORKDIR /workspace

# Set the default command to an interactive bash shell
CMD ["/bin/bash"]
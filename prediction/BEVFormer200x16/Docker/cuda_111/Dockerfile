# Use an official NVIDIA CUDA 11.1 development image as a parent image
# The 'devel' tag includes the compiler toolchain needed for mmcv-full
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies: git, wget, build tools, and ninja
RUN apt-get update && apt-get install -y \
    git \
    wget \
    build-essential \
    ca-certificates \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set comprehensive environment variables for the CUDA toolkit.
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda/include:${CPLUS_INCLUDE_PATH}"

# 8.0 for A100, 8.6 for local RTX 30-series
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"

# Tell the MMLab build systems to force CUDA compilation
ENV FORCE_CUDA="1"

# Install Miniconda for environment management
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh

# Add conda to PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Accept Anaconda Terms of Service
RUN conda config --set auto_update_conda false && conda tos accept

# Create the conda environment for BEVFormer with Python 3.8
RUN conda create -n bev-former python=3.8 -y

# Set the SHELL to use the conda environment for all subsequent RUN, CMD, and ENTRYPOINT commands
SHELL ["conda", "run", "-n", "bev-former", "/bin/bash", "-c"]

# --- Install Python Dependencies ---

# 1. Install PyTorch, torchvision, and torchaudio for CUDA 11.1
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# 2. Install mmcv-full, which depends on the correct PyTorch and CUDA versions
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# 3. Install mmdetection and mmsegmentation
RUN pip install mmdet==2.14.0 mmsegmentation==0.14.1

# --- Clone and Install mmdetection3d ---

# Set a working directory for our projects
WORKDIR /workspace

# Clone the specific version of mmdetection3d
RUN git clone https://github.com/open-mmlab/mmdetection3d.git
WORKDIR /workspace/mmdetection3d
RUN git checkout v0.17.1

# Install mmdetection3d in editable mode
RUN pip install -e .

# 4. Copy and install the remaining packages from requirements.txt
WORKDIR /workspace
COPY requirements.txt .
RUN pip install -r requirements.txt



# Set the final working directory
WORKDIR /workspace

RUN mkdir -p /truckscenes /data/mini /code

# Set the default command to an interactive bash shell
# This will drop you inside the container with the 'bev-former' environment active
CMD ["/bin/bash"]

# Stage 1: Setup a universal base environment
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

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
ENV FORCE_CUDA="1"

# Set target architectures. CRITICAL: Added "9.0" for H100 (Hopper) forward compatibility.
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0"

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

# --- Install Python Dependencies Explicitly in the Environment---

RUN conda run -n bev-former pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# 2. Install the correct version of mmcv-full for PyTorch 1.13 and CUDA 11.7
#RUN conda run -n bev-former pip install "mmcv-full==1.4.0" -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# 3. Install mmdetection and mmsegmentation
#RUN conda run -n bev-former pip install mmdet==2.14.0 mmsegmentation==0.14.1

# 4. Clone and Install mmdetection3d from source to ensure it compiles correctly
#WORKDIR /workspace
#RUN git clone -b v0.17.1 https://github.com/open-mmlab/mmdetection3d.git
#WORKDIR /workspace/mmdetection3d
#RUN conda run -n bev-former pip install -e .
RUN conda run -n bev-former pip install numba==0.53.0 llvmlite==0.36.0
RUN conda run -n bev-former pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.0/index.html
RUN conda run -n bev-former pip install mmdet==2.29.1
RUN conda run -n bev-former pip install mmsegmentation==0.30.0
RUN conda run -n bev-former pip install mmdet3d==1.0.0rc6
# 2. Install mmcv (prebuilt for CUDA 11.8 & Torch 2.0)
#RUN conda run -n bev-former pip install mmcv==2.0.0

# 3. Install mmdet + mmseg
#RUN conda run -n bev-former pip install mmdet==3.0.0 mmsegmentation==1.0.0

# 4. Install mmdet3d (latest RC compatible with mmcv 2.0)
#WORKDIR /workspace
#RUN git clone -b v1.0.0rc6 https://github.com/open-mmlab/mmdetection3d.git
#WORKDIR /workspace/mmdetection3d
#RUN conda run -n bev-former pip install -e .
#WORKDIR /workspace
#RUN git clone -b v0.17.1 https://github.com/open-mmlab/mmdetection3d.git
#WORKDIR /workspace/mmdetection3d
#RUN conda run -n bev-former pip install -e .

# 5. Copy and install the remaining packages from requirements.txt
WORKDIR /workspace
COPY requirements.txt .
RUN conda run -n bev-former pip install -r requirements.txt

# Set the final working directory
WORKDIR /workspace

# Set the default command to an interactive bash shell that activates the environment
CMD ["conda", "run", "-n", "bev-former", "/bin/bash"]


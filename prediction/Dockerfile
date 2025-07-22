# Step 1: Base Image
# Using an official NVIDIA image with CUDA 11.1, cuDNN 8, and development tools on Ubuntu 20.04.
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Step 2: Set Environment Variables
# Prevents prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive
# Sets the container's working directory.
WORKDIR /workspace

# Step 3: Install System Dependencies
# Install Python 3.8, pip, git, and other utilities.
# The `--no-install-recommends` flag reduces image size.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3-pip \
    git \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links to use `python` and `pip` directly.
RUN ln -sf /usr/bin/python3.8 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Step 4: Install Python Dependencies
# Upgrade pip first. Using `--no-cache-dir` keeps the image smaller.
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch, TorchVision, and TorchAudio for CUDA 11.1.
RUN pip install --no-cache-dir torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install MMCV-Full.
RUN pip install --no-cache-dir mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# Install the rest of the required Python packages.
RUN pip install --no-cache-dir \
    mmdet==2.14.0 \
    mmsegmentation==0.14.1 \
    einops fvcore seaborn \
    iopath==0.1.9 \
    timm==0.6.13 \
    pylint \
    ipython==8.12 \
    numpy==1.19.5 \
    matplotlib==3.5.2 \
    numba==0.48.0 \
    pandas==1.4.4 \
    scikit-image==0.19.3 \
    setuptools==59.5.0

# Step 5: Install MMDetection3D from Source
# Clone the required version of the repository.
RUN git clone https://github.com/open-mmlab/mmdetection3d.git mmdetection3d
WORKDIR /workspace/mmdetection3d
RUN git checkout v0.17.1
# Install the package in editable mode, which is common for MMLab projects.
RUN pip install --no-cache-dir -e .

# Return to the main workspace directory for a clean entry point.
WORKDIR /workspace

# Step 6: Copy Your Project Code (Optional)
# Uncomment the line below to copy your local project files into the container.
# This assumes your Dockerfile is in your project's root directory.
# COPY . .

# Step 7: Set Default Command
# This starts an interactive bash shell. You can change this to run your code directly,
# e.g., CMD ["python", "tools/train.py", "configs/my_config.py"]
CMD ["bash"]
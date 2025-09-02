# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system-level dependencies including build-essential
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    python3-pip \
    libturbojpeg-dev &&\
    rm -rf /var/lib/apt/lists/*

# 7.5 (T4), 8.0 (A100), 8.6 (RTX 30-series), 9.0 (H100, RTX 40-series)
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;9.0"
# --- CUDA Environment Variables for runtime compilation ---
# These will be used by the srun script to ensure compilation works
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV FORCE_CUDA="1"

RUN pip install --no-cache-dir --upgrade pip && \
    \
    # 1. Install PyTorch from its specific index
    pip install --no-cache-dir \
        torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118 && \
    \
    # 2. Now install openmim and other tools from the default PyPI
    pip install --no-cache-dir openmim==0.3.9 && \
    \
    # 3. Use mim to install the mm* series of packages
    mim install mmcv-full==1.6.0 && \
    mim install mmdet==2.28.2 && \
    mim install mmsegmentation==0.30.0 && \
    mim install mmdet3d==1.0.0rc6 && \
    \
    # 4. Install specific versions of core libraries
    pip install --no-cache-dir \
        setuptools==59.5.0 \
        numpy==1.23.5 \
        pyturbojpeg && \
    \
    # 5. Replace pillow with the faster pillow-simd
    pip uninstall -y pillow && \
    pip install --no-cache-dir pillow-simd==9.0.0.post1 && \
    \
    # 6. Clean up the mim cache to save space
    rm -rf ~/.mim/cache

# Set the final working directory
WORKDIR /workspace

# Set the default command to bash
CMD ["bash"]


# Use an official NVIDIA CUDA runtime as a parent image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

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

# --- CUDA Environment Variables for runtime compilation ---
# These will be used by the srun script to ensure compilation works
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH}"
ENV FORCE_CUDA="1"

# --- Install Python Dependencies ---
# These are all platform-independent and can be pre-installed
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.0.0 torchvision==0.15.0 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir openmim==0.3.9
RUN mim install mmcv-full==1.6.0
RUN mim install mmdet==2.28.2
RUN mim install mmsegmentation==0.30.0
RUN mim install mmdet3d==1.0.0rc6
RUN pip install --no-cache-dir setuptools==59.5.0 numpy==1.23.5
RUN pip install --no-cache-dir pyturbojpeg
RUN pip uninstall -y pillow
RUN pip install --no-cache-dir pillow-simd==9.0.0.post1

# NOTE: The project code is NOT copied and the CUDA extension is NOT compiled here.
# This will be done inside the srun command.

# Set the final working directory
WORKDIR /workspace

# Set the default command to bash
CMD ["bash"]


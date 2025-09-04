# Start from an official NVIDIA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set up ARG for non-interactive frontend
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Explicitly tell pytorch3d to build with CUDA support
ENV FORCE_CUDA="1"

# Install system dependencies and NVIDIA CUB. The C++ toolchain is now handled by Conda.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    bzip2 \
    unzip \
    ca-certificates \
    libgl1-mesa-glx \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtheora-dev \
    libopenblas-dev && \
    # Download and install NVIDIA CUB
    wget https://github.com/NVIDIA/cub/archive/1.17.2.zip -O cub.zip && \
    unzip cub.zip -d /opt && \
    rm cub.zip && \
    # Clean up apt cache
    rm -rf /var/lib/apt/lists/*

# Set CUB_HOME environment variable
ENV CUB_HOME /opt/cub-1.17.2

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Set up the shell for conda activation
RUN conda init bash

# Set the number of CPU threads for OpenMP
ENV OMP_NUM_THREADS=12

# === INSTALLATION IN CORRECT ORDER ===

# 1. Create the BASE conda environment using the robust conda-forge setup.
COPY Docker/environment.yaml /tmp/environment.yaml
RUN conda clean --all --yes && \
    conda env create -f /tmp/environment.yaml && \
    conda clean --all --yes

# CRITICAL FIX: Force the system's dynamic linker to prioritize the Conda C++ library.
# This solves the `CXXABI_1.3.15 not found` runtime error.
ENV LD_LIBRARY_PATH=/opt/conda/envs/occ_kiss_p3d/lib:$LD_LIBRARY_PATH
ENV TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1

# 2. Install PyTorch using Pip.
RUN conda run -n occ_kiss_p3d pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Build PyTorch3D from source.
RUN conda run -n occ_kiss_p3d sh -c '\
    pip install fvcore iopath && \
    TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX" pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
    '

# 4. Install other pip dependencies.
RUN conda run -n occ_kiss_p3d pip install \
    open3d \
    addict \
    truckscenes-devkit==1.0.0 \
    pyquaternion==0.9.9

# 5. Install mmcv-full using the robust find-links method.
RUN conda run -n occ_kiss_p3d pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

# 6. Build MinkowskiEngine, forcing CUDA compilation.
#RUN conda run -n occ_kiss_p3d pip install MinkowskiEngine -f https://nvidia-minkowskiengine.s3.amazonaws.com/whl/torch-2.1.0_cu118.html
#RUN conda run -n occ_kiss_p3d pip install -U git+https://github.com/NVIDIA/MinkowskiEngine --no-build-isolation
#RUN conda run -n occ_kiss_p3d sh -c '\
#    MINKOWSKI_FORCE_CUDA=1 pip install -U MinkowskiEngine --config-settings="--build-option=--blas=openblas" --no-deps --no-build-isolation \
#    '
#RUN conda run -n occ_kiss_p3d sh -c '\
#    FORCE_CUDA=1 python -m pip install -U git+https://github.com/NVIDIA/MinkowskiEngine.git \
#    --no-build-isolation --no-deps \
#'
# Install compiler toolchain into the conda env (so it links against the right libstdc++)
#RUN conda install -n occ_kiss_p3d -c conda-forge gxx_linux-64=12 cmake ninja

# Build MinkowskiEngine from source with CUDA enabled
#RUN conda run -n occ_kiss_p3d bash -c '\
 #   export FORCE_CUDA=1 && \
  #  export CUDA_HOME=/usr/local/cuda && \
  #  export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX" && \
  #  pip install -U git+https://github.com/NVIDIA/MinkowskiEngine.git --no-build-isolation --no-deps \
#'

# --- YOUR LOCAL CODE GOES LAST ---

# 7. Install local kiss-icp from source
COPY kiss-icp /tmp/kiss-icp
RUN cd /tmp/kiss-icp && \
    conda run -n occ_kiss_p3d make install && \
    cd / && \
    rm -rf /tmp/kiss-icp

# 8. Install local MapMOS from source
COPY MapMOS /tmp/MapMOS
RUN cd /tmp/MapMOS && \
    conda run -n occ_kiss_p3d sh -c 'CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" make install' && \
    cd / && \
    rm -rf /tmp/MapMOS

# --- Final Configuration ---
WORKDIR /app
CMD ["/bin/bash", "-c", "conda activate occ_kiss_p3d; exec bash"]


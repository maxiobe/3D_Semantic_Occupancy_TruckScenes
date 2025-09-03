# Start from an official NVIDIA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set up ARG for non-interactive frontend
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    bzip2 \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# --- CHANGE: Install Miniconda instead of Micromamba ---
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.11.0-2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# --- CHANGE: Set up the shell for conda activation ---
RUN conda init bash

# Set the number of CPU threads for OpenMP for better performance control
ENV OMP_NUM_THREADS=12

# === INSTALLATION IN CORRECT ORDER ===

# 1. Create the BASE conda environment from the minimal YAML file
# --- CHANGE: Use 'conda' instead of 'micromamba' ---
#COPY environment.yaml /tmp/environment.yaml
COPY Docker/environment.yaml /tmp/environment.yaml
RUN conda env create -f /tmp/environment.yaml && conda clean --all --yes

# 2. Install the correct GPU-enabled PyTorch using Pip.
# --- CHANGE: Use 'conda run' ---
RUN conda run -n occ_kiss_p3d pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# 3. Now, install libraries that depend on PyTorch (like pytorch3d).
# --- CHANGE: Use 'conda install' and 'conda run' ---
RUN conda install -n occ_kiss_p3d -c pytorch3d pytorch3d -y

# 4. Install the remaining pip dependencies
# --- CHANGE: Use 'conda run' ---
RUN conda run -n occ_kiss_p3d pip install addict iopath mmcv-full==1.7.2 truckscenes-devkit pyquaternion

# 5. FINALLY, build MinkowskiEngine.
# --- CHANGE: Use 'conda run' ---
RUN conda run -n occ_kiss_p3d pip install -U MinkowskiEngine --config-settings='--build-option=--blas=openblas' --no-deps --no-build-isolation

# 6. Install local kiss-icp from source
# This copies the 'kiss-icp' directory from your build context into the image.
COPY kiss-icp /tmp/kiss-icp
RUN cd /tmp/kiss-icp && \
    conda run -n occ_kiss_p3d make install && \
    cd / && \
    rm -rf /tmp/kiss-icp

# 7. Install local MapMOS from source
# This copies the 'MapMOS' directory from your build context into the image.
COPY MapMOS /tmp/MapMOS
RUN cd /tmp/MapMOS && \
    conda run -n occ_kiss_p3d sh -c 'CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5" make install' && \
    cd / && \
    rm -rf /tmp/MapMOS

# --- Final Configuration ---
WORKDIR /app
# --- CHANGE: Use 'conda activate' in the final CMD ---
CMD ["/bin/bash", "-c", "conda activate occ_kiss_p3d; exec bash"]
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

# Install Micromamba
RUN wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj -C /usr/local/bin/ --strip-components=1 bin/micromamba

# Set up the shell for micromamba activation
RUN /usr/local/bin/micromamba shell init -s bash -r ~/micromamba

# Set the number of CPU threads for OpenMP for better performance control
ENV OMP_NUM_THREADS=12

# === INSTALLATION IN CORRECT ORDER ===

# 1. Create the BASE conda environment from the minimal YAML file
COPY environment.yaml /tmp/environment.yaml
RUN /bin/bash -l -c "micromamba create -f /tmp/environment.yaml -y && micromamba clean --all --yes"

# 2. Install the correct GPU-enabled PyTorch using Pip. This is now the ONLY version.
RUN /bin/bash -l -c "micromamba run -n occ_kiss_p3d pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118"

# 3. Now, install libraries that depend on PyTorch (like pytorch3d).
#    Micromamba will see torch is already installed and will not add a new one.
RUN /bin/bash -l -c "micromamba install -n occ_kiss_p3d -c pytorch3d pytorch3d -y"

# 4. Install the remaining pip dependencies
RUN /bin/bash -l -c "micromamba run -n occ_kiss_p3d pip install addict iopath mmcv-full==1.7.2 truckscenes-devkit pyquaternion"

# 5. FINALLY, build MinkowskiEngine. It will now see the correct GPU PyTorch.
RUN /bin/bash -l -c "micromamba run -n occ_kiss_p3d pip install -U MinkowskiEngine --config-settings='--build-option=--blas=openblas' --no-deps --no-build-isolation"

# --- Final Configuration ---
WORKDIR /app
CMD ["/bin/bash", "-l", "-c", "micromamba activate occ_kiss_p3d; exec bash"]
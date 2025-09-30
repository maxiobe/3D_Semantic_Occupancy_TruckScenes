# Stage 1: Setup a universal base environment
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential system packages, including build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    cmake \
    libboost-all-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    xvfb \
    qtbase5-dev \
    libqt5x11extras5 \
    qttools5-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# --- CUDA Environment Variables for runtime compilation ---
# These will be used by the srun script later
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV CPLUS_INCLUDE_PATH="${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH}"
ENV FORCE_CUDA="1"

# Create the conda environment
RUN conda create -n gaussianformer python=3.8.16 -y

# Configure the shell to automatically use the 'gaussianformer' conda environment
SHELL ["conda", "run", "-n", "gaussianformer", "/bin/bash", "-c"]

ENV TORCH_CUDA_ARCH_LIST="8.0 9.0"

# This flag forces MMCV to compile its custom CUDA operators.
ENV MMCV_WITH_OPS=1

# Install all Python dependencies
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

RUN pip install openmim && \
    pip install mmdet==3.0.0 mmsegmentation==1.0.0 mmdet3d==1.1.1

RUN pip install mmcv==2.0.1 --no-binary :all: && \
    pip install spconv-cu118==2.3.6

RUN pip install timm einops jaxtyping

RUN pip install pyvirtualdisplay mayavi pyqt5 matplotlib pyquaternion

# Set a default working directory
WORKDIR /workspace

# Set the default command to an interactive shell
CMD ["/bin/bash"]

# Stage 1: Setup the base environment and install dependencies
# Use an official NVIDIA CUDA image compatible with PyTorch's cu118
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables to enable non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# --- NEW LINES ADDED HERE ---
# Set paths for CUDA headers and libraries so the C++ compiler can find them
ENV CPLUS_INCLUDE_PATH="/usr/local/cuda/include:$CPLUS_INCLUDE_PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
# --- END OF NEW LINES ---

# Create the conda environment
RUN conda create -n gaussianformer python=3.8.16 -y

# Configure the shell to automatically use the 'gaussianformer' conda environment
SHELL ["conda", "run", "-n", "gaussianformer", "/bin/bash", "-c"]

# Install PyTorch
RUN pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# Verify the PyTorch and GPU installation
RUN python -c "import torch; v = torch.__version__; print(f'---> PyTorch version: {v}'); a = torch.cuda.is_available(); print(f'---> GPU available: {a}'); print(f'---> GPU name: {torch.cuda.get_device_name(0)}') if a else print('---> No GPU found')"

# Install OpenMIM and the required MM libraries
RUN pip install openmim
RUN mim install mmcv==2.0.1
RUN mim install mmdet==3.0.0
RUN mim install mmsegmentation==1.0.0
RUN mim install mmdet3d==1.1.1

# Install remaining packages
RUN pip install spconv-cu117 timm
RUN pip install einops jaxtyping

# Set a working directory for the project code
WORKDIR /workspace

# Clone the project repository
RUN git clone https://github.com/maxiobe/3D_Semantic_Occupancy_TruckScenes.git

# Set the working directory to the specific sub-project folder
WORKDIR /workspace/3D_Semantic_Occupancy_TruckScenes/prediction/GaussianFormer

# Set the target GPU architecture before compiling custom CUDA code
# 8.6 is for Ampere (e.g., RTX 30-series). Add others if needed (e.g., "7.5 8.6").
ENV TORCH_CUDA_ARCH_LIST="8.6"

# Install the local, editable packages
RUN pip install -e ./model/encoder/gaussian_encoder/ops
RUN pip install -e ./model/head/localagg

# Unset the variable so it doesn't interfere with other potential builds
ENV TORCH_CUDA_ARCH_LIST=""

# Set the final working directory
WORKDIR /workspace/3D_Semantic_Occupancy_TruckScenes

# Set the default command
CMD ["/bin/bash"]
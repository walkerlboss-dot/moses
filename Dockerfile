# Dockerfile — Moses DGX Spark Environment
# Base: NVIDIA NGC PyTorch container + Isaac Sim + Isaac Lab layers

FROM nvcr.io/nvidia/pytorch:24.02-py3

LABEL maintainer="moses@boss.industries"
LABEL version="2.0"
LABEL description="Moses humanoid robotics build environment for DGX Spark"

# Build arguments
ARG CUDA_VERSION=12.3
ARG ISAAC_SIM_VERSION=4.0.0
ARG ISAAC_LAB_VERSION=1.0.0

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0"  # A100, A100-80GB, H100

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    wget \
    curl \
    vim \
    tmux \
    htop \
    nvtop \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    libusb-1.0-0 \
    libudev-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Isaac Sim (headless mode for DGX)
RUN mkdir -p /isaac-sim && cd /isaac-sim \
    && wget -q https://download.isaacsim.omniverse.nvidia.com/isaac-sim-${ISAAC_SIM_VERSION}.zip \
    && unzip -q isaac-sim-${ISAAC_SIM_VERSION}.zip \
    && rm isaac-sim-${ISAAC_SIM_VERSION}.zip

ENV ISAAC_SIM_PATH=/isaac-sim
ENV PATH=${ISAAC_SIM_PATH}:${PATH}

# Install Isaac Lab
RUN pip install --no-cache-dir \
    isaaclab==${ISAAC_LAB_VERSION} \
    isaaclab-rl==${ISAAC_LAB_VERSION}

# Install TensorRT
RUN pip install --no-cache-dir \
    tensorrt==8.6.1 \
    onnx==1.15.0 \
    onnxruntime-gpu==1.16.0

# Install ML/RL stack
RUN pip install --no-cache-dir \
    stable-baselines3==2.2.1 \
    gymnasium==0.29.1 \
    ray[train]==2.9.0 \
    wandb==0.16.0 \
    optuna==3.5.0 \
    hydra-core==1.3.2 \
    omegaconf==2.3.0

# Install ROS2 Humble (minimal)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository universe \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y ros-humble-ros-base \
    && apt-get install -y python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=humble
ENV ROS_ROOT=/opt/ros/humble
ENV PATH=${ROS_ROOT}/bin:${PATH}
ENV PYTHONPATH=${ROS_ROOT}/lib/python3.10/site-packages:${PYTHONPATH}

# Install MuJoCo (fallback)
RUN pip install --no-cache-dir mujoco==3.1.0

# Install testing utilities
RUN pip install --no-cache-dir \
    pytest==7.4.3 \
    pytest-cov==4.1.0 \
    hypothesis==6.92.0 \
    black==23.12.0 \
    ruff==0.1.8 \
    mypy==1.7.0

# Create Moses workspace
RUN mkdir -p /workspace/moses-builds
WORKDIR /workspace/moses-builds

# Copy knowledge base (if available at build time)
COPY knowledge/ /workspace/moses-builds/knowledge/

# Set up git
RUN git init && \
    git config user.name "Moses Builder" && \
    git config user.email "moses@boss.industries"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA OK')" || exit 1

# Default command: start build loop
CMD ["/bin/bash", "-c", "echo 'Moses DGX environment ready. Run build loop to start.' && /bin/bash"]

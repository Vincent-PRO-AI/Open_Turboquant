FROM pytorch/pytorch:2.11.0-cuda13.1-cudnn9-devel

# Set non-interactive to avoid prompt hangs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for Triton and model building
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy project requirements
COPY requirements.txt .

# Install dependencies natively under Linux
# Triton will install successfully here
RUN pip install -r requirements.txt

# Pre-install core library for development mode
RUN pip install -e .

# Command to run (defaults to bash overlay)
CMD ["/bin/bash"]

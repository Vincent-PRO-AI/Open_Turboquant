FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt installations
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
ENV MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"

# Install Python, git, and build-essential for Triton JIT compilation in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Fix python executable path
RUN ln -s /usr/bin/python3 /usr/bin/python

COPY . /app/

# Install PyTorch wheels specifically built for cu121, followed by dependencies to minimize layers
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir accelerate bitsandbytes scipy matplotlib triton transformers \
    && pip install --no-cache-dir -e .

CMD ["/bin/bash"]

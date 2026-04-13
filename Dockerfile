# Lighter runtime image for POC to avoid disk space issues on GH Actions
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app
ENV MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct"

# Install git, build-essential (for Triton JIT), and clean up
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

# Copy the repository contents into the container
COPY . /app/

# Install Open TurboQuant and its dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir accelerate bitsandbytes scipy matplotlib

# Set the default command to bash (model inference should be run via explicit scripts)
# Example: docker run -it --gpus all <image> python examples/poc_chat.py
CMD ["/bin/bash"]

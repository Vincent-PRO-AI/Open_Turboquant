# Base image with PyTorch and CUDA runtime pre-installed
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /app

# Install git and other system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the repository contents into the container
COPY . /app/

# Install Open TurboQuant and its dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir accelerate bitsandbytes scipy matplotlib

# Set the default command to run the POC validation script
# This proves the installation works immediately upon `docker run`
CMD ["python", "examples/local_universal_validation.py"]

# Lighter runtime image for POC to avoid disk space issues on GH Actions
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install git and other system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy the repository contents into the container
COPY . /app/

# Install Open TurboQuant and its dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir accelerate bitsandbytes scipy matplotlib

# Set the default command to run the interactive POC Chat
# Requires HF_TOKEN to be passed for gated models like Gemma
CMD ["python", "examples/poc_chat.py"]

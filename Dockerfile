FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install only the tools needed for Triton JIT and the library
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/

# Install the dependencies and the package
RUN pip install --no-cache-dir accelerate bitsandbytes scipy matplotlib transformers
RUN pip install --no-cache-dir -e .

CMD ["/bin/bash"]

# STAGE 1: Builder
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install build tools and python
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to make copying easier and cleaner
RUN pip install --no-cache-dir virtualenv
RUN virtualenv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install heavy dependencies (Torch takes up most space)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 \
    && pip install --no-cache-dir accelerate bitsandbytes scipy matplotlib triton transformers

# Install the local package
COPY . /app/
RUN pip install --no-cache-dir -e .

# STAGE 2: Final Runtime
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive

# Install only the runtime python and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Copy the entire virtual environment from the builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the code
COPY . /app/

CMD ["/bin/bash"]

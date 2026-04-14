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

# Expose the API port
EXPOSE 8000

# Run the FastAPI server using the python module syntax (more robust)
CMD ["python3", "-m", "uvicorn", "tq_impl.server:app", "--host", "0.0.0.0", "--port", "8000"]

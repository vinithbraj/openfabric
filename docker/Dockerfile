# Use Ubuntu 24.04 as the base image
FROM ubuntu:24.04

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    jq \
    realpath \
    slurm-client \
    docker.io \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the gateway port
EXPOSE 8310

# Default environment variables (can be overridden)
ENV LLM_OPS_TIMEOUT_SECONDS=300
ENV OPENFABRIC_GATEWAY_STREAM_PROGRESS=True

# Entry point for the application
ENTRYPOINT ["python3", "openwebui_gateway.py"]

# Default arguments
CMD ["--host", "0.0.0.0", "--port", "8310"]

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

# Setup a virtual environment for Python dependencies (recommended for Ubuntu 24.04 / PEP 668)
RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default shell
CMD ["/bin/bash"]

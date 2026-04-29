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
    docker.io \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Keep the virtualenv outside the bind-mounted source tree so mounting the
# repository at /opt/src does not hide the installed console scripts.
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/src/src"

# Install from pyproject.toml. Editable mode keeps the installed `aor` command
# pointed at /opt/src, which is the same path used by docker-compose bind mounts.
WORKDIR /opt/src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy the rest of the repository for non-bind-mounted image usage. In compose,
# the host repository is mounted over /opt/src.
COPY . .

# Default shell
CMD ["/bin/bash"]

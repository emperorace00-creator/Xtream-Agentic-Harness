# Dockerfile — Emperor sandbox image
# Provides the isolated environment where the bash tool runs code.
# The host (start.py) runs natively — only bash commands are proxied into this container.
#
# Build:
#   docker build -t emperor-base:latest .
#
# The container is started automatically by start.py on first run.
# You only need to build the image once (or after editing this file).

FROM python:3.12-slim

# System deps for pymupdf (PDF rendering), diagram generation, and
#    general tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmupdf-dev \
    mupdf-tools \
    graphviz \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python packages available inside the sandbox
# These let the agent run code, process data, and generate outputs.
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    scipy \
    sympy \
    rich \
    requests \
    Pillow \
    pymupdf

# Container layout
# Created so Docker can verify mount points at container start.
# Actual contents come from bind mounts configured by start.py.
RUN mkdir -p /uploads /outputs /workspace/scratch

# Entrypoint: create workspace skeleton, then keep container alive
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace/scratch
CMD ["/entrypoint.sh"]

#!/bin/bash
# entrypoint.sh — Emperor sandbox container entrypoint
#
# Creates the workspace directory skeleton on first boot, then sleeps
# forever so the container stays alive for `docker exec` commands.
#
# The actual work happens when start.py runs `docker exec` to execute
# bash commands inside this container. This script just keeps it running.

set -e

# Ensure workspace structure exists (survives named volume resets)
mkdir -p /workspace/scratch
mkdir -p /uploads
mkdir -p /outputs

echo "Emperor sandbox ready."

# Keep container alive — start.py sends commands via `docker exec`
exec sleep infinity

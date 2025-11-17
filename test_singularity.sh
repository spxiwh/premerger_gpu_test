#!/bin/bash
# Test script to verify run_single_likelihood_batch.py runs in the shared code environment

set -e  # Exit on error

SINGULARITY_IMAGE="shared_code_image.sif"

echo "========================================"
echo "Testing LISA Pre-merger GPU Pipeline"
echo "========================================"
echo ""

# Check if apptainer is available, fallback to singularity
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "ERROR: neither apptainer nor singularity command found"
    exit 1
fi

echo "Using container command: $CONTAINER_CMD"

# Check if SIF image exists
if [ ! -f "$SINGULARITY_IMAGE" ]; then
    echo "ERROR: Singularity image '$SINGULARITY_IMAGE' not found"
    echo "Please build the image first with:"
    echo "  apptainer build shared_code_image.sif docker://ghcr.io/uk-lisa-gs/shared_code_environment:latest-cuda12"
    exit 1
fi

# Get the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Running run_single_likelihood_batch.py in container..."
echo "Working directory: $SCRIPT_DIR"
echo "Singularity image: $SINGULARITY_IMAGE"
echo ""

# Run with GPU support
$CONTAINER_CMD exec --nv \
    --bind "$SCRIPT_DIR:$SCRIPT_DIR" \
    --pwd "$SCRIPT_DIR" \
    "$SINGULARITY_IMAGE" \
    python run_single_likelihood_batch.py

echo ""
echo "========================================"
echo "Test completed successfully!"
echo "========================================"

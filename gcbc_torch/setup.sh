#!/bin/bash
# Setup script for gcbc_torch (PyTorch GC-BC for ISpatialGym)
#
# Creates a conda env "torchrl" with PyTorch (CUDA 12.6, B300-compatible)
# and all dependencies needed for training and evaluation.
#
# Usage:
#   bash gcbc_torch/setup.sh

set -euo pipefail

ENV_NAME="torchrl"
PYTHON_VERSION="3.10"

echo "=== Setting up gcbc_torch environment ==="

# Create conda env
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists. Skipping creation."
else
    echo "Creating conda env '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

# PyTorch with CUDA 12.6 (supports RTX 4090 + B300/Blackwell)
echo "Installing PyTorch (CUDA 12.6)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# TensorFlow (for TFRecord data loading only, CPU is fine)
echo "Installing TensorFlow..."
pip install tensorflow

# Training & eval dependencies
echo "Installing dependencies..."
pip install wandb matplotlib tqdm Pillow av pandas ml_collections

echo ""
echo "=== Setup complete ==="
echo "Activate with:  conda activate ${ENV_NAME}"
echo ""
echo "Train example:"
echo "  python -m gcbc_torch.train \\"
echo "      --tfrecord_dir gcbc_jax/tfrecords/task-0053-final \\"
echo "      --save_dir outputs/gcbc_torch_task0053 \\"
echo "      --num_steps 50000 --batch_size 256 \\"
echo "      --use_proprio --normalize_proprio"

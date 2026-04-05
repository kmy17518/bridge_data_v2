#!/bin/bash
# Setup script for bridge_data_v2 GC-BC training environment (JAX/Flax).
#
# Tested on:
#   - Ubuntu 22.04 (kernel 6.8.0)
#   - NVIDIA RTX 4090, Driver 580.126.09, CUDA 13.0
#   - Miniconda / Miniforge
#
# Creates a conda environment "jaxrl" with:
#   - Python 3.10
#   - JAX 0.4.13 + jaxlib 0.4.13 (CUDA 12, cuDNN 8.9)
#   - TensorFlow 2.13 (CPU only, used for tf.data pipeline)
#   - Flax 0.7, optax 0.1.5, distrax 0.1.2
#   - PyAV (video decoding), pandas/pyarrow (parquet reading)
#   - WandB (experiment tracking)
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
# After setup, activate with:
#   conda activate jaxrl

set -euo pipefail

ENV_NAME="jaxrl"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Creating conda environment: ${ENV_NAME} ==="
conda create -n "${ENV_NAME}" python=3.10 -y

echo ""
echo "=== Installing bridge_data_v2 package (jaxrl_m) ==="
conda run -n "${ENV_NAME}" --no-banner pip install -e "${SCRIPT_DIR}"

echo ""
echo "=== Installing requirements.txt ==="
# This installs JAX, Flax, TensorFlow, and all bridge_data_v2 dependencies.
# Note: requirements.txt pins jax==0.4.13 but installs jaxlib without CUDA.
# We fix that in the next step.
conda run -n "${ENV_NAME}" --no-banner pip install -r "${SCRIPT_DIR}/requirements.txt"

echo ""
echo "=== Installing JAX with CUDA 12 support ==="
# jaxlib must match jax==0.4.13 exactly. The requirements.txt may install a
# newer jaxlib; we force the correct version with CUDA 12 + cuDNN 8.9.
conda run -n "${ENV_NAME}" --no-banner pip install \
    "jaxlib==0.4.13+cuda12.cudnn89" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo ""
echo "=== Downgrading scipy for JAX 0.4.13 compatibility ==="
# JAX 0.4.13 uses scipy.linalg.tril which was removed in scipy>=1.14.
# Pin to 1.11.x to avoid AttributeError.
conda run -n "${ENV_NAME}" --no-banner pip install "scipy==1.11.4"

echo ""
echo "=== Installing additional dependencies for ISpatialGym data ==="
# PyAV: decode MP4 video frames from observation recordings
# pandas + pyarrow: read parquet episode files
conda run -n "${ENV_NAME}" --no-banner pip install av pandas pyarrow

echo ""
echo "=== Verifying installation ==="
conda run -n "${ENV_NAME}" --no-banner python -c "
import jax
print(f'JAX version:    {jax.__version__}')
print(f'JAX devices:    {jax.devices()}')

import jaxlib
print(f'jaxlib version: {jaxlib.__version__}')

import flax
print(f'Flax version:   {flax.__version__}')

import tensorflow as tf
print(f'TF version:     {tf.__version__}')

from jaxrl_m.agents.continuous.gc_bc import GCBCAgent
print(f'GCBCAgent:      OK')

from jaxrl_m.vision import encoders
print(f'Encoders:       {list(encoders.keys())}')

import av
print(f'PyAV:           OK')

import pandas
print(f'pandas:         {pandas.__version__}')

print()
print('All checks passed.')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Convert data to TFRecords:"
echo "  python -m gcbc_jax.convert_to_tfrecord \\"
echo "    --data_dir <path/to/parquet/dir> \\"
echo "    --output_dir gcbc_jax/tfrecords/<task> \\"
echo "    --project_root <path/to/behavior-1k-private> \\"
echo "    --image_size 256"
echo ""
echo "Train GC-BC:"
echo "  python -m gcbc_jax.train \\"
echo "    --tfrecord_dir gcbc_jax/tfrecords/<task> \\"
echo "    --save_dir outputs/gcbc_jax_<task> \\"
echo "    --num_steps 50000 --batch_size 256 --use_wandb"

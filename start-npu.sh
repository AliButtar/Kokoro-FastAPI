#!/usr/bin/env bash

# Activate kokoro-api conda environment if uv is not already in PATH
if ! command -v uv &>/dev/null; then
    CONDA_ENV_BIN="/root/.conda/envs/kokoro-api/bin"
    if [ -f "$CONDA_ENV_BIN/uv" ]; then
        export PATH="$CONDA_ENV_BIN:$PATH"
        echo "Using Python $(${CONDA_ENV_BIN}/python --version 2>&1) environment at: /root/.conda/envs/kokoro-api"
    else
        echo "Error: uv not found. Please activate the kokoro-api conda environment first." >&2
        exit 1
    fi
fi

# Get project root directory
PROJECT_ROOT=$(pwd)

# Source Ascend CANN environment (required for torch_npu / ACL initialization)
CANN_SET_ENV="/root/cann-env/Ascend/ascend-toolkit/set_env.sh"
if [ -f "$CANN_SET_ENV" ]; then
    source "$CANN_SET_ENV"
else
    echo "Warning: CANN set_env.sh not found at $CANN_SET_ENV" >&2
fi

# Set environment variables
export USE_GPU=true
export DEVICE_TYPE=npu:0
# Append project paths to PYTHONPATH - do NOT overwrite CANN paths set by set_env.sh above
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api${PYTHONPATH:+:$PYTHONPATH}
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web
# ASCEND_RT_VISIBLE_DEVICES=0 exposes only device 0; DEVICE_TYPE=npu:0 targets it
export ASCEND_RT_VISIBLE_DEVICES=0

# Install NPU extras + decorator (needed by CANN's tbe module) in one step
uv pip install -e ".[npu]" decorator
# triton is a CUDA compilation tool, incompatible with torch 2.6.0 on aarch64 NPU;
# remove it after all installs to prevent the aarch64 override from keeping it
uv pip uninstall triton 2>/dev/null || true
# Patch kokoro's TorchSTFT.transform: Ascend NPU aclnnAbs/aclnnAngle do not support
# complex tensors; replace with real/imag component arithmetic instead.
python3 - <<'PATCH'
import re, pathlib
f = pathlib.Path(".venv/lib/python3.10/site-packages/kokoro/istftnet.py")
src = f.read_text()
old = "        return torch.abs(forward_transform), torch.angle(forward_transform)"
new = (
    "        # NPU: aclnnAbs/aclnnAngle do not support complex tensors\n"
    "        real = forward_transform.real\n"
    "        imag = forward_transform.imag\n"
    "        return torch.sqrt(real ** 2 + imag ** 2), torch.atan2(imag, real)"
)
if old in src:
    f.write_text(src.replace(old, new))
    print("Applied NPU STFT patch to kokoro/istftnet.py")
else:
    print("NPU STFT patch already applied or not needed")
PATCH
uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0
# Activate the venv directly (not uv run) so CANN PYTHONPATH is inherited by uvicorn
source .venv/bin/activate
python -m uvicorn api.src.main:app --host 0.0.0.0 --port 8880

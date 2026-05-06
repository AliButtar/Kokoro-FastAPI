#!/usr/bin/env bash

# Get project root directory
PROJECT_ROOT=$(pwd)

# Set environment variables
export USE_GPU=true
export DEVICE_TYPE=npu:0
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=src/models
export VOICES_DIR=src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web
export ASCEND_RT_VISIBLE_DEVICES=0

# Run FastAPI with NPU extras using uv run
uv pip install -e ".[npu]"
uv run --no-sync python docker/scripts/download_model.py --output api/src/models/v1_0
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880

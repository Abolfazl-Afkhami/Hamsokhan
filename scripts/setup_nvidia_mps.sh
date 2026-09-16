#!/usr/bin/env bash
# Production Host Script: NVIDIA Multi-Process Service (MPS) Initialization
set -euo pipefail

echo "[MPS] Configuring GPU 0 into EXCLUSIVE_PROCESS compute mode..."
nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

echo "[MPS] Launching NVIDIA MPS control daemon..."
nvidia-cuda-mps-control -d

# Balance execution thread allocation: 50% vLLM, 50% STS Audio Pipeline
echo "set_default_active_thread_percentage 50" | nvidia-cuda-mps-control

echo "[MPS] Status check:"
echo "get_default_active_thread_percentage" | nvidia-cuda-mps-control
echo "[MPS] Initialized successfully. Ready for docker-compose launch."

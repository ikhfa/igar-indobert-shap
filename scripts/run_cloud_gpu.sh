#!/usr/bin/env bash
# =============================================================================
# Cloud GPU Setup & Run Script
# =============================================================================
# For vast.ai, Lambda Labs, RunPod, or any Ubuntu-based GPU VM.
#
# Usage:
#   # 1. SSH into your GPU instance
#   # 2. Run this script:
#   bash run_cloud_gpu.sh [--repo URL] [--dataset PATH] [--steps RANGE]
#
# Examples:
#   bash run_cloud_gpu.sh
#   bash run_cloud_gpu.sh --repo https://github.com/ikhfa/igar-indobert-shap.git
#   bash run_cloud_gpu.sh --dataset /workspace/Rating_labeled.csv --steps 1-3
#   bash run_cloud_gpu.sh --steps 3  # only IndoBERT training
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults (override via flags)
# ---------------------------------------------------------------------------
REPO_URL="${REPO_URL:-https://github.com/ikhfa/igar-indobert-shap.git}"
DATASET_PATH=""
STEPS=""
WORKDIR="/workspace/igar-indobert-shap"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo)      REPO_URL="$2";      shift 2 ;;
        --dataset)   DATASET_PATH="$2";   shift 2 ;;
        --steps)     STEPS="$2";          shift 2 ;;
        --workdir)   WORKDIR="$2";        shift 2 ;;
        -h|--help)
            echo "Usage: bash run_cloud_gpu.sh [--repo URL] [--dataset PATH] [--steps RANGE] [--workdir DIR]"
            echo ""
            echo "  --repo URL       Git repository URL (default: \$REPO_URL)"
            echo "  --dataset PATH   Path to Rating_labeled.csv on the VM"
            echo "  --steps RANGE    Pipeline steps to run: '1', '3-5', or omit for all"
            echo "  --workdir DIR    Working directory (default: /workspace/igar-indobert-shap)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ---------------------------------------------------------------------------
# 1. System check
# ---------------------------------------------------------------------------
log "=== Cloud GPU Experiment Runner ==="
log "Date: $(date)"
log "Host: $(hostname)"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    log "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    warn "No NVIDIA GPU detected. Training will be slow on CPU."
fi

# ---------------------------------------------------------------------------
# 2. Install system dependencies
# ---------------------------------------------------------------------------
log "Installing system packages..."
apt-get update -qq && apt-get install -y -qq git curl python3-pip python3-venv > /dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# 3. Clone repository
# ---------------------------------------------------------------------------
if [ -d "$WORKDIR/.git" ]; then
    log "Repository already exists at $WORKDIR, pulling latest..."
    cd "$WORKDIR"
    git pull
else
    log "Cloning $REPO_URL ..."
    git clone "$REPO_URL" "$WORKDIR"
    cd "$WORKDIR"
fi

log "Working directory: $(pwd)"

# ---------------------------------------------------------------------------
# 4. Set up Python environment
# ---------------------------------------------------------------------------
VENV_DIR="$WORKDIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
log "Python: $(python --version) at $(which python)"

log "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Verify torch sees GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ---------------------------------------------------------------------------
# 5. Dataset setup
# ---------------------------------------------------------------------------
if [ -n "$DATASET_PATH" ]; then
    log "Copying dataset from $DATASET_PATH ..."
    cp "$DATASET_PATH" "$WORKDIR/data/Rating_labeled.csv"
    # Switch off sample mode since real data is provided
    sed -i 's/^USE_SAMPLE: bool = True/USE_SAMPLE: bool = False/' config.py
    log "Switched to real dataset mode (USE_SAMPLE=False)"
elif [ -f "$WORKDIR/data/Rating_labeled.csv" ]; then
    log "Dataset found at data/Rating_labeled.csv"
    sed -i 's/^USE_SAMPLE: bool = True/USE_SAMPLE: bool = False/' config.py
    log "Switched to real dataset mode (USE_SAMPLE=False)"
else
    warn "No dataset provided. Running in synthetic sample mode (USE_SAMPLE=True)."
    warn "To use real data: bash run_cloud_gpu.sh --dataset /path/to/Rating_labeled.csv"
fi

# ---------------------------------------------------------------------------
# 6. Run pipeline
# ---------------------------------------------------------------------------
STEP_ARG=""
if [ -n "$STEPS" ]; then
    STEP_ARG="--step $STEPS"
fi

log "Starting pipeline... ${STEP_ARG:-'(all steps)'}"
log "Logging to: $WORKDIR/output/logs/"

# Run with unbuffered output for real-time logging
PYTHONUNBUFFERED=1 python run_pipeline.py $STEP_ARG 2>&1 | tee "$WORKDIR/output/logs/run_$(date +%Y%m%d_%H%M%S).log"

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
log "=== Run Complete ==="
log "Outputs saved to: $WORKDIR/output/"

echo ""
echo "Output structure:"
find "$WORKDIR/output" -type f | head -30 | sed "s|$WORKDIR/||"

echo ""
log "To download results:"
log "  scp -r <user>@<host>:$WORKDIR/output ./output_cloud/"
log "  # or tar + download:"
log "  tar czf results.tar.gz -C $WORKDIR output && scp <user>@<host>:$WORKDIR/results.tar.gz ."

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# mini_tts — Local environment setup
#
# Tested on: Ubuntu 22.04, macOS (Apple Silicon + Intel), WSL2
# NOT supported in Alpine Linux / musl libc
#
# Usage:
#   chmod +x setup_local.sh
#   ./setup_local.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "=== mini_tts setup ==="
echo ""

# ── 1. Python check ───────────────────────────────────────────────────────────
PYTHON=$(which python3 || which python)
PY_VER=$($PYTHON --version 2>&1)
echo "Python: $PY_VER"

if ! command -v $PYTHON &> /dev/null; then
    echo "ERROR: Python not found. Install Python 3.10+ first."
    exit 1
fi

# ── 2. Virtual environment ────────────────────────────────────────────────────
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
fi

source .venv/bin/activate
echo "Virtual env: $(which python)"

# ── 3. System deps (Ubuntu/Debian) ───────────────────────────────────────────
if command -v apt-get &> /dev/null; then
    echo ""
    echo "Installing system dependencies (requires sudo)..."
    sudo apt-get install -y -q \
        espeak-ng \
        libsndfile1 \
        libsndfile1-dev \
        ffmpeg \
        2>/dev/null || echo "Warning: some system deps may be missing"
elif command -v brew &> /dev/null; then
    echo "Homebrew detected. Installing system deps..."
    brew install espeak libsndfile ffmpeg 2>/dev/null || true
fi

# ── 4. Python packages ────────────────────────────────────────────────────────
echo ""
echo "Installing Python packages..."
pip install --upgrade pip -q

# PyTorch CPU (change 'cpu' to 'cu121' etc. for GPU)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu -q

# Rest of requirements
pip install -r requirements.txt -q

# ── 5. Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "Verifying installation..."
python -c "
import torch, torchaudio, numpy, scipy, soundfile
print(f'  torch        : {torch.__version__}')
print(f'  torchaudio   : {torchaudio.__version__}')
print(f'  numpy        : {numpy.__version__}')
print(f'  scipy        : {scipy.__version__}')
print(f'  soundfile    : OK')
try:
    import phonemizer
    print(f'  phonemizer   : {phonemizer.__version__}')
except:
    print(f'  phonemizer   : not installed (needed for Phase 2)')
try:
    import speechbrain
    print(f'  speechbrain  : {speechbrain.__version__}')
except:
    print(f'  speechbrain  : not installed (needed for Phase 3)')
"

# ── 6. Phase 1 quick test ─────────────────────────────────────────────────────
echo ""
echo "Running Phase 1 smoke test..."
python -c "
import sys; sys.path.insert(0, '.')
from models.hifigan import HiFiGANGenerator, HiFiGANConfig
from models.mel import MelSpec
import torch

cfg = HiFiGANConfig()
gen = HiFiGANGenerator(cfg)
mel = torch.randn(1, 80, 100)
audio = gen(mel)
print(f'  mel {mel.shape} → audio {audio.shape}  ✓')
print(f'  Generator params: {gen.param_count:,}')
"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  1. Download HiFi-GAN weights:  python tools/download_hifigan.py"
echo "  2. Run benchmark:              python tools/benchmark_vocoder.py"
echo "  3. Start Phase 2 when RTF < 1.0"

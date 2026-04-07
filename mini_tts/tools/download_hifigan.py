"""
Download pretrained HiFi-GAN checkpoint for fast Phase 1 bootstrap.

Uses the official Jungil Kong HiFi-GAN repo weights (V2 config, 14M params).
These are the best publicly available CPU-fast HiFi-GAN weights.

Usage:
    python tools/download_hifigan.py
    python tools/download_hifigan.py --variant v1   # bigger, higher quality
    python tools/download_hifigan.py --variant v3   # smallest, fastest

Saves to: checkpoints/hifigan_{variant}.pt
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ── Checkpoint sources ────────────────────────────────────────────────────────
# These are the official HiFi-GAN releases from the original authors.
# V2 is the sweet spot for our use case (small + fast + good quality).

CHECKPOINTS = {
    "v1": {
        "url": "https://drive.google.com/uc?id=14NenuYVh9B6D70bVAHBCGMQDEPR5BCMV",
        "description": "HiFi-GAN V1 — highest quality, ~14M params",
        "config": {
            "upsample_rates": [8, 8, 2, 2],
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
        }
    },
    "v2": {
        "url": "https://drive.google.com/uc?id=1zeSofa0OX0vZuMhFuuMMT4DZsBlKCZpD",
        "description": "HiFi-GAN V2 — balanced speed/quality, ~14M params ← recommended",
        "config": {
            "upsample_rates": [8, 8, 4],
            "upsample_kernel_sizes": [16, 16, 8],
            "upsample_initial_channel": 256,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1,3,5],[1,3,5],[1,3,5]],
        }
    },
    "v3": {
        "url": "https://drive.google.com/uc?id=1-eEYTB5O4yPwqPjF_e7hROVW2O6bVR0Q",
        "description": "HiFi-GAN V3 — fastest, smallest, ~1.5M params",
        "config": {
            "upsample_rates": [8, 8, 4],
            "upsample_kernel_sizes": [16, 16, 8],
            "upsample_initial_channel": 128,
            "resblock_kernel_sizes": [3, 5, 7],
            "resblock_dilation_sizes": [[1,2],[2,6],[3,12]],
        }
    },
}

# Alternative: gdown-free download from HuggingFace
HUGGINGFACE_ALTERNATIVES = {
    "v2": "https://huggingface.co/rhasspy/piper-voices/resolve/main/vocoder/hifigan/v2/generator",
    # More available on HuggingFace — search "hifigan" for options
}


def download_with_gdown(url: str, output: Path):
    """Download from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
        import gdown
    gdown.download(url, str(output), quiet=False)


def download_with_wget(url: str, output: Path):
    """Fallback: direct HTTP download."""
    import urllib.request
    print(f"Downloading {url} → {output}")
    urllib.request.urlretrieve(url, str(output))


def main():
    parser = argparse.ArgumentParser(description="Download pretrained HiFi-GAN")
    parser.add_argument("--variant", choices=["v1", "v2", "v3"], default="v2")
    args = parser.parse_args()

    variant = args.variant
    info = CHECKPOINTS[variant]
    print(f"\nDownloading {info['description']}")

    out_dir = Path(__file__).parent.parent / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"hifigan_{variant}.pt"

    if out_path.exists():
        print(f"✓ Already exists: {out_path}")
        return

    # Try HuggingFace first (no Google auth required)
    if variant in HUGGINGFACE_ALTERNATIVES:
        try:
            download_with_wget(HUGGINGFACE_ALTERNATIVES[variant], out_path)
            print(f"✓ Downloaded to {out_path}")
            return
        except Exception as e:
            print(f"HuggingFace download failed: {e}, trying Google Drive...")

    download_with_gdown(info["url"], out_path)
    print(f"✓ Downloaded to {out_path}")
    print(f"\nTest it with:")
    print(f"  python inference/vocoder.py --checkpoint {out_path}")


if __name__ == "__main__":
    main()

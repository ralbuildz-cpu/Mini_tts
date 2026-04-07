# mini_tts

A lightweight, CPU-real-time text-to-speech system with offline speaker cloning.

---

## Design Goals

| Property | Target |
|---|---|
| Execution | CPU-only |
| Inference | Real-time (RTF < 1.0) |
| First audio latency | < 300 ms |
| Active params | ~40–70M total |
| Speaker cloning | Offline (no runtime audio reference) |
| Architecture | Non-autoregressive, parallel |

---

## Architecture Overview

```
Text
 │
 ▼
[G2P / Phoneme Lookup]
 │   models/phoneme_vocab.py
 ▼
[Acoustic Model]  ←── Speaker Embedding (cached)
 │   models/acoustic_model.py     ←── Prosody (duration, pitch, energy)
 │   ~34.5M params
 ▼  mel chunks [B, n_mels, chunk_frames]
[HiFi-GAN Vocoder]
 │   models/hifigan.py
 │   inference/vocoder.py
 ▼
Audio (streaming PCM)
```

**End-to-end pipeline:** `inference/pipeline.py`

---

## Repository Layout

```
mini_tts/
├── models/
│   ├── audio_config.py       Single source of truth for all audio constants
│   ├── phoneme_vocab.py      136-token IPA/ARPAbet vocabulary + O(1) lookups
│   ├── acoustic_model.py     FastSpeech2-style acoustic model (~34.5M params)
│   ├── hifigan.py            HiFi-GAN generator + discriminators
│   └── mel.py                Mel spectrogram extraction (librosa / numpy fallback)
├── inference/
│   ├── pipeline.py           Streaming end-to-end TTS (acoustic → vocoder)
│   └── vocoder.py            Hardened vocoder with stateful chunked streaming
├── training/
│   ├── dataset.py            TTSDataset + collate_fn
│   └── acoustic_trainer.py  Phase 2A / 2B / 2C training curriculum
├── tools/
│   ├── download_hifigan.py   Downloads pretrained HiFi-GAN weights
│   ├── benchmark_vocoder.py  Phase 1 vocoder RTF + streaming benchmark
│   └── benchmark_acoustic.py Phase 2 end-to-end RTF + streaming benchmark
├── requirements.txt
├── setup_local.sh
└── README.md                 (this file)
```

---

## Quick Start

### 1. Setup

```bash
chmod +x setup_local.sh
./setup_local.sh
```

Requires Python 3.9+ on Ubuntu / macOS / WSL2 (glibc). **Not compatible with Alpine Linux / musl libc.**

### 2. Download pretrained HiFi-GAN weights

```bash
python tools/download_hifigan.py
```

Downloads V1, V2, and V3 checkpoints to `checkpoints/hifigan/`.

### 3. Benchmark vocoder (Phase 1 exit gate)

```bash
python tools/benchmark_vocoder.py                  # RTF table across utterance lengths
python tools/benchmark_vocoder.py --streaming      # Per-chunk latency + memory growth
```

### 4. Benchmark acoustic model (Phase 2 exit gate)

```bash
python tools/benchmark_acoustic.py --acoustic-only   # Acoustic model alone (no vocoder needed)
python tools/benchmark_acoustic.py                   # Full E2E pipeline
python tools/benchmark_acoustic.py --streaming       # Streaming detail
```

---

## Performance Targets

| Check | Target |
|---|---|
| First-chunk latency | < 300 ms |
| Mean RTF (E2E) | < 1.0× |
| Max RTF | < 1.0× |
| Latency std dev | < 20% of mean |
| Memory growth | < 5 MB over session |

---

## Model Details

### Acoustic Model (`models/acoustic_model.py`)

| Component | Params |
|---|---|
| Phoneme embedding (136 → 512) | ~70K |
| Encoder: 4 × FFT block (512d, FFN 2048, 4 heads) | ~12.6M |
| Decoder: 4 × FFT block (512d, FFN 2048, 4 heads) | ~12.6M |
| Postnet: 5 × Conv1d-512 | ~6.5M |
| Duration predictor: 2 × Conv + linear | ~1.6M |
| FiLM conditioning (speaker 256d → 512) | ~0.5M |
| Prosody predictors (pitch + energy, MLP) | ~1.2M |
| Projections + misc | ~0.5M |
| **Total** | **~34.5M** |

**Key design decisions:**
- FiLM conditioning injected at mid-decoder only (not every layer) — keeps conditioning stack tiny
- Duration predictor drives length regulator — no attention over reference audio
- Streaming via fixed-size chunk emission with crossfade carryover buffer
- All ops are Conv1d / Linear / LayerNorm — INT8 quantization friendly

### Audio Config (`models/audio_config.py`)

All modules import constants from here. **Never hardcode audio constants elsewhere.**

| Constant | Default |
|---|---|
| `sample_rate` | 22050 |
| `hop_length` | 256 |
| `n_mels` | 80 |
| `n_fft` | 1024 |
| `chunk_ms` | 300 |
| `overlap_ms` | 30 |

Derived: `chunk_frames = ceil(chunk_ms * sample_rate / (1000 * hop_length))`

### Phoneme Vocab (`models/phoneme_vocab.py`)

- 136 tokens: IPA + ARPAbet + special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`, `<sil>`)
- `PhonemeVocab.encode(text)` — string → integer IDs
- `PhonemeVocab.decode(ids)` — integer IDs → string
- O(1) lookup both directions

---

## Training

### Data Format

Each sample needs:
- Phoneme ID sequence (`.npy` or `.txt`)
- Mel spectrogram (`.npy`, shape `[n_mels, T]`, computed with `models/mel.py`)
- Optional: pitch sequence, energy sequence (for Phase 2C)
- Optional: speaker ID or speaker embedding (for Phase 2B+)

### Training Phases

**Phase 2A — Single speaker, mel-only loss**
```bash
python training/acoustic_trainer.py \
    --data-dir /path/to/dataset \
    --phase 2A \
    --epochs 100
```

**Phase 2B — Multi-speaker, add speaker embedding**
```bash
python training/acoustic_trainer.py \
    --data-dir /path/to/dataset \
    --phase 2B \
    --speaker-dim 256 \
    --epochs 100
```

**Phase 2C — Full prosody (pitch + energy)**
```bash
python training/acoustic_trainer.py \
    --data-dir /path/to/dataset \
    --phase 2C \
    --epochs 100
```

### Loss Weights by Phase

| Phase | w_mel | w_dur | w_pit | w_nrg |
|---|---|---|---|---|
| 2A | 1.0 | 0.1 | 0.0 | 0.0 |
| 2B | 1.0 | 0.1 | 0.0 | 0.0 |
| 2C | 1.0 | 0.1 | 0.01 | 0.01 |

---

## Streaming Interface Contract

All models implement `AcousticModelBase`:

```python
class AcousticModelBase:
    def infer(self, phoneme_ids, speaker_emb=None) -> torch.Tensor:
        """Full utterance inference. Returns [B, n_mels, T]."""

    def infer_chunk(self, phoneme_chunk, speaker_emb=None) -> torch.Tensor:
        """Streaming inference. Returns [B, n_mels, chunk_frames]."""

    def flush_stream(self) -> Optional[torch.Tensor]:
        """Flush any remaining frames after last chunk."""

    def reset_stream(self):
        """Reset all streaming state."""
```

The vocoder (`inference/vocoder.py`) implements `VocoderBase` with an identical `infer_chunk` / `reset_stream` interface.

**Rule:** the acoustic model's output chunk shape `[B, n_mels, chunk_frames]` must match `audio_config.chunk_frames` — the vocoder depends on this.

---

## Constraints (do not violate)

- ❌ No autoregressive decoding
- ❌ No attention over raw audio
- ❌ No full-sequence buffering in streaming mode
- ❌ No large conditioning stacks (FiLM at 1–2 layers only)
- ❌ No deep transformer stacks (4–6 layers max)
- ✅ All ops must be parallelizable across time
- ✅ Design all ops for INT8 quantization (Conv1d, Linear, LayerNorm only — avoid custom CUDA kernels)

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 0 | ✅ Done | Repo scaffold, requirements, setup |
| 1 | ✅ Done | HiFi-GAN vocoder code |
| 1.5 | ✅ Done | Vocoder hardening, streaming contract, benchmarks |
| 2 | ✅ Done | Acoustic model, training pipeline, E2E benchmark |
| 3 | 🔲 Next | Speaker encoder (ECAPA-TDNN ~10M, offline cloning) |
| 4 | 🔲 | Prosody refinement, duration alignment |
| 5 | 🔲 | INT8 quantization, ONNX export |
| 6 | 🔲 | Full system integration test + final benchmarks |

---

## Version History

See `CHANGELOG.md` for full version history.

---

## License

MIT

# mini_tts — Changelog

All notable changes to this project are documented here.
Format: `[vX.Y] — Phase Name — Date`

---

## [v0.1] — Phase 0: Repo Scaffold

**Purpose:** Establish the project skeleton and local environment.

### Added
- `requirements.txt` — pinned dependencies: torch, torchaudio, numpy, librosa, soundfile, tqdm, tensorboard
- `setup_local.sh` — one-shot local environment setup (venv creation, pip install, directory creation)
- Initial `README.md`
- `models/`, `inference/`, `training/`, `tools/` package directories with `__init__.py`

### Notes for Implementers
- **glibc only.** PyTorch wheels require glibc. Alpine Linux / musl libc are incompatible. Use Ubuntu 20.04+, macOS 12+, or WSL2.
- All future phases assume this setup has been run successfully.

---

## [v0.2] — Phase 1: HiFi-GAN Vocoder

**Purpose:** Implement the neural vocoder that converts mel spectrograms → audio.

### Added
- `models/audio_config.py` — **Single source of truth** for all audio constants. Every module imports from here. Never hardcode `sample_rate`, `hop_length`, `n_mels`, etc. elsewhere.
  - `AudioConfig` dataclass with: `sample_rate=22050`, `hop_length=256`, `n_mels=80`, `n_fft=1024`, `chunk_ms=300`, `overlap_ms=30`
  - Derived property: `chunk_frames = ceil(chunk_ms * sample_rate / (1000 * hop_length))`
- `models/hifigan.py` — Full HiFi-GAN generator + multi-period discriminator (MPD) + multi-scale discriminator (MSD)
  - Small config targeting V2/V3 parameter count
  - Imports audio constants from `audio_config.py`
- `models/mel.py` — Mel spectrogram extraction
  - Primary: librosa-based (accurate)
  - Fallback: pure numpy (no librosa dependency for inference-only deployments)
- `tools/download_hifigan.py` — Downloads pretrained V1 / V2 / V3 HiFi-GAN checkpoints

### Notes for Implementers
- HiFi-GAN V2 is the recommended default: best CPU RTF vs quality tradeoff.
- V1 has highest quality but ~2× slower on CPU.
- V3 is fastest but noticeable quality drop at low bitrates.
- Pretrained weights are downloaded to `checkpoints/hifigan/` and are not committed to the repo.

---

## [v0.3] — Phase 1.5: Vocoder Hardening + Streaming Contract

**Purpose:** Lock the vocoder interface before building the acoustic model on top of it. Ensures future model swaps don't break streaming.

### Added
- `inference/vocoder.py` — Complete rewrite with:
  - `VocoderBase` ABC defining the interface contract:
    - `infer(mel)` — full utterance
    - `infer_chunk(mel_chunk)` — stateful streaming
    - `reset_stream()` — clear stream state
  - `HiFiGANVocoder` — production implementation
    - Stateful chunked streaming with **linear crossfade** between chunks (eliminates click artifacts at chunk boundaries)
    - Carryover buffer: last `overlap_frames` of audio are blended into the next chunk
    - `Vocoder = HiFiGANVocoder` alias for drop-in use
- `tools/benchmark_vocoder.py` — Two benchmark modes:
  - Standard: RTF table across utterance lengths (1s, 3s, 10s, 30s), first-chunk latency
  - `--streaming --chunk-ms 300`: per-chunk timing, latency histogram, memory growth check

### Interface Contract (frozen)
```python
class VocoderBase:
    def infer(self, mel: Tensor) -> Tensor: ...
    def infer_chunk(self, mel_chunk: Tensor) -> Tensor: ...
    def reset_stream(self): ...
```
**Do not change this contract.** The acoustic model streaming pipeline depends on it.

### Phase 1.5 Exit Criteria (run locally)
```bash
python tools/benchmark_vocoder.py
python tools/benchmark_vocoder.py --streaming
```
| Metric | Target |
|---|---|
| First-chunk latency | < 300 ms |
| Mean RTF | < 1.0× |
| Max RTF | < 1.0× |
| Latency std dev | < 20% of mean |
| Memory growth | < 5 MB |

### Notes for Implementers
- The crossfade overlap length is controlled by `audio_config.overlap_ms`. Do not change `overlap_ms` without re-benchmarking — too large → CPU budget overrun; too small → audible clicks.
- `reset_stream()` must be called between utterances or the carryover buffer from the previous utterance bleeds into the next.
- Memory growth check catches buffer leaks that only appear after many chunks (common in naive streaming implementations).

---

## [v0.4] — Phase 2: Acoustic Model + Training Pipeline

**Purpose:** Build the non-autoregressive acoustic model that converts phoneme IDs → mel chunks, plus a full training curriculum.

### Added

#### `models/phoneme_vocab.py`
- 136-token vocabulary covering IPA + ARPAbet + special tokens: `<pad>`, `<unk>`, `<bos>`, `<eos>`, `<sil>`
- `PhonemeVocab` class: O(1) encode/decode, both directions
- `PAD_ID = 0` constant used by collate functions

#### `models/acoustic_model.py`
Full FastSpeech2-style acoustic model, ~34.5M parameters.

| Component | Description |
|---|---|
| `FFTBlock` | Single Feed-Forward Transformer block (self-attention + conv FFN + LayerNorm). Used in encoder and decoder. |
| `DurationPredictor` | 2 × Conv1d + ReLU + Linear → scalar duration per phoneme. Trained with MSE loss on log-durations. |
| `LengthRegulator` | Expands phoneme-level hidden states to frame-level using predicted durations. Streaming-safe: tracks partial durations across chunks. |
| `ProsodyPredictor` | Predicts per-frame pitch and energy via small MLP (2 linear layers). Auxiliary loss only in Phase 2C. |
| `FiLMConditioner` | Speaker embedding → (γ, β) scale/shift pair. Applied at mid-decoder only. Keeps conditioning stack tiny. |
| `AcousticModelBase` | ABC defining the interface: `infer()`, `infer_chunk()`, `flush_stream()`, `reset_stream()` |
| `MiniAcousticModel` | Full model assembling all components. |
| `Postnet` | 5 × Conv1d residual refinement after decoder projection. |

**Architecture config (default):**
```python
hidden_dim = 512
encoder_layers = 4
decoder_layers = 4
n_heads = 4
ffn_dim = 2048
speaker_dim = 256
vocab_size = 136   # from PhonemeVocab
n_mels = 80        # from AudioConfig
```

#### `inference/pipeline.py`
- `StreamingTTSPipeline` — wires `MiniAcousticModel` → `HiFiGANVocoder` end-to-end
- Emits audio chunks as numpy arrays, ready for playback or file write
- Handles `reset_stream()` on both models between utterances
- Exposes `synthesize(text)` (full) and `synthesize_stream(text)` (generator, yields audio chunks)

#### `training/dataset.py`
- `TTSDataset` — loads `(phoneme_ids, mel, [pitch], [energy], [speaker_id])` from disk
- `collate_fn` — dynamic padding, returns batch tensors
- Expects dataset directory with `.npy` mel files and corresponding phoneme `.txt` files

#### `training/acoustic_trainer.py`
- `AcousticTrainer` — handles Phase 2A / 2B / 2C curriculum
- Loss composition per phase:

| Phase | w_mel | w_dur | w_pitch | w_energy |
|---|---|---|---|---|
| 2A | 1.0 | 0.1 | 0.0 | 0.0 |
| 2B | 1.0 | 0.1 | 0.0 | 0.0 |
| 2C | 1.0 | 0.1 | 0.01 | 0.01 |

- Checkpoints saved to `checkpoints/acoustic/`
- TensorBoard logging: mel loss, duration loss, pitch loss, energy loss, learning rate

#### `tools/benchmark_acoustic.py`
- `--acoustic-only`: benchmark acoustic model alone (random weights, no vocoder/audio needed)
- Default: full E2E pipeline benchmark
- `--streaming`: per-chunk timing, latency histogram, memory growth check
- Reports RTF for acoustic model alone, vocoder alone, and combined

### Streaming Interface Contract (acoustic model)
```python
class AcousticModelBase:
    def infer(self, phoneme_ids, speaker_emb=None) -> Tensor:          # [B, n_mels, T]
    def infer_chunk(self, phoneme_chunk, speaker_emb=None) -> Tensor:  # [B, n_mels, chunk_frames]
    def flush_stream(self) -> Optional[Tensor]:                        # flush carryover
    def reset_stream(self): ...
```
**Output shape:** `[B, n_mels, chunk_frames]` where `chunk_frames = AudioConfig().chunk_frames`. Must match exactly — vocoder depends on this.

### Phase 2 Exit Criteria (run locally)
```bash
python tools/benchmark_acoustic.py --acoustic-only    # model alone
python tools/benchmark_acoustic.py                    # E2E
python tools/benchmark_acoustic.py --streaming        # streaming
```
| Metric | Target |
|---|---|
| Intelligible speech | ✓ (subjective, after training) |
| Streams without gaps | ✓ |
| End-to-end RTF | < 1.0× |
| First audio | < 300 ms |

### Notes for Implementers
- **Phase 2A first.** Train single-speaker with no speaker embedding. Validate intelligibility before introducing speaker conditioning. Skipping this step makes debugging much harder.
- FiLM is injected at **mid-decoder only** (not every layer). This is intentional — injecting at every layer causes training instability at small scales and wastes CPU budget.
- `DurationPredictor` is trained on log-durations to prevent the model from predicting negative or zero durations. Take `exp()` at inference time.
- The `LengthRegulator` must handle durations that are not integer-aligned. Fractional frames are rounded, and the regulator tracks residual for the next chunk in streaming mode.
- `Postnet` is a residual add — the main decoder output is kept and the postnet delta is added. This means postnet failure degrades gracefully (doesn't corrupt mel completely).
- For INT8 quantization (Phase 5): all ops are Conv1d, Linear, LayerNorm — standard PyTorch `torch.quantization.quantize_dynamic` works without custom kernels.

---

## [Planned] v0.5 — Phase 3: Speaker Encoder + Offline Cloning

**Purpose:** Add offline speaker embedding extraction so the system can clone a voice without referencing audio at runtime.

### Planned
- `models/speaker_encoder.py` — SpeechBrain ECAPA-TDNN wrapper (~10M params)
  - Input: reference audio waveform (offline only, not at inference time)
  - Output: 256-d speaker embedding
- `tools/extract_speaker_embedding.py` — CLI to extract + cache speaker embeddings to `.npy`
- Update `inference/pipeline.py` to load cached speaker embedding from disk

### Constraints
- Speaker embedding extraction happens **offline, before inference**. Never extract at runtime.
- Embeddings are cached as `.npy` files, loaded at pipeline startup.
- The 256-d embedding shape is fixed — changing it breaks the FiLM conditioner.

---

## [Planned] v0.6 — Phase 4: Prosody Refinement

**Purpose:** Improve naturalness via better duration modeling and pitch/energy expressiveness.

### Planned
- Duration alignment using Montreal Forced Aligner (MFA) or CTC-based aligner
- Pitch extraction: REAPER or CREPE (offline, baked into dataset)
- Energy extraction: RMS per frame (cheap)
- Prosody transfer: allow scaling pitch/energy at inference for expressiveness control

---

## [Planned] v0.7 — Phase 5: Quantization + ONNX Export

**Purpose:** Maximize CPU performance for deployment.

### Planned
- INT8 dynamic quantization via `torch.quantization.quantize_dynamic`
- ONNX export for both acoustic model and vocoder
- ONNX Runtime inference wrapper (replaces PyTorch at inference time)
- Final RTF benchmark: target < 0.5× on modern x86 CPU

---

## [Planned] v1.0 — Phase 6: Full System Integration

**Purpose:** Production-ready system.

### Planned
- End-to-end smoke test: text → wav file
- Docker container (Ubuntu base, no GPU)
- Final documentation pass

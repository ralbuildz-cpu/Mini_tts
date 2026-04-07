"""
Vocoder inference — Phase 1.5 hardened.

Interface contract (stable across model swaps):

    class VocoderBase:
        infer(mel)            → full sequence, returns numpy audio
        infer_chunk(mel)      → single chunk, maintains crossfade state
        reset_stream()        → clear streaming state (call between utterances)

Streaming design:
  - No full-sequence buffering — each infer_chunk() processes only its input
  - Seamless audio via linear crossfade at chunk boundaries
  - State = only the tail buffer (~20ms of audio), NOT the mel sequence
  - Stateless between utterances via reset_stream()

Usage:
    vocoder = HiFiGANVocoder.from_config()

    # Full sequence (non-streaming)
    audio = vocoder.infer(mel)

    # Streaming — caller feeds chunks one at a time
    vocoder.reset_stream()
    for mel_chunk in mel_chunks:
        audio_chunk = vocoder.infer_chunk(mel_chunk)
        play(audio_chunk)
"""

import time
import abc
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.audio_config import AudioConfig, DEFAULT_AUDIO_CONFIG
from models.hifigan import HiFiGANGenerator, HiFiGANConfig


# ─── Interface contract ────────────────────────────────────────────────────────

class VocoderBase(abc.ABC):
    """
    Stable interface for all vocoder backends.

    Subclass this when swapping in a new vocoder (EnCodec, WaveRNN, etc.)
    All Phase 2+ code talks to this interface only — never to HiFiGAN directly.
    """

    @property
    @abc.abstractmethod
    def audio_config(self) -> AudioConfig:
        """Return the AudioConfig this vocoder was built with."""
        ...

    @abc.abstractmethod
    @torch.no_grad()
    def infer(self, mel: torch.Tensor) -> np.ndarray:
        """
        Full-sequence inference.

        Args:
            mel: (n_mels, T) or (1, n_mels, T)
        Returns:
            audio: float32 numpy (T_audio,)
        """
        ...

    @abc.abstractmethod
    @torch.no_grad()
    def infer_chunk(self, mel_chunk: torch.Tensor) -> np.ndarray:
        """
        Single-chunk streaming inference.

        Processes only mel_chunk — does NOT buffer the full sequence.
        Maintains a small crossfade state (~20ms) to avoid pops between chunks.
        Call reset_stream() between utterances.

        Args:
            mel_chunk: (n_mels, T_chunk) or (1, n_mels, T_chunk)
        Returns:
            audio: float32 numpy (T_audio_chunk,)
                   Approximately T_chunk * hop_length samples,
                   minus the overlap consumed by the crossfade.
        """
        ...

    @abc.abstractmethod
    def reset_stream(self) -> None:
        """
        Reset streaming state.

        Call this:
          - Before starting a new utterance
          - After a silence/break longer than ~100ms
          - On any error during streaming

        No-op if stream has never been started.
        """
        ...

    # ── Convenience methods (not abstract) ────────────────────────────────────

    @torch.no_grad()
    def infer_timed(self, mel: torch.Tensor) -> Tuple[np.ndarray, dict]:
        """
        infer() with timing stats for benchmarking.

        Returns:
            (audio, stats) where stats contains:
                rtf              : real-time factor (< 1.0 = real-time)
                latency_ms       : wall-clock inference time
                audio_duration_s : duration of generated audio
                realtime         : bool
        """
        mel_prepared = self._prepare_mel(mel)

        # Warmup: one pass on a tiny tensor to trigger JIT/cache
        _ = self._run_generator(mel_prepared[:, :, :min(10, mel_prepared.shape[-1])])

        t0 = time.perf_counter()
        audio = self.infer(mel)
        elapsed = time.perf_counter() - t0

        audio_duration = len(audio) / self.audio_config.sample_rate
        rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")

        return audio, {
            "rtf": rtf,
            "latency_ms": elapsed * 1000,
            "audio_duration_s": audio_duration,
            "mel_frames": mel_prepared.shape[-1],
            "realtime": rtf < 1.0,
        }

    def _prepare_mel(self, mel: torch.Tensor) -> torch.Tensor:
        """Normalize mel to (1, n_mels, T)."""
        if mel.ndim == 1:
            raise ValueError("mel must be at least 2D (n_mels, T)")
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        if mel.ndim != 3:
            raise ValueError(f"Expected 2D or 3D mel, got {mel.ndim}D")
        return mel.float()

    def _run_generator(self, mel: torch.Tensor) -> torch.Tensor:
        """Subclasses override this to call their specific generator."""
        raise NotImplementedError


# ─── HiFi-GAN implementation ───────────────────────────────────────────────────

class HiFiGANVocoder(VocoderBase):
    """
    HiFi-GAN backed vocoder.

    Phase 1 reference implementation of VocoderBase.
    """

    def __init__(
        self,
        generator: HiFiGANGenerator,
        hifigan_config: HiFiGANConfig,
        audio_cfg: AudioConfig = None,
        device: str = "cpu",
    ):
        self._audio_config = audio_cfg or DEFAULT_AUDIO_CONFIG
        self._device = torch.device(device)
        self._generator = generator.to(self._device).eval()
        self._hifigan_config = hifigan_config

        # Strip weight norm once — required for clean ONNX export later
        try:
            self._generator.remove_weight_norm()
        except Exception:
            pass

        # Streaming state
        self._stream_tail: Optional[np.ndarray] = None  # last overlap_samples of prev chunk

    # ── VocoderBase: properties ────────────────────────────────────────────────

    @property
    def audio_config(self) -> AudioConfig:
        return self._audio_config

    # ── VocoderBase: constructors ─────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        hifigan_config: HiFiGANConfig = None,
        audio_cfg: AudioConfig = None,
        device: str = "cpu",
    ) -> "HiFiGANVocoder":
        """Create a fresh (untrained) vocoder — for architecture testing."""
        if hifigan_config is None:
            hifigan_config = HiFiGANConfig()
        gen = HiFiGANGenerator(hifigan_config)
        return cls(gen, hifigan_config, audio_cfg, device)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        audio_cfg: AudioConfig = None,
        device: str = "cpu",
    ) -> "HiFiGANVocoder":
        """
        Load from checkpoint. Supports:
          1. mini_tts native format (saved by training/train_vocoder.py)
          2. Official HiFi-GAN repo format (raw state dict)
          3. PyTorch Lightning format (state_dict key)
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                "Download weights with:\n"
                "  python tools/download_hifigan.py"
            )

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        if "generator" in ckpt:
            config_dict = ckpt.get("config", {})
            hifigan_config = HiFiGANConfig()
            for k, v in config_dict.items():
                if hasattr(hifigan_config, k):
                    setattr(hifigan_config, k, v)
            gen = HiFiGANGenerator(hifigan_config)
            gen.load_state_dict(ckpt["generator"])
        elif "state_dict" in ckpt:
            hifigan_config = HiFiGANConfig()
            gen = HiFiGANGenerator(hifigan_config)
            gen.load_state_dict(ckpt["state_dict"])
        else:
            hifigan_config = HiFiGANConfig()
            gen = HiFiGANGenerator(hifigan_config)
            gen.load_state_dict(ckpt)

        print(f"✓ Loaded HiFiGANVocoder from {path} ({gen.param_count:,} params)")
        return cls(gen, hifigan_config, audio_cfg, device)

    # ── VocoderBase: inference ─────────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, mel: torch.Tensor) -> np.ndarray:
        """Full-sequence inference. Does not affect streaming state."""
        mel = self._prepare_mel(mel).to(self._device)
        audio = self._generator(mel)
        return audio.squeeze().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def infer_chunk(self, mel_chunk: torch.Tensor) -> np.ndarray:
        """
        True streaming inference — processes only the given chunk.

        No full-sequence buffering. State = only a crossfade tail buffer.

        Crossfade algorithm:
          1. Run generator on mel_chunk → raw_audio
          2. If _stream_tail exists:
             - Linear fade out over tail, fade in over head of raw_audio
             - Merge in overlap region
          3. Store last overlap_samples of raw_audio as new tail
          4. Return stitched audio (minus overlap consumed by fade-in)
        """
        mel_chunk = self._prepare_mel(mel_chunk).to(self._device)
        raw_audio = self._generator(mel_chunk).squeeze().cpu().numpy().astype(np.float32)

        overlap_n = self._audio_config.overlap_samples

        if self._stream_tail is None:
            # First chunk — no crossfade, just save the tail
            if len(raw_audio) > overlap_n:
                self._stream_tail = raw_audio[-overlap_n:].copy()
                return raw_audio[:-overlap_n]
            else:
                self._stream_tail = raw_audio.copy()
                return np.array([], dtype=np.float32)

        # Subsequent chunks — crossfade
        tail = self._stream_tail
        n = min(overlap_n, len(tail), len(raw_audio))

        # Linear fade weights
        fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
        fade_in  = np.linspace(0.0, 1.0, n, dtype=np.float32)

        blended = tail[-n:] * fade_out + raw_audio[:n] * fade_in

        # Update tail for next chunk
        if len(raw_audio) > overlap_n:
            self._stream_tail = raw_audio[-overlap_n:].copy()
            output_body = raw_audio[n:-overlap_n]
        else:
            self._stream_tail = raw_audio.copy()
            output_body = np.array([], dtype=np.float32)

        return np.concatenate([blended, output_body]).astype(np.float32)

    def reset_stream(self) -> None:
        """Clear crossfade buffer. Call before each new utterance."""
        self._stream_tail = None

    # ── Internal ───────────────────────────────────────────────────────────────

    def _run_generator(self, mel: torch.Tensor) -> torch.Tensor:
        return self._generator(mel.to(self._device))

    def save_audio(self, audio: np.ndarray, path: str) -> None:
        """Save numpy audio to WAV. Requires soundfile."""
        import soundfile as sf
        sr = self._audio_config.sample_rate
        sf.write(path, audio, sr)
        print(f"✓ Saved {len(audio)/sr:.2f}s → {path}")

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self._generator.parameters())


# ── Backwards-compat alias ─────────────────────────────────────────────────────
# Code written against the old Vocoder class still works.
Vocoder = HiFiGANVocoder


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Phase 1.5 vocoder interface...")

    cfg = AudioConfig()
    vocoder = HiFiGANVocoder.from_config(audio_cfg=cfg)

    # Full sequence
    mel = torch.randn(1, cfg.n_mels, cfg.frames_for_duration(3.0))
    audio, stats = vocoder.infer_timed(mel)

    print(f"\n── Full sequence ─────────────────────────────")
    print(f"  mel frames     : {mel.shape[-1]}")
    print(f"  audio duration : {stats['audio_duration_s']:.2f}s")
    print(f"  latency        : {stats['latency_ms']:.1f}ms")
    print(f"  RTF            : {stats['rtf']:.4f}  {'✓ REALTIME' if stats['realtime'] else '✗ SLOW'}")

    # Streaming
    print(f"\n── Streaming ({cfg.chunk_ms}ms chunks) ─────────────────")
    vocoder.reset_stream()
    mel_full = torch.randn(1, cfg.n_mels, cfg.frames_for_duration(3.0))
    T = mel_full.shape[-1]
    step = cfg.chunk_frames

    chunks_out = []
    chunk_times = []
    for start in range(0, T, step):
        chunk = mel_full[:, :, start:start + step]
        t0 = time.perf_counter()
        out = vocoder.infer_chunk(chunk)
        chunk_times.append((time.perf_counter() - t0) * 1000)
        chunks_out.append(out)

    total_audio = sum(len(c) for c in chunks_out)
    print(f"  chunks         : {len(chunks_out)}")
    print(f"  total samples  : {total_audio}")
    print(f"  first chunk    : {chunk_times[0]:.1f}ms")
    print(f"  avg chunk      : {np.mean(chunk_times):.1f}ms")
    print(f"  target <300ms  : {'✓' if chunk_times[0] < 300 else '✗'}")
    print("\n✓ Phase 1.5 vocoder interface OK")

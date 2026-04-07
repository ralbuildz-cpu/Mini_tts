"""
Centralized audio configuration.

ALL modules in mini_tts import constants from here.
Never hard-code sample_rate, hop_length, or n_mels anywhere else.

Quick swap example:
    from models.audio_config import AudioConfig
    cfg = AudioConfig()
    cfg.sample_rate  # 22050
"""

from dataclasses import dataclass, field


@dataclass
class AudioConfig:
    """
    Single source of truth for audio parameters.

    Changing values here propagates to all modules.
    """

    # ── Core audio ────────────────────────────────────────────────────────────
    sample_rate: int = 22050       # Hz
    hop_length: int = 256          # samples per mel frame  (22050/256 ≈ 86 fps)
    n_fft: int = 1024              # FFT window size
    win_length: int = 1024         # STFT window length
    n_mels: int = 80               # mel filterbank bins
    fmin: float = 0.0              # mel filter min frequency
    fmax: float = 8000.0           # mel filter max frequency

    # ── Streaming ─────────────────────────────────────────────────────────────
    chunk_ms: int = 300            # target chunk duration in ms
    overlap_ms: int = 20           # crossfade overlap between chunks in ms

    # ── Normalization ─────────────────────────────────────────────────────────
    mel_min_db: float = -80.0      # floor for log-mel (dB)
    mel_ref_db: float = 20.0       # reference for log-mel normalization

    # ── Derived (read-only properties) ────────────────────────────────────────

    @property
    def chunk_frames(self) -> int:
        """Number of mel frames per streaming chunk."""
        return max(1, int(self.chunk_ms * self.sample_rate / 1000 / self.hop_length))

    @property
    def chunk_samples(self) -> int:
        """Number of audio samples per streaming chunk."""
        return self.chunk_frames * self.hop_length

    @property
    def overlap_frames(self) -> int:
        """Mel frames used for crossfade overlap."""
        return max(1, int(self.overlap_ms * self.sample_rate / 1000 / self.hop_length))

    @property
    def overlap_samples(self) -> int:
        """Audio samples used for crossfade overlap."""
        return self.overlap_frames * self.hop_length

    @property
    def frames_per_second(self) -> float:
        """Mel frames per second."""
        return self.sample_rate / self.hop_length

    def frames_for_duration(self, seconds: float) -> int:
        """Convert duration in seconds to mel frame count."""
        return int(seconds * self.frames_per_second)

    def duration_of_frames(self, frames: int) -> float:
        """Convert mel frame count to duration in seconds."""
        return frames * self.hop_length / self.sample_rate

    def __repr__(self) -> str:
        return (
            f"AudioConfig(sr={self.sample_rate}, hop={self.hop_length}, "
            f"n_mels={self.n_mels}, chunk_ms={self.chunk_ms})"
        )


# ── Module-level singleton (import this for convenience) ──────────────────────
DEFAULT_AUDIO_CONFIG = AudioConfig()

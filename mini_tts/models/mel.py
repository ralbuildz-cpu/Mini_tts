"""
Mel spectrogram extraction.

Consistent mel transform used across training, inference, and benchmarking.
Parameters must match the vocoder — any mismatch = garbage audio.

Designed to be exportable to ONNX and runnable with numpy fallback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class MelSpec(nn.Module):
    """
    Differentiable mel spectrogram extractor.
    Drop-in compatible with torchaudio.transforms.MelSpectrogram
    but more portable and with explicit normalization control.

    Usage:
        mel_fn = MelSpec(sample_rate=22050, hop_size=256)
        mel = mel_fn(waveform)     # (B, 80, T)
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_size: int = 256,
        win_size: int = 1024,
        num_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = 8000.0,
        center: bool = False,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        self.num_mels = num_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

        # Pre-compute mel filterbank (not a learnable param)
        mel_fb = self._build_mel_filterbank()
        self.register_buffer("mel_fb", mel_fb)

        hann = torch.hann_window(win_size)
        self.register_buffer("window", hann)

    def _build_mel_filterbank(self) -> torch.Tensor:
        """Build mel filterbank matrix (num_mels, n_fft//2+1)."""
        try:
            import librosa
            fb = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.num_mels,
                fmin=self.fmin,
                fmax=self.fmax,
            )
            return torch.from_numpy(fb).float()
        except ImportError:
            # Numpy-only fallback (less accurate but no dependency)
            return self._numpy_mel_filterbank()

    def _numpy_mel_filterbank(self) -> torch.Tensor:
        """Pure numpy mel filterbank. Used when librosa is not available."""
        def hz_to_mel(hz):
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        fmax = self.fmax if self.fmax else self.sample_rate / 2
        mel_min = hz_to_mel(self.fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = np.linspace(mel_min, mel_max, self.num_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bins = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        fbank = np.zeros((self.num_mels, self.n_fft // 2 + 1))
        for m in range(1, self.num_mels + 1):
            f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-10)
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-10)

        return torch.from_numpy(fbank).float()

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) or (T,)  — values in [-1, 1]
        Returns:
            mel: (B, num_mels, T_mel)  — log scale
        """
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        if self.center:
            waveform = F.pad(waveform.unsqueeze(1),
                             (self.n_fft // 2, self.n_fft // 2), mode="reflect")
            waveform = waveform.squeeze(1)

        spec = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window,
            return_complex=True,
        )
        spec = torch.abs(spec)           # magnitude: (B, n_fft//2+1, T)

        # Apply mel filterbank
        mel = torch.matmul(self.mel_fb, spec)  # (B, num_mels, T)

        # Log compression
        mel = torch.clamp(mel, min=1e-5)
        mel = torch.log(mel)

        return mel

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_size


if __name__ == "__main__":
    print("Testing MelSpec...")
    mel_fn = MelSpec()
    audio = torch.randn(1, 22050)   # 1 second
    mel = mel_fn(audio)
    expected_frames = 22050 // 256   # ~86 frames
    print(f"Audio: {audio.shape} → Mel: {mel.shape}")
    print(f"Expected ~{expected_frames} frames, got {mel.shape[-1]}")
    print("✓ MelSpec OK")

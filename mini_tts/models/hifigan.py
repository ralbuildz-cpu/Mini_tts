"""
HiFi-GAN Vocoder — small config (V2/V3)
Converts mel spectrograms → waveforms in real time on CPU.

Architecture reference: Kong et al. 2020 (https://arxiv.org/abs/2010.05646)

Designed for quantization-friendly inference from day 1:
  - No BatchNorm (LayerNorm or weight norm only)
  - All weight_norm layers stripped after training for export
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from typing import List


# ─── Config dataclass ─────────────────────────────────────────────────────────

class HiFiGANConfig:
    """
    Hyperparameters for HiFi-GAN small (V3-ish).

    NOTE: num_mels / hop_size / sample_rate MUST match AudioConfig.
    The canonical values live in models/audio_config.py.
    These defaults are kept in sync manually.
    """
    # Input — keep in sync with AudioConfig
    num_mels: int = 80          # = AudioConfig.n_mels
    hop_size: int = 256         # = AudioConfig.hop_length
    sample_rate: int = 22050    # = AudioConfig.sample_rate

    # Generator
    upsample_rates: List[int] = None         # hop_size = product of these
    upsample_kernel_sizes: List[int] = None
    upsample_initial_channel: int = 256      # smaller = faster CPU
    resblock_kernel_sizes: List[int] = None
    resblock_dilation_sizes: List[List[int]] = None

    def __post_init__(self):
        if self.upsample_rates is None:
            self.upsample_rates = [8, 8, 4]           # 8*8*4 = 256 hop
        if self.upsample_kernel_sizes is None:
            self.upsample_kernel_sizes = [16, 16, 8]
        if self.resblock_kernel_sizes is None:
            self.resblock_kernel_sizes = [3, 7, 11]
        if self.resblock_dilation_sizes is None:
            self.resblock_dilation_sizes = [
                [1, 3, 5],
                [1, 3, 5],
                [1, 3, 5],
            ]

    def __init__(self):
        self.__post_init__()


# ─── Building blocks ───────────────────────────────────────────────────────────

def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock(nn.Module):
    """Multi-receptive-field fusion residual block."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: List[int] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=d, padding=get_padding(kernel_size, d)
            )) for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(
                channels, channels, kernel_size, 1,
                dilation=1, padding=get_padding(kernel_size, 1)
            )) for _ in dilations
        ])
        self._init_weights()

    def _init_weights(self):
        for c in self.convs1:
            c.weight.data.normal_(0, 0.01)
        for c in self.convs2:
            c.weight.data.normal_(0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


# ─── Generator ────────────────────────────────────────────────────────────────

class HiFiGANGenerator(nn.Module):
    """
    HiFi-GAN generator (mel → waveform).

    Usage:
        gen = HiFiGANGenerator(config)
        audio = gen(mel_spec)   # (B, 1, T*hop_size)
    """

    def __init__(self, config: HiFiGANConfig = None):
        super().__init__()
        if config is None:
            config = HiFiGANConfig()
        self.config = config

        ch = config.upsample_initial_channel
        self.conv_pre = weight_norm(
            nn.Conv1d(config.num_mels, ch, 7, 1, padding=3)
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        for i, (u, k) in enumerate(zip(config.upsample_rates, config.upsample_kernel_sizes)):
            in_ch = ch // (2 ** i)
            out_ch = ch // (2 ** (i + 1))
            self.ups.append(weight_norm(
                nn.ConvTranspose1d(in_ch, out_ch, k, u,
                                   padding=(k - u) // 2)
            ))
            for ks, ds in zip(config.resblock_kernel_sizes,
                               config.resblock_dilation_sizes):
                self.resblocks.append(ResBlock(out_ch, ks, ds))

        self.conv_post = weight_norm(nn.Conv1d(out_ch, 1, 7, 1, padding=3))
        self.ups.apply(self._init_weights)
        self.conv_post.weight.data.normal_(0, 0.01)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            m.weight.data.normal_(0, 0.01)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, num_mels, T_mel)
        Returns:
            audio: (B, 1, T_audio)  values in [-1, 1]
        """
        x = self.conv_pre(mel)
        num_kernels = len(self.config.resblock_kernel_sizes)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = None
            for j in range(num_kernels):
                rb = self.resblocks[i * num_kernels + j]
                xs = rb(x) if xs is None else xs + rb(x)
            x = xs / num_kernels  # mean fusion

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        """Call after training — required before ONNX export."""
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)
        for up in self.ups:
            remove_weight_norm(up)
        for rb in self.resblocks:
            rb.remove_weight_norm()

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ─── Discriminators (training only) ───────────────────────────────────────────

class PeriodDiscriminator(nn.Module):
    """Multi-period discriminator sub-discriminator."""

    def __init__(self, period: int):
        super().__init__()
        self.period = period
        ch = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(ch[i], ch[i+1], (5, 1), (3, 1),
                                  padding=(2, 0)))
            for i in range(len(ch) - 1)
        ])
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor):
        B, C, T = x.shape
        pad = (self.period - T % self.period) % self.period
        x = F.pad(x, (0, pad))
        x = x.view(B, C, -1, self.period)
        fmaps = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, real, fake):
        r_outs, f_outs = [], []
        r_fmaps, f_fmaps = [], []
        for d in self.discriminators:
            r_out, r_fm = d(real)
            f_out, f_fm = d(fake)
            r_outs.append(r_out); r_fmaps.append(r_fm)
            f_outs.append(f_out); f_fmaps.append(f_fm)
        return r_outs, f_outs, r_fmaps, f_fmaps


class ScaleDiscriminator(nn.Module):
    """Sub-discriminator for multi-scale."""

    def __init__(self, use_spectral_norm: bool = False):
        super().__init__()
        norm = nn.utils.spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmaps = []
        for c in self.convs:
            x = F.leaky_relu(c(x), 0.1)
            fmaps.append(x)
        x = self.conv_post(x)
        fmaps.append(x)
        return x.flatten(1, -1), fmaps


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
            ScaleDiscriminator(),
        ])
        self.pooling = nn.ModuleList([
            nn.Identity(),
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, real, fake):
        r_outs, f_outs, r_fmaps, f_fmaps = [], [], [], []
        for pool, d in zip(self.pooling, self.discriminators):
            r_out, r_fm = d(pool(real))
            f_out, f_fm = d(pool(fake))
            r_outs.append(r_out); r_fmaps.append(r_fm)
            f_outs.append(f_out); f_fmaps.append(f_fm)
        return r_outs, f_outs, r_fmaps, f_fmaps


if __name__ == "__main__":
    cfg = HiFiGANConfig()
    gen = HiFiGANGenerator(cfg)
    print(f"HiFi-GAN Generator params: {gen.param_count:,}")

    # Smoke test
    mel = torch.randn(1, cfg.num_mels, 100)
    audio = gen(mel)
    expected_len = 100 * cfg.hop_size  # 25600 samples
    print(f"mel shape: {mel.shape}  →  audio shape: {audio.shape}")
    assert audio.shape == (1, 1, expected_len), f"Shape mismatch: {audio.shape}"
    print("✓ Shape test passed")

"""
Mini Acoustic Model — Phase 2
Converts phoneme IDs → mel spectrogram chunks (streaming, CPU-first).

Architecture (FastSpeech2-style, non-autoregressive):
  Phoneme Embedding (512d)
  → Encoder      : 4 × FFT block  (512d, 4 heads, FFN 2048)  → ~12.6M
  → Duration Pred: 2 × Conv1d                                  →  ~1.6M
  → Length Reg   : repeat phoneme features by predicted duration
  → FiLM inject  : speaker_emb → (γ, β) via small MLP         →  ~0.5M
  → Prosody Pred : pitch + energy (2 × Conv1d each)            →  ~0.8M
  → Prosody Emb  : (pitch, energy) → 512d additive embedding   →  ~0.1M
  → Decoder      : 4 × FFT block  (512d, 4 heads, FFN 2048)   → ~12.6M
  → Postnet      : 5 × Conv1d (mel refinement)                 →  ~2.6M
  → Linear proj  : 512 → 80                                    →   0.04M
  Total: ~31M parameters

Streaming contract:
  AcousticModelBase defines the interface.
  AcousticModel implements chunked inference with minimal state:
    - leftover_frames: partial frame buffer (< chunk_frames)
    - phoneme_carry: phonemes whose duration spans chunk boundary

Hard constraints honoured:
  ✓ No autoregressive decoding
  ✓ No attention over raw audio
  ✓ No full-sequence buffering (streaming mode emits chunks incrementally)
  ✓ Parallel computation (attention is full over the phoneme window only)
  ✓ INT8-quantization-friendly (no custom CUDA, standard ops only)
"""

from __future__ import annotations
import math
import abc
from typing import Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.audio_config import AudioConfig
from models.phoneme_vocab import PhonemeVocab, VOCAB_SIZE


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

class AcousticConfig:
    """All acoustic model hyper-parameters in one place."""

    # Model dimensions
    hidden_dim:     int = 512
    encoder_layers: int = 4
    decoder_layers: int = 4
    num_heads:      int = 4
    ffn_dim:        int = 2048
    dropout:        float = 0.1

    # Speaker conditioning
    speaker_emb_dim: int = 192    # matches SpeechBrain ECAPA output

    # Prosody
    pitch_emb_dim:  int = 256
    energy_emb_dim: int = 256

    # Duration predictor
    dur_conv_channels: int = 512
    dur_conv_kernel:   int = 3
    dur_conv_layers:   int = 2

    # Prosody predictor
    pros_conv_channels: int = 256
    pros_conv_kernel:   int = 3
    pros_conv_layers:   int = 2

    # Postnet
    postnet_channels: int = 512
    postnet_kernel:   int = 5
    postnet_layers:   int = 5

    # Streaming
    lookahead_phonemes: int = 2   # max phonemes peeked ahead per chunk

    # Inference
    min_duration: int = 1         # minimum frames per phoneme
    max_duration: int = 50        # clip exploding durations


# ─────────────────────────────────────────────────────────────────────────────
# Interface contract
# ─────────────────────────────────────────────────────────────────────────────

class AcousticModelBase(abc.ABC):
    """
    Swappable interface for any acoustic model.

    Implementations must be stateless between sessions except via reset_stream().
    """

    @abc.abstractmethod
    def infer(
        self,
        phoneme_ids:  torch.Tensor,          # [B, T_ph]
        speaker_emb:  Optional[torch.Tensor], # [B, speaker_emb_dim] or None
        dur_override: Optional[torch.Tensor] = None,  # [B, T_ph] or None
    ) -> torch.Tensor:
        """Full-sequence inference. Returns mel [B, n_mels, T_frames]."""
        ...

    @abc.abstractmethod
    def infer_chunk(
        self,
        phoneme_ids: torch.Tensor,            # [B, T_ph] (one window)
        speaker_emb: Optional[torch.Tensor],  # [B, speaker_emb_dim] or None
        prosody_state: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Chunked inference.
        Returns (mel_chunk [B, n_mels, chunk_frames], next_prosody_state).
        """
        ...

    @abc.abstractmethod
    def reset_stream(self) -> None:
        """Clear all streaming state. Call before a new utterance."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learned, cache up to 8192 frames)."""

    def __init__(self, d_model: int, max_len: int = 8192, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class FFTBlock(nn.Module):
    """
    Feed-Forward Transformer (FFT) block.
    Multi-head self-attention + position-wise FFN + residuals + LayerNorm.
    Identical to FastSpeech2 FFT block.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int,
                 dropout: float = 0.1, kernel_size: int = 1):
        super().__init__()
        self.attn   = nn.MultiheadAttention(d_model, num_heads, dropout=dropout,
                                            batch_first=True)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        # Position-wise FFN implemented as two Conv1d (kernel=1 = pointwise)
        self.ffn    = nn.Sequential(
            nn.Conv1d(d_model, ffn_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ffn_dim, d_model, kernel_size, padding=kernel_size // 2),
            nn.Dropout(dropout),
        )
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, D]
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.drop(attn_out))
        # FFN expects [B, D, T]
        ffn_out = self.ffn(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm2(x + ffn_out)
        return x


class ConvPredictor(nn.Module):
    """
    Small Conv1d predictor head — used for duration, pitch, energy.
    Input:  [B, T, d_model]
    Output: [B, T, out_dim]
    """

    def __init__(self, d_model: int, hidden: int, kernel: int,
                 n_layers: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch = d_model
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(in_ch, hidden, kernel, padding=kernel // 2),
                nn.ReLU(),
                nn.LayerNorm(hidden),  # will be applied after transpose
                nn.Dropout(dropout),
            ]
            in_ch = hidden
        self.convs   = nn.ModuleList()
        self.norms   = nn.ModuleList()
        self.drops   = nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(nn.Conv1d(
                d_model if i == 0 else hidden, hidden, kernel, padding=kernel // 2))
            self.norms.append(nn.LayerNorm(hidden))
            self.drops.append(nn.Dropout(dropout))
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        h = x.transpose(1, 2)  # [B, D, T]
        for conv, norm, drop in zip(self.convs, self.norms, self.drops):
            h = F.relu(conv(h))
            h = drop(norm(h.transpose(1, 2)).transpose(1, 2))
        h = h.transpose(1, 2)  # [B, T, hidden]
        return self.proj(h)    # [B, T, out_dim]


class LengthRegulator(nn.Module):
    """
    Expand phoneme-level features to frame-level using integer durations.
    Fully parallel (scatter/repeat approach — no Python loops over time).

    streaming_safe=True: operates on a slice with carryover state.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
        """
        x:         [B, T_ph, D]
        durations: [B, T_ph]  (integer, clamped >= 1)
        Returns:   [B, T_frames, D]
        """
        B, T_ph, D = x.shape
        outputs = []
        for b in range(B):
            # repeat_interleave is contiguous and fast
            expanded = x[b].repeat_interleave(durations[b], dim=0)  # [T_frames, D]
            outputs.append(expanded)
        # Pad to same length in batch
        max_len = max(o.size(0) for o in outputs)
        padded  = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
        for b, o in enumerate(outputs):
            padded[b, :o.size(0)] = o
        return padded


class FiLMConditioner(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) conditioning.

    Converts a conditioning vector (speaker embedding) into per-channel
    scale (γ) and shift (β) applied to the hidden state:
        h = γ * h + β

    Tiny MLP: cond_dim → hidden → 2 * d_model
    """

    def __init__(self, cond_dim: int, d_model: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * d_model),
        )
        # Init γ = 1, β = 0
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].bias.data[:d_model] = 1.0   # γ init = 1

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        h:    [B, T, D]
        cond: [B, cond_dim]
        """
        params = self.net(cond)                  # [B, 2D]
        gamma, beta = params.chunk(2, dim=-1)    # each [B, D]
        gamma = gamma.unsqueeze(1)               # [B, 1, D]
        beta  = beta.unsqueeze(1)                # [B, 1, D]
        return gamma * h + beta


class Postnet(nn.Module):
    """
    5-layer conv postnet for mel refinement.
    Predicts residual correction: mel_final = mel + postnet(mel).
    """

    def __init__(self, n_mels: int, channels: int, kernel: int, n_layers: int,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        in_ch  = n_mels
        for i in range(n_layers):
            out_ch = channels if i < n_layers - 1 else n_mels
            act    = nn.Tanh() if i < n_layers - 1 else nn.Identity()
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                act,
                nn.Dropout(dropout),
            ]
            in_ch = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [B, n_mels, T]
        return self.net(mel)


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────

class AcousticModel(nn.Module, AcousticModelBase):
    """
    Mini non-autoregressive acoustic model.
    Implements AcousticModelBase interface (swappable).
    Target: ~25–31M parameters.
    """

    def __init__(
        self,
        cfg:       AcousticConfig = None,
        audio_cfg: AudioConfig    = None,
        vocab:     PhonemeVocab   = None,
    ):
        super().__init__()
        self.cfg       = cfg       or AcousticConfig()
        self.audio_cfg = audio_cfg or AudioConfig()
        self.vocab     = vocab     or PhonemeVocab()

        C = self.cfg
        n_mels = self.audio_cfg.n_mels

        # ── Embedding ──────────────────────────────────────────────────────
        self.phoneme_embed = nn.Embedding(
            self.vocab.vocab_size, C.hidden_dim, padding_idx=self.vocab.pad_id)

        # ── Positional encoding ────────────────────────────────────────────
        self.pos_enc_ph    = PositionalEncoding(C.hidden_dim, dropout=C.dropout)
        self.pos_enc_frame = PositionalEncoding(C.hidden_dim, dropout=C.dropout)

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder = nn.ModuleList([
            FFTBlock(C.hidden_dim, C.num_heads, C.ffn_dim, C.dropout)
            for _ in range(C.encoder_layers)
        ])

        # ── Duration predictor ────────────────────────────────────────────
        self.duration_predictor = ConvPredictor(
            C.hidden_dim, C.dur_conv_channels, C.dur_conv_kernel,
            C.dur_conv_layers, out_dim=1)

        # ── Length regulator ──────────────────────────────────────────────
        self.length_regulator = LengthRegulator()

        # ── FiLM conditioning (speaker) — applied after length regulation ─
        self.film = FiLMConditioner(C.speaker_emb_dim, C.hidden_dim)
        # Fallback zero embedding when no speaker given
        self.register_buffer("zero_speaker",
                             torch.zeros(1, C.speaker_emb_dim))

        # ── Prosody predictors (pitch + energy) ───────────────────────────
        self.pitch_predictor = ConvPredictor(
            C.hidden_dim, C.pros_conv_channels, C.pros_conv_kernel,
            C.pros_conv_layers, out_dim=1)
        self.energy_predictor = ConvPredictor(
            C.hidden_dim, C.pros_conv_channels, C.pros_conv_kernel,
            C.pros_conv_layers, out_dim=1)

        # Pitch + energy → embedding (added to frame-level features)
        self.pitch_embed  = nn.Linear(1, C.hidden_dim)
        self.energy_embed = nn.Linear(1, C.hidden_dim)

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = nn.ModuleList([
            FFTBlock(C.hidden_dim, C.num_heads, C.ffn_dim, C.dropout)
            for _ in range(C.decoder_layers)
        ])

        # ── Output projection ─────────────────────────────────────────────
        self.mel_proj = nn.Linear(C.hidden_dim, n_mels)

        # ── Postnet ───────────────────────────────────────────────────────
        self.postnet = Postnet(
            n_mels, C.postnet_channels, C.postnet_kernel, C.postnet_layers)

        # ── Streaming state ───────────────────────────────────────────────
        self._frame_buffer: Optional[torch.Tensor] = None   # [B, n_mels, leftover]
        self._stream_active: bool = False

        # ── Weight init ───────────────────────────────────────────────────
        self._init_weights()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.01)

    # ── Parameter count ───────────────────────────────────────────────────────

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(
        self,
        phoneme_ids:    torch.Tensor,            # [B, T_ph]
        speaker_emb:    Optional[torch.Tensor],  # [B, speaker_emb_dim] or None
        dur_targets:    Optional[torch.Tensor] = None,  # [B, T_ph] int
        src_key_mask:   Optional[torch.Tensor] = None,  # [B, T_ph] bool (True=pad)
    ) -> dict:
        """
        Full forward pass for training.

        Returns dict with:
            mel_before  : [B, n_mels, T_frames]  (pre-postnet)
            mel_after   : [B, n_mels, T_frames]  (post-postnet)
            dur_pred    : [B, T_ph]               (log-duration predictions)
            pitch_pred  : [B, T_frames]
            energy_pred : [B, T_frames]
            durations   : [B, T_ph]               (actual durations used)
        """
        B = phoneme_ids.size(0)

        # ── Embed + encode phonemes ────────────────────────────────────────
        x = self.phoneme_embed(phoneme_ids)          # [B, T_ph, D]
        x = self.pos_enc_ph(x)

        for layer in self.encoder:
            x = layer(x, key_padding_mask=src_key_mask)

        # ── Duration prediction ────────────────────────────────────────────
        dur_log = self.duration_predictor(x).squeeze(-1)   # [B, T_ph]

        if dur_targets is not None:
            # Training: use ground-truth durations
            durations = dur_targets.long().clamp(self.cfg.min_duration, self.cfg.max_duration)
        else:
            # Inference: convert log-duration to integer
            durations = (torch.exp(dur_log) - 1).round().long()
            durations = durations.clamp(self.cfg.min_duration, self.cfg.max_duration)
            if src_key_mask is not None:
                durations = durations.masked_fill(src_key_mask, 0)

        # ── Length regulation ──────────────────────────────────────────────
        h = self.length_regulator(x, durations)            # [B, T_frames, D]
        h = self.pos_enc_frame(h)

        # ── Speaker FiLM ───────────────────────────────────────────────────
        if speaker_emb is None:
            speaker_emb = self.zero_speaker.expand(B, -1)
        h = self.film(h, speaker_emb)

        # ── Prosody prediction ─────────────────────────────────────────────
        pitch_pred  = self.pitch_predictor(h).squeeze(-1)   # [B, T_frames]
        energy_pred = self.energy_predictor(h).squeeze(-1)  # [B, T_frames]

        h = h + self.pitch_embed(pitch_pred.unsqueeze(-1))
        h = h + self.energy_embed(energy_pred.unsqueeze(-1))

        # ── Decode ────────────────────────────────────────────────────────
        for layer in self.decoder:
            h = layer(h)

        # ── Project to mel ────────────────────────────────────────────────
        mel_before = self.mel_proj(h).transpose(1, 2)       # [B, n_mels, T_frames]
        mel_after  = mel_before + self.postnet(mel_before)  # [B, n_mels, T_frames]

        return {
            "mel_before":  mel_before,
            "mel_after":   mel_after,
            "dur_pred":    dur_log,
            "pitch_pred":  pitch_pred,
            "energy_pred": energy_pred,
            "durations":   durations,
        }

    # ── Inference (full sequence) ─────────────────────────────────────────────

    @torch.no_grad()
    def infer(
        self,
        phoneme_ids:  torch.Tensor,
        speaker_emb:  Optional[torch.Tensor] = None,
        dur_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full-sequence inference. Returns mel [B, n_mels, T_frames]."""
        self.eval()
        out = self.forward(phoneme_ids, speaker_emb, dur_targets=dur_override)
        return out["mel_after"]

    # ── Streaming interface ───────────────────────────────────────────────────

    def reset_stream(self) -> None:
        """Clear streaming state. Call before each new utterance."""
        self._frame_buffer = None
        self._stream_active = False

    @torch.no_grad()
    def infer_chunk(
        self,
        phoneme_ids:   torch.Tensor,             # [B, T_ph] — one phoneme window
        speaker_emb:   Optional[torch.Tensor] = None,
        prosody_state: Optional[dict]         = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Chunked inference.

        Processes a window of phonemes, manages frame-level carryover,
        and emits exactly audio_config.chunk_frames mel frames.

        Strategy:
          1. Run full forward on phoneme window (parallel, no autoregression)
          2. Prepend carryover frames from previous chunk
          3. Emit chunk_frames, store remainder as new carryover
          4. If insufficient frames, zero-pad (rare — at utterance end)

        Returns:
            (mel_chunk [B, n_mels, chunk_frames], next_prosody_state)
        """
        self.eval()
        chunk_frames = self.audio_cfg.chunk_frames
        B, n_mels    = phoneme_ids.size(0), self.audio_cfg.n_mels

        # Run the acoustic model on this window
        out = self.forward(phoneme_ids, speaker_emb)
        mel = out["mel_after"]     # [B, n_mels, T_frames]

        # Prepend carryover from previous chunk
        if self._frame_buffer is not None:
            mel = torch.cat([self._frame_buffer, mel], dim=2)

        # Emit one chunk
        if mel.size(2) >= chunk_frames:
            chunk         = mel[:, :, :chunk_frames]
            self._frame_buffer = mel[:, :, chunk_frames:]   # store leftover
        else:
            # Pad to chunk_frames (utterance end / underrun)
            pad = torch.zeros(B, n_mels, chunk_frames - mel.size(2),
                              device=mel.device, dtype=mel.dtype)
            chunk = torch.cat([mel, pad], dim=2)
            self._frame_buffer = None

        next_state = {
            "pitch_pred":  out["pitch_pred"],
            "energy_pred": out["energy_pred"],
            "durations":   out["durations"],
        }
        self._stream_active = True
        return chunk, next_state

    def flush_stream(self) -> Optional[torch.Tensor]:
        """
        Flush remaining carryover frames at utterance end.
        Returns padded final chunk or None if buffer is empty.
        """
        if self._frame_buffer is None or self._frame_buffer.size(2) == 0:
            return None
        chunk_frames = self.audio_cfg.chunk_frames
        n_mels       = self.audio_cfg.n_mels
        buf = self._frame_buffer
        B   = buf.size(0)
        if buf.size(2) < chunk_frames:
            pad = torch.zeros(B, n_mels, chunk_frames - buf.size(2),
                              device=buf.device, dtype=buf.dtype)
            buf = torch.cat([buf, pad], dim=2)
        else:
            buf = buf[:, :, :chunk_frames]
        self._frame_buffer = None
        return buf


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions (training use)
# ─────────────────────────────────────────────────────────────────────────────

class AcousticLoss(nn.Module):
    """
    Combined training loss.

    L_total = w_mel * L1(mel_after, mel_gt)
            + w_mel * L1(mel_before, mel_gt)   (auxiliary)
            + w_dur * MSE(dur_pred, log(dur_gt + 1))
            + w_pit * MSE(pitch_pred, pitch_gt)  [optional]
            + w_ene * MSE(energy_pred, energy_gt) [optional]
    """

    def __init__(
        self,
        w_mel: float = 1.0,
        w_dur: float = 0.1,
        w_pit: float = 0.01,
        w_ene: float = 0.01,
    ):
        super().__init__()
        self.w_mel = w_mel
        self.w_dur = w_dur
        self.w_pit = w_pit
        self.w_ene = w_ene

    def forward(self, outputs: dict, targets: dict) -> dict:
        """
        outputs: dict from AcousticModel.forward()
        targets: {
            mel_gt:     [B, n_mels, T_frames]
            dur_gt:     [B, T_ph]
            pitch_gt:   [B, T_frames]   optional
            energy_gt:  [B, T_frames]   optional
            mel_mask:   [B, T_frames]   optional (True = valid)
        }
        """
        mel_after  = outputs["mel_after"]
        mel_before = outputs["mel_before"]
        dur_pred   = outputs["dur_pred"]

        mel_gt   = targets["mel_gt"]
        dur_gt   = targets["dur_gt"].float()
        mel_mask = targets.get("mel_mask", None)

        # Align lengths (mel_after may differ from mel_gt due to duration mismatch)
        min_t = min(mel_after.size(2), mel_gt.size(2))
        mel_after  = mel_after[:, :, :min_t]
        mel_before = mel_before[:, :, :min_t]
        mel_gt     = mel_gt[:, :, :min_t]

        if mel_mask is not None:
            mask = mel_mask[:, :min_t].unsqueeze(1).float()  # [B, 1, T]
            n    = mask.sum().clamp(min=1)
            l_mel  = ((mel_after  - mel_gt).abs() * mask).sum() / n
            l_mel += ((mel_before - mel_gt).abs() * mask).sum() / n
        else:
            l_mel  = F.l1_loss(mel_after,  mel_gt)
            l_mel += F.l1_loss(mel_before, mel_gt)

        # Duration loss: predict log(dur + 1)
        l_dur = F.mse_loss(dur_pred, torch.log(dur_gt.clamp(min=0) + 1))

        loss = self.w_mel * l_mel + self.w_dur * l_dur

        details = {"mel": l_mel.item(), "dur": l_dur.item()}

        # Optional prosody losses
        if "pitch_gt" in targets and targets["pitch_gt"] is not None:
            p_pred = outputs["pitch_pred"][:, :min_t]
            p_gt   = targets["pitch_gt"][:, :min_t]
            l_pit  = F.mse_loss(p_pred, p_gt)
            loss   = loss + self.w_pit * l_pit
            details["pitch"] = l_pit.item()

        if "energy_gt" in targets and targets["energy_gt"] is not None:
            e_pred = outputs["energy_pred"][:, :min_t]
            e_gt   = targets["energy_gt"][:, :min_t]
            l_ene  = F.mse_loss(e_pred, e_gt)
            loss   = loss + self.w_ene * l_ene
            details["energy"] = l_ene.item()

        details["total"] = loss.item()
        return {"loss": loss, "details": details}


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building AcousticModel...")
    model     = AcousticModel()
    audio_cfg = AudioConfig()
    print(f"  Parameters: {model.param_count:,}")
    print(f"  Target:     20M–40M")
    assert 15_000_000 < model.param_count < 50_000_000, \
        f"Unexpected param count: {model.param_count:,}"

    B, T_ph = 1, 20
    ph_ids  = torch.randint(1, 50, (B, T_ph))
    spk_emb = torch.randn(B, AcousticConfig.speaker_emb_dim)

    # Full-sequence inference
    mel = model.infer(ph_ids, spk_emb)
    print(f"  Full infer output: {mel.shape}  (expect [1, 80, T])")
    assert mel.shape[1] == audio_cfg.n_mels

    # Streaming inference
    model.reset_stream()
    chunk, state = model.infer_chunk(ph_ids, spk_emb)
    print(f"  Chunk output:      {chunk.shape}  (expect [1, 80, {audio_cfg.chunk_frames}])")
    assert chunk.shape == (B, audio_cfg.n_mels, audio_cfg.chunk_frames)

    # Loss
    loss_fn = AcousticLoss()
    out_train = model.forward(ph_ids, spk_emb)
    T_f = out_train["mel_after"].size(2)
    targets = {
        "mel_gt": torch.randn(B, audio_cfg.n_mels, T_f),
        "dur_gt": torch.ones(B, T_ph) * 5,
    }
    result = loss_fn(out_train, targets)
    print(f"  Loss: {result['details']}")
    assert result["loss"].item() > 0

    print("✓ All checks passed")

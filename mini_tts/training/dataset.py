"""
Training dataset for mini_tts acoustic model.

Loads (phoneme_ids, mel, durations, pitch, energy) tuples from disk.
Supports single-speaker (Phase 2A) and multi-speaker (Phase 2B+) modes.

Expected data layout on disk:
    data/
      metadata.csv          — filename | text | speaker_id | phoneme_str
      mels/
        utt_001.npy         — [n_mels, T_frames] float32
      phonemes/
        utt_001.npy         — [T_ph] int64 (pre-encoded integer IDs)
      durations/
        utt_001.npy         — [T_ph] int64 (frames per phoneme)
      pitch/
        utt_001.npy         — [T_frames] float32 (log-F0 or 0 for unvoiced)
      energy/
        utt_001.npy         — [T_frames] float32 (frame-level energy)
      speaker_embeddings/
        speaker_001.npy     — [192] float32 (ECAPA-TDNN embedding, Phase 2B+)

Preprocessing (offline):
    python tools/preprocess_dataset.py --data-dir /path/to/wavs --out-dir data/
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.audio_config import AudioConfig
from models.phoneme_vocab import PhonemeVocab, PAD_ID


# ─────────────────────────────────────────────────────────────────────────────
# Single-item container
# ─────────────────────────────────────────────────────────────────────────────

class TTSItem:
    __slots__ = ("utt_id", "phoneme_ids", "mel", "durations",
                 "pitch", "energy", "speaker_id", "speaker_emb")

    def __init__(
        self,
        utt_id:      str,
        phoneme_ids: np.ndarray,   # [T_ph] int64
        mel:         np.ndarray,   # [n_mels, T_frames] float32
        durations:   np.ndarray,   # [T_ph] int64
        pitch:       np.ndarray,   # [T_frames] float32
        energy:      np.ndarray,   # [T_frames] float32
        speaker_id:  int = 0,
        speaker_emb: Optional[np.ndarray] = None,  # [192]
    ):
        self.utt_id      = utt_id
        self.phoneme_ids = phoneme_ids
        self.mel         = mel
        self.durations   = durations
        self.pitch       = pitch
        self.energy      = energy
        self.speaker_id  = speaker_id
        self.speaker_emb = speaker_emb


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class AcousticDataset(Dataset):
    """
    PyTorch Dataset for acoustic model training.

    Parameters
    ----------
    data_dir    : root data directory (see layout above)
    split       : "train" or "val"
    val_ratio   : fraction of data to use for validation
    max_ph_len  : discard utterances with more phonemes (avoids OOM)
    max_mel_len : discard utterances with more mel frames
    multi_speaker: if True, loads speaker embeddings
    audio_cfg   : AudioConfig (for n_mels, etc.)
    """

    def __init__(
        self,
        data_dir:      str,
        split:         str = "train",
        val_ratio:     float = 0.02,
        max_ph_len:    int = 200,
        max_mel_len:   int = 1000,
        multi_speaker: bool = False,
        audio_cfg:     AudioConfig = None,
        seed:          int = 42,
    ):
        self.data_dir      = Path(data_dir)
        self.split         = split
        self.multi_speaker = multi_speaker
        self.audio_cfg     = audio_cfg or AudioConfig()
        self.vocab         = PhonemeVocab()

        # Load metadata
        meta_path = self.data_dir / "metadata.csv"
        all_items = self._load_metadata(meta_path, max_ph_len, max_mel_len)

        # Train/val split (deterministic)
        rng = random.Random(seed)
        rng.shuffle(all_items)
        n_val = max(1, int(len(all_items) * val_ratio))
        if split == "val":
            self.items = all_items[:n_val]
        else:
            self.items = all_items[n_val:]

    def _load_metadata(self, meta_path: Path,
                       max_ph_len: int, max_mel_len: int) -> List[dict]:
        """Parse metadata.csv → list of dicts with paths and speaker info."""
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.csv not found at {meta_path}.\n"
                "Run: python tools/preprocess_dataset.py first."
            )
        items = []
        with open(meta_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                utt_id     = parts[0].strip()
                speaker_id = int(parts[2]) if len(parts) > 2 else 0

                ph_path  = self.data_dir / "phonemes"  / f"{utt_id}.npy"
                mel_path = self.data_dir / "mels"      / f"{utt_id}.npy"
                dur_path = self.data_dir / "durations" / f"{utt_id}.npy"

                if not ph_path.exists() or not mel_path.exists():
                    continue

                # Quick length check (load headers only if possible, else skip)
                try:
                    ph_arr  = np.load(ph_path, mmap_mode="r")
                    mel_arr = np.load(mel_path, mmap_mode="r")
                except Exception:
                    continue

                if len(ph_arr) > max_ph_len or mel_arr.shape[1] > max_mel_len:
                    continue

                items.append({
                    "utt_id":     utt_id,
                    "speaker_id": speaker_id,
                    "ph_path":    ph_path,
                    "mel_path":   mel_path,
                    "dur_path":   dur_path,
                    "pit_path":   self.data_dir / "pitch"  / f"{utt_id}.npy",
                    "ene_path":   self.data_dir / "energy" / f"{utt_id}.npy",
                    "spk_path":   (self.data_dir / "speaker_embeddings"
                                   / f"speaker_{speaker_id:03d}.npy"),
                })
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> TTSItem:
        m = self.items[idx]

        phoneme_ids = np.load(m["ph_path"]).astype(np.int64)
        mel         = np.load(m["mel_path"]).astype(np.float32)

        durations   = (np.load(m["dur_path"]).astype(np.int64)
                       if m["dur_path"].exists() else np.ones(len(phoneme_ids), dtype=np.int64))

        pitch  = (np.load(m["pit_path"]).astype(np.float32)
                  if m["pit_path"].exists() else np.zeros(mel.shape[1], dtype=np.float32))
        energy = (np.load(m["ene_path"]).astype(np.float32)
                  if m["ene_path"].exists() else np.zeros(mel.shape[1], dtype=np.float32))

        spk_emb = None
        if self.multi_speaker and m["spk_path"].exists():
            spk_emb = np.load(m["spk_path"]).astype(np.float32)

        return TTSItem(
            utt_id=m["utt_id"],
            phoneme_ids=phoneme_ids,
            mel=mel,
            durations=durations,
            pitch=pitch,
            energy=energy,
            speaker_id=m["speaker_id"],
            speaker_emb=spk_emb,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Collate
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(batch: List[TTSItem]) -> Dict[str, torch.Tensor]:
    """
    Pad batch to uniform lengths.

    Returns dict with all tensors on CPU (move to device in trainer).
    """
    B          = len(batch)
    max_ph_len = max(len(item.phoneme_ids) for item in batch)
    max_mel_t  = max(item.mel.shape[1]     for item in batch)

    phoneme_ids = torch.full((B, max_ph_len), PAD_ID, dtype=torch.long)
    durations   = torch.zeros(B, max_ph_len, dtype=torch.long)
    mel_gt      = torch.zeros(B, batch[0].mel.shape[0], max_mel_t)
    pitch_gt    = torch.zeros(B, max_mel_t)
    energy_gt   = torch.zeros(B, max_mel_t)
    ph_lengths  = torch.zeros(B, dtype=torch.long)
    mel_lengths = torch.zeros(B, dtype=torch.long)
    mel_mask    = torch.zeros(B, max_mel_t, dtype=torch.bool)  # True=valid

    speaker_ids  = torch.zeros(B, dtype=torch.long)
    has_spk_emb  = any(item.speaker_emb is not None for item in batch)
    if has_spk_emb:
        spk_dim   = batch[0].speaker_emb.shape[0]
        spk_embs  = torch.zeros(B, spk_dim)
    else:
        spk_embs  = None

    for i, item in enumerate(batch):
        T_ph = len(item.phoneme_ids)
        T_fr = item.mel.shape[1]

        phoneme_ids[i, :T_ph] = torch.tensor(item.phoneme_ids)
        durations[i, :T_ph]   = torch.tensor(item.durations)
        mel_gt[i, :, :T_fr]   = torch.tensor(item.mel)
        pitch_gt[i, :T_fr]    = torch.tensor(item.pitch)
        energy_gt[i, :T_fr]   = torch.tensor(item.energy)
        ph_lengths[i]          = T_ph
        mel_lengths[i]         = T_fr
        mel_mask[i, :T_fr]     = True
        speaker_ids[i]         = item.speaker_id

        if has_spk_emb and item.speaker_emb is not None:
            spk_embs[i] = torch.tensor(item.speaker_emb)

    # Padding mask for transformer (True = position to ignore)
    src_key_mask = (phoneme_ids == PAD_ID)

    out = {
        "phoneme_ids":   phoneme_ids,    # [B, T_ph]
        "durations":     durations,      # [B, T_ph]
        "mel_gt":        mel_gt,         # [B, n_mels, T_frames]
        "pitch_gt":      pitch_gt,       # [B, T_frames]
        "energy_gt":     energy_gt,      # [B, T_frames]
        "mel_mask":      mel_mask,       # [B, T_frames]
        "src_key_mask":  src_key_mask,   # [B, T_ph]
        "ph_lengths":    ph_lengths,     # [B]
        "mel_lengths":   mel_lengths,    # [B]
        "speaker_ids":   speaker_ids,    # [B]
    }
    if spk_embs is not None:
        out["speaker_embs"] = spk_embs  # [B, 192]

    return out


def build_dataloader(
    data_dir:      str,
    split:         str = "train",
    batch_size:    int = 16,
    num_workers:   int = 4,
    multi_speaker: bool = False,
    audio_cfg:     AudioConfig = None,
) -> DataLoader:
    ds = AcousticDataset(
        data_dir=data_dir,
        split=split,
        multi_speaker=multi_speaker,
        audio_cfg=audio_cfg,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=(split == "train"),
    )

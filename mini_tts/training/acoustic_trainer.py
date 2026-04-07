"""
Acoustic model trainer — Phases 2A / 2B / 2C

Phase 2A: Single speaker, no speaker embedding
Phase 2B: Multi-speaker, inject speaker embedding (load Phase 2A checkpoint)
Phase 2C: Full prosody (pitch + energy auxiliary losses)

Usage:
    # Phase 2A
    python training/acoustic_trainer.py \
        --data-dir data/ --out-dir checkpoints/ \
        --phase 2A --epochs 100 --batch-size 16

    # Phase 2B (resume from 2A checkpoint)
    python training/acoustic_trainer.py \
        --data-dir data/ --out-dir checkpoints/ \
        --phase 2B --resume checkpoints/phase2A_best.pt \
        --epochs 50 --multi-speaker

    # Phase 2C (resume from 2B)
    python training/acoustic_trainer.py \
        --data-dir data/ --out-dir checkpoints/ \
        --phase 2C --resume checkpoints/phase2B_best.pt \
        --epochs 30 --multi-speaker --prosody
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from models.audio_config import AudioConfig
from models.acoustic_model import AcousticModel, AcousticConfig, AcousticLoss
from models.phoneme_vocab import PhonemeVocab
from training.dataset import build_dataloader


# ─────────────────────────────────────────────────────────────────────────────
# Training config
# ─────────────────────────────────────────────────────────────────────────────

PHASE_LOSS_WEIGHTS = {
    "2A": dict(w_mel=1.0, w_dur=0.1, w_pit=0.0,  w_ene=0.0),
    "2B": dict(w_mel=1.0, w_dur=0.1, w_pit=0.0,  w_ene=0.0),
    "2C": dict(w_mel=1.0, w_dur=0.1, w_pit=0.01, w_ene=0.01),
}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class AcousticTrainer:

    def __init__(
        self,
        data_dir:      str,
        out_dir:       str,
        phase:         str  = "2A",
        epochs:        int  = 100,
        batch_size:    int  = 16,
        lr:            float = 1e-3,
        grad_clip:     float = 1.0,
        multi_speaker: bool = False,
        use_prosody:   bool = False,
        resume:        Optional[str] = None,
        device:        str  = "cpu",
        num_workers:   int  = 4,
        save_every:    int  = 10,
        log_every:     int  = 50,
    ):
        self.data_dir      = data_dir
        self.out_dir       = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.phase         = phase
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.lr            = lr
        self.grad_clip     = grad_clip
        self.multi_speaker = multi_speaker
        self.use_prosody   = use_prosody
        self.device        = torch.device(device)
        self.num_workers   = num_workers
        self.save_every    = save_every
        self.log_every     = log_every

        self.audio_cfg = AudioConfig()
        self.vocab     = PhonemeVocab()

        # Model
        self.model = AcousticModel(audio_cfg=self.audio_cfg, vocab=self.vocab)
        self.model.to(self.device)

        # Loss
        weights = PHASE_LOSS_WEIGHTS[phase]
        self.criterion = AcousticLoss(**weights)

        # Dataloaders
        self.train_dl = build_dataloader(
            data_dir, "train", batch_size, num_workers, multi_speaker, self.audio_cfg)
        self.val_dl = build_dataloader(
            data_dir, "val", batch_size, num_workers, multi_speaker, self.audio_cfg)

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=lr,
                               betas=(0.9, 0.98), weight_decay=1e-5)

        # Scheduler (OneCycleLR — warm up + cosine decay)
        steps_per_epoch = len(self.train_dl)
        self.scheduler  = OneCycleLR(
            self.optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
        )

        self.start_epoch  = 0
        self.best_val_loss = float("inf")
        self.global_step  = 0

        # Resume
        if resume:
            self._load_checkpoint(resume)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float, tag: str = "") -> None:
        name = f"phase{self.phase}_{tag or f'epoch{epoch:04d}'}.pt"
        path = self.out_dir / name
        torch.save({
            "epoch":     epoch,
            "step":      self.global_step,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "val_loss":  val_loss,
            "phase":     self.phase,
            "config":    {
                "multi_speaker": self.multi_speaker,
                "use_prosody":   self.use_prosody,
            },
        }, path)
        print(f"  [save] {path}")

    def _load_checkpoint(self, path: str) -> None:
        print(f"  [resume] loading {path}")
        state = torch.load(path, map_location=self.device)
        # Load model weights (allow partial match when upgrading phase)
        missing, unexpected = self.model.load_state_dict(
            state["model"], strict=False)
        if missing:
            print(f"  [resume] missing keys (expected for phase upgrade): {len(missing)}")
        self.start_epoch  = state.get("epoch", 0) + 1
        self.global_step  = state.get("step", 0)
        self.best_val_loss = state.get("val_loss", float("inf"))
        # Optimizer state only if same phase
        if state.get("phase") == self.phase:
            try:
                self.optimizer.load_state_dict(state["optimizer"])
                self.scheduler.load_state_dict(state["scheduler"])
            except Exception as e:
                print(f"  [resume] could not restore optimizer: {e}")

    # ── Train / eval steps ────────────────────────────────────────────────────

    def _batch_to_device(self, batch: dict) -> dict:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

    def _train_step(self, batch: dict) -> dict:
        self.model.train()
        batch = self._batch_to_device(batch)

        speaker_emb = batch.get("speaker_embs") if self.multi_speaker else None

        outputs = self.model(
            phoneme_ids=batch["phoneme_ids"],
            speaker_emb=speaker_emb,
            dur_targets=batch["durations"],
            src_key_mask=batch["src_key_mask"],
        )

        targets = {
            "mel_gt":    batch["mel_gt"],
            "dur_gt":    batch["durations"].float(),
            "mel_mask":  batch["mel_mask"],
        }
        if self.use_prosody:
            targets["pitch_gt"]  = batch["pitch_gt"]
            targets["energy_gt"] = batch["energy_gt"]

        result = self.criterion(outputs, targets)
        loss   = result["loss"]

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.scheduler.step()

        return result["details"]

    @torch.no_grad()
    def _val_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0
        for batch in self.val_dl:
            batch = self._batch_to_device(batch)
            speaker_emb = batch.get("speaker_embs") if self.multi_speaker else None

            outputs = self.model(
                phoneme_ids=batch["phoneme_ids"],
                speaker_emb=speaker_emb,
                dur_targets=batch["durations"],
                src_key_mask=batch["src_key_mask"],
            )
            targets = {
                "mel_gt":    batch["mel_gt"],
                "dur_gt":    batch["durations"].float(),
                "mel_mask":  batch["mel_mask"],
            }
            result     = self.criterion(outputs, targets)
            total_loss += result["details"]["total"]
            n_batches  += 1
        return total_loss / max(n_batches, 1)

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self) -> None:
        print(f"\n{'='*55}")
        print(f"  mini_tts Acoustic Trainer — Phase {self.phase}")
        print(f"  Model params : {self.model.param_count:,}")
        print(f"  Training set : {len(self.train_dl.dataset)} utterances")
        print(f"  Val set      : {len(self.val_dl.dataset)} utterances")
        print(f"  Device       : {self.device}")
        print(f"  Epochs       : {self.epochs}")
        print(f"{'='*55}\n")

        log_path = self.out_dir / f"train_log_phase{self.phase}.jsonl"

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            running = {}
            n_steps = 0

            for step, batch in enumerate(self.train_dl):
                details = self._train_step(batch)
                self.global_step += 1
                n_steps += 1
                for k, v in details.items():
                    running[k] = running.get(k, 0.0) + v

                if self.global_step % self.log_every == 0:
                    avg = {k: v / n_steps for k, v in running.items()}
                    lr  = self.scheduler.get_last_lr()[0]
                    print(f"  ep{epoch:03d} step{self.global_step:06d} "
                          f"loss={avg.get('total', 0):.4f}  "
                          f"mel={avg.get('mel', 0):.4f}  "
                          f"dur={avg.get('dur', 0):.4f}  "
                          f"lr={lr:.6f}")
                    running = {}
                    n_steps = 0

            # Validation
            val_loss = self._val_epoch()
            elapsed  = time.time() - t0
            is_best  = val_loss < self.best_val_loss

            print(f"\n  Epoch {epoch:03d}  val_loss={val_loss:.4f}  "
                  f"time={elapsed:.1f}s  {'← BEST' if is_best else ''}")

            # Log
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "val_loss": val_loss,
                    "step": self.global_step, "elapsed": elapsed,
                }) + "\n")

            # Save
            if is_best:
                self.best_val_loss = val_loss
                self._save_checkpoint(epoch, val_loss, "best")
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch, val_loss)

        print("\n✓ Training complete.")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Checkpoints in: {self.out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train mini_tts acoustic model")
    p.add_argument("--data-dir",     required=True,        help="Preprocessed data directory")
    p.add_argument("--out-dir",      default="checkpoints",help="Checkpoint output directory")
    p.add_argument("--phase",        default="2A",
                   choices=["2A", "2B", "2C"],             help="Training phase")
    p.add_argument("--epochs",       type=int, default=100)
    p.add_argument("--batch-size",   type=int, default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--multi-speaker",action="store_true",  help="Enable speaker conditioning")
    p.add_argument("--prosody",      action="store_true",  help="Enable prosody losses (Phase 2C)")
    p.add_argument("--resume",       default=None,         help="Resume from checkpoint path")
    p.add_argument("--device",       default="cpu",        help="cpu or cuda")
    p.add_argument("--num-workers",  type=int, default=4)
    p.add_argument("--save-every",   type=int, default=10, help="Save checkpoint every N epochs")
    p.add_argument("--log-every",    type=int, default=50, help="Log every N steps")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    trainer = AcousticTrainer(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        phase=args.phase,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_clip=args.grad_clip,
        multi_speaker=args.multi_speaker,
        use_prosody=args.prosody,
        resume=args.resume,
        device=args.device,
        num_workers=args.num_workers,
        save_every=args.save_every,
        log_every=args.log_every,
    )
    trainer.train()

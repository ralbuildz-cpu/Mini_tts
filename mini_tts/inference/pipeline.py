"""
Streaming end-to-end TTS pipeline.

Acoustic model → Vocoder → Audio chunks (real-time, CPU-first)

Usage:
    pipeline = TTSPipeline.from_pretrained(
        acoustic_path="checkpoints/acoustic.pt",
        vocoder_path="checkpoints/hifigan.pt",
    )
    pipeline.reset_stream()
    for audio_chunk in pipeline.stream(phoneme_ids, speaker_emb):
        play(audio_chunk)   # numpy float32, shape [samples]

Interface is swappable — acoustic model and vocoder can be replaced
independently as long as they implement their base contracts.
"""

from __future__ import annotations

import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Iterator, Optional, List

import torch
import numpy as np

from models.audio_config import AudioConfig
from models.acoustic_model import AcousticModel, AcousticModelBase, AcousticConfig
from models.phoneme_vocab import PhonemeVocab
from inference.vocoder import HiFiGANVocoder, VocoderBase


# ─────────────────────────────────────────────────────────────────────────────
# Timing stats
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChunkTiming:
    acoustic_ms:    float = 0.0
    vocoder_ms:     float = 0.0
    total_ms:       float = 0.0
    audio_duration_ms: float = 0.0

    @property
    def rtf(self) -> float:
        if self.audio_duration_ms == 0:
            return float("inf")
        return self.total_ms / self.audio_duration_ms


@dataclass
class SessionStats:
    chunk_timings: List[ChunkTiming] = field(default_factory=list)

    @property
    def first_chunk_latency_ms(self) -> float:
        return self.chunk_timings[0].total_ms if self.chunk_timings else 0.0

    @property
    def mean_rtf(self) -> float:
        if not self.chunk_timings:
            return 0.0
        return sum(t.rtf for t in self.chunk_timings) / len(self.chunk_timings)

    @property
    def max_rtf(self) -> float:
        return max((t.rtf for t in self.chunk_timings), default=0.0)

    def report(self) -> str:
        lines = [
            "── Session Stats ────────────────────────────────",
            f"  Chunks processed      : {len(self.chunk_timings)}",
            f"  First chunk latency   : {self.first_chunk_latency_ms:.1f} ms",
            f"  Mean RTF              : {self.mean_rtf:.3f}x",
            f"  Max RTF               : {self.max_rtf:.3f}x",
            f"  {'✓ REAL-TIME' if self.max_rtf < 1.0 else '✗ TOO SLOW'}",
            "─────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TTSPipeline:
    """
    Streaming TTS pipeline.

    Connects acoustic model → vocoder with matching chunk sizes.
    Both models operate on AudioConfig.chunk_frames at a time.
    """

    def __init__(
        self,
        acoustic: AcousticModelBase,
        vocoder:  VocoderBase,
        audio_cfg: AudioConfig = None,
    ):
        self.acoustic  = acoustic
        self.vocoder   = vocoder
        self.audio_cfg = audio_cfg or AudioConfig()
        self.stats     = SessionStats()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        acoustic_path: Optional[str] = None,
        vocoder_path:  Optional[str] = None,
        device: str = "cpu",
    ) -> "TTSPipeline":
        """
        Load pipeline from checkpoints.
        If paths are None, models are initialised with random weights (for testing).
        """
        audio_cfg = AudioConfig()
        vocab     = PhonemeVocab()

        # Acoustic model
        acoustic = AcousticModel(audio_cfg=audio_cfg, vocab=vocab)
        acoustic.eval()
        if acoustic_path:
            state = torch.load(acoustic_path, map_location=device)
            acoustic.load_state_dict(state["model"] if "model" in state else state)
        acoustic.to(device)

        # Vocoder
        vocoder = HiFiGANVocoder(audio_cfg=audio_cfg)
        if vocoder_path:
            vocoder.load(vocoder_path)

        return cls(acoustic=acoustic, vocoder=vocoder, audio_cfg=audio_cfg)

    # ── Streaming ─────────────────────────────────────────────────────────────

    def reset_stream(self) -> None:
        """Reset both models and stats. Call before each new utterance."""
        self.acoustic.reset_stream()
        self.vocoder.reset_stream()
        self.stats = SessionStats()

    def stream(
        self,
        phoneme_ids: torch.Tensor,         # [B, T_ph] — full utterance
        speaker_emb: Optional[torch.Tensor] = None,
        chunk_phonemes: int = 10,           # phonemes per acoustic chunk window
    ) -> Iterator[np.ndarray]:
        """
        Stream audio chunks as numpy arrays (float32, shape [samples]).

        Yields one chunk per acoustic window — each chunk is exactly
        audio_config.chunk_samples samples long.

        Args:
            phoneme_ids:    Full utterance phoneme sequence [1, T_ph]
            speaker_emb:    Speaker embedding [1, speaker_emb_dim] or None
            chunk_phonemes: Number of phonemes to process per acoustic step

        Yields:
            np.ndarray: audio chunk, float32, shape [chunk_samples]
        """
        T_ph = phoneme_ids.size(1)

        # Process phonemes in windows
        for start in range(0, T_ph, chunk_phonemes):
            end     = min(start + chunk_phonemes, T_ph)
            ph_win  = phoneme_ids[:, start:end]

            # ── Acoustic: phonemes → mel chunk ────────────────────────────
            t0 = time.perf_counter()
            mel_chunk, _ = self.acoustic.infer_chunk(ph_win, speaker_emb)
            t1 = time.perf_counter()
            acoustic_ms = (t1 - t0) * 1000

            # ── Vocoder: mel chunk → audio chunk ──────────────────────────
            t2 = time.perf_counter()
            audio_chunk = self.vocoder.infer_chunk(mel_chunk)   # np.ndarray
            t3 = time.perf_counter()
            vocoder_ms = (t3 - t2) * 1000

            total_ms      = (t3 - t0) * 1000
            audio_dur_ms  = (len(audio_chunk) / self.audio_cfg.sample_rate) * 1000

            self.stats.chunk_timings.append(ChunkTiming(
                acoustic_ms=acoustic_ms,
                vocoder_ms=vocoder_ms,
                total_ms=total_ms,
                audio_duration_ms=audio_dur_ms,
            ))

            yield audio_chunk

        # Flush any remaining acoustic frames
        final_mel = self.acoustic.flush_stream()
        if final_mel is not None:
            t0 = time.perf_counter()
            audio_chunk = self.vocoder.infer_chunk(final_mel)
            t1 = time.perf_counter()
            total_ms     = (t1 - t0) * 1000
            audio_dur_ms = (len(audio_chunk) / self.audio_cfg.sample_rate) * 1000
            self.stats.chunk_timings.append(ChunkTiming(
                vocoder_ms=total_ms,
                total_ms=total_ms,
                audio_duration_ms=audio_dur_ms,
            ))
            yield audio_chunk

        # Final vocoder flush
        audio_final = self.vocoder.flush()
        if audio_final is not None and len(audio_final) > 0:
            yield audio_final

    def synthesize(
        self,
        phoneme_ids: torch.Tensor,
        speaker_emb: Optional[torch.Tensor] = None,
        chunk_phonemes: int = 10,
    ) -> np.ndarray:
        """
        Full synthesis (non-streaming, for evaluation).
        Returns concatenated audio as numpy array.
        """
        self.reset_stream()
        chunks = list(self.stream(phoneme_ids, speaker_emb, chunk_phonemes))
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Threaded audio player helper (sounddevice optional)
# ─────────────────────────────────────────────────────────────────────────────

class StreamPlayer:
    """
    Plays audio chunks in a background thread as they arrive.
    Requires: pip install sounddevice

    Usage:
        player = StreamPlayer(sample_rate=22050)
        for chunk in pipeline.stream(...):
            player.push(chunk)
        player.wait()
    """

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def push(self, audio: np.ndarray) -> None:
        self._q.put(audio)

    def stop(self) -> None:
        self._q.put(None)

    def wait(self) -> None:
        self._q.join()

    def _run(self) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            print("[StreamPlayer] sounddevice not installed — audio not played.")
            while True:
                item = self._q.get()
                self._q.task_done()
                if item is None:
                    break
            return
        with sd.OutputStream(samplerate=self.sample_rate, channels=1,
                              dtype="float32") as stream:
            while True:
                item = self._q.get()
                self._q.task_done()
                if item is None:
                    break
                stream.write(item.reshape(-1, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building random-weight pipeline...")
    pipeline = TTSPipeline.from_pretrained()  # random weights
    audio_cfg = AudioConfig()

    B, T_ph = 1, 30
    ph_ids  = torch.randint(1, 50, (B, T_ph))
    spk_emb = torch.randn(B, AcousticConfig.speaker_emb_dim)

    print("Streaming...")
    pipeline.reset_stream()
    chunks = []
    for i, chunk in enumerate(pipeline.stream(ph_ids, spk_emb, chunk_phonemes=10)):
        print(f"  Chunk {i}: shape={chunk.shape}, "
              f"RTF={pipeline.stats.chunk_timings[i].rtf:.3f}x")
        chunks.append(chunk)

    audio = np.concatenate(chunks)
    print(f"Total audio: {len(audio)} samples = "
          f"{len(audio)/audio_cfg.sample_rate:.3f}s")
    print(pipeline.stats.report())
    print("✓ Pipeline smoke test passed")

"""
Phase 2 benchmark — End-to-end acoustic + vocoder latency.

Measures:
  - Acoustic model alone:  phonemes → mel chunk
  - End-to-end pipeline:   phonemes → audio chunk
  - First-chunk latency
  - RTF across multiple utterance lengths
  - Per-chunk timing distribution (streaming mode)

Usage:
    # Standard E2E table
    python tools/benchmark_acoustic.py

    # Streaming mode (detailed per-chunk stats)
    python tools/benchmark_acoustic.py --streaming

    # With real checkpoints
    python tools/benchmark_acoustic.py \
        --acoustic-ckpt checkpoints/phase2A_best.pt \
        --vocoder-ckpt  checkpoints/hifigan_v2.pt
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from typing import List, Optional

import torch
import numpy as np

from models.audio_config import AudioConfig
from models.acoustic_model import AcousticModel, AcousticConfig
from models.phoneme_vocab import PhonemeVocab
from inference.pipeline import TTSPipeline, SessionStats


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fake_phoneme_ids(n_phonemes: int, vocab_size: int) -> torch.Tensor:
    """Random phoneme IDs [1, n_phonemes] (skips PAD=0)."""
    return torch.randint(1, min(50, vocab_size), (1, n_phonemes))


def _fake_speaker_emb(dim: int = 192) -> torch.Tensor:
    emb = torch.randn(1, dim)
    return emb / emb.norm(dim=-1, keepdim=True)


def _seconds_to_phonemes(duration_s: float, avg_ph_rate: float = 12.0) -> int:
    """Approximate phonemes for a target utterance duration."""
    return max(5, int(duration_s * avg_ph_rate))


def _warmup(pipeline: TTSPipeline, vocab_size: int, n_reps: int = 3) -> None:
    """Run a few throw-away inferences to warm up caches."""
    ph  = _fake_phoneme_ids(15, vocab_size)
    spk = _fake_speaker_emb()
    for _ in range(n_reps):
        pipeline.reset_stream()
        _ = pipeline.synthesize(ph, spk, chunk_phonemes=10)
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Acoustic-model-alone benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_acoustic_alone(model: AcousticModel, audio_cfg: AudioConfig,
                         n_reps: int = 5) -> None:
    """Benchmark acoustic model in isolation (no vocoder)."""
    print("\n── Acoustic Model Alone ─────────────────────────────────────────")
    print(f"  Model params : {model.param_count:,}")
    print(f"  Chunk frames : {audio_cfg.chunk_frames}")
    print(f"  Chunk duration: {audio_cfg.chunk_ms}ms")
    print()
    print(f"  {'Phonemes':>10}  {'Mel frames':>12}  {'Latency (ms)':>14}  "
          f"{'RTF':>8}  {'Status':>10}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*14}  {'─'*8}  {'─'*10}")

    spk = _fake_speaker_emb()

    for n_ph in [10, 20, 40, 80, 150]:
        ph   = _fake_phoneme_ids(n_ph, model.vocab.vocab_size)
        times: List[float] = []

        model.reset_stream()
        for _ in range(n_reps):
            t0 = time.perf_counter()
            with torch.no_grad():
                mel = model.infer(ph, spk)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            model.reset_stream()

        med_ms   = statistics.median(times)
        n_frames = mel.shape[2]
        dur_ms   = (n_frames / audio_cfg.frames_per_second) * 1000
        rtf      = med_ms / max(dur_ms, 1)
        status   = "✓ fast" if rtf < 0.5 else ("⚠ slow" if rtf > 1.0 else "ok")

        print(f"  {n_ph:>10d}  {n_frames:>12d}  {med_ms:>13.1f}ms  "
              f"{rtf:>8.3f}x  {status:>10}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# E2E table benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_e2e(pipeline: TTSPipeline, audio_cfg: AudioConfig,
              n_reps: int = 3) -> None:
    """Full acoustic → vocoder RTF table."""
    print("\n── End-to-End Pipeline (Acoustic + Vocoder) ─────────────────────")
    print(f"  Sample rate  : {audio_cfg.sample_rate} Hz")
    print(f"  Chunk size   : {audio_cfg.chunk_ms}ms  ({audio_cfg.chunk_samples} samples)")
    print()
    print(f"  {'Utterance':>12}  {'Phonemes':>10}  {'First (ms)':>12}  "
          f"{'Mean RTF':>10}  {'Max RTF':>10}  {'Status':>10}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*12}  {'─'*10}  {'─'*10}  {'─'*10}")

    spk = _fake_speaker_emb()

    for dur_s in [0.5, 1.0, 2.0, 5.0, 10.0]:
        n_ph = _seconds_to_phonemes(dur_s)
        ph   = _fake_phoneme_ids(n_ph, pipeline.acoustic.vocab.vocab_size)

        first_ms_list: List[float] = []
        mean_rtf_list: List[float] = []
        max_rtf_list:  List[float] = []

        for _ in range(n_reps):
            pipeline.reset_stream()
            _ = pipeline.synthesize(ph, spk, chunk_phonemes=10)
            stats = pipeline.stats
            if stats.chunk_timings:
                first_ms_list.append(stats.first_chunk_latency_ms)
                mean_rtf_list.append(stats.mean_rtf)
                max_rtf_list.append(stats.max_rtf)

        if not first_ms_list:
            continue

        first_ms = statistics.median(first_ms_list)
        mean_rtf = statistics.median(mean_rtf_list)
        max_rtf  = statistics.median(max_rtf_list)

        first_ok = "✓" if first_ms < 300 else "✗"
        rtf_ok   = "✓" if max_rtf < 1.0  else "✗"
        status   = "✓ PASS" if first_ms < 300 and max_rtf < 1.0 else "✗ FAIL"

        print(f"  {dur_s:>10.1f}s  {n_ph:>10d}  {first_ms:>10.1f}ms{first_ok}  "
              f"{mean_rtf:>9.3f}x{rtf_ok[0]}  {max_rtf:>9.3f}x{rtf_ok[0]}  {status:>10}")

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Streaming detailed benchmark
# ─────────────────────────────────────────────────────────────────────────────

def bench_streaming(pipeline: TTSPipeline, audio_cfg: AudioConfig) -> None:
    """Per-chunk timing distribution and memory growth check."""
    print("\n── Streaming Mode (per-chunk detail) ────────────────────────────")

    n_ph = _seconds_to_phonemes(5.0)
    ph   = _fake_phoneme_ids(n_ph, pipeline.acoustic.vocab.vocab_size)
    spk  = _fake_speaker_emb()

    try:
        import tracemalloc
        tracemalloc.start()
        mem_tracking = True
    except Exception:
        mem_tracking = False

    pipeline.reset_stream()
    chunk_times: List[float] = []

    for i, chunk in enumerate(pipeline.stream(ph, spk, chunk_phonemes=10)):
        if i < len(pipeline.stats.chunk_timings):
            chunk_times.append(pipeline.stats.chunk_timings[i].total_ms)

    if mem_tracking:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        mem_growth_mb = peak / 1e6
    else:
        mem_growth_mb = 0.0

    if not chunk_times:
        print("  No chunks produced.")
        return

    rtfs = [
        ct.rtf for ct in pipeline.stats.chunk_timings
    ]

    print(f"  Chunks produced       : {len(chunk_times)}")
    print(f"  First chunk latency   : {chunk_times[0]:.1f} ms  "
          f"{'✓ < 300ms' if chunk_times[0] < 300 else '✗ > 300ms'}")
    print(f"  Median chunk latency  : {statistics.median(chunk_times):.1f} ms")
    print(f"  Stdev chunk latency   : {statistics.stdev(chunk_times) if len(chunk_times)>1 else 0:.1f} ms")
    print(f"  Mean RTF              : {statistics.mean(rtfs):.3f}x")
    print(f"  Max RTF               : {max(rtfs):.3f}x  "
          f"{'✓ < 1.0' if max(rtfs) < 1.0 else '✗ > 1.0'}")
    print(f"  Peak memory           : {mem_growth_mb:.1f} MB  "
          f"{'✓ < 50MB' if mem_growth_mb < 50 else '⚠'}")

    # Latency histogram
    bins = [0, 50, 100, 150, 200, 250, 300, 400, 500, float("inf")]
    labels = ["<50", "50-100", "100-150", "150-200", "200-250",
              "250-300", "300-400", "400-500", ">500"]
    counts = [0] * len(labels)
    for t in chunk_times:
        for j in range(len(bins) - 1):
            if bins[j] <= t < bins[j + 1]:
                counts[j] += 1
                break

    print(f"\n  Latency histogram (ms):")
    for label, count in zip(labels, counts):
        bar = "█" * count
        print(f"    {label:>8}ms : {bar} ({count})")

    # Exit condition summary
    print()
    first_ok = chunk_times[0] < 300
    rtf_ok   = max(rtfs) < 1.0
    if first_ok and rtf_ok:
        print("  ✓ STREAMING EXIT CONDITION MET")
        print("    Phase 2 streaming requirements satisfied.")
    else:
        print("  ✗ EXIT CONDITION NOT MET")
        if not first_ok:
            print(f"    First chunk too slow: {chunk_times[0]:.1f}ms (target < 300ms)")
        if not rtf_ok:
            print(f"    RTF too high: {max(rtfs):.3f}x (target < 1.0)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Benchmark Phase 2 acoustic + E2E pipeline")
    p.add_argument("--acoustic-ckpt", default=None, help="Acoustic model checkpoint")
    p.add_argument("--vocoder-ckpt",  default=None, help="HiFi-GAN checkpoint")
    p.add_argument("--streaming",     action="store_true", help="Run streaming benchmark")
    p.add_argument("--acoustic-only", action="store_true", help="Only bench acoustic model")
    p.add_argument("--reps",          type=int, default=3, help="Repetitions per measurement")
    p.add_argument("--device",        default="cpu")
    args = p.parse_args()

    print("\n" + "="*60)
    print("  mini_tts Phase 2 — Acoustic Model Benchmark")
    print("="*60)

    audio_cfg = AudioConfig()

    # Build pipeline (random weights if no checkpoints given)
    pipeline = TTSPipeline.from_pretrained(
        acoustic_path=args.acoustic_ckpt,
        vocoder_path=args.vocoder_ckpt,
        device=args.device,
    )

    print(f"\n  Acoustic params : {pipeline.acoustic.param_count:,}")
    print(f"  Device          : {args.device}")
    print(f"  Chunk config    : {audio_cfg.chunk_ms}ms / "
          f"{audio_cfg.chunk_frames} frames / "
          f"{audio_cfg.chunk_samples} samples")

    # Warm up
    print("\n  Warming up...")
    _warmup(pipeline, PhonemeVocab().vocab_size, n_reps=2)

    if args.acoustic_only:
        bench_acoustic_alone(pipeline.acoustic, audio_cfg, n_reps=args.reps)
        return

    if args.streaming:
        bench_streaming(pipeline, audio_cfg)
    else:
        bench_acoustic_alone(pipeline.acoustic, audio_cfg, n_reps=args.reps)
        bench_e2e(pipeline, audio_cfg, n_reps=args.reps)

    print("="*60 + "\n")


if __name__ == "__main__":
    main()

"""
Phase 1 / 1.5 vocoder benchmark.

Measures:
  - Real-time factor (RTF) across multiple durations
  - First-chunk latency (ms)
  - Per-chunk timing consistency
  - Memory growth during streaming session

Usage:
    python tools/benchmark_vocoder.py                              # standard RTF table
    python tools/benchmark_vocoder.py --streaming                  # streaming validation
    python tools/benchmark_vocoder.py --checkpoint path/to/ckpt    # real weights
    python tools/benchmark_vocoder.py --threads 4                  # pin CPU threads
    python tools/benchmark_vocoder.py --streaming --chunk-ms 200   # 200ms chunks

Phase 1 exit condition  : RTF < 1.0 across all durations
Phase 1.5 exit condition: first_chunk_latency < 300ms + no memory growth
"""

import argparse
import time
import tracemalloc
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.audio_config import AudioConfig
from inference.vocoder import HiFiGANVocoder


# ─── Helpers ──────────────────────────────────────────────────────────────────

def make_vocoder(args, audio_cfg: AudioConfig) -> HiFiGANVocoder:
    if args.checkpoint:
        return HiFiGANVocoder.from_pretrained(args.checkpoint, audio_cfg=audio_cfg)
    print("  ⚠  No checkpoint — using random weights (shape/speed only, not audio quality)")
    return HiFiGANVocoder.from_config(audio_cfg=audio_cfg)


def hr(width: int = 60, char: str = "─") -> str:
    return char * width


# ─── Standard benchmark ───────────────────────────────────────────────────────

def run_standard(vocoder: HiFiGANVocoder, durations, n_runs: int, audio_cfg: AudioConfig):
    """RTF table + first_chunk_latency for Phase 1 / 1.5."""

    print(f"\n{hr()}")
    print(f"  {'Duration':>10}  {'RTF mean':>10}  {'RTF min':>9}  {'Latency ms':>12}  {'Status':>10}")
    print(hr())

    results = []
    for dur in durations:
        frames = audio_cfg.frames_for_duration(dur)
        mel = torch.randn(1, audio_cfg.n_mels, frames)

        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = vocoder._generator(mel)

        # Timed runs
        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = vocoder._generator(mel)
                times.append(time.perf_counter() - t0)

        audio_duration = audio_cfg.duration_of_frames(frames)
        mean_rtf = np.mean(times) / audio_duration
        min_rtf  = min(times) / audio_duration
        mean_ms  = np.mean(times) * 1000

        status = "✓ REAL-TIME" if mean_rtf < 1.0 else "✗ TOO SLOW"
        results.append({"dur": dur, "mean_rtf": mean_rtf, "min_rtf": min_rtf,
                         "mean_ms": mean_ms, "pass": mean_rtf < 1.0})
        print(f"  {dur:>9.1f}s  {mean_rtf:>10.4f}x  {min_rtf:>9.4f}x  {mean_ms:>11.1f}ms  {status:>10}")

    print(hr())

    # First chunk latency
    chunk_frames = audio_cfg.chunk_frames
    mel_chunk = torch.randn(1, audio_cfg.n_mels, chunk_frames)
    vocoder.reset_stream()
    with torch.no_grad():
        t0 = time.perf_counter()
        _ = vocoder.infer_chunk(mel_chunk)
        first_chunk_ms = (time.perf_counter() - t0) * 1000
    vocoder.reset_stream()

    target_ms = 300
    fc_status = "✓" if first_chunk_ms < target_ms else "✗"
    print(f"\n  First-chunk latency : {first_chunk_ms:.1f}ms  {fc_status} (target < {target_ms}ms)")
    print(f"  Chunk size          : {audio_cfg.chunk_ms}ms  ({chunk_frames} mel frames)")

    # Exit condition summary
    all_pass = all(r["pass"] for r in results) and first_chunk_ms < target_ms
    print(f"\n  Phase 1.5 exit condition:")
    if all_pass:
        print("  ✓ PASSED — vocoder is real-time with low first-chunk latency")
        print("  → Ready for Phase 2 (acoustic model)")
    else:
        _print_failure_advice(results, first_chunk_ms, target_ms)

    return results, first_chunk_ms


def _print_failure_advice(results, first_chunk_ms, target_ms):
    print("  ✗ FAILED")
    slow = [r for r in results if not r["pass"]]
    if slow:
        print(f"  RTF > 1.0 at: {[r['dur'] for r in slow]}s durations")
        print("    → Try --threads N, or use V3 checkpoint (smallest HiFi-GAN config)")
    if first_chunk_ms >= target_ms:
        print(f"  First-chunk latency {first_chunk_ms:.0f}ms exceeds {target_ms}ms")
        print("    → Consider reducing --chunk-ms or pinning threads with --threads")


# ─── Streaming benchmark ──────────────────────────────────────────────────────

def run_streaming(vocoder: HiFiGANVocoder, session_seconds: float, audio_cfg: AudioConfig):
    """
    Simulate a continuous streaming session.

    Checks:
      1. First-chunk latency < 300ms
      2. Per-chunk RTF < 1.0 (consistent)
      3. No memory growth across chunks (< 5MB delta)
      4. Timing variance is low (std < 20% of mean)
    """
    print(f"\n{'='*60}")
    print(f"  Streaming Validation  ({session_seconds:.0f}s session, {audio_cfg.chunk_ms}ms chunks)")
    print(f"{'='*60}")

    chunk_frames = audio_cfg.chunk_frames
    total_frames = audio_cfg.frames_for_duration(session_seconds)
    n_chunks = max(1, total_frames // chunk_frames)

    print(f"  Chunks            : {n_chunks}")
    print(f"  Frames/chunk      : {chunk_frames}")
    print(f"  Overlap           : {audio_cfg.overlap_ms}ms ({audio_cfg.overlap_samples} samples)")

    vocoder.reset_stream()
    chunk_times_ms = []
    chunk_rtfs = []
    total_samples_out = 0
    chunk_audio_duration = audio_cfg.duration_of_frames(chunk_frames)

    # Start memory tracking
    tracemalloc.start()
    mem_snapshots = []

    for i in range(n_chunks):
        mel_chunk = torch.randn(1, audio_cfg.n_mels, chunk_frames)

        t0 = time.perf_counter()
        audio_out = vocoder.infer_chunk(mel_chunk)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        chunk_times_ms.append(elapsed_ms)
        chunk_rtfs.append((elapsed_ms / 1000) / chunk_audio_duration)
        total_samples_out += len(audio_out)

        # Memory snapshot every 10 chunks
        if i % 10 == 0:
            snapshot = tracemalloc.take_snapshot()
            stats = snapshot.statistics("lineno")
            mem_kb = sum(s.size for s in stats) / 1024
            mem_snapshots.append(mem_kb)

    tracemalloc.stop()
    vocoder.reset_stream()

    # ── Results ───────────────────────────────────────────────────────────────
    first_ms    = chunk_times_ms[0]
    mean_ms     = float(np.mean(chunk_times_ms))
    std_ms      = float(np.std(chunk_times_ms))
    max_ms      = float(np.max(chunk_times_ms))
    mean_rtf    = float(np.mean(chunk_rtfs))
    max_rtf     = float(np.max(chunk_rtfs))

    mem_delta_kb = (mem_snapshots[-1] - mem_snapshots[0]) if len(mem_snapshots) >= 2 else 0
    mem_delta_mb = mem_delta_kb / 1024

    print(f"\n{'─'*60}")
    print(f"  {'Metric':<30} {'Value':>15}  {'Status':>8}")
    print(f"{'─'*60}")

    def row(label, value_str, ok):
        sym = "✓" if ok else "✗"
        print(f"  {label:<30} {value_str:>15}  {sym:>8}")

    row("First-chunk latency",     f"{first_ms:.1f} ms",      first_ms < 300)
    row("Mean chunk latency",      f"{mean_ms:.1f} ms",        mean_ms < audio_cfg.chunk_ms)
    row("Max chunk latency",       f"{max_ms:.1f} ms",         max_ms < audio_cfg.chunk_ms * 2)
    row("Latency std dev",         f"{std_ms:.1f} ms",         std_ms < mean_ms * 0.20)
    row("Mean RTF",                f"{mean_rtf:.4f}x",         mean_rtf < 1.0)
    row("Max RTF",                 f"{max_rtf:.4f}x",          max_rtf < 1.0)
    row("Memory growth",           f"{mem_delta_mb:+.2f} MB",  mem_delta_mb < 5.0)

    print(f"{'─'*60}")
    print(f"  Total output samples  : {total_samples_out:,}")
    print(f"  Total output duration : {total_samples_out / audio_cfg.sample_rate:.2f}s")

    # Timing histogram (ASCII, compact)
    _print_timing_histogram(chunk_times_ms, audio_cfg.chunk_ms)

    # Exit condition
    checks = [
        first_ms < 300,
        mean_rtf < 1.0,
        max_rtf < 1.0,
        mem_delta_mb < 5.0,
    ]
    print(f"\n  Phase 1.5 streaming exit condition:")
    if all(checks):
        print("  ✓ PASSED — streaming is real-time, stable, no memory growth")
        print("  → Ready for Phase 2 (acoustic model)")
    else:
        print("  ✗ FAILED — see ✗ rows above")


def _print_timing_histogram(times_ms: list, chunk_ms: int):
    """ASCII histogram of chunk latencies."""
    arr = np.array(times_ms)
    bins = np.linspace(0, max(arr) * 1.1, 8)
    counts, edges = np.histogram(arr, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1

    print(f"\n  Chunk latency histogram (target < {chunk_ms}ms):")
    for i, (lo, hi, count) in enumerate(zip(edges[:-1], edges[1:], counts)):
        bar = "█" * int(count / max_count * 20)
        marker = " ◀ target" if lo < chunk_ms <= hi else ""
        print(f"  {lo:5.0f}–{hi:5.0f}ms │{bar:<20}│ {count}{marker}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="mini_tts Phase 1.5 vocoder benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to HiFi-GAN checkpoint (default: random weights)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Number of CPU threads")
    parser.add_argument("--runs", type=int, default=5,
                        help="Timing runs per duration for standard benchmark (default: 5)")
    parser.add_argument("--streaming", action="store_true",
                        help="Run streaming validation mode")
    parser.add_argument("--session-seconds", type=float, default=10.0,
                        help="Streaming session length in seconds (default: 10)")
    parser.add_argument("--chunk-ms", type=int, default=300,
                        help="Chunk duration in ms for streaming (default: 300)")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    # Build audio config
    audio_cfg = AudioConfig(chunk_ms=args.chunk_ms)

    print("\n" + "=" * 60)
    print("  mini_tts — Phase 1.5 Vocoder Benchmark")
    print("=" * 60)
    print(f"  CPU threads    : {torch.get_num_threads()}")
    print(f"  Sample rate    : {audio_cfg.sample_rate} Hz")
    print(f"  Hop length     : {audio_cfg.hop_length} samples")
    print(f"  Mel bins       : {audio_cfg.n_mels}")
    print(f"  Chunk          : {audio_cfg.chunk_ms}ms ({audio_cfg.chunk_frames} frames)")
    print(f"  Overlap        : {audio_cfg.overlap_ms}ms")

    vocoder = make_vocoder(args, audio_cfg)
    print(f"  Model params   : {vocoder.param_count:,}")

    if args.streaming:
        run_streaming(vocoder, args.session_seconds, audio_cfg)
    else:
        durations = [0.5, 1.0, 3.0, 5.0, 10.0]
        run_standard(vocoder, durations, args.runs, audio_cfg)

    print()


if __name__ == "__main__":
    main()

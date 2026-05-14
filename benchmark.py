"""
FP32 vs FP16 throughput / latency comparison.

Replays a fixed video clip through the full pipeline (face -> landmarks
-> head pose -> 2x eye state) and prints FPS, mean latency and p95
latency for each precision. Writes the same numbers to
benchmark_results.json so the README table stays in sync.

Usage:
    python benchmark.py --video assets/test_clip.mp4
    python benchmark.py --video assets/test_clip.mp4 --precisions FP32,FP16 --mode async

If you don't have a clip, record a 60-second video of yourself looking
around / closing your eyes and save it to assets/test_clip.mp4. Using a
file (not a webcam) keeps the numbers reproducible.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

import cv2

from pipeline import DriverPipeline


def run_one(video: str, models_root: str, precision: str, mode: str, max_frames: int) -> Dict:
    print(f"\n>> {precision} / {mode} — loading models")
    pipe = DriverPipeline(models_root, precision=precision, mode=mode)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video}")

    latencies_ms: List[float] = []
    frame_count = 0
    t_start = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            pipe.process(frame)
            latencies_ms.append((time.time() - t0) * 1000.0)
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break
        t_end = time.time()
    finally:
        pipe.shutdown()
        cap.release()

    elapsed = t_end - t_start
    fps = frame_count / elapsed if elapsed > 0 else 0.0
    mean_ms = statistics.mean(latencies_ms) if latencies_ms else 0.0
    p95_ms = (
        statistics.quantiles(latencies_ms, n=20)[18]
        if len(latencies_ms) >= 20
        else max(latencies_ms or [0.0])
    )
    print(f"   frames={frame_count}  elapsed={elapsed:.2f}s  fps={fps:.2f}  "
          f"mean={mean_ms:.2f}ms  p95={p95_ms:.2f}ms")
    return {
        "frames": frame_count,
        "elapsed_s": round(elapsed, 3),
        "fps": round(fps, 2),
        "mean_ms": round(mean_ms, 2),
        "p95_ms": round(p95_ms, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to a recorded clip (mp4)")
    ap.add_argument("--models", default="models")
    ap.add_argument("--precisions", default="FP32,FP16",
                    help="Comma-separated list of precisions to benchmark")
    ap.add_argument("--mode", default="async", choices=("async", "sync"))
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N frames (0 = run the whole clip)")
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    if not Path(args.video).exists():
        print(f"error: video not found: {args.video}")
        return 1

    results: Dict[str, Dict] = {}
    for prec in [p.strip().upper() for p in args.precisions.split(",") if p.strip()]:
        try:
            results[prec.lower()] = run_one(
                args.video, args.models, prec, args.mode, args.max_frames
            )
        except Exception as exc:
            print(f"!! {prec} failed: {exc}")
            results[prec.lower()] = {"error": str(exc)}

    Path(args.out).write_text(json.dumps(results, indent=2))
    print("\nwrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

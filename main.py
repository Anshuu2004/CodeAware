"""
DriveAware entry point.

Run a webcam (or a video file with --video) through the pipeline, draw the
overlay, fire alerts when PERCLOS or sustained head-pose deviation crosses
their thresholds, and append incidents to incidents.csv.

Quit with `q`.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from alerts import (
    Alerter,
    draw_alert_border,
    draw_face_box,
    draw_overlay,
    ensure_alert_sounds,
)
from pipeline import DriverPipeline
from scoring import DistractionDetector, PerclosScorer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DriveAware — drowsiness & distraction detection on Intel CPU")
    p.add_argument("--precision", choices=("FP16", "FP32"), default="FP16",
                   help="OpenVINO IR precision to load")
    p.add_argument("--mode", choices=("async", "sync"), default="async",
                   help="Inference scheduling mode")
    p.add_argument("--device", default="CPU", help="OpenVINO device (default: CPU)")
    p.add_argument("--camera", type=int, default=0, help="Webcam index")
    p.add_argument("--video", type=str, default=None,
                   help="Run on a video file instead of the webcam (handy for benchmarks)")
    p.add_argument("--models", type=str, default="models", help="Models root directory")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--no-display", action="store_true", help="Skip the cv2 window (for headless runs)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames (0 = unlimited). Useful for benchmarking.")
    return p.parse_args()


def open_capture(args: argparse.Namespace) -> cv2.VideoCapture:
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video file: {args.video}")
        return cap
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"could not open webcam index {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main() -> int:
    args = parse_args()

    print(f"[main] loading models (precision={args.precision}, mode={args.mode}, device={args.device})")
    pipeline = DriverPipeline(args.models, precision=args.precision, mode=args.mode, device=args.device)

    drowsy_wav, distracted_wav = ensure_alert_sounds(Path("assets"))
    alerter = Alerter(drowsy_wav=drowsy_wav, distracted_wav=distracted_wav)

    perclos = PerclosScorer(window_s=30.0, threshold=0.15, clear_threshold=0.10)
    distraction = DistractionDetector(yaw_thresh=30.0, pitch_thresh=20.0, sustain_s=1.5)

    cap = open_capture(args)
    last_face_seen = time.time()
    fps_window: deque = deque(maxlen=30)

    print("[main] running. press q to quit.")
    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    break
                print("[main] webcam read failed, retrying once")
                time.sleep(0.05)
                ok, frame = cap.read()
                if not ok:
                    break

            t0 = time.time()
            result = pipeline.process(frame)
            dt = time.time() - t0
            fps_window.append(dt)
            fps = len(fps_window) / sum(fps_window) if sum(fps_window) > 0 else 0.0

            yaw = pitch = 0.0
            both_closed = False
            face_visible = result.face is not None
            if face_visible:
                last_face_seen = time.time()
                f = result.face
                color = (0, 255, 0)
                if result.pose is not None:
                    yaw = result.pose.yaw
                    pitch = result.pose.pitch
                if result.eyes is not None:
                    both_closed = result.eyes.both_closed

                draw_face_box(frame, f.x1, f.y1, f.x2, f.y2, color=color)

            is_drowsy = perclos.update(both_closed if face_visible else False)
            is_distracted = distraction.update(yaw, pitch) if face_visible else False
            snap = perclos.snapshot()

            alerter.handle(
                is_drowsy=is_drowsy,
                is_distracted=is_distracted,
                perclos=snap.perclos,
                yaw=yaw,
                pitch=pitch,
                deviation_duration=distraction.deviation_duration,
            )

            # overlay
            lines = [
                f"FPS: {fps:5.1f}   inf: {result.inference_ms:5.1f} ms   mode: {args.mode}/{args.precision}",
                f"PERCLOS: {snap.perclos:.2f}   yaw: {yaw:+.1f}   pitch: {pitch:+.1f}",
            ]
            if result.eyes is not None:
                lines.append(
                    f"eyes: L={'closed' if result.eyes.left_closed else 'open  '} "
                    f"R={'closed' if result.eyes.right_closed else 'open  '}"
                )
            draw_overlay(frame, lines)

            if not face_visible and (time.time() - last_face_seen) > 2.0:
                draw_alert_border(frame, "DRIVER NOT VISIBLE", color=(0, 165, 255))
            elif is_drowsy:
                draw_alert_border(frame, "DROWSY", color=(0, 0, 255))
            elif is_distracted:
                draw_alert_border(frame, "DISTRACTED", color=(0, 0, 255))

            if not args.no_display:
                cv2.imshow("DriveAware", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        pipeline.shutdown()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

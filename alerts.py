"""
Alerts: audio (via simpleaudio if installed, winsound on Windows otherwise),
visual overlay drawn on the live preview, and CSV incident logging.

The audio backend is chosen at import time. If neither is available, the
alerter still works — it just stays silent and prints to stderr.
"""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# audio backend
# ---------------------------------------------------------------------------

_audio_backend: str = "none"

try:
    import simpleaudio  # type: ignore
    _audio_backend = "simpleaudio"
except Exception:
    if sys.platform.startswith("win"):
        try:
            import winsound  # type: ignore
            _audio_backend = "winsound"
        except Exception:
            pass


def _generate_tone_wav(path: Path, freq_hz: int, duration_s: float = 0.6, volume: float = 0.4) -> None:
    """Generate a simple sine-wave wav so we don't need to ship binary assets."""
    sr = 22050
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    wave_data = (np.sin(2 * np.pi * freq_hz * t) * volume * 32767).astype(np.int16)
    # gentle 30 ms attack/release so it doesn't click
    fade = int(0.03 * sr)
    if fade > 0:
        ramp = np.linspace(0, 1, fade)
        wave_data[:fade] = (wave_data[:fade] * ramp).astype(np.int16)
        wave_data[-fade:] = (wave_data[-fade:] * ramp[::-1]).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(wave_data.tobytes())


def ensure_alert_sounds(assets_dir: Path) -> tuple[Path, Path]:
    drowsy = assets_dir / "alert_drowsy.wav"
    distracted = assets_dir / "alert_distracted.wav"
    if not drowsy.exists():
        _generate_tone_wav(drowsy, freq_hz=440)        # A4, lower / mellow
    if not distracted.exists():
        _generate_tone_wav(distracted, freq_hz=880)    # A5, higher / urgent
    return drowsy, distracted


def _play_async(wav_path: Path) -> None:
    if _audio_backend == "simpleaudio":
        try:
            wav_obj = simpleaudio.WaveObject.from_wave_file(str(wav_path))
            wav_obj.play()
            return
        except Exception as exc:
            print(f"[alerts] simpleaudio failed: {exc}", file=sys.stderr)
    if _audio_backend == "winsound":
        try:
            winsound.PlaySound(str(wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        except Exception as exc:
            print(f"[alerts] winsound failed: {exc}", file=sys.stderr)
    # silent fallback: just print
    print(f"[alerts] (no audio backend) would play {wav_path.name}", file=sys.stderr)


# ---------------------------------------------------------------------------
# visual overlay
# ---------------------------------------------------------------------------

def draw_face_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color=(0, 255, 0)) -> None:
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


def draw_overlay(frame: np.ndarray, lines: list[str], origin=(10, 24), color=(255, 255, 255)) -> None:
    y = origin[1]
    for line in lines:
        cv2.putText(frame, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (origin[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 22


def draw_alert_border(frame: np.ndarray, label: str, color=(0, 0, 255), thickness: int = 8) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    x = (w - tw) // 2
    y = th + 16
    cv2.rectangle(frame, (x - 12, y - th - 8), (x + tw + 12, y + 8), color, -1)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# alerter (audio + cooldown + CSV logging)
# ---------------------------------------------------------------------------

@dataclass
class IncidentRecord:
    timestamp: str
    event_type: str
    perclos: float
    yaw: float
    pitch: float
    duration_s: float


class Alerter:
    """Owns the cooldown timers, plays sounds, and appends CSV incidents on rising edges."""

    def __init__(
        self,
        drowsy_wav: Path,
        distracted_wav: Path,
        csv_path: Path = Path("incidents.csv"),
        cooldown_s: float = 5.0,
    ):
        self.drowsy_wav = drowsy_wav
        self.distracted_wav = distracted_wav
        self.cooldown_s = cooldown_s
        self.csv_path = csv_path

        self._last_drowsy_audio = 0.0
        self._last_distracted_audio = 0.0
        self._prev_drowsy = False
        self._prev_distracted = False

        self._csv_lock = threading.Lock()
        self._init_csv()

    def _init_csv(self) -> None:
        if self.csv_path.exists():
            return
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "event_type", "perclos", "yaw", "pitch", "duration_s"])

    def _append_csv(self, rec: IncidentRecord) -> None:
        with self._csv_lock:
            with self.csv_path.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow([rec.timestamp, rec.event_type, f"{rec.perclos:.3f}",
                            f"{rec.yaw:.1f}", f"{rec.pitch:.1f}", f"{rec.duration_s:.2f}"])

    def _maybe_play(self, wav: Path, last_played_ref: str) -> bool:
        now = time.time()
        last = getattr(self, last_played_ref)
        if now - last < self.cooldown_s:
            return False
        setattr(self, last_played_ref, now)
        _play_async(wav)
        return True

    def handle(
        self,
        is_drowsy: bool,
        is_distracted: bool,
        perclos: float,
        yaw: float,
        pitch: float,
        deviation_duration: float,
    ) -> None:
        now_iso = datetime.now().isoformat(timespec="seconds")

        # rising edge detection -> CSV row + sound
        if is_drowsy and not self._prev_drowsy:
            self._append_csv(IncidentRecord(now_iso, "drowsy", perclos, yaw, pitch, 0.0))
        if is_distracted and not self._prev_distracted:
            self._append_csv(IncidentRecord(now_iso, "distracted", perclos, yaw, pitch, deviation_duration))

        if is_drowsy:
            self._maybe_play(self.drowsy_wav, "_last_drowsy_audio")
        if is_distracted:
            self._maybe_play(self.distracted_wav, "_last_distracted_audio")

        self._prev_drowsy = is_drowsy
        self._prev_distracted = is_distracted

"""
PERCLOS scorer + sustained-deviation distraction detector.

PERCLOS = Percentage of Eye Closure over a rolling window. We use the
"P80" flavour: fraction of time both eyes are >=80% closed. Since the
classifier just gives us a binary, we treat its "closed" decision as
the >=80% bucket.

Industry threshold for drowsy is around 0.15 (i.e. 15% of the window
spent with eyes shut). The clear threshold (hysteresis) avoids the
alert flickering on the boundary.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class ScoreSnapshot:
    perclos: float
    samples: int


class PerclosScorer:
    def __init__(
        self,
        window_s: float = 30.0,
        threshold: float = 0.15,
        clear_threshold: float = 0.10,
        time_fn=time.time,
    ):
        if clear_threshold > threshold:
            raise ValueError("clear_threshold must be <= threshold")
        self.window_s = window_s
        self.threshold = threshold
        self.clear_threshold = clear_threshold
        self._events: deque = deque()
        self._closed_count = 0
        self._is_drowsy = False
        self._time = time_fn

    @property
    def is_drowsy(self) -> bool:
        return self._is_drowsy

    def update(self, both_closed: bool) -> bool:
        now = self._time()
        self._events.append((now, both_closed))
        if both_closed:
            self._closed_count += 1

        # evict samples older than the window
        cutoff = now - self.window_s
        while self._events and self._events[0][0] < cutoff:
            _, was_closed = self._events.popleft()
            if was_closed:
                self._closed_count -= 1

        if not self._events:
            self._is_drowsy = False
            return False

        score = self._closed_count / len(self._events)
        if self._is_drowsy:
            if score < self.clear_threshold:
                self._is_drowsy = False
        else:
            if score > self.threshold:
                self._is_drowsy = True
        return self._is_drowsy

    def snapshot(self) -> ScoreSnapshot:
        n = len(self._events)
        score = self._closed_count / n if n else 0.0
        return ScoreSnapshot(perclos=score, samples=n)


class DistractionDetector:
    """Triggers when |yaw| or |pitch| stays past threshold for sustain_s seconds."""

    def __init__(
        self,
        yaw_thresh: float = 30.0,
        pitch_thresh: float = 20.0,
        sustain_s: float = 1.5,
        time_fn=time.time,
    ):
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh
        self.sustain_s = sustain_s
        self._deviation_start: float | None = None
        self._is_distracted = False
        self._time = time_fn

    @property
    def is_distracted(self) -> bool:
        return self._is_distracted

    @property
    def deviation_duration(self) -> float:
        if self._deviation_start is None:
            return 0.0
        return self._time() - self._deviation_start

    def update(self, yaw: float, pitch: float) -> bool:
        deviating = abs(yaw) > self.yaw_thresh or abs(pitch) > self.pitch_thresh
        now = self._time()
        if deviating:
            if self._deviation_start is None:
                self._deviation_start = now
            self._is_distracted = (now - self._deviation_start) >= self.sustain_s
        else:
            self._deviation_start = None
            self._is_distracted = False
        return self._is_distracted

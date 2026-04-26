"""Lightweight two-sensor motion detector.

Two circular pixel-change sensors ('blobs') are placed at the calibration
reference points.  When one blob triggers, a timer starts.  If the other
blob triggers within the timeout window (derived from the minimum speed of
interest), a speed event is emitted based on the known inter-blob distance
and the elapsed time.

This runs entirely on the host CPU — no VPU inference required — so it is
essentially free in terms of compute.  It cannot classify vehicles, but it
is more reliable than tracking for vehicles that the neural network misses
(e.g. at night, with glare, or with unusual shapes).
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class _Blob:
    center: tuple       # (x, y) in pixels
    radius: int
    background: Optional[float] = None   # rolling mean brightness


class BlobDetector:
    """Two-blob motion-based speed estimator.

    States
    ------
    IDLE       : neither blob has triggered
    WAITING_B  : blob A triggered first; waiting for blob B (vehicle going right)
    WAITING_A  : blob B triggered first; waiting for blob A (vehicle going left)
    """

    IDLE = "IDLE"
    WAITING_B = "WAITING_B"
    WAITING_A = "WAITING_A"

    def __init__(self, cfg: dict) -> None:
        ref_a = cfg["calibration_ref_a"]   # [x, y]
        ref_b = cfg["calibration_ref_b"]
        self._distance_m: float = cfg["calibration_distance_m"]

        # Radius = half of (1/20th of segment length), minimum 5 px
        dx = ref_b[0] - ref_a[0]
        dy = ref_b[1] - ref_a[1]
        segment_px = math.sqrt(dx ** 2 + dy ** 2)
        default_radius = max(5, int(segment_px / 40))
        radius = cfg.get("blob_radius_px", default_radius)

        self._blob_a = _Blob(center=(int(ref_a[0]), int(ref_a[1])), radius=radius)
        self._blob_b = _Blob(center=(int(ref_b[0]), int(ref_b[1])), radius=radius)

        self._bg_alpha: float = cfg.get("blob_bg_alpha", 0.02)
        self._threshold: float = cfg.get("blob_change_threshold", 15.0)

        # Timeout = travel time at minimum speed of interest
        min_speed_kmh: float = cfg.get("blob_min_speed_kmh", 1.0)
        self._timeout_s: float = self._distance_m / (min_speed_kmh / 3.6)

        self._state: str = self.IDLE
        self._first_time: Optional[float] = None

    # ------------------------------------------------------------------
    def process(self, frame, now: Optional[float] = None) -> list:
        """Sample both blobs and update state machine.

        Returns a (possibly empty) list of dicts:
            direction  : "left" | "right"
            speed_kmh  : float
            source     : "blob"
        """
        if now is None:
            now = time.monotonic()

        trig_a = self._sample_and_update(self._blob_a, frame)
        trig_b = self._sample_and_update(self._blob_b, frame)

        events = []

        if self._state == self.IDLE:
            if trig_a and not trig_b:
                self._state = self.WAITING_B
                self._first_time = now
            elif trig_b and not trig_a:
                self._state = self.WAITING_A
                self._first_time = now

        elif self._state == self.WAITING_B:
            if now - self._first_time > self._timeout_s:
                self._state = self.IDLE      # timeout — discard
            elif trig_b:
                events.append(self._make_event("right", now - self._first_time))
                self._state = self.IDLE

        elif self._state == self.WAITING_A:
            if now - self._first_time > self._timeout_s:
                self._state = self.IDLE
            elif trig_a:
                events.append(self._make_event("left", now - self._first_time))
                self._state = self.IDLE

        return events

    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        return self._state

    @property
    def blob_a(self) -> _Blob:
        return self._blob_a

    @property
    def blob_b(self) -> _Blob:
        return self._blob_b

    # ------------------------------------------------------------------
    def _sample_and_update(self, blob: _Blob, frame) -> bool:
        value = _mean_brightness(frame, blob.center, blob.radius)
        if blob.background is None:
            blob.background = value
            return False
        triggered = abs(value - blob.background) > self._threshold
        if not triggered:
            # Update background only when quiescent to avoid vehicle contamination
            blob.background += self._bg_alpha * (value - blob.background)
        return triggered

    def _make_event(self, direction: str, elapsed_s: float) -> dict:
        speed_kmh = round(self._distance_m / elapsed_s * 3.6, 1)
        return {"direction": direction, "speed_kmh": speed_kmh, "source": "blob"}


# ---------------------------------------------------------------------------
def _mean_brightness(frame, center: tuple, radius: int) -> float:
    """Mean pixel brightness (averaged over BGR channels) inside a circle."""
    cx, cy = center
    r = radius
    y1 = max(0, cy - r)
    y2 = min(frame.shape[0], cy + r + 1)
    x1 = max(0, cx - r)
    x2 = min(frame.shape[1], cx + r + 1)
    roi = frame[y1:y2, x1:x2]          # (h, w, 3)
    if roi.size == 0:
        return 0.0
    brightness = roi.mean(axis=2)       # (h, w) — mean over BGR
    lcy, lcx = cy - y1, cx - x1
    ys, xs = np.ogrid[:brightness.shape[0], :brightness.shape[1]]
    mask = (xs - lcx) ** 2 + (ys - lcy) ** 2 <= r ** 2
    return float(brightness[mask].mean()) if mask.any() else float(brightness.mean())

import time
from collections import defaultdict
from typing import Optional

import depthai as dai

from pipeline import VEHICLE_LABELS

# How many seconds of position history to keep per track
_HISTORY_WINDOW_S = 5.0

# Statuses that indicate an active (moving) track
_ACTIVE = {
    dai.Tracklet.TrackingStatus.NEW,
    dai.Tracklet.TrackingStatus.TRACKED,
}


class VehicleTracker:
    """Counts vehicles and estimates speed as they cross a virtual vertical line.

    The road runs left-to-right in the frame.  The trigger line is vertical,
    at counting_line_x_fraction of the frame width.  A vehicle is counted once
    per track lifetime when its centroid X crosses that line.  Speed is
    estimated from horizontal pixel displacement over time.
    """

    def __init__(self, cfg: dict) -> None:
        self._line_x: float = cfg["counting_line_x_fraction"]
        self._px_per_m: float = cfg["pixels_per_meter"]
        self._min_width_frac: float = cfg["min_track_width_fraction"]

        # {track_id: {"prev_cx": float|None, "x_history": [(mono_time, x_px)]}}
        self._state: dict = defaultdict(lambda: {"prev_cx": None, "x_history": []})
        # IDs that have already triggered a count (one count per track lifetime)
        self._counted: set = set()

    # ------------------------------------------------------------------
    def process(self, tracklets_msg, frame_width: int, frame_height: int) -> list:
        """Process one Tracklets message; return a (possibly empty) list of
        crossing events.

        Each event is a dict:
            vehicle_class : str   e.g. "car"
            direction     : str   "left" | "right"
            speed_kmh     : float | None
            _roi          : (x1, y1, x2, y2) pixel bbox — used by storage.py
        """
        events = []
        now = time.monotonic()

        for t in tracklets_msg.tracklets:
            if t.status == dai.Tracklet.TrackingStatus.REMOVED:
                self._state.pop(t.id, None)
                self._counted.discard(t.id)
                continue

            if t.label not in VEHICLE_LABELS:
                continue
            if t.status not in _ACTIVE:
                continue

            tid = t.id
            x1, y1, x2, y2 = _denorm_roi(t.roi, frame_width, frame_height)
            cx_px = (x1 + x2) / 2.0
            cx_norm = cx_px / frame_width

            state = self._state[tid]

            # Update position history
            state["x_history"].append((now, cx_px))
            cutoff = now - _HISTORY_WINDOW_S
            state["x_history"] = [
                (ts, x) for ts, x in state["x_history"] if ts >= cutoff
            ]

            prev_cx = state["prev_cx"]
            state["prev_cx"] = cx_norm

            if tid in self._counted or prev_cx is None:
                continue

            if not _crossed(prev_cx, cx_norm, self._line_x):
                continue

            self._counted.add(tid)

            direction = _direction(state["x_history"])
            speed_kmh = self._estimate_speed(state["x_history"], frame_width)

            events.append(
                {
                    "vehicle_class": VEHICLE_LABELS[t.label],
                    "direction": direction,
                    "speed_kmh": speed_kmh,
                    "_roi": (x1, y1, x2, y2),
                    "_track_id": tid,
                }
            )

        return events

    # ------------------------------------------------------------------
    def _estimate_speed(
        self, x_history: list, frame_width: int
    ) -> Optional[float]:
        if len(x_history) < 2:
            return None

        t0, x0 = x_history[0]
        t1, x1 = x_history[-1]
        dt = t1 - t0
        if dt < 0.1:
            return None

        dx_px = abs(x1 - x0)
        if dx_px / frame_width < self._min_width_frac:
            return None

        speed_m_s = (dx_px / self._px_per_m) / dt
        return round(speed_m_s * 3.6, 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _denorm_roi(roi, frame_width: int, frame_height: int) -> tuple:
    """Return (x1, y1, x2, y2) in pixels from a normalised dai.Rect."""
    tl = roi.topLeft()
    br = roi.bottomRight()
    x1 = int(tl.x * frame_width)
    y1 = int(tl.y * frame_height)
    x2 = int(br.x * frame_width)
    y2 = int(br.y * frame_height)
    return x1, y1, x2, y2


def _crossed(prev_x: float, cur_x: float, line_x: float) -> bool:
    """True when the centroid moves from one side of line_x to the other."""
    return (prev_x < line_x <= cur_x) or (prev_x > line_x >= cur_x)


def _direction(x_history: list) -> str:
    """'right' if net X motion is positive, 'left' otherwise."""
    if len(x_history) < 2:
        return "right"  # fallback; direction unknown with a single sample
    return "right" if x_history[-1][1] > x_history[0][1] else "left"

"""Ring-buffer clip recorder.

Keeps a rolling window of the last `pre_s` seconds of frames in memory.
When triggered, it continues recording for another `post_s` seconds, then
writes the clip to `data/clips/<timestamp>_<label>.mp4`.
"""

import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2

_CLIPS_DIR = Path("data/clips")


class ClipRecorder:
    def __init__(self, cfg: dict) -> None:
        self._enabled: bool = cfg.get("save_clips", False)
        if not self._enabled:
            return

        self._pre_s: float = cfg.get("clip_pre_s", 3.0)
        self._post_s: float = cfg.get("clip_post_s", 3.0)
        self._fps: float = cfg.get("clip_fps", 15.0)
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Ring buffer: deque of (mono_time, frame_bgr)
        max_frames = int((self._pre_s + 1) * self._fps)
        self._buf: deque = deque(maxlen=max_frames)

        # Active recording state
        self._recording: bool = False
        self._trigger_time: float = 0.0
        self._label: str = ""
        self._pending: list = []   # frames accumulated during post-roll

        _CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def push(self, frame, now: float = None) -> None:
        """Feed every frame here regardless of detection state."""
        if not self._enabled:
            return
        if now is None:
            now = time.monotonic()
        self._buf.append((now, frame.copy()))

        if self._recording:
            self._pending.append(frame.copy())
            if now - self._trigger_time >= self._post_s:
                self._flush()

    def trigger(self, label: str, now: float = None) -> None:
        """Call when a vehicle event fires; starts the post-roll."""
        if not self._enabled or self._recording:
            return
        if now is None:
            now = time.monotonic()
        self._recording = True
        self._trigger_time = now
        self._label = label
        self._pending = []

    # ------------------------------------------------------------------
    def _flush(self) -> None:
        """Write pre-roll + post-roll frames to an mp4 file."""
        self._recording = False

        # Collect pre-roll frames from ring buffer
        pre_frames = [f for ts, f in self._buf if ts <= self._trigger_time]
        all_frames = pre_frames + self._pending
        self._pending = []

        if not all_frames:
            return

        ts_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_label = self._label.replace(" ", "_").replace("/", "-")
        path = str(_CLIPS_DIR / f"{ts_str}_{safe_label}.mp4")

        h, w = all_frames[0].shape[:2]
        writer = cv2.VideoWriter(path, self._fourcc, self._fps, (w, h))
        for f in all_frames:
            writer.write(f)
        writer.release()

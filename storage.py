import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

log = logging.getLogger(__name__)

_DB_PATH = Path("data/counts.db")
_FRAMES_DIR = Path("data/frames")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT    NOT NULL,
    vehicle_class TEXT    NOT NULL,
    direction     TEXT    NOT NULL,
    speed_kmh     REAL,
    frame_path    TEXT
)
"""


class Storage:
    """SQLite-backed event log with optional per-vehicle JPEG crop."""

    def __init__(self, cfg: dict, dry_run: bool = False) -> None:
        self._dry_run = dry_run
        self._save_frames: bool = cfg.get("save_frames", False)
        self._conn: Optional[sqlite3.Connection] = None

        if dry_run:
            return

        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(_DB_PATH))
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

        if self._save_frames:
            _FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def log_event(self, event: dict, frame=None) -> None:
        """Persist one vehicle-crossing event.

        Args:
            event: dict from VehicleTracker.process() — must contain
                   vehicle_class, direction, speed_kmh, and optionally _roi.
            frame: full 640×640 BGR numpy array from the preview queue, or
                   None.  Used only when save_frames=true.
        """
        ts = datetime.now(timezone.utc).isoformat()

        if self._dry_run:
            log.info(
                "[dry-run] %s  class=%s  dir=%s  speed=%s km/h",
                ts,
                event["vehicle_class"],
                event["direction"],
                event.get("speed_kmh"),
            )
            return

        frame_path: Optional[str] = None
        if self._save_frames and frame is not None:
            frame_path = self._save_crop(frame, event, ts)

        self._conn.execute(
            "INSERT INTO events (timestamp, vehicle_class, direction, speed_kmh, frame_path)"
            " VALUES (?, ?, ?, ?, ?)",
            (
                ts,
                event["vehicle_class"],
                event["direction"],
                event.get("speed_kmh"),
                frame_path,
            ),
        )
        self._conn.commit()
        log.debug("Stored event: %s", event)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    def _save_crop(self, frame, event: dict, ts: str) -> Optional[str]:
        roi = event.get("_roi")
        if roi is None:
            return None

        x1, y1, x2, y2 = roi
        # Clamp to frame bounds
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        # Sanitise timestamp for use as a filename
        safe_ts = ts.replace(":", "-").replace(".", "-")
        fname = f"{safe_ts}_{event['vehicle_class']}.jpg"
        path = str(_FRAMES_DIR / fname)
        cv2.imwrite(path, crop)
        return path

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

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
    source        TEXT,
    frame_path    TEXT
)
"""


class Storage:
    """SQLite-backed event log with optional annotated full-frame JPEG."""

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
        ts = datetime.now(timezone.utc).isoformat()

        if self._dry_run:
            log.info(
                "[dry-run] %s  class=%s  dir=%s  speed=%s km/h  source=%s",
                ts,
                event["vehicle_class"],
                event["direction"],
                event.get("speed_kmh"),
                event.get("source", "yolo"),
            )
            return

        frame_path: Optional[str] = None
        if self._save_frames and frame is not None:
            frame_path = self._save_annotated(frame, event, ts)

        self._conn.execute(
            "INSERT INTO events (timestamp, vehicle_class, direction, speed_kmh, source, frame_path)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (
                ts,
                event["vehicle_class"],
                event["direction"],
                event.get("speed_kmh"),
                event.get("source", "yolo"),
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
    def _save_annotated(self, frame, event: dict, ts: str) -> Optional[str]:
        img = frame.copy()

        # Bounding box (YOLO events only)
        roi = event.get("_roi")
        if roi is not None:
            x1, y1, x2, y2 = roi
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 255), 2)

        # Annotation banner at top
        cls = event["vehicle_class"]
        direction = event["direction"]
        speed = event.get("speed_kmh")
        source = event.get("source", "yolo")
        speed_str = f"{speed} km/h" if speed is not None else "speed=?"

        # Human-readable local timestamp (UTC shown)
        ts_short = ts[:19].replace("T", " ") + " UTC"

        lines = [
            ts_short,
            f"{cls}  {direction}  {speed_str}  [{source}]",
        ]

        _draw_banner(img, lines)

        safe_ts = ts.replace(":", "-").replace(".", "-")
        fname = f"{safe_ts}_{cls}_{direction}.jpg"
        path = str(_FRAMES_DIR / fname)
        cv2.imwrite(path, img)
        return path


# ---------------------------------------------------------------------------

def _draw_banner(img, lines: list) -> None:
    """Draw a semi-transparent dark bar at the top of img with text lines."""
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    pad = 5
    line_h = 18

    bar_h = pad + len(lines) * line_h + pad
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, text in enumerate(lines):
        y = pad + (i + 1) * line_h - 3
        cv2.putText(img, text, (pad, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

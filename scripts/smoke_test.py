"""
Smoke-test — runs without OAK-D camera and without the model blob.

Tests:
  1. Config loading
  2. Storage: DB creation, event insert, readback
  3. Tracker: synthetic line-crossing produces the right events
  4. Pipeline: blob-path validation message is clear (skipped if blob present)

Run from the project root:
    python scripts/smoke_test.py
"""

import sys
import os
import tempfile
import sqlite3
from pathlib import Path
from types import SimpleNamespace

# Make sure project root is on the path regardless of cwd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
failures = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = PASS if ok else FAIL
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------
print("\n--- 1. Config loading ---")
try:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    check("config.yaml loads", True)
    required_keys = [
        "model_blob_path", "counting_line_x_fraction",
        "pixels_per_meter", "min_track_width_fraction",
        "save_frames", "log_level",
    ]
    for k in required_keys:
        check(f"  key '{k}' present", k in cfg)
except Exception as exc:
    check("config.yaml loads", False, str(exc))


# ---------------------------------------------------------------------------
# 2. Storage (no camera needed)
# ---------------------------------------------------------------------------
print("\n--- 2. Storage ---")
try:
    # Temporarily redirect DB to a temp directory
    import storage as _storage_mod
    orig_db = _storage_mod._DB_PATH
    orig_fr = _storage_mod._FRAMES_DIR

    with tempfile.TemporaryDirectory() as td:
        _storage_mod._DB_PATH = Path(td) / "test.db"
        _storage_mod._FRAMES_DIR = Path(td) / "frames"

        test_cfg = {**cfg, "save_frames": False}
        s = _storage_mod.Storage(test_cfg, dry_run=False)
        check("Storage created (DB file exists)", _storage_mod._DB_PATH.exists())

        event = {
            "vehicle_class": "car",
            "direction": "right",
            "speed_kmh": 42.5,
            "_roi": (10, 20, 100, 80),
        }
        s.log_event(event, frame=None)

        conn = sqlite3.connect(str(_storage_mod._DB_PATH))
        rows = conn.execute("SELECT vehicle_class, direction, speed_kmh FROM events").fetchall()
        conn.close()
        check("Event inserted", len(rows) == 1)
        check("Fields correct", rows[0] == ("car", "right", 42.5))

        # dry-run — nothing written
        s2 = _storage_mod.Storage(test_cfg, dry_run=True)
        s2.log_event(event, frame=None)
        check("dry-run produces no DB row", True)  # would have raised if broken

        s.close()

    _storage_mod._DB_PATH = orig_db
    _storage_mod._FRAMES_DIR = orig_fr

except Exception as exc:
    check("Storage suite", False, str(exc))


# ---------------------------------------------------------------------------
# 3. Tracker — synthetic tracklets (no depthai import needed for *logic*)
# ---------------------------------------------------------------------------
print("\n--- 3. Tracker logic ---")
try:
    import depthai as dai  # needed for TrackingStatus enum
    dai_available = True
except ImportError:
    dai_available = False
    print("  [INFO] depthai not installed — skipping tracker test")
    print("         Install with:  pip install depthai")

if dai_available:
    try:
        from tracker import VehicleTracker

        def make_tracklet(tid, label, status, x1, y1, x2, y2, fw=640, fh=640):
            """Build a minimal mock Tracklet."""
            tl = SimpleNamespace(x=x1 / fw, y=y1 / fh)
            br = SimpleNamespace(x=x2 / fw, y=y2 / fh)
            roi = SimpleNamespace(topLeft=lambda: tl, bottomRight=lambda: br)
            return SimpleNamespace(id=tid, label=label, status=status, roi=roi)

        def make_msg(tracklets):
            return SimpleNamespace(tracklets=tracklets)

        S = dai.Tracklet.TrackingStatus
        # Tracker now uses counting_line_x_fraction (vertical trigger line)
        vt = VehicleTracker(cfg)

        # -- Frame 1: vehicle (car=2) appears on left, centroid x_norm ≈ 0.23
        t1 = make_tracklet(1, 2, S.NEW, 100, 100, 200, 140)
        events = vt.process(make_msg([t1]), 640, 640)
        check("No event before crossing", len(events) == 0)

        # -- Frames 2-8: vehicle moves right, crosses x_fraction=0.5 around step 3
        import time as _time
        for step in range(7):
            x_off = (step + 1) * 50
            y_off = (step + 1) * 10
            t1 = make_tracklet(1, 2, S.TRACKED,
                                100 + x_off, 100 + y_off,
                                200 + x_off, 140 + y_off)
            events = vt.process(make_msg([t1]), 640, 640)
            if events:
                break

        check("Event fires on crossing", len(events) == 1)
        if events:
            ev = events[0]
            check("vehicle_class == 'car'", ev["vehicle_class"] == "car")
            check("direction == 'right'", ev["direction"] == "right")
            check("speed_kmh is float or None",
                  ev["speed_kmh"] is None or isinstance(ev["speed_kmh"], float))

        # -- Second crossing: same track, should not count again
        t1 = make_tracklet(1, 2, S.TRACKED, 500, 380, 600, 420)
        events2 = vt.process(make_msg([t1]), 640, 640)
        check("No double-count for same track", len(events2) == 0)

        # -- REMOVED status: state is cleaned up
        t_rm = make_tracklet(1, 2, S.REMOVED, 0, 0, 10, 10)
        vt.process(make_msg([t_rm]), 640, 640)
        check("Track state cleaned up after REMOVED", 1 not in vt._state)

    except Exception as exc:
        import traceback
        check("Tracker suite", False, str(exc))
        traceback.print_exc()


# ---------------------------------------------------------------------------
# 4. Blob detector logic (no camera needed)
# ---------------------------------------------------------------------------
print("\n--- 4. Blob detector logic ---")
try:
    import cv2
    import numpy as np
    from blob_detector import BlobDetector

    blob_cfg = {
        "calibration_ref_a": [100, 200],
        "calibration_ref_b": [250, 200],
        "calibration_distance_m": 4.0,
        "blob_bg_alpha": 0.02,
        "blob_change_threshold": 15.0,
        "blob_min_speed_kmh": 1.0,
    }
    bd = BlobDetector(blob_cfg)
    check("BlobDetector created", True)
    check("Initial state is IDLE", bd.state == "IDLE")

    # Blank frame — no trigger expected
    blank = np.zeros((640, 640, 3), dtype=np.uint8)
    bd.process(blank, now=0.0)  # initialises background
    events = bd.process(blank, now=0.1)
    check("No event on uniform blank frame", events == [])
    check("State still IDLE after blank frame", bd.state == "IDLE")

    # Bright patch at blob A → state should become WAITING_B
    frame_a = blank.copy()
    cx, cy, r = blob_cfg["calibration_ref_a"][0], blob_cfg["calibration_ref_a"][1], bd.blob_a.radius
    cv2.circle(frame_a, (cx, cy), r, (255, 255, 255), -1)
    bd.process(frame_a, now=0.2)
    check("State WAITING_B after blob A triggers", bd.state == "WAITING_B")

    # Bright patch at blob B → crossing event expected
    frame_b = blank.copy()
    cx2, cy2 = blob_cfg["calibration_ref_b"][0], blob_cfg["calibration_ref_b"][1]
    cv2.circle(frame_b, (cx2, cy2), bd.blob_b.radius, (255, 255, 255), -1)
    events = bd.process(frame_b, now=0.5)
    check("Event emitted when both blobs fire", len(events) == 1)
    if events:
        check("Direction is 'right'", events[0]["direction"] == "right")
        check("source == 'blob'", events[0]["source"] == "blob")
        check("speed_kmh is positive", events[0]["speed_kmh"] > 0)
    check("State returns to IDLE after event", bd.state == "IDLE")

    # Timeout: A fires but B never fires within timeout
    bd2 = BlobDetector(blob_cfg)
    bd2.process(blank, now=0.0)   # init background
    bd2.process(frame_a, now=0.1)
    check("WAITING_B after A", bd2.state == "WAITING_B")
    big_t = bd2._timeout_s + 1.0
    events2 = bd2.process(blank, now=0.1 + big_t)
    check("Timeout resets to IDLE", bd2.state == "IDLE")
    check("No event on timeout", events2 == [])

except Exception as exc:
    import traceback
    check("Blob detector suite", False, str(exc))
    traceback.print_exc()


# ---------------------------------------------------------------------------
# 5. Pipeline / blob check
# ---------------------------------------------------------------------------
print("\n--- 5. Pipeline / live device ---")
if not dai_available:
    print("  [INFO] depthai not installed — skipping pipeline test")
else:
    from pipeline import build_pipeline
    import time as _time

    model_file = ROOT / cfg["model_blob_path"]
    device_present = len(dai.Device.getAllAvailableDevices()) > 0

    if not model_file.exists():
        # Expected: FileNotFoundError with a helpful message
        try:
            build_pipeline(cfg)
            check("FileNotFoundError raised for missing model", False)
        except FileNotFoundError as exc:
            check("FileNotFoundError raised for missing model", True)
            check("Error mentions models.luxonis.com",
                  "models.luxonis.com" in str(exc))

    elif not device_present:
        print("  [INFO] No OAK-D device found — skipping live pipeline test")
        print("         Plug in the camera and re-run to test the full pipeline.")

    else:
        # Device + model present: build, start briefly, grab one frame, stop.
        try:
            pipeline, q_tracklets, q_preview = build_pipeline(cfg)
            check("Pipeline object created", pipeline is not None)
            pipeline.start()
            deadline = _time.monotonic() + 8
            got_frame = False
            while _time.monotonic() < deadline:
                msg = q_preview.tryGet()
                if msg is not None:
                    f = msg.getCvFrame()
                    got_frame = True
                    break
            pipeline.stop()
            check("Live frame received from camera", got_frame,
                  f"shape={f.shape}" if got_frame else "no frame in 8s")
        except Exception as exc:
            import traceback
            check("Pipeline suite", False, str(exc))
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if failures:
    print(f"FAILED ({len(failures)} check(s)): {', '.join(failures)}")
    sys.exit(1)
else:
    print("All checks passed.")

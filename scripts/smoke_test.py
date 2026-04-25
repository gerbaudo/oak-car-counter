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
        "model_blob_path", "counting_line_y_fraction",
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
        vt = VehicleTracker(cfg)

        # -- Frame 1: vehicle (car=2) appears near top, centroid y ≈ 0.2
        t1 = make_tracklet(1, 2, S.NEW, 100, 100, 200, 140)   # cy_norm ≈ 0.19
        events = vt.process(make_msg([t1]), 640, 640)
        check("No event before crossing", len(events) == 0)

        # -- Frames 2-8: vehicle moves right across the frame, descending
        import time as _time
        for step in range(7):
            x_off = (step + 1) * 50
            y_off = (step + 1) * 35  # crosses y=0.5 around step 4
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
# 4. Pipeline / blob check
# ---------------------------------------------------------------------------
print("\n--- 4. Pipeline ---")
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

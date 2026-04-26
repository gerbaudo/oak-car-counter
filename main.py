"""Entry point: argument parsing, device setup, main event loop."""

import argparse
import logging
import signal
import time

import cv2
import depthai as dai
import yaml

from blob_detector import BlobDetector
from clip_recorder import ClipRecorder
from pipeline import VEHICLE_LABELS, build_pipeline
from storage import Storage
from tracker import VehicleTracker, _denorm_roi

_FRAME_W = 640
_FRAME_H = 640
_WIN_NAME = "oak-car-counter"

_BANNER_TTL = 5.0       # seconds to show last-event banner

# Cross-check tuning
_HOLD_S = 2.5           # hold a YOLO event this long waiting for a blob match
_BLOB_YOLO_WINDOW = 3.0 # max age of an unmatched blob event before logging standalone
_SPEED_TOLERANCE = 0.30 # relative difference below which blob+YOLO speeds are "consistent"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Count and time vehicles using an OAK-D-Lite camera."
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print events to stdout without writing to the database.")
    p.add_argument("--display", action="store_true",
                   help="Show a live OpenCV window (requires a display or VNC).")
    return p.parse_args()


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Mouse callback — prints pixel X,Y to console for calibration
# ---------------------------------------------------------------------------
_last_click: dict = {}


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[click] x={x}  y={y}")
        _last_click["x"] = x
        _last_click["y"] = y


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config()

    logging.basicConfig(
        level=getattr(logging, cfg.get("log_level", "INFO")),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    log = logging.getLogger(__name__)

    pipeline, q_tracklets, q_preview = build_pipeline(cfg)
    vehicle_tracker = VehicleTracker(cfg)
    blob_detector = BlobDetector(cfg)
    storage = Storage(cfg, dry_run=args.dry_run)
    clip_recorder = ClipRecorder({**cfg, "save_clips": args.dry_run})

    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    try:
        signal.signal(signal.SIGTERM, _stop)
    except OSError:
        pass

    if not args.display:
        q_preview = None

    if args.display:
        cv2.namedWindow(_WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(_WIN_NAME, _on_mouse)
        if args.dry_run:
            log.info(
                "Calibration mode: click the display to print pixel coordinates."
                "  Press Space to freeze/unfreeze, Q to quit."
            )

    pipeline.start()
    log.info(
        "Pipeline running.  dry_run=%s  display=%s  Press Ctrl+C to stop.",
        args.dry_run,
        args.display,
    )

    paused = False
    track_speeds: dict = {}
    banner_text: str = ""
    banner_expiry: float = 0.0

    # pending_yolo: YOLO events held for blob cross-check before logging
    #   each entry: (fire_time, event_dict, frame_at_fire)
    pending_yolo: list = []

    # recent_blob: blob events not yet matched to a YOLO event
    #   each entry: (fire_time, blob_event_dict)
    recent_blob: list = []

    try:
        while running:
            now = time.monotonic()

            # ----- grab latest tracklets -----
            tracklets_msg = q_tracklets.tryGet()

            # ----- grab latest preview frame -----
            frame = None
            if q_preview is not None:
                frame_msg = q_preview.tryGet()
                if frame_msg is not None:
                    frame = frame_msg.getCvFrame()
                    clip_recorder.push(frame, now)

            if not paused:
                # ----- YOLO tracker -----
                if tracklets_msg is not None:
                    for t in tracklets_msg.tracklets:
                        if t.status == dai.Tracklet.TrackingStatus.REMOVED:
                            track_speeds.pop(t.id, None)

                    for event in vehicle_tracker.process(
                        tracklets_msg, frame_width=_FRAME_W, frame_height=_FRAME_H
                    ):
                        event.setdefault("source", "yolo")
                        # Update display immediately so the bounding box shows speed
                        track_speeds[event["_track_id"]] = event.get("speed_kmh")

                        # Try to match with an already-waiting blob event
                        idx = _match_idx(event["direction"], recent_blob)
                        if idx is not None:
                            _, bevent = recent_blob.pop(idx)
                            _cross_check(event, bevent, log)
                            track_speeds[event["_track_id"]] = event.get("speed_kmh")
                            banner_text, banner_expiry = _emit(
                                log, storage, event, frame, clip_recorder, now), now + _BANNER_TTL
                        else:
                            # Hold for up to _HOLD_S seconds for a blob to arrive
                            pending_yolo.append((now, event, frame))

                # ----- Drain pending YOLO events -----
                still_pending = []
                for t0, event, ev_frame in pending_yolo:
                    idx = _match_idx(event["direction"], recent_blob)
                    if idx is not None:
                        _, bevent = recent_blob.pop(idx)
                        _cross_check(event, bevent, log)
                        track_speeds[event["_track_id"]] = event.get("speed_kmh")
                        banner_text, banner_expiry = _emit(
                            log, storage, event, ev_frame, clip_recorder, t0), now + _BANNER_TTL
                    elif now - t0 >= _HOLD_S:
                        banner_text, banner_expiry = _emit(
                            log, storage, event, ev_frame, clip_recorder, t0), now + _BANNER_TTL
                    else:
                        still_pending.append((t0, event, ev_frame))
                pending_yolo = still_pending

                # ----- Blob detector -----
                if frame is not None:
                    for bevent in blob_detector.process(frame, now):
                        # Try to match with a pending YOLO event
                        idx = _pending_match_idx(bevent["direction"], pending_yolo)
                        if idx is not None:
                            t0, event, ev_frame = pending_yolo.pop(idx)
                            _cross_check(event, bevent, log)
                            track_speeds[event["_track_id"]] = event.get("speed_kmh")
                            banner_text, banner_expiry = _emit(
                                log, storage, event, ev_frame, clip_recorder, t0), now + _BANNER_TTL
                        else:
                            # No YOLO yet — park it and wait
                            recent_blob.append((now, bevent))

                # Flush stale unmatched blob events as standalone detections
                fresh_blob = []
                for t, bevent in recent_blob:
                    if now - t > _BLOB_YOLO_WINDOW:
                        bevent["vehicle_class"] = "unknown"
                        banner_text, banner_expiry = _emit(
                            log, storage, bevent, frame, clip_recorder, t), now + _BANNER_TTL
                    else:
                        fresh_blob.append((t, bevent))
                recent_blob = fresh_blob

            # ----- display -----
            if args.display and frame is not None:
                banner = banner_text if now < banner_expiry else ""
                _draw_overlay(frame, cfg, tracklets_msg, track_speeds,
                              blob_detector=blob_detector, banner=banner)
                cv2.imshow(_WIN_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    running = False
                elif key == ord(" "):
                    paused = not paused
                    log.info("Display %s.", "paused" if paused else "resumed")
    finally:
        pipeline.stop()

    if args.display:
        cv2.destroyAllWindows()

    storage.close()
    log.info("Stopped.")


# ---------------------------------------------------------------------------
# Cross-check and correlation helpers
# ---------------------------------------------------------------------------

def _match_idx(direction: str, recent_blob: list):
    """Return index of the most recent blob event with matching direction, or None."""
    for i in range(len(recent_blob) - 1, -1, -1):
        if recent_blob[i][1].get("direction") == direction:
            return i
    return None


def _pending_match_idx(direction: str, pending_yolo: list):
    """Return index of the most recent pending YOLO event with matching direction, or None."""
    for i in range(len(pending_yolo) - 1, -1, -1):
        if pending_yolo[i][1].get("direction") == direction:
            return i
    return None


def _cross_check(yolo_event: dict, blob_event: dict, log) -> None:
    """Merge blob speed into yolo_event, warning if they disagree significantly."""
    ys = yolo_event.get("speed_kmh")
    bs = blob_event.get("speed_kmh")

    if bs is None:
        # Blob fired but had no valid speed; counts as confirmation without speed
        yolo_event["source"] = "blob+yolo"
        return

    if ys is None:
        yolo_event["speed_kmh"] = bs
        yolo_event["source"] = "blob+yolo"
        return

    rel_diff = abs(ys - bs) / max(ys, bs)
    if rel_diff > _SPEED_TOLERANCE:
        log.warning(
            "Speed mismatch: yolo=%.1f  blob=%.1f km/h  (%.0f%% diff) — using blob estimate",
            ys, bs, rel_diff * 100,
        )
        yolo_event["speed_kmh"] = bs
        yolo_event["source"] = "blob+yolo(conflict)"
    else:
        yolo_event["speed_kmh"] = round((ys + bs) / 2, 1)
        yolo_event["source"] = "blob+yolo"


def _emit(log, storage, event: dict, frame, clip_recorder, trigger_time: float) -> str:
    """Log, store, trigger clip, and return the banner string."""
    log.info(
        "Count: class=%-12s  dir=%-5s  speed=%s km/h  source=%s",
        event["vehicle_class"],
        event["direction"],
        event.get("speed_kmh"),
        event.get("source", "yolo"),
    )
    storage.log_event(event, frame)
    clip_recorder.trigger(_clip_label(event), trigger_time)
    return _make_banner(event)


def _clip_label(event: dict) -> str:
    speed = event.get("speed_kmh")
    speed_str = f"_{speed}kmh" if speed is not None else ""
    return f"{event['vehicle_class']}_{event['direction']}{speed_str}_{event.get('source', 'yolo')}"


def _make_banner(event: dict) -> str:
    speed = event.get("speed_kmh")
    speed_str = f"  {speed} km/h" if speed is not None else ""
    return f"{event['vehicle_class']} {event['direction']}{speed_str}  [{event.get('source', 'yolo')}]"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_overlay(
    frame,
    cfg: dict,
    tracklets_msg,
    track_speeds: dict = None,
    blob_detector=None,
    banner: str = "",
) -> None:
    if track_speeds is None:
        track_speeds = {}

    line_x = int(cfg["counting_line_x_fraction"] * _FRAME_W)
    cv2.line(frame, (line_x, 0), (line_x, _FRAME_H), (0, 220, 0), 2)

    ref_a = cfg.get("calibration_ref_a")
    ref_b = cfg.get("calibration_ref_b")
    if ref_a and ref_b:
        pa, pb = tuple(ref_a), tuple(ref_b)
        cv2.circle(frame, pa, 5, (255, 255, 0), -1)
        cv2.circle(frame, pb, 5, (255, 255, 0), -1)
        cv2.line(frame, pa, pb, (255, 255, 0), 1)
        dist_m = cfg.get("calibration_distance_m", "?")
        mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2 - 8)
        cv2.putText(frame, f"{dist_m}m", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    if blob_detector is not None:
        _draw_blob_circles(frame, blob_detector)

    if tracklets_msg is not None:
        for t in tracklets_msg.tracklets:
            if t.status not in (
                dai.Tracklet.TrackingStatus.NEW,
                dai.Tracklet.TrackingStatus.TRACKED,
            ):
                continue
            x1, y1, x2, y2 = _denorm_roi(t.roi, _FRAME_W, _FRAME_H)
            vehicle = VEHICLE_LABELS.get(t.label, f"cls{t.label}")
            speed = track_speeds.get(t.id)
            speed_str = f" {speed}km/h" if speed is not None else ""
            label = f"{vehicle} #{t.id}{speed_str}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(frame, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    if banner:
        cv2.putText(frame, banner, (8, _FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, banner, (8, _FRAME_H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1)


def _draw_blob_circles(frame, blob_detector) -> None:
    state = blob_detector.state
    blob_a = blob_detector.blob_a
    blob_b = blob_detector.blob_b

    if state == blob_detector.IDLE:
        color_a = color_b = (128, 128, 128)
    elif state == blob_detector.WAITING_B:
        color_a = (0, 165, 255)
        color_b = (128, 128, 128)
    else:
        color_a = (128, 128, 128)
        color_b = (0, 165, 255)

    cv2.circle(frame, blob_a.center, blob_a.radius, color_a, 2)
    cv2.circle(frame, blob_b.center, blob_b.radius, color_b, 2)


if __name__ == "__main__":
    main()

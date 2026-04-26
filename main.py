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

# Seconds to keep the last-event banner on screen
_BANNER_TTL = 5.0

# Time window for matching a blob event to a YOLO event (seconds)
_BLOB_YOLO_WINDOW = 3.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Count and time vehicles using an OAK-D-Lite camera."
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print events to stdout without writing to the database.",
    )
    p.add_argument(
        "--display",
        action="store_true",
        help="Show a live OpenCV window (requires a display or VNC).",
    )
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
    clip_recorder = ClipRecorder(cfg)

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

    # Recent YOLO events kept for blob↔YOLO correlation: [(time, event_dict)]
    recent_yolo: list = []

    # Banner: last event string + expiry time
    banner_text: str = ""
    banner_expiry: float = 0.0

    try:
        while running:
            now = time.monotonic()

            # ----- grab latest tracklets -----
            tracklets_msg = q_tracklets.tryGet()

            # ----- grab latest preview frame (if display is on) -----
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

                    events = vehicle_tracker.process(
                        tracklets_msg, frame_width=_FRAME_W, frame_height=_FRAME_H
                    )
                    for event in events:
                        event.setdefault("source", "yolo")
                        track_speeds[event["_track_id"]] = event.get("speed_kmh")
                        recent_yolo.append((now, event))
                        _log_and_store(log, storage, event, frame)
                        banner_text, banner_expiry = _make_banner(event), now + _BANNER_TTL
                        clip_recorder.trigger(_clip_label(event), now)

                # Prune stale YOLO events from correlation buffer
                recent_yolo = [(t, e) for t, e in recent_yolo if now - t <= _BLOB_YOLO_WINDOW]

                # ----- Blob detector -----
                if frame is not None:
                    blob_events = blob_detector.process(frame, now)
                    for bevent in blob_events:
                        # Try to correlate with a recent YOLO event
                        match = _find_yolo_match(bevent, recent_yolo, now)
                        if match is not None:
                            # Enrich YOLO event with blob speed if YOLO had none
                            if match.get("speed_kmh") is None:
                                match["speed_kmh"] = bevent["speed_kmh"]
                                match["source"] = "blob+yolo"
                                tid = match.get("_track_id")
                                if tid is not None:
                                    track_speeds[tid] = match["speed_kmh"]
                            # Don't double-log the YOLO event; blob speed is enrichment only
                        else:
                            # Standalone blob event — use "unknown" as class
                            bevent["vehicle_class"] = "unknown"
                            _log_and_store(log, storage, bevent, frame)
                            banner_text, banner_expiry = _make_banner(bevent), now + _BANNER_TTL
                            clip_recorder.trigger(_clip_label(bevent), now)

            # ----- display -----
            if args.display and frame is not None:
                banner = banner_text if now < banner_expiry else ""
                _draw_overlay(
                    frame, cfg, tracklets_msg, track_speeds,
                    blob_detector=blob_detector, banner=banner,
                )
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
# Helpers
# ---------------------------------------------------------------------------

def _log_and_store(log, storage, event: dict, frame) -> None:
    log.info(
        "Count: class=%-12s  dir=%-5s  speed=%s km/h  source=%s",
        event["vehicle_class"],
        event["direction"],
        event.get("speed_kmh"),
        event.get("source", "yolo"),
    )
    storage.log_event(event, frame)


def _clip_label(event: dict) -> str:
    speed = event.get("speed_kmh")
    speed_str = f"_{speed}kmh" if speed is not None else ""
    return f"{event['vehicle_class']}_{event['direction']}{speed_str}_{event.get('source', 'yolo')}"


def _make_banner(event: dict) -> str:
    speed = event.get("speed_kmh")
    speed_str = f"  {speed} km/h" if speed is not None else ""
    return f"{event['vehicle_class']} {event['direction']}{speed_str}  [{event.get('source', 'yolo')}]"


def _find_yolo_match(blob_event: dict, recent_yolo: list, now: float):
    """Return the most recent YOLO event with the same direction, or None."""
    direction = blob_event["direction"]
    for ts, ev in reversed(recent_yolo):
        if ev.get("direction") == direction:
            return ev
    return None


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
    """Draw trigger line, calibration segment, blob circles, bounding boxes, banner."""
    if track_speeds is None:
        track_speeds = {}

    # --- Vertical trigger line (green) --------------------------------------
    line_x = int(cfg["counting_line_x_fraction"] * _FRAME_W)
    cv2.line(frame, (line_x, 0), (line_x, _FRAME_H), (0, 220, 0), 2)

    # --- Calibration reference segment (cyan) --------------------------------
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

    # --- Blob detector circles (state-coloured) ------------------------------
    if blob_detector is not None:
        _draw_blob_circles(frame, blob_detector)

    # --- Bounding boxes with class, track ID, and speed ---------------------
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
            cv2.putText(
                frame, label, (x1, max(y1 - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1,
            )

    # --- Last-event banner ---------------------------------------------------
    if banner:
        cv2.putText(
            frame, banner, (8, _FRAME_H - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
        )
        cv2.putText(
            frame, banner, (8, _FRAME_H - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1,
        )


def _draw_blob_circles(frame, blob_detector) -> None:
    """Draw blob sensor circles with colour indicating state."""
    state = blob_detector.state
    blob_a = blob_detector.blob_a
    blob_b = blob_detector.blob_b

    # IDLE: grey; WAITING_B means A fired (A=orange, B=grey); WAITING_A means B fired
    if state == blob_detector.IDLE:
        color_a = color_b = (128, 128, 128)
    elif state == blob_detector.WAITING_B:
        color_a = (0, 165, 255)   # orange — A fired
        color_b = (128, 128, 128)
    else:  # WAITING_A
        color_a = (128, 128, 128)
        color_b = (0, 165, 255)   # orange — B fired

    cv2.circle(frame, blob_a.center, blob_a.radius, color_a, 2)
    cv2.circle(frame, blob_b.center, blob_b.radius, color_b, 2)


if __name__ == "__main__":
    main()

"""Entry point: argument parsing, device setup, main event loop."""

import argparse
import logging
import signal

import cv2
import depthai as dai
import yaml

from pipeline import VEHICLE_LABELS, build_pipeline
from storage import Storage
from tracker import VehicleTracker, _denorm_roi

_FRAME_W = 640
_FRAME_H = 640
_WIN_NAME = "oak-car-counter"


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

    # build_pipeline returns (pipeline, tracklets_queue, preview_queue).
    # In depthai 3.x, MessageQueues are registered in the pipeline at
    # construction time and become active when dai.Device(pipeline) is opened.
    pipeline, q_tracklets, q_preview = build_pipeline(cfg)
    vehicle_tracker = VehicleTracker(cfg)
    storage = Storage(cfg, dry_run=args.dry_run)

    # Graceful shutdown on SIGINT / SIGTERM
    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    try:
        # SIGTERM is not catchable on Windows
        signal.signal(signal.SIGTERM, _stop)
    except OSError:
        pass

    # In depthai 3.x, pipeline.start() replaces dai.Device(pipeline).
    # Suppress q_preview when display is off to avoid accumulating frames.
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

    try:
        while running:
            # ----- grab latest tracklets -----
            tracklets_msg = q_tracklets.tryGet()

            # ----- grab latest preview frame (if display is on) -----
            frame = None
            if q_preview is not None:
                frame_msg = q_preview.tryGet()
                if frame_msg is not None:
                    frame = frame_msg.getCvFrame()

            # ----- process detections -----
            if tracklets_msg is not None and not paused:
                events = vehicle_tracker.process(
                    tracklets_msg, frame_width=_FRAME_W, frame_height=_FRAME_H
                )
                for event in events:
                    log.info(
                        "Count: class=%-12s  dir=%-5s  speed=%s km/h",
                        event["vehicle_class"],
                        event["direction"],
                        event.get("speed_kmh"),
                    )
                    storage.log_event(event, frame)

            # ----- display -----
            if args.display and frame is not None:
                _draw_overlay(frame, cfg, tracklets_msg)
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
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_overlay(frame, cfg: dict, tracklets_msg) -> None:
    """Draw counting line and bounding boxes onto *frame* in-place."""
    line_y = int(cfg["counting_line_y_fraction"] * _FRAME_H)
    cv2.line(frame, (0, line_y), (_FRAME_W, line_y), (0, 220, 0), 2)

    if tracklets_msg is None:
        return

    for t in tracklets_msg.tracklets:
        if t.status not in (
            dai.Tracklet.TrackingStatus.NEW,
            dai.Tracklet.TrackingStatus.TRACKED,
        ):
            continue

        x1, y1, x2, y2 = _denorm_roi(t.roi, _FRAME_W, _FRAME_H)
        label = VEHICLE_LABELS.get(t.label, f"cls{t.label}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(
            frame,
            f"{label} #{t.id}",
            (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 200, 255),
            1,
        )


if __name__ == "__main__":
    main()

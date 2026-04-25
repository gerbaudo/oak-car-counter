from pathlib import Path

import depthai as dai

# COCO class indices for the vehicle types we track
VEHICLE_LABELS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# YOLOv8n expects 640x640 input
_INPUT_SIZE = 640

# File extensions treated as NNArchive (blob + YOLO config bundled together)
_NNARCHIVE_SUFFIXES = {".nnarchive", ".tar.xz"}


def build_pipeline(cfg: dict):
    """Construct the DepthAI pipeline for depthai 3.x.

    NOTE: This function connects to the OAK-D device (via Camera.build()) to
    query sensor capabilities.  The device must be plugged in before calling.

    Returns (pipeline, tracklets_queue, preview_queue).
    Call pipeline.start() to begin streaming; pipeline.stop() to shut down.

    Graph (NNArchive path — .tar.xz / .nnarchive):
        Camera (640x640 BGR888p output)
            -> DetectionNetwork  (blob + YOLO post-processing from archive)
                .out         -> ObjectTracker.inputDetections
                .passthrough -> ObjectTracker.inputTrackerFrame
                .passthrough -> ObjectTracker.inputDetectionFrame

    Graph (raw .blob / .superblob):
        Camera (640x640 BGR888p output)
            -> NeuralNetwork
                .out        -> DetectionParser  (manually configured)
                                 .out -> ObjectTracker.inputDetections
                .passthrough -> ObjectTracker.inputTrackerFrame
                .passthrough -> ObjectTracker.inputDetectionFrame

    Both graphs:
        ObjectTracker.out                     -> tracklets_queue
        ObjectTracker.passthroughTrackerFrame -> preview_queue
    """
    model_path = Path(cfg["model_blob_path"]).resolve()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Download a YOLOv8n RVC2 NNArchive from https://models.luxonis.com\n"
            "and place it at the path given by model_blob_path in config.yaml."
        )

    pipeline = dai.Pipeline()

    # --- Camera --------------------------------------------------------------
    # In depthai 3.x, Camera.build() must be called before requestOutput() so
    # the node can query the connected sensor's native resolution and capabilities.
    # This requires the OAK-D to be plugged in at pipeline-construction time.
    cam = pipeline.create(dai.node.Camera)
    cam.build()
    cam_out = cam.requestOutput((_INPUT_SIZE, _INPUT_SIZE), dai.ImgFrame.Type.BGR888p)
    # TODO: Camera node in depthai 3.x doesn't expose setFps directly; FPS is
    #       controlled via the sensor control interface (cam.initialControl).
    #       Default FPS should be adequate; lower it here if needed:
    #       cam.initialControl.setFrameSyncMode(...)

    # --- Detection network ---------------------------------------------------
    is_archive = str(model_path).endswith(".tar.xz") \
                 or model_path.suffix == ".nnarchive"

    if is_archive:
        det_out, det_passthrough = _build_archive_detection(
            pipeline, cam_out, model_path
        )
    else:
        det_out, det_passthrough = _build_blob_detection(
            pipeline, cam_out, model_path
        )

    # --- ObjectTracker -------------------------------------------------------
    tracker = pipeline.create(dai.node.ObjectTracker)
    tracker.setDetectionLabelsToTrack(list(VEHICLE_LABELS.keys()))
    # ZERO_TERM_COLOR_HISTOGRAM balances accuracy and CPU cost on the VPU.
    # TODO: try ZERO_TERM_IMAGELESS if colour histogram causes false merges.
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)
    tracker.inputTrackerFrame.setBlocking(False)
    tracker.inputDetectionFrame.setBlocking(False)
    tracker.inputDetections.setBlocking(False)

    det_out.link(tracker.inputDetections)
    det_passthrough.link(tracker.inputTrackerFrame)
    det_passthrough.link(tracker.inputDetectionFrame)

    # --- Output queues (replaces XLinkOut in depthai 3.x) --------------------
    q_tracklets = tracker.out.createOutputQueue(4, False)
    q_preview = tracker.passthroughTrackerFrame.createOutputQueue(1, False)

    return pipeline, q_tracklets, q_preview


# ---------------------------------------------------------------------------
# Detection sub-graphs — each takes cam_out, links it, returns
# (detections_out, frame_passthrough) for ObjectTracker wiring
# ---------------------------------------------------------------------------

def _build_archive_detection(pipeline, cam_out, model_path: Path):
    """DetectionNetwork + NNArchive: all YOLO config is read from the bundle."""
    det = pipeline.create(dai.node.DetectionNetwork)
    det.setNNArchive(dai.NNArchive(str(model_path)))
    det.input.setBlocking(False)
    det.setNumInferenceThreads(2)
    cam_out.link(det.input)
    return det.out, det.passthrough


def _build_blob_detection(pipeline, cam_out, model_path: Path):
    """NeuralNetwork + DetectionParser for raw .blob / .superblob files."""
    nn = pipeline.create(dai.node.NeuralNetwork)
    nn.setBlobPath(str(model_path))
    nn.input.setBlocking(False)
    nn.setNumInferenceThreads(2)
    cam_out.link(nn.input)

    parser = pipeline.create(dai.node.DetectionParser)
    parser.setNNFamily(dai.DetectionNetworkType.YOLO)
    # TODO: confirm subtype matches your specific blob (v5 / v6 / v7 / v8)
    parser.setSubtype("yolov8")
    parser.setInputImageSize(_INPUT_SIZE, _INPUT_SIZE)
    parser.setConfidenceThreshold(0.5)
    parser.setNumClasses(80)
    parser.setCoordinateSize(4)
    parser.setAnchors([])       # YOLOv8 is anchor-free
    parser.setAnchorMasks({})
    parser.setIouThreshold(0.5)
    nn.out.link(parser.input)

    return parser.out, nn.passthrough

# oak-car-counter

A low-cost, privacy-respecting vehicle counter and speed estimator using a [Luxonis OAK-D-Lite](https://docs.luxonis.com/projects/hardware/en/latest/pages/DM9095.html) camera and a Raspberry Pi 4. Detection and tracking run entirely on the OAK's onboard VPU — the Pi only handles counting logic, speed estimation, and logging.

Designed for low-traffic rural settings (≤ a few hundred vehicles/day). No cloud connectivity. No GPU required.

---

## How it works

```
OAK-D-Lite (on-device VPU)          Raspberry Pi 4
──────────────────────────           ──────────────────────────
ColorCamera                          Counting line logic
  → YOLOv8n detection          →     Speed estimation
  → ObjectTracker                    SQLite logging
                                     (optional) OpenCV display
```

The camera is placed indoors behind a window, perpendicular to the road. A configurable virtual counting line triggers an event each time a tracked vehicle crosses it. Speed is estimated from lateral pixel displacement using a user-supplied calibration factor.

---

## Hardware required

| Item | Notes |
|---|---|
| Luxonis OAK-D-Lite | Connected via USB 3 |
| Raspberry Pi 4 B (4 GB recommended) | 64-bit OS |
| USB 3 cable (Type-A to Type-C) | Included with OAK-D-Lite |
| MicroSD card (32 GB+) | Class 10 / A1 or better |

**Camera placement:** mount the OAK-D-Lite so that the road runs roughly left-to-right across the frame. A window sill or a small tripod works well. Avoid shooting through tinted or dirty glass.

---

## Installation

### 1. Flash and configure the Pi

Use **Raspberry Pi OS Lite (64-bit, Bookworm)** via [Raspberry Pi Imager](https://www.raspberrypi.com/software/). Enable SSH and set your Wi-Fi credentials in the imager before flashing.

### 2. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/oak-car-counter.git
cd oak-car-counter
```

### 3. Run the install script

```bash
bash scripts/install.sh
```

This will:
- Install system dependencies (`libusb-1.0`, `python3-venv`, etc.)
- Install the OAK-D udev rules so the device is accessible without `sudo`
- Create a Python virtualenv at `.venv/`
- Install all Python dependencies from `requirements.txt`

> **Reboot or re-plug the OAK-D after udev rules are installed.**

### 4. Download the detection model

The YOLOv8n model must be downloaded separately as a pre-compiled `.blob` file for the OAK's VPU (Marvell MX-series / RVC2).

1. Go to [models.luxonis.com](https://models.luxonis.com)
2. Search for **YOLOv8n**, select target **RVC2** (this covers OAK-D-Lite)
3. Download the `.blob` file
4. Place it at `models/yolov8n.blob` (or update `model_blob_path` in `config.yaml`)

---

## Configuration

All tuneable parameters live in `config.yaml`:

```yaml
# Path to the compiled .blob model file
model_blob_path: models/yolov8n.blob

# Fraction of frame height where the counting line sits (0.0 = top, 1.0 = bottom)
counting_line_y_fraction: 0.5

# Calibration: how many pixels correspond to 1 metre at road level
# See "Calibrating pixels_per_meter" below
pixels_per_meter: 120.0

# Minimum fraction of frame width a vehicle must travel before speed is recorded
min_track_width_fraction: 0.3

# Save a cropped JPEG of each counted vehicle to data/frames/
save_frames: false

# Logging verbosity: DEBUG, INFO, WARNING, ERROR
log_level: INFO
```

---

## Calibrating `pixels_per_meter`

This is the most important setup step for accurate speed readings.

1. Place two objects with a **known distance between them** (e.g. 2 metres apart) on the road surface, in the part of the frame where vehicles will be tracked.
2. Run the app in dry-run mode with display on:
   ```bash
   source .venv/bin/activate
   python main.py --dry-run --display
   ```
3. Pause on a frame that shows both objects clearly (press **Space** to freeze).
4. Note the pixel X-coordinates of each object (printed to console when you click, or measure from a screenshot).
5. Calculate: `pixels_per_meter = pixel_distance / real_distance_metres`
6. Update `config.yaml` with the result.

> **Example:** objects are 2 m apart and appear 240 pixels apart → `pixels_per_meter: 120.0`

Speed accuracy is typically ±10–20%, which is sufficient to distinguish slow from fast but is not suitable for enforcement purposes.

---

## Running

**Dry run (no database writes) — good for testing:**
```bash
source .venv/bin/activate
python main.py --dry-run --display
```

**Normal operation, headless:**
```bash
source .venv/bin/activate
python main.py
```

**With live display (if a monitor or VNC is connected):**
```bash
source .venv/bin/activate
python main.py --display
```

Press **Ctrl+C** to stop cleanly.

---

## Running as a service (optional)

To start automatically on boot:

```bash
sudo cp scripts/oak-car-counter.service /etc/systemd/system/
sudo systemctl enable oak-car-counter
sudo systemctl start oak-car-counter
```

View logs:
```bash
journalctl -u oak-car-counter -f
```

---

## Querying the data

The SQLite database is at `data/counts.db`. The `events` table has these columns:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Auto-increment primary key |
| `timestamp` | TEXT | ISO 8601, UTC |
| `vehicle_class` | TEXT | `car`, `truck`, `bus`, `motorcycle`, `unknown` |
| `direction` | TEXT | `left` or `right` |
| `speed_kmh` | REAL | Estimated speed, or NULL if unreliable |
| `source` | TEXT | `yolo`, `blob`, `blob+yolo`, `blob+yolo(conflict)` |
| `frame_path` | TEXT | Path to annotated JPEG, or NULL |

### Query script

```bash
source .venv/bin/activate

python scripts/query.py            # last 20 events + summary
python scripts/query.py -n 50     # last 50 events
python scripts/query.py -n 0      # all events
python scripts/query.py --hours 6 # last 6 hours
python scripts/query.py --today   # today only
python scripts/query.py --summary # summary stats only (no row listing)
```

### Direct SQL

**Export to CSV:**
```bash
sqlite3 -csv -header data/counts.db "SELECT * FROM events ORDER BY timestamp;" > export.csv
```

**Average speed by class (confirmed detections only):**
```sql
SELECT vehicle_class, ROUND(AVG(speed_kmh), 1) AS avg_kmh, COUNT(*) AS n
FROM events
WHERE speed_kmh IS NOT NULL AND source != 'blob'
GROUP BY vehicle_class;
```

---

## Known limitations

- **Low-light / night:** the OAK-D-Lite RGB sensor is not optimised for night use. Detections will be unreliable after dark without additional IR illumination.
- **Speed accuracy:** lateral speed estimation from a single perpendicular camera is approximate. ±10–20% is realistic.
- **Closely spaced vehicles:** two cars bumper-to-bumper may briefly merge into one detection. At low rural traffic volumes this is rare and has minimal impact on daily counts.
- **Window reflections:** a clean window at an angle that avoids the camera's own reflection gives the best results.

---

## Project structure

```
oak-car-counter/
├── README.md
├── requirements.txt
├── config.yaml
├── main.py          # Entry point, argument parsing, main loop
├── pipeline.py      # DepthAI pipeline construction
├── tracker.py       # Counting line logic, speed estimation
├── storage.py       # SQLite logging, optional frame saving
├── models/
│   └── .gitkeep     # Place your .blob file here
├── data/
│   └── .gitkeep     # Runtime data — gitignored
└── scripts/
    ├── install.sh
    └── oak-car-counter.service
```

---

## License

MIT — see [LICENSE](LICENSE).

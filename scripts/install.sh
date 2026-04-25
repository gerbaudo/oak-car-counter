#!/usr/bin/env bash
# Install system dependencies, OAK udev rules, and Python virtualenv.
# Run from any directory; the script resolves its own location.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "==> Project directory: $PROJECT_DIR"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "==> Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    libusb-1.0-0 \
    python3-venv \
    python3-pip \
    git \
    curl

# ---------------------------------------------------------------------------
# 2. OAK-D udev rules (allows access without sudo)
# ---------------------------------------------------------------------------
echo "==> Installing OAK-D udev rules..."
UDEV_RULE='SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"'
UDEV_FILE="/etc/udev/rules.d/80-oak-d.rules"

if [ ! -f "$UDEV_FILE" ]; then
    echo "$UDEV_RULE" | sudo tee "$UDEV_FILE" > /dev/null
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "    udev rules installed. Re-plug the OAK-D-Lite USB cable now."
else
    echo "    udev rules already present at $UDEV_FILE, skipping."
fi

# ---------------------------------------------------------------------------
# 3. Python virtualenv
# ---------------------------------------------------------------------------
VENV="$PROJECT_DIR/.venv"
echo "==> Creating virtualenv at $VENV ..."
python3 -m venv "$VENV"

echo "==> Installing Python dependencies..."
"$VENV/bin/pip" install --upgrade pip --quiet
"$VENV/bin/pip" install -r "$PROJECT_DIR/requirements.txt"

# ---------------------------------------------------------------------------
# 4. Patch the systemd service file with the real project path and user
# ---------------------------------------------------------------------------
# TODO: verify that SERVICE_USER is correct for your setup.
SERVICE_USER="${SUDO_USER:-$USER}"
SERVICE_SRC="$SCRIPT_DIR/oak-car-counter.service"
SERVICE_DST="/tmp/oak-car-counter.service"

sed \
    -e "s|__PROJECT_DIR__|$PROJECT_DIR|g" \
    -e "s|__USER__|$SERVICE_USER|g" \
    "$SERVICE_SRC" > "$SERVICE_DST"

echo ""
echo "==> Installation complete."
echo ""
echo "Next steps:"
echo "  1. Download a YOLOv8n RVC2 blob from https://models.luxonis.com"
echo "     and place it at:  $PROJECT_DIR/models/yolov8n.blob"
echo "  2. Re-plug (or reboot for) the OAK-D-Lite so udev rules take effect."
echo "  3. Run a test:"
echo "     source $VENV/bin/activate"
echo "     python $PROJECT_DIR/main.py --dry-run --display"
echo ""
echo "  Optional – install as a systemd service:"
echo "     sudo cp $SERVICE_DST /etc/systemd/system/"
echo "     sudo systemctl enable oak-car-counter"
echo "     sudo systemctl start  oak-car-counter"

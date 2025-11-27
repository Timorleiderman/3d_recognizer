#!/bin/bash
# Force GXIVISION camera into 2560x720 stereo mode

echo "Setting GXIVISION camera to stereo mode (2560x720)..."

# Kill any processes using the camera
echo "Checking for processes using camera..."
lsof /dev/video0 2>/dev/null | grep -v COMMAND | awk '{print $2}' | xargs -r kill -9 2>/dev/null

# Wait a moment
sleep 1

# Set format using v4l2-ctl
echo "Setting format to MJPEG 2560x720..."
v4l2-ctl -d /dev/video0 \
  --set-fmt-video=width=2560,height=720,pixelformat=MJPG

# Verify
echo ""
echo "Current settings:"
v4l2-ctl -d /dev/video0 --get-fmt-video

echo ""
echo "Done! Now run: python3 test_stereo_camera.py"

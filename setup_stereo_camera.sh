#!/bin/bash
# Quick setup script for GXIVISION Stereo Camera on Jetson Nano

echo "========================================"
echo "GXIVISION Stereo Camera Quick Setup"
echo "========================================"
echo ""

# Check if camera is connected
if [ ! -e /dev/video0 ]; then
    echo "❌ No camera detected at /dev/video0"
    echo "   Check USB connection"
    exit 1
fi

echo "✓ Camera device found"

# Check permissions
if [ ! -r /dev/video0 ]; then
    echo "⚠️  No read permission for /dev/video0"
    echo "   Adding user to video group..."
    sudo usermod -a -G video $USER
    echo "   Please reboot and run this script again"
    exit 1
fi

echo "✓ Permissions OK"

# Test camera
echo ""
echo "Testing stereo camera..."
python3 test_stereo_camera.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✓ Setup Complete!"
    echo "========================================"
    echo ""
    echo "Run the application with:"
    echo "  ./run_jetson.sh"
    echo ""
    echo "Or directly:"
    echo "  python3 main.py"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Setup encountered issues"
    echo "========================================"
    echo ""
    echo "See STEREO_CAMERA_SETUP.md for troubleshooting"
    echo ""
fi

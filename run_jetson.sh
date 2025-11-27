#!/bin/bash
# Jetson Nano startup script for 3D gesture recognizer
# This script ensures the environment is properly configured

echo "Starting 3D Gesture Recognizer on Jetson Nano..."

# Ensure DISPLAY is set
if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
    echo "Set DISPLAY=:0"
fi

# Allow X11 connections
xhost +local: 2>/dev/null || echo "Note: Could not run xhost (X11 may not be running)"

# Unset conflicting OpenGL platform settings
unset PYOPENGL_PLATFORM

# Check if we need software rendering
if [ "$1" = "--software" ]; then
    echo "Using software rendering (slower but more compatible)"
    export LIBGL_ALWAYS_SOFTWARE=1
fi

# Run the application
echo "Launching application..."
python3 main.py

# Check exit code
if [ $? -ne 0 ]; then
    echo ""
    echo "Application exited with an error."
    echo "If you see OpenGL errors, try running with software rendering:"
    echo "  ./run_jetson.sh --software"
    echo ""
    echo "Or ensure you have the required packages:"
    echo "  sudo apt-get install libgl1-mesa-glx libgl1-mesa-dri mesa-utils"
fi

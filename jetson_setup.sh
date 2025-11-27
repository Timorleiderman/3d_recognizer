#!/bin/bash
# Jetson Nano OpenGL setup script
# This script configures the environment for running OpenGL applications on Jetson Nano

echo "Setting up OpenGL environment for Jetson Nano..."

# Export display
export DISPLAY=:0

# Use EGL for headless rendering (useful for Jetson)
export PYOPENGL_PLATFORM=egl

# Disable vsync for better performance
export __GL_SYNC_TO_VBLANK=0

# Force software rendering if needed (fallback option)
# export LIBGL_ALWAYS_SOFTWARE=1

# Set mesa library path
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH

echo "Environment configured. You can now run: python3 main.py"
echo ""
echo "If you still encounter issues, try running with:"
echo "PYOPENGL_PLATFORM=egl python3 main.py"

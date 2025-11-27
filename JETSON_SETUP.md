# Jetson Nano Setup Guide

This guide will help you resolve OpenGL/GLX context issues on Jetson Nano 2GB.

## Problem
The errors occur because:
1. **EGL/GLX Conflict**: Setting `PYOPENGL_PLATFORM=egl` conflicts with Tkinter's GLX requirements
2. **Limited GLX Support**: Jetson Nano has limited OpenGL/GLX support with the Tkinter backend
3. **NULL FBConfig**: OpenGL context creation fails due to unavailable framebuffer configurations

## Root Cause
The application uses a **Tkinter-based UI**, which requires the **Tkinter backend** for Vispy. Attempting to use PyQt5 backend or EGL platform causes incompatibility because you cannot embed Qt widgets in Tkinter frames.

## Solutions

### Quick Start (Recommended)

Use the provided startup script:

```bash
chmod +x run_jetson.sh
./run_jetson.sh
```

This script automatically:
- Sets up the display environment
- Clears conflicting OpenGL settings
- Launches the application

### If OpenGL Errors Occur

Try software rendering (slower but more compatible):
```bash
./run_jetson.sh --software
```

Or manually:
```bash
LIBGL_ALWAYS_SOFTWARE=1 python3 main.py
```

### Manual Setup

If you prefer to run without the script:

```bash
# Ensure DISPLAY is set
export DISPLAY=:0
xhost +local:

# Clear EGL platform setting (causes conflicts with Tkinter)
unset PYOPENGL_PLATFORM

# Run the application
python3 main.py
```

## Additional System Setup

### Install Required System Packages

```bash
# Install OpenGL libraries
sudo apt-get install -y \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    mesa-utils \
    libgles2-mesa-dev \
    libegl1-mesa-dev

# Install Qt5 dependencies
sudo apt-get install -y \
    qt5-default \
    python3-pyqt5 \
    python3-pyqt5.qtopengl \
    python3-opengl
```

### Verify OpenGL Setup

Test OpenGL availability:
```bash
glxinfo | grep "OpenGL"
```

### Performance Tips for Jetson Nano

1. **Enable maximum performance mode:**
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

2. **Increase swap space** (if you have limited RAM):
```bash
sudo systemctl disable nvzramconfig
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

3. **Reduce visual complexity** if rendering is slow:
   - Lower the point cloud resolution
   - Reduce the update frequency in `main.py` (change `window.after(34, ...)` to higher values)

## Troubleshooting

### Still getting NULL pointer errors?

Try forcing software rendering (slower but more compatible):
```bash
LIBGL_ALWAYS_SOFTWARE=1 python3 main.py
```

### Display issues?

Make sure DISPLAY is set:
```bash
export DISPLAY=:0
xhost +local:
```

### Check your display configuration:
```bash
echo $DISPLAY
```

### Out of Memory errors?

Reduce batch size or model complexity in your training parameters.

## Alternative: Run Headless

If you don't need the GUI, you can modify the code to run in headless mode using:
- Offscreen rendering with EGL
- Save visualizations to images instead of displaying them

## Updated Code

The following files have been updated for Jetson Nano compatibility:

- **`main.py`**: Imports jetson_fix module to apply patches before loading OpenGL libraries. Uses Tkinter backend (required for Tkinter UI).
- **`predict.py`**: Removed conflicting `vispy.use("tkinter")` call. The backend is now set once in main.py before importing other modules.
- **`jetson_fix.py`**: New module that patches OpenGL context creation to handle Jetson Nano's limited GLX support more gracefully. Automatically falls back to software rendering if needed.

These changes ensure that OpenGL works on Jetson Nano despite its limited hardware acceleration support.

## Contact

If you continue experiencing issues, please check:
1. NVIDIA driver version: `cat /usr/local/cuda/version.txt`
2. JetPack version: `sudo apt-cache show nvidia-jetpack`
3. Python version: `python3 --version`

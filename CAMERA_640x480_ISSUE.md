# GXIVISION Camera Not in Stereo Mode (640x480)

Your camera is detected but not outputting the stereo format (2560x720).

## Quick Diagnosis

Run this to see what formats are available:

```bash
python3 diagnose_camera.py
```

## Common Solutions

### 1. Check USB Connection
```bash
# Ensure USB 3.0 (blue port, faster)
lsusb -t

# You should see "5000M" (USB 3.0) not "480M" (USB 2.0)
```

### 2. List Video Devices
```bash
ls -la /dev/video*

# You might see:
# /dev/video0  <- Main stereo device
# /dev/video1  <- Metadata or second lens
```

### 3. Check Available Formats
```bash
# Install v4l-utils if not present
sudo apt-get install v4l-utils

# Check what formats are supported
v4l2-ctl --list-formats-ext -d /dev/video0
```

Look for:
- `[MJPG]` or `[YUYV]` at 2560x720
- Two separate devices at 1280x720 each

### 4. Force Stereo Format

Try setting the format manually:

```bash
# MJPEG format
v4l2-ctl -d /dev/video0 --set-fmt-video=width=2560,height=720,pixelformat=MJPG

# Or YUYV format
v4l2-ctl -d /dev/video0 --set-fmt-video=width=2560,height=720,pixelformat=YUYV

# Verify
v4l2-ctl -d /dev/video0 --get-fmt-video
```

### 5. Try Different Device Index

```bash
# Try video1 instead of video0
python3 -c "
from camera.stereo_camera import StereoCamera
cam = StereoCamera('test', camera_index=1)  # Try 1, 2, 3
cam.start()
"
```

## If Camera Has Two Separate Devices

Some stereo cameras expose left and right as separate video devices.

Check if you have:
- `/dev/video0` - Left camera (1280x720)
- `/dev/video1` - Right camera (1280x720)

If so, you'll need to modify the code to open both devices. Let me know and I can help with that.

## Camera-Specific Settings

Some GXIVISION cameras need:

1. **Windows software to configure** - Camera may remember settings
2. **Hardware switch** - Check if camera has mode selection
3. **Specific driver** - May need kernel module updates

## Workaround: Test Mode

The code will continue in "test mode" with single camera fallback:
- ⚠️  No real depth (simulated only)
- Useful for testing UI
- Not suitable for gesture recognition

To force back to stereo mode later:
```python
# In camera/stereo_camera.py
cam = StereoCamera("GXIVISION", camera_index=0, width=2560, height=720)
```

## Need More Help?

1. Run diagnostics: `python3 diagnose_camera.py`
2. Share the output
3. Note your camera's exact model and any configuration software it came with

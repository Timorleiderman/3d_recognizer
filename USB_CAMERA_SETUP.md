# USB Camera Setup

This application is designed for Intel RealSense L515 depth cameras, but you can use a regular USB webcam for testing.

## Important Limitation

⚠️ **Regular USB cameras only capture 2D RGB images, not 3D depth data.**

The USB camera adapter creates a **simulated** 3D point cloud from the 2D image using brightness as fake depth. This is useful for:
- Testing the UI
- Developing gesture recognition algorithms
- Demo purposes

For **production use**, you need a **real depth camera** like:
- Intel RealSense L515 (recommended)
- Intel RealSense D400 series
- Microsoft Kinect
- Stereolabs ZED

## Quick Start with USB Camera

### 1. Test Your Camera

```bash
python3 test_usb_camera.py
```

This will:
- Detect available USB cameras
- Test the first camera
- Show if it's working properly

### 2. Fix Permissions (if needed)

If you get permission errors:

```bash
# Check camera devices
ls -l /dev/video*

# Add your user to video group
sudo usermod -a -G video $USER

# Reboot for changes to take effect
sudo reboot
```

### 3. Run the Application

```bash
# On Jetson Nano
./run_jetson.sh

# Or directly
python3 main.py
```

The application will automatically:
1. Try to connect to RealSense L515
2. If not found, use USB camera
3. If no USB camera, use mock data

## Selecting a Specific Camera

If you have multiple cameras, you can modify `camera/__init__.py`:

```python
# Use camera at index 1 instead of 0
return USBCamera("USB Camera", camera_index=1)
```

To list available cameras:
```bash
python3 -c "from camera.usb_camera import list_usb_cameras; print(list_usb_cameras())"
```

## Camera Configuration

Edit `camera/usb_camera.py` to adjust:
- **Resolution**: Default is 640x480
- **Point cloud density**: Adjust `scale` variable (higher = fewer points, better performance)
- **Depth simulation**: Modify brightness-to-depth mapping

Example - higher resolution:
```python
cam = USBCamera("USB Camera", camera_index=0, width=1280, height=720)
```

## Troubleshooting

### "No USB cameras found"
- Check if camera is plugged in: `ls /dev/video*`
- Test with other apps: `cheese` or `guvcview`
- Check USB cable and port

### "No valid frame received - image too dark"
- Improve lighting conditions
- Point camera at something with good contrast
- Adjust brightness threshold in `usb_camera.py`

### "Could not open USB camera"
- Camera may be in use by another application
- Check permissions (see step 2 above)
- Try a different camera index

### Poor gesture recognition
This is expected - USB cameras provide fake 3D data. For real gesture recognition, you need a depth camera.

## Using OpenCV to Preview

To see what the camera sees:

```python
import cv2
cap = cv2.VideoCapture(0)  # 0 = first camera
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```

## Upgrading to a Real Depth Camera

For production use, get an Intel RealSense L515:
- ~$250-350 USD
- Provides true 3D point clouds
- Much better accuracy for gesture recognition
- No additional code changes needed

Install RealSense SDK:
```bash
# The pyrealsense2 package is already in requirements.txt
pip3 install pyrealsense2
```

The application will automatically detect and use the RealSense camera when connected.

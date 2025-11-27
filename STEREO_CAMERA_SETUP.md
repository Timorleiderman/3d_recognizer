# GXIVISION Stereo Camera Setup

Your **GXIVISION 3D Stereo VR USB Camera** provides **real depth data** through stereo vision! This is much better than a regular webcam.

## Camera Specifications

- **Model**: GXIVISION 3D Stereo VR USB Camera Module
- **Resolution**: 2560x720 (dual 1280x720)
- **Frame Rate**: 30fps
- **Features**: Synchronized stereo pairs, adjustable baseline
- **Interface**: USB

## Quick Start

### 1. Test Your Stereo Camera

```bash
python3 test_stereo_camera.py
```

This will:
- Detect the camera
- Compute real depth maps from stereo pairs
- Generate 3D point clouds
- Run calibration analysis
- Provide tuning recommendations

### 2. Run the Application

```bash
# On Jetson Nano
./run_jetson.sh

# Or directly
python3 main.py
```

The application **automatically detects** the stereo camera by its 2560x720 resolution and uses real depth computation!

## How It Works

### Stereo Vision Process

1. **Capture**: Both camera lenses capture synchronized frames
2. **Split**: The 2560x720 image is split into left (1280x720) and right (1280x720) views
3. **Stereo Matching**: Algorithm finds corresponding points between left and right images
4. **Disparity Map**: Computes pixel shift (disparity) for each point
5. **Depth Calculation**: Converts disparity to depth using: `depth = (baseline × focal_length) / disparity`
6. **Point Cloud**: Generates 3D coordinates (x, y, z) for each valid pixel

### Key Parameters

**Baseline**: Physical distance between the two camera lenses (typically 60-120mm for this camera)

**Focal Length**: Camera's optical focal length in pixels (~700px typical)

These are configured in `camera/stereo_camera.py`:
```python
baseline_mm=60.0,      # Adjust based on your camera's actual baseline
focal_length_px=700.0  # Calibrate for best results
```

## Optimization & Calibration

### Adjust Baseline

If depth seems incorrect:

1. Measure the physical distance between your camera lenses (in mm)
2. Update in `camera/stereo_camera.py`:
```python
StereoCamera("GXIVISION", baseline_mm=YOUR_MEASUREMENT)
```

### Tune Stereo Matching

Edit `camera/stereo_camera.py` in the `_setup_stereo_matcher()` method:

```python
# For better quality (slower):
self._stereo.setNumDisparities(128)  # Higher = more depth levels
self._stereo.setBlockSize(21)        # Larger = smoother, less detail

# For faster processing (lower quality):
self._stereo.setNumDisparities(64)   # Lower = faster
self._stereo.setBlockSize(11)        # Smaller = faster
```

### Switch to SGBM (Better Quality)

For improved depth quality at the cost of speed, uncomment the SGBM section in `stereo_camera.py`:

```python
# Replace StereoBM with StereoSGBM
self._stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=80,
    blockSize=5,
    # ... rest of parameters
)
```

### Adjust Depth Range

Modify the valid depth range in `_depth_map_to_point_cloud()`:

```python
if z > 2.0:      # Change max range (default 2m)
if z < 0.1:      # Change min range (default 10cm)
```

### Downsampling

Control point cloud density (higher = faster, fewer points):

```python
point_cloud = self._depth_map_to_point_cloud(depth_map, left, downsample=4)
#                                                                        ^
# Change to 2 (more points) or 8 (fewer points)
```

## Troubleshooting

### "No valid depth data"

**Causes:**
- Poor lighting (stereo needs texture to match)
- Lenses not aligned
- One lens obstructed
- Looking at featureless surfaces (white walls, etc.)

**Solutions:**
- Improve lighting
- Point at textured objects
- Clean both lenses
- Check camera alignment

### "Insufficient valid points"

**Causes:**
- Object too close or too far
- Incorrect baseline/focal length settings
- Stereo matcher parameters not optimal

**Solutions:**
- Keep objects between 0.1-1.0m for best results
- Run calibration: `python3 test_stereo_camera.py`
- Adjust baseline_mm parameter
- Tune stereo matcher settings

### Noisy/Speckled Depth

**Solutions:**
```python
# Increase speckle filtering
self._stereo.setSpeckleWindowSize(200)  # Default: 100
self._stereo.setSpeckleRange(64)        # Default: 32
```

### Poor Performance on Jetson Nano

**Solutions:**
1. **Increase downsampling** (fewer points):
   ```python
   downsample=8  # Instead of 4
   ```

2. **Reduce disparity levels**:
   ```python
   self._stereo.setNumDisparities(48)  # Instead of 80
   ```

3. **Use StereoBM** instead of SGBM (already default, it's faster)

4. **Lower resolution** (if camera supports it):
   ```python
   StereoCamera("GXIVISION", width=1280, height=360)
   ```

### USB Bandwidth Issues

If you get dropped frames:

```bash
# Check USB version
lsusb -t

# Ensure camera is on USB 3.0 port (blue connector)
# USB 2.0 may not have enough bandwidth for 2560x720@30fps
```

## Advanced: Camera Calibration

For best accuracy, perform full stereo calibration using OpenCV's calibration tools:

```python
import cv2
import numpy as np

# 1. Print checkerboard pattern (9x6 squares, 25mm each)
# 2. Capture 20+ images of the checkerboard from different angles
# 3. Run calibration to get:
#    - Intrinsic parameters (focal length, principal point)
#    - Extrinsic parameters (rotation, translation)
#    - Distortion coefficients

# Use cv2.stereoCalibrate() and cv2.stereoRectify()
```

See OpenCV documentation: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

## Comparison with RealSense L515

| Feature | GXIVISION Stereo | RealSense L515 |
|---------|-----------------|----------------|
| Technology | Passive stereo | LiDAR |
| Indoor | Good (needs texture) | Excellent |
| Outdoor | Excellent | Good |
| Range | 0.1-5m (adjustable) | 0.25-9m |
| Accuracy | ±5-10mm @ 1m | ±5mm @ 1m |
| Lighting | Needs good lighting | Works in low light |
| Processing | CPU-intensive | Hardware-accelerated |
| Cost | ~$50-100 | ~$250-350 |

## Testing Camera Alignment

Check if left and right images are properly aligned:

```python
python3 -c "
from camera.stereo_camera import StereoCamera
import cv2

cam = StereoCamera('test', 0)
cam.start()

ret, frame = cam._cap.read()
left, right = cam._split_stereo_image(frame)

cv2.imwrite('left.jpg', left)
cv2.imwrite('right.jpg', right)
print('Saved left.jpg and right.jpg - check alignment')

cam.stop()
"
```

The images should show the same scene from slightly different angles. If they're very different, check:
- Camera mounting/orientation
- Synchronization settings
- USB cable quality

## Performance Tips for Jetson Nano

1. **Enable performance mode**:
   ```bash
   sudo nvpmodel -m 0
   sudo jetson_clocks
   ```

2. **Optimize parameters**:
   - downsample=6-8 (fewer points)
   - numDisparities=48-64 (less precision)
   - blockSize=11-15 (faster matching)

3. **Update frame rate in main.py**:
   ```python
   window.after(50, self.update_camera_frame)  # Instead of 34ms
   ```

## Need Help?

If you continue having issues:
1. Run the test script: `python3 test_stereo_camera.py`
2. Check the calibration results
3. Try the recommended parameter adjustments
4. Verify both camera lenses are working

For best results, ensure:
- ✓ Good lighting conditions
- ✓ Textured surfaces in view
- ✓ Clean lenses
- ✓ USB 3.0 connection
- ✓ Objects at 0.2-1.5m distance

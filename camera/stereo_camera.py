"""
Stereo Camera adapter for GXIVISION 3D Stereo VR USB Camera
This camera has dual lenses for true stereo vision and depth estimation.

Model: GXIVISION 3D Stereo VR USB Camera Module 720P
Resolution: 2560x720 (1280x720 per eye)
Frame rate: 30fps
Features: Synchronized stereo pairs, adjustable baseline
"""

from typing import Tuple, Dict, List
import numpy as np
import cv2
from .base_camera import Camera


class StereoCamera(Camera):
    """
    Stereo camera that computes real depth from dual synchronized camera feeds.
    Uses stereo matching to generate depth maps and 3D point clouds.
    """
    
    def __init__(
            self,
            name: str,
            camera_index: int = 0,
            width: int = 2560,
            height: int = 720,
            baseline_mm: float = 85.0,  # Measured: 8.5cm baseline
            focal_length_px: float = 350.0,  # Optimized for 2.4mm 125° lens
    ):
        """
        Initialize stereo camera
        
        Args:
            name: Camera name
            camera_index: Camera device index (usually 0)
            width: Total image width (2560 for dual 1280x720)
            height: Image height (720)
            baseline_mm: Distance between left and right cameras in mm
            focal_length_px: Focal length in pixels (calibrate for best results)
        """
        super().__init__(name)
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._baseline_mm = baseline_mm / 1000.0  # Convert to meters
        self._focal_length_px = focal_length_px
        self._cap = None
        
        # Initialize stereo matcher
        self._setup_stereo_matcher()
        
    def _setup_stereo_matcher(self):
        """Configure stereo block matching for depth computation"""
        # Use SGBM for better quality with wide-angle lens
        # StereoBM struggles with wide FOV and distortion
        self._stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=96,  # Higher for wide-angle lens (must be divisible by 16)
            blockSize=5,        # Smaller block for detail
            P1=8 * 3 * 5**2,    # Smoothness penalty
            P2=32 * 3 * 5**2,   # Smoothness penalty
            disp12MaxDiff=1,
            uniquenessRatio=5,  # Lower for more matches
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # Better quality
        )
        
        # Alternative: Use StereoSGBM for better quality (slower)
        # self._stereo = cv2.StereoSGBM_create(
        #     minDisparity=0,
        #     numDisparities=80,
        #     blockSize=5,
        #     P1=8 * 3 * 5**2,
        #     P2=32 * 3 * 5**2,
        #     disp12MaxDiff=1,
        #     uniquenessRatio=10,
        #     speckleWindowSize=100,
        #     speckleRange=32
        # )
        
    def start(self) -> None:
        """Start capturing from stereo camera"""
        if self._running:
            return
        
        # Open with V4L2 backend for better format control on Linux
        try:
            self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_V4L2)
        except:
            self._cap = cv2.VideoCapture(self._camera_index)
        
        if not self._cap.isOpened():
            raise Exception(f"Could not open stereo camera at index {self._camera_index}")
        
        # Check current settings first
        current_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        current_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera opened with: {current_width}x{current_height}")
        
        # Always try to set to requested resolution
        if current_width != self._width or current_height != self._height:
            print(f"Setting to {self._width}x{self._height}...")
            
            # Set MJPEG format first
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            
            # Set resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Verify it worked
            new_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            new_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"After setting: {new_width}x{new_height}")
        else:
            print("Resolution already correct!")
        
        # Verify resolution
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        print(f"Camera format: {fourcc_str}, Resolution: {actual_width}x{actual_height}")
        
        if actual_width != self._width or actual_height != self._height:
            print(f"Warning: Requested {self._width}x{self._height}, got {actual_width}x{actual_height}")
            
            # Check if this might be a dual-camera setup on separate devices
            if actual_width == 640 and actual_height == 480:
                print("\n⚠️  Camera not in stereo mode!")
                print("\nDiagnostics:")
                print("  Run: python3 diagnose_camera.py")
                print("\nQuick fixes to try:")
                print("  1. Ensure camera is USB 3.0 connected (blue port)")
                print("  2. Try: v4l2-ctl -d /dev/video0 --set-fmt-video=width=2560,height=720,pixelformat=MJPG")
                print("  3. Check if camera has separate video devices:")
                print("     ls -la /dev/video*")
                print("\nContinuing with test mode (limited functionality)...")
                
                # Adjust parameters for lower resolution
                # Scale down focal length proportionally
                self._focal_length_px = self._focal_length_px * (actual_width / 1280.0)
                
            self._width = actual_width
            self._height = actual_height
        
        print(f"Stereo Camera started: {self._width}x{self._height} @ 30fps")
        print(f"Baseline: {self._baseline_mm*1000:.1f}mm, Focal length: {self._focal_length_px:.1f}px")
        super().start()
    
    def stop(self) -> None:
        """Stop capturing from stereo camera"""
        if not self._running:
            return
            
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        super().stop()
    
    @property
    def device_connected(self) -> bool:
        """Check if camera is connected"""
        return self._cap is not None and self._cap.isOpened()
    
    def _split_stereo_image(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split the combined stereo image into left and right views.
        The GXIVISION camera provides side-by-side images.
        
        Args:
            frame: Combined stereo image (2560x720) or fallback single image
            
        Returns:
            (left_image, right_image) each 1280x720 or simulated pair
        """
        height, width = frame.shape[:2]
        
        # Normal stereo mode - side by side
        if width >= 1280:
            half_width = width // 2
            left = frame[:, :half_width]
            right = frame[:, half_width:]
        else:
            # Fallback mode for testing - simulate stereo from single image
            # Shift the image slightly to create a pseudo-stereo pair
            print("⚠️  Using single camera fallback mode (no real depth)")
            shift = 20  # Pixel shift to simulate parallax
            left = frame
            right = np.roll(frame, shift, axis=1)  # Shift horizontally
            
        return left, right
    
    def _compute_depth_map(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Compute depth map from stereo pair using block matching.
        
        Args:
            left: Left camera image (grayscale)
            right: Right camera image (grayscale)
            
        Returns:
            Depth map in meters
        """
        # Convert to grayscale if needed
        if len(left.shape) == 3:
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left
            right_gray = right
        
        # Compute disparity
        disparity = self._stereo.compute(left_gray, right_gray)
        
        # Convert to float and normalize
        disparity = disparity.astype(np.float32) / 16.0
        
        # Convert disparity to depth: depth = (baseline * focal_length) / disparity
        # Avoid division by zero
        depth_map = np.zeros_like(disparity, dtype=np.float32)
        valid_pixels = disparity > 0
        depth_map[valid_pixels] = (self._baseline_mm * self._focal_length_px) / disparity[valid_pixels]
        
        return depth_map
    
    def _depth_map_to_point_cloud(
        self, 
        depth_map: np.ndarray, 
        left_image: np.ndarray,
        downsample: int = 4
    ) -> np.ndarray:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            depth_map: Depth in meters for each pixel
            left_image: Left camera image for reference
            downsample: Downsampling factor for performance (higher = fewer points)
            
        Returns:
            Point cloud as (N, 3) array with x, y, z coordinates
        """
        height, width = depth_map.shape
        
        # Downsample for performance
        depth_small = depth_map[::downsample, ::downsample]
        h, w = depth_small.shape
        
        # Camera intrinsics (center point and focal length)
        cx = width / 2.0
        cy = height / 2.0
        fx = self._focal_length_px
        fy = self._focal_length_px
        
        points = []
        
        for v in range(h):
            for u in range(w):
                # Get depth
                z = depth_small[v, u]
                
                # Filter invalid depths (more permissive for testing)
                if z <= 0 or z > 5.0:  # Max 5 meters
                    continue
                if z < 0.05:  # Min 5cm
                    continue
                
                # Convert pixel coordinates to 3D
                # Scale back to original coordinates
                x_pixel = u * downsample
                y_pixel = v * downsample
                
                # 3D coordinates in camera frame
                x = (x_pixel - cx) * z / fx
                y = (y_pixel - cy) * z / fy
                
                points.append([x, y, z])
        
        if len(points) == 0:
            raise Exception("No valid depth data - check lighting and camera alignment")
        
        point_cloud = np.array(points, dtype=np.float32)
        
        # Filter to region of interest (more permissive for initial testing)
        mask = (point_cloud[:, 2] > 0.05) & (point_cloud[:, 2] < 3.0)
        point_cloud = point_cloud[mask]
        
        # print(f"  Point cloud: {len(point_cloud)} points after filtering")
        
        if len(point_cloud) < 100:
            raise Exception(f"Insufficient valid points in depth range (got {len(point_cloud)})")
        
        return point_cloud
    
    def get(self, timeout_ms: int = 200) -> np.ndarray:
        """
        Get the latest point cloud from the stereo camera.
        
        Returns:
            Point cloud as numpy array of shape (N, 3) with x, y, z coordinates
        """
        if not self._running:
            raise Exception("Stereo camera is not running.")
        
        if self._cap is None or not self._cap.isOpened():
            raise Exception("Stereo camera is not connected.")
        
        # Read frame from camera
        ret, frame = self._cap.read()
        
        if not ret or frame is None:
            raise Exception("No valid frame received from stereo camera.")
        
        # Split into left and right images
        left, right = self._split_stereo_image(frame)
        
        # Compute depth map
        depth_map = self._compute_depth_map(left, right)
        
        # Debug: Check depth map statistics (comment out for production)
        # valid_depth = depth_map[depth_map > 0]
        # if len(valid_depth) > 0:
        #     print(f"  Depth map: {len(valid_depth)} valid pixels, "
        #           f"range [{valid_depth.min():.2f}, {valid_depth.max():.2f}]m")
        # else:
        #     print("  Depth map: No valid depth pixels!")
        
        # Convert to point cloud (downsample=8 for better Jetson Nano performance)
        point_cloud = self._depth_map_to_point_cloud(depth_map, left, downsample=8)
        
        self._last_cloud = point_cloud
        
        return point_cloud
    
    def calibrate(self, num_samples: int = 10) -> Dict:
        """
        Perform basic calibration by analyzing several frames.
        This helps determine optimal baseline and focal length parameters.
        
        Args:
            num_samples: Number of frames to analyze
            
        Returns:
            Dictionary with calibration statistics
        """
        if not self._running:
            raise Exception("Camera must be started before calibration")
        
        print(f"Collecting {num_samples} samples for calibration...")
        disparities = []
        
        for i in range(num_samples):
            ret, frame = self._cap.read()
            if ret:
                left, right = self._split_stereo_image(frame)
                left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                disparity = self._stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
                valid = disparity[disparity > 0]
                if len(valid) > 0:
                    disparities.extend(valid.flatten())
                print(f"  Sample {i+1}/{num_samples} - {len(valid)} valid pixels")
        
        if not disparities:
            return {"error": "No valid disparity data collected"}
        
        disparities = np.array(disparities)
        
        stats = {
            "mean_disparity": float(np.mean(disparities)),
            "median_disparity": float(np.median(disparities)),
            "min_disparity": float(np.min(disparities)),
            "max_disparity": float(np.max(disparities)),
            "std_disparity": float(np.std(disparities)),
            "num_samples": len(disparities)
        }
        
        print("\nCalibration results:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        
        return stats


if __name__ == '__main__':
    # Test stereo camera
    print("Testing GXIVISION Stereo Camera...")
    
    try:
        cam = StereoCamera("GXIVISION Stereo", camera_index=0)
        cam.start()
        
        print("\nCapturing test frames...")
        for i in range(5):
            try:
                cloud = cam.get()
                print(f"Frame {i+1}: {len(cloud)} points, "
                      f"X: [{cloud[:, 0].min():.3f}, {cloud[:, 0].max():.3f}], "
                      f"Y: [{cloud[:, 1].min():.3f}, {cloud[:, 1].max():.3f}], "
                      f"Z: [{cloud[:, 2].min():.3f}, {cloud[:, 2].max():.3f}]")
            except Exception as e:
                print(f"Frame {i+1}: Error - {e}")
        
        # Run calibration
        print("\n" + "="*50)
        cam.calibrate(num_samples=10)
        
        cam.stop()
        print("\n✓ Stereo camera test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

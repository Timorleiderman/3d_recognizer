"""
Stereo Camera adapter for GXIVISION 3D Stereo VR USB Camera
This camera has dual lenses for true stereo vision and depth estimation.

Model: GXIVISION 3D Stereo VR USB Camera Module 720P
Resolution: 2560x720 (1280x720 per eye)
Frame rate: 30fps
Features: Synchronized stereo pairs, adjustable baseline
"""

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
            baseline_mm: float = 60.0,  # Adjust based on your camera's baseline
            focal_length_px: float = 700.0,  # Approximate, needs calibration
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
        # Using StereoBM (faster) - can switch to StereoSGBM for better quality
        self._stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
        
        # Tune these parameters for your specific camera
        self._stereo.setMinDisparity(0)
        self._stereo.setNumDisparities(80)  # Must be divisible by 16
        self._stereo.setBlockSize(15)  # Odd number, 5-21 typical
        self._stereo.setSpeckleWindowSize(100)
        self._stereo.setSpeckleRange(32)
        self._stereo.setDisp12MaxDiff(1)
        
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
            
        self._cap = cv2.VideoCapture(self._camera_index)
        
        if not self._cap.isOpened():
            raise Exception(f"Could not open stereo camera at index {self._camera_index}")
        
        # Set resolution for dual camera (2560x720)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Verify resolution
        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if actual_width != self._width or actual_height != self._height:
            print(f"Warning: Requested {self._width}x{self._height}, got {actual_width}x{actual_height}")
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
    
    def _split_stereo_image(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Split the combined stereo image into left and right views.
        The GXIVISION camera provides side-by-side images.
        
        Args:
            frame: Combined stereo image (2560x720)
            
        Returns:
            (left_image, right_image) each 1280x720
        """
        half_width = frame.shape[1] // 2
        left = frame[:, :half_width]
        right = frame[:, half_width:]
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
                
                # Filter invalid depths
                if z <= 0 or z > 2.0:  # Max 2 meters
                    continue
                if z < 0.1:  # Min 10cm
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
        
        # Filter to region of interest (similar to RealSense range)
        mask = (point_cloud[:, 2] > 0.05) & (point_cloud[:, 2] < 0.6)
        point_cloud = point_cloud[mask]
        
        if len(point_cloud) < 100:
            raise Exception("Insufficient valid points in depth range")
        
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
        
        # Convert to point cloud
        point_cloud = self._depth_map_to_point_cloud(depth_map, left, downsample=4)
        
        self._last_cloud = point_cloud
        
        return point_cloud
    
    def calibrate(self, num_samples: int = 10) -> dict:
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

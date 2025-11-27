"""
USB Camera adapter for standard webcams
Note: Regular USB cameras only provide 2D RGB images, not 3D point clouds.
This adapter creates a simulated point cloud from the 2D image for testing purposes.
For production use, you need a depth camera like Intel RealSense.
"""

from typing import List
import numpy as np
import cv2
from .base_camera import Camera


class USBCamera(Camera):
    """
    USB Camera that simulates a point cloud from 2D images.
    This is useful for testing the UI without a RealSense camera.
    """
    
    def __init__(
            self,
            name: str,
            camera_index: int = 0,
            width: int = 640,
            height: int = 480,
    ):
        """
        Initialize USB camera
        
        Args:
            name: Camera name
            camera_index: Camera device index (0 for default camera, 1 for second, etc.)
            width: Image width
            height: Image height
        """
        super().__init__(name)
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._cap = None
        
    def start(self) -> None:
        """Start capturing from USB camera"""
        if self._running:
            return
            
        self._cap = cv2.VideoCapture(self._camera_index)
        
        if not self._cap.isOpened():
            raise Exception(f"Could not open USB camera at index {self._camera_index}")
        
        # Set resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        
        print(f"USB Camera {self._camera_index} started: {self._width}x{self._height}")
        super().start()
    
    def stop(self) -> None:
        """Stop capturing from USB camera"""
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
    
    def _image_to_point_cloud(self, image: np.ndarray) -> np.ndarray:
        """
        Convert 2D image to simulated 3D point cloud.
        Uses brightness to simulate depth (darker = closer, brighter = farther).
        
        This is NOT real 3D data - just a visualization hack for testing.
        """
        # Convert to grayscale for depth estimation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Downsample for performance (create fewer points)
        scale = 4
        small_gray = cv2.resize(gray, (gray.shape[1]//scale, gray.shape[0]//scale))
        
        height, width = small_gray.shape
        points = []
        
        # Create point cloud from image
        for y in range(height):
            for x in range(width):
                # Normalize coordinates to [-0.3, 0.3] range
                px = (x / width - 0.5) * 0.6
                py = (0.5 - y / height) * 0.6  # Flip Y axis
                
                # Use brightness as simulated depth (0.1 to 0.5 meters)
                brightness = small_gray[y, x] / 255.0
                pz = 0.2 + brightness * 0.3
                
                # Only include points that are reasonably bright (not pure black)
                if brightness > 0.1:
                    points.append([px, py, pz])
        
        point_cloud = np.array(points, dtype=np.float32)
        
        if len(point_cloud) < 100:
            raise Exception("No valid frame received - image too dark.")
        
        return point_cloud
    
    def get(self, timeout_ms: int = 200) -> np.ndarray:
        """
        Get the latest simulated point cloud from the USB camera.
        
        Returns:
            Point cloud as numpy array of shape (N, 3) with x, y, z coordinates
        """
        if not self._running:
            raise Exception("USB camera is not running.")
        
        if self._cap is None or not self._cap.isOpened():
            raise Exception("USB camera is not connected.")
        
        # Read frame from camera
        ret, frame = self._cap.read()
        
        if not ret or frame is None:
            raise Exception("No valid frame received from USB camera.")
        
        # Convert image to point cloud
        point_cloud = self._image_to_point_cloud(frame)
        self._last_cloud = point_cloud
        
        return point_cloud


def list_usb_cameras(max_test: int = 10) -> List[int]:
    """
    List available USB camera indices.
    
    Args:
        max_test: Maximum number of camera indices to test
        
    Returns:
        List of available camera indices
    """
    available = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available


if __name__ == '__main__':
    # Test USB camera detection
    print("Detecting USB cameras...")
    cameras = list_usb_cameras()
    
    if cameras:
        print(f"Found {len(cameras)} camera(s) at indices: {cameras}")
        print("\nTesting first camera...")
        
        cam = USBCamera("USB Test", camera_index=cameras[0])
        cam.start()
        
        try:
            cloud = cam.get()
            print(f"Successfully captured point cloud with {len(cloud)} points")
            print(f"Point cloud shape: {cloud.shape}")
            print(f"Z range: {cloud[:, 2].min():.3f} to {cloud[:, 2].max():.3f}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cam.stop()
    else:
        print("No USB cameras found!")

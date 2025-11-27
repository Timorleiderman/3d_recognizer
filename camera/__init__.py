from pathlib import Path

from dataset import Dataset
from .realsense_camera import RealsenseCamera
from .mock_camera import MockRealsenseCamera
from .stereo_camera import StereoCamera
from .usb_camera import USBCamera, list_usb_cameras
from .base_camera import Camera

import pyrealsense2.pyrealsense2 as rs


def auto_connect_camera() -> Camera:
    """
    Automatically connect to the best available camera.
    Priority: RealSense L515 > Stereo Camera > USB Camera > Mock Camera
    """
    # Try RealSense camera first
    try:
        context = rs.context()
        for device in context.query_devices():
            if device.get_info(rs.camera_info.name) == "Intel RealSense L515":
                serial = device.get_info(rs.camera_info.serial_number)
                print(f"Connected to RealSense L515: {serial}")
                return RealsenseCamera(serial, serial)
    except Exception as e:
        print(f"RealSense not available: {e}")
    
    # Try stereo camera (GXIVISION or similar)
    # Stereo cameras typically provide 2560x720 resolution (dual 1280x720)
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Detect stereo camera by resolution pattern
            if width >= 2560 and height == 720:
                print(f"Detected stereo camera: {width}x{height}")
                print("Using real depth from stereo vision")
                return StereoCamera("Stereo Camera", camera_index=0, 
                                   width=width, height=height)
            else:
                # Regular USB camera
                print(f"Detected USB camera: {width}x{height}")
                print("WARNING: USB camera provides simulated 3D data, not real depth.")
                return USBCamera("USB Camera", camera_index=0)
    except Exception as e:
        print(f"Camera detection failed: {e}")
    
    # Fall back to mock camera
    print("Using mock camera with test data")
    return MockRealsenseCamera("mock", Dataset(
        Path("__file__").parent / "data" / "mock", only_annotated=False
    ))


__all__ = [
    "Camera",
    "auto_connect_camera",
    "StereoCamera",
    "USBCamera",
    "list_usb_cameras"
]
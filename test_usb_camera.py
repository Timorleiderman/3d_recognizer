#!/usr/bin/env python3
"""
Test USB camera detection and functionality
Run this to check if your USB camera is detected
"""

import sys
from camera.usb_camera import list_usb_cameras, USBCamera

def main():
    print("=" * 50)
    print("USB Camera Detection Test")
    print("=" * 50)
    
    # List available cameras
    print("\nDetecting USB cameras...")
    cameras = list_usb_cameras(max_test=10)
    
    if not cameras:
        print("❌ No USB cameras found!")
        print("\nTroubleshooting:")
        print("1. Check if camera is plugged in")
        print("2. Check camera permissions: ls -l /dev/video*")
        print("3. Add your user to video group: sudo usermod -a -G video $USER")
        print("4. Reboot after adding to group")
        return 1
    
    print(f"✓ Found {len(cameras)} camera(s) at indices: {cameras}")
    
    # Test first camera
    print(f"\nTesting camera at index {cameras[0]}...")
    try:
        cam = USBCamera("Test Camera", camera_index=cameras[0])
        cam.start()
        
        print("✓ Camera started successfully")
        
        # Try to capture a few frames
        for i in range(3):
            try:
                cloud = cam.get(timeout_ms=1000)
                print(f"  Frame {i+1}: {len(cloud)} points, "
                      f"Z range: [{cloud[:, 2].min():.3f}, {cloud[:, 2].max():.3f}]")
            except Exception as e:
                print(f"  Frame {i+1}: Error - {e}")
        
        cam.stop()
        print("\n✓ USB camera test completed successfully!")
        print("\nYou can now run: python3 main.py")
        print("The application will automatically use your USB camera.")
        return 0
        
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

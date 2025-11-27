#!/usr/bin/env python3
"""
Test GXIVISION Stereo Camera
This script helps you verify and calibrate your stereo camera
"""

import sys
import numpy as np

def test_stereo_camera():
    """Test the GXIVISION stereo camera"""
    print("=" * 60)
    print("GXIVISION Stereo Camera Test")
    print("=" * 60)
    
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV (cv2) not installed!")
        print("Install with: pip3 install opencv-python")
        return 1
    
    print("\n1. Detecting camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera at index 0")
        print("\nTroubleshooting:")
        print("  1. Check if camera is plugged in: ls /dev/video*")
        print("  2. Try different index: ls -la /dev/video*")
        print("  3. Check permissions: sudo usermod -a -G video $USER")
        return 1
    
    # Get resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"✓ Camera detected: {width}x{height} @ {fps}fps")
    
    # Check if it's a stereo camera
    if width < 2560 or height != 720:
        print(f"\n⚠️  WARNING: Expected 2560x720 for GXIVISION stereo camera")
        print(f"   Got: {width}x{height}")
        print("\n   This might not be the stereo camera, or resolution needs adjustment.")
        print("   Continuing anyway...")
    
    cap.release()
    
    print("\n2. Testing stereo camera with real depth computation...")
    
    try:
        from camera.stereo_camera import StereoCamera
        
        cam = StereoCamera("GXIVISION", camera_index=0, width=width, height=height)
        cam.start()
        
        print("✓ Stereo camera started")
        print("\nCapturing and processing frames...")
        print("(This will take a moment - computing depth maps)\n")
        
        success_count = 0
        for i in range(5):
            try:
                cloud = cam.get(timeout_ms=1000)
                print(f"✓ Frame {i+1}: {len(cloud):,} points")
                print(f"    X: [{cloud[:, 0].min():6.3f}, {cloud[:, 0].max():6.3f}] m")
                print(f"    Y: [{cloud[:, 1].min():6.3f}, {cloud[:, 1].max():6.3f}] m")
                print(f"    Z: [{cloud[:, 2].min():6.3f}, {cloud[:, 2].max():6.3f}] m")
                success_count += 1
            except Exception as e:
                print(f"❌ Frame {i+1}: {e}")
        
        if success_count == 0:
            print("\n❌ No frames captured successfully")
            print("\nTroubleshooting:")
            print("  1. Ensure good lighting - stereo needs texture")
            print("  2. Point camera at objects 0.1-2.0 meters away")
            print("  3. Check that both lenses are clean and unobstructed")
            cam.stop()
            return 1
        
        print(f"\n✓ {success_count}/5 frames captured successfully")
        
        # Run calibration
        print("\n" + "=" * 60)
        print("3. Running calibration analysis...")
        print("=" * 60)
        print("Point the camera at a textured scene 0.5-1.0m away")
        print("Collecting samples...\n")
        
        stats = cam.calibrate(num_samples=10)
        
        if "error" in stats:
            print(f"\n⚠️  Calibration warning: {stats['error']}")
        else:
            print("\n✓ Calibration complete!")
            print("\nRecommended adjustments for camera/stereo_camera.py:")
            
            # Provide recommendations based on disparity
            mean_disp = stats['mean_disparity']
            if mean_disp < 10:
                print("  - Increase baseline_mm (cameras too close)")
            elif mean_disp > 100:
                print("  - Decrease baseline_mm (cameras too far apart)")
            else:
                print("  - Current baseline seems reasonable")
            
            if stats['std_disparity'] > 50:
                print("  - High variation - improve lighting or reduce motion")
        
        cam.stop()
        
        print("\n" + "=" * 60)
        print("✓ STEREO CAMERA TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nYou can now run: python3 main.py")
        print("or: ./run_jetson.sh")
        print("\nThe application will automatically detect and use your stereo camera.")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Stereo camera test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(test_stereo_camera())

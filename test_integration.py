#!/usr/bin/env python3
"""
Simple test to verify stereo camera integration with the app
"""

import sys
import numpy as np

print("Testing stereo camera integration...")
print("=" * 60)

# Test 1: Import camera module
print("\n1. Testing camera import...")
try:
    from camera import auto_connect_camera
    print("   ✓ Camera module imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Connect camera
print("\n2. Connecting to camera...")
try:
    camera = auto_connect_camera()
    print(f"   ✓ Connected: {camera.name}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Start camera
print("\n3. Starting camera...")
try:
    camera.start()
    print("   ✓ Camera started")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Get point clouds
print("\n4. Capturing point clouds...")
try:
    for i in range(3):
        cloud = camera.get(timeout_ms=1000)
        print(f"   Frame {i+1}: {len(cloud)} points, "
              f"Z range: [{cloud[:, 2].min():.2f}, {cloud[:, 2].max():.2f}]m")
    print("   ✓ Point cloud capture working")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    camera.stop()
    sys.exit(1)

# Test 5: Check point cloud properties
print("\n5. Analyzing point cloud...")
try:
    cloud = camera.get()
    print(f"   Shape: {cloud.shape}")
    print(f"   Data type: {cloud.dtype}")
    print(f"   X range: [{cloud[:, 0].min():.2f}, {cloud[:, 0].max():.2f}]")
    print(f"   Y range: [{cloud[:, 1].min():.2f}, {cloud[:, 1].max():.2f}]")
    print(f"   Z range: [{cloud[:, 2].min():.2f}, {cloud[:, 2].max():.2f}]")
    
    # Check if points are reasonable for hand gestures
    points_in_range = np.sum((cloud[:, 2] > 0.1) & (cloud[:, 2] < 1.0))
    print(f"   Points in gesture range (0.1-1.0m): {points_in_range}")
    
    if points_in_range > 1000:
        print("   ✓ Good point density for gesture recognition")
    else:
        print("   ⚠ Low point count - move closer or improve lighting")
    
except Exception as e:
    print(f"   ✗ Failed: {e}")
    camera.stop()
    sys.exit(1)

camera.stop()

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nYour stereo camera is properly integrated.")
print("\nTo use the application:")
print("1. Run: python3 main.py")
print("2. The LEFT panel shows live camera view")
print("3. Click 'Capture' button to save a gesture")
print("4. The MIDDLE panel shows the captured gesture")
print("5. Use mouse to rotate the 3D view")
print("\nTips:")
print("- Hold your hand 30-50cm from camera")
print("- Ensure good lighting")
print("- Move hand slowly for better tracking")

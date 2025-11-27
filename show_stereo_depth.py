#!/usr/bin/env python3
"""
Visualize stereo depth computation in real-time
Shows left image, right image, and computed depth map side-by-side
"""

import cv2
import numpy as np
from camera.stereo_camera import StereoCamera

def visualize_stereo_depth():
    """Show real-time stereo depth computation"""
    
    print("Starting stereo depth visualization...")
    print("This proves both cameras are being used!")
    print("Press 'q' to quit")
    
    cam = StereoCamera("GXIVISION", camera_index=0, width=2560, height=720)
    cam.start()
    
    print("\nYou should see 3 panels:")
    print("  LEFT: Left camera view")
    print("  RIGHT: Right camera view (notice slight parallax)")
    print("  DEPTH: Computed depth map (white=close, black=far)")
    
    while True:
        try:
            # Get raw frame
            ret, frame = cam._cap.read()
            if not ret:
                continue
            
            # Split stereo pair
            left, right = cam._split_stereo_image(frame)
            
            # Compute depth
            left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            
            disparity = cam._stereo.compute(left_gray, right_gray)
            
            # Normalize disparity for visualization
            disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disparity_vis = disparity_vis.astype(np.uint8)
            disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
            
            # Resize for display
            scale = 0.5
            left_small = cv2.resize(left, None, fx=scale, fy=scale)
            right_small = cv2.resize(right, None, fx=scale, fy=scale)
            depth_small = cv2.resize(disparity_color, None, fx=scale, fy=scale)
            
            # Add labels
            cv2.putText(left_small, "LEFT CAMERA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(right_small, "RIGHT CAMERA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(depth_small, "DEPTH MAP", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Stack horizontally
            combined = np.hstack([left_small, right_small, depth_small])
            
            cv2.imshow('Stereo Vision - BOTH CAMERAS ACTIVE', combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        except Exception as e:
            print(f"Error: {e}")
            break
    
    cam.stop()
    cv2.destroyAllWindows()
    print("\nAs you can see:")
    print("✓ LEFT camera captures scene")
    print("✓ RIGHT camera captures same scene from different angle")
    print("✓ DEPTH computed by comparing the two images")
    print("✓ The 3D point cloud comes from this depth data!")

if __name__ == '__main__':
    try:
        visualize_stereo_depth()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

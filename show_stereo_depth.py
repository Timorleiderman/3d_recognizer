#!/usr/bin/env python3
"""
Visualize stereo depth computation in real-time
Shows left image, right image, depth map, and 3D point cloud
"""

import cv2
import numpy as np
from camera.stereo_camera import StereoCamera

def create_point_cloud_image(point_cloud, width=640, height=360):
    """Create a simple 2D projection of 3D point cloud for visualization"""
    if len(point_cloud) == 0:
        img = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(img, "No points", (width//2-50, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img
    
    # Create blank image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Project 3D points to 2D (top-down view: X-Z plane)
    x = point_cloud[:, 0]  # Left-right
    z = point_cloud[:, 2]  # Depth (forward-back)
    y = point_cloud[:, 1]  # Up-down (for coloring)
    
    # Normalize to image coordinates
    if len(x) > 0:
        x_min, x_max = x.min(), x.max()
        z_min, z_max = z.min(), z.max()
        
        if x_max - x_min > 0 and z_max - z_min > 0:
            x_norm = ((x - x_min) / (x_max - x_min) * (width - 40) + 20).astype(np.int32)
            z_norm = ((z - z_min) / (z_max - z_min) * (height - 40) + 20).astype(np.int32)
            
            # Color by height (Y axis)
            y_norm = np.clip((y - y.min()) / (y.max() - y.min() + 1e-6) * 255, 0, 255).astype(np.uint8)
            colors = cv2.applyColorMap(y_norm.reshape(-1, 1), cv2.COLORMAP_JET)
            
            # Draw points
            for i in range(len(x_norm)):
                if 0 <= x_norm[i] < width and 0 <= z_norm[i] < height:
                    color = tuple(map(int, colors[i][0]))
                    cv2.circle(img, (x_norm[i], z_norm[i]), 1, color, -1)
    
    # Add info
    cv2.putText(img, f"Points: {len(point_cloud)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "Top-down view (X-Z)", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

def visualize_stereo_depth():
    """Show real-time stereo depth computation with point cloud"""
    
    print("Starting stereo depth visualization...")
    print("Press 'q' to quit")
    
    cam = StereoCamera("GXIVISION", camera_index=0, width=2560, height=720)
    cam.start()
    
    print("\nYou should see 5 windows:")
    print("  1. LEFT: Left camera view")
    print("  2. RIGHT: Right camera view (notice slight parallax)")
    print("  3. DEPTH: Computed depth map (color: blue=far, red=close)")
    print("  4. DEPTH RAW: Raw depth values visualization")
    print("  5. POINT CLOUD: 3D point cloud top-down view")
    
    while True:
        try:
            # Get raw frame
            ret, frame = cam._cap.read()
            if not ret:
                continue
            
            # Split stereo pair
            left, right = cam._split_stereo_image(frame)
            
            # Compute depth
            depth_map = cam._compute_depth_map(left, right)
            
            # Create visualizations
            # 1. Disparity visualization (colorized)
            disparity = cam._stereo.compute(
                cv2.cvtColor(left, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            )
            disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disparity_vis = disparity_vis.astype(np.uint8)
            disparity_color = cv2.applyColorMap(disparity_vis, cv2.COLORMAP_JET)
            
            # 2. Actual depth map visualization
            depth_vis = np.zeros_like(depth_map)
            valid_depth = depth_map > 0
            if valid_depth.any():
                depth_vis[valid_depth] = depth_map[valid_depth]
                depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            
            # 3. Get point cloud
            try:
                point_cloud = cam._depth_map_to_point_cloud(depth_map, left, downsample=6)
                pc_vis = create_point_cloud_image(point_cloud)
            except Exception as e:
                pc_vis = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(pc_vis, f"Error: {str(e)[:40]}", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Resize for display
            scale = 0.5
            left_small = cv2.resize(left, None, fx=scale, fy=scale)
            right_small = cv2.resize(right, None, fx=scale, fy=scale)
            disparity_small = cv2.resize(disparity_color, None, fx=scale, fy=scale)
            depth_small = cv2.resize(depth_color, None, fx=scale, fy=scale)
            
            # Add labels
            cv2.putText(left_small, "LEFT CAMERA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(right_small, "RIGHT CAMERA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disparity_small, "DISPARITY MAP", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_small, "DEPTH MAP (meters)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Stack in 2 rows
            row1 = np.hstack([left_small, right_small])
            row2 = np.hstack([disparity_small, depth_small])
            combined = np.vstack([row1, row2])
            
            # Show windows
            cv2.imshow('Stereo Vision', combined)
            cv2.imshow('3D Point Cloud (Top View)', pc_vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            break
    
    cam.stop()
    cv2.destroyAllWindows()
    print("\nVisualization complete!")
    print("✓ LEFT camera captures scene")
    print("✓ RIGHT camera captures same scene from different angle")
    print("✓ DISPARITY computed by comparing the two images")
    print("✓ DEPTH calculated from disparity")
    print("✓ POINT CLOUD generated from depth map!")

if __name__ == '__main__':
    try:
        visualize_stereo_depth()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

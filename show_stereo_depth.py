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
    
    print("\nYou should see 4 views in a single window:")
    print("  1. TOP-LEFT: Left camera view")
    print("  2. TOP-RIGHT: Right camera view (notice slight parallax)")
    print("  3. BOTTOM-LEFT: Depth map (color: blue=far, red=close)")
    print("  4. BOTTOM-RIGHT: 3D point cloud top-down view")
    
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
            # Note: We already computed depth_map which includes disparity computation
            # No need to recompute disparity separately
            
            # 2. Actual depth map visualization (grayscale)
            depth_vis = np.zeros_like(depth_map)
            valid_depth = depth_map > 0
            if valid_depth.any():
                depth_vis[valid_depth] = depth_map[valid_depth]
                depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            # Convert grayscale to BGR for consistent stacking
            depth_color = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            
            # 3. Get point cloud (with higher downsampling for speed)
            try:
                point_cloud = cam._depth_map_to_point_cloud(depth_map, left, downsample=12)
                pc_vis = create_point_cloud_image(point_cloud)
            except Exception as e:
                pc_vis = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(pc_vis, f"Error: {str(e)[:40]}", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Resize for display (larger scale = less processing)
            scale = 0.4
            left_small = cv2.resize(left, None, fx=scale, fy=scale)
            right_small = cv2.resize(right, None, fx=scale, fy=scale)
            depth_small = cv2.resize(depth_color, None, fx=scale, fy=scale)
            
            # Resize point cloud to match camera views
            pc_small = cv2.resize(pc_vis, (left_small.shape[1], left_small.shape[0]))
            
            # Add labels
            cv2.putText(left_small, "LEFT CAMERA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(right_small, "RIGHT CAMERA", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(depth_small, "DEPTH MAP (meters)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(pc_small, "3D POINT CLOUD", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Stack in 2x2 grid
            row1 = np.hstack([left_small, right_small])
            row2 = np.hstack([depth_small, pc_small])
            combined = np.vstack([row1, row2])
            
            # Show single window with all 4 views
            cv2.imshow('Stereo Vision - 4 Views', combined)
            
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

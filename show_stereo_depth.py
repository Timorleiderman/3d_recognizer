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
    print("\nControls:")
    print("  Press 'q' to quit")
    print("  Press '+' to increase focal length (objects closer)")
    print("  Press '-' to decrease focal length (objects farther)")
    print("  Press 'r' to reset to default")
    print("  Press 'f' to toggle frame skip (faster/slower)")
    print("  Press 'd' to show disparity info")
    
    cam = StereoCamera("GXIVISION", camera_index=0, width=2560, height=720)
    cam.start()
    
    # Allow runtime adjustment of focal length
    focal_length = cam._focal_length_px
    baseline = cam._baseline_mm
    show_stats = False
    
    print("\nYou should see 4 views in a single window:")
    print("  1. TOP-LEFT: Left camera view")
    print("  2. TOP-RIGHT: Right camera view (notice slight parallax)")
    print("  3. BOTTOM-LEFT: Depth map (white=close, black=far)")
    print("  4. BOTTOM-RIGHT: 3D point cloud top-down view")
    print(f"\nCurrent settings: Baseline={baseline*1000:.1f}mm, Focal={focal_length:.1f}px")
    
    # FPS tracking
    import time
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    # Skip frames for faster display (process every Nth frame)
    frame_skip = 3  # Process every 3rd frame (faster default)
    frame_counter = 0
    
    # Cache last results
    last_depth_map = None
    last_pc_vis = None
    last_depth_color = None
    
    while True:
        frame_counter += 1
        try:
            # Get raw frame
            ret, frame = cam._cap.read()
            if not ret:
                continue
            
            # Split stereo pair (always needed for display)
            left, right = cam._split_stereo_image(frame)
            
            # Skip depth computation on some frames for speed
            compute_depth = (frame_counter % frame_skip == 0)
            
            # Compute depth only on selected frames
            if compute_depth:
                # Update camera focal length if changed
                cam._focal_length_px = focal_length
                
                # Downscale for faster stereo matching (2x faster)
                scale_factor = 0.5
                left_small = cv2.resize(left, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                right_small = cv2.resize(right, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
                
                # Compute depth at lower resolution
                depth_map_small = cam._compute_depth_map(left_small, right_small)
                
                # Upscale depth map back to original size
                depth_map = cv2.resize(depth_map_small, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
                # Adjust depth values for scale
                depth_map *= (1.0 / scale_factor)
                last_depth_map = depth_map
                
                # Create visualizations
                # 2. Better depth map visualization with automatic range adjustment
                depth_vis = depth_map.copy()
                valid_depth_mask = depth_map > 0
                
                if valid_depth_mask.any():
                    # Clip to reasonable range (0.2m - 3.0m)
                    depth_vis = np.clip(depth_vis, 0.2, 3.0)
                    # Invert: closer = brighter (more intuitive)
                    depth_vis = 3.0 - depth_vis
                    # Normalize only valid regions
                    depth_vis = cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX)
                    # Mask out invalid regions
                    depth_vis[~valid_depth_mask] = 0
                else:
                    depth_vis = np.zeros_like(depth_map)
                
                depth_vis = depth_vis.astype(np.uint8)
                # Apply color map for better visibility
                depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
                # Black out invalid regions
                depth_color[~valid_depth_mask] = [0, 0, 0]
                last_depth_color = depth_color
                
                # 3. Get point cloud (very high downsampling for speed)
                try:
                    point_cloud = cam._depth_map_to_point_cloud(depth_map, left, downsample=20)
                    pc_vis = create_point_cloud_image(point_cloud)
                    last_pc_vis = pc_vis
                except Exception as e:
                    pc_vis = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(pc_vis, f"Error: {str(e)[:40]}", (10, 180),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    last_pc_vis = pc_vis
            else:
                # Reuse last computation
                if last_depth_map is not None:
                    depth_map = last_depth_map
                    depth_color = last_depth_color
                    pc_vis = last_pc_vis
                    valid_depth_mask = depth_map > 0
                else:
                    # First frame, create empty
                    depth_map = np.zeros((left.shape[0], left.shape[1]), dtype=np.float32)
                    depth_color = np.zeros((left.shape[0], left.shape[1], 3), dtype=np.uint8)
                    pc_vis = np.zeros((360, 640, 3), dtype=np.uint8)
                    valid_depth_mask = np.zeros_like(depth_map, dtype=bool)
            
            # Resize for display (smaller = faster rendering)
            scale = 0.3
            left_disp = cv2.resize(left, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            right_disp = cv2.resize(right, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            depth_small = cv2.resize(depth_color, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            
            # Resize point cloud to match camera views
            pc_small = cv2.resize(pc_vis, (left_disp.shape[1], left_disp.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # Add labels and info
            cv2.putText(left_disp, "LEFT CAMERA", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(right_disp, "RIGHT CAMERA", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Calculate FPS
            frame_count += 1
            if time.time() - fps_time >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_time = time.time()
            
            # Depth map with stats
            valid_depth = depth_map[depth_map > 0]
            if len(valid_depth) > 0:
                min_d, max_d = valid_depth.min(), valid_depth.max()
                coverage = 100.0 * len(valid_depth) / depth_map.size
                depth_info = f"{min_d:.2f}-{max_d:.2f}m ({coverage:.0f}%)"
                cv2.putText(depth_small, depth_info, (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(depth_small, "RED=close BLUE=far", (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            else:
                cv2.putText(depth_small, "No depth - improve light", (10, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # FPS and focal length info
            cv2.putText(pc_small, f"F: {focal_length:.0f}px (+/-) FPS: {fps}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Stack in 2x2 grid
            row1 = np.hstack([left_disp, right_disp])
            row2 = np.hstack([depth_small, pc_small])
            combined = np.vstack([row1, row2])
            
            # Show single window with all 4 views
            cv2.imshow('Stereo Vision - 4 Views', combined)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                focal_length += 10
                print(f"Focal length: {focal_length:.0f}px")
            elif key == ord('-') or key == ord('_'):
                focal_length = max(50, focal_length - 10)
                print(f"Focal length: {focal_length:.0f}px")
            elif key == ord('r'):
                focal_length = 350.0
                print(f"Reset focal length: {focal_length:.0f}px")
            elif key == ord('f'):
                # Cycle through frame skip modes
                if frame_skip == 1:
                    frame_skip = 3
                    print("Frame skip: 3 (faster)")
                elif frame_skip == 3:
                    frame_skip = 5
                    print("Frame skip: 5 (fastest)")
                else:
                    frame_skip = 1
                    print("Frame skip: 1 (best quality, slower)")
            elif key == ord('d'):
                show_stats = not show_stats
                if show_stats:
                    print(f"\nCurrent settings:")
                    print(f"  Baseline: {baseline*1000:.1f}mm")
                    print(f"  Focal length: {focal_length:.1f}px")
                    if len(valid_depth) > 0:
                        print(f"  Depth range: {min_d:.2f}m - {max_d:.2f}m")
                        print(f"  Valid pixels: {len(valid_depth)}/{depth_map.size} ({100*len(valid_depth)/depth_map.size:.1f}%)")
                
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

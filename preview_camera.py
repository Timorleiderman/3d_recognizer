#!/usr/bin/env python3
"""
Camera Preview - Show raw stereo camera images
This helps verify the camera is working and properly aligned
"""

import cv2
import numpy as np

def preview_stereo_camera():
    """Show live preview of stereo camera feed"""
    
    print("Opening stereo camera...")
    print("Press 'q' to quit, 's' to save snapshot")
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # Set to stereo mode
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera: {width}x{height}")
    
    if width != 2560:
        print("Warning: Not in stereo mode!")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        # Split stereo image
        half_width = frame.shape[1] // 2
        left = frame[:, :half_width]
        right = frame[:, half_width:]
        
        # Create display with both views side by side
        # Resize for display if too large
        display_width = 1280
        scale = display_width / frame.shape[1]
        display_height = int(frame.shape[0] * scale)
        
        display = cv2.resize(frame, (display_width, display_height))
        
        # Add labels
        cv2.putText(display, "LEFT", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "RIGHT", (display_width//2 + 20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Frame: {frame_count}", (20, display_height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw center line
        cv2.line(display, (display_width//2, 0), 
                (display_width//2, display_height), (0, 255, 255), 1)
        
        cv2.imshow('Stereo Camera Preview', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save snapshot
            filename = f'stereo_snapshot_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            
            # Also save split views
            cv2.imwrite(f'left_{frame_count}.jpg', left)
            cv2.imwrite(f'right_{frame_count}.jpg', right)
            print(f"Saved left_{frame_count}.jpg and right_{frame_count}.jpg")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Camera preview closed")

if __name__ == '__main__':
    try:
        preview_stereo_camera()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

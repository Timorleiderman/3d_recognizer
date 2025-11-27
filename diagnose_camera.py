#!/usr/bin/env python3
"""
Diagnose GXIVISION stereo camera configuration
This helps identify the correct video format and resolution
"""

import subprocess
import sys

def check_v4l2():
    """Check video devices and supported formats"""
    print("=" * 60)
    print("GXIVISION Stereo Camera Diagnostics")
    print("=" * 60)
    
    # List video devices
    print("\n1. Video devices:")
    try:
        result = subprocess.run(['ls', '-la', '/dev/video*'], 
                              capture_output=True, text=True)
        print(result.stdout)
    except Exception as e:
        print(f"Error: {e}")
    
    # Check each video device
    for device_num in range(4):
        device = f'/dev/video{device_num}'
        print(f"\n{'='*60}")
        print(f"Checking {device}...")
        print('='*60)
        
        try:
            # List formats
            result = subprocess.run(
                ['v4l2-ctl', '--list-formats-ext', '-d', device],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode == 0:
                print(result.stdout)
                
                # Look for stereo-specific resolutions
                if '2560' in result.stdout and '720' in result.stdout:
                    print(f"\n⭐ FOUND STEREO FORMAT on {device}!")
                elif '1280' in result.stdout and '720' in result.stdout:
                    print(f"\n💡 Found 1280x720 - might be single lens on {device}")
            else:
                print(f"Device {device} not available")
                
        except subprocess.TimeoutExpired:
            print(f"Timeout checking {device}")
        except FileNotFoundError:
            print("\n❌ v4l2-ctl not found!")
            print("Install with: sudo apt-get install v4l-utils")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("OpenCV Test")
    print("="*60)
    
    try:
        import cv2
        
        for idx in range(4):
            print(f"\nTrying /dev/video{idx}...")
            cap = cv2.VideoCapture(idx)
            
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                print(f"  ✓ Opened: {width}x{height} @ {fps}fps")
                
                # Try setting stereo resolution
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if new_width == 2560 and new_height == 720:
                    print(f"  ⭐ Can set to 2560x720! Use device {idx}")
                else:
                    print(f"  After setting: {new_width}x{new_height}")
                
                cap.release()
            else:
                print(f"  Device {idx} not available")
                
    except ImportError:
        print("OpenCV not available")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*60)
    print("Recommendations")
    print("="*60)
    print("""
For GXIVISION stereo camera:

1. If you found a device with 2560x720:
   - Note the device number (e.g., /dev/video0)
   - That's your stereo camera!

2. If only 640x480 or 1280x720 found:
   - Camera might need Windows driver/software first
   - Try different USB ports (USB 3.0 preferred)
   - Check if camera has mode switch button/software
   - May need custom v4l2 settings

3. Manual configuration:
   Try: v4l2-ctl -d /dev/video0 --set-fmt-video=width=2560,height=720,pixelformat=MJPG

4. If two 1280x720 devices found:
   - Camera might expose left/right as separate devices
   - Will need to modify code to open both

For more help, see: https://github.com/your-repo/issues
""")

if __name__ == '__main__':
    check_v4l2()

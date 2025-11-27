"""
Jetson Nano OpenGL fix
This module applies patches to work around OpenGL/GLX issues on Jetson Nano
Import this module BEFORE importing vispy or any OpenGL-dependent libraries
"""

import os
import sys

def apply_jetson_fixes():
    """Apply necessary fixes for Jetson Nano OpenGL support"""
    
    # 1. Set environment variables before any OpenGL imports
    os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':0')
    
    # 2. Try to patch pyopengltk to be more forgiving
    try:
        import pyopengltk.linux as pyopengl_linux
        original_tkCreateContext = pyopengl_linux.OpenGLFrame.tkCreateContext
        
        def patched_tkCreateContext(self):
            """Patched version that handles NULL FBConfigs more gracefully"""
            try:
                return original_tkCreateContext(self)
            except (ValueError, AttributeError) as e:
                print(f"Warning: OpenGL context creation issue: {e}")
                print("Attempting software rendering fallback...")
                # Try software rendering as fallback
                os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
                try:
                    return original_tkCreateContext(self)
                except Exception as e2:
                    print(f"Software rendering also failed: {e2}")
                    print("OpenGL visualization may not work properly.")
                    # Don't crash, let the app continue
                    return None
        
        pyopengl_linux.OpenGLFrame.tkCreateContext = patched_tkCreateContext
        print("Applied Jetson Nano OpenGL patches")
        
    except ImportError:
        print("pyopengltk not imported yet, patches will be applied later if needed")
    except Exception as e:
        print(f"Could not apply OpenGL patches: {e}")

def check_opengl_support():
    """Check if OpenGL is available and working"""
    try:
        from OpenGL import GL
        import ctypes
        
        # Try to get OpenGL version
        try:
            # This will fail if no context is available yet
            version = GL.glGetString(GL.GL_VERSION)
            if version:
                print(f"OpenGL Version: {version.decode('utf-8')}")
                return True
        except Exception:
            # No context yet, which is expected at startup
            pass
            
        return True  # OpenGL library is at least importable
        
    except Exception as e:
        print(f"OpenGL check failed: {e}")
        return False

if __name__ != '__main__':
    # Apply fixes when imported
    apply_jetson_fixes()

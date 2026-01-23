#!/usr/bin/env python3
"""
main_standalone.py - MediaPipe Version
"""
import cv2
import time
from datetime import datetime
import argparse
import sys
import os

# Add parent directory to path to import shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import config
from shared.drowsiness_analyzer import DrowsinessAnalyzer

import psutil
from gpiozero import CPUTemperature

def init_camera():
    print("[INFO] Initializing camera...")
    # Try picamera2
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        camera_config = camera.create_video_configuration(
            main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": config.CAMERA_FPS}
        )
        camera.configure(camera_config)
        camera.start()
        time.sleep(0.5)
        print(f"[INFO] PiCamera2 active: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        return camera, True
    except Exception as e:
        print(f"[WARN] PiCamera2 failed ({e}), trying OpenCV/USB...")
    
    # OpenCV Fallback
    cap = cv2.VideoCapture(0)
    cap.set(3, config.CAMERA_WIDTH)
    cap.set(4, config.CAMERA_HEIGHT)
    if not cap.isOpened(): return None, False
    print("[INFO] USB Webcam active")
    return cap, False

def capture_frame(camera, use_picamera2):
    if use_picamera2:
        frame = camera.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = camera.read()
        return frame if ret else None

def get_system_stats():
    """Returns string with CPU%, RAM%, and Temp°C"""
    try:
        # CPU Temperature
        cpu_temp = CPUTemperature().temperature
        # CPU Usage Percentage
        cpu_usage = psutil.cpu_percent()
        # RAM Usage Percentage
        ram = psutil.virtual_memory()
        
        return f"CPU: {cpu_usage}% | RAM: {ram.percent}% | Temp: {cpu_temp:.1f}°C"
    except Exception:
        return "Stats N/A"  

# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-display', action='store_true', help='Disable video output')
    args = parser.parse_args()
    
    if args.no_display: config.DISPLAY_ENABLED = False
    
    print("="*60)
    print(" DROWSINESS DETECTION - STANDALONE (MediaPipe)")
    print("="*60)
    
    detector = DrowsinessAnalyzer()
    camera, use_picamera2 = init_camera()
    if not camera: return
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame = capture_frame(camera, use_picamera2)
            if frame is None: continue
            
            # --- DETECTION ---
            processed_frame, ear, mar, is_drowsy, is_yawning = detector.detect(frame)
            frame_count += 1
            
            # Display (only if enabled)
            if config.DISPLAY_ENABLED:
                cv2.imshow("Drowsiness", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # Lightweight console log
            if frame_count % 15 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                sys_stats = get_system_stats()
                status = "DRWS" if is_drowsy else "OK"
                yawn_txt = "YAWN" if is_yawning else ""

                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"FPS: {fps:.1f} | EAR: {ear:.2f} | {status}{yawn_txt} || {sys_stats}")

    except KeyboardInterrupt:
        print("\n[INFO] User stopped")
    finally:
        if use_picamera2: camera.stop()
        else: camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
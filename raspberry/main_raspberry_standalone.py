#!/usr/bin/env python3
"""
main_standalone.py - Versione MediaPipe
"""
import cv2
import time
from datetime import datetime
import argparse
import drowsiness_config_standalone as config
from drowsiness_detector_standalone import DrowsinessDetector

import psutil
from gpiozero import CPUTemperature

def init_camera():
    print("[INFO] Inizializzazione camera...")
    # Prova picamera2
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
        print(f"[INFO] PiCamera2 attiva: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        return camera, True
    except Exception as e:
        print(f"[WARN] PiCamera2 non usata ({e}), provo OpenCV/USB...")
    
    # Fallback OpenCV
    cap = cv2.VideoCapture(0)
    cap.set(3, config.CAMERA_WIDTH)
    cap.set(4, config.CAMERA_HEIGHT)
    if not cap.isOpened(): return None, False
    print("[INFO] Webcam USB attiva")
    return cap, False

def capture_frame(camera, use_picamera2):
    if use_picamera2:
        frame = camera.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = camera.read()
        return frame if ret else None

def get_system_stats():
    """Restituisce stringa con CPU%, RAM% e Temp°C"""
    try:
        # Temperatura CPU
        cpu_temp = CPUTemperature().temperature
        # Percentuale utilizzo CPU
        cpu_usage = psutil.cpu_percent()
        # Percentuale utilizzo RAM
        ram = psutil.virtual_memory()
        
        return f"CPU: {cpu_usage}% | RAM: {ram.percent}% | Temp: {cpu_temp:.1f}°C"
    except Exception:
        return "Stats N/A"  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-display', action='store_true', help='Disabilita video output')
    args = parser.parse_args()
    
    if args.no_display: config.DISPLAY_ENABLED = False
    
    print("="*60)
    print(" DROWSINESS DETECTION - STANDALONE (MediaPipe)")
    print("="*60)
    
    detector = DrowsinessDetector()
    camera, use_picamera2 = init_camera()
    if not camera: return
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            frame = capture_frame(camera, use_picamera2)
            if frame is None: continue
            
            # --- RILEVAMENTO ---
            processed_frame, ear, mar, is_drowsy, is_yawning = detector.detect(frame)
            frame_count += 1
            
            # Mostra (solo se abilitato)
            if config.DISPLAY_ENABLED:
                cv2.imshow("Drowsiness", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            # Log console leggero
            if frame_count % 15 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                #fps = frame_count / (time.time() - start_time)
                
                sys_stats = get_system_stats()
                status = "DRWS" if is_drowsy else "OK"
                yawn_txt = "YAWN" if is_yawning else ""

                #print(f"FPS: {fps:.1f} | EAR: {ear:.2f} | MAR: {mar:.2f} | {status}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"FPS: {fps:.1f} | EAR: {ear:.2f} | {status}{yawn_txt} || {sys_stats}")

    except KeyboardInterrupt:
        print("\n[INFO] Stop utente")
    finally:
        if use_picamera2: camera.stop()
        else: camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
main_standalone.py - Drowsiness Detection standalone per Raspberry Pi
Esegue tutto in locale senza bisogno del PC.

Compatibile con Raspberry Pi OS Bookworm (64-bit)
NOTA: Performance limitata su Pi 3B+ (~1-3 FPS)
"""

import cv2
import time
from datetime import datetime
import argparse

import raspberry.drowsiness_config_standalone as config
from raspberry.drowsiness_detector_standalone import DrowsinessDetector


def init_camera():
    """Inizializza camera (picamera2 o USB)"""
    print("[INFO] Inizializzazione camera...")
    
    # Prova picamera2 (Bookworm)
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
        
        print(f"[INFO] PiCamera2: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
        return camera, True
        
    except ImportError:
        print("[WARN] picamera2 non disponibile, provo OpenCV...")
    except Exception as e:
        print(f"[WARN] Errore PiCamera2: {e}")
    
    # Fallback OpenCV
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
    
    if not cap.isOpened():
        return None, False
    
    print("[INFO] OpenCV VideoCapture inizializzata")
    return cap, False


def capture_frame(camera, use_picamera2):
    """Cattura un frame"""
    if use_picamera2:
        frame = camera.capture_array()
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = camera.read()
        return frame if ret else None


def main():
    parser = argparse.ArgumentParser(description='Drowsiness Detection Standalone')
    parser.add_argument('--no-display', action='store_true',
                       help='Disabilita visualizzazione (per Lite/SSH)')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Salva video su file')
    parser.add_argument('--low-res', action='store_true',
                       help='Usa risoluzione 160x120 per più FPS')
    args = parser.parse_args()
    
    if args.no_display:
        config.DISPLAY_ENABLED = False
    
    if args.low_res:
        config.CAMERA_WIDTH = 160
        config.CAMERA_HEIGHT = 120
        print("[INFO] Modalità low-res: 160x120")
    
    print("=" * 60)
    print("  DROWSINESS DETECTION - STANDALONE")
    print("  (Raspberry Pi - tutto in locale)")
    print("=" * 60)
    print(f"[INFO] Risoluzione: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}")
    print(f"[INFO] Display: {'ON' if config.DISPLAY_ENABLED else 'OFF'}")
    print("[WARN] Performance attese: 1-3 FPS su Pi 3B+")
    print("=" * 60)
    
    # Inizializza detector
    try:
        detector = DrowsinessDetector()
    except Exception as e:
        print(f"[ERRORE] {e}")
        print("[INFO] Assicurati di aver scaricato shape_predictor_68_face_landmarks.dat")
        print("       wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("       bunzip2 shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    # Inizializza camera
    camera, use_picamera2 = init_camera()
    if camera is None:
        print("[ERRORE] Camera non disponibile")
        return
    
    # Video writer
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(
            args.save_video, fourcc, 5.0,  # 5 FPS per il video salvato
            (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        )
        print(f"[INFO] Salvataggio video: {args.save_video}")
    
    # Tracking FPS
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    fps_update_time = start_time
    
    print("\n[INFO] Sistema attivo! Premi Ctrl+C per uscire")
    print("-" * 60)
    
    try:
        while True:
            frame_start = time.time()
            
            # Cattura frame
            frame = capture_frame(camera, use_picamera2)
            if frame is None:
                continue
            
            # Analizza frame
            processed_frame, ear, mar, is_drowsy, is_yawning = detector.detect(frame)
            frame_count += 1
            
            # Calcola FPS (aggiorna ogni secondo)
            current_time = time.time()
            if current_time - fps_update_time >= 1.0:
                fps_display = frame_count / (current_time - start_time)
                fps_update_time = current_time
            
            # Aggiungi FPS al frame
            cv2.putText(processed_frame, f"FPS: {fps_display:.1f}", 
                       (processed_frame.shape[1] - 100, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Salva video
            if video_writer:
                video_writer.write(processed_frame)
            
            # Mostra frame (se display abilitato)
            if config.DISPLAY_ENABLED:
                cv2.imshow("Drowsiness Detection", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Log periodico (ogni 10 frame per non spammare)
            if frame_count % 10 == 0:
                status = "⚠️ DROWSY" if is_drowsy else "✓ OK"
                yawn = " 🥱" if is_yawning else ""
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"FPS: {fps_display:.1f} | EAR: {ear:.2f} | MAR: {mar:.2f} | {status}{yawn}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interruzione da tastiera")
    
    finally:
        # Cleanup
        print("\n[INFO] Chiusura...")
        
        if use_picamera2:
            camera.stop()
            camera.close()
        else:
            camera.release()
        
        if video_writer:
            video_writer.release()
        
        if config.DISPLAY_ENABLED:
            cv2.destroyAllWindows()
        
        # Statistiche finali
        elapsed = time.time() - start_time
        stats = detector.get_statistics()
        
        print("\n" + "=" * 60)
        print("STATISTICHE FINALI:")
        print(f"  Frame elaborati: {frame_count}")
        print(f"  Tempo totale: {elapsed:.1f}s")
        print(f"  FPS medio: {frame_count/elapsed:.2f}")
        print(f"  Eventi sonnolenza: {stats['total_drowsy_events']}")
        print(f"  Eventi sbadiglio: {stats['total_yawn_events']}")
        print(f"  Log: {config.LOG_FILE}")
        print("=" * 60)


if __name__ == "__main__":
    main()
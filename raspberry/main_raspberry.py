#!/usr/bin/env python3
# main.py - Script principale per Drowsiness Detection su Raspberry Pi

import cv2
import argparse
import time
from datetime import datetime
import sys

# Import moduli locali
import drowsiness_config as config
from drowsiness_detector import DrowsinessDetector
from alarm_module import AlarmManager

def main():
    # Parser argomenti
    parser = argparse.ArgumentParser(description='Sistema di Drowsiness Detection')
    parser.add_argument('--no-display', action='store_true',
                       help='Disabilita visualizzazione (per SSH)')
    parser.add_argument('--no-alarm', action='store_true',
                       help='Disabilita allarme sonoro')
    parser.add_argument('--camera', type=int, default=0,
                       help='Indice camera (default: 0)')
    parser.add_argument('--save-video', type=str, default=None,
                       help='Salva video con path specificato')
    args = parser.parse_args()
    
    # Override configurazioni da argomenti
    if args.no_display:
        config.DISPLAY_ENABLED = False
    if args.no_alarm:
        config.ALARM_ENABLED = False
    
    print("=" * 60)
    print("  SISTEMA DI DROWSINESS DETECTION - Raspberry Pi")
    print("=" * 60)
    print(f"[INFO] Avvio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[INFO] Display: {'Abilitato' if config.DISPLAY_ENABLED else 'Disabilitato'}")
    print(f"[INFO] Allarme: {'Abilitato' if config.ALARM_ENABLED else 'Disabilitato'}")
    print(f"[INFO] EAR Threshold: {config.EAR_THRESHOLD}")
    print(f"[INFO] Frame consecutivi: {config.EAR_CONSEC_FRAMES}")
    print("=" * 60)
    
    # Inizializza detector e alarm manager
    try:
        detector = DrowsinessDetector()
        alarm = AlarmManager()
    except Exception as e:
        print(f"[ERRORE] Inizializzazione fallita: {e}")
        print("[INFO] Assicurati di aver scaricato shape_predictor_68_face_landmarks.dat")
        return
    
    # Inizializza camera
    print(f"[INFO] Apertura camera {args.camera}...")
    
    # Per PiCamera (versione 1.x per Buster)
    try:
        from picamera.array import PiRGBArray
        from picamera import PiCamera
        
        camera = PiCamera()
        camera.resolution = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        camera.framerate = config.CAMERA_FPS
        rawCapture = PiRGBArray(camera, size=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
        
        # Attendi inizializzazione camera
        time.sleep(0.1)
        use_picamera = True
        print("[INFO] PiCamera (v1.x) inizializzata")
    except:
        # Fallback a OpenCV VideoCapture (per webcam USB)
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        use_picamera = False
        
        if not cap.isOpened():
            print("[ERRORE] Impossibile aprire la camera")
            return
        print("[INFO] OpenCV VideoCapture inizializzata (webcam USB)")
    
    # Setup video writer se richiesto
    video_writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 20.0,
                                      (config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
        print(f"[INFO] Registrazione video su: {args.save_video}")
    
    # Variabili per FPS
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    print("\n[INFO] Sistema attivo! Premi 'q' per uscire, 's' per statistiche")
    print("-" * 60)
    
    try:
        # Generatore per PiCamera
        if use_picamera:
            frame_stream = camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
        
        frame_iterator = frame_stream if use_picamera else iter(int, 1)
        
        while True:
            # Cattura frame
            if use_picamera:
                frame_obj = next(frame_iterator)
                frame = frame_obj.array
                rawCapture.truncate(0)  # Pulisci stream per prossimo frame
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[ERRORE] Impossibile leggere frame")
                    break
            
            # Processa frame
            processed_frame, ear, mar, is_drowsy, is_yawning = detector.detect(frame)
            
            # Gestione allarme
            if is_drowsy:
                alarm.start_alarm()
            else:
                alarm.stop_alarm()
            
            # Calcola FPS
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                current_fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Mostra FPS
            cv2.putText(processed_frame, f"FPS: {current_fps:.1f}", 
                       (processed_frame.shape[1] - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Salva video se abilitato
            if video_writer:
                video_writer.write(processed_frame)
            
            # Mostra frame se display abilitato
            if config.DISPLAY_ENABLED:
                cv2.imshow("Drowsiness Detection", processed_frame)
                
                # Gestione tasti
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n[INFO] Uscita richiesta dall'utente")
                    break
                elif key == ord('s'):
                    stats = detector.get_statistics()
                    print("\n" + "=" * 40)
                    print("STATISTICHE:")
                    print(f"  Eventi Sonnolenza: {stats['total_drowsy_events']}")
                    print(f"  Eventi Sbadiglio: {stats['total_yawn_events']}")
                    print(f"  FPS medio: {current_fps:.1f}")
                    print("=" * 40 + "\n")
            else:
                # In modalità headless, controlla input da tastiera
                # (richiede input non-bloccante, semplificato per demo)
                time.sleep(0.03)  # ~30 FPS
            
            # Output testuale periodico
            if fps_counter % 30 == 0 and not config.DISPLAY_ENABLED:
                status = "ALERT!" if is_drowsy else "OK"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"EAR: {ear:.2f} | MAR: {mar:.2f} | Status: {status}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interruzione da tastiera (Ctrl+C)")
    
    finally:
        # Cleanup
        print("\n[INFO] Chiusura sistema...")
        
        if use_picamera:
            camera.close()
        else:
            cap.release()
        
        if video_writer:
            video_writer.release()
        
        if config.DISPLAY_ENABLED:
            cv2.destroyAllWindows()
        
        alarm.stop_alarm()
        
        # Statistiche finali
        stats = detector.get_statistics()
        print("\n" + "=" * 60)
        print("STATISTICHE FINALI:")
        print(f"  Eventi Sonnolenza totali: {stats['total_drowsy_events']}")
        print(f"  Eventi Sbadiglio totali: {stats['total_yawn_events']}")
        print(f"  Log salvato in: {config.LOG_FILE}")
        print("=" * 60)
        print("[INFO] Sistema terminato correttamente")

if __name__ == "__main__":
    main()
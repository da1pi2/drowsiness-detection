#!/usr/bin/env python3
"""
raspberry_client.py - Client per Raspberry Pi
Cattura frame dalla camera e li invia al PC per l'elaborazione
Riceve i risultati e gestisce l'allarme locale
"""

import socket
import pickle
import struct
import cv2
import numpy as np
import time
from datetime import datetime
import sys

# Import moduli locali
import drowsiness_config as config
from alarm_module import AlarmManager

# ===================== CONFIGURAZIONE CLIENT =====================
# Modifica questo IP con l'indirizzo del tuo PC
PC_SERVER_IP = "192.168.1.100"  # <-- CAMBIA CON L'IP DEL TUO PC
PC_SERVER_PORT = 5555

# Qualità compressione JPEG (70 = buon compromesso qualità/banda)
JPEG_QUALITY = 70

# Timeout connessione
CONNECTION_TIMEOUT = 10
RECONNECT_DELAY = 5


class RaspberryClient:
    """Client che invia frame al PC e riceve risultati"""
    
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.connected = False
        
        # Camera
        self.camera = None
        self.rawCapture = None
        self.use_picamera = False
        
        # Allarme locale
        self.alarm = AlarmManager()
        
        # Statistiche
        self.frames_sent = 0
        self.start_time = None
    
    def connect(self):
        """Connette al server PC"""
        try:
            print(f"[INFO] Connessione a {self.server_ip}:{self.server_port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(CONNECTION_TIMEOUT)
            self.socket.connect((self.server_ip, self.server_port))
            self.socket.settimeout(None)
            self.connected = True
            print("[INFO] Connesso al server!")
            return True
        except Exception as e:
            print(f"[ERRORE] Connessione fallita: {e}")
            self.connected = False
            return False
    
    def init_camera(self):
        """Inizializza la camera (PiCamera o USB)"""
        print("[INFO] Inizializzazione camera...")
        
        # Prova PiCamera
        try:
            from picamera.array import PiRGBArray
            from picamera import PiCamera
            
            self.camera = PiCamera()
            self.camera.resolution = (config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
            self.camera.framerate = config.CAMERA_FPS
            self.rawCapture = PiRGBArray(self.camera, 
                                         size=(config.CAMERA_WIDTH, config.CAMERA_HEIGHT))
            time.sleep(0.1)  # Warmup
            self.use_picamera = True
            print(f"[INFO] PiCamera inizializzata: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT} @ {config.CAMERA_FPS}fps")
            return True
        except ImportError:
            print("[WARN] PiCamera non disponibile, provo webcam USB...")
        except Exception as e:
            print(f"[WARN] Errore PiCamera: {e}")
        
        # Fallback a webcam USB
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
            
            if not self.camera.isOpened():
                raise Exception("Camera non disponibile")
            
            self.use_picamera = False
            print("[INFO] Webcam USB inizializzata")
            return True
        except Exception as e:
            print(f"[ERRORE] Impossibile inizializzare camera: {e}")
            return False
    
    def capture_frame(self):
        """Cattura un frame dalla camera"""
        if self.use_picamera:
            self.rawCapture.truncate(0)
            self.camera.capture(self.rawCapture, format="bgr", use_video_port=True)
            return self.rawCapture.array
        else:
            ret, frame = self.camera.read()
            return frame if ret else None
    
    def send_frame(self, frame):
        """Invia frame compresso al server"""
        try:
            # Comprimi in JPEG
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            _, encoded = cv2.imencode('.jpg', frame, encode_param)
            data = encoded.tobytes()
            
            # Invia dimensione (4 bytes) + dati
            self.socket.send(struct.pack('>I', len(data)))
            self.socket.send(data)
            return True
        except Exception as e:
            print(f"[ERRORE] Invio frame: {e}")
            return False
    
    def receive_result(self):
        """Riceve il risultato dell'analisi dal server"""
        try:
            # Ricevi dimensione
            size_data = self.socket.recv(4)
            if not size_data:
                return None
            
            result_size = struct.unpack('>I', size_data)[0]
            
            # Ricevi dati
            result_data = b''
            while len(result_data) < result_size:
                chunk = self.socket.recv(min(result_size - len(result_data), 4096))
                if not chunk:
                    break
                result_data += chunk
            
            return pickle.loads(result_data)
        except Exception as e:
            print(f"[ERRORE] Ricezione risultato: {e}")
            return None
    
    def run(self):
        """Loop principale"""
        print("=" * 60)
        print("  DROWSINESS DETECTION - RASPBERRY CLIENT")
        print("=" * 60)
        
        # Inizializza camera
        if not self.init_camera():
            return
        
        # Connetti al server
        while not self.connected:
            if not self.connect():
                print(f"[INFO] Riprovo tra {RECONNECT_DELAY} secondi...")
                time.sleep(RECONNECT_DELAY)
        
        self.start_time = time.time()
        print("\n[INFO] Sistema attivo! Premi Ctrl+C per uscire")
        print("-" * 60)
        
        try:
            if self.use_picamera:
                # Stream continuo per PiCamera
                for frame_obj in self.camera.capture_continuous(
                    self.rawCapture, format="bgr", use_video_port=True):
                    
                    frame = frame_obj.array
                    self.rawCapture.truncate(0)
                    
                    if not self.process_frame(frame):
                        break
            else:
                # Loop per webcam USB
                while True:
                    frame = self.capture_frame()
                    if frame is None:
                        continue
                    
                    if not self.process_frame(frame):
                        break
        
        except KeyboardInterrupt:
            print("\n[INFO] Interruzione da tastiera")
        
        finally:
            self.cleanup()
    
    def process_frame(self, frame):
        """Processa un singolo frame"""
        # Invia al server
        if not self.send_frame(frame):
            self.connected = False
            return False
        
        # Ricevi risultato
        result = self.receive_result()
        if result is None:
            self.connected = False
            return False
        
        self.frames_sent += 1
        
        # Gestisci allarme locale
        if result.get("is_drowsy", False):
            self.alarm.start_alarm()
        else:
            self.alarm.stop_alarm()
        
        # Log periodico
        if self.frames_sent % 30 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frames_sent / elapsed if elapsed > 0 else 0
            status = "⚠️ DROWSY" if result.get("is_drowsy") else "✓ OK"
            yawn = " 🥱" if result.get("is_yawning") else ""
            print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                  f"FPS: {fps:.1f} | EAR: {result.get('ear', 0):.2f} | {status}{yawn}")
        
        # Visualizzazione locale se abilitata
        if config.DISPLAY_ENABLED:
            self.show_local_frame(frame, result)
        
        return True
    
    def show_local_frame(self, frame, result):
        """Mostra frame locale con overlay dei risultati"""
        display = frame.copy()
        
        # Overlay info
        ear = result.get('ear', 0)
        mar = result.get('mar', 0)
        
        cv2.putText(display, f"EAR: {ear:.2f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"MAR: {mar:.2f}", (10, 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if result.get("is_drowsy"):
            cv2.putText(display, "SONNOLENZA!", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Raspberry Client", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        
        return True
    
    def cleanup(self):
        """Pulizia risorse"""
        print("\n[INFO] Chiusura sistema...")
        
        if self.socket:
            self.socket.close()
        
        if self.use_picamera and self.camera:
            self.camera.close()
        elif self.camera:
            self.camera.release()
        
        self.alarm.stop_alarm()
        
        if config.DISPLAY_ENABLED:
            cv2.destroyAllWindows()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frames_sent / elapsed if elapsed > 0 else 0
        
        print("\n" + "=" * 60)
        print("STATISTICHE:")
        print(f"  Frame inviati: {self.frames_sent}")
        print(f"  Tempo totale: {elapsed:.1f}s")
        print(f"  FPS medio: {fps:.1f}")
        print("=" * 60)


# ===================== MAIN =====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Raspberry Client per Drowsiness Detection')
    parser.add_argument('--server', type=str, default=PC_SERVER_IP,
                       help=f'IP del server PC (default: {PC_SERVER_IP})')
    parser.add_argument('--port', type=int, default=PC_SERVER_PORT,
                       help=f'Porta server (default: {PC_SERVER_PORT})')
    parser.add_argument('--no-display', action='store_true',
                       help='Disabilita visualizzazione locale')
    parser.add_argument('--no-alarm', action='store_true',
                       help='Disabilita allarme')
    args = parser.parse_args()
    
    if args.no_display:
        config.DISPLAY_ENABLED = False
    if args.no_alarm:
        config.ALARM_ENABLED = False
    
    client = RaspberryClient(args.server, args.port)
    client.run()

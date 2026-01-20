#!/usr/bin/env python3
"""
raspberry_client.py - Client per Raspberry Pi
Cattura frame dalla camera e li invia al PC per l'elaborazione.
Comunicazione unidirezionale: solo invio frame, nessuna ricezione.

Compatibile con Raspberry Pi OS Bookworm (64-bit) - usa picamera2
"""

import socket
import struct
import cv2
import time
from datetime import datetime

# ===================== CONFIGURAZIONE =====================
PC_SERVER_IP = "192.168.1.100"  # <-- CAMBIA CON L'IP DEL TUO PC
PC_SERVER_PORT = 5555

# Camera
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 15

# Compressione JPEG (70 = buon compromesso qualità/banda)
JPEG_QUALITY = 70

# Connessione
CONNECTION_TIMEOUT = 10
RECONNECT_DELAY = 5


class RaspberryClient:
    """Client che cattura e invia frame al PC (solo trasmissione)"""
    
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.connected = False
        
        # Camera
        self.camera = None
        self.use_picamera2 = False
        
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
        """Inizializza la camera (picamera2 per Bookworm o USB fallback)"""
        print("[INFO] Inizializzazione camera...")
        
        # Prova picamera2 (Bookworm)
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            camera_config = self.camera.create_video_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
                controls={"FrameRate": CAMERA_FPS}
            )
            self.camera.configure(camera_config)
            self.camera.start()
            
            time.sleep(0.5)  # Warmup
            self.use_picamera2 = True
            print(f"[INFO] PiCamera2 inizializzata: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
            return True
            
        except ImportError:
            print("[WARN] picamera2 non disponibile, provo webcam USB...")
        except Exception as e:
            print(f"[WARN] Errore PiCamera2: {e}")
        
        # Fallback a webcam USB / OpenCV
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
            if not self.camera.isOpened():
                raise Exception("Camera non disponibile")
            
            self.use_picamera2 = False
            print("[INFO] Webcam USB/OpenCV inizializzata")
            return True
        except Exception as e:
            print(f"[ERRORE] Impossibile inizializzare camera: {e}")
            return False
    
    def capture_frame(self):
        """Cattura un frame dalla camera"""
        if self.use_picamera2:
            frame = self.camera.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
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
            
            # Invia: [4 byte dimensione] + [dati JPEG]
            self.socket.sendall(struct.pack('>I', len(data)) + data)
            return True
        except Exception as e:
            print(f"[ERRORE] Invio frame: {e}")
            return False
    
    def run(self):
        """Loop principale"""
        print("=" * 60)
        print("  DROWSINESS DETECTION - RASPBERRY STREAMER")
        print("  (Solo cattura e invio frame)")
        print("=" * 60)
        
        # Inizializza camera
        if not self.init_camera():
            return
        
        # Connetti al server (con retry)
        while not self.connected:
            if not self.connect():
                print(f"[INFO] Riprovo tra {RECONNECT_DELAY} secondi...")
                time.sleep(RECONNECT_DELAY)
        
        self.start_time = time.time()
        print("\n[INFO] Streaming attivo! Premi Ctrl+C per uscire")
        print("-" * 60)
        
        try:
            while self.connected:
                # Cattura frame
                frame = self.capture_frame()
                if frame is None:
                    continue
                
                # Invia al server
                if not self.send_frame(frame):
                    print("[WARN] Invio fallito, riconnessione...")
                    self.connected = False
                    break
                
                self.frames_sent += 1
                
                # Log periodico (ogni 30 frame)
                if self.frames_sent % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frames_sent / elapsed if elapsed > 0 else 0
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"Frame inviati: {self.frames_sent} | FPS: {fps:.1f}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Interruzione da tastiera")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Pulizia risorse"""
        print("\n[INFO] Chiusura...")
        
        if self.socket:
            self.socket.close()
        
        if self.use_picamera2 and self.camera:
            self.camera.stop()
            self.camera.close()
        elif self.camera:
            self.camera.release()
        
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
    
    parser = argparse.ArgumentParser(description='Raspberry Streamer per Drowsiness Detection')
    parser.add_argument('--server', type=str, default=PC_SERVER_IP,
                       help=f'IP del server PC (default: {PC_SERVER_IP})')
    parser.add_argument('--port', type=int, default=PC_SERVER_PORT,
                       help=f'Porta server (default: {PC_SERVER_PORT})')
    parser.add_argument('--quality', type=int, default=JPEG_QUALITY,
                       help=f'Qualità JPEG 1-100 (default: {JPEG_QUALITY})')
    args = parser.parse_args()
    
    if args.quality:
        JPEG_QUALITY = args.quality
    
    client = RaspberryClient(args.server, args.port)
    client.run()
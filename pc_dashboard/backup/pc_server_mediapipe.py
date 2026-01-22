#!/usr/bin/env python3
"""
pc_server_mediapipe.py - Server per drowsiness detection con MediaPipe
Riceve frame dal Raspberry, esegue l'analisi e mostra preview live.
Usa il modulo condiviso shared/drowsiness_detector_standalone.py
"""

import socket
import struct
import cv2
import numpy as np
import time
from datetime import datetime
import sys
import os

# Aggiungi la cartella parent al path per importare shared
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.drowsiness_analyzer import DrowsinessAnalyzer
from shared import config

# ===================== CONFIGURAZIONE SERVER =====================
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5555
BUFFER_SIZE = 65536


class PCServer:
    """Server TCP che riceve frame dal Raspberry"""
    
    def __init__(self):
        self.analyzer = DrowsinessAnalyzer()
        self.server_socket = None
        self.running = False
        self.show_preview = True
        
        self.frames_processed = 0
        self.start_time = None
    
    def start(self):
        """Avvia il server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((SERVER_HOST, SERVER_PORT))
        self.server_socket.listen(1)
        
        print("=" * 60)
        print("  DROWSINESS DETECTION - PC SERVER (MediaPipe)")
        print("=" * 60)
        print(f"[INFO] Server in ascolto su {SERVER_HOST}:{SERVER_PORT}")
        print("[INFO] In attesa di connessione dal Raspberry Pi...")
        print("[INFO] Preview video: ATTIVA (premi 'q' per uscire, 'p' per toggle)")
        print("=" * 60)
        
        self.running = True
        
        while self.running:
            try:
                client_socket, client_addr = self.server_socket.accept()
                print(f"\n[CONNESSO] Client connesso da {client_addr}")
                self.handle_client(client_socket)
            except KeyboardInterrupt:
                print("\n[INFO] Chiusura server...")
                break
            except Exception as e:
                print(f"[ERRORE] {e}")
        
        self.cleanup()
    
    def handle_client(self, client_socket):
        """Gestisce la connessione con un client"""
        self.start_time = time.time()
        self.frames_processed = 0
        
        # Reset contatori analyzer per nuova sessione
        self.analyzer.ear_counter = 0
        self.analyzer.yawn_counter = 0
        
        try:
            while self.running:
                # Ricevi dimensione frame (4 bytes, big-endian)
                size_data = self._recv_exact(client_socket, 4)
                if not size_data:
                    print("[INFO] Client disconnesso")
                    break
                
                frame_size = struct.unpack('>I', size_data)[0]
                
                # Ricevi frame completo
                frame_data = self._recv_exact(client_socket, frame_size)
                if not frame_data:
                    continue
                
                # Decodifica frame JPEG
                frame = cv2.imdecode(
                    np.frombuffer(frame_data, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )
                
                if frame is None:
                    continue
                
                # Analizza frame con MediaPipe (usando il detector condiviso)
                # Il metodo detect() restituisce: (frame_processato, ear, mar, is_drowsy, is_yawning)
                processed_frame, ear, mar, is_drowsy, is_yawning = self.analyzer.detect(frame)
                self.frames_processed += 1
                
                # Mostra preview locale
                if self.show_preview:
                    if not self.show_frame(processed_frame, ear, mar, is_drowsy, is_yawning):
                        break
                
                # Log periodico
                if self.frames_processed % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frames_processed / elapsed if elapsed > 0 else 0
                    status = "⚠️ DROWSY" if is_drowsy else "✓ OK"
                    yawn = " 🥱" if is_yawning else ""
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"FPS: {fps:.1f} | EAR: {ear:.2f} | "
                          f"MAR: {mar:.2f} | {status}{yawn}")
        
        except Exception as e:
            print(f"[ERRORE] Gestione client: {e}")
        
        finally:
            client_socket.close()
            if self.show_preview:
                cv2.destroyAllWindows()
    
    def _recv_exact(self, sock, size):
        """Riceve esattamente 'size' bytes"""
        data = b''
        while len(data) < size:
            chunk = sock.recv(min(size - len(data), BUFFER_SIZE))
            if not chunk:
                return None
            data += chunk
        return data
    
    def show_frame(self, frame, ear, mar, is_drowsy, is_yawning):
        """
        Mostra il frame con overlay.
        Il frame è già processato dal detector con i landmark disegnati.
        Ritorna False se l'utente vuole uscire.
        """
        display = frame.copy()
        
        # Info aggiuntive (il detector già disegna EAR/MAR se config.SHOW_EAR_MAR=True)
        # Aggiungiamo solo indicatore "NO FACE" se necessario
        # (il detector non lo gestisce esplicitamente)
        
        # Mostra finestra
        cv2.imshow("Drowsiness Detection - MediaPipe", display)
        
        # Gestione tasti
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False
            return False
        elif key == ord('p'):
            self.show_preview = not self.show_preview
            print(f"[INFO] Preview: {'ON' if self.show_preview else 'OFF'}")
        
        return True
    
    def cleanup(self):
        """Pulizia risorse"""
        if self.server_socket:
            self.server_socket.close()
        cv2.destroyAllWindows()
        
        stats = self.analyzer.get_statistics()
        print("\n" + "=" * 60)
        print("STATISTICHE FINALI:")
        print(f"  Frame elaborati: {self.frames_processed}")
        print(f"  Eventi sonnolenza: {stats['total_drowsy_events']}")
        print(f"  Eventi sbadiglio: {stats['total_yawn_events']}")
        print("=" * 60)


# ===================== MAIN =====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PC Server per Drowsiness Detection (MediaPipe)')
    parser.add_argument('--port', type=int, default=SERVER_PORT,
                       help=f'Porta server (default: {SERVER_PORT})')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disabilita preview video')
    args = parser.parse_args()
    
    SERVER_PORT = args.port
    
    server = PCServer()
    server.show_preview = not args.no_preview
    server.start()

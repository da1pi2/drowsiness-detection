#!/usr/bin/env python3
"""
pc_server.py - Server per elaborazione drowsiness detection
Riceve frame dal Raspberry, esegue l'analisi e mostra preview live.
Comunicazione unidirezionale: solo ricezione frame, nessun invio.
"""

import socket
import struct
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
from datetime import datetime

# ===================== CONFIGURAZIONE SERVER =====================
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5555
BUFFER_SIZE = 65536

# Soglie per il rilevamento
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
MAR_THRESHOLD = 0.6
YAWN_CONSEC_FRAMES = 15

# Path al modello dlib
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"


class DrowsinessAnalyzer:
    """Analizza i frame per rilevare sonnolenza"""
    
    def __init__(self):
        print("[INFO] Caricamento detector volti dlib...")
        self.detector = dlib.get_frontal_face_detector()
        
        print("[INFO] Caricamento shape predictor...")
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        
        # Indici landmark (68 punti facciali)
        self.LEFT_EYE = list(range(42, 48))
        self.RIGHT_EYE = list(range(36, 42))
        self.MOUTH = list(range(60, 68))
        
        # Contatori per frame consecutivi
        self.ear_counter = 0
        self.yawn_counter = 0
        self.total_drowsy_events = 0
        self.total_yawn_events = 0
        
        print("[INFO] Analyzer pronto!")
    
    def eye_aspect_ratio(self, eye):
        """Calcola Eye Aspect Ratio"""
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def mouth_aspect_ratio(self, mouth):
        """Calcola Mouth Aspect Ratio"""
        A = distance.euclidean(mouth[2], mouth[6])
        B = distance.euclidean(mouth[3], mouth[5])
        C = distance.euclidean(mouth[0], mouth[4])
        return (A + B) / (2.0 * C)
    
    def shape_to_np(self, shape):
        """Converte shape dlib in numpy array"""
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def analyze_frame(self, frame):
        """Analizza un frame e restituisce i risultati"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        result = {
            "ear": 0.0,
            "mar": 0.0,
            "is_drowsy": False,
            "is_yawning": False,
            "face_detected": False,
            "face_rect": None,
            "landmarks": None
        }
        
        for face in faces:
            result["face_detected"] = True
            result["face_rect"] = (face.left(), face.top(), face.width(), face.height())
            
            shape = self.predictor(gray, face)
            shape_np = self.shape_to_np(shape)
            
            left_eye = shape_np[self.LEFT_EYE]
            right_eye = shape_np[self.RIGHT_EYE]
            mouth = shape_np[self.MOUTH]
            
            # Calcola EAR (media occhi)
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            result["ear"] = (left_ear + right_ear) / 2.0
            
            # Calcola MAR
            result["mar"] = self.mouth_aspect_ratio(mouth)
            
            # Controlla sonnolenza
            if result["ear"] < EAR_THRESHOLD:
                self.ear_counter += 1
                if self.ear_counter >= EAR_CONSEC_FRAMES:
                    result["is_drowsy"] = True
                    if self.ear_counter == EAR_CONSEC_FRAMES:
                        self.total_drowsy_events += 1
            else:
                self.ear_counter = 0
            
            # Controlla sbadigli
            if result["mar"] > MAR_THRESHOLD:
                self.yawn_counter += 1
                if self.yawn_counter >= YAWN_CONSEC_FRAMES:
                    result["is_yawning"] = True
                    if self.yawn_counter == YAWN_CONSEC_FRAMES:
                        self.total_yawn_events += 1
            else:
                self.yawn_counter = 0
            
            # Salva landmarks per visualizzazione
            result["landmarks"] = {
                "left_eye": left_eye.tolist(),
                "right_eye": right_eye.tolist(),
                "mouth": mouth.tolist()
            }
            
            break  # Analizza solo il primo volto
        
        return result


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
        print("  DROWSINESS DETECTION - PC SERVER")
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
        
        try:
            while self.running:
                # Ricevi dimensione frame (4 bytes)
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
                
                # Analizza frame
                result = self.analyzer.analyze_frame(frame)
                self.frames_processed += 1
                
                # Mostra preview locale (NO invio al client)
                if self.show_preview:
                    if not self.show_frame(frame, result):
                        break
                
                # Log periodico
                if self.frames_processed % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frames_processed / elapsed if elapsed > 0 else 0
                    status = "⚠️ DROWSY" if result["is_drowsy"] else "✓ OK"
                    yawn = " 🥱" if result["is_yawning"] else ""
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"FPS: {fps:.1f} | EAR: {result['ear']:.2f} | {status}{yawn}")
        
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
    
    def show_frame(self, frame, result):
        """Mostra il frame con overlay - ritorna False se l'utente vuole uscire"""
        display = frame.copy()
        
        # Disegna rettangolo volto
        if result["face_rect"]:
            x, y, w, h = result["face_rect"]
            color = (0, 0, 255) if result["is_drowsy"] else (0, 255, 0)
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
        
        # Disegna landmarks (occhi e bocca)
        if result["landmarks"]:
            # Occhi (verde)
            for eye in [result["landmarks"]["left_eye"], result["landmarks"]["right_eye"]]:
                pts = np.array(eye, dtype=np.int32)
                cv2.polylines(display, [pts], True, (0, 255, 0), 1)
            
            # Bocca (giallo)
            mouth_pts = np.array(result["landmarks"]["mouth"], dtype=np.int32)
            cv2.polylines(display, [mouth_pts], True, (0, 255, 255), 1)
        
        # Info overlay
        cv2.putText(display, f"EAR: {result['ear']:.2f}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, f"MAR: {result['mar']:.2f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Alert sonnolenza
        if result["is_drowsy"]:
            cv2.putText(display, "SONNOLENZA!", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Alert sbadiglio
        if result["is_yawning"]:
            cv2.putText(display, "SBADIGLIO!", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Mostra finestra
        cv2.imshow("Drowsiness Detection - Live Preview", display)
        
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
        
        print("\n" + "=" * 60)
        print("STATISTICHE FINALI:")
        print(f"  Frame elaborati: {self.frames_processed}")
        print(f"  Eventi sonnolenza: {self.analyzer.total_drowsy_events}")
        print(f"  Eventi sbadiglio: {self.analyzer.total_yawn_events}")
        print("=" * 60)


# ===================== MAIN =====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PC Server per Drowsiness Detection')
    parser.add_argument('--port', type=int, default=SERVER_PORT,
                       help=f'Porta server (default: {SERVER_PORT})')
    parser.add_argument('--no-preview', action='store_true',
                       help='Disabilita preview video')
    args = parser.parse_args()
    
    SERVER_PORT = args.port
    
    server = PCServer()
    server.show_preview = not args.no_preview
    server.start()
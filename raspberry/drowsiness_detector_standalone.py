#!/usr/bin/env python3
"""
drowsiness_detector_standalone.py - Versione MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
from datetime import datetime
import raspberry.drowsiness_config_standalone as config

class DrowsinessDetector:
    """Detector di sonnolenza basato su MediaPipe Face Mesh"""
    
    def __init__(self):
        print("[INFO] Caricamento MediaPipe Face Mesh...")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        # max_num_faces=1 per velocità, refine_landmarks=True per iridi (opzionale)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # INDICI MEDIAPIPE (diversi da dlib)
        # Occhio Sinistro (punti chiave per EAR)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        # Occhio Destro
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Bocca (punti chiave per MAR: sopra, sotto, sinistra, destra)
        self.MOUTH = [13, 14, 61, 291] 
        
        # Contatori
        self.ear_counter = 0
        self.yawn_counter = 0
        self.total_drowsy_events = 0
        self.total_yawn_events = 0
        
        print("[INFO] Detector MediaPipe pronto!")
    
    def eye_aspect_ratio(self, landmarks, indices):
        """Calcola EAR dati i landmark specifici"""
        # Estrai i punti in base agli indici
        pts = [landmarks[i] for i in indices]
        
        # Calcola distanze verticali
        A = distance.euclidean(pts[1], pts[5])
        B = distance.euclidean(pts[2], pts[4])
        # Calcola distanza orizzontale
        C = distance.euclidean(pts[0], pts[3])
        
        if C == 0: return 0.0
        return (A + B) / (2.0 * C)
    
    def mouth_aspect_ratio(self, landmarks, indices):
        """Calcola MAR (distanza verticale / orizzontale)"""
        pts = [landmarks[i] for i in indices]
        
        # pts[0]=Top(13), pts[1]=Bottom(14), pts[2]=Left(61), pts[3]=Right(291)
        A = distance.euclidean(pts[0], pts[1]) # Verticale
        C = distance.euclidean(pts[2], pts[3]) # Orizzontale
        
        if C == 0: return 0.0
        return A / C
    
    def detect(self, frame):
        """
        Rileva sonnolenza nel frame usando MediaPipe.
        Returns: (frame_processato, ear, mar, is_drowsy, is_yawning)
        """
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa con MediaPipe
        results = self.face_mesh.process(rgb_frame)
        
        ear = 0.0
        mar = 0.0
        is_drowsy = False
        is_yawning = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Converti landmark normalizzati in pixel
                landmarks_np = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
                
                # Calcola EAR
                left_ear = self.eye_aspect_ratio(landmarks_np, self.LEFT_EYE)
                right_ear = self.eye_aspect_ratio(landmarks_np, self.RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0
                
                # Calcola MAR
                mar = self.mouth_aspect_ratio(landmarks_np, self.MOUTH)
                
                # --- LOGICA RILEVAMENTO (uguale a prima) ---
                
                # Sonnolenza
                if ear < config.EAR_THRESHOLD:
                    self.ear_counter += 1
                    if self.ear_counter >= config.EAR_CONSEC_FRAMES:
                        is_drowsy = True
                        if self.ear_counter == config.EAR_CONSEC_FRAMES:
                            self.total_drowsy_events += 1
                            self._log_event("DROWSINESS_DETECTED")
                            print(f"[⚠️ ALLARME] SONNOLENZA! Evento #{self.total_drowsy_events}")
                else:
                    self.ear_counter = 0
                
                # Sbadiglio
                if mar > config.MAR_THRESHOLD:
                    self.yawn_counter += 1
                    if self.yawn_counter >= config.YAWN_CONSEC_FRAMES:
                        is_yawning = True
                        if self.yawn_counter == config.YAWN_CONSEC_FRAMES:
                            self.total_yawn_events += 1
                            self._log_event("YAWN_DETECTED")
                            print(f"[🥱 INFO] SBADIGLIO! Evento #{self.total_yawn_events}")
                else:
                    self.yawn_counter = 0
                
                # --- DISEGNO ---
                if config.SHOW_LANDMARKS:
                    color_drowsy = (0, 0, 255) if is_drowsy else (0, 255, 0)
                    color_yawn = (0, 0, 255) if is_yawning else (0, 255, 255)
                    
                    # Disegna occhi
                    for idx in self.LEFT_EYE:
                        cv2.circle(frame, tuple(landmarks_np[idx]), 1, color_drowsy, -1)
                    for idx in self.RIGHT_EYE:
                        cv2.circle(frame, tuple(landmarks_np[idx]), 1, color_drowsy, -1)
                    # Disegna bocca
                    for idx in self.MOUTH:
                        cv2.circle(frame, tuple(landmarks_np[idx]), 2, color_yawn, -1)

                # Mostra Info a video
                if config.SHOW_EAR_MAR:
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                if is_drowsy:
                    cv2.putText(frame, "SONNOLENZA!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                if is_yawning:
                    cv2.putText(frame, "SBADIGLIO!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                break # Solo primo volto
        
        return frame, ear, mar, is_drowsy, is_yawning

    def _log_event(self, event_type):
        if not config.LOG_EVENTS: return
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(config.LOG_FILE, "a") as f:
                f.write(f"[{timestamp}] {event_type}\n")
        except Exception: pass

    def get_statistics(self):
        return {
            "total_drowsy_events": self.total_drowsy_events,
            "total_yawn_events": self.total_yawn_events
        }
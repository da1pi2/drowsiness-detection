#!/usr/bin/env python3
"""
drowsiness_detector.py - Classe per rilevamento sonnolenza
Versione standalone per Raspberry Pi
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from datetime import datetime
import raspberry.drowsiness_config_standalone as config


class DrowsinessDetector:
    """Detector di sonnolenza basato su EAR e MAR"""
    
    def __init__(self):
        print("[INFO] Caricamento detector volti dlib...")
        self.detector = dlib.get_frontal_face_detector()
        
        print("[INFO] Caricamento shape predictor...")
        self.predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR_PATH)
        
        # Indici landmark 68 punti facciali
        self.LEFT_EYE = list(range(42, 48))
        self.RIGHT_EYE = list(range(36, 42))
        self.MOUTH = list(range(60, 68))
        
        # Contatori
        self.ear_counter = 0
        self.yawn_counter = 0
        self.total_drowsy_events = 0
        self.total_yawn_events = 0
        
        print("[INFO] Detector pronto!")
    
    def eye_aspect_ratio(self, eye):
        """Calcola Eye Aspect Ratio (EAR)"""
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def mouth_aspect_ratio(self, mouth):
        """Calcola Mouth Aspect Ratio (MAR)"""
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
    
    def detect(self, frame):
        """
        Rileva sonnolenza nel frame.
        
        Returns:
            tuple: (frame_processato, ear, mar, is_drowsy, is_yawning)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        ear = 0.0
        mar = 0.0
        is_drowsy = False
        is_yawning = False
        
        for face in faces:
            # Estrai landmark
            shape = self.predictor(gray, face)
            shape_np = self.shape_to_np(shape)
            
            left_eye = shape_np[self.LEFT_EYE]
            right_eye = shape_np[self.RIGHT_EYE]
            mouth = shape_np[self.MOUTH]
            
            # Calcola EAR (media dei due occhi)
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Calcola MAR
            mar = self.mouth_aspect_ratio(mouth)
            
            # Disegna landmarks se abilitato
            if config.SHOW_LANDMARKS:
                cv2.polylines(frame, [cv2.convexHull(left_eye)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [cv2.convexHull(right_eye)], True, (0, 255, 0), 1)
                cv2.polylines(frame, [cv2.convexHull(mouth)], True, (0, 255, 255), 1)
            
            # Rettangolo volto
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            color = (0, 255, 0)  # Verde default
            
            # Controlla sonnolenza
            if ear < config.EAR_THRESHOLD:
                self.ear_counter += 1
                if self.ear_counter >= config.EAR_CONSEC_FRAMES:
                    is_drowsy = True
                    color = (0, 0, 255)  # Rosso
                    if self.ear_counter == config.EAR_CONSEC_FRAMES:
                        self.total_drowsy_events += 1
                        self._log_event("DROWSINESS_DETECTED")
                        print(f"[⚠️ ALLARME] SONNOLENZA RILEVATA! (evento #{self.total_drowsy_events})")
            else:
                self.ear_counter = 0
            
            # Controlla sbadigli
            if mar > config.MAR_THRESHOLD:
                self.yawn_counter += 1
                if self.yawn_counter >= config.YAWN_CONSEC_FRAMES:
                    is_yawning = True
                    if self.yawn_counter == config.YAWN_CONSEC_FRAMES:
                        self.total_yawn_events += 1
                        self._log_event("YAWN_DETECTED")
                        print(f"[🥱 INFO] Sbadiglio rilevato (evento #{self.total_yawn_events})")
            else:
                self.yawn_counter = 0
            
            # Disegna rettangolo volto
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Mostra valori EAR/MAR
            if config.SHOW_EAR_MAR:
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Alert visivi
            if is_drowsy:
                cv2.putText(frame, "SONNOLENZA!", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if is_yawning:
                cv2.putText(frame, "SBADIGLIO!", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            break  # Analizza solo il primo volto
        
        return frame, ear, mar, is_drowsy, is_yawning
    
    def _log_event(self, event_type):
        """Salva evento su file log"""
        if not config.LOG_EVENTS:
            return
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(config.LOG_FILE, "a") as f:
                f.write(f"[{timestamp}] {event_type}\n")
        except Exception as e:
            print(f"[WARN] Errore scrittura log: {e}")
    
    def get_statistics(self):
        """Ritorna statistiche correnti"""
        return {
            "total_drowsy_events": self.total_drowsy_events,
            "total_yawn_events": self.total_yawn_events
        }
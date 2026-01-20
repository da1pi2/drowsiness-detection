# drowsiness_detector.py - Classe per il rilevamento della sonnolenza

import cv2
import dlib
import numpy as np
from scipy.spatial import distance
from datetime import datetime
import drowsiness_config as config

class DrowsinessDetector:
    def __init__(self):
        # Inizializza il detector di volti di dlib
        print("[INFO] Caricamento detector volti...")
        self.detector = dlib.get_frontal_face_detector()
        
        # Inizializza il predictor dei landmark facciali
        print("[INFO] Caricamento shape predictor...")
        self.predictor = dlib.shape_predictor(config.SHAPE_PREDICTOR_PATH)
        
        # Indici dei landmark per occhi e bocca
        self.LEFT_EYE = list(range(42, 48))
        self.RIGHT_EYE = list(range(36, 42))
        self.MOUTH = list(range(60, 68))
        
        # Contatori
        self.ear_counter = 0
        self.yawn_counter = 0
        self.total_drowsy_events = 0
        self.total_yawn_events = 0
        
        # Stato
        self.alarm_on = False
        
    def eye_aspect_ratio(self, eye):
        """Calcola l'Eye Aspect Ratio (EAR)"""
        # Distanze verticali
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        # Distanza orizzontale
        C = distance.euclidean(eye[0], eye[3])
        # EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def mouth_aspect_ratio(self, mouth):
        """Calcola il Mouth Aspect Ratio (MAR) per rilevare sbadigli"""
        # Distanze verticali
        A = distance.euclidean(mouth[2], mouth[6])
        B = distance.euclidean(mouth[3], mouth[5])
        # Distanza orizzontale
        C = distance.euclidean(mouth[0], mouth[4])
        # MAR
        mar = (A + B) / (2.0 * C)
        return mar
    
    def shape_to_np(self, shape):
        """Converte shape di dlib in array numpy"""
        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def detect(self, frame):
        """
        Rileva sonnolenza nel frame
        Returns: (frame_processato, ear, mar, is_drowsy, is_yawning)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector(gray, 0)
        
        ear = 0.0
        mar = 0.0
        is_drowsy = False
        is_yawning = False
        
        # Per ogni volto rilevato
        for rect in rects:
            # Ottieni i landmark facciali
            shape = self.predictor(gray, rect)
            shape = self.shape_to_np(shape)
            
            # Estrai coordinate occhi e bocca
            left_eye = shape[self.LEFT_EYE]
            right_eye = shape[self.RIGHT_EYE]
            mouth = shape[self.MOUTH]
            
            # Calcola EAR per entrambi gli occhi
            left_ear = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Calcola MAR per la bocca
            mar = self.mouth_aspect_ratio(mouth)
            
            # Disegna contorni degli occhi e bocca
            if config.SHOW_LANDMARKS:
                left_eye_hull = cv2.convexHull(left_eye)
                right_eye_hull = cv2.convexHull(right_eye)
                mouth_hull = cv2.convexHull(mouth)
                
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 255), 1)
            
            # Disegna rettangolo intorno al volto
            (x, y, w, h) = (rect.left(), rect.top(), 
                           rect.width(), rect.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Controlla sonnolenza (occhi chiusi)
            if ear < config.EAR_THRESHOLD:
                self.ear_counter += 1
                
                if self.ear_counter >= config.EAR_CONSEC_FRAMES:
                    is_drowsy = True
                    if not self.alarm_on:
                        self.alarm_on = True
                        self.total_drowsy_events += 1
                        self.log_event("DROWSINESS DETECTED")
                    
                    cv2.putText(frame, "SONNOLENZA!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.ear_counter = 0
                self.alarm_on = False
            
            # Controlla sbadigli
            if mar > config.MAR_THRESHOLD:
                self.yawn_counter += 1
                
                if self.yawn_counter >= config.YAWN_CONSEC_FRAMES:
                    is_yawning = True
                    if self.yawn_counter == config.YAWN_CONSEC_FRAMES:
                        self.total_yawn_events += 1
                        self.log_event("YAWN DETECTED")
                    
                    cv2.putText(frame, "SBADIGLIO!", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                self.yawn_counter = 0
            
            # Mostra valori EAR e MAR
            if config.SHOW_EAR_MAR:
                cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (300, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Mostra statistiche
        cv2.putText(frame, f"Eventi Sonnolenza: {self.total_drowsy_events}", 
                   (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Eventi Sbadiglio: {self.total_yawn_events}", 
                   (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, ear, mar, is_drowsy, is_yawning
    
    def log_event(self, event_type):
        """Registra evento su file"""
        if config.LOG_EVENTS:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_line = f"[{timestamp}] {event_type}\n"
            with open(config.LOG_FILE, "a") as f:
                f.write(log_line)
            print(f"[LOG] {log_line.strip()}")
    
    def get_statistics(self):
        """Ritorna statistiche di utilizzo"""
        return {
            "total_drowsy_events": self.total_drowsy_events,
            "total_yawn_events": self.total_yawn_events
        }
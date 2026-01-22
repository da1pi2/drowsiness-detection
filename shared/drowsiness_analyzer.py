#!/usr/bin/env python3
"""
drowsiness_analyzer.py - Analyzer MediaPipe per drowsiness detection
Modulo condiviso usato sia dal PC Server che dal Raspberry standalone.
Compatibile con MediaPipe >= 0.10.0 (nuova API tasks)
"""

import cv2
import numpy as np
from scipy.spatial import distance
from datetime import datetime

# Import config dalla stessa cartella shared
try:
    from . import config  # Quando importato come package
except ImportError:
    import config  # Quando eseguito direttamente

# Prova la nuova API (MediaPipe >= 0.10.0)
import mediapipe as mp

# Verifica quale API è disponibile
USE_NEW_API = not hasattr(mp, 'solutions')


class DrowsinessAnalyzer:
    """Analyzer di sonnolenza basato su MediaPipe Face Mesh"""
    
    def __init__(self):
        print("[INFO] Caricamento MediaPipe Face Mesh...")
        
        if USE_NEW_API:
            self._init_new_api()
        else:
            self._init_legacy_api()
        
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
        
        print("[INFO] Analyzer MediaPipe pronto!")
    
    def _init_new_api(self):
        """Inizializza con la nuova API MediaPipe Tasks (>= 0.10.0)"""
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        
        print("[INFO] Usando MediaPipe Tasks API (nuova)")
        
        # Usa FaceLandmarker dalla nuova API
        base_options = mp_python.BaseOptions(
            model_asset_path=self._get_model_path()
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
        self.use_new_api = True
    
    def _init_legacy_api(self):
        """Inizializza con la vecchia API MediaPipe solutions (< 0.10.0)"""
        print("[INFO] Usando MediaPipe Solutions API (legacy)")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.use_new_api = False
    
    def _get_model_path(self):
        """Ottiene il path del modello per la nuova API"""
        import os
        
        # Cerca il modello nella cartella del modulo
        module_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(module_dir, "face_landmarker.task")
        
        if os.path.exists(model_path):
            return model_path
        
        # Scarica il modello se non esiste
        print("[INFO] Scaricamento modello face_landmarker.task...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"[INFO] Modello salvato in: {model_path}")
        
        return model_path
    
    def eye_aspect_ratio(self, landmarks, indices):
        """Calcola EAR dati i landmark specifici"""
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
        A = distance.euclidean(pts[0], pts[1])  # Verticale
        C = distance.euclidean(pts[2], pts[3])  # Orizzontale
        
        if C == 0: return 0.0
        return A / C
    
    def _process_frame_new_api(self, frame):
        """Processa frame con la nuova API"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.face_landmarker.detect(mp_image)
        
        if result.face_landmarks:
            h, w = frame.shape[:2]
            landmarks = result.face_landmarks[0]
            landmarks_np = np.array([(int(lm.x * w), int(lm.y * h)) for lm in landmarks])
            return landmarks_np
        return None
    
    def _process_frame_legacy_api(self, frame):
        """Processa frame con la vecchia API"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            h, w = frame.shape[:2]
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_np = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])
            return landmarks_np
        return None
    
    def detect(self, frame):
        """
        Rileva sonnolenza nel frame usando MediaPipe.
        Returns: (frame_processato, ear, mar, is_drowsy, is_yawning)
        """
        h, w = frame.shape[:2]
        
        # Processa con l'API appropriata
        if self.use_new_api:
            landmarks_np = self._process_frame_new_api(frame)
        else:
            landmarks_np = self._process_frame_legacy_api(frame)
        
        ear = 0.0
        mar = 0.0
        is_drowsy = False
        is_yawning = False
        
        if landmarks_np is not None:
            # Calcola EAR
            left_ear = self.eye_aspect_ratio(landmarks_np, self.LEFT_EYE)
            right_ear = self.eye_aspect_ratio(landmarks_np, self.RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            
            # Calcola MAR
            mar = self.mouth_aspect_ratio(landmarks_np, self.MOUTH)
            
            # --- LOGICA RILEVAMENTO ---
            
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
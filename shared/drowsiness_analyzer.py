#!/usr/bin/env python3
"""
drowsiness_analyzer.py - MediaPipe Analyzer for Drowsiness Detection
Shared module used by both the PC Server and the Standalone Raspberry Pi.
Compatible with MediaPipe >= 0.10.0 (new tasks API).
"""

import os
import json
import cv2
import numpy as np
from scipy.spatial import distance
from datetime import datetime

# SILENCE MEDIAPIPE LOGS
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ['GLOG_minloglevel'] = '3' 
os.environ["GLOG_logtostderr"] = '0'
os.environ['MAGLEV_HTTP_RESOLVER'] = '0'
os.environ['ABSL_LOG_LEVEL'] = 'error'

# Import config from the same 'shared' folder
try:
    from . import config  # When imported as a package
except ImportError:
    import config  # When executed directly

# Try the new API (MediaPipe >= 0.10.0)
import mediapipe as mp

# Check which API is available
USE_NEW_API = not hasattr(mp, 'solutions')


class DrowsinessAnalyzer:
    """Drowsiness analyzer based on MediaPipe Face Mesh"""
    
    def __init__(self):
        print("[INFO] Loading MediaPipe Face Mesh...")
        
        if USE_NEW_API:
            self._init_new_api()
        else:
            self._init_legacy_api()
        
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ear_config.json")
        self.ear_threshold = self.load_threshold()

        # MEDIAPIPE INDICES (different from dlib)
        # Left Eye (key points for EAR)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        # Right Eye
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Mouth (key points for MAR: top, bottom, left, right)
        self.MOUTH = [13, 14, 61, 291] 
        
        # Counters
        self.ear_counter = 0
        self.yawn_counter = 0
        self.total_drowsy_events = 0
        self.total_yawn_events = 0
        self.face_lost_counter = 0
        self.face_lost_threshold = config.CAMERA_FPS # 1 second of lost face
        self.drowsiness_score = 0.0

        print("[INFO] MediaPipe Analyzer ready!")
    
    def calculate_drowsiness_score(self, ear, mar):
        """
        Calcola uno score composito di sonnolenza (0-100).
        Combina EAR (occhi chiusi), durata occhi chiusi, MAR (sbadigli), e durata sbadigli.
        
        Args:
            ear: Eye Aspect Ratio (0-0.5)
            mar: Mouth Aspect Ratio (0-1.0)
            ear_counter: numero di frame consecutivi occhi chiusi
            yawn_counter: numero di frame consecutivi bocca aperta
        
        Returns:
            score: 0-100 (100 = massima sonnolenza)
        """
        # 1. Score EAR (occhi chiusi)
        # Normalizza EAR rispetto alla soglia
        ear_value_score = max(0, (self.ear_threshold - ear) / self.ear_threshold) * 100
        ear_value_score = min(ear_value_score, 100)
        
        # 2. Score durata occhi chiusi
        # Massimizza quando raggiunge EAR_CONSEC_FRAMES
        ear_duration_score = min(100, (self.total_drowsy_events / config.EAR_CONSEC_FRAMES) * 100)
        
        # Combina EAR value + duration (50% + 50%)
        ear_total_score = (ear_value_score * 0.5) + (ear_duration_score * 0.5)
        
        # 3. Score MAR (sbadigli)
        mar_value_score = max(0, (mar - config.MAR_THRESHOLD) / (1.0 - config.MAR_THRESHOLD)) * 100
        mar_value_score = min(mar_value_score, 100)
        
        # 4. Score durata sbadagli
        yawn_duration_score = min(100, (self.total_yawn_events / config.YAWN_CONSEC_FRAMES) * 100)
        
        # Combina MAR value + duration (50% + 50%)
        mar_total_score = (mar_value_score * 0.5) + (yawn_duration_score * 0.5)
        
        # 5. Punteggio finale
        # 80% occhi chiusi, 20% sbadigli
        drowsiness_score = (ear_total_score * 0.8) + (mar_total_score * 0.2)
        
        return drowsiness_score

    def _init_new_api(self):
        """Initializes with the new MediaPipe Tasks API (>= 0.10.0)"""
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        
        print("[INFO] Using MediaPipe Tasks API (New)")
        
        # Use FaceLandmarker from the new API
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
        """Initializes with the old MediaPipe Solutions API (< 0.10.0)"""
        print("[INFO] Using MediaPipe Solutions API (Legacy)")
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # To improve mouth and eye landmarks on standalone version
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.use_new_api = False
    
    def _get_model_path(self):
        """Gets the model path for the new API"""
        import os
        
        # Look for the model in the module folder
        module_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(module_dir, "face_landmarker.task")
        
        if os.path.exists(model_path):
            return model_path
        
        # Download the model if it doesn't exist
        print("[INFO] Downloading face_landmarker.task model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"[INFO] Model saved at: {model_path}")
        
        return model_path
    
    def load_threshold(self):
        """Loads the threshold from JSON or uses the default from config.py"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                    print(f"[INFO] Loaded custom EAR threshold: {data['threshold']:.2f}")
                    return data['threshold']
            except Exception as e:
                print(f"[WARN] Error loading config: {e}")
        return config.EAR_THRESHOLD

    def save_threshold(self, value):
        """Saves the new threshold to a shared JSON file"""
        self.ear_threshold = value
        try:
            with open(self.config_path, 'w') as f:
                json.dump({"threshold": value, "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f)
            print(f"[INFO] Threshold saved to {self.config_path}")
        except Exception as e:
            print(f"[ERROR] Could not save threshold: {e}")

    def eye_aspect_ratio(self, landmarks, indices):
        """Calculates EAR given specific landmarks """
        pts = [landmarks[i] for i in indices]
        
        # Calculate vertical distances
        A = distance.euclidean(pts[1], pts[5])
        B = distance.euclidean(pts[2], pts[4])
        # Calculate horizontal distance
        C = distance.euclidean(pts[0], pts[3])
        
        if C == 0: return 0.0
        return (A + B) / (2.0 * C)
    
    def mouth_aspect_ratio(self, landmarks, indices):
        """Calculates MAR (vertical distance / horizontal distance)"""
        pts = [landmarks[i] for i in indices]
        
        # pts[0]=Top(13), pts[1]=Bottom(14), pts[2]=Left(61), pts[3]=Right(291)
        A = distance.euclidean(pts[0], pts[1])  # Vertical
        C = distance.euclidean(pts[2], pts[3])  # Horizontal
        
        if C == 0: return 0.0
        return A / C
    
    def _process_frame_new_api(self, frame):
        """Processes frame with the new API"""
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
        """Processes frame with the legacy API"""
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
        Detects drowsiness in the frame using MediaPipe.
        Returns: (processed_frame, ear, mar, is_drowsy, is_yawning)
        """
        h, w = frame.shape[:2]
        
        # Process with the appropriate API
        if self.use_new_api:
            landmarks_np = self._process_frame_new_api(frame)
        else:
            landmarks_np = self._process_frame_legacy_api(frame)
        
        ear = 0.0
        mar = 0.0
        is_drowsy = False
        is_yawning = False
        face_detected = landmarks_np is not None
        
        if landmarks_np is not None:
            self.face_lost_counter = 0
            # Calculate EAR
            left_ear = self.eye_aspect_ratio(landmarks_np, self.LEFT_EYE)
            right_ear = self.eye_aspect_ratio(landmarks_np, self.RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            mar = self.mouth_aspect_ratio(landmarks_np, self.MOUTH)
            
            # --- DETECTION LOGIC ---
            
            # Calcola score composito
            new_drowsiness_score = self.calculate_drowsiness_score(ear, mar)

            # Drowsiness
            if ear < self.ear_threshold:
                self.ear_counter += 1
                if self.ear_counter >= config.EAR_CONSEC_FRAMES:
                    is_drowsy = True
                    if self.ear_counter == config.EAR_CONSEC_FRAMES:
                        self.total_drowsy_events += 1
                        self.drowsiness_score = new_drowsiness_score
                        self._log_event("DROWSINESS_DETECTED")
                        print(f"[⚠️ ALERT] DROWSINESS! Event #{self.total_drowsy_events} (Score: {self.drowsiness_score:.1f})")
            else:
                self.ear_counter = 0
            
            # Yawning
            if mar > config.MAR_THRESHOLD:
                self.yawn_counter += 1
                if self.yawn_counter >= config.YAWN_CONSEC_FRAMES:
                    is_yawning = True
                    if self.yawn_counter == config.YAWN_CONSEC_FRAMES:
                        self.total_yawn_events += 1
                        self.drowsiness_score = new_drowsiness_score
                        self._log_event("YAWN_DETECTED")
                        print(f"[🥱 INFO] YAWN! Event #{self.total_yawn_events} (Score: {self.drowsiness_score:.1f})")
            else:
                self.yawn_counter = 0
            
            # --- DRAWING ---
            if config.SHOW_LANDMARKS:
                color_drowsy = (0, 0, 255) if is_drowsy else (0, 255, 0)
                color_yawn = (0, 0, 255) if is_yawning else (0, 255, 255)
                
                # Draw Eyes 
                for idx in self.LEFT_EYE:
                    cv2.circle(frame, tuple(landmarks_np[idx]), 1, color_drowsy, -1)
                for idx in self.RIGHT_EYE:
                    cv2.circle(frame, tuple(landmarks_np[idx]), 1, color_drowsy, -1)
                # Draw Mouth 
                for idx in self.MOUTH:
                    cv2.circle(frame, tuple(landmarks_np[idx]), 2, color_yawn, -1)

            # Show Info on video
            if config.SHOW_EAR_MAR:
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, f"Score: {self.drowsiness_score:.1f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if is_drowsy:
                cv2.putText(frame, "DROWSINESS!", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if is_yawning:
                cv2.putText(frame, "YAWN!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # No face detected
            self.face_lost_counter += 1

            if self.face_lost_counter > self.face_lost_threshold:
                #face_detected = False
                self.face_lost_counter = 0
                # Disegno l'alert sul frame solo dopo il ritardo
                text = "!!! FACE LOST !!!"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.2
                thickness = 3
                # Calcola la posizione centrale
                (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

                x = (w - text_w) // 2
                y = (h + text_h) // 2  

                cv2.putText(frame, text, (x, y),
                            font, scale, (0, 0, 255), thickness)
            #else:
                #face_detected = True
            
        return frame, ear, mar, is_drowsy, is_yawning, face_detected, self.drowsiness_score

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
"""
Streamlit Dashboard - Demo Mode (PC Webcam)
Uses the computer's webcam, no Raspberry Pi required
"""

import streamlit as st
import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
import winsound
import threading
from datetime import datetime
from collections import deque

# Page configuration
st.set_page_config(
    page_title="Drowsiness Detection - Demo",
    page_icon="👁️",
    layout="wide"
)

# Path to dlib model
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6

# Load the model once
@st.cache_resource
def load_detector():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    return detector, predictor

# Global state
if 'data' not in st.session_state:
    st.session_state.data = {
        'ear': 0.0,
        'mar': 0.0,
        'is_drowsy': False,
        'is_yawning': False,
        'drowsy_count': 0,
        'yawn_count': 0,
        'events': deque(maxlen=15),
        'start_time': datetime.now(),
        'connected': True,
        'ear_counter': 0,
        'yawn_counter': 0
    }

if 'muted' not in st.session_state:
    st.session_state.muted = False

if 'running' not in st.session_state:
    st.session_state.running = True


def play_alert():
    """Emits alarm beep (Windows only)"""
    try:
        for _ in range(3):
            winsound.Beep(800, 200)
            time.sleep(0.1)
    except:
        # winsound is Windows only; this pass avoids errors on Linux/Mac
        pass


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[6])
    B = distance.euclidean(mouth[3], mouth[5])
    C = distance.euclidean(mouth[0], mouth[4])
    return (A + B) / (2.0 * C)


def shape_to_np(shape):
    coords = np.zeros((68, 2), dtype=int)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def analyze_frame_webcam(frame, detector, predictor):
    """Analyzes frame from webcam"""
    d = st.session_state.data
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)
    
    if not faces:
        d['is_drowsy'] = False
        d['is_yawning'] = False
        return frame
    
    face = faces[0]
    shape = predictor(gray, face)
    shape_np = shape_to_np(shape)
    
    # Landmark indices
    LEFT_EYE = list(range(42, 48))
    RIGHT_EYE = list(range(36, 42))
    MOUTH = list(range(60, 68))
    
    left_eye = shape_np[LEFT_EYE]
    right_eye = shape_np[RIGHT_EYE]
    mouth = shape_np[MOUTH]
    
    # Calculate EAR/MAR
    d['ear'] = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    d['mar'] = mouth_aspect_ratio(mouth)
    
    # Check Drowsiness
    prev_drowsy = d['is_drowsy']
    if d['ear'] < EAR_THRESHOLD:
        d['ear_counter'] += 1
        d['is_drowsy'] = d['ear_counter'] >= 20
    else:
        d['ear_counter'] = 0
        d['is_drowsy'] = False
    
    if d['is_drowsy'] and not prev_drowsy:
        d['drowsy_count'] += 1
        d['events'].appendleft(f"🔴 {datetime.now().strftime('%H:%M:%S')} - DROWSINESS (EAR: {d['ear']:.3f})")
        if not st.session_state.muted:
            threading.Thread(target=play_alert, daemon=True).start()
    
    # Check Yawning
    prev_yawn = d['is_yawning']
    if d['mar'] > MAR_THRESHOLD:
        d['yawn_counter'] += 1
        d['is_yawning'] = d['yawn_counter'] >= 15
    else:
        d['yawn_counter'] = 0
        d['is_yawning'] = False
    
    if d['is_yawning'] and not prev_yawn:
        d['yawn_count'] += 1
        d['events'].appendleft(f"🥱 {datetime.now().strftime('%H:%M:%S')} - YAWN (MAR: {d['mar']:.3f})")
    
    # Draw Overlay
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    color = (0, 0, 255) if d['is_drowsy'] else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Landmarks
    cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
    cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
    cv2.polylines(frame, [mouth], True, (0, 255, 255), 1)
    
    # Info Text
    cv2.putText(frame, f"EAR: {d['ear']:.2f}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"MAR: {d['mar']:.2f}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if d['is_drowsy']:
        cv2.putText(frame, "DROWSINESS!", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if d['is_yawning']:
        cv2.putText(frame, "YAWN!", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame


# === FIXED LAYOUT ===
st.title("👁️ Drowsiness Detection - DEMO (Webcam)")

col_ctrl1, col_ctrl2 = st.columns([8, 1])
with col_ctrl2:
    st.session_state.muted = st.checkbox("🔇 Mute", value=st.session_state.muted)

# Load model
detector, predictor = load_detector()

# Create FIXED placeholders (only once)
frame_col, events_col = st.columns([1, 1])

with frame_col:
    st.subheader("📹 Webcam")
    frame_placeholder = st.empty()

with events_col:
    st.subheader("📋 Recent Alerts")
    events_placeholder = st.empty()

alert_placeholder = st.empty()
metrics_placeholder = st.empty()
duration_placeholder = st.empty()

# Webcam feed
d = st.session_state.data

try:
    cap = cv2.VideoCapture(0)
    
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Error capturing frame from webcam")
            break
        
        # Resize frame to square 300x300
        frame = cv2.resize(frame, (300, 300))
        
        # Analyze frame
        frame = analyze_frame_webcam(frame, detector, predictor)
        
        # Update ONLY placeholders (no duplication)
        with frame_col:
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=300)
        
        # Alert banner
        with alert_placeholder.container():
            if d['is_drowsy']:
                st.error("⚠️ DROWSINESS DETECTED!", icon="🚨")
            elif d['is_yawning']:
                st.warning("🥱 Yawn Detected", icon="😴")
        
        # Metrics
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Status", "⚠️ ALERT" if d['is_drowsy'] else "✅ OK")
            with col2:
                st.metric("EAR", f"{d['ear']:.3f}")
            with col3:
                st.metric("MAR", f"{d['mar']:.3f}")
            with col4:
                st.metric("Events", f"🔴 {d['drowsy_count']}  🥱 {d['yawn_count']}")
        
        # Duration
        with duration_placeholder.container():
            duration = datetime.now() - d['start_time']
            st.caption(f"⏱️ Duration: {str(duration).split('.')[0]}")
        
        # Events list
        with events_col:
            with events_placeholder.container():
                if d['events']:
                    for event in list(d['events'])[:8]:
                        st.text(event)
                else:
                    st.caption("No events")
        
        time.sleep(0.05)

except Exception as e:
    st.error(f"Error: {e}")
finally:
    cap.release()
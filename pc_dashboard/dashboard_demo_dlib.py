"""
Streamlit Dashboard - Demo Mode (Webcam del PC)
Usa la webcam del computer, non serve Raspberry
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

# Configurazione pagina
st.set_page_config(
    page_title="Drowsiness Detection - Demo",
    page_icon="👁️",
    layout="wide"
)

# Path al modello dlib
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6

# Carica il modello una sola volta
@st.cache_resource
def load_detector():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    return detector, predictor

# Stato globale
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
    """Emette beep di allarme (Windows)"""
    try:
        for _ in range(3):
            winsound.Beep(800, 200)
            time.sleep(0.1)
    except:
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
    """Analizza frame dalla webcam"""
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
    
    # Indici landmark
    LEFT_EYE = list(range(42, 48))
    RIGHT_EYE = list(range(36, 42))
    MOUTH = list(range(60, 68))
    
    left_eye = shape_np[LEFT_EYE]
    right_eye = shape_np[RIGHT_EYE]
    mouth = shape_np[MOUTH]
    
    # Calcola EAR/MAR
    d['ear'] = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
    d['mar'] = mouth_aspect_ratio(mouth)
    
    # Controlla sonnolenza
    prev_drowsy = d['is_drowsy']
    if d['ear'] < EAR_THRESHOLD:
        d['ear_counter'] += 1
        d['is_drowsy'] = d['ear_counter'] >= 20
    else:
        d['ear_counter'] = 0
        d['is_drowsy'] = False
    
    if d['is_drowsy'] and not prev_drowsy:
        d['drowsy_count'] += 1
        d['events'].appendleft(f"🔴 {datetime.now().strftime('%H:%M:%S')} - SONNOLENZA (EAR: {d['ear']:.3f})")
        if not st.session_state.muted:
            threading.Thread(target=play_alert, daemon=True).start()
    
    # Controlla sbadiglio
    prev_yawn = d['is_yawning']
    if d['mar'] > MAR_THRESHOLD:
        d['yawn_counter'] += 1
        d['is_yawning'] = d['yawn_counter'] >= 15
    else:
        d['yawn_counter'] = 0
        d['is_yawning'] = False
    
    if d['is_yawning'] and not prev_yawn:
        d['yawn_count'] += 1
        d['events'].appendleft(f"🥱 {datetime.now().strftime('%H:%M:%S')} - SBADIGLIO (MAR: {d['mar']:.3f})")
    
    # Disegna overlay
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    color = (0, 0, 255) if d['is_drowsy'] else (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    
    # Landmarks
    cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
    cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
    cv2.polylines(frame, [mouth], True, (0, 255, 255), 1)
    
    # Info
    cv2.putText(frame, f"EAR: {d['ear']:.2f}", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"MAR: {d['mar']:.2f}", (10, 50),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if d['is_drowsy']:
        cv2.putText(frame, "SONNOLENZA!", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    if d['is_yawning']:
        cv2.putText(frame, "SBADIGLIO!", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return frame


# === LAYOUT FISSO ===
st.title("👁️ Drowsiness Detection - DEMO (Webcam)")

col_ctrl1, col_ctrl2 = st.columns([8, 1])
with col_ctrl2:
    st.session_state.muted = st.checkbox("🔇 Muto", value=st.session_state.muted)

# Carica modello
detector, predictor = load_detector()

# Crea placeholders FISSI (una sola volta)
frame_col, events_col = st.columns([1, 1])

with frame_col:
    st.subheader("📹 Webcam")
    frame_placeholder = st.empty()

with events_col:
    st.subheader("📋 Avvisi Recenti")
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
            st.error("Errore nel catturare il frame dalla webcam")
            break
        
        # Resize frame a quadrato 300x300
        frame = cv2.resize(frame, (300, 300))
        
        # Analizza frame
        frame = analyze_frame_webcam(frame, detector, predictor)
        
        # Aggiorna SOLO i placeholders (no duplicazione)
        with frame_col:
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=300)
        
        # Alert banner
        with alert_placeholder.container():
            if d['is_drowsy']:
                st.error("⚠️ SONNOLENZA RILEVATA!", icon="🚨")
            elif d['is_yawning']:
                st.warning("🥱 Sbadiglio Rilevato", icon="😴")
        
        # Metriche
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stato", "⚠️ ALLARME" if d['is_drowsy'] else "✅ OK")
            with col2:
                st.metric("EAR", f"{d['ear']:.3f}")
            with col3:
                st.metric("MAR", f"{d['mar']:.3f}")
            with col4:
                st.metric("Eventi", f"🔴 {d['drowsy_count']}  🥱 {d['yawn_count']}")
        
        # Durata
        with duration_placeholder.container():
            duration = datetime.now() - d['start_time']
            st.caption(f"⏱️ Durata: {str(duration).split('.')[0]}")
        
        # Events list
        with events_col:
            with events_placeholder.container():
                if d['events']:
                    for event in list(d['events'])[:8]:
                        st.text(event)
                else:
                    st.caption("Nessun evento")
        
        time.sleep(0.05)

except Exception as e:
    st.error(f"Errore: {e}")
finally:
    cap.release()

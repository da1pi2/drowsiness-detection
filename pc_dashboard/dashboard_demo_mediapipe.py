"""
Streamlit Dashboard - Demo Mode with MediaPipe
Uses the computer webcam to test drowsiness detection.
Does not require Raspberry Pi - uses MediaPipe instead of dlib.
"""

import streamlit as st
import cv2
import numpy as np
import time
import threading
from datetime import datetime
from collections import deque
import sys
import os

# Add parent directory to path to import shared module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from shared.drowsiness_analyzer import DrowsinessAnalyzer
except ImportError:
    st.error("Error: Could not import 'shared' module. Please ensure the directory structure is correct.")
    st.stop()

# Page Configuration
st.set_page_config(
    page_title="Drowsiness Detection - Demo MediaPipe",
    page_icon="👁️",
    layout="wide"
)

# Load analyzer only once
@st.cache_resource
def load_analyzer():
    return DrowsinessAnalyzer()


# Global State
if 'data' not in st.session_state:
    st.session_state.data = {
        'ear': 0.0,
        'mar': 0.0,
        'drowsiness_score': 0.0,  # NUOVO
        'face_detected': True,      # NUOVO
        'is_drowsy': False,
        'is_yawning': False,
        'drowsy_count': 0,
        'yawn_count': 0,
        'events': deque(maxlen=15),
        'start_time': datetime.now(),
        'connected': True
    }

if 'muted' not in st.session_state:
    st.session_state.muted = False

if 'running' not in st.session_state:
    st.session_state.running = True


def play_alert():
    """Emits alarm beep (Windows only)"""
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(800, 200)
            time.sleep(0.1)
    except:
        # winsound is Windows only; this pass avoids errors on Linux/Mac
        pass


def analyze_frame_webcam(frame, analyzer):
    """Analyzes frame from webcam using MediaPipe"""
    d = st.session_state.data
    
    # Use the analyzer's detect method - ORA RITORNA 7 VALORI
    processed_frame, ear, mar, is_drowsy, is_yawning, face_detected, drowsiness_score = analyzer.detect(frame)
    
    # Update state
    prev_drowsy = d['is_drowsy']
    prev_yawn = d['is_yawning']
    
    d['ear'] = ear
    d['mar'] = mar
    d['drowsiness_score'] = drowsiness_score  # NUOVO
    d['face_detected'] = face_detected          # NUOVO
    d['is_drowsy'] = is_drowsy
    d['is_yawning'] = is_yawning
    
    # Check for new drowsiness events
    if is_drowsy and not prev_drowsy:
        d['drowsy_count'] += 1
        d['events'].appendleft(f"🔴 {datetime.now().strftime('%H:%M:%S')} - DROWSINESS (Score: {drowsiness_score:.1f})")
        if not st.session_state.muted:
            threading.Thread(target=play_alert, daemon=True).start()
    
    # Check for new yawn events
    if is_yawning and not prev_yawn:
        d['yawn_count'] += 1
        d['events'].appendleft(f"🥱 {datetime.now().strftime('%H:%M:%S')} - YAWN (MAR: {mar:.3f})")
    
    return processed_frame


# === FIXED LAYOUT ===
st.title("👁️ Drowsiness Detection - DEMO (MediaPipe)")

col_ctrl1, col_ctrl2 = st.columns([8, 1])
with col_ctrl2:
    st.session_state.muted = st.checkbox("🔇 Mute", value=st.session_state.muted)

# Load analyzer
analyzer = load_analyzer()

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
        
        # Resize frame to square 300x300 for display consistency
        frame = cv2.resize(frame, (300, 300))
        
        # Analyze frame
        frame = analyze_frame_webcam(frame, analyzer)
        
        # Update ONLY placeholders (no duplication)
        with frame_col:
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=300)
        
        # Alert banner
        with alert_placeholder.container():
            if not d['face_detected']:  # NUOVO - alert face lost
                st.error("🚨 FACE NOT DETECTED - PLEASE ADJUST CAMERA", icon="👤")
            elif d['is_drowsy']:
                st.error("⚠️ DROWSINESS DETECTED!", icon="🚨")
            elif d['is_yawning']:
                st.warning("🥱 Yawn Detected", icon="😴")
            else:
                st.info("✅ Face detected - Monitoring active")  # NUOVO - feedback positivo
        
        # Metrics
        with metrics_placeholder.container():
            col1, col2, col3, col4, col5 = st.columns(5)  # AGGIUNTA 5a colonna
            with col1:
                st.metric("Status", "⚠️ ALERT" if d['is_drowsy'] else "✅ OK")
            with col2:
                st.metric("EAR", f"{d['ear']:.3f}")
            with col3:
                st.metric("MAR", f"{d['mar']:.3f}")
            with col4:
                st.metric("Score", f"{d['drowsiness_score']:.1f}")  # NUOVO
            with col5:
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
"""
Streamlit Dashboard - Demo Mode con MediaPipe
Usa la webcam del computer per testare il rilevamento sonnolenza
Non richiede Raspberry - usa MediaPipe invece di dlib
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

# Aggiungi la cartella parent al path per importare shared
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.drowsiness_analyzer import DrowsinessAnalyzer

# Configurazione pagina
st.set_page_config(
    page_title="Drowsiness Detection - Demo MediaPipe",
    page_icon="👁️",
    layout="wide"
)

# Carica l'analyzer una sola volta
@st.cache_resource
def load_analyzer():
    return DrowsinessAnalyzer()


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
        'connected': True
    }

if 'muted' not in st.session_state:
    st.session_state.muted = False

if 'running' not in st.session_state:
    st.session_state.running = True


def play_alert():
    """Emette beep di allarme (Windows)"""
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(800, 200)
            time.sleep(0.1)
    except:
        pass


def analyze_frame_webcam(frame, analyzer):
    """Analizza frame dalla webcam usando MediaPipe"""
    d = st.session_state.data
    
    # Usa il metodo detect dell'analyzer
    processed_frame, ear, mar, is_drowsy, is_yawning = analyzer.detect(frame)
    
    # Aggiorna stato
    prev_drowsy = d['is_drowsy']
    prev_yawn = d['is_yawning']
    
    d['ear'] = ear
    d['mar'] = mar
    d['is_drowsy'] = is_drowsy
    d['is_yawning'] = is_yawning
    
    # Controlla nuovi eventi sonnolenza
    if is_drowsy and not prev_drowsy:
        d['drowsy_count'] += 1
        d['events'].appendleft(f"🔴 {datetime.now().strftime('%H:%M:%S')} - SONNOLENZA (EAR: {ear:.3f})")
        if not st.session_state.muted:
            threading.Thread(target=play_alert, daemon=True).start()
    
    # Controlla nuovi eventi sbadiglio
    if is_yawning and not prev_yawn:
        d['yawn_count'] += 1
        d['events'].appendleft(f"🥱 {datetime.now().strftime('%H:%M:%S')} - SBADIGLIO (MAR: {mar:.3f})")
    
    return processed_frame


# === LAYOUT FISSO ===
st.title("👁️ Drowsiness Detection - DEMO (MediaPipe)")

col_ctrl1, col_ctrl2 = st.columns([8, 1])
with col_ctrl2:
    st.session_state.muted = st.checkbox("🔇 Muto", value=st.session_state.muted)

# Carica analyzer
analyzer = load_analyzer()

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
        frame = analyze_frame_webcam(frame, analyzer)
        
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

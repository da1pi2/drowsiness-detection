"""
Streamlit Dashboard + Server TCP per Raspberry
Server completo che riceve frame dal Raspberry e mostra dashboard Streamlit
Usa questo al posto di pc_server.py quando vuoi la dashboard web
"""

import streamlit as st
import socket
import struct
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
    page_title="Drowsiness Detection - Server",
    page_icon="👁️",
    layout="wide"
)

# ===================== CONFIGURAZIONE =====================
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5555
BUFFER_SIZE = 65536
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EAR_THRESHOLD = 0.25
EAR_CONSEC_FRAMES = 20
MAR_THRESHOLD = 0.6
YAWN_CONSEC_FRAMES = 15


# ===================== SHARED DATA =====================
class SharedData:
    """Dati condivisi thread-safe tra server TCP e Streamlit"""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.ear = 0.0
        self.mar = 0.0
        self.is_drowsy = False
        self.is_yawning = False
        self.drowsy_count = 0
        self.yawn_count = 0
        self.events = deque(maxlen=15)
        self.start_time = datetime.now()
        self.connected = False
        self.frames_processed = 0
        self._prev_drowsy = False
        self._prev_yawning = False
        self._trigger_alert = False
    
    def update(self, ear, mar, is_drowsy, is_yawning):
        with self.lock:
            self.ear = ear
            self.mar = mar
            self.is_drowsy = is_drowsy
            self.is_yawning = is_yawning
            self.frames_processed += 1
            self.connected = True
            
            # Nuovo evento sonnolenza
            if is_drowsy and not self._prev_drowsy:
                self.drowsy_count += 1
                self.events.appendleft(f"🔴 {datetime.now().strftime('%H:%M:%S')} - SONNOLENZA (EAR: {ear:.3f})")
                self._trigger_alert = True
            
            # Nuovo evento sbadiglio
            if is_yawning and not self._prev_yawning:
                self.yawn_count += 1
                self.events.appendleft(f"🥱 {datetime.now().strftime('%H:%M:%S')} - SBADIGLIO (MAR: {mar:.3f})")
            
            self._prev_drowsy = is_drowsy
            self._prev_yawning = is_yawning
    
    def get_snapshot(self):
        with self.lock:
            return {
                'ear': self.ear,
                'mar': self.mar,
                'is_drowsy': self.is_drowsy,
                'is_yawning': self.is_yawning,
                'drowsy_count': self.drowsy_count,
                'yawn_count': self.yawn_count,
                'events': list(self.events),
                'start_time': self.start_time,
                'connected': self.connected,
                'frames_processed': self.frames_processed
            }
    
    def should_alert(self):
        with self.lock:
            if self._trigger_alert:
                self._trigger_alert = False
                return True
            return False
    
    def disconnect(self):
        with self.lock:
            self.connected = False


# Istanza globale condivisa
shared_data = SharedData()


# ===================== ANALYZER =====================
class DrowsinessAnalyzer:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
        self.LEFT_EYE = list(range(42, 48))
        self.RIGHT_EYE = list(range(36, 42))
        self.MOUTH = list(range(60, 68))
        self.ear_counter = 0
        self.yawn_counter = 0
    
    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    
    def mouth_aspect_ratio(self, mouth):
        A = distance.euclidean(mouth[2], mouth[6])
        B = distance.euclidean(mouth[3], mouth[5])
        C = distance.euclidean(mouth[0], mouth[4])
        return (A + B) / (2.0 * C)
    
    def shape_to_np(self, shape):
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def analyze(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        ear, mar, is_drowsy, is_yawning = 0.0, 0.0, False, False
        
        for face in faces:
            shape = self.predictor(gray, face)
            shape_np = self.shape_to_np(shape)
            
            left_eye = shape_np[self.LEFT_EYE]
            right_eye = shape_np[self.RIGHT_EYE]
            mouth = shape_np[self.MOUTH]
            
            ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
            mar = self.mouth_aspect_ratio(mouth)
            
            # Sonnolenza
            if ear < EAR_THRESHOLD:
                self.ear_counter += 1
                is_drowsy = self.ear_counter >= EAR_CONSEC_FRAMES
            else:
                self.ear_counter = 0
            
            # Sbadiglio
            if mar > MAR_THRESHOLD:
                self.yawn_counter += 1
                is_yawning = self.yawn_counter >= YAWN_CONSEC_FRAMES
            else:
                self.yawn_counter = 0
            
            break
        
        return ear, mar, is_drowsy, is_yawning


# ===================== TCP SERVER =====================
def tcp_server_thread():
    """Thread che gestisce il server TCP per ricevere frame dal Raspberry"""
    global shared_data
    
    analyzer = DrowsinessAnalyzer()
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(1)
    
    print(f"[SERVER] In ascolto su {SERVER_HOST}:{SERVER_PORT}")
    
    while True:
        try:
            client_socket, addr = server_socket.accept()
            print(f"[SERVER] Client connesso da {addr}")
            shared_data.start_time = datetime.now()
            
            while True:
                # Ricevi dimensione frame
                size_data = b''
                while len(size_data) < 4:
                    chunk = client_socket.recv(4 - len(size_data))
                    if not chunk:
                        raise ConnectionError("Client disconnesso")
                    size_data += chunk
                
                frame_size = struct.unpack('>I', size_data)[0]
                
                # Ricevi frame
                frame_data = b''
                while len(frame_data) < frame_size:
                    chunk = client_socket.recv(min(frame_size - len(frame_data), BUFFER_SIZE))
                    if not chunk:
                        raise ConnectionError("Client disconnesso")
                    frame_data += chunk
                
                # Decodifica e analizza
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    ear, mar, is_drowsy, is_yawning = analyzer.analyze(frame)
                    shared_data.update(ear, mar, is_drowsy, is_yawning)
        
        except Exception as e:
            print(f"[SERVER] Errore: {e}")
            shared_data.disconnect()
        
        finally:
            try:
                client_socket.close()
            except:
                pass
            print("[SERVER] In attesa di nuova connessione...")


# ===================== AUDIO ALERT =====================
def play_alert():
    try:
        for _ in range(3):
            winsound.Beep(800, 200)
            time.sleep(0.1)
    except:
        pass


# ===================== STREAMLIT UI =====================

# Avvia server TCP in thread separato (una sola volta)
if 'server_started' not in st.session_state:
    st.session_state.server_started = True
    threading.Thread(target=tcp_server_thread, daemon=True).start()

if 'muted' not in st.session_state:
    st.session_state.muted = False

# Layout
st.title("👁️ Drowsiness Detection - Server (Raspberry)")

col_ctrl1, col_ctrl2 = st.columns([8, 1])
with col_ctrl2:
    st.session_state.muted = st.checkbox("🔇 Muto", value=st.session_state.muted)

# Info connessione
st.info(f"📡 Server TCP in ascolto su porta **{SERVER_PORT}** - Connetti il Raspberry a questo PC")

# Placeholders FISSI
frame_col, events_col = st.columns([1, 1])

with frame_col:
    st.subheader("📹 Stato Connessione")
    status_placeholder = st.empty()

with events_col:
    st.subheader("📋 Avvisi Recenti")
    events_placeholder = st.empty()

alert_placeholder = st.empty()
metrics_placeholder = st.empty()
duration_placeholder = st.empty()

# Loop principale
try:
    while True:
        d = shared_data.get_snapshot()
        
        # Status connessione
        with status_placeholder.container():
            if d['connected']:
                st.success(f"🟢 Raspberry Connesso - Frame: {d['frames_processed']}")
            else:
                st.warning("🟡 In attesa di connessione dal Raspberry...")
        
        # Alert banner
        with alert_placeholder.container():
            if d['is_drowsy']:
                st.error("⚠️ SONNOLENZA RILEVATA!", icon="🚨")
            elif d['is_yawning']:
                st.warning("🥱 Sbadiglio Rilevato", icon="😴")
        
        # Audio alert (solo su nuovo evento)
        if shared_data.should_alert() and not st.session_state.muted:
            threading.Thread(target=play_alert, daemon=True).start()
        
        # Metriche
        with metrics_placeholder.container():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                status = "⚠️ ALLARME" if d['is_drowsy'] else ("✅ OK" if d['connected'] else "⏳ Attesa")
                st.metric("Stato", status)
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
        with events_placeholder.container():
            if d['events']:
                for event in d['events'][:8]:
                    st.text(event)
            else:
                st.caption("Nessun evento")
        
        time.sleep(0.1)

except Exception as e:
    st.error(f"Errore: {e}")

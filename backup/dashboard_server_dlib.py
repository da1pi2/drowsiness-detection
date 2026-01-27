"""
Streamlit Dashboard + TCP Server for Raspberry Pi
Complete server that receives frames from Raspberry Pi and displays a Streamlit dashboard.
CORRECTED: Now displays the video stream.
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

# Page Configuration
st.set_page_config(
    page_title="Drowsiness Detection - Server",
    page_icon="👁️",
    layout="wide"
)

# ===================== CONFIGURATION =====================
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
    """Thread-safe shared data between TCP server and Streamlit"""
    
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
        self.frame = None 
        self._prev_drowsy = False
        self._prev_yawning = False
        self._trigger_alert = False
    
    def update(self, ear, mar, is_drowsy, is_yawning, frame):
        with self.lock:
            self.ear = ear
            self.mar = mar
            self.is_drowsy = is_drowsy
            self.is_yawning = is_yawning
            self.frame = frame 
            self.frames_processed += 1
            self.connected = True
            
            # New Drowsiness Event
            if is_drowsy and not self._prev_drowsy:
                self.drowsy_count += 1
                self.events.appendleft(f"🔴 {datetime.now().strftime('%H:%M:%S')} - DROWSINESS (EAR: {ear:.3f})")
                self._trigger_alert = True
            
            # New Yawn Event
            if is_yawning and not self._prev_yawning:
                self.yawn_count += 1
                self.events.appendleft(f"🥱 {datetime.now().strftime('%H:%M:%S')} - YAWN (MAR: {mar:.3f})")
            
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
                'frames_processed': self.frames_processed,
                'frame': self.frame.copy() if self.frame is not None else None # <--- NEW: Return frame copy
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
            self.frame = None

# Global shared instance
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
            # Draw rectangle around face (Optional, for visual debug)
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            shape = self.predictor(gray, face)
            shape_np = self.shape_to_np(shape)
            
            left_eye = shape_np[self.LEFT_EYE]
            right_eye = shape_np[self.RIGHT_EYE]
            mouth = shape_np[self.MOUTH]
            
            ear = (self.eye_aspect_ratio(left_eye) + self.eye_aspect_ratio(right_eye)) / 2.0
            mar = self.mouth_aspect_ratio(mouth)
            
            # Draw eye and mouth contours on the original frame
            cv2.polylines(frame, [left_eye], True, (0, 255, 255), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 255), 1)
            cv2.polylines(frame, [mouth], True, (0, 0, 255), 1)

            # Drowsiness check
            if ear < EAR_THRESHOLD:
                self.ear_counter += 1
                is_drowsy = self.ear_counter >= EAR_CONSEC_FRAMES
            else:
                self.ear_counter = 0
            
            # Yawn check
            if mar > MAR_THRESHOLD:
                self.yawn_counter += 1
                is_yawning = self.yawn_counter >= YAWN_CONSEC_FRAMES
            else:
                self.yawn_counter = 0
            
            break
        
        return ear, mar, is_drowsy, is_yawning, frame

# ===================== TCP SERVER =====================
def tcp_server_thread():
    """Thread handling the TCP server to receive frames from the Raspberry"""
    global shared_data
    
    analyzer = DrowsinessAnalyzer()
    
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(1)
    
    print(f"[SERVER] Listening on {SERVER_HOST}:{SERVER_PORT}")
    
    while True:
        try:
            client_socket, addr = server_socket.accept()
            print(f"[SERVER] Client connected from {addr}")
            shared_data.start_time = datetime.now()
            
            while True:
                # Receive frame size
                size_data = b''
                while len(size_data) < 4:
                    chunk = client_socket.recv(4 - len(size_data))
                    if not chunk:
                        raise ConnectionError("Client disconnected")
                    size_data += chunk
                
                frame_size = struct.unpack('>I', size_data)[0]
                
                # Receive frame
                frame_data = b''
                while len(frame_data) < frame_size:
                    chunk = client_socket.recv(min(frame_size - len(frame_data), BUFFER_SIZE))
                    if not chunk:
                        raise ConnectionError("Client disconnected")
                    frame_data += chunk
                
                # Decode and analyze
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Analysis + Draw landmarks on frame
                    ear, mar, is_drowsy, is_yawning, processed_frame = analyzer.analyze(frame)
                    # Update shared data INCLUDING FRAME
                    shared_data.update(ear, mar, is_drowsy, is_yawning, processed_frame)
        
        except Exception as e:
            print(f"[SERVER] Error: {e}")
            shared_data.disconnect()
        
        finally:
            try:
                client_socket.close()
            except:
                pass
            print("[SERVER] Waiting for new connection...")

# ===================== AUDIO ALERT =====================
def play_alert():
    try:
        for _ in range(3):
            winsound.Beep(800, 200)
            time.sleep(0.1)
    except:
        pass

# ===================== STREAMLIT UI =====================

# Start TCP server in a separate thread (only once)
if 'server_started' not in st.session_state:
    st.session_state.server_started = True
    threading.Thread(target=tcp_server_thread, daemon=True).start()

if 'muted' not in st.session_state:
    st.session_state.muted = False

# Layout
st.title("👁️ Drowsiness Detection - Server (Raspberry)")

col_ctrl1, col_ctrl2 = st.columns([8, 1])
with col_ctrl2:
    st.session_state.muted = st.checkbox("🔇 Mute", value=st.session_state.muted)

# Connection Info
st.info(f"📡 TCP Server listening on port **{SERVER_PORT}** - Connect the Raspberry Pi to this PC")

# Fixed Placeholders
frame_col, events_col = st.columns([3, 2]) # Slightly widened video space

with frame_col:
    st.subheader("📹 Video Stream")
    video_placeholder = st.empty() # Placeholder for video

with events_col:
    st.subheader("📋 Status & Log")
    status_placeholder = st.empty()
    alert_placeholder = st.empty()
    metrics_placeholder = st.empty()
    events_placeholder = st.empty()

# Main Loop
try:
    while True:
        d = shared_data.get_snapshot()
        
        # 1. VIDEO DISPLAY
        with video_placeholder.container():
            if d['frame'] is not None:
                # Convert BGR (OpenCV) -> RGB (Streamlit)
                frame_rgb = cv2.cvtColor(d['frame'], cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, channels="RGB", use_container_width=True)
            else:
                st.image("https://via.placeholder.com/640x480.png?text=Waiting+for+video...", use_container_width=True)

        # 2. CONNECTION STATUS
        with status_placeholder.container():
            if d['connected']:
                st.success(f"🟢 Connected - Frames processed: {d['frames_processed']}")
            else:
                st.warning("🟡 Waiting for Raspberry Pi...")
        
        # 3. ALERT BANNER
        with alert_placeholder.container():
            if d['is_drowsy']:
                st.error("⚠️ DROWSINESS DETECTED!", icon="🚨")
            elif d['is_yawning']:
                st.warning("🥱 Yawn Detected", icon="😴")
            else:
                st.markdown("---") # Spacer if everything is OK
        
        # Audio alert
        if shared_data.should_alert() and not st.session_state.muted:
            threading.Thread(target=play_alert, daemon=True).start()
        
        # 4. METRICS
        with metrics_placeholder.container():
            c1, c2 = st.columns(2)
            c1.metric("EAR (Eyes)", f"{d['ear']:.2f}", delta="-Low" if d['ear'] < EAR_THRESHOLD else None)
            c2.metric("MAR (Mouth)", f"{d['mar']:.2f}", delta="+High" if d['mar'] > MAR_THRESHOLD else None)
        
        # 5. EVENTS LIST
        with events_placeholder.container():
            st.write("##### Event History")
            if d['events']:
                for event in d['events'][:5]:
                    st.text(event)
            else:
                st.caption("No recent events")
        
        # Small pause to avoid overloading Streamlit CPU render
        time.sleep(0.05)

except Exception as e:
    st.error(f"UI Error: {e}")
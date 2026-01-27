"""
Streamlit Dashboard + TCP Server (MediaPipe) - OPTIMIZED VERSION
Fixes: FPS calculation, buffering latency with frame skipping
Now receives system stats from Raspberry Pi
"""
import socket
import struct
import cv2
import numpy as np
import streamlit as st
import threading
import time
import json
from datetime import datetime
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from shared.drowsiness_analyzer import DrowsinessAnalyzer
    from shared import config
except ImportError:
    st.error("Error: Could not import 'shared' module.")
    st.stop()

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5555
BUFFER_SIZE = 65536

st.set_page_config(page_title="Drowsiness Server - MediaPipe", page_icon="👁️", layout="wide")

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.ear = 0.0
        self.mar = 0.0
        self.is_drowsy = False
        self.is_yawning = False
        self.face_detected = True
        self.drowsy_count = 0
        self.yawn_count = 0
        self.events = deque(maxlen=20)
        self.start_time = datetime.now()
        self.connected = False
        self.frames_processed = 0
        self.last_frame = None
        self._prev_drowsy = False
        self._prev_yawn = False
        self._trigger_alert = False
        # Raspberry Pi stats
        self.rpi_cpu_temp = 0.0
        self.rpi_cpu_usage = 0.0
        self.rpi_ram_usage = 0.0
        self.rpi_fps = 0.0
        self.rpi_ip = ""

    def update(self, ear, mar, is_drowsy, is_yawning, face_detected, frame_rgb):
        with self.lock:
            self.ear = ear
            self.mar = mar
            self.is_drowsy = is_drowsy
            self.is_yawning = is_yawning
            self.face_detected = face_detected
            self.frames_processed += 1
            self.connected = True
            self.last_frame = frame_rgb

            if is_drowsy and not self._prev_drowsy:
                self.drowsy_count += 1
                self.events.appendleft(f"🔴 {datetime.now().strftime('%H:%M:%S')} - DROWSINESS (EAR: {ear:.3f})")
                self._trigger_alert = True
            if is_yawning and not self._prev_yawn:
                self.yawn_count += 1
                self.events.appendleft(f"🥱 {datetime.now().strftime('%H:%M:%S')} - YAWN (MAR: {mar:.3f})")
            
            self._prev_drowsy = is_drowsy
            self._prev_yawn = is_yawning

    def update_rpi_stats(self, cpu_temp, cpu_usage, ram_usage, fps, ip=""):
        with self.lock:
            self.rpi_cpu_temp = cpu_temp
            self.rpi_cpu_usage = cpu_usage
            self.rpi_ram_usage = ram_usage
            self.rpi_fps = fps
            if ip:
                self.rpi_ip = ip

    def snapshot(self):
        with self.lock:
            return {
                "ear": self.ear,
                "mar": self.mar,
                "is_drowsy": self.is_drowsy,
                "is_yawning": self.is_yawning,
                "drowsy_count": self.drowsy_count,
                "yawn_count": self.yawn_count,
                "face_detected": self.face_detected,
                "events": list(self.events),
                "start_time": self.start_time,
                "connected": self.connected,
                "frames_processed": self.frames_processed,
                "last_frame": self.last_frame.copy() if self.last_frame is not None else None,
                "rpi_cpu_temp": self.rpi_cpu_temp,
                "rpi_cpu_usage": self.rpi_cpu_usage,
                "rpi_ram_usage": self.rpi_ram_usage,
                "rpi_fps": self.rpi_fps,
                "rpi_ip": self.rpi_ip,
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
            self.last_frame = None
            self.rpi_fps = 0.0
            self.rpi_cpu_temp = 0.0
            self.rpi_cpu_usage = 0.0
            self.rpi_ram_usage = 0.0

state = SharedState()

def _recv_exact(sock, size):
    """Receive exact number of bytes"""
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(size - len(data), BUFFER_SIZE))
        if not chunk:
            return None
        data += chunk
    return data

def tcp_server_loop():
    """
    Receives frames + stats from Raspberry Pi.
    Protocol: [4 bytes stats_size][JSON stats][4 bytes frame_size][JPEG frame]
    """
    analyzer = DrowsinessAnalyzer()
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(1)
    print(f"[SERVER] Listening on {SERVER_HOST}:{SERVER_PORT}")
    
    while True:
        client_socket = None
        try:
            client_socket, addr = server_socket.accept()
            analyzer.ear_threshold = analyzer.load_threshold()
            print(f"[SERVER] Client connected from {addr}")
            state.start_time = datetime.now()
            state.update_rpi_stats(0, 0, 0, 0, addr[0])  # Store client IP
            analyzer.ear_counter = 0
            analyzer.yawn_counter = 0
            
            while True:
                # 1. Read stats JSON size
                stats_size_data = _recv_exact(client_socket, 4)
                if not stats_size_data:
                    raise ConnectionError("Client disconnected")
                stats_size = struct.unpack('>I', stats_size_data)[0]
                
                # 2. Read stats JSON
                stats_data = _recv_exact(client_socket, stats_size)
                if not stats_data:
                    raise ConnectionError("Incomplete stats")
                
                try:
                    rpi_stats = json.loads(stats_data.decode('utf-8'))
                    state.update_rpi_stats(
                        rpi_stats.get('cpu_temp', 0),
                        rpi_stats.get('cpu_usage', 0),
                        rpi_stats.get('ram_usage', 0),
                        rpi_stats.get('fps', 0)
                    )
                except:
                    pass
                
                # 3. Read frame size
                frame_size_data = _recv_exact(client_socket, 4)
                if not frame_size_data:
                    raise ConnectionError("Client disconnected")
                frame_size = struct.unpack('>I', frame_size_data)[0]
                
                # 4. Read frame data
                frame_data = _recv_exact(client_socket, frame_size)
                if not frame_data:
                    raise ConnectionError("Incomplete frame")
                
                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                
                # Process with MediaPipe
                processed, ear, mar, is_drowsy, is_yawning, face_detected, _ = analyzer.detect(frame)
                
                # Prepare preview
                preview = cv2.resize(processed, (320, 240))
                
                state.update(ear, mar, is_drowsy, is_yawning, face_detected, preview)
                
        except Exception as e:
            print(f"[SERVER] Error: {e}")
            state.disconnect()
        finally:
            if client_socket:
                try:
                    client_socket.close()
                except:
                    pass
            print("[SERVER] Waiting for new connection...")

def play_beep():
    try:
        import winsound
        winsound.Beep(800, 200)
    except:
        pass

# Start server thread
if 'server_thread' not in st.session_state:
    st.session_state.server_thread = threading.Thread(target=tcp_server_loop, daemon=True)
    st.session_state.server_thread.start()

if 'muted' not in st.session_state:
    st.session_state.muted = False

# ===================== UI LAYOUT =====================
st.title("👁️ Drowsiness Detection - Server (MediaPipe)")
st.caption(f"📡 TCP Port {SERVER_PORT} - Connect the Raspberry Pi to the PC")

ctrl_col, mute_col = st.columns([8, 1])
with mute_col:
    st.session_state.muted = st.checkbox("🔇 Mute", value=st.session_state.muted)

frame_col, events_col = st.columns([2, 1])

with frame_col:
    st.subheader("Live Preview")
    frame_placeholder = st.empty()

with events_col:
    st.subheader("Recent Alerts")
    events_placeholder = st.empty()

alert_placeholder = st.empty()
info_placeholder = st.empty()
metrics_placeholder = st.empty()

while True:
    snap = state.snapshot()
    
    # Connection Status
    with info_placeholder.container():
        if snap["connected"]:
            st.success(f"🟢 Raspberry Connected ({snap['rpi_ip']}) - Frames: {snap['frames_processed']}")
        else:
            st.warning("🟡 Waiting for Raspberry Pi connection...")
    
    # Video Feed
    if snap["last_frame"] is not None:
        frame_placeholder.image(snap["last_frame"], channels="RGB", width=320)
    else:
        frame_placeholder.image("https://via.placeholder.com/300x300.png?text=Waiting+for+Video", width=320)
    
    # Alerts
    with alert_placeholder.container():
        if snap["connected"] and not snap.get("face_detected", True):
            st.error("🚨 FACE NOT DETECTED - PLEASE ADJUST CAMERA", icon="👤")
        elif snap["is_drowsy"]:
            st.error("⚠️ DROWSINESS DETECTED!", icon="🚨")
        elif snap["is_yawning"]:
            st.warning("🥱 Yawn Detected", icon="😴")
        else:
            st.markdown("---")
    
    # Audio Alert
    if state.should_alert() and not st.session_state.muted:
        threading.Thread(target=play_beep, daemon=True).start()
    
    # Metrics
    with metrics_placeholder.container():
        c1, c2, c3, c4 = st.columns(4)
        status_text = "⚠️ ALERT" if snap["is_drowsy"] else ("✅ OK" if snap["connected"] else "⏳ Waiting")
        
        c1.metric("Status", status_text)
        c2.metric("EAR", f"{snap['ear']:.3f}")
        c3.metric("MAR", f"{snap['mar']:.3f}")
        c4.metric("Events", f"🔴 {snap['drowsy_count']}  🥱 {snap['yawn_count']}")
    
    # Raspberry Pi System Stats
    with st.container():
        if snap["connected"]:
            st.caption("🍓 Raspberry Pi Stats")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("RPi FPS", f"{snap['rpi_fps']:.1f}")
            r2.metric("RPi CPU", f"{snap['rpi_cpu_usage']:.1f}%")
            r3.metric("RPi RAM", f"{snap['rpi_ram_usage']:.1f}%")
            r4.metric("RPi Temp", f"{snap['rpi_cpu_temp']:.1f}°C" if snap['rpi_cpu_temp'] > 0 else "N/A")
    
    # Event Log
    with events_placeholder.container():
        if snap["events"]:
            for event in snap["events"][:8]:
                st.text(event)
        else:
            st.caption("No events yet")
    
    time.sleep(0.05)
"""
Streamlit Dashboard + TCP Server (MediaPipe)
Receives frames from Raspberry Pi, analyzes them with MediaPipe (shared module),
and displays the web dashboard.
"""
import socket
import struct
import cv2
import numpy as np
import streamlit as st
import threading
import time
from datetime import datetime
from collections import deque
import sys
import os

# Import shared analyzer/config
# Assumes the 'shared' folder is in the parent directory of this script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from shared.drowsiness_analyzer import DrowsinessAnalyzer
    from shared import config
except ImportError:
    st.error("Error: Could not import 'shared' module. Make sure the directory structure is correct.")
    st.stop()

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5555
BUFFER_SIZE = 65536

st.set_page_config(page_title="Drowsiness Server - MediaPipe", page_icon="👁️", layout="wide")


class SharedState:
    """Thread-safe shared data between server thread and UI."""
    def __init__(self):
        self.lock = threading.Lock()
        self.ear = 0.0
        self.mar = 0.0
        self.is_drowsy = False
        self.is_yawning = False
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

    def update(self, ear, mar, is_drowsy, is_yawning, frame_rgb):
        with self.lock:
            self.ear = ear
            self.mar = mar
            self.is_drowsy = is_drowsy
            self.is_yawning = is_yawning
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

    def snapshot(self):
        with self.lock:
            return {
                "ear": self.ear,
                "mar": self.mar,
                "is_drowsy": self.is_drowsy,
                "is_yawning": self.is_yawning,
                "drowsy_count": self.drowsy_count,
                "yawn_count": self.yawn_count,
                "events": list(self.events),
                "start_time": self.start_time,
                "connected": self.connected,
                "frames_processed": self.frames_processed,
                # Return a copy if needed, but for display reference is usually fine in this scope
                "last_frame": self.last_frame.copy() if self.last_frame is not None else None,
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


state = SharedState()


def tcp_server_loop():
    """Receives frames from Raspberry Pi and updates shared state."""
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
            state.start_time = datetime.now()

            # Reset counters for new session
            analyzer.ear_counter = 0
            analyzer.yawn_counter = 0

            while True:
                size_data = _recv_exact(client_socket, 4)
                if not size_data:
                    raise ConnectionError("Client disconnected")
                frame_size = struct.unpack('>I', size_data)[0]

                frame_data = _recv_exact(client_socket, frame_size)
                if not frame_data:
                    raise ConnectionError("Incomplete frame")

                frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Process using shared MediaPipe analyzer
                processed, ear, mar, is_drowsy, is_yawning = analyzer.detect(frame)

                # Convert to RGB and resize for UI preview
                preview = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
                # Ensure the resize matches the layout aspect ratio
                preview = cv2.resize(preview, (640, 480)) 

                state.update(ear, mar, is_drowsy, is_yawning, preview)
        
        except Exception as e:
            print(f"[SERVER] Error: {e}")
            state.disconnect()
        finally:
            try:
                client_socket.close()
            except Exception:
                pass
            print("[SERVER] Waiting for new connection...")


def _recv_exact(sock, size):
    data = b''
    while len(data) < size:
        chunk = sock.recv(min(size - len(data), BUFFER_SIZE))
        if not chunk:
            return None
        data += chunk
    return data


def play_beep():
    """Emits alarm beep (Windows only)"""
    try:
        import winsound
        winsound.Beep(800, 200)
    except Exception:
        # winsound is Windows only; this pass avoids errors on Linux/Mac
        pass


# Start server thread once
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

# Layout: Video on Left, Events on Right
frame_col, events_col = st.columns([2, 1])

with frame_col:
    st.subheader("Live Preview")
    frame_placeholder = st.empty()

with events_col:
    st.subheader("Recent Alerts")
    events_placeholder = st.empty()

# Information and Metrics below
alert_placeholder = st.empty()
info_placeholder = st.empty()
metrics_placeholder = st.empty()

while True:
    snap = state.snapshot()

    # Connection Status
    with info_placeholder.container():
        if snap["connected"]:
            st.success(f"🟢 Raspberry Connected - Frames: {snap['frames_processed']}")
        else:
            st.warning("🟡 Waiting for Raspberry Pi connection...")

    # Video Feed
    if snap["last_frame"] is not None:
        frame_placeholder.image(snap["last_frame"], channels="RGB", use_container_width=True)
    else:
        frame_placeholder.image("https://via.placeholder.com/640x480.png?text=Waiting+for+Video", use_container_width=True)

    # Alerts Banner
    with alert_placeholder.container():
        if snap["is_drowsy"]:
            st.error("⚠️ DROWSINESS DETECTED!", icon="🚨")
        elif snap["is_yawning"]:
            st.warning("🥱 Yawn Detected", icon="😴")
        else:
            # Clear space or show separator
            st.markdown("---")

    # Audio Alert
    if state.should_alert() and not st.session_state.muted:
        threading.Thread(target=play_beep, daemon=True).start()

    # Metrics Grid
    with metrics_placeholder.container():
        c1, c2, c3, c4 = st.columns(4)
        status_text = "⚠️ ALERT" if snap["is_drowsy"] else ("✅ OK" if snap["connected"] else "⏳ Waiting")
        
        c1.metric("Status", status_text)
        c2.metric("EAR", f"{snap['ear']:.3f}")
        c3.metric("MAR", f"{snap['mar']:.3f}")
        c4.metric("Events", f"🔴 {snap['drowsy_count']}  🥱 {snap['yawn_count']}")

    # Event Log
    with events_placeholder.container():
        if snap["events"]:
            for event in snap["events"][:8]:
                st.text(event)
        else:
            st.caption("No events yet")

    time.sleep(0.05) # Reduced slightly for smoother UI updates
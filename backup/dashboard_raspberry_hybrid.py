"""
Streamlit Dashboard - Raspberry Pi Standalone Mode
Runs locally on Raspberry when PC server is not available.
Switches to CLIENT mode when server becomes available.

streamlit run dashboard_raspberry_hybrid.py --server.address 0.0.0.0 --server.port 8501
"""
import os
# Suppress MediaPipe/TF Lite logging (only show errors)
os.environ['GLOG_minloglevel'] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ["GLOG_logtostderr"] = '0'
os.environ['MAGLEV_HTTP_RESOLVER'] = '0'

import socket
import struct
import cv2
import time
import psutil
import json
import streamlit as st
import threading
from datetime import datetime
from collections import deque
import sys

# Add parent directory to path to import shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from shared.drowsiness_analyzer import DrowsinessAnalyzer
    from shared import config
except ImportError:
    st.error("Error: Could not import 'shared' module.")
    st.stop()

# Try to import Raspberry Pi specific modules
try:
    from gpiozero import CPUTemperature
    HAS_GPIOZERO = True
except ImportError:
    HAS_GPIOZERO = False

def get_local_ip():
    """Get the actual local IP address of the Raspberry Pi"""
    try:
        # Create a socket to determine the outgoing IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))  # Doesn't actually send data
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

st.set_page_config(page_title="Drowsiness - Raspberry Standalone", page_icon="🍓", layout="wide")

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
        self.connected_to_server = False
        self.standalone_active = False
        self.frames_processed = 0
        self.last_frame = None
        self._prev_drowsy = False
        self._prev_yawn = False
        self._trigger_alert = False
        self.cpu_temp = 0.0
        self.cpu_usage = 0.0
        self.ram_usage = 0.0
        self.fps = 0.0
        # Calibration state
        self.calibrating = False
        self.calibration_done = False
        self.calibration_remaining = 0
        self.calibration_ear_samples = 0
        self.calibration_avg_ear = 0.0
        self.calibration_message = ""

    def update(self, ear, mar, is_drowsy, is_yawning, face_detected, frame_rgb):
        with self.lock:
            self.ear = ear
            self.mar = mar
            self.is_drowsy = is_drowsy
            self.is_yawning = is_yawning
            self.face_detected = face_detected
            self.frames_processed += 1
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

    def update_system_stats(self, cpu_temp, cpu_usage, ram_usage, fps):
        with self.lock:
            self.cpu_temp = cpu_temp
            self.cpu_usage = cpu_usage
            self.ram_usage = ram_usage
            self.fps = fps

    def update_calibration(self, remaining, ear_samples, avg_ear, message, face_detected, frame_rgb):
        with self.lock:
            self.calibration_remaining = remaining
            self.calibration_ear_samples = ear_samples
            self.calibration_avg_ear = avg_ear
            self.calibration_message = message
            self.face_detected = face_detected
            self.last_frame = frame_rgb

    def start_calibration(self):
        with self.lock:
            self.calibrating = True
            self.calibration_done = False

    def finish_calibration(self, threshold):
        with self.lock:
            self.calibrating = False
            self.calibration_done = True
            self.events.appendleft(f"✅ {datetime.now().strftime('%H:%M:%S')} - Calibration complete (threshold: {threshold:.3f})")

    def skip_calibration(self):
        with self.lock:
            self.calibrating = False
            self.calibration_done = True

    def set_mode(self, connected_to_server, standalone_active):
        with self.lock:
            self.connected_to_server = connected_to_server
            self.standalone_active = standalone_active
            if connected_to_server:
                self.events.appendleft(f"🟢 {datetime.now().strftime('%H:%M:%S')} - Connected to PC Server")
            elif standalone_active:
                self.events.appendleft(f"🟡 {datetime.now().strftime('%H:%M:%S')} - Standalone Mode Active")

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
                "connected_to_server": self.connected_to_server,
                "standalone_active": self.standalone_active,
                "frames_processed": self.frames_processed,
                "last_frame": self.last_frame.copy() if self.last_frame is not None else None,
                "cpu_temp": self.cpu_temp,
                "cpu_usage": self.cpu_usage,
                "ram_usage": self.ram_usage,
                "fps": self.fps,
                "calibrating": self.calibrating,
                "calibration_done": self.calibration_done,
                "calibration_remaining": self.calibration_remaining,
                "calibration_ear_samples": self.calibration_ear_samples,
                "calibration_avg_ear": self.calibration_avg_ear,
                "calibration_message": self.calibration_message,
            }

    def should_alert(self):
        with self.lock:
            if self._trigger_alert:
                self._trigger_alert = False
                return True
            return False

    def reset_for_standalone(self):
        with self.lock:
            self.start_time = datetime.now()
            self.frames_processed = 0

state = SharedState()

class HybridClient:
    def __init__(self, shared_state):
        self.state = shared_state
        self.server_ip = config.PC_SERVER_IP
        self.server_port = config.PC_SERVER_PORT
        self.socket = None
        self.connected = False
        
        # Camera
        self.camera = None
        self.use_picamera2 = False
        
        # Analyzer
        self.analyzer = None
        self.last_reconnect_attempt = 0
        self.reconnect_interval = 5
        self.frame_count = 0
        self.start_time = time.time()
        self.running = True
        
        # Calibration
        self.calibration_requested = False

    def get_system_stats(self):
        try:
            cpu_temp = CPUTemperature().temperature if HAS_GPIOZERO else 0.0
            cpu_usage = psutil.cpu_percent(percpu=True)
            cpu_usage = sum(cpu_usage)
            ram = psutil.virtual_memory().percent
            return cpu_temp, cpu_usage, ram
        except:
            return 0.0, 0.0, 0.0

    def connect_to_server(self):
        now = time.time()
        if now - self.last_reconnect_attempt < self.reconnect_interval:
            return False
        
        self.last_reconnect_attempt = now
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(1.0)
            self.socket.connect((self.server_ip, self.server_port))
            self.socket.settimeout(None)
            self.connected = True
            self.state.set_mode(connected_to_server=True, standalone_active=False)
            self.start_time = time.time()
            self.frame_count = 0
            print(f"[CONNECTED] Server found at {self.server_ip}:{self.server_port}")
            return True
        except:
            self.connected = False
            return False

    def init_camera(self):
        print("[INFO] Initializing camera...")
        try:
            from picamera2 import Picamera2
            self.camera = Picamera2()
            cam_config = self.camera.create_video_configuration(
                main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT), "format": "RGB888"},
                controls={"FrameRate": config.CAMERA_FPS}
            )
            self.camera.configure(cam_config)
            self.camera.start()
            self.use_picamera2 = True
            print("[INFO] PiCamera2 active")
            return True
        except:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(3, config.CAMERA_WIDTH)
            self.camera.set(4, config.CAMERA_HEIGHT)
            print("[INFO] USB Webcam active")
            return self.camera.isOpened()

    def capture_frame(self):
        if self.use_picamera2:
            return cv2.cvtColor(self.camera.capture_array(), cv2.COLOR_RGB2BGR)
        ret, frame = self.camera.read()
        return frame if ret else None

    def send_frame_with_stats(self, frame):
        """
        Send frame + system stats to server.
        Protocol: [4 bytes stats_size][JSON stats][4 bytes frame_size][JPEG frame]
        """
        try:
            # Get current system stats
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            cpu_temp, cpu_usage, ram = self.get_system_stats()
            
            # Prepare stats JSON
            stats = {
                'cpu_temp': cpu_temp,
                'cpu_usage': cpu_usage,
                'ram_usage': ram,
                'fps': fps
            }
            stats_json = json.dumps(stats).encode('utf-8')
            
            # Encode frame
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            frame_data = encoded.tobytes()
            
            # Send: stats_size + stats + frame_size + frame
            self.socket.sendall(struct.pack('>I', len(stats_json)))
            self.socket.sendall(stats_json)
            self.socket.sendall(struct.pack('>I', len(frame_data)))
            self.socket.sendall(frame_data)
            return True
        except:
            self.connected = False
            if self.socket:
                self.socket.close()
            self.state.set_mode(connected_to_server=False, standalone_active=True)
            print("[LOST] Connection lost! Switching to standalone...")
            return False

    def run_calibration(self):
        """10-second calibration to personalize EAR threshold"""
        self.state.start_calibration()
        
        ear_values = []
        start_time = time.time()
        calibration_duration = 10
        
        while self.running:
            elapsed = time.time() - start_time
            if elapsed >= calibration_duration:
                break
            
            frame = self.capture_frame()
            if frame is None:
                continue
            
            # Process frame to get EAR
            processed, ear, mar, _, _, face_detected, _ = self.analyzer.detect(frame)
            preview = cv2.resize(processed, (320, 240))
            
            remaining = calibration_duration - int(elapsed)
            
            if face_detected:
                if ear > 0.1:
                    ear_values.append(ear)
                avg_ear = sum(ear_values) / len(ear_values) if ear_values else 0.0
                message = f"Keep eyes open naturally - {remaining}s remaining"
                self.state.update_calibration(remaining, len(ear_values), avg_ear, message, True, preview)
            else:
                # Face lost - reset
                if elapsed > 0.5:
                    ear_values = []
                    start_time = time.time()
                    message = "⚠️ Face lost! Restarting calibration..."
                    self.state.update_calibration(calibration_duration, 0, 0.0, message, False, preview)
                    time.sleep(0.5)
        
        # Calculate and save threshold
        if len(ear_values) > 0:
            avg_ear = sum(ear_values) / len(ear_values)
            new_threshold = avg_ear * 0.85
            self.analyzer.save_threshold(new_threshold)
            self.state.finish_calibration(new_threshold)
            print(f"[CALIBRATION] Complete! Threshold: {new_threshold:.3f}")
        else:
            self.state.skip_calibration()
            print("[CALIBRATION] Failed - using defaults")

    def request_calibration(self):
        self.calibration_requested = True

    def run(self):
        if not self.init_camera():
            print("[ERROR] Camera initialization failed!")
            return
        
        print("[SYSTEM] Starting MediaPipe engine...")
        self.analyzer = DrowsinessAnalyzer()

        # Warm-up
        print("[SYSTEM] Warming up landmarks engine...")
        dummy_frame = self.capture_frame()
        if dummy_frame is not None:
            self.analyzer.detect(dummy_frame)

        # Check for existing calibration or wait for user action
        existing_threshold = self.analyzer.load_threshold()
        if os.path.exists(self.analyzer.config_path):
            self.state.skip_calibration()  # Use existing
            print(f"[CALIBRATION] Using saved threshold: {existing_threshold:.3f}")
        # else: wait for user to click calibrate button

        # Start in standalone mode (will try to connect to server)
        self.state.set_mode(connected_to_server=False, standalone_active=True)
        self.state.reset_for_standalone()

        try:
            while self.running:
                # Check if calibration was requested
                if self.calibration_requested:
                    self.calibration_requested = False
                    self.run_calibration()
                    continue

                frame = self.capture_frame()
                if frame is None:
                    continue

                self.frame_count += 1

                # Try to connect to server periodically
                if not self.connected:
                    self.connect_to_server()

                # OPERATIONAL LOGIC
                if self.connected:
                    # CLIENT MODE - Send frame + stats to PC server
                    if not self.send_frame_with_stats(frame):
                        # Connection lost, will switch back to standalone
                        self.state.reset_for_standalone()
                        self.frame_count = 0
                        self.start_time = time.time()
                else:
                    # STANDALONE MODE - Process locally and update dashboard
                    processed, ear, mar, drowsy, yawn, face, _ = self.analyzer.detect(frame)
                    
                    # Prepare preview (resize for dashboard)
                    preview = cv2.resize(processed, (320, 240))
                    
                    self.state.update(ear, mar, drowsy, yawn, face, preview)

                # Update system stats periodically
                if self.frame_count % 15 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    cpu_temp, cpu_usage, ram = self.get_system_stats()
                    self.state.update_system_stats(cpu_temp, cpu_usage, ram, fps)

        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        self.running = False
        if self.socket:
            self.socket.close()
        if self.use_picamera2 and self.camera:
            self.camera.stop()
        elif self.camera:
            self.camera.release()

def play_beep():
    """Play alert sound (works on Linux/Raspberry Pi)"""
    try:
        import subprocess
        # Try multiple methods for Linux audio alert
        # 1. Try aplay with ALSA
        result = subprocess.run(
            ['aplay', '-q', '/usr/share/sounds/alsa/Front_Center.wav'],
            capture_output=True, timeout=2
        )
        if result.returncode != 0:
            # 2. Try speaker-test for a beep
            subprocess.run(['speaker-test', '-t', 'sine', '-f', '800', '-l', '1'], 
                          capture_output=True, timeout=1)
    except:
        try:
            # 3. Fallback: system beep
            os.system('echo -e "\\a"')
        except:
            pass

# Get local IP for display
LOCAL_IP = get_local_ip()
print(f"\n{'='*60}")
print(f"  🍓 RASPBERRY PI DASHBOARD")
print(f"  Access from PC browser: http://{LOCAL_IP}:8501")
print(f"{'='*60}\n")

# Start client thread
if 'client_thread' not in st.session_state:
    client = HybridClient(state)
    st.session_state.client = client
    st.session_state.client_thread = threading.Thread(target=client.run, daemon=True)
    st.session_state.client_thread.start()

if 'muted' not in st.session_state:
    st.session_state.muted = False

# ===================== UI LAYOUT =====================
st.title("🍓 Drowsiness Detection - Raspberry Pi")
st.caption(f"📡 Dashboard accessible at: **http://{LOCAL_IP}:8501** | Auto-connect to PC server ({config.PC_SERVER_IP}:{config.PC_SERVER_PORT})")

ctrl_col, calib_col, mute_col = st.columns([6, 2, 1])
with calib_col:
    if st.button("🎯 Calibrate EAR"):
        if 'client' in st.session_state:
            st.session_state.client.request_calibration()
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
calibration_placeholder = st.empty()
metrics_placeholder = st.empty()
system_placeholder = st.empty()

while True:
    snap = state.snapshot()
    
    # Connection/Mode Status
    with info_placeholder.container():
        if snap["connected_to_server"]:
            st.info("🟢 Connected to PC Server - Processing on PC side")
        elif snap["calibrating"]:
            st.warning("🎯 Calibration in progress...")
        elif snap["standalone_active"]:
            st.success(f"🍓 Standalone Mode - Local Processing | Frames: {snap['frames_processed']}")
        else:
            st.warning("🟡 Initializing...")
    
    # Calibration UI
    with calibration_placeholder.container():
        if snap["calibrating"]:
            st.info(f"📷 **{snap['calibration_message']}**")
            col1, col2, col3 = st.columns(3)
            col1.metric("Time Remaining", f"{snap['calibration_remaining']}s")
            col2.metric("Samples", snap['calibration_ear_samples'])
            col3.metric("Avg EAR", f"{snap['calibration_avg_ear']:.3f}")
            if not snap["face_detected"]:
                st.error("👤 Face not detected! Please look at the camera.")
    
    # Video Feed
    if snap["calibrating"] and snap["last_frame"] is not None:
        frame_placeholder.image(snap["last_frame"], channels="RGB", width=320)
    elif snap["standalone_active"] and snap["last_frame"] is not None:
        frame_placeholder.image(snap["last_frame"], channels="RGB", width=320)
    elif snap["connected_to_server"]:
        frame_placeholder.info("📡 Video streaming to PC Server\n\nView the dashboard on PC for live preview.")
    else:
        frame_placeholder.image("https://via.placeholder.com/320x240.png?text=Initializing...", width=320)
    
    # Alerts (only in standalone mode, not during calibration)
    with alert_placeholder.container():
        if snap["calibrating"]:
            st.markdown("---")
        elif snap["standalone_active"]:
            if not snap.get("face_detected", True):
                st.error("🚨 FACE NOT DETECTED - PLEASE ADJUST CAMERA", icon="👤")
            elif snap["is_drowsy"]:
                st.error("⚠️ DROWSINESS DETECTED!", icon="🚨")
            elif snap["is_yawning"]:
                st.warning("🥱 Yawn Detected", icon="😴")
            else:
                st.markdown("---")
        elif snap["connected_to_server"]:
            st.info("Alerts managed by PC Server")
        else:
            st.markdown("---")
    
    # Audio Alert (only in standalone mode, not during calibration)
    if snap["standalone_active"] and not snap["calibrating"] and state.should_alert() and not st.session_state.muted:
        threading.Thread(target=play_beep, daemon=True).start()
    
    # Metrics (only meaningful in standalone mode)
    with metrics_placeholder.container():
        if snap["calibrating"]:
            pass  # Hide normal metrics during calibration
        elif snap["connected_to_server"]:
            c1, c2, c3, c4 = st.columns(4)
            status_text = "📡 PC-Side"
            c1.metric("Status", status_text)
            c2.metric("EAR", "PC-Side")
            c3.metric("MAR", "PC-Side")
            c4.metric("Events", f"🔴 {snap['drowsy_count']}  🥱 {snap['yawn_count']}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            status_text = "⚠️ ALERT" if snap["is_drowsy"] else ("✅ OK" if snap["standalone_active"] else "⏳ Init")
            c1.metric("Status", status_text)
            c2.metric("EAR", f"{snap['ear']:.3f}")
            c3.metric("MAR", f"{snap['mar']:.3f}")
            c4.metric("Events", f"🔴 {snap['drowsy_count']}  🥱 {snap['yawn_count']}")
    
    # System Stats (Raspberry specific)
    with system_placeholder.container():
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("FPS", f"{snap['fps']:.1f}")
        s2.metric("CPU", f"{snap['cpu_usage']:.1f}%")
        s3.metric("RAM", f"{snap['ram_usage']:.1f}%")
        if HAS_GPIOZERO:
            s4.metric("Temp", f"{snap['cpu_temp']:.1f}°C")
        else:
            s4.metric("Temp", "N/A")
    
    # Event Log
    with events_placeholder.container():
        if snap["events"]:
            for event in snap["events"][:8]:
                st.text(event)
        else:
            st.caption("No events yet")
    
    time.sleep(0.05)
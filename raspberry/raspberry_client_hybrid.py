import os
# Suppress MediaPipe/TF Lite logging (only show errors)
os.environ['GLOG_minloglevel'] = '2'

import socket
import struct
import cv2
import time
import psutil
from gpiozero import CPUTemperature
from datetime import datetime
import sys

# Add parent directory to path to import shared modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import config
try:
    from shared.drowsiness_analyzer import DrowsinessAnalyzer
except ImportError:
    print("[ERROR] DrowsinessAnalyzer not found!")

class SmartRaspberryClient:
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.connected = False
        
        # Camera
        self.camera = None
        self.use_picamera2 = False
        
        # Analyzer & Stats
        self.local_detector = None
        self.last_reconnect_attempt = 0
        self.reconnect_interval = 5
        self.frame_count = 0
        self.start_time = time.time()
        
    def get_system_stats(self):
        try:
            cpu_temp = CPUTemperature().temperature
            cpu_usage = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            return f"CPU: {cpu_usage}% | RAM: {ram}% | Temp: {cpu_temp:.1f}C"
        except: return "Stats N/A"

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
            print(f"\n[CONNECTED] Server found! Switching to CLIENT mode.")
            self.local_detector = None 
            self.start_time = time.time() # Reset FPS timer
            self.frame_count = 0
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

    def send_frame(self, frame):
        try:
            _, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY])
            data = encoded.tobytes()
            self.socket.sendall(struct.pack('>I', len(data)) + data)
            return True
        except:
            self.connected = False
            if self.socket: self.socket.close()
            return False

    def run(self):
        if not self.init_camera(): return
        
        print("="*60)
        print(" DROWSINESS DETECTION")
        print("="*60)
        
        try:
            while True:
                frame = self.capture_frame()
                if frame is None: continue

                self.frame_count += 1
                current_ear = 0.0
                status_label = "OK"

                # 1. HANDLING CONNECTION
                if not self.connected:
                    self.connect_to_server()

                # 2. OPERATIONAL LOGIC
                if self.connected:
                    mode_label = "CLIENT"
                    if not self.send_frame(frame):
                        print("\n[LOST] Connection lost! Loading local analyzer...")
                else:
                    mode_label = "STNDAL" # Standalone
                    if self.local_detector is None:
                        self.local_detector = DrowsinessAnalyzer()
                        self.start_time = time.time()
                        self.frame_count = 0
                    
                    processed, ear, mar, drowsy, yawn = self.local_detector.detect(frame)
                    current_ear = ear
                    if drowsy: status_label = "DRWS!"
                    elif yawn: status_label = "YAWN"

                    if config.DISPLAY_ENABLED:
                        cv2.imshow("Raspberry Standalone", processed)
                        if cv2.waitKey(1) & 0xFF == ord('q'): break

                # 3. STATUS PRINT
                if self.frame_count % 15 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    sys_stats = self.get_system_stats()

                    # EAR is shown only in Standalone (in Client the PC computes it)
                    ear_str = f"EAR: {current_ear:.2f}" if not self.connected else "EAR: PC-Side"
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"MODE: {mode_label} | FPS: {fps:.1f} | {ear_str} | {status_label} || {sys_stats}")

        except KeyboardInterrupt:
            print("\n[STOP] User interrupted")
        finally:
            if self.socket: self.socket.close()
            if self.use_picamera2: self.camera.stop()
            else: self.camera.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    client = SmartRaspberryClient(config.PC_SERVER_IP, config.PC_SERVER_PORT)
    client.run()
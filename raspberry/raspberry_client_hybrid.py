"""raspberry_client_hybrid.py"""
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

# Suppress MediaPipe/TF Lite logging (0=all, 1=info, 2=warnings, 3=errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
os.environ['GLOG_minloglevel'] = '3' 
os.environ["GLOG_logtostderr"] = '0'
os.environ['MAGLEV_HTTP_RESOLVER'] = '0'

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
        
    def run_calibration(self, analyzer):
        """10-second initial setup to personalize EAR threshold"""
        print("\n" + "="*60)
        print(" INITIAL SETUP - EAR CALIBRATION")
        print("="*60)
        
        # Check for existing config
        #temp_analyzer = DrowsinessAnalyzer()
        existing_threshold = analyzer.load_threshold()
        
        if os.path.exists(analyzer.config_path):
            choice = input(f"[PROMPT] Found saved threshold ({existing_threshold:.2f}). Use previous? (y/n): ")
            if choice.lower() == 'y':
                print("[INFO] Using existing configuration.")
                return

        print("\n[ACTION] Please look at the camera with a normal expression.")
        print("[ACTION] Keep your eyes naturally open for 10 seconds.")
        time.sleep(2)
        
        ear_values = []
        start_time = time.time()
        calibration_duration = 10 # Secondi richiesti
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= calibration_duration:
                break # Calibrazione completata con successo
                
            frame = self.capture_frame()
            if frame is not None:
                # Riceve i 6 valori dall'analyzer aggiornato
                _, ear, _, _, _, face_detected = analyzer.detect(frame)
                
                if face_detected:
                    if ear > 0.1:
                        ear_values.append(ear)
                    
                    remaining = calibration_duration - int(elapsed)
                    print(f"Calibrating... {remaining}s remaining | Current EAR: {ear:.2f}      ", end="\r")
                else:
                    # RESET: Se il viso viene perso, resetta timer e campioni
                    if elapsed > 0.5: # Evita reset per micro-glitch istantanei
                        print("\n[⚠️ RESET] Face lost! Restarting 10s timer...          ")
                        ear_values = []
                        start_time = time.time()
                        time.sleep(1) # Breve pausa per permettere all'utente di posizionarsi
        
        if len(ear_values) > 0:
            avg_ear = sum(ear_values) / len(ear_values)
            # Threshold set at 85% of average open-eye EAR
            new_threshold = avg_ear * 0.85
            analyzer.save_threshold(new_threshold)
            print(f"\n[SUCCESS] Calibration complete! Average EAR: {avg_ear:.2f}")
            print(f"[SUCCESS] New Alert Threshold: {new_threshold:.2f}")
        else:
            print("\n[ERROR] Calibration failed: No face detected. Using defaults.")
        
        print("="*60 + "\n")

    def get_system_stats(self):
        try:
            cpu_temp = CPUTemperature().temperature
            cpu_usage = psutil.cpu_percent(percpu=True)
            cpu_usage = sum(cpu_usage)
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
        
        print("[SYSTEM] Starting MediaPipe engine...")
        startup_analyzer = DrowsinessAnalyzer()

        # --- WARM-UP STEP ---
        # Catturiamo un frame e lo processiamo subito per far uscire i warning di MediaPipe/TFLite
        print("[SYSTEM] Warming up landmarks engine...")
        dummy_frame = self.capture_frame()
        if dummy_frame is not None:
            startup_analyzer.detect(dummy_frame) # Questo scatena i warning

        # START CALIBRATION BEFORE THE MAIN LOOP
        self.run_calibration(startup_analyzer)

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
                        self.local_detector = startup_analyzer
                        self.start_time = time.time()
                        self.frame_count = 0

                    processed, ear, mar, drowsy, yawn, face, score = self.local_detector.detect(frame)
                    current_ear = ear
                    if not face: status_label = "!!! NO FACE !!!"
                    elif drowsy: status_label = "DRWS!"
                    elif yawn: status_label = "YAWN"
                    else: status_label = "OK"

                    if config.DISPLAY_ENABLED: #? in standalone perchè display abilitato?
                        cv2.imshow("Raspberry Standalone", processed)
                        if cv2.waitKey(1) & 0xFF == ord('q'): break

                # 3. STATUS PRINT
                if self.frame_count % config.CAMERA_FPS == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    sys_stats = self.get_system_stats()

                    # EAR is shown only in Standalone (in Client the PC computes it)
                    ear_str = f"EAR: {current_ear:.2f}" if not self.connected else "EAR: PC-Side"
                    score = f"SCORE: {score:.1f}" if not self.connected else "SCORE: PC-Side"
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"MODE: {mode_label} | {score} | FPS: {fps:.1f} | {ear_str} | {status_label} || {sys_stats}")

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
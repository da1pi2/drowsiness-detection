#!/usr/bin/env python3
"""
raspberry_client.py - Raspberry Pi Client
Captures frames from the camera and sends them to the PC for processing.
Unidirectional communication: only frame transmission, no reception.

Compatible with Raspberry Pi OS Bookworm (64-bit) - uses picamera2
"""

import socket
import struct
import cv2
import time
import psutil
from gpiozero import CPUTemperature
from datetime import datetime

# ===================== CONFIGURATION =====================
PC_SERVER_IP = "192.168.1.219"  # PC'S IP
PC_SERVER_PORT = 5555

# Camera
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 15

# JPEG Compression (70 = good quality/bandwidth compromise)
JPEG_QUALITY = 70

# Connection
CONNECTION_TIMEOUT = 10
RECONNECT_DELAY = 5


class RaspberryClient:
    """Client that captures and sends frames to the PC (transmission only)"""
    
    def __init__(self, server_ip, server_port):
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None
        self.connected = False
        
        # Camera
        self.camera = None
        self.use_picamera2 = False
        
        # Statistics
        self.frames_sent = 0
        self.start_time = None
    
    def get_system_stats(self):
        """Returns string with CPU%, RAM%, and Temp°C"""
        try:
            # CPU Temperature
            cpu_temp = CPUTemperature().temperature
            # CPU Usage Percentage
            cpu_usage = psutil.cpu_percent()
            # RAM Usage Percentage
            ram = psutil.virtual_memory()
            
            return f"CPU: {cpu_usage}% | RAM: {ram.percent}% | Temp: {cpu_temp:.1f}°C"
        except Exception:
            return "Stats N/A"

    def connect(self):
        """Connects to PC server"""
        try:
            print(f"[INFO] Connecting to {self.server_ip}:{self.server_port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(CONNECTION_TIMEOUT)
            self.socket.connect((self.server_ip, self.server_port))
            self.socket.settimeout(None)
            self.connected = True
            print("[INFO] Connected to server!")
            return True
        except Exception as e:
            print(f"[ERROR] Connection failed: {e}")
            self.connected = False
            return False
    
    def init_camera(self):
        """Initializes the camera (picamera2 for Bookworm or USB fallback)"""
        print("[INFO] Initializing camera...")
        
        # Try picamera2 (Bookworm)
        try:
            from picamera2 import Picamera2
            
            self.camera = Picamera2()
            camera_config = self.camera.create_video_configuration(
                main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
                controls={"FrameRate": CAMERA_FPS}
            )
            self.camera.configure(camera_config)
            self.camera.start()
            
            time.sleep(0.5)  # Warmup
            self.use_picamera2 = True
            print(f"[INFO] PiCamera2 initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT} @ {CAMERA_FPS}fps")
            return True
            
        except ImportError:
            print("[WARN] picamera2 not available, trying USB webcam...")
        except Exception as e:
            print(f"[WARN] PiCamera2 error: {e}")
        
        # Fallback to USB webcam / OpenCV
        try:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            
            if not self.camera.isOpened():
                raise Exception("Camera not available")
            
            self.use_picamera2 = False
            print("[INFO] USB/OpenCV Webcam initialized")
            return True
        except Exception as e:
            print(f"[ERROR] Unable to initialize camera: {e}")
            return False
    
    def capture_frame(self):
        """Captures a frame from the camera"""
        if self.use_picamera2:
            frame = self.camera.capture_array()
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = self.camera.read()
            return frame if ret else None
    
    def send_frame(self, frame):
        """Sends compressed frame to the server"""
        try:
            # Compress to JPEG
            encode_param = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            _, encoded = cv2.imencode('.jpg', frame, encode_param)
            data = encoded.tobytes()
            
            # Send: [4 bytes size] + [JPEG data]
            self.socket.sendall(struct.pack('>I', len(data)) + data)
            return True
        except Exception as e:
            print(f"[ERROR] Sending frame: {e}")
            return False
    
    def run(self):
        """Main loop"""
        print("=" * 60)
        print("  DROWSINESS DETECTION - RASPBERRY STREAMER")
        print("  (Frame capture and transmission only)")
        print("=" * 60)
        
        # Initialize camera
        if not self.init_camera():
            return
        
        # Connect to server (with retry)
        while not self.connected:
            if not self.connect():
                print(f"[INFO] Retrying in {RECONNECT_DELAY} seconds...")
                time.sleep(RECONNECT_DELAY)
        
        self.start_time = time.time()
        print("\n[INFO] Streaming active! Press Ctrl+C to exit")
        print("-" * 60)
        
        try:
            while self.connected:
                # Capture frame
                frame = self.capture_frame()
                if frame is None:
                    continue
                
                # Send to server
                if not self.send_frame(frame):
                    print("[WARN] Transmission failed, reconnecting...")
                    self.connected = False
                    break
                
                self.frames_sent += 1
                
                # Periodic log (every 30 frames -> approx every 2 seconds at 15fps)
                if self.frames_sent % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frames_sent / elapsed if elapsed > 0 else 0
                    
                    # Get system stats
                    sys_stats = self.get_system_stats()
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                          f"FPS: {fps:.1f} | Frame: {self.frames_sent} || {sys_stats}")
        
        except KeyboardInterrupt:
            print("\n[INFO] Keyboard interrupt")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Resource cleanup"""
        print("\n[INFO] Closing...")
        
        if self.socket:
            self.socket.close()
        
        if self.use_picamera2 and self.camera:
            self.camera.stop()
            self.camera.close()
        elif self.camera:
            self.camera.release()
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frames_sent / elapsed if elapsed > 0 else 0
        
        print("\n" + "=" * 60)
        print("STATISTICS:")
        print(f"  Frames sent: {self.frames_sent}")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Average FPS: {fps:.1f}")
        print("=" * 60)


# ===================== MAIN =====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Raspberry Streamer for Drowsiness Detection')
    parser.add_argument('--server', type=str, default=PC_SERVER_IP,
                        help=f'PC Server IP (default: {PC_SERVER_IP})')
    parser.add_argument('--port', type=int, default=PC_SERVER_PORT,
                        help=f'Server Port (default: {PC_SERVER_PORT})')
    parser.add_argument('--quality', type=int, default=JPEG_QUALITY,
                        help=f'JPEG Quality 1-100 (default: {JPEG_QUALITY})')
    args = parser.parse_args()
    
    if args.quality:
        JPEG_QUALITY = args.quality
    
    client = RaspberryClient(args.server, args.port)
    client.run()
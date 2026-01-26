# drowsiness_config_standalone.py
# Standalone configurations for MediaPipe-based version

# ===================== CONNECTION (Server-version) =====================
PC_SERVER_IP = "192.168.1.219"  # PC'S IP
PC_SERVER_PORT = 5555
CONNECTION_TIMEOUT = 10
RECONNECT_DELAY = 5

# ===================== DETECTION THRESHOLDS (Standalone-only) =====================
# MediaPipe is very accurate, standard thresholds work well
EAR_THRESHOLD = 0.25       # Default Eye Aspect Ratio threshold (eyes closed if < 0.25)
EAR_CONSEC_FRAMES = 10     # Consecutive frames for alert
MAR_THRESHOLD = 0.6        # Default Mouth Aspect Ratio threshold for yawning
YAWN_CONSEC_FRAMES = 8     # Consecutive frames for yawn detection
# ===================== CAMERA (Both standalone and server)===================================
# With MediaPipe we can dare a slightly higher resolution if we want,
# but 320x240 is the ideal resolution for maximizing FPS on Pi 3B+
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 20  # Slightly increased (it was 15) because MediaPipe is faster
# JPEG Compression (70 = good quality/bandwidth compromise)
JPEG_QUALITY = 70

# ===================== VIEW (Standalone-only) ====================================
SHOW_LANDMARKS = True      # Show eye/mouth landmarks
SHOW_EAR_MAR = True        # Show EAR/MAR values
DISPLAY_ENABLED = False    # False for Lite version (no desktop)

# ===================== LOGGING (Standalone-only) =================================
LOG_EVENTS = False
LOG_FILE = "drowsiness_log.txt"
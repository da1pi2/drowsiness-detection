# drowsiness_config_standalone.py
# Standalone configurations for MediaPipe-based version

# ===================== DETECTION THRESHOLDS =====================
# MediaPipe is very accurate, standard thresholds work well
EAR_THRESHOLD = 0.25       # Eye Aspect Ratio threshold (eyes closed if < 0.25)
EAR_CONSEC_FRAMES = 10     # Consecutive frames for alert
MAR_THRESHOLD = 0.6        # Mouth Aspect Ratio threshold for yawning
YAWN_CONSEC_FRAMES = 8     # Consecutive frames for yawn detection
# ===================== CAMERA ===================================
# With MediaPipe we can dare a slightly higher resolution if we want,
# but 320x240 is the ideal resolution for maximizing FPS on Pi 3B+
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 20  # Slightly increased because MediaPipe is faster

# ===================== VIEW ====================================
SHOW_LANDMARKS = True      # Show eye/mouth landmarks
SHOW_EAR_MAR = True        # Show EAR/MAR values
DISPLAY_ENABLED = False    # False for Lite version (no desktop)

# ===================== LOGGING =================================
LOG_EVENTS = False
LOG_FILE = "drowsiness_log.txt"
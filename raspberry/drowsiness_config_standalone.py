# drowsiness_config.py - Configurazioni per versione standalone
# Raspberry Pi OS Bookworm (64-bit)

# ===================== SOGLIE RILEVAMENTO =====================
EAR_THRESHOLD = 0.25       # Soglia Eye Aspect Ratio (occhi chiusi se < 0.25)
EAR_CONSEC_FRAMES = 10     # Frame consecutivi per allarme (ridotto per basso FPS)
MAR_THRESHOLD = 0.6        # Soglia Mouth Aspect Ratio per sbadigli
YAWN_CONSEC_FRAMES = 8     # Frame consecutivi per sbadiglio

# ===================== CAMERA =====================
# Risoluzione bassa per performance su Pi 3B+
CAMERA_WIDTH = 320         # Usa 160 per più FPS
CAMERA_HEIGHT = 240        # Usa 120 per più FPS
CAMERA_FPS = 15

# ===================== MODELLO =====================
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# ===================== VISUALIZZAZIONE =====================
SHOW_LANDMARKS = True      # Mostra contorni occhi/bocca
SHOW_EAR_MAR = True        # Mostra valori EAR/MAR
DISPLAY_ENABLED = False    # False per versione Lite (senza desktop)

# ===================== LOGGING =====================
LOG_EVENTS = True
LOG_FILE = "drowsiness_log.txt"
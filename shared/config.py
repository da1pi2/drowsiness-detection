# drowsiness_config_standalone.py
# Configurazioni per versione standalone con MediaPipe

# ===================== SOGLIE RILEVAMENTO =====================
# MediaPipe è molto preciso, le soglie standard funzionano bene
EAR_THRESHOLD = 0.25       # Soglia Eye Aspect Ratio (occhi chiusi se < 0.25)
EAR_CONSEC_FRAMES = 10     # Frame consecutivi per allarme
MAR_THRESHOLD = 0.6        # Soglia Mouth Aspect Ratio per sbadigli
YAWN_CONSEC_FRAMES = 8     # Frame consecutivi per sbadiglio

# ===================== CAMERA =====================
# Con MediaPipe possiamo osare una risoluzione leggermente maggiore se vuoi,
# ma 320x240 resta l'ideale per massimizzare gli FPS su Pi 3B+
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 20  # Alzato leggermente visto che MediaPipe è più veloce

# ===================== VISUALIZZAZIONE =====================
SHOW_LANDMARKS = True      # Mostra scheletro occhi/bocca
SHOW_EAR_MAR = True        # Mostra valori EAR/MAR
DISPLAY_ENABLED = False    # False per versione Lite (senza desktop)

# ===================== LOGGING =====================
LOG_EVENTS = False
LOG_FILE = "drowsiness_log.txt"
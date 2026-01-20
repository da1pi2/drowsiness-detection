# config.py - Configurazioni del sistema di drowsiness detection

# Soglie per il rilevamento
EAR_THRESHOLD = 0.25  # Soglia Eye Aspect Ratio (occhi chiusi se < 0.25)
EAR_CONSEC_FRAMES = 20  # Numero di frame consecutivi per attivare allarme
MAR_THRESHOLD = 0.6  # Soglia Mouth Aspect Ratio per sbadigli
YAWN_CONSEC_FRAMES = 15  # Frame consecutivi per rilevare sbadiglio

# Configurazione camera
CAMERA_WIDTH = 320 # abbassato da 640
CAMERA_HEIGHT = 240 # abbassato da 480
CAMERA_FPS = 15

# Configurazione allarme
ALARM_SOUND_PATH = "/usr/share/sounds/alsa/Front_Center.wav"  # Suono sistema
ALARM_ENABLED = True

# Modello pre-addestrato
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# Visualizzazione
SHOW_LANDMARKS = True  # Mostra i punti facciali
SHOW_EAR_MAR = True  # Mostra valori EAR/MAR sullo schermo
DISPLAY_ENABLED = True  # False se via SSH senza X11 forwarding

# Logging
LOG_EVENTS = True
LOG_FILE = "drowsiness_log.txt"
# 🚗 Drowsiness Detection - Guida Completa Setup

## 📋 Architettura del Sistema

```
┌─────────────────────┐         WiFi/LAN        ┌─────────────────────┐
│   RASPBERRY Pi 3B+  │ ─────────────────────►  │         PC          │
│                     │      TCP Socket         │                     │
│  • Cattura video    │      (porta 5555)       │  • Face detection   │
│  • Invio frame JPEG │   (solo TX, no RX)      │  • Calcolo EAR/MAR  │
│                     │                         │  • Preview live     │
└─────────────────────┘                         └─────────────────────┘
```

Il Raspberry cattura i frame dalla camera e li invia al PC.
Il PC esegue l'analisi (dlib o MediaPipe) e mostra la preview video in tempo reale.

---

## 🖥️ PARTE 1: Setup PC (Windows)

### 1.1 Prerequisiti
- Python 3.8+ installato
- (Solo per dlib) Visual Studio Build Tools

### 1.2 Crea Virtual Environment

```cmd
cd c:\..\drowsiness-detection

python -m venv venv_pc
venv_pc\Scripts\activate
```

### 1.3 Installa Dipendenze PC

```cmd
python -m pip install --upgrade pip
pip install -r pc_dashboard\requirements_pc.txt
```

> ⚠️ **Nota su dlib**: Se l'installazione fallisce:
> - Scarica CMake: https://cmake.org/download/
> - Installa VS Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/
> 
> 💡 **Alternativa**: Usa la versione **MediaPipe** che non richiede compilazione!

### 1.4 Scarica il Modello dlib (solo per versione dlib)

```cmd
cd pc_dashboard

# Download
powershell -Command "Invoke-WebRequest -Uri 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2' -OutFile 'shape_predictor_68_face_landmarks.dat.bz2'"

# Decomprimi
python -c "import bz2; open('shape_predictor_68_face_landmarks.dat','wb').write(bz2.open('shape_predictor_68_face_landmarks.dat.bz2').read())"
```

> 💡 **MediaPipe non richiede download di modelli** - sono inclusi nella libreria!

### 1.5 Trova l'IP del PC

```cmd
ipconfig
```

### 1.6 Avvia Dashboard/Server PC (solo Streamlit)

**Server + dashboard dlib (riceve dal Raspberry):**
```cmd
venv_pc\Scripts\activate
cd pc_dashboard
streamlit run dashboard_server_dlib.py
```

**Server + dashboard MediaPipe (riceve dal Raspberry, consigliata):**
```cmd
venv_pc\Scripts\activate
cd pc_dashboard
streamlit run dashboard_server_mediapipe.py
```

### 1.7 Demo con Webcam (senza Raspberry)

**Dashboard demo MediaPipe (consigliata):**
```cmd
venv_pc\Scripts\activate
cd pc_dashboard
streamlit run dashboard_demo_mediapipe.py
```

**Dashboard demo dlib:**
```cmd
venv_pc\Scripts\activate
cd pc_dashboard
streamlit run dashboard_demo_dlib.py
```

### 🖼️ Preview Video

Vedrai una finestra video live con:
- 🟢 Rettangolo verde = stato normale
- 🔴 Rettangolo rosso = sonnolenza rilevata
- 👁️ Contorni occhi (verde)
- 👄 Contorni bocca (giallo)
- 📊 Valori EAR e MAR
- ⚠️ Alert "SONNOLENZA!" / "SBADIGLIO!"

**Controlli:**
- `q` = Esci
- `p` = Toggle preview on/off

---

## 🍓 PARTE 2: Setup Raspberry Pi OS Bookworm (64-bit) Lite

### 2.1 Prerequisiti Sistema

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libcamera-dev libcamera-apps
sudo apt install -y python3-libcamera python3-picamera2
sudo apt install -y python3-opencv
```

### 2.2 Verifica Camera

```bash
libcamera-hello --list-cameras
libcamera-still -o test.jpg
```

### 2.3 Crea Virtual Environment

```bash
cd /home/pi/drowsiness-detection/raspberry

python3 -m venv --system-site-packages venv_raspberry
source venv_raspberry/bin/activate
pip install --upgrade pip
```

### 2.4 Installa Dipendenze

```bash
pip install -r requirements_raspberry.txt
```

### 2.5 Configura IP Server

Modifica `raspberry_client.py`:
```python
PC_SERVER_IP = "192.168.1.219"  # <-- IP del tuo PC
```

### 2.6 Avvia il Client

**Modalità Client (invia frame al PC):**
```bash
source venv_raspberry/bin/activate
python raspberry_client.py --server 192.168.1.219
```

**Modalità Standalone (analisi locale con MediaPipe):**
```bash
source venv_raspberry/bin/activate
python main_raspberry_standalone.py
```

---

## 📁 Struttura File

```
drowsiness-detection/
├── SETUP_GUIDE.md
├── shared/                              # Moduli condivisi
│   ├── config.py                        # Configurazioni
│   └── drowsiness_analyzer.py           # Analyzer MediaPipe
├── pc_dashboard/
│   ├── requirements_pc.txt              # Requirements PC
│   ├── dashboard_demo_dlib.py           # Dashboard demo dlib (webcam PC)
│   ├── dashboard_demo_mediapipe.py      # Dashboard demo MediaPipe (webcam PC)
│   ├── dashboard_server_dlib.py         # Dashboard+server dlib (Raspberry -> PC)
│   ├── dashboard_server_mediapipe.py    # Dashboard+server MediaPipe (Raspberry -> PC)
│   ├── backup/pc_server.py              # Vecchio server CLI dlib (backup)
│   ├── backup/pc_server_mediapipe.py    # Vecchio server CLI MediaPipe (backup)
│   └── shape_predictor_68_face_landmarks.dat
└── raspberry/
    ├── requirements_raspberry.txt       # Requirements Raspberry
    ├── raspberry_client.py              # Client (invia frame al PC)
    └── main_raspberry_standalone.py     # Standalone MediaPipe
```

---

## 🚀 Avvio Rapido

### PC (server + dashboard MediaPipe - consigliato):
```cmd
cd pc_dashboard
..\venv_pc\Scripts\activate
streamlit run dashboard_server_mediapipe.py
```

### PC Demo (webcam locale):
```cmd
cd pc_dashboard
..\venv_pc\Scripts\activate
streamlit run dashboard_demo_mediapipe.py
```

### Raspberry (client):
```bash
cd /home/pi/drowsiness-detection/raspberry
source venv_raspberry/bin/activate
python raspberry_client.py --server <IP_PC>
```

### Raspberry (standalone):
```bash
cd /home/pi/drowsiness-detection/raspberry
source venv_raspberry/bin/activate
python main_raspberry_standalone.py --no-display
```

---

## 🐛 Troubleshooting

### "Connection refused"
- Server PC in esecuzione?
- Firewall Windows: apri porta 5555
- Stessa rete LAN?

### "Camera not found" (Raspberry)
```bash
libcamera-hello --list-cameras
```

### FPS bassi
- Riduci risoluzione/JPEG_QUALITY in `raspberry_client.py`
- Verifica WiFi

---

## 📊 Performance

| Parametro | Valore |
|-----------|--------|
| Risoluzione | 320x240 |
| FPS | 15-20 |
| Latenza | 20-50ms |
| RAM Raspberry | ~100MB |
| RAM PC | ~500MB |

---

## ✅ Checklist

- [ ] PC: venv + dipendenze
- [ ] PC: shape_predictor scaricato
- [ ] PC: server avviato
- [ ] Raspberry: camera funzionante
- [ ] Raspberry: IP server configurato
- [ ] Firewall: porta 5555 aperta
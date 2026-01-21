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
Il PC esegue l'analisi (dlib) e mostra la preview video in tempo reale.

---

## 🖥️ PARTE 1: Setup PC (Windows)

### 1.1 Prerequisiti
- Python 3.8+ installato
- Visual Studio Build Tools (per compilare dlib)

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

### 1.4 Scarica il Modello dlib

```cmd
cd pc_dashboard

# Download - salta
powershell -Command "Invoke-WebRequest -Uri 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2' -OutFile 'shape_predictor_68_face_landmarks.dat.bz2'"

# Decomprimi - salta
python -c "import bz2; open('shape_predictor_68_face_landmarks.dat','wb').write(bz2.open('shape_predictor_68_face_landmarks.dat.bz2').read())"
```

### 1.5 Trova l'IP del PC

```cmd
ipconfig
```

### 1.6 Avvia il Server PC

```cmd
venv_pc\Scripts\activate
cd pc_dashboard
python pc_server.py
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
pip install -r ../requirements_raspberry.txt
```

### 2.5 Configura IP Server

Modifica `raspberry_client.py`:
```python
PC_SERVER_IP = "192.168.1.219"  # <-- IP del tuo PC
```

### 2.6 Avvia il Client

```bash
source venv_raspberry/bin/activate
python raspberry_client.py --server 192.168.1.219
```

---

## 📁 Struttura File

```
drowsiness-detection/
├── requirements_file.txt          # Requirements Raspberry
├── pc_dashboard/
│   ├── requirements_pc.txt        # Requirements PC
│   ├── pc_server.py               # Server + preview
│   └── shape_predictor_68_face_landmarks.dat
└── raspberry/
    └── raspberry_client.py        # Streamer video
```

---

## 🚀 Avvio Rapido

### PC:
```cmd
cd pc_dashboard
..\venv_pc\Scripts\activate
python pc_server.py
```

### Raspberry:
```bash
cd /home/pi/drowsiness-detection/raspberry
source venv_raspberry/bin/activate
python raspberry_client.py --server <IP_PC>
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
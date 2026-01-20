# 🚗 Drowsiness Detection - Guida Completa Setup

## 📋 Architettura del Sistema

```
┌─────────────────────┐         WiFi/LAN        ┌─────────────────────┐
│   RASPBERRY Pi 3B+  │ ─────────────────────►  │         PC          │
│                     │      TCP Socket         │                     │
│  • Cattura video    │      (porta 5555)       │  • Face detection   │
│  • Invio frame JPEG │ ◄───────────────────── │  • Calcolo EAR/MAR  │
│  • Allarme locale   │      Risultati          │  • Dashboard        │
└─────────────────────┘                         └─────────────────────┘
```

Il Raspberry cattura i frame e li invia al PC che esegue il processing pesante (dlib).
Il PC restituisce i risultati e il Raspberry attiva l'allarme locale se necessario.

---

## 🖥️ PARTE 1: Setup PC (Windows)

### 1.1 Prerequisiti
- Python 3.8+ installato
- Visual Studio Build Tools (per compilare dlib)

### 1.2 Crea Virtual Environment

```cmd
cd c:\Users\danie\OneDrive\Desktop\AIDE - UniPi LM\2 Anno\INDUSTRIAL APPLICATIONS (9 cfu)\drowsiness-detection

# Crea venv per PC
python -m venv venv_pc

# Attiva venv
venv_pc\Scripts\activate
```

### 1.3 Installa Dipendenze PC

```cmd
# Aggiorna pip
python -m pip install --upgrade pip

# Installa requirements
pip install -r pc_dashboard\requirements_pc.txt
```

> ⚠️ **Nota su dlib**: Se l'installazione fallisce, installa CMake e Visual Studio Build Tools:
> - Scarica CMake: https://cmake.org/download/
> - Installa VS Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### 1.4 Scarica il Modello dlib

```cmd
# Scarica shape_predictor (~ 100MB)
cd pc_dashboard

# Opzione 1: PowerShell
powershell -Command "Invoke-WebRequest -Uri 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2' -OutFile 'shape_predictor_68_face_landmarks.dat.bz2'"

# Decomprimi con 7zip o Python:
python -c "import bz2; open('shape_predictor_68_face_landmarks.dat','wb').write(bz2.open('shape_predictor_68_face_landmarks.dat.bz2').read())"
```

### 1.5 Trova l'IP del PC

```cmd
ipconfig
```
Cerca l'indirizzo IPv4 della tua interfaccia di rete (es. `192.168.1.100`).

### 1.6 Avvia il Server PC

```cmd
# Attiva venv se non già attivo
venv_pc\Scripts\activate

# Avvia server
cd pc_dashboard
python pc_server.py
```

Opzioni disponibili:
```cmd
python pc_server.py --port 5555 --no-preview
```

---

## 🍓 PARTE 2: Setup Raspberry Pi 3B+ (Raspbian Buster)

### 2.1 Prerequisiti Sistema

```bash
# Aggiorna sistema
sudo apt update && sudo apt upgrade -y

# Installa dipendenze sistema
sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y libjasper-dev libqtgui4 libqt4-test
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

### 2.2 Abilita Camera

```bash
sudo raspi-config
```
Vai su: **Interface Options → Camera → Enable**

Riavvia:
```bash
sudo reboot
```

### 2.3 Crea Virtual Environment

```bash
# Vai nella cartella del progetto
cd /home/pi/drowsiness-detection/raspberry

# Crea venv
python3 -m venv venv_raspberry

# Attiva venv
source venv_raspberry/bin/activate

# Aggiorna pip
pip install --upgrade pip
```

### 2.4 Installa Dipendenze

```bash
# Installa da requirements
pip install -r ../requirements_file.txt

# Nota: picamera potrebbe richiedere installazione sistema
pip install picamera
```

> 💡 **Tip**: Su Raspberry non serve dlib! Il processing viene fatto sul PC.

### 2.5 Copia i File sul Raspberry

Dal PC, trasferisci la cartella `raspberry/` sul Raspberry Pi:

**Opzione A - SCP (da terminale PC):**
```cmd
scp -r raspberry pi@<IP_RASPBERRY>:/home/pi/drowsiness-detection/
```

**Opzione B - FileZilla o WinSCP** (GUI)

**Opzione C - USB/SD Card**

### 2.6 Configura IP del Server

Modifica il file `raspberry_client.py` con l'IP del tuo PC:

```bash
nano raspberry_client.py
```

Trova e modifica questa riga:
```python
PC_SERVER_IP = "192.168.1.100"  # <-- CAMBIA CON L'IP DEL TUO PC
```

### 2.7 Avvia il Client Raspberry

```bash
# Attiva venv
source venv_raspberry/bin/activate

# Avvia client
python raspberry_client.py --server 192.168.1.100
```

Opzioni disponibili:
```bash
python raspberry_client.py --server 192.168.1.100 --port 5555 --no-display --no-alarm
```

---

## 📁 Struttura File Finale

```
drowsiness-detection/
├── requirements_file.txt          # Requirements Raspberry
│
├── pc_dashboard/
│   ├── requirements_pc.txt        # Requirements PC
│   ├── pc_server.py               # Server che elabora i frame
│   └── shape_predictor_68_face_landmarks.dat  # Modello dlib (da scaricare)
│
├── raspberry/
│   ├── raspberry_client.py        # Client che invia frame
│   ├── drowsiness_config.py       # Configurazioni
│   ├── alarm_module.py            # Gestione allarme
│   ├── drowsiness_detector.py     # Detector (per uso standalone)
│   └── main_raspberry.py          # Main standalone (senza PC)
│
└── models/                        # Modelli aggiuntivi (opzionale)
```

---

## 🔧 Configurazione Parametri

Modifica `raspberry/drowsiness_config.py`:

```python
# Risoluzione (abbassata per performance)
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
CAMERA_FPS = 15

# Soglie rilevamento
EAR_THRESHOLD = 0.25      # Soglia occhi chiusi
EAR_CONSEC_FRAMES = 20    # Frame consecutivi per allarme
MAR_THRESHOLD = 0.6       # Soglia sbadiglio

# Display (False per SSH senza X11)
DISPLAY_ENABLED = False
```

---

## 🚀 Avvio Rapido

### Sul PC:
```cmd
cd pc_dashboard
..\venv_pc\Scripts\activate
python pc_server.py
```

### Sul Raspberry:
```bash
cd /home/pi/drowsiness-detection/raspberry
source venv_raspberry/bin/activate
python raspberry_client.py --server <IP_PC>
```

---

## 🐛 Troubleshooting

### Errore: "Connection refused"
- Verifica che il server PC sia in esecuzione
- Controlla firewall Windows (apri porta 5555)
- Verifica che PC e Raspberry siano sulla stessa rete

### Errore: "Camera not found" su Raspberry
```bash
# Verifica camera
vcgencmd get_camera
# Deve mostrare: supported=1 detected=1

# Test camera
raspistill -o test.jpg
```

### FPS bassi
- Riduci risoluzione in `drowsiness_config.py`
- Riduci JPEG_QUALITY in `raspberry_client.py`
- Verifica qualità connessione WiFi

### Errore dlib su PC
```cmd
pip install cmake
pip install dlib --verbose
```

---

## 📊 Performance Attese

| Componente | Valore |
|------------|--------|
| Risoluzione | 320x240 |
| FPS Raspberry | 15-20 |
| Latenza rete | 20-50ms |
| RAM Raspberry | ~200MB |
| RAM PC | ~500MB |

---

## 🔌 Schema Connessioni Hardware (Opzionale)

Per aggiungere un buzzer fisico al Raspberry:

```
Raspberry Pi 3B+
    GPIO 17 ────── (+) Buzzer
    GND     ────── (-) Buzzer
```

Modifica `alarm_module.py` per usare GPIO:
```python
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)
GPIO.output(17, GPIO.HIGH)  # Buzzer ON
```

---

## ✅ Checklist Pre-Avvio

- [ ] PC: venv creato e attivato
- [ ] PC: dipendenze installate
- [ ] PC: shape_predictor scaricato
- [ ] PC: server avviato
- [ ] Raspberry: venv creato e attivato  
- [ ] Raspberry: dipendenze installate
- [ ] Raspberry: camera abilitata e funzionante
- [ ] Raspberry: IP server configurato
- [ ] Rete: PC e Raspberry sulla stessa LAN
- [ ] Firewall: porta 5555 aperta

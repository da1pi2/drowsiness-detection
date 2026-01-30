# Drowsiness Detection - Complete Setup Guide

## PART 1: PC Setup (Windows)

### 1.1 Prerequisites
- Python 3.8+ installed

### 1.2 Create Virtual Environment

```cmd
cd c:\..\drowsiness-detection

python -m venv venv_pc
venv_pc\Scripts\activate
```

### 1.3 Install PC Dependencies

```cmd
python -m pip install --upgrade pip
pip install -r pc_dashboard\requirements_pc.txt
```

### 1.4 Find PC IP Address

```cmd
ipconfig
```

### 1.5 Start PC Dashboard/Server (Streamlit only)

**Server + MediaPipe dashboard (receives from Raspberry, recommended):**
```cmd
venv_pc\Scripts\activate
cd pc_dashboard
streamlit run dashboard_server_mediapipe.py
```

### Video Preview

You will see a live video window with:
- 🟢 Green rectangle = normal state
- 🔴 Red rectangle = drowsiness detected
- 👁️ Eye contours (green)
- 👄 Mouth contours (yellow)
- 📊 EAR and MAR values
- ⚠️ Alerts "DROWSINESS!" / "YAWNING!"

---

## 🍓 PART 2: Raspberry Pi OS Bookworm (64-bit) Lite Setup

### 2.1 System Prerequisites

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y python3-pip python3-venv python3-dev
sudo apt install -y libcamera-dev libcamera-apps
sudo apt install -y python3-libcamera python3-picamera2
sudo apt install -y python3-opencv
```

### 2.2 Create Virtual Environment

```bash
cd /home/pi/drowsiness-detection/raspberry

python3 -m venv --system-site-packages venv_raspberry
source venv_raspberry/bin/activate
pip install --upgrade pip
```

### 2.3 Install Dependencies

```bash
pip install -r requirements_raspberry.txt
```

### 2.4 Configure Server IP

Edit `config.py`:
```python
PC_SERVER_IP = "192.168.1.219"  # <-- Your PC IP address
```

### 2.5 Start the Client

**No dashboard mode:**
```bash
source venv_raspberry/bin/activate
python raspberry_client_hybrid.py
```

**Dashboard mode:**
```bash
source venv_raspberry/bin/activate
python dashboard_raspberry_hybrid.py
```
# alarm_module.py - Gestione dell'allarme sonoro

import subprocess
import threading
import time
import drowsiness_config as config

class AlarmManager:
    def __init__(self):
        self.alarm_playing = False
        self.alarm_thread = None
        
    def play_alarm_sound(self):
        """Riproduce il suono di allarme usando aplay"""
        try:
            subprocess.call(['aplay', config.ALARM_SOUND_PATH], 
                          stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[ERRORE] Impossibile riprodurre suono: {e}")
    
    def play_beep(self):
        """Riproduce un beep usando il buzzer del sistema"""
        try:
            # Usa il beep del PC speaker se disponibile
            subprocess.call(['beep', '-f', '1000', '-l', '500'], 
                          stderr=subprocess.DEVNULL)
        except:
            # Fallback: stampa a terminale
            print('\a')  # Bell character
    
    def start_alarm(self):
        """Avvia l'allarme in un thread separato"""
        if not config.ALARM_ENABLED:
            return
            
        if not self.alarm_playing:
            self.alarm_playing = True
            print("[ALLARME] Attivato!")
            
            # Prova prima con aplay, poi con beep
            try:
                self.play_alarm_sound()
            except:
                self.play_beep()
    
    def stop_alarm(self):
        """Ferma l'allarme"""
        if self.alarm_playing:
            self.alarm_playing = False
            print("[ALLARME] Disattivato")
    
    def trigger_alert(self):
        """Trigger rapido per alert (per uso in loop principale)"""
        if config.ALARM_ENABLED:
            self.play_beep()
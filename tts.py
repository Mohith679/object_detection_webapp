import pyttsx3
import threading
def speak(text):
    def run_speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
    threading.Thread(target=run_speak, daemon=True).start()

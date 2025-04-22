# speech.py
import pyttsx3
import speech_recognition as sr
from typing import Optional

class SpeechInterface:
    """Handles Text-to-Speech and Speech-to-Text functionality."""
    def __init__(self, rate: int = 150, volume: float = 1.0):
        self.tts_engine = self._initialize_tts(rate, volume)
        self.recognizer = sr.Recognizer()

    def _initialize_tts(self, rate: int, volume: float) -> Optional[pyttsx3.Engine]:
        """Initializes the TTS engine."""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.setProperty("volume", volume)
            return engine
        except Exception as e:
            print(f"Error initializing TTS engine: {e}. TTS functionality will be disabled.")
            return None

    def speak(self, text: str):
        """Speaks text aloud using pyttsx3 and prints it."""
        print(f"Amico: {text}")
        if self.tts_engine:
            try:
                # Ensure the engine hasn't crashed previously or is busy
                # Note: runAndWait() is blocking, so checking is_busy might not be needed
                # unless you were trying to interrupt, which stop() handles.
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except RuntimeError as e:
                 print(f"TTS Runtime Error (Possibly engine busy or invalid state): {e}")
                 # Attempt to re-initialize or handle gracefully? For now, just print.
            except Exception as e:
                print(f"Error during TTS: {e}")
        else:
            print("(TTS Disabled)")

    def stop_speaking(self):
        """Stops the TTS engine immediately."""
        if self.tts_engine:
            try:
                # Check if engine is currently speaking before stopping
                # Note: isBusy property might not be reliable across all pyttsx3 drivers/OS
                # self.tts_engine.isBusy()
                self.tts_engine.stop()
            except Exception as e:
                 print(f"Error stopping TTS engine: {e}")


    def listen(self) -> Optional[str]:
        """Listens for voice input."""
        # Give feedback *before* blocking listen call
        self.speak("I'm listening...") # This call will block until finished speaking
        print("Listening...")
        with sr.Microphone() as source:
            try:
                # Dynamic energy adjustment based on ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                print("Adjusted for ambient noise. Ready for command.")
                # Listen for audio input
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
                print("Audio received, recognizing...")
                # Recognize speech using Google Web Speech API
                command = self.recognizer.recognize_google(audio)
                print(f"You said: {command}")
                return command.lower()
            except sr.WaitTimeoutError:
                print("Timeout waiting for speech.")
                # self.speak("I didn't hear anything.") # Optional feedback
                return None
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
                self.speak("Sorry, I couldn't understand the audio.")
                return None
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                self.speak(f"Speech service request failed.") # Don't include error detail in speech
                return None
            except Exception as e:
                print(f"An unexpected error occurred during listening: {e}")
                self.speak("An error occurred while trying to listen.")
                return None
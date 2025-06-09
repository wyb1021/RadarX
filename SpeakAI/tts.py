import pyttsx3

def synthesize(text: str, filename="response.wav"):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()
    print(f"✅ TTS 완료: {filename}")

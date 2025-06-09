import sounddevice as sd
import soundfile as sf

def record_audio(filename="input.wav", duration=5, fs=16000):
    print("🎙 녹음 시작...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    sf.write(filename, audio, fs)
    print(f"✅ 녹음 완료: {filename}")

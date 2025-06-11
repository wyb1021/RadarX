import pyaudio
import wave

# 녹음 설정
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
RECORD_SECONDS = 5
OUTPUT_WAV = "test.wav"

p = pyaudio.PyAudio()

# input_device_index: arecord -l 로 확인한 카드/디바이스 인덱스
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=1,  # 보통 1번 카드
                frames_per_buffer=CHUNK)

print("⏺ 녹음 시작… (5초간)")
frames = []
for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print("⏹ 녹음 완료")

stream.stop_stream()
stream.close()
p.terminate()

# WAV 파일로 저장
wf = wave.open(OUTPUT_WAV, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"✅ {OUTPUT_WAV} 파일 생성. 재생을 위해 아래 명령 실행하세요:")
print(f"   aplay {OUTPUT_WAV}")

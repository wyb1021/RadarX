# PC에서 음성 입력받고 텍스트로 변환, 감정분류하고 생성된 응답을 다시 음성으로 합성, 라즈베리파이로 송신
# emotion_client.py, record.py, stt.py, tts.py, transfer.py, config.py, llm_server_monologg와 연관
import os
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-7.1.1-essentials_build\bin"  # ✅ ffmpeg 경로 명시

from record import record_audio
from stt import transcribe
from emotion_client import get_emotion_response
from tts import synthesize
from transfer import send_to_rpi

def run_pipeline():
    record_audio("input.wav")
    text = transcribe("input.wav")
    print(f"📝 텍스트 추출 결과: {text}")

    result = get_emotion_response(text)
    if not result:
        print("❌ 감정 분석 실패")
        return

    print(f"🧠 감정: {result['emotion']}")
    print(f"💬 응답: {result['response']}")

    synthesize(result['response'], filename="response.wav")
    send_to_rpi("response.wav")

if __name__ == "__main__":
    run_pipeline()
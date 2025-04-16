# 라즈베리파이 음성 챗봇

이 프로젝트는 라즈베리파이4에서 Vosk, TinyLLM, eSpeak을 활용한 음성 기반 챗봇입니다.

## 기능
- 음성 인식 (Vosk)
- 자연어 처리 (TinyLLM)
- 음성 합성 (eSpeak)

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. Vosk 모델 다운로드:
```bash
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 model
```

3. eSpeak 설치:
```bash
sudo apt-get update
sudo apt-get install espeak
```

## 실행 방법
```bash
python voice_chatbot.py
```

## 주의사항
- 라즈베리파이4의 성능 제한으로 인해 TinyLLM 모델의 응답 시간이 다소 길 수 있습니다.
- 음성 인식 정확도를 높이기 위해 조용한 환경에서 사용하는 것을 권장합니다. 
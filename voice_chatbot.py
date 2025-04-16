import os
import json
import queue
import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess

class VoiceChatbot:
    def __init__(self):
        # Vosk 모델 초기화
        self.model = Model("model")  # Vosk 모델 경로 지정 필요
        self.recognizer = KaldiRecognizer(self.model, 16000)
        
        # TinyLLM 초기화
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.llm = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        
        # 오디오 설정
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        
        # 오디오 장치 설정
        try:
            self.input_device = sd.default.device[0]  # 기본 입력 장치 사용
            print(f"사용 중인 오디오 장치: {sd.query_devices(self.input_device)['name']}")
        except:
            print("오디오 장치를 찾을 수 없습니다. 마이크가 연결되어 있는지 확인해주세요.")
            self.input_device = None
        
    def record_audio(self, duration=5):
        """음성 녹음"""
        if self.input_device is None:
            print("오디오 장치가 없습니다.")
            return None
            
        print("음성 녹음 중...")
        try:
            recording = sd.rec(int(duration * self.sample_rate), 
                             samplerate=self.sample_rate, 
                             channels=1, 
                             dtype='float32',
                             device=self.input_device)
            sd.wait()
            return recording
        except Exception as e:
            print(f"녹음 중 오류 발생: {e}")
            return None
        
    def transcribe_speech(self, audio_data):
        """음성을 텍스트로 변환"""
        if audio_data is None:
            return ""
            
        if self.recognizer.AcceptWaveform(audio_data.tobytes()):
            result = json.loads(self.recognizer.Result())
            return result.get("text", "")
        return ""
        
    def generate_response(self, text):
        """텍스트에 대한 응답 생성"""
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.llm.generate(**inputs, max_length=100)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
    def text_to_speech(self, text):
        """텍스트를 음성으로 변환"""
        subprocess.run(['espeak', text])
        
    def run(self):
        """메인 실행 루프"""
        print("음성 챗봇 시작...")
        while True:
            try:
                # 음성 녹음
                audio = self.record_audio()
                if audio is None:
                    print("마이크를 확인하고 다시 시도해주세요.")
                    continue
                
                # 음성을 텍스트로 변환
                text = self.transcribe_speech(audio)
                if text:
                    print(f"인식된 텍스트: {text}")
                    
                    # 응답 생성
                    response = self.generate_response(text)
                    print(f"생성된 응답: {response}")
                    
                    # 음성 출력
                    self.text_to_speech(response)
                    
            except KeyboardInterrupt:
                print("챗봇 종료")
                break
            except Exception as e:
                print(f"오류 발생: {e}")
                continue

if __name__ == "__main__":
    chatbot = VoiceChatbot()
    chatbot.run() 
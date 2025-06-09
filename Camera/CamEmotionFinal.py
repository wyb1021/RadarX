import socket
import cv2
import tensorflow as tf
from deepface import DeepFace
import time
import os
import numpy as np
from collections import deque
import threading
from queue import Queue
import sys
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

DIRS = {'DATA_OUTPUT': 'data'}
EMOTION_CONFIG = {
    'smoothing_window': 5,
    'va_clip_range': (-1.0, 1.0)
}

class EmotionDetectionSystem:
    def __init__(self): 
        self.analysis_frame_count = 0

        # JSON 저장 경로 설정 전에 폴더부터 생성
        data_dir = DIRS['DATA_OUTPUT']
        os.makedirs(data_dir, exist_ok=True)

        # 디렉토리 설정
        self.valence_arousal_path = os.path.join(str(DIRS['DATA_OUTPUT']), 'v_a_camera.json')

        cv2.ocl.setUseOpenCL(True)

        try:
            from openvino import Core
            self.ie = Core()
            self.devices = self.ie.available_devices
            print(f"사용 가능한 OpenVINO 디바이스: {self.devices}")
        except ImportError:
            print("OpenVINO를 설치하세요: pip install openvino")

        self.EMOTION_VALUES = {
            'angry': {'valence': -0.3, 'arousal': 0.4},
            'disgust': {'valence': -0.4, 'arousal': -0.1},
            'fear': {'valence': -0.4, 'arousal': 0.4},
            'happy': {'valence': 0.4, 'arousal': 0.3},
            'sad': {'valence': -0.3, 'arousal': -0.2},
            'surprise': {'valence': 0.1, 'arousal': 0.4},
            'neutral': {'valence': 0.0, 'arousal': 0.0}
        }

        self.backends = [
            'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
            'retinaface', 'mediapipe', 'yolov8', 'yolov11s',
            'yolov11n', 'yolov11m', 'yunet', 'centerface',
        ]
        self.detector = 'fastmtcnn'  # 기본 얼굴 검출기 설정

        # 시간 측정 위한 변수
        self.detection_time = 0
        self.emotion_analysis_time = 0
        self.total_frames = 0
        # 프레임 큐와 결과 큐 생성
        self.frame_queue = deque(maxlen=2)
        self.result_queue = deque(maxlen=1)
        self.emotions_queue = deque(maxlen=5)
        self.last_result = None

    def detect_faces(self, frame):
        start_time = time.time()
        faces = []
        try:
            if cv2.ocl.haveOpenCL():
                frame_gpu = cv2.UMat(frame)
                frame_gpu = cv2.cvtColor(frame_gpu, cv2.COLOR_BGR2RGB)
            else:
                frame_gpu = frame

            result = DeepFace.extract_faces(img_path=frame, detector_backend=self.detector, enforce_detection=False)

            if result:
                for face in result:
                    fa = face['facial_area']
                    if isinstance(fa, dict):
                        x, y, w, h = fa['x'], fa['y'], fa['w'], fa['h']
                    else:
                        x1, y1, x2, y2 = fa
                        x, y = x1, y1
                        w, h = x2 - x1, y2 - y1
                    if w < frame.shape[1] and h < frame.shape[0]:
                        faces.append((x, y, w, h))

            faces.sort(key=lambda x: x[2]*x[3], reverse=True)
        except Exception as e:
            print(f"[❌ 얼굴 검출 오류] {e}")

        self.detection_time += time.time() - start_time
        return faces

    def calculate_va_values(self, emotions):
        total_val, total_aro, total_prob = 0, 0, 0
        for e, p in emotions.items():
            if e in self.EMOTION_VALUES:
                total_val += p * self.EMOTION_VALUES[e]['valence']
                total_aro += p * self.EMOTION_VALUES[e]['arousal']
                total_prob += p
        return (total_val / total_prob, total_aro / total_prob) if total_prob > 0 else (0, 0)

    def average_emotions(self, emotions_list):
        if not emotions_list:
            return None
        avg_emotions = {}
        for emotion in emotions_list[0].keys():
            avg_emotions[emotion] = sum(e[emotion] for e in emotions_list) / len(emotions_list)
        return avg_emotions

    def draw_bar_graph(self, frame, emotions, start_x, start_y, bar_width=30, bar_height=100, spacing=40):
        colors = {
            'angry': (0, 0, 255), 'disgust': (0, 255, 0), 'fear': (255, 0, 0),
            'happy': (0, 255, 255), 'sad': (255, 0, 255), 'surprise': (255, 255, 0),
            'neutral': (128, 128, 128)
        }
        for i, (emotion, score) in enumerate(emotions.items()):
            height = int(score * bar_height / 100)
            x = start_x + i * spacing
            y = start_y + bar_height - height
            cv2.rectangle(frame, (x, y), (x + bar_width, start_y + bar_height), colors[emotion], -1)
            cv2.putText(frame, emotion[:3], (x, start_y + bar_height + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.putText(frame, f"{score:.1f}%", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def draw_va_values(self, frame, valence, arousal, x, y):
        cv2.putText(frame, f"Valence: {valence:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, f"Arousal: {arousal:.2f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    def analyze_frames(self):
        while True:
            if not self.frame_queue:
                time.sleep(0.05)
                continue

            frame = self.frame_queue.popleft()  
            self.analysis_frame_count += 1
            if self.analysis_frame_count % 6 != 0:
                continue

            try:
                result = DeepFace.analyze(
                    frame,
                    actions=['emotion'],
                    detector_backend=self.detector,
                    enforce_detection=False,
                    silent=True
                )
                emotions = result[0]['emotion']
                self.emotions_queue.append(emotions)
                avg_emotions = self.average_emotions(list(self.emotions_queue))
                valence, arousal = self.calculate_va_values(avg_emotions)

                # 📁 JSON 저장
                try:
                    if os.path.exists(self.valence_arousal_path):
                        try:
                            with open(self.valence_arousal_path, 'r') as f:
                                data_list = json.load(f)
                        except json.JSONDecodeError:
                            data_list = []
                            with open(self.valence_arousal_path, 'w') as f:
                                json.dump(data_list, f, indent=4)
                    else:
                        data_list = []

                    va_data = {
                        'timestamp': time.time(),
                        'valence': float(valence),
                        'arousal': float(arousal)
                    }

                    data_list.append(va_data)

                    if len(data_list) > 100:
                        data_list = data_list[-100:]

                    with open(self.valence_arousal_path, 'w') as f:
                        json.dump(data_list, f, indent=4)

                except Exception as e:
                    print(f"[❌ JSON 저장 오류] {str(e)}")

                # 🖼 시각화
                output_frame = frame.copy()
                self.draw_bar_graph(output_frame, avg_emotions, 10, 10)
                self.draw_va_values(output_frame, valence, arousal, 10, 150)
                self.result_queue.clear()
                self.result_queue.append(output_frame)

            except Exception as e:
                print(f"[❌ 감정 분석 오류] {e}")

    def start(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(('0.0.0.0', 5001))
        print("📡 수신 대기 중...")

        window_name = "📡 Emotion Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)

        idle_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(idle_frame, "Waiting for frames...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        analysis_thread = threading.Thread(target=self.analyze_frames)
        analysis_thread.daemon = True
        analysis_thread.start()

        while True:
            try:
                sock.settimeout(0.1)
                received = False
                display_frame = idle_frame.copy()
                try:
                    data, addr = sock.recvfrom(65536)
                    img_array = np.frombuffer(data, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if frame is not None:
                        received = True
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        self.frame_queue.append(frame)
                        if self.result_queue:
                            display_frame = self.result_queue[-1].copy()
                        else:
                            display_frame = frame.copy()
                    else:
                        continue
                except socket.timeout:
                    if not received:
                        display_frame = idle_frame.copy()
                cv2.imshow(window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), 27]:
                    print("🛑 종료 요청됨")
                    break
            except Exception as e:
                print(f"[❌ 예외 발생] {e}")
                break
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    emotion_system = EmotionDetectionSystem()
    emotion_system.start()

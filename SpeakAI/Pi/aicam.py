import cv2
import tensorflow as tf
print(tf.__version__)
from deepface import DeepFace
import time
import numpy as np
from collections import deque
import threading
from queue import Queue

# OpenCL 사용 (Intel GPU 가속용)
cv2.ocl.setUseOpenCL(True)

# 감정별 Valence/Arousal 값
EMOTION_VALUES = {
    'angry': {'valence': -0.3, 'arousal': 0.4},
    'disgust': {'valence': -0.4, 'arousal': -0.1},
    'fear': {'valence': -0.4, 'arousal': 0.4},
    'happy': {'valence': 0.4, 'arousal': 0.3},
    'sad': {'valence': -0.3, 'arousal': -0.2},
    'surprise': {'valence': 0.1, 'arousal': 0.4},
    'neutral': {'valence': 0.0, 'arousal': 0.0}
}

# 얼굴 검출 백엔드 설정
detector = 'ssd'

# 프레임/결과 큐
frame_queue = Queue(maxsize=2)
result_queue = Queue(maxsize=1)
last_result = None

def detect_faces(frame):
    faces = []
    try:
        result = DeepFace.extract_faces(img_path=frame, detector_backend=detector, enforce_detection=False)
        for face in result:
            x, y, w, h = face['facial_area'].values()
            if w != frame.shape[1] and h != frame.shape[0]:
                faces.append((x, y, w, h))
        faces.sort(key=lambda x: x[2]*x[3], reverse=True)
    except Exception as e:
        print(f"얼굴 검출 오류: {str(e)}")
    return faces

def calculate_va_values(emotions):
    valence, arousal, total = 0, 0, 0
    for e, prob in emotions.items():
        if e in EMOTION_VALUES:
            valence += prob * EMOTION_VALUES[e]['valence']
            arousal += prob * EMOTION_VALUES[e]['arousal']
            total += prob
    return (valence / total, arousal / total) if total else (0, 0)

def average_emotions(emotions_list):
    avg = {}
    for key in emotions_list[0]:
        avg[key] = sum(e[key] for e in emotions_list) / len(emotions_list)
    return avg

def draw_bar_graph(frame, emotions, x, y, width=30, height=100, spacing=40):
    colors = {
        'angry': (0, 0, 255), 'disgust': (0, 255, 0), 'fear': (255, 0, 0),
        'happy': (0, 255, 255), 'sad': (255, 0, 255), 'surprise': (255, 255, 0),
        'neutral': (128, 128, 128)
    }
    for i, (emotion, score) in enumerate(emotions.items()):
        h = int(score * height / 100)
        x_pos = x + i * spacing
        y_pos = y + height - h
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + width, y + height), colors[emotion], -1)
        cv2.putText(frame, emotion[:3], (x_pos, y + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{score:.1f}%", (x_pos, y_pos - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_va_values(frame, valence, arousal, x, y):
    cv2.putText(frame, f"Valence: {valence:.2f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Arousal: {arousal:.2f}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def capture_frames(cap):
    global last_result
    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임 읽기 실패")
            break
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except: pass
        frame_queue.put(frame.copy())
        result = result_queue.get() if not result_queue.empty() else last_result
        if result:
            last_result = result
            disp = frame.copy()
            for (x, y, w, h) in result['faces']:
                cv2.rectangle(disp, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if 'emotions' in result:
                draw_bar_graph(disp, result['emotions'], 20, 40)
            if 'valence' in result and 'arousal' in result:
                draw_va_values(disp, result['valence'], result['arousal'], 20, 200)
            cv2.imshow("Emotion Detection", disp)
        else:
            cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def analyze_frames():
    while True:
        if frame_queue.empty():
            time.sleep(0.01)
            continue
        frame = frame_queue.get()
        faces = detect_faces(frame)
        if not faces:
            continue
        x, y, w, h = faces[0]
        roi = frame[max(0, y-20):y+h+20, max(0, x-20):x+w+20]
        if roi.shape[0] < 100 or roi.shape[1] < 100:
            continue
        try:
            res = DeepFace.analyze(img_path=roi, actions=['emotion'], enforce_detection=False, silent=True)
            emotions_queue.append(res[0]['emotion'])
            if emotions_queue:
                avg = average_emotions(list(emotions_queue))
                valence, arousal = calculate_va_values(avg)
                if result_queue.full():
                    try: result_queue.get_nowait()
                    except: pass
                result_queue.put({
                    'frame': frame,
                    'faces': faces,
                    'emotions': avg,
                    'valence': valence,
                    'arousal': arousal
                })
        except Exception as e:
            print(f"감정 분석 오류: {str(e)}")

def main():
    # 감정 결과 큐 초기화
    global emotions_queue
    emotions_queue = deque(maxlen=5)

    # 📸 V4L2 기반 IMX500 카메라 열기 (/dev/video10 등)
    cap = cv2.VideoCapture("/dev/video0")

    # 해상도 설정
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("카메라 열기 실패")
        return

    # 스레드 시작
    threading.Thread(target=capture_frames, args=(cap,), daemon=True).start()
    threading.Thread(target=analyze_frames, daemon=True).start()

    try:
        while True: time.sleep(0.1)
    except KeyboardInterrupt:
        print("종료합니다")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import socket
import cv2
import numpy as np

# 소켓 생성 및 바인딩
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 5001))  # 라즈베리파이에서 전송한 영상 수신

print("✅ 수신 대기 중...")

while True:
    data, _ = sock.recvfrom(65536)
    
    # JPEG 디코딩
    img_array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is not None:
        # ✅ 색상 보정 (RGB → BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 화면 출력
        cv2.imshow("📡 Received from Pi", frame)
    else:
        print("❌ 디코딩 실패 - 손상된 패킷입니다.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

sock.close()
cv2.destroyAllWindows()

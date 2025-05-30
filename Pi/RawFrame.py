import cv2
import socket

# ✅ 카메라 초기화
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()

# ✅ UDP 소켓 설정
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_ip = "192.168.137.1"  # ← 여기에 PC IP 넣기
server_port = 5001

# ✅ 전송 루프
while True:
    frame = picam2.capture_array()


    # JPEG 압축 시 품질 낮추기 (0~100, 낮을수록 더 압축됨)
    _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    # ✅ 소켓으로 전송
    sock.sendto(buf.tobytes(), (server_ip, server_port))

    print(f"✅ 프레임 전송 완료 ({len(buf)} bytes)")
import os
from config import RPI_USER, RPI_IP, RPI_PATH

def send_to_rpi(local_file="response.wav"):
    os.system(f"scp {local_file} {RPI_USER}@{RPI_IP}:{RPI_PATH}")
    print("📤 전송 완료")


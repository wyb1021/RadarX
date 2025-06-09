import requests
from config import SERVER_IP, SERVER_PORT

def get_emotion_response(text: str):
    try:
        res = requests.post(f"http://{SERVER_IP}:{SERVER_PORT}/generate", json={"prompt": text})
        return res.json() if res.status_code == 200 else None
    except:
        return None


# 한국어 기반 가장 반응속도 빠른 모델, 프롬프트로 감정분류, 아직은 온전하지 않음. 긍정/부정 뒤바뀔때 있음

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import random

# ✅ 모델 변경: monologg/koelectra-base-discriminator
model_name = "monologg/koelectra-base-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cpu")
model.to(device)

# ✅ 감정 라벨 정의
labels = ["부정", "긍정"]

# ✅ 응답 템플릿 다양화 (20개씩)
response_map = {
    "긍정": [
        "좋은 기분이시군요! 그 기분 오래 유지되길 바라요 ☀️",
        "오늘도 기분 좋게 하루 보내세요! 😊",
        "행복한 하루가 되시길 바랍니다!",
        "그 에너지로 오늘도 멋지게 보내세요!",
        "당신의 미소가 참 보기 좋아요 😄",
        "오늘은 뭔가 잘 풀릴 것 같네요!",
        "긍정적인 기운이 전해져요!",
        "당신의 하루가 즐겁기를 바랍니다 🌼",
        "그 기분 오래오래 유지되기를!",
        "웃는 얼굴이 참 잘 어울려요!",
        "기분 좋을 땐 주변 사람도 행복해져요 😊",
        "당신의 밝은 모습이 멋져요!",
        "좋은 일이 생길 것 같은 예감이에요!",
        "기분 좋은 하루 보내세요!",
        "당신이 행복하니 저도 기뻐요!",
        "그 감정, 마음껏 누리세요 ☀️",
        "행복한 하루 보내세요 💛",
        "오늘 당신에게 좋은 일이 가득하길!",
        "기분이 좋아지는 말이네요!",
        "당신의 행복이 오래가기를 바랍니다!"
    ],
    "부정": [
        "힘든 마음 이해해요. 잠시 쉬어가도 괜찮아요.",
        "요즘 많이 지치셨죠? 제가 옆에 있어줄게요.",
        "그런 날도 있죠. 괜찮아요. 당신은 충분히 잘하고 있어요 💙",
        "마음이 힘들 땐 누구나 무너질 수 있어요. 괜찮아요.",
        "지금 그 감정도 중요한 당신의 일부예요.",
        "스스로를 너무 몰아붙이지 마세요.",
        "누구나 약해질 수 있어요. 당신은 혼자가 아니에요.",
        "천천히 가도 괜찮아요. 함께 걸어갈게요.",
        "당신의 속도대로, 당신의 방식대로 살아가도 괜찮아요.",
        "그 감정을 외면하지 말고 잠시 안아주세요.",
        "당신은 이미 충분히 잘 해내고 있어요.",
        "조금 쉬어가도 괜찮아요. 당신은 소중한 사람이에요.",
        "무슨 일이 있었든, 당신 잘못이 아니에요.",
        "힘들다고 말해줘서 고마워요.",
        "당신의 마음을 이해해주고 싶은 사람이 여기에 있어요.",
        "너무 오래 참고 있진 않나요? 당신을 응원해요.",
        "지금 이 순간도 지나갈 거예요.",
        "당신의 마음을 토닥토닥 해주고 싶어요.",
        "그 어떤 감정도 괜찮아요. 그건 당신이 살아있다는 증거예요.",
        "마음이 지쳤다면, 충분히 쉬어도 돼요."
    ]
}

# ✅ Flask 앱 초기화
app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    try:
        text = request.json.get("prompt", "")
        if not text.strip():
            return jsonify({"response": "❗ 입력이 비어 있습니다."}), 400

        # 감정 분류
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            label_index = torch.argmax(probs, dim=-1).item()
            label = labels[label_index]

        # 랜덤 응답 선택
        response = random.choice(response_map[label])
        return jsonify({"response": response, "emotion": label})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"response": f"❗ 서버 오류: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

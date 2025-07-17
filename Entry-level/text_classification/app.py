from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# 加载模型和tokenizer
model = BertForSequenceClassification.from_pretrained("./fine_tuned_bert")
tokenizer = BertTokenizer.from_pretrained("./fine_tuned_bert")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# 分类函数
def predict(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()[0]


@app.route("/predict", methods=["POST"])
def inference():
    data = request.get_json()
    text = data["text"]
    probabilities = predict(text)

    # 返回概率分布（或转换为标签）
    return jsonify({
        "probabilities": probabilities.tolist(),
        "predicted_label": int(probabilities.argmax())
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
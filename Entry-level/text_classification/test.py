from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 加载分词器和模型（根据保存路径修改）
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_bert")
model = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_bert")

# 若有GPU可用，将模型移动到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 设置为评估模式

def predict(text):
    # 文本编码
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    ).to(device)

    # 模型推理
    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"]
        )

    # 获取预测结果
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class, probabilities.cpu().numpy()[0]

# 使用示例
text = "你猜这家酒店好不好"
predicted_class, probs = predict(text)
print(f"预测类别: {predicted_class}")
print(f"类别概率: {probs}")

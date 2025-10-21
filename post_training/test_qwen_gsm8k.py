import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import argparse

import re
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def extract_answer(output: str) -> str:
    """
    从模型输出中提取最终答案（鲁棒版）
    支持格式：
      - ... The answer is 42.
      - ... #### 42
      - ... 答案是 42。
      - ... 42
    """
    # 方法1: 匹配 "#### X"（GSM8K 官方格式）
    match = re.search(r"####\s*(\d+)", output)
    if match:
        return match.group(1).strip()

    # 方法2: 匹配 "The answer is X"（含中英文）
    match = re.search(r"(?:answer is|答案是|最终答案|is)\s*[:：]?\s*(\d+)", output, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 方法3: 提取最后一个正整数（最通用）
    numbers = re.findall(r"\d+", output)
    if numbers:
        return numbers[-1]

    # 无法解析，返回空
    return ""


def evaluate_gsm8k(model, tokenizer, test_data, max_new_tokens=512):
    correct = 0
    total = len(test_data)

    for item in tqdm(test_data, desc="Evaluating GSM8K"):
        question = item["question"]
        ground_truth = item["answer"].replace(",", "").strip()

        # 构造 prompt（根据模型调整，这里用简单格式）
        prompt = f"Question: {question}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
            pad_token_id=tokenizer.eos_token_id
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        model_output = output_text[len(prompt):].strip()

        pred = extract_answer(model_output)
        if pred == ground_truth:
            correct += 1

    accuracy = correct / total
    print(f"GSM8K Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy

def is_correct(predicted, ground_truth):
    """检查预测答案是否正确"""
    if predicted is None:
        return False

    # 提取数字部分
    pred_match = re.search(r'[\d.\-\+]+', str(predicted))
    gt_match = re.search(r'[\d.\-\+]+', str(ground_truth))

    if pred_match and gt_match:
        try:
            pred_num = float(pred_match.group())
            gt_num = float(gt_match.group())
            return abs(pred_num - gt_num) < 1e-6
        except ValueError:
            return False
    return False

def evaluate_model(model_name, subset_size=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    """评估模型在GSM8K数据集上的性能"""
    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="../model_cache")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir="../model_cache",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    # 如果分词器没有pad_token，设置一个
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main")
    test_dataset = dataset["test"]

    # 如果指定了子集大小，则只使用该数量的样本
    if subset_size:
        test_dataset = test_dataset.select(range(min(subset_size, len(test_dataset))))

    print(f"Evaluating on {len(test_dataset)} examples...")

    correct = 0
    total = len(test_dataset)

    # 格式化提示
    def format_prompt(question):
        messages = [
            {"role": "user", "content": f"Solve the math problem below. At the end of your response, write 'Answer: ' followed by the numerical answer.\n\nProblem: {question}"}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    for i, example in enumerate(test_dataset):
        question = example["question"]
        ground_truth = extract_answer(example["answer"])

        # 格式化提示
        prompt = format_prompt(question)

        # 编码输入
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        # 生成答案
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        # 解码输出
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取模型答案（改进方法）
        model_answer = generated_text

        predicted = extract_answer(model_answer)

        # 检查答案是否正确
        if is_correct(predicted, ground_truth):
            correct += 1

        # 打印前几个示例的结果
        # if i < 3:  # 显示前3个示例的详细信息
        print(f"\n=== Example {i + 1} ===")
        print(f"Question: {question}")
        print(f"Expected answer: {ground_truth}")
        print(f"Full generated text: {repr(generated_text)}")
        print(f"Extracted prediction: {predicted}")
        print(f"Correct: {is_correct(predicted, ground_truth)}")
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{total} examples, Current Accuracy: {correct/(i+1):.4f}")

    accuracy = correct / total
    print(f"\nFinal Results:")
    print(f"Model: {model_name}")
    print(f"Tested on: {total} examples")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Qwen model on GSM8K dataset")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name or path")
    parser.add_argument("--subset_size", type=int, default=None, help="Evaluate on a subset of examples")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")

    args = parser.parse_args()

    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_model(args.model_name, args.subset_size, device)
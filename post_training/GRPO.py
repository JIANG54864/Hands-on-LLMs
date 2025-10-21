# Warning control
import warnings
warnings.filterwarnings('ignore')


import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import re

# 用于记录训练loss的回调函数
class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.steps = []
        self.rewards = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)
            print(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")
        
        # 打印生成的文本示例，帮助调试
        if logs and "rewards" in logs:
            self.rewards.extend(logs["rewards"])
            avg_reward = sum(logs["rewards"]) / len(logs["rewards"])
            print(f"Step {state.global_step}: Average Reward = {avg_reward:.4f}")
            
            # 如果有生成的文本示例，也打印出来帮助调试
            if "responses" in logs:
                print(f"Step {state.global_step}: Sample responses = {logs['responses'][:2]}...")

# 1. 加载模型和分词器
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 改为Qwen2.5-0.5B模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="../model_cache")
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True,
                                                            cache_dir="../model_cache")

# 如果分词器没有pad_token，设置一个
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2. 加载并预处理GSM8K数据集
dataset = load_dataset("openai/gsm8k", "main")  # 使用OpenAI的GSM8K数据集:cite[2]
# 假设我们使用训练集
train_dataset = dataset["train"]


# 定义一个函数来格式化数据，适配模型的聊天模板
def format_dataset(example):
    # 根据Qwen2.5的聊天模板格式化数据:cite[8]
    # 例如: [{"role": "user", "content": "问题..."}]
    # 具体格式请参考Qwen2.5的官方文档
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": example["question"]}],
        tokenize=False,
        add_generation_prompt=True
    )
    return {"prompt": formatted_prompt, "answer": example["answer"]}


# 应用格式化函数
train_dataset = train_dataset.map(format_dataset)

# 添加一个函数来为GRPO训练准备数据
def prepare_grpo_dataset(example):
    # 保留原始格式化后的prompt和answer
    return {"prompt": example["prompt"], "answer": example["answer"]}


# 应用GRPO数据准备函数
train_dataset = train_dataset.map(prepare_grpo_dataset)

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

# 3. 定义奖励函数
# 奖励函数是GRPO的核心，用于评估生成内容的质量:cite[9]
def math_reasoning_reward(completions, prompts=None, **kwargs):
    """
    根据数学推理的正确性和格式给予奖励。
    由于GSM8K的答案有固定格式（以####结尾），可以据此判断正误:cite[2]
    """
    rewards = []
    
    # 从kwargs中获取标准答案
    answers = None
    if 'answer' in kwargs:
        answers = kwargs['answer']
    
    for i, completion in enumerate(completions):
        reward = 0.0
        # 更精确的奖励：尝试提取答案并比较
        if answers is not None:
            # 提取模型生成的答案
            model_answer = extract_answer(completion)
            # 提取标准答案
            ground_truth = extract_answer(str(answers[i]))

            if model_answer and ground_truth:
                try:
                    model_answer = float(model_answer)
                    ground_truth = float(ground_truth)
                    # 如果答案正确，给予高奖励
                    if abs(model_answer - ground_truth) < 1e-6:
                        reward += 1.0
                except ValueError:
                    # 如果无法转换为数字，给予较小的奖励
                    pass
        rewards.append(reward)

    return rewards


# 4. 配置训练参数
training_args = GRPOConfig(
    output_dir="./qwen2.5-0.5b-grpo-gsm8k",
    # 批次大小相关参数，适用于24GB显存的RTX 3090
    per_device_train_batch_size=8,  # 增加批次大小以更好地利用显存
    gradient_accumulation_steps=1,  # 减少梯度累积步数
    num_generations=4,  # 每个提示生成的样本数

    # 学习率与优化器
    learning_rate=5e-6,  # 降低学习率以提高训练稳定性
    max_grad_norm=0.5,   # 增加梯度裁剪阈值

    # 生成参数
    max_prompt_length=512,
    max_completion_length=256,

    # 训练步数
    # max_steps=1000,  # 训练步数
    max_steps=100,  # 小型实验
    logging_steps=1,  # 更频繁地记录loss
    save_steps=50,

    # 重要：GRPO特定参数:cite[4]
    beta=0.05,  # 降低KL散度系数以减少正则化强度

    # 启用vLLM加速（如果已安装）:cite[9]
    # use_vllm=True,
)

# 初始化loss记录回调
loss_callback = LossLoggingCallback()

# 5. 初始化GRPOTrainer并开始训练
trainer = GRPOTrainer(
    model=model,
    # 将tokenizer参数名改为processing_class
    processing_class=tokenizer,
    reward_funcs=math_reasoning_reward,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[loss_callback],  # 添加回调函数
)

# 开始训练
trainer.train()

# 保存训练好的模型
trainer.save_model("./qwen2.5-0.5b-grpo-gsm8k-final")

# 绘制loss曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_callback.steps, loss_callback.losses, marker='o', linestyle='-')
plt.title('GRPO Training Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('grpo_training_loss.png')
plt.show()

# 如果有奖励数据，也绘制奖励曲线
if loss_callback.rewards:
    plt.figure(figsize=(10, 6))
    # 计算每个step的平均奖励
    avg_rewards = []
    num_generations = training_args.num_generations
    for i in range(0, len(loss_callback.rewards), num_generations):
        avg_reward = sum(loss_callback.rewards[i:i+num_generations]) / num_generations
        avg_rewards.append(avg_reward)
    plt.plot(range(len(avg_rewards)), avg_rewards, marker='o', linestyle='-', color='orange')
    plt.title('GRPO Average Reward')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Reward')
    plt.grid(True)
    plt.savefig('grpo_training_reward.png')
    plt.show()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import deque
import gymnasium as gym
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from modelscope import AutoModelForCausalLM as ModelScopeAutoModelForCausalLM
from modelscope import AutoTokenizer as ModelScopeAutoTokenizer
import matplotlib.pyplot as plt  # 添加绘图库


# 设置随机种子以确保可重复性
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(66)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 1. 环境设置 - 使用一个简化的文本环境
class TextEnv(gym.Env):
    def __init__(self, base_model, tokenizer, max_length=20):
        super(TextEnv, self).__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.current_text = ""
        self.action_space = gym.spaces.Discrete(tokenizer.vocab_size)
        self.observation_space = gym.spaces.Box(
            low=0, high=tokenizer.vocab_size,
            shape=(max_length,), dtype=np.int32
        )

    def reset(self):
        self.current_text = ""
        return self._get_obs()

    def _get_obs(self):
        # 将当前文本转换为token id序列
        if not self.current_text:  # 处理空文本情况
            tokens = np.array([self.tokenizer.bos_token_id] if self.tokenizer.bos_token_id is not None else [0])
        else:
            tokens = self.tokenizer.encode(self.current_text, return_tensors="np", add_special_tokens=False)[0]
        
        # 填充或截断到固定长度
        if len(tokens) < self.max_length:
            tokens = np.pad(tokens, (0, self.max_length - len(tokens)), 'constant')
        else:
            tokens = tokens[:self.max_length]
        return tokens

    def step(self, action):
        # 将动作（token id）添加到当前文本
        new_token = self.tokenizer.decode([action])
        self.current_text += new_token

        # 每一步都给予奖励，而不仅仅在结束时
        # 基于生成token的质量给予即时奖励
        reward = 0

        # 对生成的token给予即时奖励
        if new_token.strip():  # 如果不是空白字符
            reward += 0.1
        if new_token.isalpha():  # 如果是字母
            reward += 0.1
        if len(new_token.strip()) > 1:  # 如果是较长的token
            reward += 0.2

        # 新增: 鼓励生成中文或英文字符
        # 检查是否包含中文字符
        import re
        if re.search(r'[\u4e00-\u9fff]', new_token):  # 中文字符范围
            reward += 0.3
        # 英文单词奖励
        if re.match(r'^[a-zA-Z]+$', new_token.strip()):
            reward += 0.2

        # 惩罚连续生成标点符号
        if len(self.current_text) > 1:
            if not self.current_text[-2].isalnum() and not new_token.isalnum() and new_token.strip():
                reward -= 0.1

        # 强化惩罚：连续相同标点符号
        if len(self.current_text) > 2:
            # 检查最后三个字符是否相同且都是标点符号
            last_chars = self.current_text[-3:]
            if len(set(last_chars)) == 1 and not last_chars[0].isalnum():
                reward -= 0.5  # 更强的惩罚
                
        # 鼓励生成有意义的文本片段
        current_words = self.current_text.split()
        if len(current_words) > 0:
            # 如果最后一个词有一定长度，给予奖励
            if len(current_words[-1]) > 2:
                reward += 0.1
            # 如果有多个词，给予额外奖励
            if len(current_words) > 1:
                reward += 0.2

        # 检查是否结束（修改：基于文本长度和标点符号）
        done = len(self.tokenizer.encode(self.current_text, add_special_tokens=False)) >= self.max_length
        # 修改结束条件，增加随机性，避免过早收敛
        if ("." in new_token or "!" in new_token or "?" in new_token) and random.random() < 0.7:
            done = True

        # 在结束时给予完成奖励
        if done:
            # 基于最终文本长度和质量的奖励
            reward += min(len(self.current_text) / 5, 2.0)  # 长度奖励
            if "good" in self.current_text.lower() or "great" in self.current_text.lower():
                reward += 1.0
            if "bad" in self.current_text.lower() or "terrible" in self.current_text.lower():
                reward -= 0.5
                
            # 惩罚无意义文本
            if len(self.current_text.strip()) == 0:
                reward -= 1.0
            elif len(set(self.current_text.replace(" ", ""))) == 1:  # 所有字符相同
                reward -= 2.0

        # 获取新观察值
        obs = self._get_obs()

        return obs, reward, done, {}


# 2. 奖励模型
class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size=256):
        super(RewardModel, self).__init__()
        self.base_model = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # 使用最后一个token的隐藏状态
        last_hidden_state = outputs.hidden_states[-1]
        # 获取序列中最后一个非填充token的隐藏状态
        if attention_mask is not None:
            last_token_positions = attention_mask.sum(1) - 1
            last_hidden_state = last_hidden_state[torch.arange(last_hidden_state.size(0)), last_token_positions]
        else:
            last_hidden_state = last_hidden_state[:, -1, :]

        reward = self.reward_head(last_hidden_state)
        return reward

    def compute_reward(self, input_ids, attention_mask=None):
        """
        计算文本序列的奖励值
        """
        with torch.no_grad():
            reward = self.forward(input_ids, attention_mask)
        return reward.squeeze(-1) if reward.dim() > 1 else reward


# 3. 策略模型（基于预训练语言模型）
class PolicyModel(nn.Module):
    def __init__(self, base_model):
        super(PolicyModel, self).__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        return logits

    def get_action(self, state, sample=True):
        # 确保state是二维的 [1, seq_len]
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # 将state移到GPU上
        state = state.to(device)
        
        # 创建attention mask以正确处理序列
        pad_token_id = self.base_model.config.pad_token_id
        if pad_token_id is None:
            # 如果没有设置pad_token_id，使用eos_token_id或者默认值0
            pad_token_id = getattr(self.base_model.config, 'eos_token_id', 0)
        
        # 确保pad_token_id不是None后再进行比较
        if pad_token_id is not None:
            attention_mask = (state != pad_token_id).long()
        else:
            # 如果pad_token_id为None，则创建全1的attention_mask
            attention_mask = torch.ones_like(state, dtype=torch.long)

        # 获取当前状态的logits
        with torch.no_grad():
            logits = self.forward(state.long(), attention_mask=attention_mask)
            # 获取最后一个位置的logits
            next_token_logits = logits[0, -1, :]

        # 采样或选择最高概率的token
        if sample:
            # 添加温度参数控制探索程度
            temperature = 1.2  # 提高温度以增加探索性
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            action = torch.multinomial(probs, 1).item()
        else:
            action = torch.argmax(next_token_logits).item()

        return action


# 4. 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


# 5. PPO算法实现
class PPO:
    def __init__(self, policy_model, reward_model, tokenizer, lr=1e-4, gamma=0.99, clip_epsilon=0.2):
        self.policy = policy_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        # 将模型移到GPU上
        self.policy.to(device)
        self.reward_model.to(device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=lr*0.1)  # 奖励模型学习率更低
        self.losses = []  # 添加用于记录loss的列表

    def compute_advantages(self, rewards, values, dones, next_values):
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        # 使用广义优势估计(GAE)
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * next_values[t] - values[t]
            advantages[t] = delta + self.gamma * last_advantage
            last_advantage = advantages[t]

        return advantages

    def update(self, states, actions, rewards, next_states, dones, epochs=3):
        # 确定设备
        device = next(self.policy.parameters()).device

        # 创建attention masks
        pad_token_id = self.policy.base_model.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = getattr(self.policy.base_model.config, 'eos_token_id', 0)
        
        # 处理pad_token_id为None的情况
        if pad_token_id is not None:
            attention_masks = (torch.tensor(states, dtype=torch.long) != pad_token_id).long().to(device)
            next_attention_masks = (torch.tensor(next_states, dtype=torch.long) != pad_token_id).long().to(device)
        else:
            # 如果pad_token_id为None，则创建全1的attention_mask
            attention_masks = torch.ones(torch.tensor(states, dtype=torch.long).shape, dtype=torch.long).to(device)
            next_attention_masks = torch.ones(torch.tensor(next_states, dtype=torch.long).shape, dtype=torch.long).to(device)
        
        states = torch.tensor(states, dtype=torch.long).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.long).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        # 多轮更新
        epoch_losses = []  # 记录每个epoch的loss
        for _ in range(epochs):
            # 获取当前策略的logits
            logits = self.policy(states, attention_masks)
            # 获取对应动作的log概率
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

            # 计算价值函数（简化版本）
            with torch.no_grad():
                values = self.reward_model(states, attention_masks).squeeze()
                next_values = self.reward_model(next_states, next_attention_masks).squeeze()

            # 计算优势函数
            # 确保所有输入都是正确形状的数组
            batch_size = states.shape[0]

            # 确保所有值都是一维数组
            if values.dim() == 0:  # 标量
                values_np = values.cpu().numpy().repeat(batch_size)
            else:
                values_np = values.cpu().numpy()

            if next_values.dim() == 0:  # 标量
                next_values_np = next_values.cpu().numpy().repeat(batch_size)
            else:
                next_values_np = next_values.cpu().numpy()

            if dones.dim() == 0:  # 标量
                dones_np = dones.cpu().numpy().repeat(batch_size)
            else:
                dones_np = dones.cpu().numpy()

            # 如果数组形状不正确，需要调整
            if values_np.ndim == 0:
                values_np = np.full(batch_size, values_np.item())
            if next_values_np.ndim == 0:
                next_values_np = np.full(batch_size, next_values_np.item())
            if dones_np.ndim == 0:
                dones_np = np.full(batch_size, dones_np.item())

            # 使用奖励作为优势的简化版本（替代复杂的GAE计算）
            advantages = rewards.cpu().numpy() - values_np
            
            advantages = torch.tensor(advantages, dtype=torch.float).to(device)

            # 计算旧策略的概率（用于比率计算）
            with torch.no_grad():
                old_logits = self.policy(states, attention_masks)
                old_log_probs = torch.log_softmax(old_logits[:, -1, :], dim=-1)
                old_action_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze()

            # 计算概率比率
            ratio = torch.exp(action_log_probs - old_action_log_probs)

            # 计算PPO目标函数
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            # 最终损失函数
            policy_loss = -torch.min(surr1, surr2).mean()

            # 添加损失值检查，确保训练信号正常
            if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                print("Warning: NaN or Inf loss detected")
                continue

            # 更新策略
            self.optimizer.zero_grad()
            policy_loss.backward()

            # 添加梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

            self.optimizer.step()

            # 记录当前epoch的loss
            epoch_losses.append(policy_loss.item())
            
        # 只在所有epochs结束后打印一次平均loss
        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"Policy loss: {avg_loss:.4f}")
        
        # 记录本轮更新的平均loss
        if epoch_losses:
            self.losses.extend(epoch_losses)
        return epoch_losses  # 返回本轮的loss值列表


# 6. 人类反馈数据收集（简化版本）
def collect_human_feedback(prompts, responses):
    # 在实际应用中，这里会有真实的人类评估
    # 这里我们使用一个简单的启发式规则来模拟人类反馈
    rewards = []
    for response in responses:
        # 简单规则：响应长度和包含关键词的奖励
        score = min(len(response.split()) / 10, 1.0)  # 长度奖励
        if "good" in response.lower() or "great" in response.lower():
            score += 0.5
        if "bad" in response.lower() or "terrible" in response.lower():
            score -= 0.3
        rewards.append(score)
    return np.array(rewards)

# 7. 奖励模型训练函数
def train_reward_model(reward_model, reward_optimizer, texts, human_rewards, tokenizer, device):
    reward_model.train()
    
    # 编码文本
    encodings = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    human_rewards = torch.tensor(human_rewards, dtype=torch.float32).to(device)
    
    # 计算模型预测的奖励
    predicted_rewards = reward_model(input_ids, attention_mask=attention_mask).squeeze()
    
    # 计算损失（均方误差）
    loss = nn.MSELoss()(predicted_rewards, human_rewards)
    
    # 更新奖励模型
    reward_optimizer.zero_grad()
    loss.backward()
    reward_optimizer.step()
    
    reward_model.eval()
    return loss.item()

# 8. 主训练循环
def main():
    os.environ['MODELSCOPE_CACHE'] = "../model_cache"
    os.environ['HF_HOME'] = "../model_cache"
    # 初始化模型和tokenizer
    model_name = "qwen/Qwen3-0.6B"  # ModelScope上的模型名称
    tokenizer = ModelScopeAutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir="../model_cache")
    base_model = ModelScopeAutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir="../model_cache")

    # 添加pad token如果不存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 初始化环境、策略和奖励模型
    env = TextEnv(base_model, tokenizer)
    policy_model = PolicyModel(ModelScopeAutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir="../model_cache"))
    reward_model = RewardModel(ModelScopeAutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir="../model_cache"))

    # 初始化PPO
    ppo = PPO(policy_model, reward_model, tokenizer)

    # 初始化经验回放缓冲区
    replay_buffer = ReplayBuffer(10000)

    # 训练参数
    num_episodes = 500
    batch_size = 32  # 增加批量大小以提高训练稳定性
    reward_update_interval = 10

    # 记录训练过程中的奖励和loss
    episode_rewards = []
    all_losses = []

    # 训练循环
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        # 收集一个episode的数据
        states, actions, rewards, next_states, dones = [], [], [], [], []

        while not done:
            # 使用当前策略选择动作
            action = policy_model.get_action(torch.tensor(state, dtype=torch.long))


            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储转换
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            total_reward += reward

        # 将episode数据添加到回放缓冲区
        for i in range(len(states)):
            replay_buffer.push(states[i], actions[i], rewards[i], next_states[i], dones[i])

        # 定期更新奖励模型
        if episode % reward_update_interval == 0 and len(replay_buffer) >= batch_size:
            # 采样一批数据
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
                batch_size)

            # 解码响应以获取文本
            responses = []
            for state_seq in batch_states:
                # 解码非填充token
                non_padded_tokens = state_seq[state_seq != tokenizer.pad_token_id]
                # 确保token是整数类型
                non_padded_tokens = non_padded_tokens.astype(np.int64)
                text = tokenizer.decode(non_padded_tokens)
                responses.append(text)

            # 收集人类反馈（模拟）
            human_rewards = collect_human_feedback([""] * batch_size, responses)

            # 训练奖励模型
            reward_loss = train_reward_model(reward_model, ppo.reward_optimizer, responses, human_rewards, tokenizer, device)
            print(f"Episode {episode}, Reward Model Loss: {reward_loss:.4f}")

        # 更新策略
        if len(replay_buffer) >= batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
                    batch_size)
            losses = ppo.update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)
            all_losses.extend(losses)

        episode_rewards.append(total_reward)
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

        # 打印episode中的一些文本示例，帮助调试
        if episode % 10 == 0:
            print(f"Sample text: {env.current_text}")

        # 定期评估策略
        if episode % 50 == 0 and episode > 0:
            print("Evaluating current policy...")
            eval_rewards = []
            for _ in range(5):  # 运行5个评估episode
                state = env.reset()
                done = False
                total_eval_reward = 0
                while not done:
                    action = policy_model.get_action(torch.tensor(state, dtype=torch.long), sample=False)  # 评估时使用贪婪策略
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    total_eval_reward += reward
                eval_rewards.append(total_eval_reward)
            print(f"Evaluation - Average reward: {np.mean(eval_rewards):.2f}")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    
    # 绘制奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # 绘制loss曲线
    plt.subplot(1, 2, 2)
    if all_losses:  # 只有在有loss数据时才绘制
        plt.plot(all_losses)
    plt.title('Training Loss')
    plt.xlabel('Update Step')
    plt.ylabel('Policy Loss')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()


if __name__ == "__main__":
    main()
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

    def step(self, action):
        # 将动作（token id）添加到当前文本
        new_token = self.tokenizer.decode([action])
        self.current_text += new_token

        # 检查是否结束（简化：达到最大长度或生成了句号）
        done = len(self.current_text) >= self.max_length or "." in new_token

        # 奖励初始为0，将由奖励模型提供
        reward = 0

        # 获取新观察值
        obs = self._get_obs()

        return obs, reward, done, {}

    def _get_obs(self):
        # 将当前文本转换为token id序列
        tokens = self.tokenizer.encode(self.current_text, return_tensors="np")[0]
        # 填充或截断到固定长度
        if len(tokens) < self.max_length:
            tokens = np.pad(tokens, (0, self.max_length - len(tokens)), 'constant')
        else:
            tokens = tokens[:self.max_length]
        return tokens


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
        
        # 获取当前状态的logits
        with torch.no_grad():
            logits = self.forward(state.long())
            # 获取最后一个位置的logits
            next_token_logits = logits[0, -1, :]

        # 采样或选择最高概率的token
        if sample:
            probs = torch.softmax(next_token_logits, dim=-1)
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

        states = torch.tensor(states, dtype=torch.long).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        next_states = torch.tensor(next_states, dtype=torch.long).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)

        # 计算奖励（使用奖励模型）
        with torch.no_grad():
            reward_values = self.reward_model(states).squeeze()

        # 多轮更新
        for _ in range(epochs):
            # 获取当前策略的logits
            logits = self.policy(states)
            # 获取对应动作的log概率
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze()

            # 计算价值函数（简化版本）
            with torch.no_grad():
                values = self.reward_model(states).squeeze()
                next_values = self.reward_model(next_states).squeeze()

            # 计算优势函数
            # 确保所有输入都是正确形状的数组
            batch_size = states.shape[0]

            # 确保所有值都是一维数组
            if reward_values.dim() == 0:  # 标量
                rewards_np = reward_values.cpu().numpy().repeat(batch_size)
            else:
                rewards_np = reward_values.cpu().numpy()

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
            if rewards_np.ndim == 0:
                rewards_np = np.full(batch_size, rewards_np.item())
            if values_np.ndim == 0:
                values_np = np.full(batch_size, values_np.item())
            if next_values_np.ndim == 0:
                next_values_np = np.full(batch_size, next_values_np.item())
            if dones_np.ndim == 0:
                dones_np = np.full(batch_size, dones_np.item())

            advantages = self.compute_advantages(rewards_np, values_np, dones_np, next_values_np)
            advantages = torch.tensor(advantages, dtype=torch.float).to(device)

            # 计算旧策略的概率（用于比率计算）
            with torch.no_grad():
                old_logits = self.policy(states)
                old_log_probs = torch.log_softmax(old_logits[:, -1, :], dim=-1)
                old_action_log_probs = old_log_probs.gather(1, actions.unsqueeze(1)).squeeze()

            # 计算概率比率
            ratio = torch.exp(action_log_probs - old_action_log_probs)

            # 计算PPO目标函数
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            # 最终损失函数
            policy_loss = -torch.min(surr1, surr2).mean()

            # 更新策略
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()


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


# 7. 主训练循环
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
    batch_size = 1
    reward_update_interval = 10

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
        if episode % reward_update_interval == 0 and len(replay_buffer) > batch_size:
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

            # 更新奖励模型
            # 这里简化处理，实际应用中需要训练奖励模型
            print(f"Collected human feedback with average reward: {human_rewards.mean():.3f}")

        # 更新策略
        if len(replay_buffer) > batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(
                batch_size)
            ppo.update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones)

        print(f"Episode {episode}, Total Reward: {total_reward:.2f}")

        # 定期评估策略
        if episode % 50 == 0:
            print("Evaluating current policy...")
            eval_rewards = []
            for _ in range(5):  # 运行5个评估episode
                state = env.reset()
                done = False
                total_eval_reward = 0
                while not done:
                    action = policy_model.get_action(torch.tensor(state, dtype=torch.long), sample=False)
                    next_state, reward, done, _ = env.step(action)
                    state = next_state
                    total_eval_reward += reward
                eval_rewards.append(total_eval_reward)
            print(f"Evaluation - Average reward: {np.mean(eval_rewards):.2f}")


if __name__ == "__main__":
    main()
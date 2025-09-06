import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import time

# ========================
# 1. 模型定义
# ========================

class TinyGPTConfig:
    def __init__(self):
        self.vocab_size = 50257  # GPT-2 tokenizer 的词表大小
        self.block_size = 128    # 序列长度
        self.n_layer = 4
        self.n_head = 4
        self.n_embd = 256
        self.dropout = 0.1

class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # 权重共享
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"输入长度 {t} 超过 {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            x = checkpoint(block, x)  # 👈 启用梯度检查点！

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1)

        return logits, loss

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x

# ========================
# 2. 数据集类
# ========================

class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.examples = []
        tokenized_text = tokenizer.encode(text)

        # 滑动窗口切分
        for i in range(0, len(tokenized_text) - block_size + 1, block_size // 2):  # 步长设为一半，增加样本
            self.examples.append(tokenized_text[i:i + block_size])

        print(f"共加载 {len(self.examples)} 个训练样本")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

# ========================
# 3. 训练主函数
# ========================

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 初始化模型
    config = TinyGPTConfig()
    model = TinyGPT(config)
    model.to(device)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")

    # 加载数据
    dataset = TextDataset(tokenizer, "凡人utf8.txt", block_size=config.block_size)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # ⚠️ batch_size=8 是安全起点

    # 优化器 & 混合精度
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = GradScaler()  # 👈 混合精度训练核心

    # 训练参数
    epochs = 3
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            inputs, targets = batch[:, :-1], batch[:, 1:]  # 语言模型：用前T-1个预测后T-1个

            optimizer.zero_grad()

            # 👇 混合精度上下文
            with autocast():
                logits, loss = model(inputs, targets)

            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            if step % 50 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} 完成，平均损失: {avg_loss:.4f}，耗时: {epoch_time:.2f}秒")

        # 每个 epoch 后保存模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f"nanoGPT2_epoch_{epoch+1}.pt")

    print("训练完成！")

# ========================
# 4. 推理函数
# ========================

def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    model.eval()
    device = next(model.parameters()).device

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids

    for _ in range(max_new_tokens):
        logits, _ = model(generated)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        generated = torch.cat([generated, next_token], dim=1)

        # 如果生成了结束符，可以提前停止（可选）
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)

# ========================
# 5. 主程序入口
# ========================

if __name__ == "__main__":
    train()

    config = TinyGPTConfig()
    model = TinyGPT(config)
    checkpoint_data = torch.load("nanoGPT2_epoch_epoch_3.pt", map_location='cpu')
    model.load_state_dict(checkpoint_data['model_state_dict'])
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(generate_text(model, tokenizer, "所谓元婴之下第一人，"))
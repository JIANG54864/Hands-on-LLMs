import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.wq = nn.Linear(hidden_size, hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.wo = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)  # 现代大模型通过海量数据防止过拟合，可以不用dropout

    def forward(self, q,k,v, mask=None):  # x: [batch_size, seq_len, hidden_size]
        batch_size = q.size()[0]
        query = self.wq(q)
        key = self.wk(k)
        value = self.wv(v)

        query = self.split_head(query)
        key = self.split_head(key)
        value = self.split_head(value)

        attn = torch.matmul(query, key.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim))
        if mask is not None:
            attn += mask * (-1e9)
            # attn = attn.masked_fill(mask == 0, -1e9) 另一种写法，mask中1表示保留的0被去掉
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)  # 在注意力权重上应用dropout
        output = torch.matmul(attn, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.wo(output)
        # 可选: 在输出投影后也添加dropout
        output = self.dropout(output)
        return output

    def split_head(self, x):
        batch_size = x.size()[0]
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]

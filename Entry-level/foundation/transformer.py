import torch
import torch.nn as nn

from MHA import MHA


class Encoder(nn.Module):
    def __init__(self,hidden_size,num_heads,ffn_hidden_size,dropout=0.1):
        super().__init__()
        self.self_attention=MHA(hidden_size,num_heads)
        self.ln1=nn.LayerNorm(hidden_size)
        self.ln2=nn.LayerNorm(hidden_size)
        self.ffn=nn.Sequential(
            nn.Linear(hidden_size,ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffn_hidden_size,hidden_size),
            nn.Dropout(dropout) # 现代大模型通过海量数据防止过拟合，可以不用dropout
        )

    def forward(self,x):
        # 现代大模型都用pre- norm，原始模型用post-norm
        x=x+self.self_attention(self.ln1(x),self.ln1(x),self.ln1(x))
        x=x+self.ffn(self.ln2(x))
        return x

class EncoderDecoderAttn(nn.Module):
    def __init__(self,hidden_size,num_heads, mask= None):
        super().__init__()
        self.hidden_size=hidden_size
        self.cross_attn=MHA(hidden_size,num_heads)
        self.mask = mask
    def forward(self,x,encoder_out):
        return self.cross_attn(encoder_out,x,x, mask=self.mask)

class Decoder(nn.Module):
    def __init__(self,hidden_size,num_heads,ffn_hidden_size,mask=None,dropout=0.1):
        super().__init__()
        self.self_attention=MHA(hidden_size,num_heads)
        self.cross_attention=EncoderDecoderAttn(hidden_size,num_heads)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size,ffn_hidden_size),
            nn.ReLU(),
            nn.Linear(ffn_hidden_size,hidden_size),
            nn.Dropout(dropout)
        )
        self.mask = mask

    def forward(self,x,encoder_out):
        x=x+self.self_attention(self.ln1(x),self.ln1(x),self.ln1(x), mask=self.mask)
        x=x+self.cross_attention(self.ln2(x),encoder_out)
        x=x+self.ffn(self.ln3(x))
        return x
import torch
import torch.nn as nn

class MHA(nn.Module):
    def __init__(self,hidden_size,num_heads):
        super().__init__()
        self.num_heads=num_heads
        self.head_dim=hidden_size//num_heads

        self.q=nn.Linear(hidden_size,hidden_size)
        self.k=nn.Linear(hidden_size,hidden_size)
        self.v=nn.Linear(hidden_size,hidden_size)
        self.o=nn.Linear(hidden_size,hidden_size)

    def forward(self,x,mask=None): # x: [batch_size, seq_len, hidden_size]
        batch_size=x.size()[0]
        query=self.q(x)
        key=self.k(x)
        value=self.v(x)

        query=self.split_head(query)
        key=self.split_head(key)
        value=self.split_head(value)

        attn=torch.matmul(query,key.transpose(1,2))/torch.sqrt(self.head_dim)
        if mask is not None:
            attn+=mask*(-1e9)
        attn=torch.softmax(attn,-1)
        output=torch.matmul(attn,value)
        output=output.transpose(-1,-2).contiguous().view(batch_size,-1,self.num_heads*self.head_dim)
        output=self.o(output)
        return output



    def split_head(self,x):
        batch_size=x.size()[0]
        return x.view(batch_size,-1,self.num_heads,self.head_dim).transpose(1,2) # [batch_size, seq_len, num_heads, head_dim]
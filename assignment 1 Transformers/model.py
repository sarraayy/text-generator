# model.py
import torch
import torch.nn as nn
import math

def attention(q, k, v, mask=None):
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return torch.matmul(torch.softmax(scores, dim=-1), v)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model*3)
        self.out = nn.Linear(d_model, d_model)
        self.seq_len = seq_len

    def forward(self, x):
        B, T, _ = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        out = attention(q, k, v, mask)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, self.head_dim*self.n_heads)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, seq_len):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, seq_len)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_len=128, n_layers=4, d_model=128, n_heads=4, d_ff=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, d_ff, seq_len) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    def forward(self, idx):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln(x)
        return self.head(x)

"""
contains basic building blocks of GPT architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as torch_f

from model.moe_blocks import SparseMoe


class SelfAttention(nn.Module):
    def __init__(self, emb_dim, head_dim, block_size):
        super().__init__()
        self.head_dim = head_dim

        self.query = nn.Linear(emb_dim, head_dim, bias=False)
        self.key = nn.Linear(emb_dim, head_dim, bias=False)
        self.value = nn.Linear(emb_dim, head_dim, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        b, t, c = x.shape
        # all with shape B, T, C
        query_vec = self.query(x)
        key_vec = self.key(x)
        value_vec = self.value(x)
        # dot product between (B, T, C) and (B, C, T) -> (B, T, T)
        scaled_dot_prod = (query_vec @ key_vec.transpose(-2, -1)) * (self.head_dim**-0.5)
        # masking to conceal "future" tokens
        masked_dot_prod = scaled_dot_prod.masked_fill(self.tril[:t, :t] == 0, float("-inf"))

        attn_prob = torch_f.softmax(masked_dot_prod, dim=-1)
        attn_prob = self.dropout(attn_prob)
        attn_scores = attn_prob @ value_vec
        return attn_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, block_size):
        super().__init__()

        self.mh_sa = nn.ModuleList([SelfAttention(emb_dim, emb_dim//num_heads, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.mh_sa], dim=-1)
        x = self.dropout(self.proj(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(p=0.5),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # configs["emb_dim"], configs["num_heads"],
        # configs["block_size"]
        self.mh_sa = MultiHeadAttention(configs["num_heads"], configs["emb_dim"], configs["block_size"])
        self.norm1 = nn.LayerNorm(configs["emb_dim"])
        if configs["is_moe"]:
            self.ffw = SparseMoe(configs["num_experts"], configs["num_topk"], configs["emb_dim"])
        else:
            self.ffw = FeedForward(configs["emb_dim"])
        self.norm2 = nn.LayerNorm(configs["emb_dim"])

    def forward(self, x):

        x = x + self.mh_sa(x)
        x = self.norm1(x)
        x = x + self.ffw(x)
        x = self.norm2(x)
        return x

"""
contains basic building blocks of GPT architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as torch_f

from model.moe_blocks import SparseMoe


class Attention(nn.Module):
    def __init__(self, emb_dim, head_dim, block_size):
        super().__init__()
        self.head_dim = head_dim

        self.query = nn.Linear(emb_dim, head_dim, bias=False)
        self.key = nn.Linear(emb_dim, head_dim, bias=False)
        self.value = nn.Linear(emb_dim, head_dim, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        return self.self_attention(x, q, k, v)

    def self_attention(self, x, query_vec, key_vec, value_vec):
        b, t, c = x.shape
        # all with shape B, T, C

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

        self.mh_sa = nn.ModuleList([Attention(emb_dim, emb_dim//num_heads, block_size) for _ in range(num_heads)])
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.mh_sa], dim=-1)
        x = self.dropout(self.proj(x))
        return x


class MultiQueryAttention(Attention):
    """
    implement MQA from the paper https://arxiv.org/abs/1911.02150
    """
    def __init__(self, num_heads, emb_dim, block_size):
        super().__init__(emb_dim, emb_dim//num_heads, block_size)
        delattr(self, "query")

        self.queries = nn.ModuleList([nn.Linear(emb_dim, emb_dim//num_heads, bias=False) for _ in range(num_heads)])
        self.key = nn.Linear(emb_dim, emb_dim//num_heads, bias=False)
        self.values = nn.Linear(emb_dim, emb_dim // num_heads, bias=False)

        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        v_vec = self.values(x)
        k_vec = self.key(x)

        x = torch.cat([self.self_attention(x, q_vec(x), k_vec, v_vec)for  q_vec in self.queries], dim=-1)
        return self.dropout(self.proj(x))


class GroupQueryAttention(Attention):
    def __init__(self, num_heads, num_groups, emb_dim, block_size):
        super().__init__(emb_dim, emb_dim // num_heads, block_size)
        if num_heads % num_groups != 0:
            raise ValueError("Number of heads must be divisible by number of groups")

        delattr(self, "query")  # Remove the individual query from base class

        self.num_groups = num_groups
        self.group_size = num_heads // num_groups  # Heads per group

        self.queries = nn.ModuleList([nn.Linear(emb_dim, emb_dim // num_heads, bias=False) for _ in range(num_heads)])
        # single k-v per group
        self.keys = nn.ModuleList([nn.Linear(emb_dim, emb_dim // num_heads, bias=False) for _ in range(num_groups)])
        self.values = nn.ModuleList([nn.Linear(emb_dim, emb_dim // num_heads, bias=False) for _ in range(num_groups)])

        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        all_head_outputs = []

        for g in range(self.num_groups):
            # shared K vector for group g
            k_vec = self.keys[g](x)
            v_vec = self.values[g](x)

            for h in range(self.group_size):
                head_idx = g * self.group_size + h  # Calculate head index
                q_vec = self.queries[head_idx](x)  # Query for this specific head
                head_output = self.self_attention(x, q_vec, k_vec, v_vec)  # Use group's K and V
                all_head_outputs.append(head_output)

        x = torch.cat(all_head_outputs, dim=-1)
        return self.dropout(self.proj(x))


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
        if configs["attention"] == "vanilla":
            self.attn = MultiHeadAttention(configs["num_heads"], configs["emb_dim"], configs["block_size"])
        elif configs["attention"] == "multi_query":
            self.attn = MultiQueryAttention(configs["num_heads"], configs["emb_dim"], configs["block_size"])
        elif configs["attention"] == "group_query":
            self.attn = GroupQueryAttention(configs["num_heads"], configs["emb_dim"], configs["block_size"],
                                            configs["num_groups"])

        self.norm1 = nn.LayerNorm(configs["emb_dim"])
        if configs["is_moe"]:
            self.ffw = SparseMoe(configs["num_experts"], configs["num_topk"], configs["emb_dim"])
        else:
            self.ffw = FeedForward(configs["emb_dim"])
        self.norm2 = nn.LayerNorm(configs["emb_dim"])

    def forward(self, x):

        x = x + self.attn(x)
        x = self.norm1(x)
        x = x + self.ffw(x)
        x = self.norm2(x)
        return x

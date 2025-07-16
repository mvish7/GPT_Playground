"""
contains basic building blocks of GPT architecture
"""

import torch
import torch.nn as nn
from fontTools.unicodedata import block

from model.positional_embeddings import RotaryPosEmbed
import torch.nn.functional as torch_f

from model.moe_blocks import SparseMoe


class Attention(nn.Module):

    def __init__(self, emb_dim, num_heads, block_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.query = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, emb_dim, bias=False)
        self.value = nn.Linear(emb_dim, emb_dim, bias=False)

        self.register_buffer("tril",
                             torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # Calculate queries for current input
        # reshape into (b, t, mum_heads, head_dim) and transpose it to (B, num_heads, T, head_dim)
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is None:
            # Prefill mode - compute and store KV cache
            k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            kv_cache = (k, v)
        else:
            # Decode mode - use and update KV cache
            k_cache, v_cache = kv_cache
            # Compute K,V for current token only
            k = self.key(x[:, -1:, :]).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.value(x[:, -1:, :]).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            # Concatenate with cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
            kv_cache = (k, v)

        return self.self_attention(x, q, k, v), kv_cache

    def self_attention(self, x, query_vec, key_vec, value_vec):
        b, t, c = x.shape
        # all with shape B, T, C

        # dot product between (B, H, T, C) and (B, H, C, T) -> (B, H, T, T)
        scaled_dot_prod = (query_vec @ key_vec.transpose(-2, -1)) * (
            self.head_dim**-0.5)
        # masking to conceal "future" tokens
        masked_dot_prod = scaled_dot_prod.masked_fill(self.tril[:t, :t] == 0,
                                                      float("-inf"))

        attn_prob = torch_f.softmax(masked_dot_prod, dim=-1)
        attn_prob = self.dropout(attn_prob)
        attn_scores = attn_prob @ value_vec
        attn_scores = attn_scores.transpose(1, 2).contiguous().view(b, t, c)
        return attn_scores


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, emb_dim, block_size):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads

        self.mh_sa = Attention(emb_dim, num_heads, block_size)

        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, kv_cache=None):
        x, updated_kv_cache = self.mh_sa(x, kv_cache)
        x = self.dropout(self.proj(x))
        return x, updated_kv_cache


class MultiQueryAttention(Attention):
    """
    implement MQA from the paper https://arxiv.org/abs/1911.02150
    """

    def __init__(self, num_heads, emb_dim, block_size):
        super().__init__(emb_dim, num_heads, block_size)
        delattr(self, "query")
        delattr(self, "value")
        delattr(self, "key")

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        self.queries = nn.Linear(emb_dim, emb_dim, bias=False)
        self.key = nn.Linear(emb_dim, self.head_dim, bias=False)
        self.value = nn.Linear(emb_dim, self.head_dim, bias=False)

        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # Calculate queries for all heads - (B, T, nH, Hd) -> (B, nH, T, Hd)
        q_vecs = self.queries(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is None:
            # Prefill mode - compute and store KV cache
            # both k, v in (B, T, D)
            k_vec = self.key(x)
            v_vec = self.value(x)
            kv_cache = (k_vec, v_vec)
        else:
            # Decode mode - use and update KV cache
            k_cache, v_cache = kv_cache
            # Compute K,V for current token only
            # k and v in (B, 1, D)
            k_vec = self.key(x[:, -1:, :])
            v_vec = self.value(x[:, -1:, :])
            # Concatenate with cache
            k_vec = torch.cat([k_cache, k_vec], dim=1)
            v_vec = torch.cat([v_cache, v_vec], dim=1)
            kv_cache = (k_vec, v_vec)

        # Compute attention for each head using shared K,V
        # q is (B, nH, T, Hd) and (k,v) are in (B, T, D) -> (k and v) are shared for each head hence broadcasting
        x = self.self_attention(x, q_vecs, k_vec.unsqueeze(1), v_vec.unsqueeze(1))

        return self.dropout(self.proj(x)), kv_cache


class GroupQueryAttention(Attention):

    def __init__(self, num_heads, emb_dim, block_size, num_groups):
        super().__init__(emb_dim, num_heads, block_size)
        if num_heads % num_groups != 0:
            raise ValueError(
                "Number of heads must be divisible by number of groups")

        delattr(self, "query")  # Remove the individual query from base class
        delattr(self, "key")
        delattr(self, "value")

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.num_groups = num_groups
        self.group_size = num_heads // num_groups  # Heads per group

        self.queries = nn.Linear(emb_dim, emb_dim, bias=False)
        self.keys = nn.Linear(emb_dim, emb_dim, bias=False)
        self.values = nn.Linear(emb_dim, emb_dim, bias=False)

        self.proj = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        # q: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = self.queries(x).view(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if kv_cache is None:
            # k/v: (B, T, C) -> (B, T, num_groups, group_size, head_dim)
            k_full = self.keys(x).view(B, T, self.num_groups, self.group_size, self.head_dim)
            # Since KV are shared across group_size heads, pick only the first head in each group
            k = k_full[:, :, :, 0, :]  # (B, T, num_groups, head_dim)
            k = k.permute(0, 2, 1, 3).contiguous()  # (B, num_groups, T, head_dim)
            # broadcasting to match the dimensions to query -- just repeat across num_groups dimensions
            k = k.repeat_interleave(self.group_size, dim=1)

            v_full = self.values(x).view(B, T, self.num_groups, self.group_size, self.head_dim)
            v = v_full[:, :, :, 0, :]  # (B, T, num_groups, head_dim)
            v = v.permute(0, 2, 1, 3).contiguous()  # (B, num_groups, T, head_dim)
            v = v.repeat_interleave(self.group_size, dim=1)

            kv_cache = (k, v)
        else:
            k_cache, v_cache = kv_cache
            # calculating k for only last token
            # k with shape (B, T, num_groups, group_size, head_dim)
            k =  self.keys(x[:, -1, :]).view(B, T, self.num_groups, self.group_size, self.head_dim)
            k = k[:, :, :, 0, :]  # (B, T, num_groups, head_dim)
            k = k.permute(0, 2, 1, 3).contiguous()  # (B, num_groups, T, head_dim)
            # broadcasting to match the dimensions to query -- just repeat across num_groups dimensions
            k = k.repeat_interleave(self.group_size, dim=1)
            k_cache = torch.cat((k_cache, k), dim=2)

            # calculating v for last token
            v = self.values(x[:, -1, :]).view(B, T, self.num_groups, self.group_size, self.head_dim)
            v = v[:, :, :, 0, :]  # (B, T, num_groups, head_dim)
            v = v.permute(0, 2, 1, 3).contiguous()  # (B, num_groups, T, head_dim)
            v = v.repeat_interleave(self.group_size, dim=1)
            v_cache = torch.cat((v_cache, v), dim=2)
            kv_cache = (k_cache, v_cache)

        x = self.self_attention(x, q, k, v)

        return self.dropout(self.proj(x)), kv_cache

class MultiLatentAttention(Attention):
    def __init__(self, emb_dim, latent_dim, num_heads, block_size):
        super().__init__(emb_dim, num_heads, block_size)

        head_dim = emb_dim // num_heads

        self.rope = RotaryPosEmbed(block_size, emb_dim, device="cuda:0")

        self.wdkv = nn.Linear(emb_dim, latent_dim,  bias=False)
        self.dq = nn.Linear(emb_dim, latent_dim, bias=False)
        self.wuq = nn.Linear(emb_dim, emb_dim,  bias=False)
        self.wuk = nn.Linear(latent_dim, emb_dim,  bias=False)
        self.wuv = nn.Linear(latent_dim, emb_dim,  bias=False)
        self.wkr = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wo = nn.Linear(emb_dim, emb_dim, bias=False)

        self.register_buffer("absorbed_k", None)

    def forward(self, x, kv_cache=None):
        ckv = self.wdkv(x)
        kc = self.wuk(ckv)
        temp_wkr = self.wkr(x)
        kr = self.rope(temp_wkr)

        keys = torch.cat((kc, kr))

        values = self.wuv(ckv)

        a = 1









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
            self.attn = MultiHeadAttention(configs["num_heads"],
                                           configs["emb_dim"],
                                           configs["block_size"])
        elif configs["attention"] == "multi_query":
            self.attn = MultiQueryAttention(configs["num_heads"],
                                            configs["emb_dim"],
                                            configs["block_size"])
        elif configs["attention"] == "group_query":
            self.attn = GroupQueryAttention(configs["num_heads"],
                                            configs["emb_dim"],
                                            configs["block_size"],
                                            configs["num_groups"])
        elif configs["attention"] == "multi_latent":
            self.attn = MultiLatentAttention(configs["emb_dim"],
                                             configs["latent_dim"],
                                             configs["num_heads"],
                                             configs["block_size"])

        self.norm1 = nn.LayerNorm(configs["emb_dim"])
        if configs["is_moe"]:
            self.ffw = SparseMoe(configs["num_experts"], configs["num_topk"],
                                 configs["emb_dim"])
        else:
            self.ffw = FeedForward(configs["emb_dim"])
        self.norm2 = nn.LayerNorm(configs["emb_dim"])

    def forward(self, x, kv_cache=None):
        # Attention block
        attn_out, updated_kv_cache = self.attn(x, kv_cache)
        x = x + attn_out
        x = self.norm1(x)

        # Feed forward block
        x = x + self.ffw(x)
        x = self.norm2(x)
        return x, updated_kv_cache

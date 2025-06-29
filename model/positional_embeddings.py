import torch
import torch.nn as nn


class AbsoluteLearntPosEmbed(nn.Module):
    """
    implements a learnt positional embedding wrt model's context length i.e. block size
    """
    def __init__(self, block_size, emb_dim):
        super().__init__()

        self.pos_emb = nn.Embedding(block_size, emb_dim)

    def forward(self, seq_len):
        token_ids_vec = torch.arange(seq_len).to("cuda")
        return self.pos_emb(token_ids_vec)


class AbsoluteSinusoidalPosEmbed:
    """
    implements absolute sinusoidal positional embeddings
    """
    def __init__(self, emb_dim, n=10000):
        self.emb_dim = emb_dim
        self.n = n

    def __call__(self, x):
        B, T, _ = x.shape
        pos = torch.arange(T).unsqueeze(1)
        dimensions = torch.arange(self.emb_dim).unsqueeze(0)

        pos_emb = torch.zeros(T, self.emb_dim)
        angle_rates = 1 / torch.pow(self.n, (2 * (dimensions // 2)) / self.emb_dim)
        angle_rads = pos * angle_rates

        pos_emb[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_emb[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        return pos_emb


class RotaryPosEmbed:
    pass





if __name__ == "__main__":
    abs_pos_emb = AbsoluteSinusoidalPosEmbed(16)
    ip = torch.randn((2, 20, 16), dtype=torch.float)
    abs_pos_emb(ip)
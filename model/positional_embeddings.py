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
    """
    implements rotary positional embedding
    """
    def __init__(self, block_size, emb_dim):
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.rotary_emb = None
        self._create_rotary_embeddings()

    def _create_rotary_embeddings(self):
        assert self.emb_dim % 2 == 0, "embeddings dimension must be a multiple of 2"

        pos = torch.arange(self.block_size).unsqueeze(1)
        dimensions = torch.arange(0, self.emb_dim, 2, dtype=torch.float16).unsqueeze(0)
        inv_freq = torch.exp(dimensions * -(torch.log(torch.tensor(10000.0)) / self.emb_dim))

        angle_rates = pos * inv_freq

        self.rotary_emb = torch.zeros(self.block_size, self.emb_dim)
        self.rotary_emb[:, 0::2] = torch.sin(angle_rates)
        self.rotary_emb[:, 1::2] = torch.cos(angle_rates)

    def __call__(self, x):
        B, T, C = x.shape

        # pair up the dimensions of input tensor
        x_paired = x.reshape(B, T, C//2, 2)

        # Get the sin and cos components of the rotary embeddings
        sin_embeddings = self.rotary_emb[:, 0::2].unsqueeze(0)
        cos_embeddings = self.rotary_emb[:, 1::2].unsqueeze(0)

        # Apply the rotation
        x_rotated = torch.zeros_like(x_paired)
        x_rotated[..., 0] = x_paired[..., 0] * cos_embeddings - x_paired[..., 1] * sin_embeddings
        x_rotated[..., 1] = x_paired[..., 0] * sin_embeddings + x_paired[..., 1] * cos_embeddings

        # Reshape back to the original shape
        return x_rotated.reshape(B, T, C)






if __name__ == "__main__":
    # abs_pos_emb = AbsoluteSinusoidalPosEmbed(16)
    rot_pos_emb = RotaryPosEmbed(8, 16)

    ip = torch.randn((2, 8, 16), dtype=torch.float)
    # abs_pos_emb(ip)
    rot_pos_emb(ip)
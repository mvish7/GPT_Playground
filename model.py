"""
implements a gpt2-nano model
"""

import torch
import torch.nn as nn
import torch.nn.functional as torch_f


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
    def __init__(self, emb_dim, num_heads, block_size):
        super().__init__()

        self.mh_sa = MultiHeadAttention(num_heads, emb_dim, block_size)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffw = FeedForward(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x):

        x = x + self.mh_sa(x)
        x = self.norm1(x)
        x = x + self.ffw(x)
        x = self.norm2(x)
        return x


class NanoGPT(nn.Module):
    def __init__(self, num_blocks, emb_dim, num_heads, block_size, vocab_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        # positional embeddings -- as a  block_size of input will be fed to n/w at a time,shape of block_size.
        self.positional_embedding = nn.Embedding(block_size, emb_dim)
        # all transformer blocks stacked together
        self.blocks = nn.Sequential(*[TransformerBlock(emb_dim, num_heads, block_size) for _ in range(num_blocks)])
        # final prediction over all words in the vocab
        self.lm_pred = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape

        token_emb = self.token_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(T, device="cuda"))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.lm_pred(x)
        return x

    def generate_greedy(self, idx, max_new_tokens):
        """
        implements greedy decoding
        :param idx: ids of tokens
        :param max_new_tokens: max num tokens to generate
        :return:
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[0, -1, :] # becomes (B, C)
            # find token with max score
            token_id = torch.argmax(logits).unsqueeze(0)

            # optionally --- apply softmax to get probabilities
            # probs = torch_f.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            # idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx.squeeze(-1), token_id), dim=-1).unsqueeze(-1) # (B, T+1)
        return idx[:, 0]

    def generate_beam_search(self, idx, max_new_tokens, beam_size):
        """
        implements beam search decoding
        :param idx: ids of tokens
        :param max_new_tokens: max num tokens to generate
        :param beam_size: num of beams (individual streams) to keep track of
        :return:
        """
        B = idx.shape[0]
        # initializing the beams
        beams = [(idx, 0)]
        finished_beams = []

        for _ in range(max_new_tokens):
            candidates = []

            for seq, score in beams:
                # Crop seq to the last block_size tokens (context window)
                seq_cond = seq[:, -self.block_size:]

                # Get logits from the model for the current beam
                logits = self(seq_cond)

                # Focus only on the last time step (the new token predictions)
                logits = logits[0, -1, :]  # Shape (vocab_size,)

                # Apply log-softmax to get log-probabilities
                log_probs = torch.log_softmax(logits, dim=-1)

                # Find the top beam_size candidates for this beam
                top_log_probs, top_token_ids = torch.topk(log_probs, beam_size)

                for log_prob, token_id in zip(top_log_probs, top_token_ids):
                    new_seq = torch.cat([seq.squeeze(-1), token_id.unsqueeze(0)], dim=-1).unsqueeze(-1)  # (B, T+1)
                    new_score = score + log_prob.item()  # Sum of log-probabilities
                    candidates.append((new_seq, new_score))

            ordered = sorted(candidates, key=lambda tup: tup[1], reverse=True)
            beams = ordered[:beam_size]

        best_seq = max(beams, key=lambda tup: tup[1])[0]
        return best_seq[:, 0]


if __name__ == "__main__":
    ip = torch.randn((4, 8, 32))

    msa = NanoGPT(6, 64, 8, 8, 50)
    op = msa(ip)
    a = 1
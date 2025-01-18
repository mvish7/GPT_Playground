"""
implements a gpt2-nano model
"""

import torch
import torch.nn as nn

from model.gpt_blocks import TransformerBlock


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
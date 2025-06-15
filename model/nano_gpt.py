"""
implements a gpt2-nano model
"""
import yaml
import torch
import torch.nn as nn

from model.gpt_blocks import TransformerBlock
from model.multi_token_prediction import MultiTokenPred


class NanoGPT(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.block_size = configs["block_size"]
        self.token_embedding = nn.Embedding(configs["vocab_size"],
                                            configs["emb_dim"])
        # positional embeddings -- as a  block_size of input will be fed to n/w at a time,shape of block_size.
        self.positional_embedding = nn.Embedding(configs["block_size"],
                                                 configs["emb_dim"])
        # all transformer blocks stacked together
        self.blocks = nn.Sequential(
            *[TransformerBlock(configs) for _ in range(configs["num_blocks"])])
        if configs["multi_token_pred"]["do_mtp"]:
            self.mtp_head = MultiTokenPred(configs)
        else:
            # final prediction over all words in the vocab
            self.lm_pred = nn.Linear(configs["emb_dim"], configs["vocab_size"])

    def forward(self, x, kv_cache=None):
        B, T = x.shape

        token_emb = self.token_embedding(x)
        pos_emb_seq = torch.arange(T).to("cuda")
        pos_emb = self.positional_embedding(pos_emb_seq)
        x = token_emb + pos_emb

        # Initialize kv_cache if not provided
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        # Process through transformer blocks
        for block_idx, block in enumerate(self.blocks):
            x, block_kv_cache = block(x, kv_cache[block_idx])
            kv_cache[block_idx] = block_kv_cache

        if self.configs["multi_token_pred"]["do_mtp"]:
            x = self.mtp_head(x, token_emb)
        else:
            x = self.lm_pred(x)
        return x, kv_cache

    def generate_greedy(self, idx, max_new_tokens=100):
        """
        implements greedy decoding with KV cache
        :param idx: ids of tokens (B, T) where B is batch size, T is sequence length
        :param max_new_tokens: max num tokens to generate
        :return: generated sequence of shape (B, T+max_new_tokens)
        """
        B, T = idx.shape

        # Ensure input sequence is within block size
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = self.block_size

        # Prefill phase - establish KV cache
        logits, kv_cache = self(idx)

        # Decode phase - use and update KV cache
        for _ in range(max_new_tokens):
            # Get the last token's logits
            logits = logits[:, -1, :]  # (B, C) where C is vocab size

            # Sample next token
            if self.configs["multi_token_pred"]["do_mtp"]:
                token_id = torch.argmax(logits, dim=2)  # (B, 1)
            else:
                token_id = torch.argmax(logits, dim=-1).unsqueeze(-1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat([idx, token_id], dim=1)  # (B, T+1)

            # Get next token's logits using KV cache
            # Only pass the last token to the model since we have KV cache
            # The model will use KV cache for previous tokens and only compute
            # attention for the new token
            logits, kv_cache = self(token_id, kv_cache)

            # If sequence length exceeds block size, we need to update KV cache
            # by recomputing it for the last block_size tokens
            if idx.shape[1] > self.block_size:
                idx_cond = idx[:, -self.block_size:]
                logits, kv_cache = self(idx_cond)

        return idx

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
                    new_seq = torch.cat(
                        [seq.squeeze(-1),
                         token_id.unsqueeze(0)],
                        dim=-1).unsqueeze(-1)  # (B, T+1)
                    new_score = score + log_prob.item(
                    )  # Sum of log-probabilities
                    candidates.append((new_seq, new_score))

            ordered = sorted(candidates, key=lambda tup: tup[1], reverse=True)
            beams = ordered[:beam_size]

        best_seq = max(beams, key=lambda tup: tup[1])[0]
        return best_seq[:, 0]


if __name__ == "__main__":
    with open("../configs/model_config.yaml", "r") as mcf:
        model_configs = yaml.safe_load(mcf)

    model_configs[
        "vocab_size"] = 71  # simulate harry_potter_text data vocab_size
    ip = torch.randint(0, 10, (2, 32)).to("cuda")
    ip = ip.to(torch.int64)

    msa = NanoGPT(model_configs).to("cuda")
    op = msa.generate_greedy(ip)
    a = 1

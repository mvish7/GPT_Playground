import torch
import torch.nn as nn
from setuptools.command.setopt import config_file

from model.gpt_blocks import TransformerBlock


class MultiTokenPred(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.configs = configs
        self.lm_head0 = nn.Linear(configs["emb_dim"], configs["vocab_size"])

        self.rms_norm = nn.RMSNorm(configs["emb_dim"])

        self.projections_layers = nn.ModuleList([
            nn.Linear(2*configs["emb_dim"], configs["emb_dim"]) for _ in range(configs["multi_token_pred"]["num_tx_head"])
        ])

        self.tx_layers = nn.ModuleList([
            TransformerBlock(configs) for _ in range(configs["multi_token_pred"]["num_tx_head"])
        ])

    def forward(self, init_hidden, token_emb):
        B, T, _ = token_emb.shape

        outputs = []
        max_i = T - self.configs["multi_token_pred"]["total_num_token"] - 1
        for i in range(0, max_i+1):
            h_prev = init_hidden[:, i, :]

            logits_k = []
            # obtain all predicted tokens for this i by iterating over all token prediction heads
            for k in range(self.configs["multi_token_pred"]["num_tx_head"]):
                future_pos = i + (k+1)
                future_token_emb = token_emb[:, future_pos, :]

                norm_init_hidden = self.rms_norm(h_prev)
                norm_future_emb =self.rms_norm(future_token_emb)

                merged_emb = torch.cat([norm_init_hidden, norm_future_emb], dim=-1)

                projected_emb = self.projections_layers[k](merged_emb)

                curr_token_emb = self.tx_layers[k](projected_emb.unsqueeze(0)).squeeze(0)

                logits_k.append(self.lm_head0(curr_token_emb))

                h_prev = curr_token_emb

            logits_k = torch.stack(logits_k, dim=1)
            outputs.append(logits_k)

        out = torch.stack(outputs, dim=0)
        out = out.permute(1, 0, 2, 3).contiguous()
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expect(nn.Module):
    """
    An expert is actually an MLP
    """

    def __init__(self, emb_dim):
        super().__init__()

        self.net = nn.Sequential(nn.Linear(emb_dim, 4* emb_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(4 * emb_dim, emb_dim),
                                 nn.Dropout(p=0.5))

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):

    """
    implements a Router based on top-k expert selection
    """

    def __init__(self, emb_dim, num_experts, num_topk):
        super().__init__()
        # first, lets convert the input of shape (B, T, emb) into (B, T, num_experts)
        self.topk = num_topk
        self.router_linear = nn.Linear(emb_dim, num_experts)


    def forward(self, x):
        logits = self.router_linear(x)
        # now logits are in shape (B, T, num_experts)
        # let's find topk logits (simply top k biggest logits)
        topk_logits, topk_indices = logits.topk(self.topk, dim=-1)
        # create a placeholder with all -inf, so their softmax output will be 0
        sparse_logits = torch.full_like(logits, float('-inf'))
        # update the placeholders with topk logits
        sparse_logits = sparse_logits.scatter(-1, topk_indices, topk_logits)
        # convert logits into prob
        router_scores = F.softmax(sparse_logits, dim=-1)

        return router_scores, topk_indices


class NoisyRouter(nn.Module):
    """
    implements a load balancing scheme as per https://arxiv.org/pdf/1701.06538

    this is simply same router block as above with an addition of trainable gaussian noise helping the model to
    choose different experts
    """

    def __init__(self, emb_dim, num_experts, num_topk):
        super().__init__()
        self.topk = num_topk
        # linear layer to convert the input of shape (B, T, emb) into (B, T, num_experts)
        self.router_linear = nn.Linear(emb_dim, num_experts)
        # linear layer to convert the input of shape (B, T, emb) into (B, T, num_experts) for trainable noise vector
        self.noisy_linear = nn.Linear(emb_dim, num_experts)

    def forward(self, x):
        # (B, T, emb_dim) to (B, T, num_experts)
        logits = self.router_linear(x)
        noisy_vec = self.noisy_linear(x)

        # gaussian noise
        # todo: investigate why F.softplus(noisy_logits) was done in other implementations?
        noise = torch.randn_like(logits) * noisy_vec
        noisy_logits = logits + noise

        # from this step onwards, exact same implementation of simple router
        topk_logits, topk_indices = logits.topk(self.topk, dim=-1)
        sparse_logits = torch.full_like(logits, float('-inf'))
        sparse_logits = sparse_logits.scatter(-1, topk_indices, topk_logits)
        router_scores = F.softmax(sparse_logits, dim=-1)

        return router_scores, topk_indices


class SparseMoe(nn.Module):
    """
    implements a complete MoE layer i.e Experts + Router
    """

    def __init__(self, num_experts, num_topk, emb_dim):
        super().__init__()
        self.num_experts = num_experts
        self.router = NoisyRouter(emb_dim, num_experts, num_topk)
        self.experts = nn.ModuleList([Expect(emb_dim) for _ in range(num_experts)])

    def forward(self, x):

        final_output = torch.zeros_like(x)
        # feed the inputs to router
        routing_scores, routing_indices = self.router(x)
        # reshaping for easy calculations
        x = x.view(-1, x.size(-1))
        flat_indices = routing_scores.view(-1, routing_scores.size(-1))

        # routing indices tell us which expert is selected for which tokens.
        # feed the applicable tokens to each router

        for expert_id in range(self.num_experts):
            expert_mask_original = (routing_indices == expert_id).any(dim=-1)
            expert_mask = expert_mask_original.view(-1)

            if expert_mask.any():
                expert_ip = x[expert_mask]
                expert_op = self.experts[expert_id](expert_ip)

                # use the expert mask to find its routing scores
                curr_router_scores = flat_indices[expert_mask, expert_id].unsqueeze(1)
                weighted_output = expert_op * curr_router_scores

                final_output[expert_mask_original] += weighted_output.squeeze(1)

        return final_output

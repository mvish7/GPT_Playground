import torch.nn.functional as F

def single_token_pred_loss(logits, gt):
    B, T, C = logits.shape

    logits = logits.view(B*T, C)
    gt = gt.view(B*T)
    loss = F.cross_entropy(logits, gt)
    return loss

def multi_token_pred_loss(logits, gt):

    B, L, D, V = logits.shape  # batch_size, len_tokens, num_tokens, vocab_size
    _, T = gt.shape

    assert L == T - D  # check that enough gt is available

    loss = 0
    for i in range(L):
        for k in range(D):
            # for current i, fetch the logits predicted for all next tokens and apply loss for one single token at
            # a time
            logits_ik = logits[:, i, k, :]
            gt_ik = gt[:, i+(k+1)]
            gt_ik += F.cross_entropy(logits_ik, gt_ik)
    # scale the loss wrt. length(of tokens used for loss calc) and depth of each token_pred
    loss = loss / (L * D)
    return loss
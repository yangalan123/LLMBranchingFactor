import torch


def eta_truncation_logit_processor(token_ids, logits, eta=0.1):
    probs = torch.softmax(logits, dim=-1)
    prob_mask = probs < eta
    logits = logits.masked_fill_(prob_mask, float('-inf'))
    return logits

import torch
import torch.nn.functional as F

def mse_loss(target, out):
    return torch.mean((out - target) ** 2)

def sce_loss(target, out):
    cosine_similarity = F.cosine_similarity(target, out, dim=-1)
    loss = 1 - cosine_similarity
    return loss.mean()
"""Experimental drop-in adapter for raw-window to sampled-point mapping only."""
import torch
from .hybrid_nearest import hybrid_nearest

def remapping_knn(refs, queries, k=1):
    if k != 1:
        raise ValueError("The remapping adapter supports k=1 only")
    indices, _ = hybrid_nearest(refs, queries, workers=10)
    return torch.stack((torch.arange(len(queries), device=queries.device), indices))

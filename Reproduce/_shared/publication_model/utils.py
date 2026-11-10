import torch
import torch.nn as nn
import math
from torch.optim.lr_scheduler import _LRScheduler

# -------------------------
# Neighborhood utilities
# -------------------------
def build_knn_mask(
    coords_tensor,
    k=8,
    include_self=True,
    *,
    return_weights=False,
    symmetric=True,
    sigma=None,
):
    coords_tensor = torch.as_tensor(coords_tensor, dtype=torch.float32)
    if coords_tensor.dim() != 2:
        raise ValueError("`coords_tensor` must have shape (N, D)")
    n = coords_tensor.size(0)
    if n == 0:
        raise ValueError("No coordinates provided.")
    device = coords_tensor.device
    weights = torch.zeros(n, n, dtype=torch.float32, device=device)
    if k <= 0:
        mask = torch.zeros(n, n, dtype=torch.bool, device=device)
        if include_self:
            eye = torch.eye(n, dtype=torch.bool, device=device)
            mask |= eye
            weights = weights.masked_fill(eye, 1.0)
        return (mask, weights) if return_weights else mask

    k_eff = min(k + 1, n)
    dists = torch.cdist(coords_tensor, coords_tensor)
    knn_dists, knn = torch.topk(dists, k=k_eff, largest=False)
    knn = knn[:, 1:]
    knn_dists = knn_dists[:, 1:]

    mask = torch.zeros(n, n, dtype=torch.bool, device=device)
    row_idx = torch.arange(n, device=device).unsqueeze(1).expand_as(knn)
    mask[row_idx, knn] = True

    positive_dists = knn_dists[knn_dists > 0]
    if sigma is None:
        if positive_dists.numel() == 0:
            sigma_t = torch.tensor(1.0, dtype=torch.float32, device=device)
        else:
            sigma_t = positive_dists.median()
    else:
        sigma_t = torch.tensor(float(sigma), dtype=torch.float32, device=device)
    sigma_t = sigma_t.clamp_min(torch.finfo(torch.float32).eps)
    weights[row_idx, knn] = torch.exp(-(knn_dists ** 2) / (2.0 * sigma_t ** 2))

    if symmetric:
        mask = mask | mask.T
        weights = torch.maximum(weights, weights.T)

    if include_self:
        eye = torch.eye(n, dtype=torch.bool, device=device)
        mask |= eye
        weights = weights.masked_fill(eye, 1.0)

    if not include_self:
        weights = weights * mask.to(dtype=torch.float32)

    return (mask, weights) if return_weights else mask

# ----------------------------------------------------------------------
# Missing utils from CoMM package implementations
# ----------------------------------------------------------------------

def set_weight_decay_per_param(model, weight_decay):
    """
    Set weight decay for parameters. 
    Typically we don't apply weight decay to bias and LayerNorm parameters.
    """
    p_wd, p_non_wd = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or 'bias' in name or 'ln' in name or 'bn' in name:
            p_non_wd.append(param)
        else:
            p_wd.append(param)
    
    optim_params = [
        {"params": p_wd, "weight_decay": weight_decay},
        {"params": p_non_wd, "weight_decay": 0.0},
    ]
    return optim_params


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr=0.0,
        eta_min=0.0,
        last_epoch=-1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr
                + (base_lr - self.warmup_start_lr)
                * self.last_epoch
                / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch < self.max_epochs:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.last_epoch - self.warmup_epochs)
                        / (self.max_epochs - self.warmup_epochs)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]
        else:
            return [self.eta_min for _ in self.base_lrs]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    
    For single GPU/CPU: just returns the tensor or list of tensors.
    """
    if isinstance(tensors, list):
        return [all_gather_batch_with_grad(t) for t in tensors]
        
    # Check if distributed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # Implementation for DDP gathering with grad
        tensors_gather = [torch.ones_like(tensors) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, tensors, async_op=False)
        return torch.cat(tensors_gather, dim=0)
        
    return tensors

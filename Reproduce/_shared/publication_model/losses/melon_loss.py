import torch.nn.functional as func
import torch
import torch.nn as nn
from ..utils import all_gather_batch_with_grad


class MELONLoss(nn.Module):
    """
        Normalized Temperature Cross-Entropy Loss for Multi-Modal Contrastive Learning as defined in MELON [1]

        [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self, temperature=0.1, weights=None, learnable_weights=False, n_losses=None):
        super().__init__()
        self.temperature = temperature
        self.weights = weights
        self.learnable_weights = learnable_weights
        self.INF = 1e8
        if learnable_weights:
            if n_losses is None:
                raise ValueError("n_losses must be provided when learnable_weights=True")
            # Kendall et al. (2018) homoscedastic-uncertainty weighting. One learnable
            # log-variance s_i per loss term; effective loss = exp(-s_i)*L_i + 0.5*s_i.
            # exp(-s_i) is the modality's precision (weight); the +0.5*s_i regulariser
            # makes down-weighting cost something, preventing collapse-to-zero. A
            # discordant modality (high residual InfoNCE loss) learns a large s_i and is
            # automatically down-weighted instead of dragging the shared embedding.
            self.log_vars = nn.Parameter(torch.zeros(int(n_losses)))

    def infonce(self, z1, z2, negative_exclusion_mask=None):
        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j)
        eye = torch.eye(N, dtype=torch.bool, device=z1.device)
        excluded = eye
        if negative_exclusion_mask is not None:
            negative_exclusion_mask = negative_exclusion_mask.to(
                device=z1.device, dtype=torch.bool
            )
            if negative_exclusion_mask.shape != (N, N):
                raise ValueError("negative_exclusion_mask must match the gathered batch")
            excluded = excluded | negative_exclusion_mask
            sim_zij = sim_zij.masked_fill(negative_exclusion_mask & ~eye, -self.INF)
        # Remove self-pairs and optional false negatives from the denominator.
        sim_zii = sim_zii.masked_fill(excluded, -self.INF)
        sim_zjj = sim_zjj.masked_fill(excluded, -self.INF)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N
        return loss, acc

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - "aug1_embed", List of tensors with shape (bsize, feature_dim), 1st aug.
                - "aug2_embed", List of tensors with shape (bsize, feature_dim), 2nd aug.
                - "prototype", integer indicating where the multimodal representation Z 
                    is stored in "aug1_embed" and "aug2_embed".
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        # Prepare embeddings (normalize + gather across all GPU)
        z1, z2, prototype = outputs["aug1_embed"], outputs["aug2_embed"], outputs["prototype"]
        negative_exclusion_mask = outputs.get("negative_exclusion_mask")
        assert len(z1) == len(z2)
        n_emb = len(z1)
        z1 = [func.normalize(z, p=2, dim=-1) for z in z1]
        z2 = [func.normalize(z, p=2, dim=-1) for z in z2]
        Z = all_gather_batch_with_grad(z1 + z2)
        z1, z2 = Z[:n_emb], Z[n_emb:]

        # Apply InfoNCE between a "prototype embedding" and all the others
        loss = []
        acc = []
        for i in range(n_emb):
            loss1, acc1 = self.infonce(
                z1[i], z2[prototype], negative_exclusion_mask
            )
            loss2, acc2 = self.infonce(
                z2[i], z1[prototype], negative_exclusion_mask
            )
            loss.append((loss1 + loss2) / 2.)
            acc.append((acc1 + acc2) / 2.)
        ssl_acc = {"ssl_acc_%i"%i: acc_ for i, acc_ in enumerate(acc)}
        losses = {"ssl_loss_%i"%i: l for i, l in enumerate(loss)}
        if self.learnable_weights:
            stacked = torch.stack(loss)
            s = self.log_vars.to(stacked.device)
            loss = torch.mean(torch.exp(-s) * stacked + 0.5 * s)
        elif self.weights is not None:
            loss = torch.mean(torch.stack(loss) * torch.tensor(self.weights, device=z1[0].device))
        else:
            loss = torch.mean(torch.stack(loss))
        acc = torch.mean(torch.stack(acc))
        return {"loss": loss, "ssl_acc": acc, **ssl_acc, **losses}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)

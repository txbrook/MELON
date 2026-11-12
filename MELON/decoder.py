"""MELON shared-factor decoder and gene-peak activity."""
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_val_split(index: pd.Index, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(index))
    rng.shuffle(order)
    n_val = max(1, int(round(len(index) * val_frac)))
    val_idx = np.sort(order[:n_val])
    train_idx = np.sort(order[n_val:])
    return train_idx, val_idx


def specificity_weights(prevalence: pd.Series, power: float) -> np.ndarray:
    if power <= 0:
        return np.ones(len(prevalence), dtype=np.float32)
    weights = np.clip(1.0 - prevalence.to_numpy(dtype=np.float32, copy=False), 0.05, 1.0)
    return np.power(weights, float(power), dtype=np.float32)


@dataclass
class DecoderOutputs:
    h: np.ndarray
    rna_loadings: np.ndarray
    atac_loadings: np.ndarray
    rna_bias: np.ndarray
    atac_bias: np.ndarray


class SharedFactorDecoder(nn.Module):
    def __init__(self, z_dim: int, n_factors: int, n_rna: int, n_atac: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_factors),
        )
        self.rna_weight_raw = nn.Parameter(torch.randn(n_rna, n_factors) * 0.02)
        self.atac_weight_raw = nn.Parameter(torch.randn(n_atac, n_factors) * 0.02)
        self.rna_bias = nn.Parameter(torch.zeros(n_rna))
        self.atac_bias = nn.Parameter(torch.zeros(n_atac))

    def loadings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return F.softplus(self.rna_weight_raw), F.softplus(self.atac_weight_raw)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.encoder(z))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(z)
        rna_w, atac_w = self.loadings()
        rna_pred = F.softplus(h @ rna_w.T + self.rna_bias)
        atac_logits = h @ atac_w.T + self.atac_bias
        return h, rna_pred, atac_logits


def factor_decorrelation_penalty(h: torch.Tensor) -> torch.Tensor:
    if h.shape[1] <= 1:
        return h.new_tensor(0.0)
    centered = h - h.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(h.shape[0] - 1, 1)
    std = torch.sqrt(torch.diag(cov).clamp_min(1e-6))
    corr = cov / (std[:, None] * std[None, :] + 1e-6)
    eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
    off_diag = corr - eye
    return off_diag.pow(2).mean()


def make_dataloader(z: np.ndarray, rna: np.ndarray, atac: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(z).float(), torch.from_numpy(rna).float(), torch.from_numpy(atac).float())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def compute_atac_pos_weight(atac_train: np.ndarray) -> torch.Tensor:
    pos = atac_train.sum(axis=0, dtype=np.float64)
    neg = atac_train.shape[0] - pos
    pos = np.maximum(pos, 1.0)
    weight = np.clip(neg / pos, 1.0, 50.0)
    return torch.from_numpy(weight.astype(np.float32))


def evaluate(model, loader, pos_weight, atac_weight, l1_weight, activity_weight, factor_decorrelation_weight, device):
    model.eval()
    total = 0
    metrics = {"loss": 0.0, "rna_mse": 0.0, "atac_bce": 0.0}
    with torch.no_grad():
        for z_batch, rna_batch, atac_batch in loader:
            z_batch, rna_batch, atac_batch = z_batch.to(device), rna_batch.to(device), atac_batch.to(device)
            h, rna_pred, atac_logits = model(z_batch)
            rna_loss = F.mse_loss(rna_pred, rna_batch)
            atac_loss = F.binary_cross_entropy_with_logits(atac_logits, atac_batch, pos_weight=pos_weight.to(device))
            rna_w, atac_w = model.loadings()
            l1_penalty = rna_w.mean() + atac_w.mean()
            activity_penalty = h.mean()
            decorrelation_penalty = factor_decorrelation_penalty(h)
            loss = rna_loss + atac_weight * atac_loss + l1_weight * l1_penalty + activity_weight * activity_penalty + factor_decorrelation_weight * decorrelation_penalty
            n = z_batch.shape[0]
            total += n
            metrics["loss"] += float(loss) * n
            metrics["rna_mse"] += float(rna_loss) * n
            metrics["atac_bce"] += float(atac_loss) * n
    return {k: v / max(total, 1) for k, v in metrics.items()}


def fit_decoder(z, rna, atac, *, n_factors, hidden_dim, batch_size, epochs, lr, weight_decay, atac_weight, l1_weight, activity_weight, factor_decorrelation_weight, val_frac, seed):
    device = torch.device("cpu")
    train_idx, val_idx = train_val_split(pd.Index(range(len(z))), val_frac, seed)
    train_loader = make_dataloader(z[train_idx], rna[train_idx], atac[train_idx], batch_size, True)
    val_loader = make_dataloader(z[val_idx], rna[val_idx], atac[val_idx], batch_size, False)

    model = SharedFactorDecoder(z_dim=z.shape[1], n_factors=n_factors, n_rna=rna.shape[1], n_atac=atac.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = compute_atac_pos_weight(atac[train_idx])

    history_rows: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        for z_batch, rna_batch, atac_batch in train_loader:
            z_batch, rna_batch, atac_batch = z_batch.to(device), rna_batch.to(device), atac_batch.to(device)
            h, rna_pred, atac_logits = model(z_batch)
            rna_loss = F.mse_loss(rna_pred, rna_batch)
            atac_loss = F.binary_cross_entropy_with_logits(atac_logits, atac_batch, pos_weight=pos_weight.to(device))
            rna_w, atac_w = model.loadings()
            l1_penalty = rna_w.mean() + atac_w.mean()
            activity_penalty = h.mean()
            decorrelation_penalty = factor_decorrelation_penalty(h)
            loss = rna_loss + atac_weight * atac_loss + l1_weight * l1_penalty + activity_weight * activity_penalty + factor_decorrelation_weight * decorrelation_penalty
            opt.zero_grad()
            loss.backward()
            opt.step()

        train_metrics = evaluate(model, train_loader, pos_weight, atac_weight, l1_weight, activity_weight, factor_decorrelation_weight, device)
        val_metrics = evaluate(model, val_loader, pos_weight, atac_weight, l1_weight, activity_weight, factor_decorrelation_weight, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"], "train_rna_mse": train_metrics["rna_mse"], "train_atac_bce": train_metrics["atac_bce"],
            "val_loss": val_metrics["loss"], "val_rna_mse": val_metrics["rna_mse"], "val_atac_bce": val_metrics["atac_bce"],
        }
        history_rows.append(row)
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 10) == 0:
            print(f"epoch={epoch:03d} train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} train_rna={row['train_rna_mse']:.4f} train_atac={row['train_atac_bce']:.4f}")

    return model, pd.DataFrame(history_rows), (train_idx, val_idx)


def extract_decoder_outputs(model: SharedFactorDecoder, z: np.ndarray) -> DecoderOutputs:
    model.eval()
    with torch.no_grad():
        z_t = torch.from_numpy(z).float()
        h = model.encode(z_t).cpu().numpy()
        rna_w, atac_w = model.loadings()
        return DecoderOutputs(h=h, rna_loadings=rna_w.cpu().numpy(), atac_loadings=atac_w.cpu().numpy(), rna_bias=model.rna_bias.detach().cpu().numpy(), atac_bias=model.atac_bias.detach().cpu().numpy())


def row_profile_matrix(loadings):
    centered = loadings - loadings.mean(axis=1, keepdims=True)
    denom = loadings.std(axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    centered = centered / denom
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    profiles = centered / norm
    strength = np.linalg.norm(loadings, axis=1)
    return profiles.astype(np.float32, copy=False), strength.astype(np.float32, copy=False)


def positive_standardized_support(mat: np.ndarray) -> np.ndarray:
    mu = mat.mean(axis=0, keepdims=True)
    sigma = mat.std(axis=0, keepdims=True)
    sigma[sigma == 0.0] = 1.0
    z = (mat - mu) / sigma
    return np.clip(z, 0.0, None).astype(np.float32, copy=False)


def positive_binary_residual_support(mat: np.ndarray) -> np.ndarray:
    p = mat.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.clip(p * (1.0 - p), 1e-6, None))
    z = (mat - p) / denom
    return np.clip(z, 0.0, None).astype(np.float32, copy=False)


def compute_pair_activity(
    h: pd.DataFrame,
    rna_loadings: pd.DataFrame,
    atac_loadings: pd.DataFrame,
    gene_weights: pd.Series,
    peak_weights: pd.Series,
    gene_support_df: pd.DataFrame,
    peak_support_df: pd.DataFrame,
    gene: str,
    peak: str,
) -> pd.Series:
    adj_gene = rna_loadings.loc[gene].to_numpy(dtype=np.float32, copy=False) * float(gene_weights.loc[gene])
    adj_peak = atac_loadings.loc[peak].to_numpy(dtype=np.float32, copy=False) * float(peak_weights.loc[peak])
    factor_core = h.to_numpy(dtype=np.float32, copy=False) @ (adj_gene * adj_peak)
    gene_support = gene_support_df[gene].to_numpy(dtype=np.float32, copy=False)
    peak_support = peak_support_df[peak].to_numpy(dtype=np.float32, copy=False)
    score = factor_core * gene_support * peak_support
    return pd.Series(score, index=h.index, name=f"{gene}|{peak}")

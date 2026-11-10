"""Publication model construction and exact, query-chunked embedding export."""
from contextlib import contextmanager
from types import MethodType
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .publication_model.data import SpatialOmicsAugmentations
from .publication_model.model import SpotTokenAdapter
from .publication_model.models.mmfusion import MMFusion, GroupedQueryAttention
from .publication_model.pl_modules.melon import MELONModel
from .publication_model.losses.melon_loss import MELONLoss
from .publication_model.utils import build_knn_mask


class PaperAugmentations(SpatialOmicsAugmentations):
    @staticmethod
    def neighbor_mix(x, neighbor_mask=None, alpha=0.3):
        # Table 9: one permutation and mixing coefficient shared by modalities.
        permutation = np.random.permutation(len(x[0]))
        coefficient = np.random.beta(alpha, alpha)
        return [(coefficient * a + (1 - coefficient) * a[permutation]).astype(np.float32) for a in x]


class HistoricalLoss(MELONLoss):
    def infonce(self, z1, z2, negative_exclusion_mask=None):
        if negative_exclusion_mask is not None:
            raise ValueError("The recorded five-modality objective has no negative exclusion")
        n = len(z1)
        target = torch.arange(n, device=z1.device)
        eye = torch.eye(n, dtype=torch.bool, device=z1.device)
        cross = z1 @ z2.T / self.temperature
        same1 = (z1 @ z1.T / self.temperature).masked_fill(eye, -self.INF)
        same2 = (z2 @ z2.T / self.temperature).masked_fill(eye, -self.INF)
        loss = .5 * (F.cross_entropy(torch.cat([cross, same1], 1), target)
                     + F.cross_entropy(torch.cat([cross.T, same2], 1), target))
        accuracy = (cross.argmax(1) == target).float().mean() * 100
        return loss, accuracy


def make_model(dimensions, config):
    embed_dim = config.get("embed_dim", 128)
    spatial_k = config.get("training_spatial_k", 16)
    encoder = MMFusion(
        encoders=[nn.Sequential(nn.Linear(d, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)) for d in dimensions],
        input_adapters=[SpotTokenAdapter() for _ in dimensions],
        embed_dim=embed_dim, fusion="concat", pool="tokens", n_heads=8, n_layers=1,
    )
    model = MELONModel(
        encoder=encoder, projection=MELONModel._build_mlp(embed_dim, 256, embed_dim),
        optim_kwargs={"lr": config.get("learning_rate", .001), "weight_decay": config.get("weight_decay", 1e-5)},
        loss_kwargs={"temperature": config.get("temperature", .1)},
        neighborhood_cfg={"hidden_dim": 64, "heads": 8, "dropout": .1,
                          "k_neighbors": spatial_k, "include_self": True},
    )
    if config["historical_five_modal"]:
        model.loss = HistoricalLoss(temperature=config.get("temperature", .1))
    else:
        # Supplementary Table 9 specifies Adam; the archived five-modal run used AdamW.
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), **self.optim_kwargs, betas=(.9, .999))
        model.configure_optimizers = MethodType(configure_optimizers, model)
    return model


@contextmanager
def export_attention(encoder, bias, spots, chunk_size):
    """Slice queries only. Every query still attends to every key and value."""
    originals = []
    for module in encoder.modules():
        if not isinstance(module, GroupedQueryAttention):
            continue
        original = module.forward
        def forward(self, query, key, value, key_padding_mask=None, attn_mask=None,
                    need_weights=False, _original=original):
            if self.training or not self.batch_first or need_weights:
                raise ValueError("Chunked export requires eval mode and batch-first attention")
            if key_padding_mask is not None and key_padding_mask.any():
                raise ValueError("Unexpected padded tokens in aligned multimodal export")
            keys = torch.arange(key.shape[1], device=query.device) % spots
            outputs = []
            for start in range(0, query.shape[1], chunk_size):
                q = query[:, start:start + chunk_size]
                queries = torch.arange(start, start + q.shape[1], device=query.device) % spots
                mask = None if bias is None else bias[queries[:, None], keys[None, :]]
                outputs.append(_original(q, key, value, key_padding_mask=None, attn_mask=mask, need_weights=False)[0])
            return torch.cat(outputs, dim=1), None
        originals.append((module, original))
        module.forward = MethodType(forward, module)
    try:
        yield
    finally:
        for module, original in originals:
            module.forward = original


def export_embedding(model, tensors, coords, spatial_k, query_chunk=128):
    model.eval()
    with torch.inference_mode():
        bias = None
        if spatial_k is not None:
            mask, weight = build_knn_mask(coords, k=spatial_k, include_self=True, return_weights=True)
            bias = model.neighborhood_attn(torch.cat(tensors, dim=1), coords, mask, weight)
        with export_attention(model.encoder, bias, len(coords), query_chunk):
            embedding = model.encoder(tensors)
    return embedding.cpu().numpy()

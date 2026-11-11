"""Export MELON embeddings with bounded attention memory."""
from contextlib import contextmanager
from types import MethodType
import torch
from .models.mmfusion import GroupedQueryAttention
from .utils import build_knn_mask


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
            mask, weight = build_knn_mask(coords, k=spatial_k, include_self=True, return_weights=True, symmetric=True)
            bias = model.neighborhood_attn(torch.cat(tensors, dim=1), coords, mask, weight)
        with export_attention(model.encoder, bias, len(coords), query_chunk):
            embedding = model.encoder(tensors)
    return embedding.cpu().numpy()

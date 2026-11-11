import numpy as np
import pytest
import torch

from melon_pkg.data import DualAugDataModule, SpatialDataset, SpatialOmicsAugmentations, collate_with_dual_augment
from melon_pkg.model import make_model
from melon_pkg.utils import build_knn_mask


@pytest.mark.parametrize("modalities", [2, 3, 5])
def test_all_modalities_reach_model_and_receive_gradients(modalities):
    torch.manual_seed(12)
    tensors = [torch.randn(6, i + 3) for i in range(modalities)]
    coords = torch.randn(6, 2)
    mask, weight = build_knn_mask(coords, k=2, return_weights=True, symmetric=True)
    data = DualAugDataModule(mods_raw=tensors, coords=coords, neighbor_mask=mask,
                             neighbor_weight=weight, batch_size=6, spot_tokenization=True)
    view_a, view_b, meta = next(iter(data.train_dataloader()))
    assert len(view_a) == len(view_b) == modalities
    assert meta["attn_mask_tokens"] == 6 * modalities
    model = make_model([x.shape[1] for x in tensors],
                       {"embed_dim": 8, "historical_five_modal": modalities == 5})
    output = model(view_a, view_b, meta)
    assert len(output["aug1_embed"]) == modalities + 1
    loss = model.loss(output)["loss"]
    loss.backward()
    assert torch.isfinite(loss)
    for encoder in model.encoder.encoders:
        assert encoder[0].weight.grad is not None
        assert torch.isfinite(encoder[0].weight.grad).all()


def test_legacy_positional_data_and_custom_augmentation():
    x, y, coords = torch.randn(5, 3), torch.randn(5, 2), torch.randn(5, 2)
    data = DualAugDataModule(x, y, coords, None, 5, lambda xs: [a + 1 for a in xs])
    a, b, meta = next(iter(data.train_dataloader()))
    torch.testing.assert_close(a[0], x + 1)
    torch.testing.assert_close(a[1], y + 1)
    torch.testing.assert_close(b[0], x)
    assert "weight" not in meta
    assert len(collate_with_dual_augment([data.ds[0], data.ds[1]])) == 3


def test_weighted_graph_is_explicit_and_symmetric():
    coords = torch.tensor([[0., 0.], [1., 0.], [4., 0.]])
    legacy = build_knn_mask(coords, k=1)
    assert legacy.dtype == torch.bool
    assert not torch.equal(legacy, legacy.T)
    mask, weights = build_knn_mask(coords, k=1, return_weights=True, symmetric=True)
    torch.testing.assert_close(mask, mask.T)
    torch.testing.assert_close(weights, weights.T)
    assert torch.equal(weights.diagonal(), torch.ones(3))
    assert torch.equal(weights > 0, mask)


def test_spatial_mix_only_uses_neighbors_and_preserves_modalities():
    x = np.arange(4, dtype=np.float32).reshape(-1, 1)
    mask = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=bool)
    augment = SpatialOmicsAugmentations(["neighbor_mix"], p=1, spatial_mix=True)
    a, b = augment([x, 2 * x + 7], neighbor_mask=mask)
    np.testing.assert_allclose(b, 2 * a + 7, atol=1e-5)
    assert (a[:2] >= 0).all() and (a[:2] <= 1).all()
    assert (a[2:] >= 2).all() and (a[2:] <= 3).all()


def test_misaligned_modalities_fail_before_training():
    with pytest.raises(ValueError, match="aligned"):
        SpatialDataset(mods_raw=[torch.ones(3, 2), torch.ones(4, 2)])
    with pytest.raises(ValueError, match="not both"):
        SpatialDataset(torch.ones(3, 2), torch.ones(3, 2), mods_raw=[torch.ones(3, 2)] * 2)

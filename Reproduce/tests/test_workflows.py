import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared.io import load_features, unique_index
from _shared.engine import make_model, export_embedding, PaperAugmentations
from _shared.publication_model.utils import build_knn_mask


def test_preprocessed_features_are_preserved(tmp_path):
    rng = np.random.default_rng(9)
    rna, protein = rng.normal(size=(6, 4)), rng.normal(size=(6, 3))
    coords = rng.normal(size=(6, 2))
    path = tmp_path / "features.npz"
    np.savez(path, barcodes=np.array(["b", "a", "c", "d", "e", "f"]),
             coords=coords, RNA=rna, Protein=protein)
    ids, xy, arrays = load_features(path, ["RNA", "Protein"])
    assert ids.tolist()[:2] == ["b", "a"]
    np.testing.assert_array_equal(arrays[0], rna.astype(np.float32))
    np.testing.assert_array_equal(arrays[1], protein.astype(np.float32))
    np.testing.assert_array_equal(xy, coords.astype(np.float32))


@pytest.mark.parametrize("ids", [["a", "a"], ["a", None], ["a", ""]])
def test_invalid_barcodes_are_rejected(ids):
    with pytest.raises(ValueError):
        unique_index(ids, "features")


def test_mismatched_modality_rows_are_rejected(tmp_path):
    path = tmp_path / "features.npz"
    np.savez(path, barcodes=["a", "b"], coords=np.zeros((2, 2)), RNA=np.ones((3, 4)))
    with pytest.raises(ValueError, match="Invalid features for RNA"):
        load_features(path, ["RNA"])


@pytest.mark.parametrize("modalities", [2, 3, 5])
def test_chunked_export_equals_full_attention(modalities):
    torch.manual_seed(7)
    config = {"historical_five_modal": modalities == 5}
    model = make_model([4] * modalities, config).eval()
    tensors = [torch.randn(9, 4) for _ in range(modalities)]
    coords = torch.randn(9, 2)
    mask, weight = build_knn_mask(coords, k=3, return_weights=True)
    with torch.inference_mode():
        bias = model.neighborhood_attn(torch.cat(tensors, 1), coords, mask, weight)
        expected = model.encoder(tensors, attn_mask=bias.repeat(modalities, modalities)).numpy()
    actual = export_embedding(model, tensors, coords, spatial_k=3, query_chunk=4)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    # The context manager must restore attention for the next full forward pass.
    with torch.inference_mode():
        restored = model.encoder(tensors, attn_mask=bias.repeat(modalities, modalities)).numpy()
    np.testing.assert_allclose(restored, expected)


def test_paper_mix_keeps_modalities_synchronized():
    x = np.arange(30, dtype=np.float32).reshape(10, 3)
    a, b = PaperAugmentations.neighbor_mix([x, 2 * x + 7])
    np.testing.assert_allclose(b, 2 * a + 7, atol=1e-5)
    assert not np.array_equal(a, x)



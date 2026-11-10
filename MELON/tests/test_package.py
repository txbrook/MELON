from pathlib import Path

import numpy as np
import pandas as pd

from melon_pkg import MELON, SpatialOmicsAugmentations, build_knn_mask


def test_public_exports():
    assert callable(MELON)
    assert callable(build_knn_mask)
    assert isinstance(SpatialOmicsAugmentations(augmentations=()), SpatialOmicsAugmentations)


def test_melon_smoke_run(tmp_path: Path):
    n_spots = 6
    barcodes = [f"spot_{i}" for i in range(n_spots)]
    mod1_df = pd.DataFrame(np.random.rand(n_spots, 5).astype("float32"), index=barcodes)
    mod2_df = pd.DataFrame(np.random.rand(n_spots, 3).astype("float32"), index=barcodes)
    coords = pd.DataFrame(
        {"x": np.arange(n_spots, dtype="float32"), "y": np.arange(n_spots, dtype="float32")},
        index=barcodes,
    )
    coords_path = tmp_path / "coords.csv"
    coords.to_csv(coords_path)

    model, mod1_t, mod2_t, coords_t, neighbor_mask = MELON(
        mod1_df,
        mod2_df,
        str(coords_path),
        out_dir=str(tmp_path / "out"),
        epochs=1,
        embed_dim=8,
        batch_size=3,
        neighbor_k=2,
        augmentations=(),
    )

    assert model.__class__.__name__ == "MELONModel"
    assert tuple(mod1_t.shape) == (n_spots, 5)
    assert tuple(mod2_t.shape) == (n_spots, 3)
    assert tuple(coords_t.shape) == (n_spots, 2)
    assert neighbor_mask is not None
    assert tuple(neighbor_mask.shape) == (n_spots, n_spots)


def test_melon_neighbor_mask_output(tmp_path: Path):
    n_spots = 6
    barcodes = [f"spot_{i}" for i in range(n_spots)]
    mod1_df = pd.DataFrame(np.random.rand(n_spots, 5).astype("float32"), index=barcodes)
    mod2_df = pd.DataFrame(np.random.rand(n_spots, 3).astype("float32"), index=barcodes)
    coords = pd.DataFrame(
        {"x": np.arange(n_spots, dtype="float32"), "y": np.arange(n_spots, dtype="float32")},
        index=barcodes,
    )
    coords_path = tmp_path / "coords.csv"
    coords.to_csv(coords_path)

    model, mod1_t, mod2_t, coords_t, neighbor_mask = MELON(
        mod1_df,
        mod2_df,
        str(coords_path),
        out_dir=str(tmp_path / "out_meta"),
        epochs=1,
        embed_dim=8,
        batch_size=3,
        neighbor_k=2,
        augmentations=(),
    )

    assert model.__class__.__name__ == "MELONModel"
    assert tuple(mod1_t.shape) == (n_spots, 5)
    assert tuple(mod2_t.shape) == (n_spots, 3)
    assert tuple(coords_t.shape) == (n_spots, 2)
    assert neighbor_mask is not None
    assert tuple(neighbor_mask.shape) == (n_spots, n_spots)

"""Read aligned, already-preprocessed modality features."""
import numpy as np
import pandas as pd


def unique_index(index, name):
    index = pd.Index(index)
    if index.isna().any():
        raise ValueError(f"{name}: barcodes must be unique and nonmissing")
    index = index.astype(str)
    if index.has_duplicates or (index == "").any():
        raise ValueError(f"{name}: barcodes must be unique and nonempty")
    return index


def load_features(path, modalities):
    with np.load(path, allow_pickle=False) as z:
        ids = unique_index(z["barcodes"], "features")
        xy = np.asarray(z["coords"], dtype=np.float32)
        xs = [np.asarray(z[m], dtype=np.float32) for m in modalities]
    if xy.shape != (len(ids), 2) or not np.isfinite(xy).all():
        raise ValueError("Invalid coordinates")
    if len(ids) < 2:
        raise ValueError("At least two spots are required")
    for m, x in zip(modalities, xs):
        if x.ndim != 2 or x.shape[0] != len(ids) or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError(f"Invalid features for {m}")
    return ids, xy, xs

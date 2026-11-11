"""Inspect assay input nodes without transforming their values."""
from pathlib import Path
import h5py
import pandas as pd


def assay_shape(path):
    path = Path(path)
    with h5py.File(path, "r") as f:
        x = f["X"]
        if isinstance(x, h5py.Dataset):
            shape = tuple(x.shape)
        else:
            shape = tuple(int(v) for v in x.attrs["shape"])
        if "spot" in f:
            n = len(f["spot"])
            if shape[0] != n:
                shape = shape[::-1]
        return shape


def align_partition(path, embedding, column="MELON"):
    frame = pd.read_csv(path, index_col=0, converters={0: str})
    frame.index = frame.index.astype(str)
    if not frame.index.is_unique or set(frame.index) != set(embedding.index):
        raise ValueError("Partition and embedding barcodes differ")
    labels = frame.loc[embedding.index, column]
    if labels.isna().any():
        raise ValueError("Missing domain assignments")
    return labels

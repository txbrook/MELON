import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

# --- Helper to load your specific H5 format ---
def load_custom_h5(path, feature_key='gene'):
    with h5py.File(path, 'r') as f:
        X = np.array(f['X'])
        if 'spot' not in f:
            raise ValueError(f"{path} must contain dataset 'spot'.")
        spots = np.array(f['spot']).astype(str)
        features = np.array(f[feature_key]).astype(str)
        
        # Decode bytes if necessary
        if isinstance(spots[0], (bytes, np.bytes_)):
            spots = np.char.decode(spots, 'utf-8')
        if isinstance(features[0], (bytes, np.bytes_)):
            features = np.char.decode(features, 'utf-8')
            
        # Transpose if needed (spots should be rows)
        if X.shape[1] == len(spots):
            X = X.T
            
        return pd.DataFrame(X, index=spots, columns=features)

# --- Normalization (Manual Implementation to avoid Scanpy issues) ---
def normalize_modality(df, label="Modality"):
    print(f"Normalizing {label}...")
    
    # 1. Calculate total counts per spot
    total_counts = df.sum(axis=1)
    print(f"  - Mean counts before: {total_counts.mean():.2f}")
    
    # 2. Normalize to 10,000 reads per spot
    target_sum = 1e4
    # Avoid division by zero
    norm_factor = total_counts.replace(0, 1) 
    df_norm = df.div(norm_factor, axis=0) * target_sum
    
    # 3. Log transform (log(x+1))
    df_norm = np.log1p(df_norm)
    
    # Check new mean (approximate since log changes scale)
    print(f"  - Normalization complete.")
    return df_norm


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

#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Guard against OpenMP shared-memory issues before importing torch.
os.environ["KMP_USE_SHM"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


REPO_ROOT = Path(__file__).resolve().parent

from melon_pkg.api import MELON


def to_numpy(mat):
    if hasattr(mat, "toarray"):
        return mat.toarray()
    if hasattr(mat, "A"):
        return mat.A
    return np.asarray(mat)


def normalize_df(df: pd.DataFrame, label: str, target_sum: float = 1e4) -> pd.DataFrame:
    print(f"Normalizing {label}...")
    totals = df.sum(axis=1).astype(np.float64)
    print(f"  - Mean counts before: {totals.mean():.2f}")
    denom = totals.replace(0, 1.0)
    out = df.div(denom, axis=0) * float(target_sum)
    out = np.log1p(out).astype(np.float32, copy=False)
    print("  - Normalization complete.")
    return out


def tfidf_lsi_modality(
    df: pd.DataFrame,
    *,
    label: str,
    n_components: int,
    seed: int,
    drop_first: bool = False,
    standardize: bool = False,
) -> pd.DataFrame:
    print(f"Applying TF-IDF + LSI to {label}...")
    if n_components < 1:
        raise ValueError("atac_lsi_components must be >= 1")

    X = df.to_numpy(dtype=np.float32, copy=False)
    if np.isnan(X).any():
        raise ValueError(f"{label} contains NaN values; cannot run TF-IDF/LSI.")
    if (X < 0).any():
        raise ValueError(f"{label} contains negative values; TF-IDF/LSI expects non-negative counts.")

    X = sp.csr_matrix(X)
    n_cells, n_features = X.shape
    if n_cells < 2 or n_features < 2:
        raise ValueError(f"{label} shape {X.shape} is too small for LSI.")

    # TF: per-cell normalization
    cell_sum = np.asarray(X.sum(axis=1)).ravel().astype(np.float64, copy=False)
    cell_sum[cell_sum == 0.0] = 1.0
    tf = X.multiply(1.0 / cell_sum[:, None])

    # IDF: down-weight ubiquitous peaks/features
    feature_sum = np.asarray(X.sum(axis=0)).ravel().astype(np.float64, copy=False)
    idf = np.log1p(n_cells / (1.0 + feature_sum))
    tfidf = tf.multiply(idf)

    max_components = min(n_components, n_cells - 1, n_features - 1)
    if max_components < 1:
        raise ValueError(f"Cannot compute LSI with shape {X.shape}.")
    if max_components != n_components:
        print(f"  - Requested {n_components} LSI components; using {max_components} due to matrix shape.")

    try:
        from sklearn.decomposition import TruncatedSVD
    except Exception as exc:
        raise ImportError("scikit-learn is required for --atac_preprocess tfidf_lsi.") from exc

    svd = TruncatedSVD(n_components=max_components, random_state=int(seed))
    lsi = svd.fit_transform(tfidf).astype(np.float32, copy=False)

    start_idx = 0
    if drop_first and lsi.shape[1] > 1:
        lsi = lsi[:, 1:]
        start_idx = 1
        print("  - Dropped first LSI component.")

    if standardize and lsi.shape[1] > 0:
        mu = lsi.mean(axis=0, keepdims=True)
        sigma = lsi.std(axis=0, keepdims=True)
        sigma[sigma == 0] = 1.0
        lsi = ((lsi - mu) / sigma).astype(np.float32, copy=False)
        print("  - Standardized LSI components (z-score).")

    explained = float(np.sum(svd.explained_variance_ratio_))
    cols = [f"lsi_{i + 1 + start_idx}" for i in range(lsi.shape[1])]
    out = pd.DataFrame(lsi, index=df.index, columns=cols)
    print(f"  - LSI complete: {out.shape[1]} components, cumulative explained variance={explained:.4f}")
    return out


def decode_str_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype.kind == "S":
        return np.char.decode(arr, "utf-8")
    return arr.astype(str)


def save_model_artifacts(
    output_dir: str,
    model: torch.nn.Module,
    args: argparse.Namespace,
    mod1_shape: tuple[int, int],
    mod2_shape: tuple[int, int],
    coords_shape: tuple[int, int],
    neighbor_mask_present: bool,
) -> None:
    artifacts_dir = Path(output_dir) / "model_artifacts"
    code_dir = artifacts_dir / "code"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "run": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "embed_dim": int(args.embed_dim),
            "batch_size": int(args.batch_size),
            "augment_p": float(args.augment_p),
            "neighbor_k": int(args.neighbor_k),
            "gat_hidden": int(args.gat_hidden),
            "gat_heads": int(args.gat_heads),
            "gat_dropout": float(args.gat_dropout),
            "no_augment": bool(args.no_augment),
            "no_spatial_attention": bool(args.no_spatial_attention),
            "export_with_attention_mask": bool(args.export_with_attention_mask),
            "feature_selection": str(args.feature_selection),
            "atac_preprocess": str(args.atac_preprocess),
            "atac_lsi_components": int(args.atac_lsi_components),
            "atac_lsi_drop_first": bool(args.atac_lsi_drop_first),
            "atac_lsi_standardize": bool(args.atac_lsi_standardize),
            "shuffle_batches": bool(args.shuffle_batches),
        },
        "inputs": {
            "mod1_h5ad": args.mod1_h5ad,
            "mod2_h5ad": args.mod2_h5ad,
            "mod1_h5": args.mod1_h5,
            "mod2_h5": args.mod2_h5,
            "coords_csv": args.coords_csv,
            "mod1_feature_key": args.mod1_feature_key,
            "mod2_feature_key": args.mod2_feature_key,
            "skip_normalize": bool(args.skip_normalize),
        },
        "data_shapes": {
            "mod1": list(mod1_shape),
            "mod2": list(mod2_shape),
            "coords": list(coords_shape),
            "neighbor_mask_present": bool(neighbor_mask_present),
        },
    }
    with open(artifacts_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    with open(artifacts_dir / "model_architecture.txt", "w", encoding="utf-8") as f:
        f.write("=== MELON Model ===\n")
        f.write(repr(model))
        f.write("\n\n=== Encoder ===\n")
        f.write(repr(model.encoder))
        f.write("\n\n=== Head ===\n")
        f.write(repr(getattr(model, "head", None)))
        f.write("\n\n=== Neighborhood Attention ===\n")
        f.write(repr(getattr(model, "neighborhood_attn", None)))
        f.write("\n")

    # Save PyTorch state_dict weights for full model and key components.
    torch.save(model.state_dict(), artifacts_dir / "melon_model_state_dict.pt")
    torch.save(model.encoder.state_dict(), artifacts_dir / "encoder_state_dict.pt")
    if hasattr(model, "head") and model.head is not None:
        torch.save(model.head.state_dict(), artifacts_dir / "head_state_dict.pt")
    if hasattr(model, "neighborhood_attn") and model.neighborhood_attn is not None:
        torch.save(model.neighborhood_attn.state_dict(), artifacts_dir / "gat_state_dict.pt")

    # Save safetensors when available (weights-only, portable).
    safetensors_status = "not attempted"
    try:
        from safetensors.torch import save_file

        save_file(model.state_dict(), str(artifacts_dir / "melon_model.safetensors"))
        save_file(model.encoder.state_dict(), str(artifacts_dir / "encoder.safetensors"))
        if hasattr(model, "head") and model.head is not None:
            save_file(model.head.state_dict(), str(artifacts_dir / "head.safetensors"))
        if hasattr(model, "neighborhood_attn") and model.neighborhood_attn is not None:
            save_file(model.neighborhood_attn.state_dict(), str(artifacts_dir / "gat.safetensors"))
        safetensors_status = "saved"
    except Exception as exc:
        safetensors_status = f"skipped ({exc.__class__.__name__}: {exc})"

    # Snapshot the resolved package source used for this run.
    shutil.copy2(Path(__file__).resolve(), code_dir / "cli_run.py")
    pkg_root = Path(sys.modules["melon_pkg"].__file__).resolve().parent
    for src in pkg_root.rglob("*.py"):
        rel = src.relative_to(pkg_root.parent)
        dst = code_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    with open(artifacts_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write("This folder contains model architecture/code snapshot and weights.\n")
        f.write(f"safetensors_status: {safetensors_status}\n")

    print(f"Saved model artifacts to {artifacts_dir}")


def load_custom_h5(path: str, feature_key: str) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        if "X" not in f:
            raise ValueError(f"{path} must contain dataset 'X'.")
        if "spot" not in f:
            raise ValueError(f"{path} must contain dataset 'spot'.")
        X = np.array(f["X"])
        spots = decode_str_array(f["spot"][:])
        if feature_key in f:
            feats = decode_str_array(f[feature_key][:])
        elif "gene" in f:
            feats = decode_str_array(f["gene"][:])
        elif "peak" in f:
            feats = decode_str_array(f["peak"][:])
        else:
            feats = np.array([f"feature_{i}" for i in range(X.shape[1])], dtype=str)
        if X.shape[1] == len(spots):
            X = X.T
    return pd.DataFrame(X, index=spots, columns=feats)


def load_from_h5ad(
    mod1_h5ad: str,
    mod2_h5ad: str,
    coords_csv: str | None,
    feature_selection: str,
    atac_preprocess: str,
    atac_lsi_components: int,
    atac_lsi_drop_first: bool,
    atac_lsi_standardize: bool,
    seed: int,
):
    import scanpy as sc

    print(f"Loading mod1 from {mod1_h5ad}")
    adata_mod1 = sc.read_h5ad(mod1_h5ad)
    print(f"Loading mod2 from {mod2_h5ad}")
    adata_mod2 = sc.read_h5ad(mod2_h5ad)

    common = adata_mod1.obs_names.intersection(adata_mod2.obs_names)
    if len(common) == 0:
        raise ValueError("No overlapping barcodes between mod1 and mod2.")
    adata_mod1 = adata_mod1[common].copy()
    adata_mod2 = adata_mod2[common].copy()

    if feature_selection == "mod1_hvg":
        top_n = min(3000, adata_mod1.n_vars)
        if top_n < adata_mod1.n_vars:
            try:
                sc.pp.highly_variable_genes(adata_mod1, n_top_genes=top_n, flavor="seurat_v3")
            except Exception:
                sc.pp.highly_variable_genes(adata_mod1, n_top_genes=top_n, flavor="seurat")
            adata_mod1 = adata_mod1[:, adata_mod1.var.highly_variable].copy()
            print(f"Feature selection: mod1_hvg ({top_n} genes)")
        else:
            print("Feature selection: mod1_hvg requested but all genes are retained.")

    if coords_csv:
        coords_df = pd.read_csv(coords_csv, index_col=0)
    else:
        if "spatial" not in adata_mod1.obsm:
            raise ValueError("mod1_h5ad must include adata.obsm['spatial'] or pass --coords_csv.")
        coords_df = pd.DataFrame(adata_mod1.obsm["spatial"], index=adata_mod1.obs_names, columns=["x", "y"])

    if coords_df.shape[1] < 2:
        raise ValueError("Coordinates must have at least two columns.")
    if "x" not in coords_df.columns or "y" not in coords_df.columns:
        coords_df = coords_df.iloc[:, :2].copy()
        coords_df.columns = ["x", "y"]

    common = adata_mod1.obs_names.intersection(adata_mod2.obs_names).intersection(coords_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping barcodes between mod1/mod2/coords.")

    adata_mod1 = adata_mod1[common].copy()
    adata_mod2 = adata_mod2[common].copy()
    coords_df = coords_df.loc[common, ["x", "y"]].copy()

    mod1_df = pd.DataFrame(to_numpy(adata_mod1.X), index=adata_mod1.obs_names, columns=adata_mod1.var_names)
    mod2_df = pd.DataFrame(to_numpy(adata_mod2.X), index=adata_mod2.obs_names, columns=adata_mod2.var_names)

    if atac_preprocess == "tfidf_lsi":
        mod2_df = tfidf_lsi_modality(
            mod2_df,
            label="mod2 (ATAC)",
            n_components=atac_lsi_components,
            seed=seed,
            drop_first=atac_lsi_drop_first,
            standardize=atac_lsi_standardize,
        )

    return mod1_df, mod2_df, coords_df


def select_mod1_top_features(mod1_df: pd.DataFrame, feature_selection: str) -> pd.DataFrame:
    if feature_selection != "mod1_hvg":
        return mod1_df

    n_vars = mod1_df.shape[1]
    top_n = min(3000, n_vars)
    if top_n >= n_vars:
        print("Feature selection: mod1_hvg requested but all genes are retained.")
        return mod1_df

    # Prefer Scanpy HVG selection when available; fall back to variance ranking.
    try:
        import anndata as ad
        import scanpy as sc

        adata = ad.AnnData(X=mod1_df.to_numpy(dtype=np.float32, copy=False))
        adata.var_names = mod1_df.columns.astype(str)
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=top_n, flavor="seurat_v3")
        except Exception:
            sc.pp.highly_variable_genes(adata, n_top_genes=top_n, flavor="seurat")
        hvg_mask = np.asarray(adata.var["highly_variable"].to_numpy(), dtype=bool)
        if hvg_mask.any():
            selected = mod1_df.columns[hvg_mask]
            print(f"Feature selection: mod1_hvg ({len(selected)} genes)")
            return mod1_df.loc[:, selected]
    except Exception as exc:
        print(f"Feature selection: scanpy HVG unavailable ({exc.__class__.__name__}); using variance top-{top_n}.")

    var = mod1_df.var(axis=0).to_numpy(dtype=np.float64, copy=False)
    top_idx = np.argsort(var)[::-1][:top_n]
    selected = mod1_df.columns[top_idx]
    print(f"Feature selection: variance top-{len(selected)} genes")
    return mod1_df.loc[:, selected]


def load_from_raw_h5(
    mod1_h5: str,
    mod2_h5: str,
    coords_csv: str,
    mod1_feature_key: str,
    mod2_feature_key: str,
    skip_normalize: bool,
    feature_selection: str,
    atac_preprocess: str,
    atac_lsi_components: int,
    atac_lsi_drop_first: bool,
    atac_lsi_standardize: bool,
    seed: int,
):
    if not coords_csv:
        raise ValueError("--coords_csv is required with --mod1_h5/--mod2_h5.")

    print(f"Loading mod1 from {mod1_h5}")
    mod1_df = load_custom_h5(mod1_h5, mod1_feature_key)
    print(f"Loading mod2 from {mod2_h5}")
    mod2_df = load_custom_h5(mod2_h5, mod2_feature_key)
    print(f"Loading coords from {coords_csv}")
    coords_df = pd.read_csv(coords_csv, index_col=0)

    if coords_df.shape[1] < 2:
        raise ValueError("Coordinates must have at least two columns.")
    if "x" not in coords_df.columns or "y" not in coords_df.columns:
        coords_df = coords_df.iloc[:, :2].copy()
        coords_df.columns = ["x", "y"]

    common = mod1_df.index.intersection(mod2_df.index).intersection(coords_df.index)
    if len(common) == 0:
        raise ValueError("No overlapping barcodes between mod1/mod2/coords.")

    mod1_df = mod1_df.loc[common]
    mod2_df = mod2_df.loc[common]
    coords_df = coords_df.loc[common, ["x", "y"]].copy()

    if not skip_normalize:
        mod1_df = normalize_df(mod1_df, "mod1")

    if atac_preprocess == "tfidf_lsi":
        mod2_df = tfidf_lsi_modality(
            mod2_df,
            label="mod2 (ATAC)",
            n_components=atac_lsi_components,
            seed=seed,
            drop_first=atac_lsi_drop_first,
            standardize=atac_lsi_standardize,
        )
    elif not skip_normalize:
        mod2_df = normalize_df(mod2_df, "mod2")

    mod1_df = select_mod1_top_features(mod1_df, feature_selection)

    return mod1_df, mod2_df, coords_df


def main():
    parser = argparse.ArgumentParser(description="Run MELON completely standalone (outside benchmark pipeline).")
    parser.add_argument("--mod1_h5ad", type=str, default=None)
    parser.add_argument("--mod2_h5ad", type=str, default=None)
    parser.add_argument("--mod1_h5", type=str, default=None)
    parser.add_argument("--mod2_h5", type=str, default=None)
    parser.add_argument("--coords_csv", type=str, default=None)
    parser.add_argument("--mod1_feature_key", type=str, default="gene")
    parser.add_argument("--mod2_feature_key", type=str, default="peak")
    parser.add_argument("--skip_normalize", action="store_true")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embed_dim", type=int, default=128, help="Latent embedding dimension for MELON.")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--augment_p", type=float, default=0.8)
    parser.add_argument("--neighbor_k", type=int, default=16, help="k-nearest neighbors for spatial masking.")
    parser.add_argument("--gat_hidden", type=int, default=64, help="Neighborhood GAT hidden size per head.")
    parser.add_argument("--gat_heads", type=int, default=8, help="Number of Neighborhood GAT heads.")
    parser.add_argument("--gat_dropout", type=float, default=0.1, help="Neighborhood GAT dropout.")
    parser.add_argument("--feature_selection", choices=["none", "mod1_hvg"], default="none")
    parser.add_argument(
        "--atac_preprocess",
        choices=["none", "tfidf_lsi"],
        default="none",
        help="Optional preprocessing for mod2/ATAC.",
    )
    parser.add_argument(
        "--atac_lsi_components",
        type=int,
        default=50,
        help="Number of LSI components when --atac_preprocess tfidf_lsi.",
    )
    parser.add_argument(
        "--atac_lsi_drop_first",
        action="store_true",
        help="Drop the first LSI component after SVD.",
    )
    parser.add_argument(
        "--atac_lsi_standardize",
        action="store_true",
        help="Z-score LSI components.",
    )
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--no_spatial_attention", action="store_true")
    parser.add_argument("--shuffle_batches", action="store_true")
    parser.add_argument(
        "--export_with_attention_mask",
        dest="export_with_attention_mask",
        action="store_true",
        help="Use spatial attention mask when exporting embeddings (default: enabled).",
    )
    parser.add_argument(
        "--no_export_with_attention_mask",
        dest="export_with_attention_mask",
        action="store_false",
        help="Disable spatial attention mask during embedding export.",
    )
    parser.set_defaults(export_with_attention_mask=True)
    args = parser.parse_args()

    has_h5ad_inputs = bool(args.mod1_h5ad and args.mod2_h5ad)
    has_raw_inputs = bool(args.mod1_h5 and args.mod2_h5)
    if not has_h5ad_inputs and not has_raw_inputs:
        parser.error("Provide either --mod1_h5ad/--mod2_h5ad or --mod1_h5/--mod2_h5.")
    if has_h5ad_inputs and has_raw_inputs:
        parser.error("Use either h5ad inputs or raw h5 inputs, not both.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")

    print(f"Using melon_pkg from: {Path(sys.modules['melon_pkg'].__file__).resolve()}")

    if has_h5ad_inputs:
        mod1_df, mod2_df, coords_df = load_from_h5ad(
            mod1_h5ad=args.mod1_h5ad,
            mod2_h5ad=args.mod2_h5ad,
            coords_csv=args.coords_csv,
            feature_selection=args.feature_selection,
            atac_preprocess=args.atac_preprocess,
            atac_lsi_components=args.atac_lsi_components,
            atac_lsi_drop_first=bool(args.atac_lsi_drop_first),
            atac_lsi_standardize=bool(args.atac_lsi_standardize),
            seed=args.seed,
        )
    else:
        mod1_df, mod2_df, coords_df = load_from_raw_h5(
            mod1_h5=args.mod1_h5,
            mod2_h5=args.mod2_h5,
            coords_csv=args.coords_csv,
            mod1_feature_key=args.mod1_feature_key,
            mod2_feature_key=args.mod2_feature_key,
            skip_normalize=bool(args.skip_normalize),
            feature_selection=args.feature_selection,
            atac_preprocess=args.atac_preprocess,
            atac_lsi_components=args.atac_lsi_components,
            atac_lsi_drop_first=bool(args.atac_lsi_drop_first),
            atac_lsi_standardize=bool(args.atac_lsi_standardize),
            seed=args.seed,
        )

    coords_path = os.path.join(args.output_dir, "coords.csv")
    coords_df.to_csv(coords_path)

    augmentations = () if args.no_augment else ("gaussian_noise", "feature_dropout", "neighbor_mix")
    augment_p = 0.0 if args.no_augment else float(args.augment_p)

    torch.set_num_threads(1)
    model, mod1_t, mod2_t, coords_t, neighbor_mask = MELON(
        mod1_df,
        mod2_df,
        coords_path=coords_path,
        out_dir=args.output_dir,
        epochs=args.epochs,
        embed_dim=args.embed_dim,
        use_spatial_attention=not args.no_spatial_attention,
        augmentations=augmentations,
        augment_p=augment_p,
        batch_size=args.batch_size,
        neighbor_k=args.neighbor_k,
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        gat_dropout=args.gat_dropout,
        shuffle_batches=args.shuffle_batches,
        seed=args.seed,
    )

    model.eval()
    with torch.no_grad():
        view_data = [mod1_t, mod2_t]
        emb = None
        attn_weights = None
        if args.export_with_attention_mask and neighbor_mask is not None and hasattr(model, "_compute_attention_mask"):
            full_meta = {
                "coords": coords_t,
                "mask": neighbor_mask,
                "token_multiplier": len(view_data),
                "attn_mask_tokens": view_data[0].shape[0] * len(view_data),
            }
            attn_mask = model._compute_attention_mask(view_data, full_meta)
            enc_out = model.encoder(view_data, attn_mask=attn_mask, return_attention=True)
            if isinstance(enc_out, tuple):
                emb, attn_weights = enc_out
            else:
                emb = enc_out
        if emb is None:
            emb = model.encoder(view_data)
        emb_np = emb.detach().cpu().numpy() if hasattr(emb, "detach") else np.asarray(emb)

    latent_df = pd.DataFrame(emb_np, index=mod1_df.index)
    latent_path = os.path.join(args.output_dir, "latent_melon.csv")
    latent_df.to_csv(latent_path)
    print(f"Saved MELON latent embedding to {latent_path}")

    if attn_weights is not None:
        # Keep parity with old CoMM standalone workflow.
        attn_arrays = []
        for w in attn_weights:
            if w is None:
                continue
            if hasattr(w, "detach"):
                attn_arrays.append(w.detach().cpu().numpy())
            else:
                attn_arrays.append(np.asarray(w))
        if attn_arrays:
            attn_path = os.path.join(args.output_dir, "melon_attention_weights.npz")
            np.savez(attn_path, *attn_arrays)
            print(f"Saved attention weights to {attn_path}")

    save_model_artifacts(
        output_dir=args.output_dir,
        model=model,
        args=args,
        mod1_shape=mod1_df.shape,
        mod2_shape=mod2_df.shape,
        coords_shape=coords_df.shape,
        neighbor_mask_present=neighbor_mask is not None,
    )


if __name__ == "__main__":
    main()

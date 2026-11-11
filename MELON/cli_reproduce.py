"""Train and export a tissue-specific MELON representation."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from .model import make_model
from .embedding import export_embedding
from .preprocessing import load_features
from .data import DualAugDataModule, SpatialOmicsAugmentations
from .utils import build_knn_mask


def main(defaults):
    defaults = dict(defaults)
    p = argparse.ArgumentParser(description=defaults["name"])
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--accelerator", choices=["cpu", "gpu", "mps", "auto"], default="cpu")
    p.add_argument("--checkpoint", type=Path, help="Export a trusted local model checkpoint without retraining")
    p.add_argument("--epochs", type=int, default=defaults["epochs"])
    p.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    p.add_argument("--seed", type=int, default=defaults["training_seed"])
    p.add_argument("--query-chunk", type=int, default=128)
    p.add_argument("--omit", choices=defaults["modalities"], help="Retrain a modality ablation (tonsil only)")
    p.add_argument("--allow-subset", action="store_true", help="Permit a cohort size different from the manuscript")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.query_chunk < 1:
        p.error("epochs/query-chunk must be positive and batch-size at least two")
    if args.omit and defaults["dataset"] != "human_tonsil":
        p.error("--omit is provided for the tonsil modality ablations")
    modalities = [m for m in defaults["modalities"] if m != args.omit]
    ids, xy, arrays = load_features(args.features, modalities)
    if len(ids) != defaults["spots"] and not args.allow_subset:
        p.error(f"Expected {defaults['spots']} manuscript spots; got {len(ids)}. Use --allow-subset for an intentional subset")
    if len(ids) % args.batch_size == 1 and not args.checkpoint:
        p.error("The last training batch has one spot, incompatible with SyncBatchNorm; adjust --batch-size")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "embedding.csv").exists() or (args.out / "model.ckpt").exists():
        raise FileExistsError("Use a new output directory to preserve the existing run")
    pl.seed_everything(args.seed, workers=True)
    tensors = [torch.from_numpy(a) for a in arrays]
    coords = torch.from_numpy(xy)
    model = make_model([a.shape[1] for a in arrays], defaults)
    effective = dict(defaults, modalities=modalities, epochs=args.epochs, batch_size=args.batch_size,
                     training_seed=args.seed, actual_spots=len(ids), accelerator=args.accelerator,
                     omitted_modality=args.omit, export_query_chunk=args.query_chunk,
                     input_sha256=hashlib.sha256(args.features.read_bytes()).hexdigest())
    effective["software"] = {name: importlib.metadata.version(name) for name in
                             ["torch", "pytorch-lightning", "numpy", "pandas", "scikit-learn", "leidenalg"]}
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        effective["checkpoint"] = str(args.checkpoint)
        effective["checkpoint_epoch"] = int(checkpoint["epoch"]) + 1
        effective["epochs"] = effective["checkpoint_epoch"]
        effective["batch_size"] = defaults["batch_size"]
        effective["training_seed"] = defaults["training_seed"]
        effective["checkpoint_sha256"] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if "hyper_parameters" in checkpoint and "reproduction_settings" in checkpoint["hyper_parameters"]:
            saved = checkpoint["hyper_parameters"]["reproduction_settings"]
            if saved["modalities"] != modalities:
                raise ValueError("Checkpoint modalities do not match this run")
            effective["training_settings_from_checkpoint"] = saved
            for key in ["batch_size", "training_seed", "optimizer", "learning_rate", "weight_decay", "augment_p"]:
                effective[key] = saved[key]
        effective["export_device"] = "cpu"
    else:
        mask, weight = build_knn_mask(coords, k=defaults["training_spatial_k"], include_self=True, return_weights=True, symmetric=True)
        aug = dict(augmentations=["gaussian_noise", "feature_dropout", "neighbor_mix"], p=defaults["augment_p"], spatial_mix=defaults["historical_five_modal"])
        data = DualAugDataModule(mods_raw=tensors, coords=coords, neighbor_mask=mask, neighbor_weight=weight,
                                 batch_size=args.batch_size, augmentA=SpatialOmicsAugmentations(**aug), augmentB=SpatialOmicsAugmentations(**aug),
                                 spot_tokenization=True, shuffle=defaults["shuffle_batches"])
        model.save_hyperparameters({"reproduction_settings": effective})
        trainer = pl.Trainer(max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
                             logger=False, enable_checkpointing=False, enable_progress_bar=True,
                             deterministic=True, enable_model_summary=False)
        trainer.fit(model, datamodule=data)
        trainer.save_checkpoint(args.out / "model.ckpt")
        model.cpu()
        effective["export_device"] = "cpu"
    embedding = export_embedding(model, tensors, coords, defaults["export_spatial_k"], args.query_chunk)
    if embedding.shape != (len(ids), defaults["embed_dim"]) or not np.isfinite(embedding).all():
        raise RuntimeError("Invalid embedding")
    pd.DataFrame(embedding, index=ids).to_csv(args.out / "embedding.csv", index_label="barcode")
    pd.DataFrame(xy, index=ids, columns=["x", "y"]).to_csv(args.out / "coords.csv", index_label="barcode")
    (args.out / "run.json").write_text(json.dumps(effective, indent=2) + "\n")
    print(f"Saved {len(ids)} × {embedding.shape[1]} embedding to {args.out / 'embedding.csv'}")

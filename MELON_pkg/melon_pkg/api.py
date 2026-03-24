import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import pytorch_lightning as pl

# Use modular import structure as in original notebook
# Assuming MELONModel is in python path
from .models.mmfusion import MMFusion
from .pl_modules.melon import MELONModel

from .utils import build_knn_mask
from .data import SpatialOmicsAugmentations, DualAugDataModule, SEED
from .model import SpotTokenAdapter

def MELON(
    mod1_df,
    mod2_df,
    coords_path,
    out_dir="melon_out",
    epochs=100,
    embed_dim=128,
    *,
    use_spatial_attention=True,
    fusion="concat",
    fusion_heads=8,
    fusion_layers=1,
    projection_mlp_dim=256,
    # Augmentations: pass `augmentations=()` (or None) to disable for noise-free evaluation.
    augmentations=("gaussian_noise", "feature_dropout", "neighbor_mix"),
    augment_p=0.8,
    augment_random_choice=False,
    augmentA=None,
    augmentB=None,
    batch_size=2048,
    neighbor_k=16,
    gat_hidden=64,
    gat_heads=8,
    gat_dropout=0.1,
    spot_tokenization=True,
    shuffle_batches=False,
    learning_rate=1e-3,
    weight_decay=1e-5,
    temperature=0.1,
    trainer_logger=False,
    enable_progress_bar=True,
    seed=SEED,
):
    """
    Train MELON on two modalities aligned by shared spot/cell barcodes.

    Inputs
    - mod1_df: pandas DataFrame (spots x features)
    - mod2_df: pandas DataFrame (spots x features)
    - coords_path: CSV path with index=barcodes and columns ['x','y']

    Options
    - use_spatial_attention: if True, build a kNN graph from coords and use a trainable
      NeighborhoodGAT to produce an attention-bias matrix for the fusion transformer.
    - augmentations / augment_p / augment_random_choice: configure view augmentations.
      Use `augmentations=()` (or None) to disable for noise-free evaluation.
    - augmentA / augmentB: explicitly pass SpatialOmicsAugmentations modules for each view.
    """
    
    # -------------------------
    # Config
    # -------------------------
    # SEED is imported from .data
    EMBED_DIM   = int(embed_dim)     # latent size inside the model (inputs remain raw)
    EPOCHS_MELON = epochs

    NEIGHBOR_K  = int(neighbor_k)       # k-NN for spatial masking
    GAT_HIDDEN  = int(gat_hidden)
    GAT_HEADS   = int(gat_heads)
    GAT_DROPOUT = float(gat_dropout)
    if NEIGHBOR_K < 1:
        raise ValueError("neighbor_k must be >= 1")
    if GAT_HIDDEN < 1:
        raise ValueError("gat_hidden must be >= 1")
    if GAT_HEADS < 1:
        raise ValueError("gat_heads must be >= 1")
    if not (0.0 <= GAT_DROPOUT < 1.0):
        raise ValueError("gat_dropout must be in [0, 1)")
    if EMBED_DIM < 1:
        raise ValueError("embed_dim must be >= 1")
    if int(fusion_heads) < 1:
        raise ValueError("fusion_heads must be >= 1")
    if int(fusion_layers) < 1:
        raise ValueError("fusion_layers must be >= 1")
    if int(projection_mlp_dim) < 1:
        raise ValueError("projection_mlp_dim must be >= 1")

    if fusion == "x-attn" and use_spatial_attention:
        print("⚠️  Cross-attention fusion does not support spatial attention mask yet; disabling spatial attention.")
        use_spatial_attention = False
    if fusion == "x-attn" and spot_tokenization:
        print("⚠️  Cross-attention fusion is incompatible with per-spot tokenization; disabling spot tokenization.")
        spot_tokenization = False
    
    os.makedirs(out_dir, exist_ok=True)
    
    # -------------------------
    # COMPREHENSIVE SEED SETTING (CPU-OPTIMIZED FOR MAC)
    # -------------------------
    # Set seeds for all libraries
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # PyTorch Lightning seed setting
    pl.seed_everything(seed, workers=True)
    
    # Python hash seed for consistent ordering
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Force CPU usage (optimal for Mac)
    device = torch.device("cpu")
    print(f"🎯 All seeds set to {seed} for reproducible results on CPU (Mac optimized)")
    
    # -------------------------
    # 1) Load & align data
    # -------------------------
    print("📊 Loading and aligning data...")
    if not os.path.exists(coords_path):
         raise FileNotFoundError(f"Coords file not found: {coords_path}")

    coords_df = pd.read_csv(coords_path, index_col=0)

    # Normalize barcode index dtype to string so numeric barcodes survive CSV round-trips.
    coords_df.index = coords_df.index.astype(str)
    mod1_df = mod1_df.copy()
    mod2_df = mod2_df.copy()
    mod1_df.index = mod1_df.index.astype(str)
    mod2_df.index = mod2_df.index.astype(str)

    common = coords_df.index.intersection(mod1_df.index).intersection(mod2_df.index)
    if len(common) == 0:
        raise ValueError("No shared barcodes across coords/Mod1/Mod2.")
    
    coords = coords_df.loc[common, ['x', 'y']].to_numpy()
    X_mod1  = mod1_df.loc[common].to_numpy(dtype=np.float32)   # (N, G)
    X_mod2 = mod2_df.loc[common].to_numpy(dtype=np.float32)  # (N, P)
    N, G = X_mod1.shape
    P    = X_mod2.shape[1]
    print(f"✅ Data loaded: N={N}, mod1_feats={G}, mod2_feats={P}, device={device}")
    
    # Convert to tensors for downstream training
    mod1_raw_t  = torch.from_numpy(X_mod1)
    mod2_raw_t = torch.from_numpy(X_mod2)
    
    
    coords_tensor = torch.from_numpy(coords).to(torch.float32)
    # create knn mask using utility (only needed if spatial attention is enabled)
    if use_spatial_attention:
        neighbor_mask, neighbor_weight = build_knn_mask(
            coords_tensor,
            k=NEIGHBOR_K,
            include_self=True,
            return_weights=True,
        )
    else:
        neighbor_mask = None
        neighbor_weight = None
    
    print("✅ Tensor shapes:")
    print(f"   • Mod1:  {mod1_raw_t.shape}")
    print(f"   • Mod2: {mod2_raw_t.shape}")
    print(f"   • Coords: {coords_tensor.shape}")
    if neighbor_mask is not None:
        print(f"   • Neighbor mask density: {neighbor_mask.float().mean().item():.4f}")
        nz_weights = neighbor_weight[neighbor_mask]
        print(f"   • Mean neighbor weight: {nz_weights.mean().item():.4f}")
    else:
        print("   • Neighbor mask: disabled")
    
    # -------------------------
    # Augmentations (two views)
    # -------------------------
    # Back-compat: if only one view augmentation is provided, reuse it for the other view.
    if augmentA is None and augmentB is not None:
        augmentA = augmentB
    if augmentB is None and augmentA is not None:
        augmentB = augmentA
    if augmentA is None and augmentB is None:
        aug_list = [] if augmentations is None else list(augmentations)
        if len(aug_list) > 0 and float(augment_p) > 0:
            augmentA = SpatialOmicsAugmentations(
                augmentations=aug_list,
                p=float(augment_p),
                random_choice=bool(augment_random_choice),
            )
            augmentB = SpatialOmicsAugmentations(
                augmentations=aug_list,
                p=float(augment_p),
                random_choice=bool(augment_random_choice),
            )
        else:
            augmentA = None
            augmentB = None

    # ------------- build datamodule -------------
    data_mod = DualAugDataModule(
        mod1_raw=mod1_raw_t, mod2_raw=mod2_raw_t,
        coords=coords_tensor, neighbor_mask=neighbor_mask, neighbor_weight=neighbor_weight,
        batch_size=int(batch_size),
        augmentA=augmentA,
        augmentB=augmentB,
        device=device,
        spot_tokenization=bool(spot_tokenization),
        shuffle=bool(shuffle_batches),
    )
    
    # -------------------------
    # 4) Model setup with modular imports
    # -------------------------
    print("🧠 Setting up model with modular imports...")
    
    # Using imports from top of file
    
    print("✅ Successfully imported from modular structure:")
    print(f"   • MMFusion from models.mmfusion")
    print(f"   • MELONModel from pl_modules.melon")
    
    # Encoders map raw -> latent (EMBED_DIM). Inputs remain raw-sized.
    encoder_mod1  = nn.Sequential(nn.Linear(G, EMBED_DIM), nn.ReLU(), nn.Linear(EMBED_DIM, EMBED_DIM))
    encoder_mod2 = nn.Sequential(nn.Linear(P, EMBED_DIM), nn.ReLU(), nn.Linear(EMBED_DIM, EMBED_DIM))
    
    print(f"✅ Encoder dimensions:")
    print(f"   • Mod1 encoder: {G} -> {EMBED_DIM}")
    print(f"   • Mod2 encoder: {P} -> {EMBED_DIM}")
    
    fusion_encoder = MMFusion(
        encoders=[encoder_mod1, encoder_mod2],
        input_adapters=[
            SpotTokenAdapter(collapse_batch=(fusion != "x-attn")),
            SpotTokenAdapter(collapse_batch=(fusion != "x-attn")),
        ],
        embed_dim=EMBED_DIM,
        fusion=fusion,
        pool="mean" if fusion == "x-attn" else "tokens",
        n_heads=int(fusion_heads),
        n_layers=int(fusion_layers),
    )
    
    projection = MELONModel._build_mlp(
        in_dim=EMBED_DIM,
        mlp_dim=int(projection_mlp_dim),
        out_dim=EMBED_DIM,
    )
    
    neighborhood_cfg = None
    if use_spatial_attention:
        neighborhood_cfg = {
            "hidden_dim": GAT_HIDDEN,
            "heads": GAT_HEADS,
            "dropout": GAT_DROPOUT,
            "k_neighbors": NEIGHBOR_K,
            "include_self": True
        }
    
    model = MELONModel(
        encoder=fusion_encoder,
        projection=projection,
        optim_kwargs={"lr": float(learning_rate), "weight_decay": float(weight_decay)},
        loss_kwargs={"temperature": float(temperature)},
        neighborhood_cfg=neighborhood_cfg
    )
    
    trainer = pl.Trainer(
        max_epochs=EPOCHS_MELON,
        accelerator="cpu",          # Force CPU for Mac
        devices=1,
        logger=trainer_logger,
        enable_checkpointing=False,
        enable_progress_bar=bool(enable_progress_bar),
        deterministic=True          # Enable deterministic training
    )
    trainer.fit(model, datamodule=data_mod)
    
    print("✅ Training completed successfully!")

    return model, mod1_raw_t, mod2_raw_t, coords_tensor, neighbor_mask

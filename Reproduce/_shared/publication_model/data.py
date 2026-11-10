import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
import pytorch_lightning as pl
from typing import List, Union, Iterable
from copy import deepcopy

# Placeholder seed constant if needed, but usually passed or set globally.
SEED = 2027


def build_spatial_patch_batches(coords, neighbor_mask, core_size=256):
    """Partition spots into compact cores and add one graph-neighbor halo."""
    xy = coords.detach().cpu().numpy() if torch.is_tensor(coords) else np.asarray(coords)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("coords must have shape (n_spots, 2)")
    if int(core_size) < 2:
        raise ValueError("core_size must be >= 2")
    leaves = []

    def split(indices):
        if len(indices) <= int(core_size):
            leaves.append(np.asarray(indices, dtype=np.int64))
            return
        axis = int(np.argmax(np.ptp(xy[indices], axis=0)))
        ordered = indices[np.argsort(xy[indices, axis], kind="mergesort")]
        midpoint = len(ordered) // 2
        split(ordered[:midpoint])
        split(ordered[midpoint:])

    split(np.arange(xy.shape[0], dtype=np.int64))
    mask = neighbor_mask
    if mask is not None and not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, dtype=torch.bool)
    patches = []
    for core_np in leaves:
        core = torch.as_tensor(core_np, dtype=torch.long)
        if mask is None:
            batch = core
        else:
            halo_flag = mask.index_select(0, core).any(dim=0)
            halo_flag[core] = False
            batch = torch.cat([core, torch.where(halo_flag)[0]])
        patches.append((core, batch))
    return patches


class SpatialPatchBatchSampler(Sampler):
    """Yield compact spatial cores with a one-hop graph halo."""

    def __init__(self, coords, neighbor_mask, core_size=256, shuffle=True, seed=SEED):
        self.patches = build_spatial_patch_batches(coords, neighbor_mask, core_size)
        self.shuffle = bool(shuffle)
        self.generator = torch.Generator().manual_seed(int(seed))

    def __iter__(self):
        order = (
            torch.randperm(len(self.patches), generator=self.generator).tolist()
            if self.shuffle else range(len(self.patches))
        )
        for i in order:
            yield self.patches[i][1].tolist()

    def __len__(self):
        return len(self.patches)

class SpatialOmicsAugmentations(torch.nn.Module):
    """
    Define spatial-omics-specific augmentations to apply to multimodal data
    (e.g., Mod1 + Mod2 matrices). Inspired by:
        [1] SimCLR (Chen et al., ICML 2020)
        [2] Factorized Contrastive Learning (Liang et al., NeurIPS 2023)
        [3] SpatialGlue / BANKSY augmentations for spatial omics

    Each input sample is a list: [X_mod1, X_mod2] (numpy arrays or tensors)
    """

    def __init__(self,
                 augmentations: Union[str, List[str]] = None,
                 p: float = 0.5,
                 random_choice: bool = False):
        super().__init__()
        if augmentations is None:
            augmentations = []
        assert isinstance(augmentations, str) or isinstance(augmentations, Iterable)
        if isinstance(augmentations, str):
            augmentations = [augmentations]

        self.augmentations = augmentations
        self.p = p
        self.random_choice = random_choice

        # mapping from string name -> function
        self.registry = {
            "gaussian_noise": self.gaussian_noise,
            "feature_dropout": self.feature_dropout,
            "spot_dropout": self.spot_dropout,
            "scaled_perturb": self.scaled_perturb,
            "neighbor_mix": self.neighbor_mix,
            "simclr": self.simclr_like,
        }

    def forward(self, x: List[np.ndarray], *, neighbor_mask=None):
        """
        x: list of modalities [X_mod1, X_mod2], shape (N, G) and (N, P)
        """
        if self.random_choice and len(self.augmentations) > 0:
            aug_name = np.random.choice(self.augmentations)
            fn = self.registry.get(aug_name, None)
            if fn is not None:
                return self._apply(fn, aug_name, deepcopy(x), neighbor_mask=neighbor_mask)
        else:
            for aug_name in self.augmentations:
                if np.random.rand() < self.p:
                    fn = self.registry.get(aug_name, None)
                    if fn is not None:
                        x = self._apply(fn, aug_name, x, neighbor_mask=neighbor_mask)
        return x

    @staticmethod
    def _apply(fn, aug_name, x, *, neighbor_mask=None):
        if aug_name == "neighbor_mix":
            return fn(x, neighbor_mask=neighbor_mask)
        return fn(x)

    # --------------------------------------------------
    # Augmentation functions
    # --------------------------------------------------

    @staticmethod
    def gaussian_noise(x, std=0.05):
        """Add Gaussian noise independently to each modality."""
        return [x_ + np.random.randn(*x_.shape).astype(np.float32) * std for x_ in x]

    @staticmethod
    def feature_dropout(x, frac=0.2):
        """Randomly zero out a fraction of features (columns)."""
        x_aug = []
        for x_ in x:
            drop_idx = np.random.choice(x_.shape[1],
                                        size=int(frac * x_.shape[1]),
                                        replace=False)
            x_aug_ = np.copy(x_)
            x_aug_[:, drop_idx] = 0
            x_aug.append(x_aug_)
        return x_aug

    @staticmethod
    def spot_dropout(x, frac=0.1):
        """Drop a fraction of spots (rows)."""
        x_aug = []
        drop_idx = np.random.choice(x[0].shape[0],
                                    size=int(frac * x[0].shape[0]),
                                    replace=False)
        for x_ in x:
            x_aug_ = np.copy(x_)
            x_aug_[drop_idx, :] = 0
            x_aug.append(x_aug_)
        return x_aug

    @staticmethod
    def scaled_perturb(x, scale_range=(0.9, 1.1)):
        """Randomly scale expression intensity across genes/features."""
        x_aug = []
        for x_ in x:
            scale = np.random.uniform(*scale_range, size=(1, x_.shape[1]))
            x_aug.append(x_ * scale.astype(np.float32))
        return x_aug

    @staticmethod
    def neighbor_mix(x, neighbor_mask=None, alpha=0.3):
        """Mix each spot with a true spatial neighbor from the current batch."""
        n = x[0].shape[0]
        if n == 0:
            return x

        pair_idx = np.arange(n)
        if neighbor_mask is not None:
            if torch.is_tensor(neighbor_mask):
                mask_np = neighbor_mask.detach().cpu().numpy().astype(bool, copy=False)
            else:
                mask_np = np.asarray(neighbor_mask, dtype=bool)
            if mask_np.shape != (n, n):
                raise ValueError("neighbor_mask must be (batch_size, batch_size) for neighbor_mix.")

            mask_np = mask_np.copy()
            np.fill_diagonal(mask_np, False)
            for i in range(n):
                candidates = np.flatnonzero(mask_np[i])
                pair_idx[i] = np.random.choice(candidates) if candidates.size > 0 else i

        lam = np.random.beta(alpha, alpha, size=(n, 1)).astype(np.float32)
        mixed = []
        for x_ in x:
            arr = np.asarray(x_, dtype=np.float32)
            mixed_arr = lam * arr + (1.0 - lam) * arr[pair_idx]
            mixed.append(mixed_arr.astype(arr.dtype, copy=False))
        return mixed

    @staticmethod
    def simclr_like(x):
        """SimCLR-style composite augmentation: noise + feature dropout."""
        x = SpatialOmicsAugmentations.gaussian_noise(x, std=0.05)
        x = SpatialOmicsAugmentations.feature_dropout(x, frac=0.1)
        return x

def collate_with_dual_augment(batch, augmentA=None, augmentB=None, device=None, neighbor_mask=None):
    """
    batch: list of ([mod_1, ..., mod_M], idx) where tensors are 1D per modality.
    Returns:
      (viewA, viewB, batch_indices, batch_neighbor_mask)
      where viewA/viewB are tuples of length M, each entry shaped (B, F_m).
      The two views are generated via augmentations from the same base sample.
    """
    n_mod = len(batch[0][0])
    mods = [torch.stack([item[0][m] for item in batch], dim=0) for m in range(n_mod)]
    batch_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    batch_neighbor_mask = None
    if neighbor_mask is not None:
        batch_neighbor_mask = neighbor_mask[batch_indices][:, batch_indices]

    modsA = [m.clone() for m in mods]
    modsB = [m.clone() for m in mods]

    # --- Augment View A ---
    if augmentA is not None:
        np_views = [m.detach().cpu().numpy() for m in modsA]
        np_views = augmentA(np_views, neighbor_mask=batch_neighbor_mask)
        modsA = [torch.from_numpy(np_views[m]).type_as(modsA[m]) for m in range(n_mod)]

    # --- Augment View B ---
    if augmentB is not None:
        np_views = [m.detach().cpu().numpy() for m in modsB]
        np_views = augmentB(np_views, neighbor_mask=batch_neighbor_mask)
        modsB = [torch.from_numpy(np_views[m]).type_as(modsB[m]) for m in range(n_mod)]

    if device is not None:
        modsA = [m.to(device) for m in modsA]
        modsB = [m.to(device) for m in modsB]

    return tuple(modsA), tuple(modsB), batch_indices, batch_neighbor_mask

# -------------------------
# 3) Dataset preparation
# -------------------------

class SpatialDataset(Dataset):
    """Dataset over an arbitrary number of modality tensors aligned by row index.

    Accepts either a list of modality tensors via ``mods_raw`` or the legacy
    two-argument form ``(mod1_raw, mod2_raw)`` for backward compatibility.
    """

    def __init__(self, *args, mods_raw=None, mod1_raw=None, mod2_raw=None):
        if mods_raw is None:
            if args:
                # Positional form. Treat as list of modality tensors.
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    mods_raw = list(args[0])
                else:
                    mods_raw = list(args)
            elif mod1_raw is not None and mod2_raw is not None:
                mods_raw = [mod1_raw, mod2_raw]
            else:
                raise ValueError("SpatialDataset requires mods_raw or (mod1_raw, mod2_raw).")

        if len(mods_raw) < 2:
            raise ValueError("SpatialDataset requires at least 2 modalities.")
        n0 = mods_raw[0].size(0)
        for i, t in enumerate(mods_raw):
            if t.size(0) != n0:
                raise ValueError(f"Modality {i} has {t.size(0)} rows; expected {n0}.")
        self.mods_raw = list(mods_raw)

    def __len__(self):
        return self.mods_raw[0].size(0)

    def __getitem__(self, idx):
        return [t[idx] for t in self.mods_raw], idx

class DualAugDataModule(pl.LightningDataModule):
    def __init__(self, *args, mods_raw=None, mod1_raw=None, mod2_raw=None,
                 coords=None, neighbor_mask=None, neighbor_weight=None,
                 batch_size=1024, augmentA=None, augmentB=None, device=None,
                 spot_tokenization=False, shuffle=False,
                 spatial_patch_batches=False, spatial_patch_core_size=256):
        super().__init__()
        # Resolve modality tensors. Supports:
        #   DualAugDataModule(mod1, mod2, coords=..., ...)              # legacy positional
        #   DualAugDataModule(mod1_raw=mod1, mod2_raw=mod2, ...)        # legacy kwargs
        #   DualAugDataModule(mods_raw=[mod1, mod2, mod3, ...], ...)    # N-modality
        if mods_raw is None:
            if args:
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    mods_raw = list(args[0])
                else:
                    mods_raw = list(args)
            elif mod1_raw is not None and mod2_raw is not None:
                mods_raw = [mod1_raw, mod2_raw]
            else:
                raise ValueError("DualAugDataModule requires mods_raw or (mod1_raw, mod2_raw).")
        self.ds = SpatialDataset(mods_raw=mods_raw)
        self.bs = batch_size
        self.augmentA = augmentA
        self.augmentB = augmentB
        self.device = device
        self.spot_tokenization = spot_tokenization
        self.shuffle = bool(shuffle)
        self.spatial_patch_batches = bool(spatial_patch_batches)
        self.spatial_patch_core_size = int(spatial_patch_core_size)
        if coords is None:
            raise ValueError("coords tensor is required for neighborhood masking.")
        self.coords = coords
        self.neighbor_mask = neighbor_mask
        self.neighbor_weight = neighbor_weight

        self.generator = torch.Generator()
        self.generator.manual_seed(SEED)
        self.patch_sampler = None
        if self.spatial_patch_batches:
            self.patch_sampler = SpatialPatchBatchSampler(
                self.coords,
                self.neighbor_mask,
                core_size=self.spatial_patch_core_size,
                shuffle=True,
                seed=SEED,
            )

        def _collate(batch):
            viewA, viewB, batch_indices, mask = collate_with_dual_augment(
                batch,
                augmentA=self.augmentA,
                augmentB=self.augmentB,
                device=self.device,
                neighbor_mask=self.neighbor_mask,
            )
            coords_batch = self.coords[batch_indices]
            weight = None
            if self.neighbor_weight is not None:
                weight = self.neighbor_weight[batch_indices][:, batch_indices]
            meta = {
                "coords": coords_batch,
                "mask": mask,
                "weight": weight,
                "indices": batch_indices
            }
            if self.spot_tokenization:
                n_modalities = len(viewA) if isinstance(viewA, (list, tuple)) else 1
                meta["token_multiplier"] = n_modalities
                meta["attn_mask_tokens"] = batch_indices.numel() * n_modalities
                if mask is not None:
                    # print(meta["attn_mask_tokens"], mask.shape, mask.sum().item())
                    pass
            return viewA, viewB, meta
        self._collate = _collate

    def train_dataloader(self):
        if self.patch_sampler is not None:
            return DataLoader(
                self.ds, batch_sampler=self.patch_sampler,
                num_workers=0, collate_fn=self._collate
            )
        return DataLoader(
            self.ds, batch_size=self.bs, shuffle=self.shuffle,
            num_workers=0, generator=self.generator, collate_fn=self._collate
        )

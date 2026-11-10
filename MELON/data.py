import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import List, Union, Iterable
from copy import deepcopy

# Placeholder seed constant if needed, but usually passed or set globally.
SEED = 2027

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

    def forward(self, x: List[np.ndarray]):
        """
        x: list of modalities [X_mod1, X_mod2], shape (N, G) and (N, P)
        """
        if self.random_choice and len(self.augmentations) > 0:
            aug_name = np.random.choice(self.augmentations)
            fn = self.registry.get(aug_name, None)
            if fn is not None:
                return fn(deepcopy(x))
        else:
            for aug_name in self.augmentations:
                if np.random.rand() < self.p:
                    fn = self.registry.get(aug_name, None)
                    if fn is not None:
                        x = fn(x)
        return x

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
    def neighbor_mix(x, alpha=0.3):
        """Mix each spot with a random neighbor (spatial mixup style)."""
        n = x[0].shape[0]
        idx = np.random.permutation(n)
        lam = np.random.beta(alpha, alpha)
        return [lam * x_ + (1 - lam) * x_[idx] for x_ in x]

    @staticmethod
    def simclr_like(x):
        """SimCLR-style composite augmentation: noise + feature dropout."""
        x = SpatialOmicsAugmentations.gaussian_noise(x, std=0.05)
        x = SpatialOmicsAugmentations.feature_dropout(x, frac=0.1)
        return x

def collate_with_dual_augment(batch, augmentA=None, augmentB=None, device=None):
    """
    batch: list of ([mod1, mod2], idx) where tensors are 1D per modality.
    Returns:
      ((mod1A, mod2A), (mod1B, mod2B), batch_indices) with shapes (B, F) and (B,)
      The two views are generated via augmentations from the same base sample.
    """
    mod1 = torch.stack([item[0][0] for item in batch], dim=0)
    mod2 = torch.stack([item[0][1] for item in batch], dim=0)
    batch_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)

    mod1A, mod2A = mod1.clone(), mod2.clone()
    mod1B, mod2B = mod1.clone(), mod2.clone()

    # --- Augment View A ---
    if augmentA is not None:
        mod1_np, mod2_np = mod1A.detach().cpu().numpy(), mod2A.detach().cpu().numpy()
        mod1_np, mod2_np = augmentA([mod1_np, mod2_np])
        mod1A = torch.from_numpy(mod1_np).type_as(mod1A)
        mod2A = torch.from_numpy(mod2_np).type_as(mod2A)

    # --- Augment View B ---
    if augmentB is not None:
        mod1_np, mod2_np = mod1B.detach().cpu().numpy(), mod2B.detach().cpu().numpy()
        mod1_np, mod2_np = augmentB([mod1_np, mod2_np])
        mod1B = torch.from_numpy(mod1_np).type_as(mod1B)
        mod2B = torch.from_numpy(mod2_np).type_as(mod2B)

    if device is not None:
        mod1A = mod1A.to(device); mod2A = mod2A.to(device)
        mod1B = mod1B.to(device); mod2B = mod2B.to(device)

    return (mod1A, mod2A), (mod1B, mod2B), batch_indices

# -------------------------
# 3) Dataset preparation
# -------------------------

class SpatialDataset(Dataset):
    def __init__(self, mod1_raw, mod2_raw):
        self.mod1_raw = mod1_raw
        self.mod2_raw = mod2_raw

    def __len__(self):
        return self.mod1_raw.size(0)

    def __getitem__(self, idx):
        return [self.mod1_raw[idx], self.mod2_raw[idx]], idx

class DualAugDataModule(pl.LightningDataModule):
    def __init__(self, mod1_raw, mod2_raw, coords, neighbor_mask,
                 batch_size=1024, augmentA=None, augmentB=None, device=None,
                 spot_tokenization=False, shuffle=False):
        super().__init__()
        self.ds = SpatialDataset(mod1_raw, mod2_raw)
        self.bs = batch_size
        self.augmentA = augmentA
        self.augmentB = augmentB
        self.device = device
        self.spot_tokenization = spot_tokenization
        self.shuffle = bool(shuffle)
        if coords is None:
            raise ValueError("coords tensor is required for neighborhood masking.")
        self.coords = coords
        self.neighbor_mask = neighbor_mask

        self.generator = torch.Generator()
        self.generator.manual_seed(SEED)

        def _collate(batch):
            viewA, viewB, batch_indices = collate_with_dual_augment(
                batch, augmentA=self.augmentA, augmentB=self.augmentB, device=self.device
            )
            coords_batch = self.coords[batch_indices]
            mask = None
            if self.neighbor_mask is not None:
                mask = self.neighbor_mask[batch_indices][:, batch_indices]
            meta = {
                "coords": coords_batch,
                "mask": mask,
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
        return DataLoader(
            self.ds, batch_size=self.bs, shuffle=self.shuffle,
            num_workers=0, generator=self.generator, collate_fn=self._collate
        )

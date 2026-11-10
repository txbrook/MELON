from torch import nn
import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List, Optional
# Local imports
from .base import BaseModel
from ..losses.melon_loss import MELONLoss
from ..models.mmfusion import MMFusion


class NeighborhoodGAT(nn.Module):
    """
    Lightweight Graph Attention module that outputs an attention-bias matrix
    suitable for `nn.MultiheadAttention(attn_mask=...)`.
    """

    def __init__(self,
    # ... (NeighborhoodGAT arguments) ...
                 hidden_dim: int = 64,
                 heads: int = 4,
                 dropout: float = 0.0,
                 negative_slope: float = 0.2,
                 k_neighbors: Optional[int] = None,
                 include_self: bool = True):
        super().__init__()
        # ... (rest of NeighborhoodGAT) ...
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.k_neighbors = k_neighbors
        self.include_self = include_self
        self.input_proj = nn.LazyLinear(hidden_dim * heads, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, hidden_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, hidden_dim))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        if hasattr(self.input_proj, "reset_parameters"):
            self.input_proj.reset_parameters()

    def forward(self,
                node_features: torch.Tensor,
                coords: Optional[torch.Tensor] = None,
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if node_features.dim() != 2:
            raise ValueError("`node_features` must be rank-2 (n_nodes, feat_dim).")

        device = node_features.device
        n_nodes = node_features.size(0)

        mask = None
        if neighbor_mask is not None:
            mask = neighbor_mask.to(device=device, dtype=torch.bool)
        elif coords is not None and self.k_neighbors is not None:
            mask = self._knn_mask(coords.to(device), n_nodes)

        if mask is not None and self.include_self:
            mask = mask.clone()
            mask |= torch.eye(n_nodes, dtype=torch.bool, device=device)

        projected = self.input_proj(node_features)  # (n_nodes, heads * hidden_dim)
        projected = projected.view(n_nodes, self.heads, self.hidden_dim)

        attn_src = torch.einsum("nhd,hd->nh", projected, self.attn_src)
        attn_dst = torch.einsum("nhd,hd->nh", projected, self.attn_dst)
        logits = attn_src.unsqueeze(1) + attn_dst.unsqueeze(0)
        logits = self.leaky_relu(logits)

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(-1), float("-inf"))

        weights = torch.softmax(logits, dim=1)
        weights = self.dropout(weights)
        weights = weights.mean(dim=-1)

        if mask is not None:
            weights = weights * mask.float()

        weights = weights + 1e-6
        return torch.log(weights)

    def _knn_mask(self, coords: torch.Tensor, n_nodes: int) -> torch.Tensor:
        if coords.dim() == 1:
            coords = coords.unsqueeze(-1)
        coords = coords.to(dtype=torch.float32)
        dists = torch.cdist(coords, coords)
        k = min(self.k_neighbors + 1, n_nodes)
        knn = torch.topk(dists, k=k, largest=False).indices[:, 1:]
        mask = torch.zeros(n_nodes, n_nodes, dtype=torch.bool, device=coords.device)
        row_idx = torch.arange(n_nodes, device=coords.device).unsqueeze(1).expand_as(knn)
        mask[row_idx, knn] = True
        return mask


class MELONModel(BaseModel):
    """ Contrastive MultiModal learning allowing the communication between modalities 
    in a single multimodal space [1].
    
    It encodes a pair of mulitmodal data and outputs a pair of representations through
    a single multimodal encoder.

    [1] What to align in multimodal contrastive learning, Dufumier & Castillo-Navarro et al., ICLR 2025
    """

    def __init__(self,
                 encoder: MMFusion,
                 projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict,
                 neighborhood_cfg: Optional[Dict] = None):
        """
        Args:
            encoder: Multi-modal fusion encoder
            projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters
            loss_kwargs: Hyper-parameters for the MELON loss.
            neighborhood_cfg: Optional dictionary to configure the
                `NeighborhoodGAT` mask builder.
        """
        super(MELONModel, self).__init__(optim_kwargs)

        # create the encoder
        self.encoder = encoder

        # build a 3-layers projector
        self.head = projection

        # Build the loss
        self.loss = MELONLoss(**loss_kwargs)

        self.neighborhood_attn = NeighborhoodGAT(**neighborhood_cfg) if neighborhood_cfg else None
        self._neighbor_mask_warned = False


    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))


    def forward(self,
                x1: List[torch.Tensor],
                x2: List[torch.Tensor],
                neighborhood_meta: Optional[Dict] = None):
        # compute features for all modalities
        all_masks = self.gen_all_possible_masks(len(x1))
        meta1, meta2 = self._split_meta(neighborhood_meta)
        attn_mask_1 = self._compute_attention_mask(x1, meta1)
        attn_mask_2 = self._compute_attention_mask(x2, meta2)
        z1 = self.encoder(x1, mask_modalities=all_masks, attn_mask=attn_mask_1)
        z2 = self.encoder(x2, mask_modalities=all_masks, attn_mask=attn_mask_2)
        z1 = [self.head(z) for z in z1]
        z2 = [self.head(z) for z in z2]
        return {'aug1_embed': z1,
                'aug2_embed': z2,
                "prototype": -1}
    

    def gen_all_possible_masks(self, n_mod: int):
        """
        :param n_mod: int
        :return: a list of `n_mod` + 1 boolean masks [Mi] such that all but one bool are False.
            A last bool mask is added where all bool are True
        Examples:
        *   For n_mod==2:
            masks == [[True, False], [False, True], [True, True]]
        *   For n_mod == 3:
            masks == [[True, False, False], [False, True, False], [False, False, True], [True, True, True]]
        """
        masks = []
        for L in range(n_mod):
            mask = [s == L for s in range(n_mod)]
            masks.append(mask)
        masks.append([True for _ in range(n_mod)])
        return masks

    def _split_meta(self, neighborhood_meta: Optional[Dict]):
        if neighborhood_meta is None:
            return None, None
        if isinstance(neighborhood_meta, (list, tuple)):
            if len(neighborhood_meta) != 2:
                raise ValueError("`neighborhood_meta` as list/tuple must have length 2.")
            return neighborhood_meta
        if isinstance(neighborhood_meta, dict):
            meta1 = neighborhood_meta.get("view1", neighborhood_meta)
            meta2 = neighborhood_meta.get("view2", neighborhood_meta)
            return meta1, meta2
        return neighborhood_meta, neighborhood_meta

    def _compute_attention_mask(
        self,
        modalities: List[torch.Tensor],
        neighbor_meta: Optional[Dict]
    ) -> Optional[torch.Tensor]:
        if self.neighborhood_attn is None or neighbor_meta is None:
            return None

        device = modalities[0].device

        # Assume each modality is (n_spots, feat_dim_m)
        n_spots = modalities[0].shape[0]
        for m in modalities[1:]:
            if m.shape[0] != n_spots:
                raise ValueError("All modalities must share the same number of spots / nodes")
    
        # Build per-spot node features by concatenating modalities along feature dim
        node_features = torch.cat(
            [m.reshape(n_spots, -1).to(device=device, dtype=torch.float32)
            for m in modalities],
            dim=-1,  # (n_spots, sum_m feat_dim_m)
        )

        coords = neighbor_meta.get("coords", None)
        neighbor_mask = neighbor_meta.get("mask", None)
        if neighbor_mask is None:
            neighbor_mask = neighbor_meta.get("neighbor_mask", None)
        token_multiplier = int(neighbor_meta.get("token_multiplier", 1))

        if coords is not None and not torch.is_tensor(coords):
            coords = torch.as_tensor(coords, device=device, dtype=torch.float32)

        if neighbor_mask is not None and not torch.is_tensor(neighbor_mask):
            neighbor_mask = torch.as_tensor(neighbor_mask, device=device, dtype=torch.bool)

    # Optionally check consistency here
        if coords is not None and coords.shape[0] != n_spots:
            raise ValueError("coords must have the same number of rows as spots")
        if neighbor_mask is not None and neighbor_mask.shape != (n_spots, n_spots):
            raise ValueError("neighbor_mask must be (n_spots, n_spots)")

        # This returns (n_spots, n_spots) float bias for MultiheadAttention
        attn_bias = self.neighborhood_attn(
            node_features,
            coords=coords,
            neighbor_mask=neighbor_mask
        )
        if token_multiplier > 1 and attn_bias is not None:
            attn_bias = attn_bias.repeat(token_multiplier, token_multiplier)
        # Current MMFusion adapters operate with one token per modality, so the fusion
        # transformer expects an attn_mask shaped (n_modalities, n_modalities).  If the
        # computed neighborhood bias does not match that shape (the usual case when the
        # mask encodes relationships between samples within the batch), skip it to avoid
        # shape errors downstream.
        expected_tokens = neighbor_meta.get("attn_mask_tokens", len(modalities))
        if attn_bias.shape[0] != expected_tokens:
            if not self._neighbor_mask_warned:
                print(
                    f"⚠️  Skipping neighborhood attention mask: got {attn_bias.shape} "
                    f"but encoder expects a ({expected_tokens}, {expected_tokens}) mask. "
                    "Set a custom adapter producing per-token masks if spatial biasing is required."
                )
                self._neighbor_mask_warned = True
            return None
        return attn_bias
    
    
    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
           Extract multimodal features from the encoder.
           Args:
                loader: Dataset loader to serve `(X, y)` tuples.
                kwargs: given to `encoder.forward()`
           Returns: 
                Pair (Z,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        for X_, y_ in loader:
            if isinstance(X_, torch.Tensor): # needs to cast it as list of one modality
                X_ = [X_]
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            y_ = y_.to(self.device)
            with torch.inference_mode():
                # compute output
                output = self.encoder(X_, **kwargs)
                X.extend(output.view(len(output), -1).detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
from einops import repeat
from collections import OrderedDict

print("DEBUG: Loading CoMM/models/mmfusion.py")

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 dropout: float = 0., bias: bool = True, batch_first: bool = False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = embed_dim // n_heads
        self.batch_first = batch_first

        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = dropout

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False):
        
        # Handle batch_first=False input (seq, batch, dim) -> (batch, seq, dim)
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        B, L_q, _ = query.shape
        B, L_k, _ = key.shape
        B, L_v, _ = value.shape

        q = self.q_proj(query).view(B, L_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, L_k, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, L_v, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat k/v heads if n_kv_heads < n_heads (GQA)
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # Scaled Dot Product Attention
        # attn_mask handling: PyTorch SDPA expects (Batch, Heads, L_q, L_k) or (L_q, L_k)
        # If attn_mask is provided, ensure it's compatible.
        
        # Note: F.scaled_dot_product_attention handles dropout and masking efficiently
        # However, it expects attn_mask to be float for additive mask or bool for padding mask
        # Our attn_mask is likely additive float mask from previous code context (-inf/0)
        
        is_causal = False # We don't strictly enforce causal here unless passed in attn_mask
        
        # Adjust key_padding_mask to be merged into attn_mask if needed or passed separately
        # F.scaled_dot_product_attention supports attn_mask.
        # If key_padding_mask is present (Batch, L_k), we can merge it.
        
        final_mask = attn_mask
        if key_padding_mask is not None:
            # key_padding_mask is (Batch, L_k), True where padded (ignore)
            # Create a mask (Batch, 1, 1, L_k) to broadcast
            kp_mask = key_padding_mask.view(B, 1, 1, L_k)
            if final_mask is None:
                final_mask = torch.zeros((B, self.n_heads, L_q, L_k), device=q.device, dtype=q.dtype)
                final_mask = final_mask.masked_fill(kp_mask.bool(), float("-inf"))
            else:
                # If final_mask is 2D (L_q, L_k), expand it
                if final_mask.dim() == 2:
                    final_mask = final_mask.unsqueeze(0).unsqueeze(0) # (1, 1, L_q, L_k)
                final_mask = final_mask.masked_fill(kp_mask.bool(), float("-inf"))

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=final_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal
        )

        x = x.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
        x = self.out_proj(x)

        if not self.batch_first:
            x = x.transpose(0, 1)

        if need_weights:
             # Re-computing weights is expensive and usually not needed for GQA optimization
             # Returning None for weights to simplify
             return x, None
        
        return x, None


class ResidualCrossAttentionBlock(nn.Module):
    """Cross-attention module between 2 inputs. """
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = False):
        super().__init__()

        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads=n_kv_heads, bias=add_bias_kv,
                                          dropout=dropout,  batch_first=batch_first)
        self.ln_1x = nn.LayerNorm(d_model)
        self.ln_1y = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None,
                  attn_mask: torch.Tensor = None):
        return self.attn(x, y, y, need_weights=False, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None,
                attn_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1x(x), self.ln_1y(y), key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, d_model: int, n_head: int, n_kv_heads: Optional[int] = None,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = False):
        super().__init__()
        self.attn = GroupedQueryAttention(d_model, n_head, n_kv_heads=n_kv_heads, bias=add_bias_kv,
                                          dropout=dropout,  batch_first=batch_first)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, attn_mask: torch.Tensor = None):
        return self.attn(x.clone(), x, x, need_weights=False,
                         key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None, attn_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class FusionTransformer(nn.Module):
    """Fusion of features from multiple modalities using attention.
    in_shape: (N, L1, E), (N, L2, E), out_shape: (N, E)
    We use either:
        - "concat": concatenation over tokens + self-attention module
        - "x-attn": cross-attention between two sets of tokens + concatenation over tokens
    An attention mask can be applied eventually for each modality with shape (N, Li) for modality i.
    """
    def __init__(self, width: int,
                 n_heads: int,
                 n_layers: int,
                 n_kv_heads: Optional[int] = None,
                 fusion: str = "concat",
                 pool: str = "cls",
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = True):
        """
        :param width: embedding size
        :param n_heads: number of heads in multi-head attention blocks
        :param n_layers: number of attention blocks
        :param fusion: "concat" or "x-attn"
        :param pool: "cls" or "pool"
        :param add_bias_kv: If specified, adds bias to the key and value sequences at dim=0.
        :param dropout: Dropout probability on `attn_output_weights`
        :param batch_first: input tensor is either (batch, tokens, features) if `True` or (tokens, batch, features)
        """
        super().__init__()

        self.fusion = fusion
        self.width = width
        self.layers = n_layers
        self.norm = nn.LayerNorm(width)
        self.token_dim = 1 if batch_first else 0
        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, width)) if self.pool == "cls" else None
        if fusion == "concat":
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock(width, n_heads, n_kv_heads=n_kv_heads, add_bias_kv=add_bias_kv,
                                       dropout=dropout, batch_first=batch_first)
                for _ in range(n_layers)])
        elif fusion == "x-attn":
            self.resblocks = [
                nn.Sequential(*[
                    ResidualCrossAttentionBlock(width, n_heads, n_kv_heads=n_kv_heads, add_bias_kv=add_bias_kv,
                                                dropout=dropout, batch_first=batch_first)
                    for _ in range(n_layers)])
                for _ in range(2)]
        else:
            raise ValueError("Unknown fusion %s" % fusion)
        self.initialize()

    def initialize(self):
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            # nn.init.normal_(block.attn.in_proj_weight, std=attn_std) # Removed as GQA uses separate q,k,v projs
            nn.init.normal_(block.attn.q_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.k_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.v_proj.weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self,
                x: List[torch.Tensor],
                key_padding_mask: List[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None):
        """
        :param x: input tensors
        :param key_padding_mask: torch mask of type bool. `True` indicates unattended tokens.
        :return:
        """
        # Concatenate over tokens + self-attention
        if self.fusion == "concat":
            x = torch.cat(x, dim=self.token_dim)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(key_padding_mask, dim=self.token_dim)
            if self.pool == "cls": # append cls token at the beginning
                cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
                x = torch.cat((cls_token, x), dim=self.token_dim)
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        (torch.zeros_like(cls_token[:, :, 0]), key_padding_mask), dim=self.token_dim)

            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.masked_fill(key_padding_mask.bool(), float("-inf")).float()
            if attn_mask is not None:
                if attn_mask.dim() != 2:
                    raise ValueError("`attn_mask` must be 2D (seq_len, seq_len).")
                
                if self.pool == "cls" and attn_mask.shape[0] == x.shape[self.token_dim] - 1:
                    attn_mask = torch.nn.functional.pad(attn_mask, (1, 0, 1, 0), value=0.0)

                if attn_mask.shape[0] != x.shape[self.token_dim] or attn_mask.shape[1] != x.shape[self.token_dim]:
                    raise ValueError(f"`attn_mask` size must match token dimension. Got {attn_mask.shape} but expected {x.shape[self.token_dim]}.")
                attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)

            for layer in self.resblocks:
                x = layer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

            x = self.norm(x)

            if self.pool == "cls":
                x = x[:, 0] if self.token_dim == 1 else x[0]
            elif self.pool == "mean":
                x = x.mean(dim=self.token_dim)
            elif self.pool == "tokens":
                return x
            else:
                raise ValueError(f"Unsupported pooling '{self.pool}'")
            return x
        # Cross-attention + concatenate over tokens
        elif self.fusion == "x-attn":
            if self.pool == "cls":
                raise ValueError("Only `mean` pool is implemented for cross-attention.")
            if len(x) != 2:
                raise ValueError("Only 2 modalities are currently accepted for cross-attention")
            if key_padding_mask is not None:
                raise NotImplementedError()
            if attn_mask is not None:
                raise NotImplementedError("Cross-attention fusion does not support `attn_mask` yet.")
            x1, x2 = x
            x = torch.cat([self.resblocks[0](x1, x2, key_padding_mask),
                           self.resblocks[1](x2, x1, key_padding_mask)], dim=self.token_dim)
            x = self.norm(x).mean(dim=self.token_dim)
            return x


class MMFusion(nn.Module):
    def __init__(self,
                 encoders: List[nn.Module],
                 input_adapters: List[nn.Module],
                 embed_dim: int = 512,
                 fusion: str = "concat",
                 pool: str = "cls",
                 n_heads: int = 8,
                 n_kv_heads: Optional[int] = None,
                 n_layers: int = 1,
                 add_bias_kv: bool = False,
                 dropout: float = 0.):
        """ Multi-Modal (MM) fusion model using `FusionTransformer` in the latent space.
        It can handle an arbitrary number of input modalities.
        Each modality is encoded through either a:
            - Transformer (e.g. for text or audio) -> no adapters
            - CNN (e.g. for images) -> `PatchedInputAdapter` for tokenization
            - MLP (e.g. tabular data) -> `FeaturesInputAdapter` for tokenization
        Once each modality is encoded and tokenized, it then goes to `FusionTransformer` to output
        the final embedding.

        :param encoders: List of Torch encoders (CNN, Transformer, MLP, etc.) for each modality
        :param input_adapters: List of Torch adapters for each modality (can be None if not required)
        :param embed_dim: Embedding size
        :param fusion: "concat" or "x-attn". For "x-attn", only "mean" pool is accepted.
        :param pool: "cls" or "mean", pooling strategy for the tokens
        :param n_heads: Number of heads in multi-heads attention blocks
        :param n_layers: Number of attention layers in latent fusion
        :param add_bias_kv: If `True`, add bias term in key/values mapping
        :param dropout: attention matrix dropout rate
        """
        super().__init__()
        assert len(encoders) == len(input_adapters), "Each encoder must have an adapter."
        assert pool in {'cls', 'mean', 'tokens'}, "pool type must be either cls, mean, or tokens"
        self.input_adapters = nn.ModuleList(input_adapters)
        self.encoders = nn.ModuleList(encoders)
        self.pool = pool
        self.num_modalities = len(self.encoders)
        self.adapter_collapses_batch = [
            getattr(adapter, "collapse_batch", False) for adapter in self.input_adapters
        ]
        self.fusion_transformer = FusionTransformer(embed_dim, n_heads, n_layers, n_kv_heads,
                                                    fusion, pool, add_bias_kv, dropout,
                                                    batch_first=True)

    def forward(self, x: List[torch.Tensor],
                mask_modalities: Optional[Union[List[bool], List[List[bool]]]] = None,
                attn_mask: Optional[torch.Tensor] = None):
        """
        :param x: List of tensors
        :param mask_modalities: Mask indicating which modalities are given.
            By default, `x` should have all modalities.
            If a list of lists is given, assume `x` has all modalities and computes
            a list of output by masking out modalitites according to `mask_modalities`.
        :param attn_mask: Optional attention bias matrix (either modality-level
            or token-level) to be passed to the fusion transformer.
        :return: a latent vector z or list of vector if `mask_modalities` is a list of list.
        """
        list_mask_mod = None
        if mask_modalities is None:
            mask_modalities = self.num_modalities * [True]
        elif isinstance(mask_modalities, list) and len(mask_modalities)>0 and isinstance(mask_modalities[0], list):
            list_mask_mod = mask_modalities
            mask_modalities = self.num_modalities * [True]

        assert len(mask_modalities) == self.num_modalities, (
            f"Mask size does not match `num_modalities`: {len(mask_modalities)} != {self.num_modalities}")

        orig_batch = x[0].shape[0] if len(x) > 0 else 0
        num_modalities = sum(mask_modalities)
        assert len(x) == num_modalities, (
                f"Incorrect number of inputs: {len(x)} != {num_modalities}")

        encoders = [enc for (enc, m) in zip(self.encoders, mask_modalities) if m]
        input_adapters = [adapter for (adapter, m) in zip(self.input_adapters, mask_modalities) if m]
        adapter_flags = [getattr(adapter, "collapse_batch", False) for adapter in input_adapters]
        # collapse_batch = len(adapter_flags) > 0 and all(adapter_flags)
        collapse_batch = True
        if collapse_batch and self.pool != "tokens":
            raise ValueError("Per-spot tokenization requires FusionTransformer(pool='tokens').")
        padding_masks = []

        # 1. Encode input modalities
        z = []
        for (enc, xi) in zip(encoders, x):
            embedding = enc(xi)
            attn_mask_ = None
            if isinstance(embedding, dict):  # attention mask must be considered
                attn_mask_ = embedding["attention_mask"]
                embedding = embedding["token_embeddings"]
            z.append(embedding)
            padding_masks.append(attn_mask_)

        # 2. Tokenize each latent features
        latent_tokens = [adapter(zi) if adapter is not None else zi
                         for (adapter, zi) in zip(input_adapters, z)]
        padding_masks = [attn_mask_ if attn_mask_ is not None else torch.zeros_like(zi[:, :, 0]).bool()
                         for (attn_mask_, zi) in zip(padding_masks, latent_tokens)]
        token_lengths = [lt.shape[1] for lt in latent_tokens]
        token_ranges = self._token_ranges(token_lengths)
        attn_mask_full = None
        if attn_mask is not None:
            attn_mask_full = self._prepare_attn_mask(attn_mask, token_lengths, latent_tokens[0].device)

        if list_mask_mod is None:
            # 3. FusionTransformer forward pass
            z = self.fusion_transformer(latent_tokens,
                                        key_padding_mask=padding_masks,
                                        attn_mask=attn_mask_full)
            if self.pool == "tokens" and collapse_batch:
                z = self._restore_batch_tokens(z, token_lengths, orig_batch)
        else:
            # 3.bis Drop modalities according to `mask_modalities`
            z = []
            for mask_mod in list_mask_mod:
                latent_tokens_ = [z for (z, m) in zip(latent_tokens, mask_mod) if m]
                padding_masks_ = [attn for (attn, m) in zip(padding_masks, mask_mod) if m]
                subset_mask = None
                if attn_mask_full is not None:
                    idx = self._indices_from_mask(mask_mod, token_ranges, attn_mask_full.device)
                    subset_mask = self._subset_attn_mask(attn_mask_full, idx)
                # 3. FusionTransformer forward pass
                z_mod = self.fusion_transformer(latent_tokens_,
                                                key_padding_mask=padding_masks_,
                                                attn_mask=subset_mask)
                token_lengths_subset = [tl for (tl, keep) in zip(token_lengths, mask_mod) if keep]
                if self.pool == "tokens" and collapse_batch:
                    z_mod = self._restore_batch_tokens(z_mod,
                                                       token_lengths_subset,
                                                       orig_batch)
                z.append(z_mod)
        return z

    def encode_single_mod(self, x: torch.Tensor, mod: int):
        assert 0 <= mod < self.num_modalities, "Wrong input modality"
        return self.encoders[mod](x)

    @staticmethod
    def _token_ranges(token_lengths: List[int]) -> List[tuple]:
        ranges = []
        cursor = 0
        for length in token_lengths:
            ranges.append((cursor, cursor + length))
            cursor += length
        return ranges

    @staticmethod
    def _prepare_attn_mask(attn_mask: torch.Tensor,
                           token_lengths: List[int],
                           device: torch.device) -> torch.Tensor:
        if attn_mask.dim() != 2:
            raise ValueError("`attn_mask` must be 2D.")
        attn_mask = attn_mask.to(device=device)
        total_tokens = sum(token_lengths)
        if attn_mask.shape[0] == len(token_lengths):
            expanded = torch.zeros(total_tokens, total_tokens,
                                   dtype=attn_mask.dtype, device=device)
            row_start = 0
            for i, len_i in enumerate(token_lengths):
                col_start = 0
                for j, len_j in enumerate(token_lengths):
                    expanded[row_start:row_start + len_i,
                             col_start:col_start + len_j] = attn_mask[i, j]
                    col_start += len_j
                row_start += len_i
            return expanded
        if attn_mask.shape[0] != total_tokens or attn_mask.shape[1] != total_tokens:
            raise ValueError("`attn_mask` size does not match token layout.")
        return attn_mask

    @staticmethod
    def _indices_from_mask(mask_mod: List[bool],
                           token_ranges: List[tuple],
                           device: torch.device) -> torch.Tensor:
        idx = []
        for keep, (start, end) in zip(mask_mod, token_ranges):
            if keep:
                idx.extend(range(start, end))
        if not idx:
            raise ValueError("At least one modality must be selected.")
        return torch.tensor(idx, device=device, dtype=torch.long)

    @staticmethod
    def _subset_attn_mask(attn_mask: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if indices.numel() == attn_mask.shape[0]:
            return attn_mask
        mask = attn_mask.index_select(0, indices)
        mask = mask.index_select(1, indices)
        return mask

    @staticmethod
    def _restore_batch_tokens(tokens: torch.Tensor,
                              token_lengths: List[int],
                              batch_size: int) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError("Expected token tensor with shape (batch, tokens, dim).")
        tokens = tokens.squeeze(0)  # assume collapsed batch -> size 1
        total_tokens = sum(token_lengths)
        if tokens.shape[0] != total_tokens:
            raise ValueError("Token lengths do not match fused token count.")
        modalities = len(token_lengths)
        embedding_dim = tokens.shape[-1]
        tokens = tokens.view(modalities, batch_size, embedding_dim)
        return tokens.mean(dim=0)

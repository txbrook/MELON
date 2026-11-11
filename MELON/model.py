import torch
import torch.nn as nn

class SpotTokenAdapter(nn.Module):
    def __init__(self, collapse_batch=True):
        super().__init__()
        self.collapse_batch = bool(collapse_batch)

    def forward(self, x):
        # collapse_batch=True:  (N, E) -> (1, N, E) for per-spot tokenization
        # collapse_batch=False: (N, E) -> (N, 1, E) for one token per modality
        return x.unsqueeze(0) if self.collapse_batch else x.unsqueeze(1)


def make_model(dimensions, config):
    from types import MethodType
    from .models.mmfusion import MMFusion
    from .pl_modules.melon import MELONModel
    from .losses.melon_loss import SymmetricMELONLoss
    embed_dim = config.get("embed_dim", 128)
    spatial_k = config.get("training_spatial_k", 16)
    encoder = MMFusion(
        encoders=[nn.Sequential(nn.Linear(d, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)) for d in dimensions],
        input_adapters=[SpotTokenAdapter() for _ in dimensions],
        embed_dim=embed_dim, fusion="concat", pool="tokens", n_heads=8, n_layers=1,
    )
    model = MELONModel(
        encoder=encoder, projection=MELONModel._build_mlp(embed_dim, 256, embed_dim),
        optim_kwargs={"lr": config.get("learning_rate", .001), "weight_decay": config.get("weight_decay", 1e-5)},
        loss_kwargs={"temperature": config.get("temperature", .1)},
        neighborhood_cfg={"hidden_dim": 64, "heads": 8, "dropout": .1,
                          "k_neighbors": spatial_k, "include_self": True},
    )
    if config["historical_five_modal"]:
        model.loss = SymmetricMELONLoss(temperature=config.get("temperature", .1))
    else:
        # Supplementary Table 9 specifies Adam; the archived five-modal run used AdamW.
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), **self.optim_kwargs, betas=(.9, .999))
        model.configure_optimizers = MethodType(configure_optimizers, model)
    return model

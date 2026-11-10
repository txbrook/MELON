import torch.nn as nn

class SpotTokenAdapter(nn.Module):
    def __init__(self, collapse_batch=True):
        super().__init__()
        self.collapse_batch = bool(collapse_batch)

    def forward(self, x):
        # collapse_batch=True:  (N, E) -> (1, N, E) for per-spot tokenization
        # collapse_batch=False: (N, E) -> (N, 1, E) for one token per modality
        return x.unsqueeze(0) if self.collapse_batch else x.unsqueeze(1)

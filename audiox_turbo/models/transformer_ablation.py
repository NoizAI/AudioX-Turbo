import torch

from .transformer import ContinuousMMDiTTransformer


class ContinuousMMDiTTransformerAblation(ContinuousMMDiTTransformer):
    """
    Ablation wrapper for partial MMDiT backbone usage.
    """

    def forward(self, *args, active_num_layers=None, **kwargs):
        if active_num_layers is None:
            return super().forward(*args, **kwargs)

        total_layers = len(self.layers)
        n = max(1, min(int(active_num_layers), total_layers))
        original_layers = self.layers
        try:
            self.layers = torch.nn.ModuleList(list(original_layers)[:n])
            return super().forward(*args, **kwargs)
        finally:
            self.layers = original_layers


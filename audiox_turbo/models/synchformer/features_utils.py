from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from .synchformer import Synchformer


class FeaturesUtils(nn.Module):
    """Synchformer-based video feature extractor.

    Used at inference time to turn a 224x224 / 25fps video clip into the
    synchronization features consumed by the video conditioner. Only the
    Synchformer path is needed; audio/VAE branches are intentionally omitted.
    """

    def __init__(self, *, synchformer_ckpt: Optional[str] = None,
                 enable_conditions: bool = True, **kwargs):
        super().__init__()
        if enable_conditions:
            self.synchformer = Synchformer()
            self.synchformer.load_state_dict(
                torch.load(synchformer_ckpt, weights_only=True, map_location="cpu"))
        else:
            self.synchformer = None

    def train(self, mode: bool) -> None:
        # Always keep the feature extractor in eval mode.
        return super().train(False)

    @torch.inference_mode()
    def encode_video_with_sync(self, x: torch.Tensor, batch_size: int = -1) -> torch.Tensor:
        assert self.synchformer is not None, "Synchformer is not loaded"
        # x: (B, T, C, H, W), expects H = W = 224
        b, t, c, h, w = x.shape
        assert c == 3 and h == 224 and w == 224

        # Partition the video into overlapping 16-frame segments (stride 8).
        segment_size = 16
        step_size = 8
        num_segments = (t - segment_size) // step_size + 1
        segments = [x[:, i * step_size:i * step_size + segment_size] for i in range(num_segments)]
        x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

        if batch_size < 0:
            batch_size = b
        x = rearrange(x, "b s t c h w -> (b s) 1 t c h w")
        outputs = []
        for i in range(0, b * num_segments, batch_size):
            outputs.append(self.synchformer(x[i:i + batch_size]))
        x = torch.cat(outputs, dim=0)
        x = rearrange(x, "(b s) 1 t d -> b (s t) d", b=b)
        return x

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

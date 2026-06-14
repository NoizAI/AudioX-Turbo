from __future__ import annotations
from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

# from x_transformers.attend import Attend

# from .blocks import (RMSNorm, FeedForward)
# from mmdit.mmdit_pytorch import (
#     JointAttention
# )

# from hyper_connections import (
#     HyperConnections,
#     Residual
# )


import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))
        self.scale = dim ** 0.5

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.g * self.scale


# Faked placeholder for FeedForward as its definition is not provided.
# You should replace this with the actual implementation of your FeedForward block.
class FeedForward(nn.Module):
    def __init__(self, dim, **kwargs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    def forward(self, x):
        return self.net(x)

# Start of integrated code from x_transformers.attend
class Attend(nn.Module):
    def __init__(self,
                 flash = False,
                 softclamp_logits = False,
                 logit_softclamp_value = 50.):
        super().__init__()
        self.flash = flash and hasattr(F, 'scaled_dot_product_attention')
        self.softclamp_logits = softclamp_logits
        self.logit_softclamp_value = logit_softclamp_value

    def forward(self, q, k, v, mask = None):
        if self.flash:
            # Use torch's built-in flash attention
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False), None

        scale = q.shape[-1] ** -0.5
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * scale

        if self.softclamp_logits:
            sim = softclamp(sim, self.logit_softclamp_value)

        if mask is not None:
            mask = rearrange(mask, 'b j -> b 1 1 j')
            mask_value = -torch.finfo(sim.dtype).max
            sim = sim.masked_fill(~mask, mask_value)

        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        return out, attn
# End of integrated code from x_transformers.attend


# Start of integrated code from hyper_connections
class Residual(Module):
    """
    A simple residual connection implementation that fits the calling pattern in MMDiTBlock.
    It takes an input `x` and returns `x` along with a function to add the residual back later.
    """
    def __init__(self, num_streams: int, dim: int):
        super().__init__()
        # num_streams and dim are unused in this simple case but kept for API consistency
        # with HyperConnections.

    def forward(self, x: Tensor) -> Tuple[Tensor, Callable[[Tensor], Tensor]]:
        def apply_residual(output: Tensor) -> Tensor:
            return output + x
        return x, apply_residual

class HyperConnections(Module):
    """
    A placeholder for HyperConnections that matches the required API.
    The actual implementation might be more complex, involving learnable parameters
    for combining streams, but this placeholder ensures the code runs.
    """
    def __init__(self, num_streams: int, dim: int):
        super().__init__()
        # This is a functional placeholder. The true implementation of HyperConnections
        # would likely use num_streams and dim to create learnable parameters.
        # For now, it behaves like a simple Residual connection.

    def forward(self, x: Tensor) -> Tuple[Tensor, Callable[[Tensor], Tensor]]:
        def apply_residual(output: Tensor) -> Tensor:
            return output + x
        return x, apply_residual

    @staticmethod
    def get_expand_reduce_stream_functions(num_streams: int, disable: bool = False) -> Tuple[Callable, Callable]:
        if disable or num_streams <= 1:
            return nn.Identity(), nn.Identity()

        # Note: The exact implementation of stream expansion/reduction can vary.
        # This is a plausible implementation.
        def expand_streams(x: Tensor) -> Tensor:
            return x.unsqueeze(-1).repeat(1, 1, 1, num_streams)

        def reduce_streams(x: Tensor) -> Tensor:
            return x.mean(dim=-1)

        return expand_streams, reduce_streams
# End of integrated code from hyper_connections














from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# from x_transformers.attend import Attend
# from x_transformers import (
#     RMSNorm,
#     FeedForward
# )

# from hyper_connections import (
#     HyperConnections,
#     Residual
# )

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def softclamp(t, value):
    return (t / value).tanh() * value

# rmsnorm

class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale

# attention

class JointAttention(Module):
    def __init__(
        self,
        *,
        dim_inputs: tuple[int, ...],
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash = False,
        softclamp = False,
        softclamp_value = 50.,
        attend_kwargs: dict = dict()
    ):
        super().__init__()
        """
        ein notation

        b - batch
        h - heads
        n - sequence
        d - feature dimension
        """

        dim_inner = dim_head * heads

        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        self.attend = Attend(
            flash = flash,
            softclamp_logits = softclamp,
            logit_softclamp_value = softclamp_value,
            **attend_kwargs
        )

        self.merge_heads = Rearrange('b h n d -> b n (h d)')

        self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

        self.qk_rmsnorm = qk_rmsnorm
        self.q_rmsnorms = (None,) * num_inputs
        self.k_rmsnorms = (None,) * num_inputs

        if qk_rmsnorm:
            self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
            self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])

        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    def forward(
        self,
        inputs: tuple[Tensor],
        masks: tuple[Tensor | None] | None = None
    ):

        device = self.dummy.device

        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        # project each modality separately for qkv
        # also handle masks, assume None means attend to all tokens

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):

            qkv = to_qkv(x)
            qkv = self.split_heads(qkv)

            # optional qk rmsnorm per modality

            if self.qk_rmsnorm:
                q, k, v = qkv
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            # handle mask per modality

            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

            all_masks.append(mask)

        # combine all qkv and masks

        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        # attention

        q, k, v = all_qkvs

        outs, *_ = self.attend(q, k, v, mask = all_masks)

        # merge heads and then separate by modality for combine heads projection

        outs = self.merge_heads(outs)
        outs = unpack(outs, packed_shape, 'b * d')

        # separate combination of heads for each modality

        all_outs = []

        for out, to_out in zip(outs, self.to_out):
            out = to_out(out)
            all_outs.append(out)

        return tuple(all_outs)

# class

class MMDiTBlock(Module):
    def __init__(
        self,
        *,
        dim_text,
        dim_image,
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        qk_rmsnorm = False,
        flash_attn = False,
        num_residual_streams = 1,
        ff_kwargs: dict = dict()
    ):
        super().__init__()

        # residual functions / maybe hyper connections

        residual_klass = Residual if num_residual_streams == 1 else HyperConnections

        self.text_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_text)
        self.text_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_text)

        self.image_attn_residual_fn = residual_klass(num_residual_streams, dim = dim_image)
        self.image_ff_residual_fn = residual_klass(num_residual_streams, dim = dim_image)

        # handle optional time conditioning

        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            dim_gammas = (
                *((dim_text,) * 4),
                *((dim_image,) * 4)
            )

            dim_betas = (
                *((dim_text,) * 2),
                *((dim_image,) * 2),
            )

            self.cond_dims = (*dim_gammas, *dim_betas)

            to_cond_linear = nn.Linear(dim_cond, sum(self.cond_dims))

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                to_cond_linear
            )

            nn.init.zeros_(to_cond_linear.weight)
            nn.init.zeros_(to_cond_linear.bias)
            nn.init.constant_(to_cond_linear.bias[:sum(dim_gammas)], 1.)

        # handle adaptive norms

        self.text_attn_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_attn_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

        self.text_ff_layernorm = nn.LayerNorm(dim_text, elementwise_affine = not has_cond)
        self.image_ff_layernorm = nn.LayerNorm(dim_image, elementwise_affine = not has_cond)

        # attention and feedforward

        self.joint_attn = JointAttention(
            dim_inputs = (dim_text, dim_image),
            dim_head = dim_head,
            heads = heads,
            flash = flash_attn
        )

        self.text_ff = FeedForward(dim_text, **ff_kwargs)
        self.image_ff = FeedForward(dim_image, **ff_kwargs)

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        text_mask = None,
        time_cond = None,
        skip_feedforward_text_tokens = True
    ):
        assert not (exists(time_cond) ^ self.has_cond), 'time condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        if self.has_cond:
            (
                text_pre_attn_gamma,
                text_post_attn_gamma,
                text_pre_ff_gamma,
                text_post_ff_gamma,
                image_pre_attn_gamma,
                image_post_attn_gamma,
                image_pre_ff_gamma,
                image_post_ff_gamma,
                text_pre_attn_beta,
                text_pre_ff_beta,
                image_pre_attn_beta,
                image_pre_ff_beta,
            ) = self.to_cond(time_cond).split(self.cond_dims, dim = -1)

        # handle attn adaptive layernorm

        text_tokens, add_text_residual = self.text_attn_residual_fn(text_tokens)
        image_tokens, add_image_residual = self.image_attn_residual_fn(image_tokens)

        text_tokens = self.text_attn_layernorm(text_tokens)
        image_tokens = self.image_attn_layernorm(image_tokens)

        if self.has_cond:
            text_tokens = text_tokens * text_pre_attn_gamma + text_pre_attn_beta
            image_tokens = image_tokens * image_pre_attn_gamma + image_pre_attn_beta

        # attention

        text_tokens, image_tokens = self.joint_attn(
            inputs = (text_tokens, image_tokens),
            masks = (text_mask, None)
        )

        # condition attention output

        if self.has_cond:
            text_tokens = text_tokens * text_post_attn_gamma
            image_tokens = image_tokens * image_post_attn_gamma

        # add attention residual

        text_tokens = add_text_residual(text_tokens)
        image_tokens = add_image_residual(image_tokens)

        # handle feedforward adaptive layernorm

        if not skip_feedforward_text_tokens:
            text_tokens, add_text_residual = self.text_ff_residual_fn(text_tokens)
            text_tokens = self.text_ff_layernorm(text_tokens)

            if self.has_cond:
                text_tokens = text_tokens * text_pre_ff_gamma + text_pre_ff_beta

        image_tokens, add_image_residual = self.image_ff_residual_fn(image_tokens)
        image_tokens = self.image_ff_layernorm(image_tokens)

        if self.has_cond:
            image_tokens = image_tokens * image_pre_ff_gamma + image_pre_ff_beta

        # images feedforward

        image_tokens = self.image_ff(image_tokens)

        # images condition feedforward output

        if self.has_cond:
            image_tokens = image_tokens * image_post_ff_gamma

        # images feedforward residual

        image_tokens = add_image_residual(image_tokens)

        # early return, for last block in mmdit

        if skip_feedforward_text_tokens:
            return text_tokens, image_tokens

        # text feedforward

        text_tokens = self.text_ff(text_tokens)

        # text condition feedforward output

        if self.has_cond:
            text_tokens = text_tokens * text_post_ff_gamma

        # text feedforward residual

        text_tokens = add_text_residual(text_tokens)

        # return

        return text_tokens, image_tokens

# mm dit transformer - simply many blocks

class MMDiT(Module):
    def __init__(
        self,
        *,
        depth,
        dim_image,
        num_register_tokens = 0,
        final_norm = True,
        num_residual_streams = 4,
        **block_kwargs
    ):
        super().__init__()

        self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.has_register_tokens = num_register_tokens > 0
        self.register_tokens = nn.Parameter(torch.zeros(num_register_tokens, dim_image))
        nn.init.normal_(self.register_tokens, std = 0.02)

        self.blocks = ModuleList([])

        for _ in range(depth):
            block = MMDiTBlock(
                dim_image = dim_image,
                num_residual_streams = num_residual_streams,
                **block_kwargs
            )

            self.blocks.append(block)

        self.norm = RMSNorm(dim_image) if final_norm else nn.Identity()

    def forward(
        self,
        *,
        text_tokens,
        image_tokens,
        text_mask = None,
        time_cond = None,
        should_skip_last_feedforward = True
    ):

        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b = image_tokens.shape[0])
            image_tokens, packed_shape = pack([register_tokens, image_tokens], 'b * d')

        text_tokens = self.expand_streams(text_tokens)
        image_tokens = self.expand_streams(image_tokens)

        for ind, block in enumerate(self.blocks):
            is_last = ind == (len(self.blocks) - 1)

            text_tokens, image_tokens = block(
                time_cond = time_cond,
                text_tokens = text_tokens,
                image_tokens = image_tokens,
                text_mask = text_mask,
                skip_feedforward_text_tokens = is_last and should_skip_last_feedforward
            )

        if self.has_register_tokens:
            _, image_tokens = unpack(image_tokens, packed_shape, 'b * d')

        text_tokens = self.reduce_streams(text_tokens)
        image_tokens = self.reduce_streams(image_tokens)

        image_tokens = self.norm(image_tokens)

        return text_tokens, image_tokens


# # helpers

# def exists(v):
#     return v is not None

# def default(v, d):
#     return v if exists(v) else d

# # adaptive layernorm
# # aim for clarity in generalized version

# class AdaptiveLayerNorm(Module):
#     def __init__(
#         self,
#         dim,
#         dim_cond = None
#     ):
#         super().__init__()
#         has_cond = exists(dim_cond)
#         self.has_cond = has_cond

#         self.ln = nn.LayerNorm(dim, elementwise_affine = not has_cond)
 
#         if has_cond:
#             cond_linear = nn.Linear(dim_cond, dim * 2)

#             self.to_cond = nn.Sequential(
#                 Rearrange('b d -> b 1 d'),
#                 nn.SiLU(),
#                 cond_linear
#             )

#             nn.init.zeros_(cond_linear.weight)

#             nn.init.constant_(cond_linear.bias[:dim], 1.)
#             nn.init.zeros_(cond_linear.bias[dim:])

#     def forward(
#         self,
#         x,
#         cond = None
#     ):
#         assert not (exists(cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

#         x = self.ln(x)

#         if self.has_cond:
#             gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
#             x = x * gamma + beta

#         return x

# # class

# def softclamp(t, value):
#     return (t / value).tanh() * value



# class MultiHeadRMSNorm(Module):
#     def __init__(self, dim, heads = 1):
#         super().__init__()
#         self.scale = dim ** 0.5
#         self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

#     def forward(self, x):
#         return F.normalize(x, dim = -1) * self.gamma * self.scale


# class JointAttention(Module):
#     def __init__(
#         self,
#         *,
#         dim_inputs: tuple[int, ...],
#         dim_head = 64,
#         heads = 8,
#         qk_rmsnorm = False,
#         flash = False,
#         softclamp = False,
#         softclamp_value = 50.,
#         attend_kwargs: dict = dict()
#     ):
#         super().__init__()
#         """
#         ein notation

#         b - batch
#         h - heads
#         n - sequence
#         d - feature dimension
#         """

#         dim_inner = dim_head * heads

#         num_inputs = len(dim_inputs)
#         self.num_inputs = num_inputs

#         self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])

#         self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

#         self.attend = Attend(
#             flash = flash,
#             softclamp_logits = softclamp,
#             logit_softclamp_value = softclamp_value,
#             **attend_kwargs
#         )

#         self.merge_heads = Rearrange('b h n d -> b n (h d)')

#         self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

#         self.qk_rmsnorm = qk_rmsnorm
#         self.q_rmsnorms = (None,) * num_inputs
#         self.k_rmsnorms = (None,) * num_inputs

#         if qk_rmsnorm:
#             self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
#             self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])

#         self.register_buffer('dummy', torch.tensor(0), persistent = False)

#     def forward(
#         self,
#         inputs: tuple[Tensor],
#         masks: tuple[Tensor | None] | None = None
#     ):

#         device = self.dummy.device

#         assert len(inputs) == self.num_inputs

#         masks = default(masks, (None,) * self.num_inputs)

#         # project each modality separately for qkv
#         # also handle masks, assume None means attend to all tokens

#         all_qkvs = []
#         all_masks = []

#         for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):

#             qkv = to_qkv(x)
#             qkv = self.split_heads(qkv)

#             # optional qk rmsnorm per modality

#             if self.qk_rmsnorm:
#                 q, k, v = qkv
#                 q = q_rmsnorm(q)
#                 k = k_rmsnorm(k)
#                 qkv = torch.stack((q, k, v))

#             all_qkvs.append(qkv)

#             # handle mask per modality

#             if not exists(mask):
#                 mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

#             all_masks.append(mask)

#         # combine all qkv and masks

#         all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
#         all_masks, _ = pack(all_masks, 'b *')

#         # attention

#         q, k, v = all_qkvs

#         outs, *_ = self.attend(q, k, v, mask = all_masks)

#         # merge heads and then separate by modality for combine heads projection

#         outs = self.merge_heads(outs)
#         outs = unpack(outs, packed_shape, 'b * d')

#         # separate combination of heads for each modality

#         all_outs = []

#         for out, to_out in zip(outs, self.to_out):
#             out = to_out(out)
#             all_outs.append(out)

#         return tuple(all_outs)



# class MMDiTBlock(Module):
#     def __init__(
#         self,
#         *,
#         dim_modalities: tuple[int, ...],
#         dim_cond = None,
#         dim_head = 64,
#         heads = 8,
#         qk_rmsnorm = False,
#         flash_attn = False,
#         softclamp = False,
#         softclamp_value = 50.,
#         num_residual_streams = 1,
#         ff_kwargs: dict = dict()
#     ):
#         super().__init__()
#         self.num_modalities = len(dim_modalities)
#         self.dim_modalities = dim_modalities

#         # residuals / maybe hyper connections

#         residual_klass = Residual if num_residual_streams == 1 else HyperConnections

#         self.attn_residual_fns = ModuleList([residual_klass(num_residual_streams, dim = dim) for dim in dim_modalities])
#         self.ff_residual_fns = ModuleList([residual_klass(num_residual_streams, dim = dim) for dim in dim_modalities])

#         # handle optional time conditioning

#         has_cond = exists(dim_cond)
#         self.has_cond = has_cond

#         if has_cond:
#             cond_linear = nn.Linear(dim_cond, sum(dim_modalities) * 2)

#             self.to_post_branch_gammas = nn.Sequential(
#                 Rearrange('b d -> b 1 d'),
#                 nn.SiLU(),
#                 cond_linear
#             )

#             nn.init.zeros_(cond_linear.weight)
#             nn.init.constant_(cond_linear.bias, 1.)

#         # joint modality attention

#         attention_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities]
#         self.attn_layernorms = ModuleList(attention_layernorms)

#         self.joint_attn = JointAttention(
#             dim_inputs = dim_modalities,
#             dim_head = dim_head,
#             heads = heads,
#             flash = flash_attn,
#             softclamp = softclamp,
#             softclamp_value = softclamp_value,
#         )

#         # feedforwards

#         feedforward_layernorms = [AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities]
#         self.ff_layernorms = ModuleList(feedforward_layernorms)

#         feedforwards = [FeedForward(dim, **ff_kwargs) for dim in dim_modalities]
#         self.feedforwards = ModuleList(feedforwards)

#     def forward(
#         self,
#         *,
#         modality_tokens: tuple[Tensor, ...],
#         modality_masks: tuple[Tensor | None, ...] | None = None,
#         time_cond = None
#     ):
#         import pdb; pdb.set_trace()
#         assert len(modality_tokens) == self.num_modalities
#         assert not (exists(time_cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

#         ln_kwargs = dict()

#         if self.has_cond:
#             ln_kwargs = dict(cond = time_cond)

#             gammas = self.to_post_branch_gammas(time_cond)
#             attn_gammas, ff_gammas = gammas.chunk(2, dim = -1)

#         # attention layernorms

#         modality_tokens, modality_tokens_residual_fns = tuple(zip(*[residual_fn(modality_token) for residual_fn, modality_token in zip(self.attn_residual_fns, modality_tokens)]))

#         modality_tokens = [ln(tokens, **ln_kwargs) for tokens, ln in zip(modality_tokens, self.attn_layernorms)]

#         # attention

#         modality_tokens = self.joint_attn(inputs = modality_tokens, masks = modality_masks)

#         # post attention gammas

#         if self.has_cond:
#             attn_gammas = attn_gammas.split(self.dim_modalities, dim = -1)
#             modality_tokens = [(tokens * g) for tokens, g in zip(modality_tokens, attn_gammas)]

#         # add attention residual

#         modality_tokens = [add_attn_residual(tokens) for add_attn_residual, tokens in zip(modality_tokens_residual_fns, modality_tokens)]

#         # handle feedforward adaptive layernorm

#         modality_tokens, modality_tokens_residual_fns = tuple(zip(*[residual_fn(modality_token) for residual_fn, modality_token in zip(self.ff_residual_fns, modality_tokens)]))

#         modality_tokens = [ln(tokens, **ln_kwargs) for tokens, ln in zip(modality_tokens, self.ff_layernorms)]

#         modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]

#         # post feedforward gammas

#         if self.has_cond:
#             ff_gammas = ff_gammas.split(self.dim_modalities, dim = -1)
#             modality_tokens = [(tokens * g) for tokens, g in zip(modality_tokens, ff_gammas)]

#         # add feedforward residual

#         modality_tokens = [add_residual_fn(tokens) for add_residual_fn, tokens in zip(modality_tokens_residual_fns, modality_tokens)]

#         # returns

#         return modality_tokens

# # mm dit transformer - simply many blocks

# class MMDiT(Module):
#     def __init__(
#         self,
#         *,
#         depth,
#         dim_modalities,
#         final_norms = True,
#         num_residual_streams = 4,
#         **block_kwargs
#     ):
#         super().__init__()

#         self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

#         blocks = [MMDiTBlock(dim_modalities = dim_modalities, num_residual_streams = num_residual_streams, **block_kwargs) for _ in range(depth)]
#         self.blocks = ModuleList(blocks)

#         norms = [RMSNorm(dim) for dim in dim_modalities]
#         self.norms = ModuleList(norms)

#     def forward(
#         self,
#         *,
#         modality_tokens: tuple[Tensor, ...],
#         modality_masks: tuple[Tensor | None, ...] | None = None,
#         time_cond = None
#     ):

#         modality_tokens = [self.expand_streams(modality) for modality in modality_tokens]

#         for block in self.blocks:
#             modality_tokens = block(
#                 time_cond = time_cond,
#                 modality_tokens = modality_tokens,
#                 modality_masks = modality_masks
#             )

#         modality_tokens = [self.reduce_streams(modality) for modality in modality_tokens]

#         modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]

#         return tuple(modality_tokens)



# # from __future__ import annotations
# # from typing import Tuple, Callable

# # import torch
# # from torch import nn
# # from torch import Tensor
# # import torch.nn.functional as F
# # from torch.nn import Module, ModuleList

# # from einops import rearrange, pack, unpack
# # from einops.layers.torch import Rearrange

# # # Faked placeholder for FeedForward as its definition is not provided.
# # # You should replace this with the actual implementation of your FeedForward block.
# # class FeedForward(nn.Module):
# #     def __init__(self, dim, **kwargs):
# #         super().__init__()
# #         self.net = nn.Sequential(
# #             nn.Linear(dim, dim * 4),
# #             nn.GELU(),
# #             nn.Linear(dim * 4, dim)
# #         )
# #     def forward(self, x):
# #         return self.net(x)

# # # Start of integrated code from x_transformers.attend
# # class Attend(nn.Module):
# #     def __init__(self,
# #                  flash = False,
# #                  softclamp_logits = False,
# #                  logit_softclamp_value = 50.):
# #         super().__init__()
# #         self.flash = flash and hasattr(F, 'scaled_dot_product_attention')
# #         self.softclamp_logits = softclamp_logits
# #         self.logit_softclamp_value = logit_softclamp_value

# #     def forward(self, q, k, v, mask = None):
# #         if self.flash:
# #             # Use torch's built-in flash attention
# #             return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False), None

# #         scale = q.shape[-1] ** -0.5
# #         sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * scale

# #         if self.softclamp_logits:
# #             sim = softclamp(sim, self.logit_softclamp_value)

# #         if mask is not None:
# #             mask = rearrange(mask, 'b j -> b 1 1 j')
# #             mask_value = -torch.finfo(sim.dtype).max
# #             sim = sim.masked_fill(~mask, mask_value)

# #         attn = sim.softmax(dim=-1)
# #         out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

# #         return out, attn
# # # End of integrated code from x_transformers.attend


# # # Start of integrated code from hyper_connections
# # class Residual(Module):
# #     """
# #     A simple residual connection implementation that fits the calling pattern in MMDiTBlock.
# #     It takes an input `x` and returns `x` along with a function to add the residual back later.
# #     """
# #     def __init__(self, num_streams: int, dim: int):
# #         super().__init__()
# #         # num_streams and dim are unused in this simple case but kept for API consistency
# #         # with HyperConnections.

# #     def forward(self, x: Tensor) -> Tuple[Tensor, Callable[[Tensor], Tensor]]:
# #         def apply_residual(output: Tensor) -> Tensor:
# #             return output + x
# #         return x, apply_residual

# # class HyperConnections(Module):
# #     """
# #     A placeholder for HyperConnections that matches the required API.
# #     The actual implementation might be more complex, involving learnable parameters
# #     for combining streams, but this placeholder ensures the code runs.
# #     """
# #     def __init__(self, num_streams: int, dim: int):
# #         super().__init__()
# #         # This is a functional placeholder. The true implementation of HyperConnections
# #         # would likely use num_streams and dim to create learnable parameters.
# #         # For now, it behaves like a simple Residual connection.

# #     def forward(self, x: Tensor) -> Tuple[Tensor, Callable[[Tensor], Tensor]]:
# #         def apply_residual(output: Tensor) -> Tensor:
# #             return output + x
# #         return x, apply_residual

# #     @staticmethod
# #     def get_expand_reduce_stream_functions(num_streams: int, disable: bool = False) -> Tuple[Callable, Callable]:
# #         if disable or num_streams <= 1:
# #             return nn.Identity(), nn.Identity()

# #         # Note: The exact implementation of stream expansion/reduction can vary.
# #         # This is a plausible implementation.
# #         def expand_streams(x: Tensor) -> Tensor:
# #             return x.unsqueeze(-1).repeat(1, 1, 1, num_streams)

# #         def reduce_streams(x: Tensor) -> Tensor:
# #             return x.mean(dim=-1)

# #         return expand_streams, reduce_streams
# # # End of integrated code from hyper_connections


# # # helpers

# # def exists(v):
# #     return v is not None

# # def default(v, d):
# #     return v if exists(v) else d

# # # adaptive layernorm
# # # aim for clarity in generalized version

# # class AdaptiveLayerNorm(Module):
# #     def __init__(
# #         self,
# #         dim,
# #         dim_cond = None
# #     ):
# #         super().__init__()
# #         has_cond = exists(dim_cond)
# #         self.has_cond = has_cond

# #         self.ln = nn.LayerNorm(dim, elementwise_affine = not has_cond)

# #         if has_cond:
# #             cond_linear = nn.Linear(dim_cond, dim * 2)

# #             self.to_cond = nn.Sequential(
# #                 nn.SiLU(),
# #                 cond_linear,
# #                 Rearrange('b d -> b 1 d'),
# #             )

# #             nn.init.zeros_(cond_linear.weight)
# #             nn.init.constant_(cond_linear.bias[:dim], 1.)
# #             nn.init.zeros_(cond_linear.bias[dim:])

# #     def forward(
# #         self,
# #         x,
# #         cond = None
# #     ):
# #         assert not (exists(cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

# #         x = self.ln(x)

# #         if self.has_cond:
# #             gamma, beta = self.to_cond(cond).chunk(2, dim = -1)
# #             x = x * gamma + beta

# #         return x

# # # class

# # def softclamp(t, value):
# #     return (t / value).tanh() * value


# # class RMSNorm(nn.Module):
# #     def __init__(self, dim):
# #         super().__init__()
# #         self.g = nn.Parameter(torch.ones(dim))
# #         self.scale = dim ** 0.5

# #     def forward(self, x):
# #         return F.normalize(x, dim = -1) * self.g * self.scale


# # class MultiHeadRMSNorm(Module):
# #     def __init__(self, dim, heads = 1):
# #         super().__init__()
# #         self.scale = dim ** 0.5
# #         self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

# #     def forward(self, x):
# #         return F.normalize(x, dim = -1) * self.gamma * self.scale


# # class JointAttention(Module):
# #     def __init__(
# #         self,
# #         *,
# #         dim_inputs: tuple[int, ...],
# #         dim_head = 64,
# #         heads = 8,
# #         qk_rmsnorm = False,
# #         flash = False,
# #         softclamp = False,
# #         softclamp_value = 50.,
# #         attend_kwargs: dict = dict()
# #     ):
# #         super().__init__()
# #         dim_inner = dim_head * heads
# #         num_inputs = len(dim_inputs)
# #         self.num_inputs = num_inputs

# #         self.to_qkv = ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs])
# #         self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

# #         self.attend = Attend(
# #             flash = flash,
# #             softclamp_logits = softclamp,
# #             logit_softclamp_value = softclamp_value,
# #             **attend_kwargs
# #         )

# #         self.merge_heads = Rearrange('b h n d -> b n (h d)')
# #         self.to_out = ModuleList([nn.Linear(dim_inner, dim_input, bias = False) for dim_input in dim_inputs])

# #         self.qk_rmsnorm = qk_rmsnorm
# #         if qk_rmsnorm:
# #             self.q_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
# #             self.k_rmsnorms = ModuleList([MultiHeadRMSNorm(dim_head, heads = heads) for _ in range(num_inputs)])
# #         else:
# #             self.q_rmsnorms = (None,) * num_inputs
# #             self.k_rmsnorms = (None,) * num_inputs

# #         self.register_buffer('dummy', torch.tensor(0), persistent = False)

# #     def forward(
# #         self,
# #         inputs: tuple[Tensor],
# #         masks: tuple[Tensor | None] | None = None
# #     ):
# #         device = self.dummy.device
# #         assert len(inputs) == self.num_inputs
# #         masks = default(masks, (None,) * self.num_inputs)

# #         all_qkvs = []
# #         all_masks = []

# #         for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.to_qkv, self.q_rmsnorms, self.k_rmsnorms):
# #             qkv = to_qkv(x)
# #             qkv = self.split_heads(qkv)

# #             if self.qk_rmsnorm:
# #                 q, k, v = qkv
# #                 q = q_rmsnorm(q)
# #                 k = k_rmsnorm(k)
# #                 qkv = torch.stack((q, k, v))
# #             all_qkvs.append(qkv)

# #             if not exists(mask):
# #                 mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)
# #             all_masks.append(mask)

# #         all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
# #         all_masks, _ = pack(all_masks, 'b *')

# #         q, k, v = all_qkvs
# #         outs, *_ = self.attend(q, k, v, mask = all_masks)

# #         outs = self.merge_heads(outs)
# #         outs = unpack(outs, packed_shape, 'b * d')

# #         all_outs = [to_out(out) for out, to_out in zip(outs, self.to_out)]
# #         return tuple(all_outs)


# # class MMDiTBlock(Module):
# #     def __init__(
# #         self,
# #         *,
# #         dim_modalities: tuple[int, ...],
# #         dim_cond = None,
# #         dim_head = 64,
# #         heads = 8,
# #         qk_rmsnorm = False,
# #         flash_attn = False,
# #         softclamp = False,
# #         softclamp_value = 50.,
# #         num_residual_streams = 1,
# #         ff_kwargs: dict = dict()
# #     ):
# #         super().__init__()
# #         self.num_modalities = len(dim_modalities)
# #         self.dim_modalities = dim_modalities

# #         residual_klass = Residual if num_residual_streams == 1 else HyperConnections
# #         self.attn_residual_fns = ModuleList([residual_klass(num_residual_streams, dim=dim) for dim in dim_modalities])
# #         self.ff_residual_fns = ModuleList([residual_klass(num_residual_streams, dim=dim) for dim in dim_modalities])

# #         has_cond = exists(dim_cond)
# #         self.has_cond = has_cond
# #         if has_cond:
# #             cond_linear = nn.Linear(dim_cond, sum(dim_modalities) * 2)
# #             self.to_post_branch_gammas = nn.Sequential(
# #                 nn.SiLU(),
# #                 cond_linear,
# #                 Rearrange('b d -> b 1 d'),
# #             )
# #             nn.init.zeros_(cond_linear.weight)
# #             nn.init.constant_(cond_linear.bias, 1.)

# #         self.attn_layernorms = ModuleList([AdaptiveLayerNorm(dim, dim_cond=dim_cond) for dim in dim_modalities])
# #         self.joint_attn = JointAttention(
# #             dim_inputs = dim_modalities,
# #             dim_head = dim_head,
# #             heads = heads,
# #             flash = flash_attn,
# #             softclamp = softclamp,
# #             softclamp_value = softclamp_value,
# #         )

# #         self.ff_layernorms = ModuleList([AdaptiveLayerNorm(dim, dim_cond=dim_cond) for dim in dim_modalities])
# #         self.feedforwards = ModuleList([FeedForward(dim, **ff_kwargs) for dim in dim_modalities])

# #     def forward(
# #         self,
# #         *,
# #         modality_tokens: tuple[Tensor, ...],
# #         modality_masks: tuple[Tensor | None, ...] | None = None,
# #         time_cond = None
# #     ):
# #         assert len(modality_tokens) == self.num_modalities
# #         assert not (exists(time_cond) ^ self.has_cond), 'condition must be passed in if dim_cond is set at init.'

# #         ln_kwargs = dict(cond = time_cond) if self.has_cond else dict()
        
# #         # Attention block
# #         attn_inputs, attn_residual_apply_fns = zip(*[res(modality) for res, modality in zip(self.attn_residual_fns, modality_tokens)])
# #         normed_attn_inputs = [ln(tokens, **ln_kwargs) for tokens, ln in zip(attn_inputs, self.attn_layernorms)]
# #         attn_outs = self.joint_attn(inputs=normed_attn_inputs, masks=modality_masks)

# #         if self.has_cond:
# #             gammas = self.to_post_branch_gammas(time_cond)
# #             attn_gammas, ff_gammas = gammas.chunk(2, dim = -1)
# #             attn_gammas_split = attn_gammas.split(self.dim_modalities, dim=-1)
# #             attn_outs = [(out * g) for out, g in zip(attn_outs, attn_gammas_split)]
        
# #         modality_tokens = [apply_res(out) for apply_res, out in zip(attn_residual_apply_fns, attn_outs)]

# #         # Feedforward block
# #         ff_inputs, ff_residual_apply_fns = zip(*[res(modality) for res, modality in zip(self.ff_residual_fns, modality_tokens)])
# #         normed_ff_inputs = [ln(tokens, **ln_kwargs) for tokens, ln in zip(ff_inputs, self.ff_layernorms)]
# #         ff_outs = [ff(tokens) for tokens, ff in zip(normed_ff_inputs, self.feedforwards)]

# #         if self.has_cond:
# #             # ff_gammas were calculated earlier
# #             ff_gammas_split = ff_gammas.split(self.dim_modalities, dim=-1)
# #             ff_outs = [(out * g) for out, g in zip(ff_outs, ff_gammas_split)]
            
# #         modality_tokens = [apply_res(out) for apply_res, out in zip(ff_residual_apply_fns, ff_outs)]

# #         return tuple(modality_tokens)


# # class MMDiT(Module):
# #     def __init__(
# #         self,
# #         *,
# #         depth,
# #         dim_modalities,
# #         final_norms = True,
# #         num_residual_streams = 1,
# #         **block_kwargs
# #     ):
# #         super().__init__()
# #         self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

# #         self.blocks = ModuleList([MMDiTBlock(dim_modalities = dim_modalities, num_residual_streams = num_residual_streams, **block_kwargs) for _ in range(depth)])
        
# #         self.final_norms = final_norms
# #         if self.final_norms:
# #             self.norms = ModuleList([RMSNorm(dim) for dim in dim_modalities])

# #     def forward(
# #         self,
# #         *,
# #         modality_tokens: tuple[Tensor, ...],
# #         modality_masks: tuple[Tensor | None, ...] | None = None,
# #         time_cond = None
# #     ):
# #         modality_tokens = tuple(self.expand_streams(modality) for modality in modality_tokens)

# #         for block in self.blocks:
# #             modality_tokens = block(
# #                 time_cond = time_cond,
# #                 modality_tokens = modality_tokens,
# #                 modality_masks = modality_masks
# #             )
        
# #         modality_tokens = tuple(self.reduce_streams(modality) for modality in modality_tokens)

# #         if self.final_norms:
# #             modality_tokens = tuple(norm(tokens) for tokens, norm in zip(modality_tokens, self.norms))

# #         return modality_tokens
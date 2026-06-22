import functools
from functools import reduce, partial
from packaging import version

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.cuda.amp import autocast
from typing import Callable, Literal
import pdb
        
try:
    from flash_attn import flash_attn_func, flash_attn_kvpacked_func
except ImportError as e:
    print(e)
    print('flash_attn not installed, disabling Flash Attention')
    flash_attn_kvpacked_func = None
    flash_attn_func = None

try:
    import natten
except ImportError:
    natten = None

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


# Copied and modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/attend.py under MIT License
# License can be found in LICENSES/LICENSE_XTRANSFORMERS.txt

def create_causal_mask(i, j, device):
    return torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)

def or_reduce(masks):
    head, *body = masks
    for rest in body:
        head = head | rest
    return head

# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = (pos - seq_start_pos[..., None]).clamp(min = 0)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb

class ScaledSinusoidalEmbedding(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        assert (dim % 2) == 0, 'dimension must be divisible by 2'
        self.scale = nn.Parameter(torch.ones(1) * dim ** -0.5)

        half_dim = dim // 2
        freq_seq = torch.arange(half_dim).float() / half_dim
        inv_freq = theta ** -freq_seq
        self.register_buffer('inv_freq', inv_freq, persistent = False)

    def forward(self, x, pos = None, seq_start_pos = None):
        seq_len, device = x.shape[1], x.device

        if pos is None:
            pos = torch.arange(seq_len, device = device)

        if seq_start_pos is not None:
            pos = pos - seq_start_pos[..., None]

        emb = einsum('i, j -> i j', pos, self.inv_freq)
        emb = torch.cat((emb.sin(), emb.cos()), dim = -1)
        return emb * self.scale
    
class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512,
        interpolation_factor = 1.,
        base = 10000,
        base_rescale_factor = 1.
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature
        # https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/
        base *= base_rescale_factor ** (dim / (dim - 2))

        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        assert interpolation_factor >= 1.
        self.interpolation_factor = interpolation_factor

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        device = self.inv_freq.device

        t = torch.arange(seq_len, device = device)
        return self.forward(t)

    @autocast(enabled = False)
    def forward(self, t):
        device = self.inv_freq.device

        t = t.to(torch.float32)

        t = t / self.interpolation_factor

        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if self.scale is None:
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)

@autocast(enabled = False)
def apply_rotary_pos_emb(t, freqs, scale = 1):
    out_dtype = t.dtype

    # cast to float32 if necessary for numerical stability
    dtype = reduce(torch.promote_types, (t.dtype, freqs.dtype, torch.float32))
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs, t = freqs.to(dtype), t.to(dtype)
    freqs = freqs[-seq_len:, :]

    if t.ndim == 4 and freqs.ndim == 3:
        freqs = rearrange(freqs, 'b n d -> b 1 n d')

    # partial rotary embeddings, Wang et al. GPT-J
    t, t_unrotated = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)

    t, t_unrotated = t.to(out_dtype), t_unrotated.to(out_dtype)

    return torch.cat((t, t_unrotated), dim = -1)

# norms
class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False, fix_scale=False):
        """
        bias-less layernorm has been shown to be more stable. most newer models have moved towards rmsnorm, also bias-less
        """
        super().__init__()

        if fix_scale:
            self.register_buffer("gamma", torch.ones(dim))
        else:
            self.gamma = nn.Parameter(torch.ones(dim))

        if bias:
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_buffer("beta", torch.zeros(dim))


    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta)

# feedforward

class GLU(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        activation: Callable,
        use_conv = False,
        conv_kernel_size = 3,
    ):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2) if not use_conv else nn.Conv1d(dim_in, dim_out * 2, conv_kernel_size, padding = (conv_kernel_size // 2))
        self.use_conv = use_conv

    def forward(self, x):
        if self.use_conv:
            x = rearrange(x, 'b n d -> b d n')
            x = self.proj(x)
            x = rearrange(x, 'b d n -> b n d')
        else:
            x = self.proj(x)

        x, gate = x.chunk(2, dim = -1)
        return x * self.act(gate)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        no_bias = False,
        glu = True,
        use_conv = False,
        conv_kernel_size = 3,
        zero_init_output = True,
    ):
        super().__init__()
        inner_dim = int(dim * mult)

        # Default to SwiGLU

        activation = nn.SiLU()

        dim_out = dim if dim_out is None else dim_out

        if glu:
            linear_in = GLU(dim, inner_dim, activation)
        else:
            linear_in = nn.Sequential(
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                nn.Linear(dim, inner_dim, bias = not no_bias) if not use_conv else nn.Conv1d(dim, inner_dim, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias),
                Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
                activation
            )

        linear_out = nn.Linear(inner_dim, dim_out, bias = not no_bias) if not use_conv else nn.Conv1d(inner_dim, dim_out, conv_kernel_size, padding = (conv_kernel_size // 2), bias = not no_bias)

        # init last linear layer to 0
        if zero_init_output:
            nn.init.zeros_(linear_out.weight)
            if not no_bias:
                nn.init.zeros_(linear_out.bias)


        self.ff = nn.Sequential(
            linear_in,
            Rearrange('b d n -> b n d') if use_conv else nn.Identity(),
            linear_out,
            Rearrange('b n d -> b d n') if use_conv else nn.Identity(),
        )

    def forward(self, x):
        return self.ff(x)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_heads = 64,
        dim_context = None,
        causal = False,
        zero_init_output=True,
        qk_norm: Literal['l2', 'ln', 'none'] = 'none',
        natten_kernel_size = None
    ):
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.causal = causal

        dim_kv = dim_context if dim_context is not None else dim
        
        self.num_heads = dim // dim_heads
        self.kv_heads = dim_kv // dim_heads

        if dim_context is not None:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.to_out = nn.Linear(dim, dim, bias=False)

        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)

        self.qk_norm = qk_norm

        if self.qk_norm == "ln":
            self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
            self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)

        # Using 1d neighborhood attention
        self.natten_kernel_size = natten_kernel_size
        if natten_kernel_size is not None:
            return

        self.use_pt_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')

        self.use_fa_flash = torch.cuda.is_available() and flash_attn_func is not None

        self.sdp_kwargs = dict(
            enable_flash = True,
            enable_math = True,
            enable_mem_efficient = True
        )

    def flash_attn(
            self,
            q, 
            k, 
            v,
            mask = None,
            causal = None
    ):
        batch, heads, q_len, _, k_len, device = *q.shape, k.shape[-2], q.device
        kv_heads = k.shape[1]
        # Recommended for multi-query single-key-value attention by Tri Dao
        # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

        if heads != kv_heads:
            # Repeat interleave kv_heads to match q_heads
            heads_per_kv_head = heads // kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

        if k.ndim == 3:
            k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

        if v.ndim == 3:
            v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

        causal = self.causal if causal is None else causal

        if q_len == 1 and causal:
            causal = False
        
        if mask is not None:
            assert mask.ndim == 4
            mask = mask.expand(batch, heads, q_len, k_len)

        # handle kv cache - this should be bypassable in updated flash attention 2

        if k_len > q_len and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            if mask is None:
                mask = ~causal_mask
            else:
                mask = mask & ~causal_mask
            causal = False

        # manually handle causal mask, if another mask was given

        row_is_entirely_masked = None

        if mask is not None and causal:
            causal_mask = self.create_causal_mask(q_len, k_len, device = device)
            mask = mask & ~causal_mask

            # protect against an entire row being masked out

            row_is_entirely_masked = ~mask.any(dim = -1)
            mask[..., 0] = mask[..., 0] | row_is_entirely_masked

            causal = False
        
        with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                is_causal = causal
            )

        # for a row that is entirely masked out, should zero out the output of that row token

        if row_is_entirely_masked is not None:
            out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

        return out

    def forward(
        self,
        x,
        context = None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        causal = None
    ):
        h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

        kv_input = context if has_context else x

        if hasattr(self, 'to_q'):
            # Use separate linear projections for q and k/v
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h = h) # [B, 24, 1025, 64]

            k, v = self.to_kv(kv_input).chunk(2, dim=-1)

            k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
        else:
            # Use fused linear projection
            q, k, v = self.to_qkv(x).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        
        # Normalize q and k for cosine sim attention
        if self.qk_norm == "l2":
            q = F.normalize(q, dim=-1)
            k = F.normalize(k, dim=-1)
        elif self.qk_norm == "ln":
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rotary_pos_emb is not None and not has_context:
            freqs, _ = rotary_pos_emb

            q_dtype = q.dtype
            k_dtype = k.dtype

            q = q.to(torch.float32)
            k = k.to(torch.float32)
            freqs = freqs.to(torch.float32)

            q = apply_rotary_pos_emb(q, freqs)
            k = apply_rotary_pos_emb(k, freqs)

            q = q.to(q_dtype)
            k = k.to(k_dtype)
        
        input_mask = context_mask 

        if input_mask is None and not has_context:
            input_mask = mask

        # determine masking
        masks = []
        final_attn_mask = None # The mask that will be applied to the attention matrix, taking all masks into account

        if input_mask is not None:
            input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
            masks.append(~input_mask)

        # Other masks will be added here later

        if len(masks) > 0:
            final_attn_mask = ~or_reduce(masks)

        n, device = q.shape[-2], q.device

        causal = self.causal if causal is None else causal

        if n == 1 and causal:
            causal = False

        if self.natten_kernel_size is not None:
            if natten is None:
                raise ImportError('natten not installed, please install natten to use neighborhood attention')
            
            dtype_in = q.dtype
            q, k, v = map(lambda t: t.to(torch.float32), (q, k, v))

            attn = natten.functional.natten1dqk(q, k, kernel_size = self.natten_kernel_size, dilation=1)

            if final_attn_mask is not None:
                attn = attn.masked_fill(final_attn_mask, -torch.finfo(attn.dtype).max)

            attn = F.softmax(attn, dim=-1, dtype=torch.float32)

            out = natten.functional.natten1dav(attn, v, kernel_size = self.natten_kernel_size, dilation=1).to(dtype_in)

        # Prioritize Flash Attention 2
        elif self.use_fa_flash:
            assert final_attn_mask is None, 'masking not yet supported for Flash Attention 2'
            # Flash Attention 2 requires FP16 inputs
            fa_dtype_in = q.dtype
            q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d').to(torch.float16), (q, k, v))
            
            out = flash_attn_func(q, k, v, causal = causal)
            
            out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')

        # Fall back to PyTorch implementation
        elif self.use_pt_flash:
            out = self.flash_attn(q, k, v, causal = causal, mask = final_attn_mask)

        else:
            # Fall back to custom implementation

            if h != kv_h:
                # Repeat interleave kv_heads to match q_heads
                heads_per_kv_head = h // kv_h
                k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

            scale = 1. / (q.shape[-1] ** 0.5)

            kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

            dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
            
            i, j, dtype = *dots.shape[-2:], dots.dtype

            mask_value = -torch.finfo(dots.dtype).max

            if final_attn_mask is not None:
                dots = dots.masked_fill(~final_attn_mask, mask_value)

            if causal:
                causal_mask = self.create_causal_mask(i, j, device = device)
                dots = dots.masked_fill(causal_mask, mask_value)

            attn = F.softmax(dots, dim=-1, dtype=torch.float32)
            attn = attn.type(dtype)

            out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, ' b h n d -> b n (h d)')

        # Communicate between heads
        
        # with autocast(enabled = False):
        #     out_dtype = out.dtype
        #     out = out.to(torch.float32)
        #     out = self.to_out(out).to(out_dtype)
        out = self.to_out(out)

        if mask is not None:
            mask = rearrange(mask, 'b n -> b n 1')
            out = out.masked_fill(~mask, 0.)

        return out

# class Attention(nn.Module):
#     def __init__(
#         self,
#         dim,
#         dim_heads = 64,
#         dim_context = None,
#         causal = False,
#         zero_init_output=True,
#         qk_norm: Literal['l2', 'ln', 'none'] = 'none',
#         natten_kernel_size = None
#     ):
#         super().__init__()
#         self.dim = dim
#         self.dim_heads = dim_heads
#         self.causal = causal

#         dim_kv = dim_context if dim_context is not None else dim
        
#         self.num_heads = dim // dim_heads
#         self.kv_heads = dim_kv // dim_heads

#         if dim_context is not None:
#             self.to_q = nn.Linear(dim, dim, bias=False)
#             self.to_kv = nn.Linear(dim_kv, dim_kv * 2, bias=False)
            
#             if_adapter = True
#             if if_adapter:
#                 self.to_adapter_kv = nn.Linear(dim, dim * 2, bias=False)
#         else:
#             self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

#         self.to_out = nn.Linear(dim, dim, bias=False)

#         if zero_init_output:
#             nn.init.zeros_(self.to_out.weight)

#         self.qk_norm = qk_norm

#         if self.qk_norm == "ln":
#             self.q_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)
#             self.k_norm = nn.LayerNorm(dim_heads, elementwise_affine=True, eps=1.0e-6)

#         # Using 1d neighborhood attention
#         self.natten_kernel_size = natten_kernel_size
#         if natten_kernel_size is not None:
#             return

#         self.use_pt_flash = torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')

#         self.use_fa_flash = torch.cuda.is_available() and flash_attn_func is not None

#         self.sdp_kwargs = dict(
#             enable_flash = True,
#             enable_math = True,
#             enable_mem_efficient = True
#         )
#         self.gate = nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))

#     def flash_attn(
#             self,
#             q, 
#             k, 
#             v,
#             mask = None,
#             causal = None
#     ):
#         batch, heads, q_len, _, k_len, device = *q.shape, k.shape[-2], q.device
#         kv_heads = k.shape[1]
#         # Recommended for multi-query single-key-value attention by Tri Dao
#         # kv shape torch.Size([1, 512, 64]) -> torch.Size([1, 8, 512, 64])

#         if heads != kv_heads:
#             # Repeat interleave kv_heads to match q_heads
#             heads_per_kv_head = heads // kv_heads
#             k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

#         if k.ndim == 3:
#             k = rearrange(k, 'b ... -> b 1 ...').expand_as(q)

#         if v.ndim == 3:
#             v = rearrange(v, 'b ... -> b 1 ...').expand_as(q)

#         causal = self.causal if causal is None else causal

#         if q_len == 1 and causal:
#             causal = False
        
#         if mask is not None:
#             assert mask.ndim == 4
#             mask = mask.expand(batch, heads, q_len, k_len)

#         # handle kv cache - this should be bypassable in updated flash attention 2

#         if k_len > q_len and causal:
#             causal_mask = self.create_causal_mask(q_len, k_len, device = device)
#             if mask is None:
#                 mask = ~causal_mask
#             else:
#                 mask = mask & ~causal_mask
#             causal = False

#         # manually handle causal mask, if another mask was given

#         row_is_entirely_masked = None

#         if mask is not None and causal:
#             causal_mask = self.create_causal_mask(q_len, k_len, device = device)
#             mask = mask & ~causal_mask

#             # protect against an entire row being masked out

#             row_is_entirely_masked = ~mask.any(dim = -1)
#             mask[..., 0] = mask[..., 0] | row_is_entirely_masked

#             causal = False
        
#         with torch.backends.cuda.sdp_kernel(**self.sdp_kwargs):
#             out = F.scaled_dot_product_attention(
#                 q, k, v,
#                 attn_mask = mask,
#                 is_causal = causal
#             )

#         # for a row that is entirely masked out, should zero out the output of that row token

#         if row_is_entirely_masked is not None:
#             out = out.masked_fill(row_is_entirely_masked[..., None], 0.)

#         return out

#     def forward(
#         self,
#         x,
#         context = None,
#         mask = None,
#         context_mask = None,
#         rotary_pos_emb = None,
#         causal = None,
#         adapter=None  # 添加适配器参数
#     ):
#         h, kv_h, has_context = self.num_heads, self.kv_heads, context is not None

#         kv_input = context if has_context else x

#         if hasattr(self, 'to_q'):
#             # Use separate linear projections for q and k/v
#             q = self.to_q(x)
#             q = rearrange(q, 'b n (h d) -> b h n d', h = h) # [B, 24, 1025, 64]

#             k, v = self.to_kv(kv_input).chunk(2, dim=-1)

#             k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = kv_h), (k, v))
#         else:
#             # Use fused linear projection
#             q, k, v = self.to_qkv(x).chunk(3, dim=-1)
#             q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))


        
#         # Normalize q and k for cosine sim attention
#         if self.qk_norm == "l2":
#             q = F.normalize(q, dim=-1)
#             k = F.normalize(k, dim=-1)
#         elif self.qk_norm == "ln":
#             q = self.q_norm(q)
#             k = self.k_norm(k)
#         # pdb.set_trace()
#         if rotary_pos_emb is not None and not has_context:
#             freqs, _ = rotary_pos_emb

#             q_dtype = q.dtype
#             k_dtype = k.dtype

#             q = q.to(torch.float32)
#             k = k.to(torch.float32)
#             freqs = freqs.to(torch.float32)

#             q = apply_rotary_pos_emb(q, freqs)
#             k = apply_rotary_pos_emb(k, freqs)

#             q = q.to(q_dtype)
#             k = k.to(k_dtype)


#         if adapter is not None:
#             # pdb.set_trace()
#             bsz = x.shape[0]
#             adapter_len = adapter.shape[1]

#             if hasattr(self, 'to_kv'):
#                 adapter_kv = self.to_adapter_kv(adapter)
#                 adapter_k, adapter_v = adapter_kv.chunk(2, dim=-1)
#             else:
#                 adapter_qkv = self.to_qkv(adapter)
#                 _, adapter_k, adapter_v = adapter_qkv.chunk(3, dim=-1)

#             adapt_h = int(h/2)
#             # adapt_d = int(adapter_k.shape[2]/h)
#             # adapt_n = int(adapter_k.shape[1]*2)
#             # import einops            
            
#             # adapter_k = rearrange(adapter_k, 'b n (h d) -> b adapt_h adapt_n adapt_d', adapt_h=adapt_h, adapt_d=adapt_d, adapt_n)
#             # adapter_v = rearrange(adapter_v, 'b n (h d) -> b h n d', h=adapt_h)
#             adapter_k = rearrange(adapter_k, 'b n (h d) -> b h n d', h=h)
#             adapter_v = rearrange(adapter_v, 'b n (h d) -> b h n d', h=h)
            
#             if has_context:
#                 adapter_k = rearrange(adapter_k, 'b (h1 h2) n d -> b h1 (n h2) d', h1=adapt_h)
#                 adapter_v = rearrange(adapter_v, 'b (h1 h2) n d -> b h1 (n h2) d', h1=adapt_h)

#             k = torch.cat([adapter_k, k], dim=2)
#             v = torch.cat([adapter_v, v], dim=2)

#             # 调整掩码
#             if context_mask is not None: # False
#                 adapter_mask = torch.ones(bsz, adapter_len, device=context_mask.device, dtype=context_mask.dtype)
#                 context_mask = torch.cat([adapter_mask, context_mask], dim=1)
#             else:
#                 if mask is not None:
#                     adapter_mask = torch.ones(bsz, adapter_len, device=mask.device, dtype=mask.dtype)
#                     mask = torch.cat([adapter_mask, mask], dim=1)
#                 else:
#                     pass
#                     # 如果没有提供 mask，则创建一个全为 True 的掩码
#                     # pdb.set_trace()
#                     # mask = torch.ones(bsz, adapter_len + k.shape[2], device=k.device, dtype=torch.bool)
#                     # mask = torch.ones(bsz, k.shape[2], device=k.device, dtype=torch.bool)

        
#         input_mask = context_mask 
#         # pdb.set_trace()
#         if input_mask is None and not has_context:
#             input_mask = mask

#         # determine masking
#         masks = []
#         final_attn_mask = None # The mask that will be applied to the attention matrix, taking all masks into account

#         if input_mask is not None:
#             input_mask = rearrange(input_mask, 'b j -> b 1 1 j')
#             masks.append(~input_mask)

#         # Other masks will be added here later

#         if len(masks) > 0:
#             final_attn_mask = ~or_reduce(masks)

#         n, device = q.shape[-2], q.device

#         causal = self.causal if causal is None else causal

#         if n == 1 and causal:
#             causal = False
#         # pdb.set_trace()
#         if self.natten_kernel_size is not None:
#             # pdb.set_trace()
#             if natten is None:
#                 raise ImportError('natten not installed, please install natten to use neighborhood attention')
            
#             dtype_in = q.dtype
#             q, k, v = map(lambda t: t.to(torch.float32), (q, k, v))

#             attn = natten.functional.natten1dqk(q, k, kernel_size = self.natten_kernel_size, dilation=1)

#             if final_attn_mask is not None:
#                 attn = attn.masked_fill(final_attn_mask, -torch.finfo(attn.dtype).max)

#             attn = F.softmax(attn, dim=-1, dtype=torch.float32)

#             out = natten.functional.natten1dav(attn, v, kernel_size = self.natten_kernel_size, dilation=1).to(dtype_in)

#         # Prioritize Flash Attention 2
#         elif self.use_fa_flash:
#             # pdb.set_trace()
#             assert final_attn_mask is None, 'masking not yet supported for Flash Attention 2'
#             # Flash Attention 2 requires FP16 inputs
#             fa_dtype_in = q.dtype
#             q, k, v = map(lambda t: rearrange(t, 'b h n d -> b n h d').to(torch.float16), (q, k, v))
            
#             out = flash_attn_func(q, k, v, causal = causal)
            
#             out = rearrange(out.to(fa_dtype_in), 'b n h d -> b h n d')

#         # Fall back to PyTorch implementation
#         elif self.use_pt_flash:
#             # pdb.set_trace()
#             out = self.flash_attn(q, k, v, causal = causal, mask = final_attn_mask)

#         else:
#             # Fall back to custom implementation
#             # pdb.set_trace()
#             if h != kv_h:
#                 # Repeat interleave kv_heads to match q_heads
#                 heads_per_kv_head = h // kv_h
#                 k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim = 1), (k, v))

#             scale = 1. / (q.shape[-1] ** 0.5)

#             kv_einsum_eq = 'b j d' if k.ndim == 3 else 'b h j d'

#             dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale
            
#             i, j, dtype = *dots.shape[-2:], dots.dtype

#             mask_value = -torch.finfo(dots.dtype).max

#             if final_attn_mask is not None:
#                 dots = dots.masked_fill(~final_attn_mask, mask_value)

#             if causal:
#                 causal_mask = self.create_causal_mask(i, j, device = device)
#                 dots = dots.masked_fill(causal_mask, mask_value)

#             if adapter is not None:
#                 # pdb.set_trace()
#                 adapter_len = adapter.shape[1]
#                 adapter_scores = dots[..., :adapter_len]
#                 rest_scores = dots[..., adapter_len:]

#                 adapter_attn = F.softmax(adapter_scores.float(), dim=-1).type_as(dots)
#                 rest_attn = F.softmax(rest_scores.float(), dim=-1).type_as(dots)

#                 adapter_attn = self.gate.tanh() * adapter_attn

#                 attn = torch.cat([adapter_attn, rest_attn], dim=-1)
#             else:
#                 attn = F.softmax(dots, dim=-1, dtype=torch.float32)
#                 attn = attn.type(dtype)
                
#             # attn = F.softmax(dots, dim=-1, dtype=torch.float32)
#             # attn = attn.type(dtype)

#             out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
#         # pdb.set_trace()
#         # merge heads
#         out = rearrange(out, ' b h n d -> b n (h d)')

#         # Communicate between heads
        
#         # with autocast(enabled = False):
#         #     out_dtype = out.dtype
#         #     out = out.to(torch.float32)
#         #     out = self.to_out(out).to(out_dtype)
#         out = self.to_out(out)

#         if mask is not None:
#             mask = rearrange(mask, 'b n -> b n 1')
#             out = out.masked_fill(~mask, 0.)

#         return out


class ConformerModule(nn.Module):
    def __init__(
        self,
        dim,
        norm_kwargs = {},
    ):     

        super().__init__()

        self.dim = dim
        
        self.in_norm = LayerNorm(dim, **norm_kwargs)
        self.pointwise_conv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.glu = GLU(dim, dim, nn.SiLU())
        self.depthwise_conv = nn.Conv1d(dim, dim, kernel_size=17, groups=dim, padding=8, bias=False)
        self.mid_norm = LayerNorm(dim, **norm_kwargs) # This is a batch norm in the original but I don't like batch norm
        self.swish = nn.SiLU()
        self.pointwise_conv_2 = nn.Conv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.in_norm(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.glu(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.depthwise_conv(x)
        x = rearrange(x, 'b d n -> b n d')
        x = self.mid_norm(x)
        x = self.swish(x)
        x = rearrange(x, 'b n d -> b d n')
        x = self.pointwise_conv_2(x)
        x = rearrange(x, 'b d n -> b n d')

        return x

class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_heads = 64,
            cross_attend = False,
            dim_context = None,
            global_cond_dim = None,
            causal = False,
            zero_init_branch_outputs = True,
            conformer = False,
            layer_ix = -1,
            remove_norms = False,
            attn_kwargs = {},
            ff_kwargs = {},
            norm_kwargs = {}
    ):
        
        super().__init__()
        self.dim = dim
        self.dim_heads = dim_heads
        self.cross_attend = cross_attend
        self.dim_context = dim_context
        self.causal = causal

        self.pre_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()

        self.self_attn = Attention(
            dim,
            dim_heads = dim_heads,
            causal = causal,
            zero_init_output=zero_init_branch_outputs,
            **attn_kwargs
        )

        if cross_attend: # True
            self.cross_attend_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
            self.cross_attn = Attention(
                dim,
                dim_heads = dim_heads,
                dim_context=dim_context,
                causal = causal,
                zero_init_output=zero_init_branch_outputs,
                **attn_kwargs
            )
        
        self.ff_norm = LayerNorm(dim, **norm_kwargs) if not remove_norms else nn.Identity()
        self.ff = FeedForward(dim, zero_init_output=zero_init_branch_outputs, **ff_kwargs)

        self.layer_ix = layer_ix

        self.conformer = ConformerModule(dim, norm_kwargs=norm_kwargs) if conformer else None

        self.global_cond_dim = global_cond_dim

        if global_cond_dim is not None:
            self.to_scale_shift_gate = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_cond_dim, dim * 6, bias=False)
            )

            nn.init.zeros_(self.to_scale_shift_gate[1].weight)
            #nn.init.zeros_(self.to_scale_shift_gate_self[1].bias)

    def forward(
        self,
        x,
        context = None,
        global_cond=None,
        mask = None,
        context_mask = None,
        rotary_pos_emb = None,
        adapter=None
    ):

        # pdb.set_trace()        
        if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None: # False
            
            scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = self.to_scale_shift_gate(global_cond).unsqueeze(1).chunk(6, dim = -1)

            # self-attention with adaLN
            residual = x
            x = self.pre_norm(x)
            x = x * (1 + scale_self) + shift_self
            
            # x = self.self_attn(x, mask = mask, rotary_pos_emb = rotary_pos_emb, adapter=adapter)
            x = self.self_attn(x, mask = mask, rotary_pos_emb = rotary_pos_emb)
            x = x * torch.sigmoid(1 - gate_self)
            x = x + residual

            if context is not None:
                
                # x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask, adapter=adapter)
                x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask)
                
            if self.conformer is not None:
                x = x + self.conformer(x)

            # feedforward with adaLN
            residual = x
            x = self.ff_norm(x)
            x = x * (1 + scale_ff) + shift_ff
            x = self.ff(x)
            x = x * torch.sigmoid(1 - gate_ff)
            x = x + residual

        else:
            # x = x + self.self_attn(self.pre_norm(x), mask = mask, rotary_pos_emb = rotary_pos_emb, adapter=adapter)
            x = x + self.self_attn(self.pre_norm(x), mask = mask, rotary_pos_emb = rotary_pos_emb)

            if context is not None:
                # pdb.set_trace()      
                # x: [1, 1025, 1536], context: [1, 130, 768]
                # x: [6, 216, 1536],  context: [6, 130, 768], context_mask=None, adapter:[6, 10, 1536]
                # x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask, adapter=adapter)
                x = x + self.cross_attn(self.cross_attend_norm(x), context = context, context_mask = context_mask)

            if self.conformer is not None: # False
                x = x + self.conformer(x)

            x = x + self.ff(self.ff_norm(x))

        return x
        
class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        # self.adapter_len = 10
        # self.adapter_query = nn.Embedding(self.adapter_len * depth, dim)

        if rotary_pos_emb:
            self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        else:
            self.rotary_pos_emb = None

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)

        for i in range(depth):
            self.layers.append(
                TransformerBlock(
                    dim,
                    dim_heads = dim_heads,
                    cross_attend = cross_attend,
                    dim_context = cond_token_dim,
                    global_cond_dim = global_cond_dim,
                    causal = causal,
                    zero_init_branch_outputs = zero_init_branch_outputs,
                    conformer=conformer,
                    layer_ix=i,
                    **kwargs
                )
            )
        
        # # 冻结所有层的参数
        # for param in self.parameters():
        #     param.requires_grad = False
        
    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        return_info = False,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        info = {
            "hidden_states": [],
        }
        # import pdb
        # pdb.set_trace()
        # self.project_in = self.project_in.eval()
        # self.project_out = self.project_out.eval()
        # with torch.no_grad():
        #     x = self.project_in(x)

        x = self.project_in(x)
        # import pdb; pdb.set_trace()
        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                mask = torch.cat((prepend_mask, mask), dim = -1)

        # Attention layers

        if self.rotary_pos_emb is not None:
            rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        else:
            rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb: # False
            x = x + self.pos_emb(x)

        # Iterate over the transformer layers
        # for layer in self.layers:
        for index, layer in enumerate(self.layers):
            # self.layers[index].eval()
            # with torch.no_grad():
            #     x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
            # import pdb; pdb.set_trace()
            x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
            #x = layer(x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)

            if return_info:
                info["hidden_states"].append(x)

        # ## 加入adapter
        # # 准备适配器嵌入
        # batch_size = x.shape[0]  # 获取批次大小
        # adapter = self.adapter_query.weight.reshape(-1, self.adapter_len, self.dim).unsqueeze(1)
        # adapter = adapter.expand(-1, batch_size, -1, -1)  # [adapter_layer, batch_size, adapter_len, dim]
        
        # # 处理含有适配器的层
        # # pdb.set_trace()
        # for adapter_index, layer in enumerate(self.layers):
        #     adapter_embedding = adapter[adapter_index]
        #     # pdb.set_trace()
        #     x = checkpoint(layer, x, rotary_pos_emb=rotary_pos_emb, global_cond=global_cond, adapter=adapter_embedding, **kwargs)
            
        #     if return_info:
        #         info["hidden_states"].append(x)

        # pdb.set_trace()
        # with torch.no_grad():
        #     x = self.project_out(x)
        x = self.project_out(x)

        if return_info:
            return x, info
        
        return x






class ContinuousMMDiTTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        fusion_depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        _latent_seq_len=237,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        # self.adapter_len = 10
        # self.adapter_query = nn.Embedding(self.adapter_len * depth, dim)

        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)

        # from .mmdit import MMDiTBlock
        # for i in range(depth):
        #     # self.layers.append(
        #     #     TransformerBlock(
        #     #         dim,
        #     #         dim_heads = dim_heads,
        #     #         cross_attend = cross_attend,
        #     #         dim_context = cond_token_dim,
        #     #         global_cond_dim = global_cond_dim,
        #     #         causal = causal,
        #     #         zero_init_branch_outputs = zero_init_branch_outputs,
        #     #         conformer=conformer,
        #     #         layer_ix=i,
        #     #         **kwargs
        #     #     )
        #     # )
        #     self.layers.append(
        #         MMDiTBlock(
        #         dim_cond = 1536,
        #         dim_text = 768,
        #         dim_image = 1536,
        #         qk_rmsnorm = True
        #     )
        #     )

        from .mm_transformer_layers import MMDitSingleBlock
        hidden_dim = dim
        num_heads = dim_heads
        mlp_ratio = kwargs.get('mlp_ratio', 4.0)
        kernel_size = kwargs.get('kernel_size', 3)
        padding_size = (kernel_size - 1) // 2
        fused_depth = depth
        cross_attend = True
        self.cross_attend = cross_attend
        # self._latent_seq_len = 237  # 237, 256, 384
        # self._latent_seq_len = 276  # 237, 256, 384
        # import pdb; pdb.set_trace()
        self._latent_seq_len = _latent_seq_len

        # import pdb; pdb.set_trace()
        self.layers = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size, padding=padding_size, cross_attend=cross_attend)
            for i in range(fused_depth)
        ])
        self.proj_mm_tokens = nn.Linear(768, hidden_dim) if dim != 768 else nn.Identity()
        # self.proj_mm_seq_len = nn.Linear(128, self._latent_seq_len) if self._latent_seq_len != 384 else nn.Identity() # text condition
        self.proj_mm_seq_len = nn.Linear(384, self._latent_seq_len) if self._latent_seq_len != 384 else nn.Identity() # all condition

        # from .mmdit import MMDiT
        # self.mmdit = MMDiT(
        #     depth = depth, 
        #     dim_modalities = (768, 768, 768),
        #     dim_cond = 768,
        #     qk_rmsnorm = True
        # )


        # v1
        # if rotary_pos_emb:
        #     self.rotary_pos_emb = RotaryEmbedding(max(dim_heads // 2, 32))
        # else:
        #     self.rotary_pos_emb = None

        # v2
        base_freq = 1.0
        from .mm_embeddings import compute_rope_rotations
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            hidden_dim // num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        self.register_buffer('latent_rot', latent_rot, persistent=False)

        # # 冻结所有层的参数
        # for param in self.parameters():
        #     param.requires_grad = False
    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        context = None,
        context_mask = None,
        return_info = False,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        info = {
            "hidden_states": [],
        }
        # import pdb
        # pdb.set_trace()
        # self.project_in = self.project_in.eval()
        # self.project_out = self.project_out.eval()
        # with torch.no_grad():
        #     x = self.project_in(x)
        # import pdb; pdb.set_trace()
        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                mask = torch.cat((prepend_mask, mask), dim = -1)

        # Attention layers

        # if self.rotary_pos_emb is not None:
        #     rotary_pos_emb = self.rotary_pos_emb.forward_from_seq_len(x.shape[1])
        # else:
        #     rotary_pos_emb = None

        if self.use_sinusoidal_emb or self.use_abs_pos_emb: # False
            x = x + self.pos_emb(x)

        time_cond = prepend_embeds.squeeze(1)
        # import pdb; pdb.set_trace()
        # video_tokens, text_tokens, audio_tokens = context[0], context[1], context[2]
        # mm_tokens = torch.cat((video_tokens, text_tokens, audio_tokens), dim=1)
        mm_tokens = context

        # x, text_tokens, mm_tokens = self.mmdit(
        #     modality_tokens = (x, text_tokens, mm_tokens),
        #     modality_masks = (None, None, None),
        #     time_cond = time_cond,
        # )
        # import pdb; pdb.set_trace()

        # # v1
        # for index, layer in enumerate(self.layers):
        #     # import pdb; pdb.set_trace()
        #     mm_tokens, x = checkpoint(layer, time_cond = time_cond, text_tokens = mm_tokens, text_mask=None, image_tokens=x)
        #     if return_info:
        #         info["hidden_states"].append(x)
        # # import pdb; pdb.set_trace()
        # # for index, layer in enumerate(self.layers):
        # #     import pdb; pdb.set_trace()
        # #     x = checkpoint(layer, x, rotary_pos_emb = rotary_pos_emb, global_cond=global_cond, **kwargs)
        # #     if return_info:
        # #         info["hidden_states"].append(x)

        # v2
        # import pdb; pdb.set_trace()

                
        mm_tokens = self.proj_mm_tokens(mm_tokens)
        mm_tokens = mm_tokens.transpose(1, 2)  # (B, D, VN)
        mm_tokens = self.proj_mm_seq_len(mm_tokens)
        mm_tokens = mm_tokens.transpose(1, 2)  # (B, N, D)


        time_cond = time_cond.unsqueeze(1)
        for block in self.layers:
            # x:torch.Size([16, 215, 1024]), mm_tokens:[16, 215, 1024], self.latent_rot:[1, 215, 32, 2, 2], time_cond:[16, 256, 1024]
            # import pdb; pdb.set_trace()
            if self.cross_attend:
                x = block(x, mm_tokens, self.latent_rot, context=time_cond)
            else:
                x = block(x, mm_tokens, self.latent_rot)

        x = self.project_out(x)

        if return_info:
            return x, info
        
        return x












from typing import List
class QwenEmbedRope(nn.Module):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = {}

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        """
        Args:
            index: [0, 1, 2, 3] 1D Tensor representing the position index of the token
        """
        
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        """
        Args: video_fhw: [frame, height, width] a list of 3 integers representing the shape of the video Args:
        txt_length: [bs] a list of 1 integers representing the length of the text
        """
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        if isinstance(video_fhw, list):
            video_fhw = video_fhw[0]
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                video_freq = self.rope_cache[rope_key]
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()

class ContinuousQwenMMDiTTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        fusion_depth,
        *,
        dim_in = None,
        dim_out = None,
        dim_heads = 64,
        cross_attend=False,
        cond_token_dim=None,
        global_cond_dim=None,
        causal=False,
        rotary_pos_emb=True,
        zero_init_branch_outputs=True,
        conformer=False,
        use_sinusoidal_emb=False,
        use_abs_pos_emb=False,
        abs_pos_emb_max_length=10000,
        **kwargs
        ):

        super().__init__()

        self.dim = dim
        self.depth = depth
        self.causal = causal
        self.layers = nn.ModuleList([])

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        # self.adapter_len = 10
        # self.adapter_query = nn.Embedding(self.adapter_len * depth, dim)



        self.use_sinusoidal_emb = use_sinusoidal_emb
        if use_sinusoidal_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_emb = AbsolutePositionalEmbedding(dim, abs_pos_emb_max_length)

        from .qwen_mmdit import QwenImageTransformerBlock
        hidden_dim = dim
        num_heads = dim_heads
        mlp_ratio = kwargs.get('mlp_ratio', 4.0)
        kernel_size = kwargs.get('kernel_size', 3)
        padding_size = (kernel_size - 1) // 2
        fused_depth = depth
        cross_attend = True
        self.cross_attend = cross_attend
        self._latent_seq_len = 237  # 237, 256, 384



        # # qwen
        # axes_dims_rope = [2, 238, 16]
        # self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        # # import pdb; pdb.set_trace()
        # self.layers = nn.ModuleList([
        #     MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size, padding=padding_size, cross_attend=cross_attend)
        #     for i in range(fused_depth)
        # ])
        # import pdb; pdb.set_trace()
        # self.layers = nn.ModuleList([
        #     MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=kernel_size, padding=padding_size, cross_attend=cross_attend)
        #     for i in range(fused_depth)
        # ])

        self.layers = nn.ModuleList(
            [
                QwenImageTransformerBlock(
                    dim=hidden_dim, # self.inner_dim=3072
                    num_attention_heads=num_heads, # num_attention_heads=24
                    attention_head_dim=hidden_dim // num_heads,
                )
                for _ in range(fused_depth)
            ]
        )

        self.proj_mm_tokens = nn.Linear(768, hidden_dim) if dim != 768 else nn.Identity()
        # self.proj_mm_seq_len = nn.Linear(384, self._latent_seq_len) if self._latent_seq_len != 384 else nn.Identity()

        # v2
        base_freq = 1.0
        from .mm_embeddings import compute_rope_rotations
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            hidden_dim // num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        self.register_buffer('latent_rot', latent_rot, persistent=False)

    def create_audio_rope_embeddings(self, x, mm_tokens, device):
        """
        为音频数据创建 RoPE 位置编码
        """
        batch_size, time_steps, features = x.shape
        text_len = mm_tokens.shape[1]
        
        # 计算每个注意力头的维度
        # 假设注意力头数是 128，则每个头的维度是 1536 // 128 = 12
        head_dim = 24  # 或者动态计算：features // num_heads
        
        # 生成位置索引
        pos_indices = torch.arange(time_steps, device=device, dtype=torch.float32)
        text_indices = torch.arange(text_len, device=device, dtype=torch.float32)
        
        # 计算频率
        theta = 10000.0
        dim = head_dim  # 使用每个头的维度，不是总特征维度
        
        # 为音频生成 RoPE
        freqs_audio = torch.outer(pos_indices, 1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=device, dtype=torch.float32).div(dim)))
        freqs_audio = torch.polar(torch.ones_like(freqs_audio), freqs_audio)
        
        # 为文本生成 RoPE
        freqs_text = torch.outer(text_indices, 1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=device, dtype=torch.float32).div(dim)))
        freqs_text = torch.polar(torch.ones_like(freqs_text), freqs_text)
        
        return freqs_audio, freqs_text

    @property
    def device(self):
        return next(self.parameters()).device
        
    def forward(
        self,
        x,
        mask = None,
        prepend_embeds = None,
        prepend_mask = None,
        global_cond = None,
        context = None,
        context_mask = None,
        return_info = False,
        **kwargs
    ):
        batch, seq, device = *x.shape[:2], x.device

        info = {
            "hidden_states": [],
        }
        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_length, prepend_dim = prepend_embeds.shape[1:]

            assert prepend_dim == x.shape[-1], 'prepend dimension must match sequence dimension'

            x = torch.cat((prepend_embeds, x), dim = -2)

            if prepend_mask is not None or mask is not None:
                mask = mask if mask is not None else torch.ones((batch, seq), device = device, dtype = torch.bool)
                prepend_mask = prepend_mask if prepend_mask is not None else torch.ones((batch, prepend_length), device = device, dtype = torch.bool)

                mask = torch.cat((prepend_mask, mask), dim = -1)



        if self.use_sinusoidal_emb or self.use_abs_pos_emb: # False
            x = x + self.pos_emb(x)

        time_cond = prepend_embeds.squeeze(1)

        mm_tokens = context


                
        mm_tokens = self.proj_mm_tokens(mm_tokens)
        # mm_tokens = mm_tokens.transpose(1, 2)  # (B, D, VN)
        # mm_tokens = self.proj_mm_seq_len(mm_tokens)
        # mm_tokens = mm_tokens.transpose(1, 2)  # (B, N, D)


        # time_cond = time_cond.unsqueeze(1)
        # for block in self.layers:
        #     # x:torch.Size([16, 215, 1024]), mm_tokens:[16, 215, 1024], self.latent_rot:[1, 215, 32, 2, 2], time_cond:[16, 256, 1024]
        #     # import pdb; pdb.set_trace()
        #     if self.cross_attend:
        #         x = block(x, mm_tokens, self.latent_rot, context=time_cond)
        #     else:
        #         x = block(x, mm_tokens, self.latent_rot)

        encoder_hidden_states_mask = torch.ones((batch, mm_tokens.shape[1]), device = device, dtype = torch.bool)

        txt_seq_lens = [mm_tokens.shape[1]] * batch
        img_shapes = [(1, x.shape[1], 64)] * batch
        # image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=x.device)
        audio_rotary_emb, text_rotary_emb = self.create_audio_rope_embeddings(x, mm_tokens, device=x.device)

        for index_block, block in enumerate(self.layers):
            mm_tokens, x = block(
                hidden_states=x, # torch.Size([1, 6032, 3072]),                                     x: [48, 237, 1536]
                encoder_hidden_states=mm_tokens, # torch.Size([1, 117, 3072]),                      mm_tokens: [48, 384, 1536]
                encoder_hidden_states_mask=encoder_hidden_states_mask, # torch.Size([1, 117]),      [48, 237]
                temb=time_cond, # torch.Size([1, 3072]),                                            time_cond: torch.Size([48, 1536])
                image_rotary_emb=(audio_rotary_emb, text_rotary_emb), # (torch.Size([6032, 64]), torch.Size([117, 64])).     (torch.Size([237, 768]), torch.Size([384, 768]))
                joint_attention_kwargs=None,
            )




        x = self.project_out(x)

        if return_info:
            return x, info
        
        return x


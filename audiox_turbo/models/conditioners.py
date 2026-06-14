# Heavily influenced by https://github.com/facebookresearch/audiocraft/blob/main/audiocraft/modules/conditioners.py

import gc
import logging
import math
import os
import string
import typing as tp
import warnings

import einops
import torch
import torch.nn.init as init
import torchaudio
from einops import rearrange
from torch import nn
from torchvision import transforms
from transformers import AutoProcessor, CLIPVisionModelWithProjection

from .adp import NumberEmbedder
from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from ..training.utils import copy_state_dict
from .utils import load_ckpt_state_dict
from .SA_transformer_module import SA_Attention, SA_PreNorm, SA_FeedForward


def _clip_pretrained_path(clip_model_name):
    if clip_model_name == "clip-vit-base-patch32":
        default_model_id = "openai/clip-vit-base-patch32"
    else:
        default_model_id = clip_model_name
    return os.environ.get("AUDIOX_TURBO_CLIP_MODEL_PATH") or os.environ.get("CLIP_MODEL_PATH") or default_model_id


class Conditioner(nn.Module):
    def __init__(self, dim: int, output_dim: int, project_out: bool = False):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()


class IntConditioner(Conditioner):
    def __init__(self, output_dim: int, min_val: int = 0, max_val: int = 512):
        super().__init__(output_dim, output_dim)
        self.min_val = min_val
        self.max_val = max_val
        self.int_embedder = nn.Embedding(max_val - min_val + 1, output_dim).requires_grad_(True)

    def forward(self, ints: tp.List[int], device=None) -> tp.Any:
        ints = torch.tensor(ints).to(device)
        ints = ints.clamp(self.min_val, self.max_val)
        int_embeds = self.int_embedder(ints).unsqueeze(1)
        return [int_embeds, torch.ones(int_embeds.shape[0], 1).to(device)]


class NumberConditioner(Conditioner):
    """Conditioner that takes a list of floats, normalizes them for a given range, and returns embeddings."""

    def __init__(self, output_dim: int, min_val: float = 0, max_val: float = 1):
        super().__init__(output_dim, output_dim)
        self.min_val = min_val
        self.max_val = max_val
        self.embedder = NumberEmbedder(features=output_dim)

    def forward(self, floats: tp.List[float], device=None) -> tp.Any:
        floats = [float(x) for x in floats]
        floats = torch.tensor(floats).to(device)
        floats = floats.clamp(self.min_val, self.max_val)
        normalized_floats = (floats - self.min_val) / (self.max_val - self.min_val)
        embedder_dtype = next(self.embedder.parameters()).dtype
        normalized_floats = normalized_floats.to(embedder_dtype)
        float_embeds = self.embedder(normalized_floats).unsqueeze(1)
        return [float_embeds, torch.ones(float_embeds.shape[0], 1).to(device)]


class CLAPTextConditioner(Conditioner):
    def __init__(self, output_dim: int, clap_ckpt_path, use_text_features=False,
                 feature_layer_ix: int = -1, audio_model_type="HTSAT-base",
                 enable_fusion=True, project_out: bool = False, finetune: bool = False):
        super().__init__(768 if use_text_features else 512, output_dim, project_out=project_out)
        self.use_text_features = use_text_features
        self.feature_layer_ix = feature_layer_ix
        self.finetune = finetune

        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')
                if self.finetune:
                    self.model = model
                else:
                    self.__dict__["model"] = model
                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)
                if self.finetune:
                    self.model.model.text_branch.requires_grad_(True)
                    self.model.model.text_branch.train()
                else:
                    self.model.model.text_branch.requires_grad_(False)
                    self.model.model.text_branch.eval()
            finally:
                logging.disable(previous_level)

        del self.model.model.audio_branch
        gc.collect()
        torch.cuda.empty_cache()

    def get_clap_features(self, prompts, layer_ix=-2, device: tp.Any = "cuda"):
        prompt_tokens = self.model.tokenizer(prompts)
        attention_mask = prompt_tokens["attention_mask"].to(device=device, non_blocking=True)
        prompt_features = self.model.model.text_branch(
            input_ids=prompt_tokens["input_ids"].to(device=device, non_blocking=True),
            attention_mask=attention_mask,
            output_hidden_states=True,
        )["hidden_states"][layer_ix]
        return prompt_features, attention_mask

    def forward(self, texts: tp.List[str], device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)
        if self.use_text_features:
            if len(texts) == 1:
                text_features, text_attention_mask = self.get_clap_features(
                    [texts[0], ""], layer_ix=self.feature_layer_ix, device=device)
                text_features = text_features[:1, ...]
                text_attention_mask = text_attention_mask[:1, ...]
            else:
                text_features, text_attention_mask = self.get_clap_features(
                    texts, layer_ix=self.feature_layer_ix, device=device)
            return [self.proj_out(text_features), text_attention_mask]

        if len(texts) == 1:
            text_embedding = self.model.get_text_embedding([texts[0], ""], use_tensor=True)[:1, ...]
        else:
            text_embedding = self.model.get_text_embedding(texts, use_tensor=True)
        text_embedding = text_embedding.unsqueeze(1).to(device)
        return [self.proj_out(text_embedding), torch.ones(text_embedding.shape[0], 1).to(device)]


class CLAPAudioConditioner(Conditioner):
    def __init__(self, output_dim: int, clap_ckpt_path, audio_model_type="HTSAT-base",
                 enable_fusion=True, project_out: bool = False):
        super().__init__(512, output_dim, project_out=project_out)

        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                import laion_clap
                from laion_clap.clap_module.factory import load_state_dict as clap_load_state_dict
                model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=audio_model_type, device='cpu')
                if self.finetune:
                    self.model = model
                else:
                    self.__dict__["model"] = model
                state_dict = clap_load_state_dict(clap_ckpt_path)
                self.model.model.load_state_dict(state_dict, strict=False)
                if self.finetune:
                    self.model.model.audio_branch.requires_grad_(True)
                    self.model.model.audio_branch.train()
                else:
                    self.model.model.audio_branch.requires_grad_(False)
                    self.model.model.audio_branch.eval()
            finally:
                logging.disable(previous_level)

        del self.model.model.text_branch
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, audios: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]],
                device: tp.Any = "cuda") -> tp.Any:
        self.model.to(device)
        if isinstance(audios, (list, tuple)):
            audios = torch.cat(audios, dim=0)
        mono_audios = audios.mean(dim=1)
        with torch.cuda.amp.autocast(enabled=False):
            audio_embedding = self.model.get_audio_embedding_from_data(mono_audios.float(), use_tensor=True)
        audio_embedding = audio_embedding.unsqueeze(1).to(device)
        return [self.proj_out(audio_embedding), torch.ones(audio_embedding.shape[0], 1).to(device)]


class SA_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SA_PreNorm(dim, SA_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                SA_PreNorm(dim, SA_FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, x1, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.depth = x1 // num_heads
        self.query = nn.Linear(x1, x1)
        self.key = nn.Linear(x1, x1)
        self.value = nn.Linear(x1, x1)
        self.final_linear = nn.Linear(x1, x1)
        self.norm1 = nn.LayerNorm(x1)
        self.norm2 = nn.LayerNorm(x1)
        init.constant_(self.final_linear.weight, 0)
        if self.final_linear.bias is not None:
            init.constant_(self.final_linear.bias, 0)

    def split_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.permute(0, 2, 1, 3)

    def forward(self, tensor_A, tensor_B):
        batch_size = tensor_A.size(0)
        Q = self.split_heads(self.query(tensor_A), batch_size)
        K = self.split_heads(self.key(tensor_B), batch_size)
        V = self.split_heads(self.value(tensor_B), batch_size)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.depth ** 0.5)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_scores, V)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        output = attention_output.view(batch_size, -1, self.num_heads * self.depth)
        output = self.norm1(output + tensor_A)
        output = self.norm2(self.final_linear(output) + output)
        return output


class CLIPConditioner(Conditioner):
    CLIP_MODELS = ["clip-vit-base-patch32"]

    def __init__(self, output_dim: int, clip_model_name: str = "clip-vit-base-patch32",
                 video_fps: int = 5, out_features: str = 128, enable_grad: bool = False,
                 in_features: int = 5000, project_out: bool = False,
                 mask_ratio: float = 0.0, mask_type: str = "input"):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim=768, output_dim=output_dim, project_out=project_out)

        sa_depth = 4
        num_heads = 16
        dim_head = 64
        hidden_scale = 4
        duration = 10
        fps = 5
        self.clip_model_name = 'clip-vit-base-patch32'
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type

        if self.clip_model_name == 'clip-vit-base-patch32':
            if self.mask_type == "input":
                in_features = round(50 * (1 - self.mask_ratio)) * fps * duration
            else:
                in_features = 50 * fps * duration
            out_features = 128
            temporal_dim = 768
            model_path = _clip_pretrained_path(self.clip_model_name)
            self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(model_path)
            self.proj = nn.Linear(in_features=in_features, out_features=out_features)
            self.in_features = in_features
            self.out_features = out_features
            self.SA_type = 'temporal_SA'
            if self.SA_type == 'temporal_SA':
                self.Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim * hidden_scale, 0.)
                self.Temp_pos_embedding = nn.Parameter(torch.randn(1, duration * fps, temporal_dim))
            elif self.SA_type == 'spatial_SA':
                self.Spatial_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim * hidden_scale, 0.)
                self.Spatial_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim))

            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            self.preprocess_CLIP = transforms.Compose([
                transforms.Normalize(mean=clip_mean, std=clip_std)
            ])

    def process_video_with_custom_preprocessing(self, video_tensor):
        video_tensor = video_tensor / 255.0
        video_tensor = self.preprocess_CLIP(video_tensor)
        return video_tensor

    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        new_model = {key.replace("module.", ""): val for key, val in model.items()}
        missing, unexpected = self.visual_encoder_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def random_masking_pad_zero(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=torch.argsort(ids_shuffle, dim=1))
        x_masked = x * (1 - mask.unsqueeze(-1))
        return x_masked, mask

    def mask_video_tensor(self, Video_tensors, mask_ratio=0.6, patch_size=32):
        batch_size, channels, height, width = Video_tensors.shape
        grid_size = height // patch_size
        num_patches = grid_size * grid_size
        num_patches_to_mask = int(num_patches * mask_ratio)
        mask_indices = torch.randperm(num_patches)[:num_patches_to_mask]
        masked_video = Video_tensors.clone()
        for i in range(batch_size):
            for idx in mask_indices:
                row = idx // grid_size
                col = idx % grid_size
                masked_video[i, :, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = 0
        return masked_video

    def forward(self, Video_tensors: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)

        Video_tensors = torch.cat(Video_tensors, dim=0).to(device)
        batch_size, time_length, _, _, _ = Video_tensors.size()
        Video_tensors = einops.rearrange(Video_tensors, 'b t c h w -> (b t) c h w')

        if self.mask_type == "input-patch":
            if self.mask_ratio > 0:
                if batch_size == 1:
                    self.mask_ratio = 0
                Video_tensors = self.mask_video_tensor(Video_tensors, self.mask_ratio)

        video_cond_pixel_values = self.process_video_with_custom_preprocessing(video_tensor=Video_tensors.to(device)).to(device)

        if self.clip_model_name == 'clip-vit-base-patch32':
            with torch.no_grad():
                outputs = visual_encoder_model(pixel_values=video_cond_pixel_values)
            video_hidden = outputs.last_hidden_state

            if self.mask_ratio > 0:
                class_token_embeddings = video_hidden[:, 0, :]
                if self.mask_type == "input":
                    video_hidden, mask, ids_restore = self.random_masking(video_hidden[:, 1:, :], self.mask_ratio)
                    video_hidden = torch.cat([class_token_embeddings.unsqueeze(1), video_hidden], dim=1)
                elif self.mask_type == "input-pad":
                    if batch_size == 1:
                        self.mask_ratio = 0
                    video_hidden, mask = self.random_masking_pad_zero(video_hidden[:, 1:, :], self.mask_ratio)
                    video_hidden = torch.cat([class_token_embeddings.unsqueeze(1), video_hidden], dim=1)

            if self.SA_type == 'temporal_SA':
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> (b q) t h', b=batch_size, t=time_length)
                video_hidden += self.Temp_pos_embedding
                video_hidden = self.Temp_transformer(video_hidden)
                video_hidden = einops.rearrange(video_hidden, '(b q) t h -> b (t q) h', b=batch_size, t=time_length)
            elif self.SA_type == 'spatial_SA':
                video_hidden += self.Spatial_pos_embedding
                video_hidden = self.Spatial_transformer(video_hidden)
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> b (t q) h', b=batch_size, t=time_length)

        video_hidden = proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)
        return video_hidden, torch.ones(video_hidden.shape[0], 1).to(device)


class CLIPWithSyncWithEmptyFeatureConditioner(Conditioner):
    CLIP_MODELS = ["clip-vit-base-patch32"]

    def __init__(self, output_dim: int, clip_model_name: str = "clip-vit-base-patch32",
                 video_fps: int = 5, out_features: str = 128, enable_grad: bool = False,
                 in_features: int = 5000, project_out: bool = False,
                 mask_ratio: float = 0.0, mask_type: str = "input",
                 sync_type: str = "add", temporal_transformer: bool = True):
        assert clip_model_name in self.CLIP_MODELS, f"Unknown clip model name: {clip_model_name}"
        super().__init__(dim=768, output_dim=output_dim, project_out=project_out)

        sa_depth = 4
        num_heads = 16
        dim_head = 64
        hidden_scale = 4
        duration = 10
        print(f"video_fps: {video_fps}")
        fps = video_fps
        self.clip_model_name = 'clip-vit-base-patch32'
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.sync_type = sync_type
        self.temporal_transformer = temporal_transformer

        if self.clip_model_name == 'clip-vit-base-patch32':
            if self.mask_type == "input":
                in_features = round(50 * (1 - self.mask_ratio)) * fps * duration
            else:
                in_features = 50 * fps * duration
            out_features = 128
            temporal_dim = 768

            self.empty_visual_feat = nn.Parameter(torch.zeros(1, out_features, temporal_dim), requires_grad=True)
            nn.init.constant_(self.empty_visual_feat, 0)

            model_path = _clip_pretrained_path(self.clip_model_name)
            self.visual_encoder_model = CLIPVisionModelWithProjection.from_pretrained(model_path)

            self.proj = nn.Linear(in_features=in_features, out_features=out_features)
            self.proj_sync = nn.Linear(in_features=240, out_features=out_features)
            if self.sync_type == 'add':
                self.sync_weight = nn.Parameter(torch.tensor(0.0))
            elif self.sync_type == "cross-attention":
                self.multi_head_cross_attention = MultiHeadCrossAttention(temporal_dim, 3)

            self.in_features = in_features
            self.out_features = out_features
            self.SA_type = 'temporal_SA'
            if self.temporal_transformer:
                if self.SA_type == 'temporal_SA':
                    self.Temp_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim * hidden_scale, 0.)
                    self.Temp_pos_embedding = nn.Parameter(torch.randn(1, duration * fps, temporal_dim))
                elif self.SA_type == 'spatial_SA':
                    self.Spatial_transformer = SA_Transformer(temporal_dim, sa_depth, num_heads, dim_head, temporal_dim * hidden_scale, 0.)
                    self.Spatial_pos_embedding = nn.Parameter(torch.randn(1, 50, temporal_dim))

            clip_mean = [0.48145466, 0.4578275, 0.40821073]
            clip_std = [0.26862954, 0.26130258, 0.27577711]
            self.preprocess_CLIP = transforms.Compose([
                transforms.Normalize(mean=clip_mean, std=clip_std)
            ])

    def process_video_with_custom_preprocessing(self, video_tensor):
        video_tensor = video_tensor / 255.0
        video_tensor = self.preprocess_CLIP(video_tensor)
        return video_tensor

    def init_first_from_ckpt(self, path):
        model = torch.load(path, map_location="cpu")
        if "state_dict" in list(model.keys()):
            model = model["state_dict"]
        new_model = {key.replace("module.", ""): val for key, val in model.items()}
        missing, unexpected = self.visual_encoder_model.load_state_dict(new_model, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def random_masking_pad_zero(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=torch.argsort(ids_shuffle, dim=1))
        x_masked = x * (1 - mask.unsqueeze(-1))
        return x_masked, mask

    def mask_video_tensor(self, Video_tensors, mask_ratio=0.6, patch_size=32):
        batch_size, channels, height, width = Video_tensors.shape
        grid_size = height // patch_size
        num_patches = grid_size * grid_size
        num_patches_to_mask = int(num_patches * mask_ratio)
        mask_indices = torch.randperm(num_patches)[:num_patches_to_mask]
        masked_video = Video_tensors.clone()
        for i in range(batch_size):
            for idx in mask_indices:
                row = idx // grid_size
                col = idx % grid_size
                masked_video[i, :, row * patch_size:(row + 1) * patch_size, col * patch_size:(col + 1) * patch_size] = 0
        return masked_video

    def forward(self, Video_list: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        Video_tensors = [item["video_tensors"] for item in Video_list]
        video_sync_frames = [item["video_sync_frames"] for item in Video_list]
        video_sync_frames = torch.cat(video_sync_frames, dim=0).to(device)

        visual_encoder_model = self.visual_encoder_model.eval().to(device)
        proj = self.proj.to(device)

        original_videos = torch.cat(Video_tensors, dim=0).to(device)
        batch_size, time_length, _, _, _ = original_videos.size()
        is_zero = torch.all(original_videos == 0, dim=(1, 2, 3, 4))
        Video_tensors = original_videos

        Video_tensors = einops.rearrange(Video_tensors, 'b t c h w -> (b t) c h w')

        if self.mask_type == "input-patch":
            if self.mask_ratio > 0:
                if batch_size == 1:
                    self.mask_ratio = 0
                Video_tensors = self.mask_video_tensor(Video_tensors, self.mask_ratio)

        video_cond_pixel_values = self.process_video_with_custom_preprocessing(video_tensor=Video_tensors.to(device)).to(device)

        if self.clip_model_name == 'clip-vit-base-patch32':
            with torch.no_grad():
                outputs = visual_encoder_model(pixel_values=video_cond_pixel_values)
            video_hidden = outputs.last_hidden_state

            if self.mask_ratio > 0:
                class_token_embeddings = video_hidden[:, 0, :]
                if self.mask_type == "input":
                    video_hidden, mask, ids_restore = self.random_masking(video_hidden[:, 1:, :], self.mask_ratio)
                    video_hidden = torch.cat([class_token_embeddings.unsqueeze(1), video_hidden], dim=1)
                elif self.mask_type == "input-pad":
                    if batch_size == 1:
                        self.mask_ratio = 0
                    video_hidden, mask = self.random_masking_pad_zero(video_hidden[:, 1:, :], self.mask_ratio)
                    video_hidden = torch.cat([class_token_embeddings.unsqueeze(1), video_hidden], dim=1)

            if self.temporal_transformer:
                if self.SA_type == 'temporal_SA':
                    video_hidden = einops.rearrange(video_hidden, '(b t) q h -> (b q) t h', b=batch_size, t=time_length)
                    video_hidden += self.Temp_pos_embedding
                    video_hidden = self.Temp_transformer(video_hidden)
                    video_hidden = einops.rearrange(video_hidden, '(b q) t h -> b (t q) h', b=batch_size, t=time_length)
                elif self.SA_type == 'spatial_SA':
                    video_hidden += self.Spatial_pos_embedding
                    video_hidden = self.Spatial_transformer(video_hidden)
                    video_hidden = einops.rearrange(video_hidden, '(b t) q h -> b (t q) h', b=batch_size, t=time_length)
            else:
                video_hidden = einops.rearrange(video_hidden, '(b t) q h -> b (t q) h', b=batch_size, t=time_length)

        video_hidden = proj(video_hidden.view(-1, self.in_features))
        video_hidden = video_hidden.view(batch_size, self.out_features, -1)

        video_sync_frames = self.proj_sync(video_sync_frames.view(-1, 240))
        video_sync_frames = video_sync_frames.view(batch_size, self.out_features, -1)

        if self.sync_type == 'add':
            video_hidden = video_hidden + self.sync_weight * video_sync_frames
        elif self.sync_type == 'cross-attention':
            video_hidden = self.multi_head_cross_attention(video_hidden, video_sync_frames)

        empty_visual_feat = self.empty_visual_feat.expand(batch_size, -1, -1)
        is_zero_expanded = is_zero.view(batch_size, 1, 1)
        video_hidden = torch.where(is_zero_expanded, empty_visual_feat, video_hidden)
        return video_hidden, torch.ones(video_hidden.shape[0], 1).to(device)


class T5Conditioner(Conditioner):

    T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
                 "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
                 "google/flan-t5-xl", "google/flan-t5-xxl"]

    T5_MODEL_DIMS = {
        "t5-small": 512, "t5-base": 768, "t5-large": 1024, "t5-3b": 1024, "t5-11b": 1024,
        "t5-xl": 2048, "t5-xxl": 4096,
        "google/flan-t5-small": 512, "google/flan-t5-base": 768, "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024, "google/flan-t5-11b": 1024,
        "google/flan-t5-xl": 2048, "google/flan-t5-xxl": 4096,
    }

    def __init__(self, output_dim: int, t5_model_name: str = "t5-base", max_length: str = 128,
                 enable_grad: bool = False, project_out: bool = False,
                 mask_ratio: float = 0, mask_type: str = "input"):
        assert t5_model_name in self.T5_MODELS, f"Unknown T5 model name: {t5_model_name}"
        super().__init__(self.T5_MODEL_DIMS[t5_model_name], output_dim, project_out=project_out)

        from transformers import T5EncoderModel, AutoTokenizer

        self.max_length = max_length
        self.enable_grad = enable_grad
        self.mask_ratio = mask_ratio

        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
                model = T5EncoderModel.from_pretrained(t5_model_name).train(enable_grad).requires_grad_(enable_grad).to(torch.float16)
            finally:
                logging.disable(previous_level)

        if self.enable_grad:
            self.model = model
        else:
            self.__dict__["model"] = model

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.model.to(device)
        self.proj_out.to(device)

        encoded = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)

        self.model.eval()
        with torch.cuda.amp.autocast(dtype=torch.float16), torch.set_grad_enabled(self.enable_grad):
            embeddings = self.model(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]

        embeddings = self.proj_out(embeddings.float())
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask


class PhonemeConditioner(Conditioner):
    """Conditioner that turns text into phonemes and embeds them using a lookup table (English only)."""

    def __init__(self, output_dim: int, max_length: int = 1024, project_out: bool = False):
        super().__init__(output_dim, output_dim, project_out=project_out)
        from g2p_en import G2p
        self.max_length = max_length
        self.g2p = G2p()
        self.phoneme_embedder = nn.Embedding(len(self.g2p.phonemes) + 2, output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.phoneme_embedder.to(device)
        self.proj_out.to(device)

        batch_phonemes = [self.g2p(text) for text in texts]
        phoneme_ignore = [" ", *string.punctuation]
        batch_phonemes = [[p if p not in phoneme_ignore else "_" for p in phonemes] for phonemes in batch_phonemes]
        phoneme_ids = [[self.g2p.p2idx[p] + 2 if p in self.g2p.p2idx else 1 for p in phonemes] for phonemes in batch_phonemes]
        longest = max([len(ids) for ids in phoneme_ids])
        phoneme_ids = [ids + [0] * (longest - len(ids)) for ids in phoneme_ids]
        phoneme_ids = torch.tensor(phoneme_ids).to(device)
        phoneme_embeds = self.phoneme_embedder(phoneme_ids)
        phoneme_embeds = self.proj_out(phoneme_embeds)
        return phoneme_embeds, torch.ones(phoneme_embeds.shape[0], phoneme_embeds.shape[1]).to(device)


class TokenizerLUTConditioner(Conditioner):
    """Conditioner that embeds text via a lookup table on a pretrained tokenizer's vocabulary."""

    def __init__(self, tokenizer_name: str, output_dim: int, max_length: int = 1024, project_out: bool = False):
        super().__init__(output_dim, output_dim, project_out=project_out)

        from transformers import AutoTokenizer
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            finally:
                logging.disable(previous_level)

        self.max_length = max_length
        self.token_embedder = nn.Embedding(len(self.tokenizer), output_dim)

    def forward(self, texts: tp.List[str], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)
        encoded = self.tokenizer(
            texts, truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device).to(torch.bool)
        embeddings = self.token_embedder(input_ids)
        embeddings = self.proj_out(embeddings)
        embeddings = embeddings * attention_mask.unsqueeze(-1).float()
        return embeddings, attention_mask


class PretransformConditioner(Conditioner):
    """Conditioner that uses a pretransform's encoder for conditioning."""

    def __init__(self, pretransform: Pretransform, output_dim: int):
        super().__init__(pretransform.encoded_channels, output_dim)
        self.pretransform = pretransform

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]],
                device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.pretransform.to(device)
        self.proj_out.to(device)
        if isinstance(audio, (list, tuple)):
            audio = torch.cat(audio, dim=0)
        audio = set_audio_channels(audio, self.pretransform.io_channels)
        latents = self.pretransform.encode(audio)
        latents = self.proj_out(latents)
        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]


def get_vocos_mel_spectrogram(waveform, n_fft=1024, n_mel_channels=100,
                              target_sample_rate=24000, hop_length=256, win_length=1024):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate, n_fft=n_fft, win_length=win_length,
        hop_length=hop_length, n_mels=n_mel_channels, power=1,
        center=True, normalized=False, norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.mean(dim=1, keepdim=True)
        waveform = waveform.squeeze(1)
    assert len(waveform.shape) == 2
    waveform = waveform.to(torch.float32)
    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class AudioMelConditioner(Conditioner):

    Mel_MODELS = ["mel_features", "vocos"]
    MEL_MODEL_DIMS = {"vocos": 768, "mel_features": 768}

    def __init__(self, output_dim: int, mel_spec_type: str = "mel_features", n_fft: int = 1024,
                 hop_length: int = 256, win_length: int = 1024, n_mel_channels: int = 100,
                 target_sample_rate: int = 24000, mask_ratio_start: float = 0.7,
                 mask_ratio_end: float = 1, project_out: bool = False):
        assert mel_spec_type in self.Mel_MODELS, f"Unknown mel model: {mel_spec_type}"
        super().__init__(self.MEL_MODEL_DIMS[mel_spec_type], output_dim, project_out=project_out)

        self.mel_spec_type = mel_spec_type
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        if mel_spec_type == "mel_features":
            self.extractor = get_vocos_mel_spectrogram
        self.proj_features = nn.Linear(in_features=self.n_mel_channels, out_features=768)

    def resample_and_pad(self, wavs, device):
        wavs = [wav.to(device).float() for wav in wavs]
        max_length = max([wav.shape[-1] for wav in wavs])
        padded_wavs = [torch.nn.functional.pad(wav, (0, max_length - wav.shape[-1])) for wav in wavs]
        stacked_wavs = torch.stack(padded_wavs)
        resampler = torchaudio.transforms.Resample(orig_freq=44100, new_freq=self.target_sample_rate).to(device)
        return resampler(stacked_wavs)

    def mask_mel_spectrogram(self, mels: torch.Tensor) -> torch.Tensor:
        batch_size, n_mels, seq_len = mels.shape
        device = mels.device
        mask_ratios = torch.rand(batch_size, device=device) * (self.mask_ratio_end - self.mask_ratio_start) + self.mask_ratio_start
        masked_mels = mels.clone()
        for i in range(batch_size):
            mask_len = int(seq_len * mask_ratios[i])
            start_pos = torch.randint(0, seq_len - mask_len + 1, (1,), device=device)
            masked_mels[i, :, start_pos:start_pos + mask_len] = 0
        return masked_mels, mask_ratios

    def forward(self, wavs: tp.List[torch.tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.proj_out.to(device)
        wavs = torch.cat(wavs, dim=0).to(device).float()
        mels = self.extractor(
            waveform=wavs, n_fft=self.n_fft, n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate, hop_length=self.hop_length, win_length=self.win_length,
        )
        if self.mask_ratio_start < self.mask_ratio_end:
            mels, mask_ratios = self.mask_mel_spectrogram(mels)
        mels = mels.transpose(1, 2)
        mels_emb = self.proj_features(mels)
        return mels_emb, torch.ones(mels_emb.shape[0], 1).to(device)


class AudioAutoencoderConditioner(Conditioner):
    """Conditioner that uses a pretransform's encoder for audio conditioning."""

    def __init__(self, pretransform: Pretransform, output_dim: int,
                 latent_seq_len: int = 237, mask_ratio_start: float = 0, mask_ratio_end: float = 0):
        super().__init__(pretransform.encoded_channels, output_dim)
        self.pretransform = pretransform
        self.latent_seq_len = latent_seq_len
        self.mask_ratio_start = mask_ratio_start
        self.mask_ratio_end = mask_ratio_end
        self.proj_features_128 = nn.Linear(in_features=self.latent_seq_len, out_features=128)

    def mask_audio(self, audio: torch.Tensor) -> torch.Tensor:
        batch_size, channels, seq_len = audio.shape
        device = audio.device
        mask_ratios = torch.rand(batch_size, device=device) * (self.mask_ratio_end - self.mask_ratio_start) + self.mask_ratio_start
        masked_audio = audio.clone()
        for i in range(batch_size):
            mask_len = int(seq_len * mask_ratios[i])
            start_pos = torch.randint(0, seq_len - mask_len + 1, (1,), device=device)
            masked_audio[i, :, start_pos:start_pos + mask_len] = 0
        return masked_audio, mask_ratios

    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]],
                device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.pretransform.to(device)
        self.proj_out.to(device)
        bs = len(audio)
        max_len = max([a.shape[-1] for a in audio])
        for i in range(bs):
            audio[i] = audio[i].to(device)
            pad_len = max_len - audio[i].shape[-1]
            if pad_len > 0:
                audio[i] = torch.nn.functional.pad(audio[i], (0, pad_len))
        audio = torch.cat(audio, dim=0)
        audio = set_audio_channels(audio, self.pretransform.io_channels)
        if self.mask_ratio_start < self.mask_ratio_end:
            audio, mask_ratios = self.mask_audio(audio)
        latents = self.pretransform.encode(audio)
        latents = self.proj_features_128(latents)
        latents = latents.permute(0, 2, 1)
        latents = self.proj_out(latents)
        return latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)


class MultiConditioner(nn.Module):
    """Applies multiple conditioners to an input dictionary based on the keys."""

    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}):
        super().__init__()
        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]],
                device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}
        for key, conditioner in self.conditioners.items():
            condition_key = key
            conditioner_inputs = []
            for x in batch_metadata:
                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")
                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]
                else:
                    conditioner_input = x[condition_key]
                conditioner_inputs.append(conditioner_input)
            output[key] = conditioner(conditioner_inputs, device)
        return output


def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any]) -> MultiConditioner:
    """Create a MultiConditioner from a conditioning config dictionary."""
    conditioners = {}
    cond_dim = config["cond_dim"]
    default_keys = config.get("default_keys", {})

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]
        conditioner_type = conditioner_info["type"]
        conditioner_config = {"output_dim": cond_dim}
        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "t5":
            conditioners[id] = T5Conditioner(**conditioner_config)
        elif conditioner_type == "clip" or conditioner_type == "mask-clip":
            conditioners[id] = CLIPConditioner(**conditioner_config)
        elif conditioner_type == "mel_spec-w-empty-feature":
            conditioners[id] = AudioMelConditioner(**conditioner_config)
        elif conditioner_type == "clip-with-sync-w-empty-feat":
            conditioners[id] = CLIPWithSyncWithEmptyFeatureConditioner(**conditioner_config)
        elif conditioner_type == "clap_text":
            conditioners[id] = CLAPTextConditioner(**conditioner_config)
        elif conditioner_type == "clap_audio":
            conditioners[id] = CLAPAudioConditioner(**conditioner_config)
        elif conditioner_type == "int":
            conditioners[id] = IntConditioner(**conditioner_config)
        elif conditioner_type == "number":
            conditioners[id] = NumberConditioner(**conditioner_config)
        elif conditioner_type == "phoneme":
            conditioners[id] = PhonemeConditioner(**conditioner_config)
        elif conditioner_type == "lut":
            conditioners[id] = TokenizerLUTConditioner(**conditioner_config)
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"
            pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))
            conditioners[id] = PretransformConditioner(pretransform, **conditioner_config)
        elif conditioner_type == "audio_autoencoder":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"
            pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")))
            conditioners[id] = AudioAutoencoderConditioner(pretransform, **conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys)

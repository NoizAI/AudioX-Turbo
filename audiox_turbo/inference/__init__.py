import json

import torch
from safetensors.torch import load_file

from ..models.factory import create_model_from_config


def _read_state_dict(path):
    """Read a checkpoint into a flat ``{name: tensor}`` dict.

    Handles raw state dicts (e.g. the distilled ``audiox_turbo.ckpt`` student),
    Lightning-style ``{"state_dict": ...}`` checkpoints, and ``.safetensors`` files.
    """
    path = str(path)
    if path.endswith(".safetensors"):
        raw = load_file(path)
    else:
        raw = torch.load(path, map_location="cpu")
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "student", "student_state_dict"):
            value = raw.get(key)
            if isinstance(value, dict):
                return value
    return raw


def _copy_matching_state_dict(model, state_dict):
    """Copy parameters by name, tolerating common prefixes (module./model./student.)."""
    model_state = model.state_dict()
    prefixes = ("module.", "model.", "student.", "student.model.")
    loaded = 0
    for key, value in state_dict.items():
        candidates = [key]
        for prefix in prefixes:
            if key.startswith(prefix):
                candidates.append(key[len(prefix):])
        for candidate in candidates:
            if candidate in model_state and model_state[candidate].shape == value.shape:
                model_state[candidate] = value.data if isinstance(value, torch.nn.Parameter) else value
                loaded += 1
                break
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    return loaded, missing, unexpected


def load_audiox_turbo_model(model_config, ckpt_path, pretransform_ckpt_path=None,
                            device="cuda", verbose=True):
    """Build the AudioX-Turbo model and load the distilled student weights.

    Args:
        model_config: path to a model config JSON, or an already-loaded config dict.
        ckpt_path: path to the AudioX-Turbo student checkpoint (``audiox_turbo.ckpt``).
        pretransform_ckpt_path: optional path to the VAE pretransform (``vae.ckpt``).
        device: device to move the model onto.

    Returns:
        (model, model_config)
    """
    if isinstance(model_config, (str, bytes)):
        with open(model_config, "r", encoding="utf-8") as f:
            model_config = json.load(f)

    model = create_model_from_config(model_config)

    loaded, missing, unexpected = _copy_matching_state_dict(model, _read_state_dict(ckpt_path))
    if verbose:
        print(f"[AudioX-Turbo] loaded {ckpt_path}: matched={loaded} "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    if pretransform_ckpt_path is not None and model.pretransform is not None:
        model.pretransform.load_state_dict(_read_state_dict(pretransform_ckpt_path), strict=False)
        if verbose:
            print(f"[AudioX-Turbo] loaded pretransform {pretransform_ckpt_path}")

    model = model.to(device).eval().requires_grad_(False)
    return model, model_config

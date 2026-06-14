import argparse
import configparser
import glob
import hashlib
import json
import os
import random
import shlex

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
HF_HOME = os.path.join(REPO_ROOT, "checkpoints", "huggingface")
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(HF_HOME, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_HOME, "hub"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault(
    "AUDIOX_TURBO_CLIP_MODEL_PATH",
    os.path.join(
        HF_HOME,
        "hub",
        "models--openai--clip-vit-base-patch32",
        "snapshots",
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
    ),
)

import pytorch_lightning as pl
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from audiox_turbo.data.dataset import create_dataloader_from_config
from audiox_turbo.models import create_model_from_config
from audiox_turbo.models.utils import load_ckpt_state_dict, remove_weight_norm_from_model
from audiox_turbo.training.utils import copy_state_dict
from audiox_turbo.training.diffusion_lightning_dmd_withgan_noisylatent_ablation import DiffusionDMDTrainingWrapperAblation
from audiox_turbo.models.transformer_ablation import ContinuousMMDiTTransformerAblation
import audiox_turbo.models.dit as dit_mod

torch.backends.cudnn.allow_tf32 = False


def _coerce_ini_value(value):
    value = value.strip()
    if value == "":
        return value
    try:
        return shlex.split(value)[0]
    except ValueError:
        return value.strip("\"'")


def _load_defaults(defaults_path="defaults.ini"):
    defaults = {
        "name": "audiox_turbo",
        "batch_size": 8,
        "num_gpus": 1,
        "num_nodes": 1,
        "strategy": "",
        "precision": "16-mixed",
        "num_workers": 8,
        "seed": 42,
        "accum_batches": 1,
        "checkpoint_every": 10000,
        "ckpt_path": "",
        "pretrained_ckpt_path": "",
        "pretransform_ckpt_path": "",
        "model_config": "",
        "dataset_config": "",
        "save_dir": "",
        "gradient_clip_val": 0.0,
        "remove_pretransform_weight_norm": "",
    }
    if not os.path.exists(defaults_path):
        return defaults

    parser = configparser.ConfigParser()
    parser.read(defaults_path)
    for key, value in parser["DEFAULTS"].items():
        defaults[key] = _coerce_ini_value(value)
    return defaults


def get_train_args():
    defaults = _load_defaults()
    parser = argparse.ArgumentParser(description="Train AudioX-Turbo with DMD + GAN distillation.")
    parser.add_argument("--name", default=defaults["name"])
    parser.add_argument("--batch-size", type=int, default=int(defaults["batch_size"]))
    parser.add_argument("--num-gpus", type=int, default=int(defaults["num_gpus"]))
    parser.add_argument("--num-nodes", type=int, default=int(defaults["num_nodes"]))
    parser.add_argument("--strategy", default=defaults["strategy"])
    parser.add_argument("--precision", default=defaults["precision"])
    parser.add_argument("--num-workers", type=int, default=int(defaults["num_workers"]))
    parser.add_argument("--seed", type=int, default=int(defaults["seed"]))
    parser.add_argument("--accum-batches", type=int, default=int(defaults["accum_batches"]))
    parser.add_argument("--checkpoint-every", type=int, default=int(defaults["checkpoint_every"]))
    parser.add_argument("--ckpt-path", default=defaults["ckpt_path"])
    parser.add_argument("--pretrained-ckpt-path", default=defaults["pretrained_ckpt_path"])
    parser.add_argument("--pretransform-ckpt-path", default=defaults["pretransform_ckpt_path"])
    parser.add_argument("--model-config", required=not bool(defaults["model_config"]), default=defaults["model_config"])
    parser.add_argument("--dataset-config", required=not bool(defaults["dataset_config"]), default=defaults["dataset_config"])
    parser.add_argument("--save-dir", required=not bool(defaults["save_dir"]), default=defaults["save_dir"])
    parser.add_argument("--gradient-clip-val", type=float, default=float(defaults["gradient_clip_val"]))
    parser.add_argument("--remove-pretransform-weight-norm", default=defaults["remove_pretransform_weight_norm"])
    return parser.parse_args()


def apply_mmdit_ablation_patch():
    dit_mod.ContinuousMMDiTTransformer = ContinuousMMDiTTransformerAblation


def infer_feature_channels_from_ckpt(ckpt_path, fallback=64):
    if not ckpt_path or not os.path.exists(ckpt_path):
        return fallback
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
    except Exception:
        return fallback
    for key in ("model.model.preprocess_conv.weight", "preprocess_conv.weight"):
        if key in state_dict and hasattr(state_dict[key], "shape"):
            return int(state_dict[key].shape[0])
    return fallback


def generate_run_id(args, model_config, dataset_config):
    relevant_args = {
        "name": args.name,
        "pretrained_ckpt_path": args.pretrained_ckpt_path,
        "model_config": model_config,
        "dataset_config": dataset_config,
    }
    return hashlib.md5(json.dumps(relevant_args, sort_keys=True).encode("utf-8")).hexdigest()[:8]


def is_valid_checkpoint(ckpt_path):
    if os.path.isdir(ckpt_path):
        return True
    return os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 1024


def assert_and_fix_contiguous(module, name_prefix=""):
    has_fixed = False
    for _, param in module.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
            has_fixed = True
    for _, buffer in module.named_buffers():
        if not buffer.is_contiguous():
            buffer.data = buffer.data.contiguous()
            has_fixed = True
    if has_fixed:
        print(f"[DeepSpeed Fix] {name_prefix} fixed successfully.")


def _parse_bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _parse_timestep_probs(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(x.strip()) for x in str(v).split(",") if x.strip()]


def _collect_lora_target_modules(inner_model):
    target_modules_set = set()
    for name, module in inner_model.named_modules():
        if isinstance(module, nn.Linear):
            target_name = name.split(".")[-1]
            if not target_name.isdigit():
                target_modules_set.add(target_name)
    target_modules_list = list(target_modules_set)
    if not target_modules_list:
        target_modules_list = ["to_q", "to_k", "to_v", "to_out", "net", "proj_out", "proj_in"]
    return target_modules_list


def apply_teacher_gan_lora(teacher_model, lora_r, lora_alpha):
    """
    Wrap teacher inner diffusion trunk (teacher.model) with LoRA for GAN backbone tuning only.
    Base weights stay frozen; only adapter weights are trainable (enforced again in the wrapper).
    """
    inner = teacher_model.model
    targets = _collect_lora_target_modules(inner)
    print(f"[GAN-Teacher-LoRA] target_modules={targets}, r={lora_r}, alpha={lora_alpha}")
    peft_config = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        target_modules=targets,
        lora_dropout=0.05,
        bias="none",
    )
    teacher_model.model = get_peft_model(inner, peft_config)
    teacher_model.model.print_trainable_parameters()


def _resolve_teacher_ckpt_path(path):
    if not path:
        return path
    if os.path.isfile(path):
        return path
    if path.endswith(".ckp"):
        alt = path[:-4] + ".ckpt"
        if os.path.isfile(alt):
            print(f"[AudioX-Turbo] teacher_ckpt_path: using {alt} (fixed .ckp -> .ckpt)")
            return alt
    return path


def _env_int(name, default=None):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(value)


def create_dmd_training_wrapper(model_config, model, pretrained_ckpt_path, ablation_overrides):
    training_config = model_config.get("training", {})
    diffusion_depth = model_config.get("model", {}).get("diffusion", {}).get("config", {}).get("depth", 24)
    feature_channels = infer_feature_channels_from_ckpt(
        pretrained_ckpt_path, fallback=int(model_config.get("io_channels", 64))
    )
    print(f"[AudioX-Turbo] inferred feature_channels from ckpt: {feature_channels}")

    teacher_model = create_model_from_config(model_config)
    teacher_ckpt_path = _resolve_teacher_ckpt_path(training_config.get("teacher_ckpt_path"))
    if teacher_ckpt_path:
        try:
            ckpt = torch.load(teacher_ckpt_path, map_location="cpu")
            sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
            teacher_model.load_state_dict(sd, strict=False)
        except Exception as e:
            print(f"[Warning] Failed to load teacher checkpoint: {e}")

    if ablation_overrides["gan_backbone_trainable"]:
        apply_teacher_gan_lora(
            teacher_model,
            ablation_overrides["gan_backbone_lora_r"],
            ablation_overrides["gan_backbone_lora_alpha"],
        )

    fake_model = create_model_from_config(model_config)
    fake_model.load_state_dict(model.state_dict())

    return DiffusionDMDTrainingWrapperAblation(
        model=model,
        teacher_model=teacher_model,
        fake_model=fake_model,
        lr=training_config.get("learning_rate", 1e-5),
        optimizer_configs=training_config.get("optimizer_configs", None),
        pre_encoded=training_config.get("pre_encoded", False),
        cfg_dropout_prob=training_config.get("cfg_dropout_prob", 0.1),
        student_sample_steps=training_config.get("dmd_sample_steps", 4),
        dmd_loss_weight=training_config.get("dmd_loss_weight", 1.0),
        fake_loss_weight=training_config.get("fake_loss_weight", 1.0),
        use_gan=training_config.get("use_gan", True),
        gan_loss_weight=training_config.get("gan_loss_weight", 1.0),
        feature_channels=feature_channels,
        timestep_sampling_probs=ablation_overrides["timestep_sampling_probs"],
        gan_discriminator_head_mode=ablation_overrides["gan_discriminator_head_mode"],
        gan_backbone_num_blocks=ablation_overrides["gan_backbone_num_blocks"],
        gan_backbone_total_blocks=diffusion_depth,
        gan_backbone_trainable=ablation_overrides["gan_backbone_trainable"],
        teacher_guidance_scale=training_config.get("teacher_guidance_scale", 6.0),
        mask_padding=training_config.get("mask_padding", False),
        mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
        use_ema=training_config.get("use_ema", True),
        update_ratio_s_f=training_config.get("update_ratio_s_f", 5),
    )


class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config


def main():
    apply_mmdit_ablation_patch()
    args = get_train_args()
    rank_offset = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    seed = args.seed + rank_offset
    random.seed(seed)
    torch.manual_seed(seed)

    with open(args.model_config) as f:
        model_config = json.load(f)
    with open(args.dataset_config) as f:
        dataset_config = json.load(f)

    training_cfg = model_config.get("training", {})
    ablation_overrides = {
        "timestep_sampling_probs": _parse_timestep_probs(
            os.environ.get("TIMESTEP_SAMPLING_PROBS", training_cfg.get("timestep_sampling_probs", None))
        ),
        "gan_discriminator_head_mode": os.environ.get(
            "GAN_DISC_HEAD_MODE", training_cfg.get("gan_discriminator_head_mode", "last_block")
        ),
        "gan_backbone_num_blocks": int(
            os.environ.get("GAN_BACKBONE_NUM_BLOCKS", training_cfg.get("gan_backbone_num_blocks", -1))
        ),
        "gan_backbone_trainable": _parse_bool(
            os.environ.get("GAN_BACKBONE_TRAINABLE", training_cfg.get("gan_backbone_trainable", False)), default=False
        ),
        "gan_backbone_lora_r": int(
            os.environ.get("GAN_BACKBONE_LORA_R", training_cfg.get("gan_backbone_lora_r", 16))
        ),
        "gan_backbone_lora_alpha": int(
            os.environ.get("GAN_BACKBONE_LORA_ALPHA", training_cfg.get("gan_backbone_lora_alpha", 32))
        ),
    }
    print(
        "[AudioX-Turbo Args] "
        f"timestep_sampling_probs={ablation_overrides['timestep_sampling_probs']}, "
        f"gan_discriminator_head_mode={ablation_overrides['gan_discriminator_head_mode']}, "
        f"gan_backbone_num_blocks={ablation_overrides['gan_backbone_num_blocks']}, "
        f"gan_backbone_trainable={ablation_overrides['gan_backbone_trainable']}, "
        f"gan_backbone_lora_r={ablation_overrides['gan_backbone_lora_r']}, "
        f"gan_backbone_lora_alpha={ablation_overrides['gan_backbone_lora_alpha']}"
    )

    train_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 2),
        video_fps=model_config.get("video_fps", 5),
    )

    model = create_model_from_config(model_config)
    if args.pretrained_ckpt_path:
        ckpt_sd = load_ckpt_state_dict(args.pretrained_ckpt_path)
        copy_state_dict(model, ckpt_sd)
        del ckpt_sd
    if args.pretransform_ckpt_path:
        model.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))
    if args.remove_pretransform_weight_norm == "post_load":
        remove_weight_norm_from_model(model.pretransform)

    if model_config.get("model_type", None) == "diffusion_dmd":
        training_wrapper = create_dmd_training_wrapper(model_config, model, args.pretrained_ckpt_path, ablation_overrides)
    else:
        from audiox_turbo.training import create_training_wrapper_from_config
        training_wrapper = create_training_wrapper_from_config(model_config, model)

    run_id = generate_run_id(args, model_config, dataset_config)
    tensorboard_logger = pl.loggers.TensorBoardLogger(save_dir=os.path.join(args.save_dir, "logs"), name=args.name, version=run_id)
    checkpoint_dir = os.path.join(args.save_dir, args.name, run_id, "checkpoints") if args.save_dir and run_id else None
    ckpt_path = args.ckpt_path if (hasattr(args, "ckpt_path") and args.ckpt_path) else None
    if checkpoint_dir and os.path.exists(checkpoint_dir) and ckpt_path is None:
        checkpoint_paths = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
        checkpoint_paths = [cp for cp in checkpoint_paths if is_valid_checkpoint(cp)]
        if checkpoint_paths:
            checkpoint_paths.sort(key=os.path.getmtime, reverse=True)
            ckpt_path = checkpoint_paths[0]
            print(f"Found existing checkpoint at {ckpt_path}, resuming.")
    if ckpt_path == "":
        ckpt_path = None

    max_steps = _env_int("AUDIOX_TURBO_MAX_STEPS", -1)
    if max_steps and max_steps > 0:
        print(f"[AudioX-Turbo] limiting training to max_steps={max_steps}")

    trainer = pl.Trainer(
        devices=args.num_gpus,
        accelerator="gpu",
        num_nodes=args.num_nodes,
        strategy=(args.strategy if args.strategy else ("ddp_find_unused_parameters_true" if args.num_gpus > 1 else "auto")),
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches,
        callbacks=[
            pl.callbacks.ModelCheckpoint(every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1, save_last=True),
            ModelConfigEmbedderCallback(model_config),
        ],
        logger=tensorboard_logger,
        log_every_n_steps=1,
        max_steps=max_steps,
        max_epochs=10000000,
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=1,
    )

    assert_and_fix_contiguous(training_wrapper, name_prefix="Wrapper.")
    trainer.fit(training_wrapper, train_dl, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()

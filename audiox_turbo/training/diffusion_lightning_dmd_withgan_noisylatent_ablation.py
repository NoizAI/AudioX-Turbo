import random
from contextlib import ExitStack, contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from audiox_turbo.training.utils import create_optimizer_from_config, create_scheduler_from_config
from . import diffusion_lightning_dmd_withgan_noisylatent as base_mod


class DiffusionDMDTrainingWrapperAblation(base_mod.DiffusionDMDTrainingWrapper):
    def __init__(
        self,
        *args,
        timestep_sampling_probs=None,
        gan_discriminator_head_mode="last_block",
        gan_backbone_num_blocks=-1,
        gan_backbone_total_blocks=24,
        gan_backbone_trainable=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.gan_discriminator_head_mode = gan_discriminator_head_mode
        self.gan_backbone_trainable = bool(gan_backbone_trainable)
        self.gan_backbone_total_blocks = max(1, int(gan_backbone_total_blocks))
        if gan_backbone_num_blocks is None or int(gan_backbone_num_blocks) <= 0:
            self.gan_backbone_num_blocks = self.gan_backbone_total_blocks
        else:
            self.gan_backbone_num_blocks = max(1, min(int(gan_backbone_num_blocks), self.gan_backbone_total_blocks))

        if timestep_sampling_probs is None:
            timestep_sampling_probs = [1.0 / self.student_sample_steps] * self.student_sample_steps
        self.timestep_sampling_probs = self._normalize_timestep_probs(timestep_sampling_probs, self.student_sample_steps)

        if self.use_gan and self.gan_discriminator_head_mode == "all_blocks":
            self.discriminator_heads = nn.ModuleList(
                [base_mod.DiscriminatorHead(input_channels=self.feature_channels) for _ in range(self.gan_backbone_total_blocks)]
            )
            self.discriminator_heads.train()
            self.discriminator_head = None
        else:
            self.discriminator_heads = None

        if self.gan_backbone_trainable:
            # Teacher trunk uses PEFT LoRA only (applied in train script); never full finetune.
            self.teacher.train()
            for name, param in self.teacher.named_parameters():
                param.requires_grad = "lora_" in name
        else:
            self.teacher.eval()
            self.teacher.requires_grad_(False)

    def _normalize_timestep_probs(self, probs, num_steps):
        if isinstance(probs, str):
            probs = [float(v.strip()) for v in probs.split(",") if v.strip() != ""]
        probs = list(probs)
        if len(probs) != num_steps:
            raise ValueError(f"timestep_sampling_probs length={len(probs)} must equal student_sample_steps={num_steps}")
        probs_tensor = torch.tensor(probs, dtype=torch.float32)
        if torch.any(probs_tensor < 0):
            raise ValueError("timestep_sampling_probs must be non-negative")
        total = probs_tensor.sum().item()
        if total <= 0:
            raise ValueError("timestep_sampling_probs sum must be > 0")
        return (probs_tensor / total).tolist()

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs["diffusion"]
        params = list(self.student.parameters()) + list(self.fake.parameters())
        if self.use_gan and self.discriminator_head is not None:
            params += list(self.discriminator_head.parameters())
        if self.use_gan and self.discriminator_heads is not None:
            for head in self.discriminator_heads:
                params += list(head.parameters())
        if self.use_gan and self.gan_backbone_trainable:
            params += [p for p in self.teacher.parameters() if p.requires_grad]

        opt = create_optimizer_from_config(diffusion_opt_config["optimizer"], params)
        schedulers = []
        if "scheduler" in diffusion_opt_config:
            sched = create_scheduler_from_config(diffusion_opt_config["scheduler"], opt)
            schedulers.append({"scheduler": sched, "interval": "step"})
        return [opt], schedulers

    def compute_discriminator_score(self, x0, cond, extra_args, use_grad_on_x0=False):
        if not self.use_gan:
            return None
        batch_size = x0.shape[0]
        t_disc = torch.rand(batch_size, device=self.device) * 0.2
        x_in = x0 if use_grad_on_x0 else x0.detach()
        z_disc, _ = base_mod.get_zt_from_x0(x_in, t_disc, self.diffusion_objective)
        train_teacher_backbone = (not use_grad_on_x0) and self.gan_backbone_trainable

        def _teacher_forward(kwargs):
            if use_grad_on_x0:
                with base_mod.toggle_grad(self.teacher, False):
                    return self.teacher(z_disc, t_disc, cond=cond, cfg_dropout_prob=0.0, cfg_scale=1.0, **kwargs)
            if train_teacher_backbone:
                return self.teacher(z_disc, t_disc, cond=cond, cfg_dropout_prob=0.0, cfg_scale=1.0, **kwargs)
            with torch.no_grad():
                return self.teacher(z_disc, t_disc, cond=cond, cfg_dropout_prob=0.0, cfg_scale=1.0, **kwargs)

        if self.gan_discriminator_head_mode == "all_blocks" and self.discriminator_heads is not None:
            all_scores = []
            for block_idx in range(1, self.gan_backbone_num_blocks + 1):
                teacher_output = _teacher_forward({**extra_args, "active_num_layers": block_idx})
                all_scores.append(self.discriminator_heads[block_idx - 1](teacher_output))
            return torch.stack(all_scores, dim=0).mean(dim=0)

        teacher_output = _teacher_forward({**extra_args, "active_num_layers": self.gan_backbone_num_blocks})
        return self.discriminator_head(teacher_output)

    @contextmanager
    def _dmd_teacher_pretrained_only_ctx(self):
        """
        DMD 的 teacher 信号必须与「预训练 base」一致：不经过 GAN 上训练的 LoRA。
        teacher.model 为 PEFT 包装时使用 disable_adapter；否则为空操作。
        """
        inner = getattr(self.teacher, "model", None)
        if inner is not None and hasattr(inner, "disable_adapter"):
            with inner.disable_adapter():
                yield
        else:
            yield

    @contextmanager
    def _freeze_disc_heads_ctx(self):
        with ExitStack() as stack:
            if self.discriminator_head is not None:
                stack.enter_context(base_mod.toggle_grad(self.discriminator_head, False))
            elif self.discriminator_heads is not None:
                for h in self.discriminator_heads:
                    stack.enter_context(base_mod.toggle_grad(h, False))
            yield

    def training_step(self, batch, batch_idx):
        old_randint = random.randint

        def _sample_timestep_idx(a, b):
            _ = (a, b)
            probs = torch.tensor(self.timestep_sampling_probs, device=self.device, dtype=torch.float32)
            return int(torch.multinomial(probs, num_samples=1).item())

        random.randint = _sample_timestep_idx
        try:
            return self._training_step_ablation(batch, batch_idx)
        finally:
            random.randint = old_randint

    def _training_step_ablation(self, batch, batch_idx):
        """与 base 一致，但 DMD 的 teacher 前向强制关闭 LoRA；GAN 仍用带 LoRA 的 teacher。"""
        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout
        if use_padding_mask:
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device)
        else:
            padding_masks = None

        if self.student.pretransform is not None:
            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.student.pretransform.enable_grad):
                    self.student.pretransform.train(self.student.pretransform.enable_grad)
                    self.student.pretransform = self.student.pretransform.to(dtype=torch.float32).eval()

                    with torch.cuda.amp.autocast(enabled=False):
                        diffusion_input = self.student.pretransform.encode(diffusion_input.to(dtype=torch.float32))

                    if use_padding_mask:
                        padding_masks = F.interpolate(
                            padding_masks.unsqueeze(1).float(),
                            size=diffusion_input.shape[2],
                            mode="nearest",
                        ).squeeze(1).bool()
            else:
                if hasattr(self.student.pretransform, "scale") and self.student.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.student.pretransform.scale

        batch_size = diffusion_input.shape[0]

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                student_cond = self.student.conditioner(metadata, self.device)
                teacher_cond = self.teacher.conditioner(metadata, self.device)

        extra_args = {}

        cycle_len = self.update_ratio_s_f + 1
        is_student_update = (self.global_step % cycle_len) < self.update_ratio_s_f

        target_step_idx = random.randint(0, self.student_sample_steps - 1)
        initial_noise = torch.randn_like(diffusion_input)

        with torch.cuda.amp.autocast():
            x0_student = self.iterative_dmd_forward(initial_noise, target_step_idx, student_cond, extra_args)

        if is_student_update:
            t_dmd = self.rng.draw(batch_size)[:, 0].to(self.device)
            z_t_dmd, _ = base_mod.get_zt_from_x0(x0_student, t_dmd, self.diffusion_objective)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    with self._dmd_teacher_pretrained_only_ctx():
                        v_teacher = self.teacher(
                            z_t_dmd,
                            t_dmd,
                            cond=teacher_cond,
                            cfg_dropout_prob=0.0,
                            cfg_scale=self.teacher_guidance_scale,
                            rescale_cfg=True,
                            **extra_args,
                        )
                    x0_teacher = base_mod.get_x0_from_noise(z_t_dmd, v_teacher, t_dmd, self.diffusion_objective)

                    v_fake = self.fake(
                        z_t_dmd,
                        t_dmd,
                        cond=student_cond,
                        cfg_dropout_prob=0.0,
                        cfg_scale=1.0,
                        **extra_args,
                    )
                    x0_fake = base_mod.get_x0_from_noise(z_t_dmd, v_fake, t_dmd, self.diffusion_objective)

            p_real = z_t_dmd - x0_teacher
            p_fake = z_t_dmd - x0_fake
            normalization = torch.abs(p_real).mean(dim=[1, 2], keepdim=True) + 1e-8
            grad = (p_real - p_fake) / normalization
            grad = torch.nan_to_num(grad)

            if use_padding_mask:
                mask_broadcast = padding_masks.unsqueeze(1).type_as(grad)
                grad = grad * mask_broadcast

            loss_dmd = 0.5 * F.mse_loss(x0_student, (x0_student - grad).detach(), reduction="mean") * self.dmd_loss_weight

            total_loss = loss_dmd
            log_dict = {"train/loss_dmd": loss_dmd}

            if self.use_gan:
                with self._freeze_disc_heads_ctx():
                    with torch.cuda.amp.autocast():
                        logits_fake = self.compute_discriminator_score(
                            x0_student, teacher_cond, extra_args, use_grad_on_x0=True
                        )
                        loss_gan_gen = -torch.mean(logits_fake) * self.gan_loss_weight
                        total_loss = total_loss + loss_gan_gen
                        log_dict["train/gan_g_loss"] = loss_gan_gen

            log_dict.update({"train/total_loss": total_loss, "train/update_mode": 0.0})
            self.log_dict(log_dict, prog_bar=True, on_step=True)
            return total_loss

        target_x0 = x0_student.detach()

        t_fake_train = self.rng.draw(batch_size)[:, 0].to(self.device)
        z_t_fake, noise_epsilon = base_mod.get_zt_from_x0(target_x0, t_fake_train, self.diffusion_objective)
        target_v = base_mod.get_target_v_from_x0(
            target_x0, z_t_fake, t_fake_train, self.diffusion_objective, noise=noise_epsilon
        )

        with torch.cuda.amp.autocast():
            pred_v_fake = self.fake(
                z_t_fake,
                t_fake_train,
                cond=student_cond,
                cfg_dropout_prob=0.0,
                cfg_scale=1.0,
                **extra_args,
            )
            loss_raw = F.mse_loss(pred_v_fake, target_v, reduction="none")

            if use_padding_mask:
                mask_broadcast = padding_masks.unsqueeze(1).type_as(loss_raw)
                loss_fake_model = (loss_raw * mask_broadcast).sum() / (mask_broadcast.sum() + 1e-8)
            else:
                loss_fake_model = loss_raw.mean()

            loss_fake_model = loss_fake_model * self.fake_loss_weight

        total_loss = loss_fake_model
        log_dict = {"train/loss_fake": loss_fake_model}

        if self.use_gan:
            with torch.cuda.amp.autocast():
                logits_real = self.compute_discriminator_score(diffusion_input, teacher_cond, extra_args, use_grad_on_x0=False)
                logits_fake = self.compute_discriminator_score(target_x0, teacher_cond, extra_args, use_grad_on_x0=False)
                loss_d_real = torch.mean(F.relu(1.0 - logits_real))
                loss_d_fake = torch.mean(F.relu(1.0 + logits_fake))
                loss_discriminator = (loss_d_real + loss_d_fake) * self.gan_loss_weight
                total_loss = total_loss + loss_discriminator
                log_dict.update(
                    {
                        "train/gan_d_loss": loss_discriminator,
                        "train/score_real": logits_real.mean(),
                        "train/score_fake": logits_fake.mean(),
                    }
                )

        log_dict["train/update_mode"] = 1.0
        self.log_dict(log_dict, prog_bar=True, on_step=True)
        return total_loss


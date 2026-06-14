import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import math
from contextlib import contextmanager
from audiox_turbo.training.utils import create_optimizer_from_config, create_scheduler_from_config


@contextmanager
def toggle_grad(model, requires_grad):
    old_states = {}
    for name, param in model.named_parameters():
        old_states[name] = param.requires_grad
        param.requires_grad = requires_grad
    try:
        yield
    finally:
        for name, param in model.named_parameters():
            param.requires_grad = old_states[name]

def get_x0_from_noise(z_t, pred_v, t, objective):
    if objective == "rectified_flow":
        t_expand = t.view(-1, 1, 1)
        return z_t - t_expand * pred_v
    elif objective == "v":
        alpha = torch.cos(t * math.pi / 2).view(-1, 1, 1)
        sigma = torch.sin(t * math.pi / 2).view(-1, 1, 1)
        return alpha * z_t - sigma * pred_v
    else:
        raise NotImplementedError(f"Objective {objective} not supported.")

def get_zt_from_x0(x0, t, objective):
    noise = torch.randn_like(x0)
    t_expand = t.view(-1, 1, 1)
    if objective == "rectified_flow":
        return t_expand * noise + (1 - t_expand) * x0, noise
    elif objective == "v":
        alpha = torch.cos(t * math.pi / 2).view(-1, 1, 1)
        sigma = torch.sin(t * math.pi / 2).view(-1, 1, 1)
        return alpha * x0 + sigma * noise, noise
    else:
        raise NotImplementedError(f"Objective {objective} not supported.")

def get_target_v_from_x0(x0, z_t, t, objective, noise=None):
    if objective == "rectified_flow":
        if noise is None:
             t_expand = t.view(-1, 1, 1)
             return (z_t - x0) / (t_expand + 1e-8)
        return noise - x0
    elif objective == "v":
        alpha = torch.cos(t * math.pi / 2).view(-1, 1, 1)
        sigma = torch.sin(t * math.pi / 2).view(-1, 1, 1)
        if noise is None:
             raise ValueError("Noise is required for analytical v-target in v-pred")
        return alpha * noise - sigma * x0
    else:
        raise NotImplementedError(f"Objective {objective} not supported.")


class DiscriminatorHead(nn.Module):
    def __init__(self, input_channels, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(hidden_dim * 4, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)


class DiffusionDMDTrainingWrapper(pl.LightningModule):
    def __init__(
        self,
        model,
        teacher_model,
        fake_model,
        lr: float = None,
        optimizer_configs: dict = None,
        pre_encoded: bool = False,
        cfg_dropout_prob: float = 0.0,
        student_sample_steps: int = 4,
        dmd_loss_weight: float = 1.0,
        fake_loss_weight: float = 1.0,
        use_gan: bool = True,
        gan_loss_weight: float = 1.0,
        feature_channels: int = 64,
        teacher_guidance_scale: float = 6.0,
        mask_padding: bool = False,
        mask_padding_dropout: float = 0.0,
        use_ema: bool = True,
        update_ratio_s_f: int = 5
    ):
        super().__init__()

        self.student = model
        self.teacher = teacher_model
        self.fake = fake_model

        self.use_gan = use_gan
        self.gan_loss_weight = gan_loss_weight
        self.feature_channels = feature_channels

        if self.use_gan:
            self.discriminator_head = DiscriminatorHead(input_channels=self.feature_channels)
            self.discriminator_head.train()
        else:
            self.discriminator_head = None

        self.teacher.eval()
        self.teacher.requires_grad_(False)

        self.student.train()
        self.fake.train()

        self.lr = lr
        self.optimizer_configs = optimizer_configs
        self.pre_encoded = pre_encoded
        self.cfg_dropout_prob = cfg_dropout_prob

        self.student_sample_steps = student_sample_steps
        self.dmd_loss_weight = dmd_loss_weight
        self.fake_loss_weight = fake_loss_weight
        self.teacher_guidance_scale = teacher_guidance_scale

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout
        self.diffusion_objective = self.student.diffusion_objective
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.update_ratio_s_f = update_ratio_s_f

        if self.optimizer_configs is None:
            self.optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {"lr": lr}
                    }
                }
            }

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']

        params = list(self.student.parameters()) + list(self.fake.parameters())

        if self.use_gan and self.discriminator_head is not None:
            params += list(self.discriminator_head.parameters())

        opt = create_optimizer_from_config(diffusion_opt_config['optimizer'], params)
        schedulers = []
        if "scheduler" in diffusion_opt_config:
            sched = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt)
            schedulers.append({"scheduler": sched, "interval": "step"})
        return [opt], schedulers

    def get_timesteps_schedule(self, steps, device):
        return torch.linspace(1.0, 0.0, steps + 1, device=device)

    def iterative_dmd_forward(self, initial_noise, target_step_idx, cond, extra_args):
        batch_size = initial_noise.shape[0]
        device = initial_noise.device
        timesteps = self.get_timesteps_schedule(self.student_sample_steps, device)

        current_z = initial_noise
        final_x0 = None

        for i in range(target_step_idx + 1):
            is_last_step = (i == target_step_idx)
            t_curr_val = timesteps[i]
            t_curr = t_curr_val.expand(batch_size)

            with torch.set_grad_enabled(is_last_step):
                pred_v = self.student(
                    current_z,
                    t_curr,
                    cond=cond,
                    cfg_dropout_prob=0.0,
                    cfg_scale=1.0,
                    **extra_args
                )
                pred_x0 = get_x0_from_noise(current_z, pred_v, t_curr, self.diffusion_objective)

            if is_last_step:
                final_x0 = pred_x0
            else:
                pred_x0 = pred_x0.detach()
                t_next_val = timesteps[i+1]
                t_next = t_next_val.expand(batch_size)
                current_z, _ = get_zt_from_x0(pred_x0, t_next, self.diffusion_objective)

        return final_x0

    def compute_discriminator_score(self, x0, cond, extra_args, use_grad_on_x0=False):
        if not self.use_gan:
            return None

        batch_size = x0.shape[0]

        t_disc = torch.rand(batch_size, device=self.device) * 0.2

        if use_grad_on_x0:
            x_in = x0
        else:
            x_in = x0.detach()

        z_disc, _ = get_zt_from_x0(x_in, t_disc, self.diffusion_objective)

        if use_grad_on_x0:
            teacher_output = self.teacher(
                z_disc,
                t_disc,
                cond=cond,
                cfg_dropout_prob=0.0,
                cfg_scale=1.0,
                **extra_args
            )
        else:
            with torch.no_grad():
                teacher_output = self.teacher(
                    z_disc,
                    t_disc,
                    cond=cond,
                    cfg_dropout_prob=0.0,
                    cfg_scale=1.0,
                    **extra_args
                )

        score = self.discriminator_head(teacher_output)
        return score

    def training_step(self, batch, batch_idx):
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
                            mode="nearest"
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
            x0_student = self.iterative_dmd_forward(
                initial_noise,
                target_step_idx,
                student_cond,
                extra_args
            )

        if is_student_update:
            t_dmd = self.rng.draw(batch_size)[:, 0].to(self.device)
            z_t_dmd, _ = get_zt_from_x0(x0_student, t_dmd, self.diffusion_objective)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    v_teacher = self.teacher(
                        z_t_dmd, t_dmd, cond=teacher_cond, cfg_dropout_prob=0.0,
                        cfg_scale=self.teacher_guidance_scale, rescale_cfg=True, **extra_args
                    )
                    x0_teacher = get_x0_from_noise(z_t_dmd, v_teacher, t_dmd, self.diffusion_objective)

                    v_fake = self.fake(
                        z_t_dmd, t_dmd, cond=student_cond, cfg_dropout_prob=0.0,
                        cfg_scale=1.0, **extra_args
                    )
                    x0_fake = get_x0_from_noise(z_t_dmd, v_fake, t_dmd, self.diffusion_objective)

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
                with toggle_grad(self.discriminator_head, False):
                    with torch.cuda.amp.autocast():
                        logits_fake = self.compute_discriminator_score(
                            x0_student, teacher_cond, extra_args, use_grad_on_x0=True
                        )

                        loss_gan_gen = -torch.mean(logits_fake) * self.gan_loss_weight

                        total_loss = total_loss + loss_gan_gen
                        log_dict["train/gan_g_loss"] = loss_gan_gen

            log_dict.update({
                "train/total_loss": total_loss,
                "train/update_mode": 0.0
            })

            self.log_dict(log_dict, prog_bar=True, on_step=True)
            return total_loss

        else:
            target_x0 = x0_student.detach()

            t_fake_train = self.rng.draw(batch_size)[:, 0].to(self.device)
            z_t_fake, noise_epsilon = get_zt_from_x0(target_x0, t_fake_train, self.diffusion_objective)
            target_v = get_target_v_from_x0(target_x0, z_t_fake, t_fake_train, self.diffusion_objective, noise=noise_epsilon)

            with torch.cuda.amp.autocast():
                pred_v_fake = self.fake(
                    z_t_fake, t_fake_train, cond=student_cond,
                    cfg_dropout_prob=0.0, cfg_scale=1.0, **extra_args
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
                    logits_real = self.compute_discriminator_score(
                        diffusion_input, teacher_cond, extra_args, use_grad_on_x0=False
                    )

                    logits_fake = self.compute_discriminator_score(
                        target_x0, teacher_cond, extra_args, use_grad_on_x0=False
                    )

                    loss_d_real = torch.mean(F.relu(1.0 - logits_real))
                    loss_d_fake = torch.mean(F.relu(1.0 + logits_fake))

                    loss_discriminator = (loss_d_real + loss_d_fake) * self.gan_loss_weight

                    total_loss = total_loss + loss_discriminator

                    log_dict.update({
                        "train/gan_d_loss": loss_discriminator,
                        "train/score_real": logits_real.mean(),
                        "train/score_fake": logits_fake.mean()
                    })

            log_dict["train/update_mode"] = 1.0

            self.log_dict(log_dict, prog_bar=True, on_step=True)
            return total_loss

    def on_before_zero_grad(self, *args, **kwargs):
        if hasattr(self.student, "diffusion_ema") and self.student.diffusion_ema is not None:
            self.student.diffusion_ema.update()

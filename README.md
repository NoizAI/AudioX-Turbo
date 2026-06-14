# 🎧 AudioX-Turbo: A Unified Framework for Efficient Anything-to-Audio Generation

[![arXiv](https://img.shields.io/badge/arXiv-2606.12555-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2606.12555)
[![Project Page](https://img.shields.io/badge/GitHub.io-Project-blue?logo=Github&style=flat-square)](https://zeyuet.github.io/AudioX-Turbo/)
[![🤗 Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/HKUSTAudio/AudioX-Turbo)

## ✨ Abstract

Audio and music generation based on flexible multimodal control signals is a widely applicable topic, with the following key challenges: 1) a unified multimodal modeling framework, 2) large-scale, high-quality training data, and 3) the prohibitive inference cost of multi-step diffusion sampling.

As such, we propose **AudioX-Turbo**, a unified and efficient framework for anything-to-audio generation that integrates varied multimodal conditions (i.e., text, video, and audio signals). AudioX-Turbo follows a *teacher–student* paradigm. The teacher **AudioX-Base** is built on a Multimodal Diffusion Transformer with a Multimodal Adaptive Fusion module that aligns diverse multimodal inputs for high-fidelity synthesis, and is then distilled into the few-step student **AudioX-Turbo** via Distribution Matching Distillation adapted to flow matching, complemented by a diffusion-based discriminator for high-quality few-step generation.

To support training, we construct a large-scale, high-quality dataset, **IF-caps-Pro**, comprising approximately **9.2M** samples curated through a two-stage data collection and annotation pipeline. We benchmark AudioX-Turbo across a wide range of tasks, finding that our model achieves superior performance, especially on text-to-audio and text-to-music generation, while operating at only **4 sampling steps** and requiring up to **~25×** fewer function evaluations (NFE) than multi-step baselines. These results demonstrate that our method is capable of audio generation under flexible multimodal control, showing efficient and powerful instruction-following capabilities.

## ✨ Teaser

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a4ddcbb-fa6b-4dd1-acc9-f5f0484e84e9" width="100%" alt="teaser"/>
</p>
<p align="center">
  Performance comparison against baselines: (a) Inception Score across benchmarks, (b) instruction-following results, (c) quality–efficiency trade-off.
</p>

## ✨ Method

<p align="center">
  <img src="https://github.com/user-attachments/assets/bd129a15-a116-4ec9-b1f9-01e3854ce1a9" width="100%" alt="method"/>
</p>
<p align="center">
  Overview of the AudioX-Turbo framework.
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/306d75a4-ac12-40f0-b78c-26658ee2c201" width="100%" alt="distillation_framework"/>
</p>
<p align="center">
  Few-step distillation with DMD, fake-model diffusion loss, and adversarial supervision.
</p>

---

## 🛠️ Environment Setup

### Prerequisites

- Python 3.8 (the pinned dependencies are verified against Python 3.8.20)
- CUDA 12.1 capable GPU (an A100/H800-class card is recommended for training)
- FFmpeg and libsndfile (for audio/video I/O)
- A full CUDA toolkit (with `CUDA_HOME/bin/nvcc`) is required **only** for the DeepSpeed training path

### Installation

```bash
# Clone the repository
git clone https://github.com/NoizAI/AudioX-Turbo.git
cd AudioX-Turbo

# Create a conda environment
conda create -n audiox-turbo python=3.8.20
conda activate audiox-turbo

# Install media libraries
conda install -c conda-forge ffmpeg libsndfile

# Install dependencies
pip install -r requirements.txt
pip install -e . --no-deps

pip install soundfile==0.12.1
```

## 🪄 Checkpoints

Large model files are **not** tracked in git. The trained checkpoints are hosted on the Hugging Face Hub at [HKUSTAudio/AudioX-Turbo](https://huggingface.co/HKUSTAudio/AudioX-Turbo).

For **inference** you need the few-step `AudioX-Turbo` student model, the VAE, and
(for video-conditioned generation) the Synchformer checkpoint. For **training**
(distillation) you additionally need the teacher / base model, which is also used
to initialize the student.

Download with `huggingface-cli`:

```bash
pip install -U "huggingface_hub[cli]"

# Inference checkpoints (student + VAE + Synchformer)
huggingface-cli download HKUSTAudio/AudioX-Turbo \
  audiox_turbo/audiox_turbo.ckpt pretransform/vae.ckpt synchformer/synchformer_state_dict.pth \
  --local-dir checkpoints

# Training only: teacher / base model
huggingface-cli download HKUSTAudio/AudioX-Turbo \
  pretrained_ckpt/pretrained_ckpt.ckpt \
  --local-dir checkpoints
```

…or with `wget`:

```bash
# AudioX-Turbo: distilled 4-step student model (inference)
wget https://huggingface.co/HKUSTAudio/AudioX-Turbo/resolve/main/audiox_turbo/audiox_turbo.ckpt -O checkpoints/audiox_turbo/audiox_turbo.ckpt

# VAE pretransform
wget https://huggingface.co/HKUSTAudio/AudioX-Turbo/resolve/main/pretransform/vae.ckpt -O checkpoints/pretransform/vae.ckpt

# Synchformer, for video-conditioned (V2A/V2M) generation
wget https://huggingface.co/HKUSTAudio/AudioX-Turbo/resolve/main/synchformer/synchformer_state_dict.pth -O checkpoints/synchformer/synchformer_state_dict.pth

# Training only: teacher / base model (student init + teacher)
wget https://huggingface.co/HKUSTAudio/AudioX-Turbo/resolve/main/pretrained_ckpt/pretrained_ckpt.ckpt -O checkpoints/pretrained_ckpt/pretrained_ckpt.ckpt
```

Either way produces:

```text
checkpoints/audiox_turbo/audiox_turbo.ckpt              # AudioX-Turbo: distilled 4-step student model (inference)
checkpoints/pretransform/vae.ckpt                       # VAE pretransform
checkpoints/synchformer/synchformer_state_dict.pth      # Synchformer, for video-conditioned (V2A/V2M) generation
checkpoints/pretrained_ckpt/pretrained_ckpt.ckpt        # teacher / base model (training only)
```

The text/vision encoders are fetched automatically: on first run the scripts download `openai/clip-vit-base-patch32` and `t5-base` from the Hugging Face Hub into `checkpoints/huggingface`. If that cache already exists locally, the scripts switch to offline mode automatically. To force a mode, set `HF_HUB_OFFLINE` and/or `AUDIOX_TURBO_CLIP_MODEL_PATH` yourself.

## 📁 Project Layout

```text
configs/audiox_turbo_distill_4step.json      # model and DMD/GAN training config
configs/audiox_turbo_infer_4step.json        # 4-step inference config
configs/audiox_turbo_dataset.json            # local manifest config
data/custom_metadata.py                      # metadata adapter used by the dataloader
data/train_manifest_10.jsonl                 # 5 records sampled from the real training manifest
data/media/                                  # packaged audio, video, and sync-feature files for the 5 records
example/                                      # sample videos for the V2A / V2M inference demo
checkpoints/audiox_turbo/audiox_turbo.ckpt   # AudioX-Turbo: distilled 4-step student model (inference)
checkpoints/pretrained_ckpt/pretrained_ckpt.ckpt   # teacher / base model (training only: student init + teacher)
checkpoints/pretransform/vae.ckpt                  # VAE pretransform
checkpoints/synchformer/synchformer_state_dict.pth # Synchformer, for video-conditioned generation
checkpoints/huggingface/hub/                  # CLIP/T5 cache (auto-downloaded on first run)
train_audiox_turbo.py                        # Lightning training entrypoint
scripts/train_audiox_turbo.sh                # main two-GPU torchrun script
run_gradio.py                                # 4-step AudioX-Turbo Gradio demo
audiox_turbo/                                # model, data, inference, and training library
```

The packaged manifest (5 records) uses only project-local paths, so training runs out of the box.

## 🎯 Supported Tasks

Like AudioX, AudioX-Turbo is a unified model that accepts text, video, and audio conditions in any combination:

| Task                 | `video_path`       | `text_prompt`                                 | `audio_path` |
|:---------------------|:-------------------|:----------------------------------------------|:-------------|
| Text-to-Audio (T2A)  | `None`             | `"Typing on a keyboard"`                      | `None`       |
| Text-to-Music (T2M)  | `None`             | `"A music with piano and violin"`             | `None`       |
| Video-to-Audio (V2A) | `"video_path.mp4"` | `"Generate general audio for the video"`      | `None`       |
| Video-to-Music (V2M) | `"video_path.mp4"` | `"Generate music for the video"`              | `None`       |
| TV-to-Audio (TV2A)   | `"video_path.mp4"` | `"Ocean waves crashing with people laughing"` | `None`       |
| TV-to-Music (TV2M)   | `"video_path.mp4"` | `"Generate music with piano instrument"`      | `None`       |

The `example/` videos demonstrate the V2A / V2M paths; set `video_path=None` for pure text-to-audio/music. See the [Inference](#-inference) section for runnable code.

## 🏋️ Training

```bash
bash scripts/train_audiox_turbo.sh
```

The launcher uses `torchrun` directly (no Slurm logic). By default it exposes GPU `0,1` and starts two workers, reading `data/train_manifest_10.jsonl`.

Useful overrides:

```bash
# Validate the data pipeline without optimization
AUDIOX_TURBO_MAX_STEPS=0 bash scripts/train_audiox_turbo.sh

# Single optimization step with batch size 1
AUDIOX_TURBO_MAX_STEPS=1 BATCH_SIZE=1 bash scripts/train_audiox_turbo.sh

# Single-GPU smoke test without DeepSpeed (no nvcc required)
AUDIOX_TURBO_MAX_STEPS=1 BATCH_SIZE=1 STRATEGY=auto NUM_GPUS=1 \
  CUDA_VISIBLE_DEVICES=0 bash scripts/train_audiox_turbo.sh
```

Main environment overrides:

```text
MODEL_CONFIG=configs/audiox_turbo_distill_4step.json
DATASET_CONFIG=configs/audiox_turbo_dataset.json
PRETRAINED_CKPT=checkpoints/pretrained_ckpt/pretrained_ckpt.ckpt
PRETRANSFORM_CKPT=checkpoints/pretransform/vae.ckpt
STRATEGY=deepspeed
CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
AUDIOX_TURBO_TIMESTEP_PROBS=0.25,0.25,0.25,0.25
AUDIOX_TURBO_GAN_DISC_HEAD_MODE=all_blocks
AUDIOX_TURBO_GAN_BACKBONE_NUM_BLOCKS=6
AUDIOX_TURBO_GAN_BACKBONE_TRAINABLE=false
```

`STRATEGY=deepspeed` matches the original large-scale training setup. The script points `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` to `checkpoints/huggingface` and enables offline mode by default.

## 🔊 Inference

AudioX-Turbo generates audio in **4 steps** with the distilled student. The same
conditioning interface covers every task — set `video_path` / `audio_path` to `None`
to drop a modality.

### Gradio demo

```bash
python run_gradio.py            # http://localhost:7860
python run_gradio.py --share    # public link
```

### Python API

```python
import torch
import torchaudio
from einops import rearrange

from audiox_turbo.inference import load_audiox_turbo_model
from audiox_turbo.inference.generation import generate_diffusion_cond_dmd
from audiox_turbo.data.utils import (
    read_video, load_and_process_audio, encode_video_with_synchformer, merge_video_audio,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the distilled 4-step student
model, model_config = load_audiox_turbo_model(
    "configs/audiox_turbo_infer_4step.json",
    "checkpoints/audiox_turbo/audiox_turbo.ckpt",
    pretransform_ckpt_path="checkpoints/pretransform/vae.ckpt",
    device=device,
)
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
target_fps = model_config.get("video_fps", 5)
seconds_total = 10

# --- Choose a task by setting the inputs below ---
# Text-to-Audio:  video_path=None,  text_prompt="Typing on a keyboard"
# Video-to-Music: video_path="example/V2M_sample-1.mp4", text_prompt="Generate music for the video"
video_path = "example/V2M_sample-1.mp4"
text_prompt = "Generate music for the video"
audio_path = None

if video_path:
    video_tensor = read_video(video_path, seek_time=0, duration=seconds_total, target_fps=target_fps)
    sync_features = encode_video_with_synchformer(video_path, 0, seconds_total, device=device)
else:
    video_tensor = torch.zeros(seconds_total * target_fps, 3, 224, 224)
    sync_features = torch.zeros(1, 240, 768, device=device)

audio_tensor = load_and_process_audio(audio_path, sample_rate, 0, seconds_total)

conditioning = [{
    "video_prompt": {"video_tensors": video_tensor.unsqueeze(0), "video_sync_frames": sync_features},
    "text_prompt": text_prompt or "",
    "audio_prompt": audio_tensor.unsqueeze(0),
    "seconds_start": 0,
    "seconds_total": seconds_total,
}]

# 4-step generation (no classifier-free guidance)
output = generate_diffusion_cond_dmd(
    model, steps=4, conditioning=conditioning,
    sample_size=sample_size, seed=0, device=device,
)

output = output[:, :, : sample_rate * seconds_total]
output = rearrange(output, "b d n -> d (b n)")
output = output.to(torch.float32).div(torch.max(torch.abs(output)).clamp_min(1e-8)).clamp(-1, 1)
torchaudio.save("output.wav", output.cpu(), sample_rate)

# Optional: mux the audio back onto the source video
if video_path:
    merge_video_audio(video_path, "output.wav", "output.mp4", 0, seconds_total)
```

---

## 🚀 Citation

If you find our work useful, please consider citing:

```bibtex
@article{tian2026audioxturbo,
  title={AudioX-Turbo: A Unified Framework for Efficient Anything-to-Audio Generation},
  author={Tian, Zeyue and Ke, Lei and Liu, Zhaoyang and Yuan, Ruibin and Xue, Liumeng and Yang, Yujiu and Chen, Weijia and Tan, Xu and Chen, Qifeng and Xue, Wei and Guo, Yike},
  journal={arXiv preprint arXiv:2606.12555},
  year={2026}
}
@inproceedings{tian2026audiox,
  title={AudioX: a unified framework for anything-to-audio generation},
  author={Tian, Zeyue and Jin, Y and Liu, Z and others},
  booktitle={Proceedings of the Fourteenth International Conference on Learning Representations},
  year={2026}
}
@inproceedings{tian2025vidmuse,
  title={Vidmuse: A simple video-to-music generation framework with long-short-term modeling},
  author={Tian, Zeyue and Liu, Zhaoyang and Yuan, Ruibin and Pan, Jiahao and Liu, Qifeng and Tan, Xu and Chen, Qifeng and Xue, Wei and Guo, Yike},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18782--18793},
  year={2025}
}
```

---

## 📭 Contact

- **Zeyue Tian**: ztianad@connect.ust.hk
- **Lei Ke**: kelei2002@gmail.com

---

## 📄 License

Please follow [CC-BY-NC 4.0](./LICENSE).

**Note:** The models are watermarked and are strictly for non-commercial use only.

---

## 🙏 Acknowledgments

We thank [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools), [AudioX](https://github.com/NoizAI/AudioX), [VidMuse](https://github.com/ZeyueT/VidMuse), and [MMAudio](https://github.com/hkchengrex/MMAudio) for their valuable contributions.

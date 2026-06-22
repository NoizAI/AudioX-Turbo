"""Minimal, directly-deployable Gradio demo for the 4-step AudioX-Turbo student.

Launch with:

    python run_gradio.py                       # text / video / audio -> audio
    python run_gradio.py --share               # public Gradio link
    python run_gradio.py --ckpt path/to.ckpt   # custom checkpoint

It loads the distilled student once and runs the few-step DMD sampler
(``generate_diffusion_cond_dmd``). The same conditioning interface covers
Text-to-Audio, Video-to-Audio/Music, and combined Text+Video conditioning.
"""

import argparse
import os
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = REPO_ROOT / "outputs"

# Point the Hugging Face caches at the packaged checkpoints/huggingface folder and
# use it offline when the CLIP/T5 snapshots are already present.
HF_HOME = REPO_ROOT / "checkpoints" / "huggingface"
os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_HOME / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "hub"))
_clip_snapshot = HF_HOME / "hub" / "models--openai--clip-vit-base-patch32" / "snapshots" / "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
if _clip_snapshot.is_dir():
    os.environ.setdefault("AUDIOX_TURBO_CLIP_MODEL_PATH", str(_clip_snapshot))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torchaudio
import gradio as gr
from einops import rearrange

from audiox_turbo.inference import load_audiox_turbo_model
from audiox_turbo.inference.generation import generate_diffusion_cond_dmd
from audiox_turbo.data.utils import (
    read_video,
    load_and_process_audio,
    encode_video_with_synchformer,
    merge_video_audio,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL = None
MODEL_CONFIG = None


def _make_output_paths(video_path=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex[:8]
    out_wav = OUTPUT_DIR / f"audiox_turbo_{run_id}.wav"
    if video_path:
        video_stem = Path(video_path).stem or "video"
        out_video = OUTPUT_DIR / f"{video_stem}_{run_id}.mp4"
    else:
        out_video = None
    return out_wav, out_video


def load(ckpt_path, config_path, pretransform_ckpt_path):
    global MODEL, MODEL_CONFIG
    MODEL, MODEL_CONFIG = load_audiox_turbo_model(
        config_path, ckpt_path, pretransform_ckpt_path, device=DEVICE,
    )
    return MODEL, MODEL_CONFIG


@torch.no_grad()
def generate(prompt, video_path, audio_path, seconds_total, steps, seed):
    sample_rate = MODEL_CONFIG["sample_rate"]
    sample_size = MODEL_CONFIG["sample_size"]
    target_fps = MODEL_CONFIG.get("video_fps", 5)
    seconds_total = int(seconds_total)

    video_path = video_path or None
    audio_path = audio_path or None

    if video_path:
        video_tensor = read_video(video_path, seek_time=0, duration=seconds_total, target_fps=target_fps)
        sync_features = encode_video_with_synchformer(video_path, 0, seconds_total, device=DEVICE)
    else:
        video_tensor = torch.zeros(seconds_total * target_fps, 3, 224, 224)
        sync_features = torch.zeros(1, 240, 768, device=DEVICE)

    audio_tensor = load_and_process_audio(audio_path, sample_rate, 0, seconds_total)

    conditioning = [{
        "video_prompt": {"video_tensors": video_tensor.unsqueeze(0), "video_sync_frames": sync_features},
        "text_prompt": prompt or "",
        "audio_prompt": audio_tensor.unsqueeze(0),
        "seconds_start": 0,
        "seconds_total": seconds_total,
    }]

    audio = generate_diffusion_cond_dmd(
        MODEL,
        steps=int(steps),
        conditioning=conditioning,
        sample_size=sample_size,
        seed=int(seed),
        device=DEVICE,
    )

    audio = audio[:, :, : sample_rate * seconds_total]
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio)).clamp_min(1e-8)).clamp(-1, 1)
    out_wav, out_video = _make_output_paths(video_path)
    torchaudio.save(str(out_wav), audio.cpu(), sample_rate)

    if video_path and os.path.exists(video_path):
        merge_video_audio(video_path, str(out_wav), str(out_video), 0, seconds_total)

    return str(out_wav), str(out_video) if out_video else None


def build_ui():
    with gr.Blocks(title="AudioX-Turbo") as demo:
        gr.Markdown(
            "# AudioX-Turbo\n"
            "4-step anything-to-audio generation. Provide a text prompt and/or a "
            "video (V2A / V2M) and/or an audio prompt."
        )
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Text prompt", placeholder="e.g. Generate music for the video")
                video_path = gr.Textbox(label="Video path (optional)", placeholder="example/V2M_sample-1.mp4")
                audio_path = gr.Textbox(label="Audio prompt path (optional)")
                seconds_total = gr.Slider(1, 60, value=10, step=1, label="Seconds")
                steps = gr.Slider(1, 20, value=4, step=1, label="DMD steps")
                seed = gr.Number(value=0, label="Seed (-1 for random)")
                run = gr.Button("Generate", variant="primary")
            with gr.Column():
                audio_out = gr.Audio(label="Generated audio", interactive=False)
                video_out = gr.Video(label="Merged video (if a video was given)", interactive=False)
        run.click(generate, [prompt, video_path, audio_path, seconds_total, steps, seed], [audio_out, video_out])
    return demo


def main():
    parser = argparse.ArgumentParser(description="AudioX-Turbo 4-step Gradio demo")
    parser.add_argument("--ckpt", default="checkpoints/audiox_turbo/audiox_turbo.ckpt")
    parser.add_argument("--config", default="configs/audiox_turbo_infer_4step.json")
    parser.add_argument("--pretransform-ckpt", default="checkpoints/pretransform/vae.ckpt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    load(args.ckpt, args.config, args.pretransform_ckpt)
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

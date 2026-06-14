import gc
import platform
import os
import subprocess as sp  # For merging audio and video

import numpy as np
import gradio as gr
import json 
import torch
import torchaudio
import torchvision
import decord
from decord import VideoReader
from decord import cpu
import math
import einops
import torchvision.transforms as transforms

from aeiou.viz import audio_spectrogram_image
from einops import rearrange
from safetensors.torch import load_file
from torch.nn import functional as F
from torchaudio import transforms as T

from ..inference.generation import generate_diffusion_cond, generate_diffusion_uncond
from ..models.factory import create_model_from_config
from ..models.pretrained import get_pretrained_model
from ..models.utils import load_ckpt_state_dict
from ..inference.utils import prepare_audio
from ..training.utils import copy_state_dict

from PIL import Image

# 全局变量，用于存储模型配置
model_configurations = {}
device = torch.device("cpu")  # 默认设备

# 设置临时目录（可通过 AUDIOX_TURBO_TMPDIR 覆盖；否则沿用现有 TMPDIR）
_tmpdir = os.environ.get("AUDIOX_TURBO_TMPDIR")
if _tmpdir:
    os.environ["TMPDIR"] = _tmpdir

current_model_name = None
current_model = None
current_sample_rate = None
current_sample_size = None


_SYNC_SIZE = 224
from torchvision.transforms import v2        
sync_transform = v2.Compose([
    v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
    v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


from audiox_turbo.models.synchformer.features_utils import FeaturesUtils

# Synchformer checkpoint for video-conditioned generation.
# Override with AUDIOX_TURBO_SYNCHFORMER_CKPT; defaults to a project-local path.
SYNCHFORMER_CKPT = os.environ.get(
    "AUDIOX_TURBO_SYNCHFORMER_CKPT",
    "checkpoints/synchformer/synchformer_state_dict.pth",
)

# Lazily created so that importing this module (and text/audio-only generation)
# does not require a GPU or the Synchformer checkpoint.
_sync_feature_extractor = None


def get_sync_feature_extractor():
    global _sync_feature_extractor
    if _sync_feature_extractor is None:
        if not os.path.exists(SYNCHFORMER_CKPT):
            raise FileNotFoundError(
                f"Synchformer checkpoint not found: {SYNCHFORMER_CKPT}. "
                "Set AUDIOX_TURBO_SYNCHFORMER_CKPT to enable video-conditioned generation."
            )
        _sync_feature_extractor = FeaturesUtils(
            synchformer_ckpt=SYNCHFORMER_CKPT,
            enable_conditions=True,
        ).eval().cuda()
    return _sync_feature_extractor

def adjust_video_duration(video_tensor, duration, target_fps):
    current_duration = video_tensor.shape[0]
    target_duration = duration * target_fps
    if current_duration > target_duration:
        video_tensor = video_tensor[:target_duration]
    elif current_duration < target_duration:
        last_frame = video_tensor[-1:]
        repeat_times = target_duration - current_duration
        video_tensor = torch.cat((video_tensor, last_frame.repeat(repeat_times, 1, 1, 1)), dim=0)
    return video_tensor



def video_read_local(filepath, seek_time=0., duration=-1, target_fps=2):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        # 处理图像文件
        resize_transform = transforms.Resize((224, 224))
        image = Image.open(filepath).convert("RGB")  # 打开图像并转换为RGB格式
        frame = transforms.ToTensor()(image).unsqueeze(0)  # 转换为张量并增加一个维度 [1, C, H, W]
        # Resize the image to 224x224
        frame = resize_transform(frame)
        # 假设持续时间为10秒，并复制帧以匹配目标帧率
        target_frames = int(duration * target_fps)
        frame = frame.repeat(int(math.ceil(target_frames / frame.shape[0])), 1, 1, 1)[:target_frames]
        assert frame.shape[0] == target_frames, f"The shape of frame is {frame.shape}"
        return frame  # [N, C, H, W]

    vr = VideoReader(filepath, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    seek_frame = int(seek_time * fps)
    if duration > 0:
        total_frames_to_read = int(target_fps * duration)
        frame_interval = int(math.ceil(fps / target_fps))
        end_frame = min(seek_frame + total_frames_to_read * frame_interval, total_frames)
        frame_ids = list(range(seek_frame, end_frame, frame_interval))
    else:
        frame_interval = int(math.ceil(fps / target_fps))
        frame_ids = list(range(0, total_frames, frame_interval))

    # 批量读取指定的帧
    frames = vr.get_batch(frame_ids).asnumpy()
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

    if frames.shape[2] != 224 or frames.shape[3] != 224:
        # 仅在必要时调整大小
        print(f'resizing...--->224x224')
        resize_transform = transforms.Resize((224, 224))
        frames = resize_transform(frames)

    # 调整视频持续时间
    video_tensor = adjust_video_duration(frames, duration, target_fps)

    assert video_tensor.shape[0] == duration * target_fps, f"The shape of video_tensor is {video_tensor.shape}"

    return video_tensor


def merge_video_audio(video_path, audio_path, output_path, start_time, duration):
    command = [
        'ffmpeg',
        '-y',                   # Overwrite output files without asking
        '-ss', str(start_time), # Start time
        '-t', str(duration),    # Duration
        '-i', video_path,       # Input video file
        '-i', audio_path,       # Input audio file
        '-c:v', 'copy',         # Copy the video codec (no re-encoding)
        '-c:a', 'aac',          # Use AAC audio codec
        '-map', '0:v:0',        # Map the video from the first input
        '-map', '1:a:0',        # Map the audio from the second input
        '-shortest',            # Stop encoding when the shortest input ends
        '-strict', 'experimental',  # Allow experimental codecs if needed
        output_path             # Output file path
    ]
    
    try:
        sp.run(command, check=True)
        print(f"Successfully merged audio and video into {output_path}")
        return output_path
    except sp.CalledProcessError as e:
        print(f"Error merging audio and video: {e}")
        return None

def load_model(model_name, model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model_configurations
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        # Load checkpoint
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")

    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
        
    print(f"Done loading model {model_name}")

    return model, model_config, sample_rate, sample_size


def load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total):
    audio_tensor, sr = torchaudio.load(audio_path)
    # 确保音频长度为 `seconds_total` 秒
    start_index = int(sample_rate * seconds_start)  # 计算起始索引
    target_length = int(sample_rate * seconds_total)
    end_index = start_index + target_length  # 计算结束索引     
    audio_tensor = audio_tensor[:, start_index:end_index]  # 切片音频张量               
    if audio_tensor.shape[1] < target_length:
        pad_length = target_length - audio_tensor.shape[1]
        audio_tensor = F.pad(audio_tensor, (pad_length, 0))  # 在开头填充
    return audio_tensor

def generate_cond(
        prompt,
        model_name,
        negative_prompt=None,
        video_file=None,  # 改为文件上传
        video_path=None,
        sync_feature_path=None,
        audio_prompt_file=None,  # 改为文件上传     
        audio_prompt_path=None,
        seconds_start=0,
        seconds_total=10,
        cfg_scale=6.0,
        steps=250,
        preview_every=None,
        seed=-1,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
        cfg_rescale=0.0,
        use_init=False,
        init_audio=None,
        init_noise_level=1.0,
        mask_cropfrom=None,
        mask_pastefrom=None,
        mask_pasteto=None,
        mask_maskstart=None,
        mask_maskend=None,
        mask_softnessL=None,
        mask_softnessR=None,
        mask_marination=None,
        custom_model_config=None,  # 新增参数
        custom_ckpt_path=None,     # 新增参数
        batch_size=1
    ):

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    print(f"Prompt: {prompt}")
    print(f"Model: {model_name}")

    preview_images = []
    if preview_every == 0:
        preview_every = None
        
    print(f'seconds_total: {seconds_total}')
    try:
        length_set = int(seconds_total)
    except ValueError:
        print("Invalid input for seconds_total, using default value of 10.")
        length_set = 10  # 或者你可以选择其他合适的默认值
    print(f'length_set: {length_set}')
    # length_set = seconds_total


    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        # In case this version of Torch doesn't even have `torch.backends.mps`...
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if model_name == "Custom Model":
        if not (custom_model_config and custom_ckpt_path):
            raise ValueError("For Custom Model, 'Custom Model Config Path' and 'Custom Model Checkpoint Path' must be provided.")
        
        model_type = "diffusion_cond"  # 固定为 'diffusion_cond'
        model_config_path = custom_model_config
        ckpt_path = custom_ckpt_path

        # 验证路径是否存在
        if not os.path.exists(model_config_path):
            raise FileNotFoundError(f"Model config file not found: {model_config_path}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model checkpoint file not found: {ckpt_path}")

        # 按需加载模型
        with open(model_config_path) as f:
            model_config = json.load(f)


        global current_model_name, current_model, current_sample_rate, current_sample_size  # 引入全局变量

        if current_model is None or model_name != current_model_name:
            # 如果没有加载模型或模型名称不同，加载模型
            current_model, model_config, sample_rate, sample_size = load_model(
            model_name="Custom Model",
            model_config=model_config,
            model_ckpt_path=ckpt_path,
            pretrained_name=None,
            pretransform_ckpt_path=None,
            device=device,
            model_half=False  # 根据需要设置
        )
            current_model_name = model_name  # 更新当前模型名称
        else:
            # 使用已加载的模型
            model = current_model

    else:
        if model_name not in model_configurations:
            raise ValueError(f"Model {model_name} configuration is not available.")

        cfg = model_configurations[model_name]
        model_config_path = cfg.get("model_config")
        ckpt_path = cfg.get("ckpt_path")
        pretrained_name = cfg.get("pretrained_name")
        pretransform_ckpt_path = cfg.get("pretransform_ckpt_path")
        model_type = cfg.get("model_type", "diffusion_cond")

        # 按需加载模型
        if model_config_path:
            with open(model_config_path) as f:
                model_config = json.load(f)
        else:
            model_config = None
        target_fps=model_config.get("video_fps", 5)
        # 检查当前模型是否已加载
        if current_model is None or model_name != current_model_name:
            # 如果没有加载模型或模型名称不同，加载模型
            current_model, model_config, sample_rate, sample_size = load_model(
                model_name=model_name,
                model_config=model_config,
                model_ckpt_path=ckpt_path,
                pretrained_name=pretrained_name,
                pretransform_ckpt_path=pretransform_ckpt_path,
                device=device,
                model_half=False  # 根据需要设置
            )
            current_model_name = model_name  # 更新当前模型名称
            model = current_model
            current_sample_rate = sample_rate
            current_sample_size = sample_size
        else:
            # 使用已加载的模型
            model = current_model
            sample_rate = current_sample_rate
            sample_size = current_sample_size

    if video_file is not None:
        video_path = video_file.name  # 获取上传文件的临时路径
    elif video_path:  # 如果没有上传文件，则使用输入的路径
        video_path = video_path.strip()  # 去除路径两端的空格
    else:
        video_path = None

    if audio_prompt_file is not None:
        print(f'audio_prompt_file: {audio_prompt_file}')
        audio_path = audio_prompt_file.name  # 获取上传文件的临时路径
    elif audio_prompt_path:  # 如果没有上传文件，则使用输入的路径
        audio_path = audio_prompt_path.strip()  # 去除路径两端的空格
    else:
        audio_path = None

    # target_fps=10
    if video_path is None and audio_path is None:
        mask_type = "mask_video_audio"
        Video_tensors = torch.zeros(int(target_fps * seconds_total), 3, 224, 224)
        audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))
        sync_features = torch.zeros(1, 240, 768).to(device)

    elif video_path is None:
        mask_type = "mask_video"
        Video_tensors = torch.zeros(int(target_fps * seconds_total), 3, 224, 224)
        sync_features = torch.zeros(1, 240, 768).to(device)
        try:
            audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)
        except Exception as e:
            print("Audio prompt file is empty or invalid, using zero audio tensor.")
            audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))  # 假设立体声音频            
    elif audio_path is None:
        mask_type = "mask_audio"
        try:
            Video_tensors = video_read_local(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=target_fps)

            sync_video_tensor = video_read_local(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=25)
            sync_video=sync_transform(sync_video_tensor)
            sync_video = sync_video.unsqueeze(0).to(device)
            sync_features = get_sync_feature_extractor().encode_video_with_sync(sync_video)
        except Exception as e:
            print("Video file is empty or invalid, using zero video tensor.")
            Video_tensors = torch.zeros((seconds_total * target_fps, 3, 224, 224))   
            sync_features = torch.zeros(1, 240, 768).to(device)         
        audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))
    else:
        mask_type = None  # 如果两个都提供，则不需要 mask_type
        try:
            Video_tensors = video_read_local(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=target_fps)

            sync_video_tensor = video_read_local(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=25)
            sync_video=sync_transform(sync_video_tensor)
            sync_video = sync_video.unsqueeze(0).to(device)
            sync_features = get_sync_feature_extractor().encode_video_with_sync(sync_video)

        except Exception as e:
            print("Video file is empty or invalid, using zero video tensor.")
            Video_tensors = torch.zeros((seconds_total * target_fps, 3, 224, 224))
            sync_features = torch.zeros(1, 240, 768).to(device)
        try:
            audio_tensor = load_and_process_audio(audio_path, sample_rate, seconds_start, seconds_total)
        except Exception as e:
            print("Audio prompt file is empty or invalid, using zero audio tensor.")
            audio_tensor = torch.zeros((2, int(sample_rate * seconds_total)))  # 假设立体声音频
            
    # import pdb; pdb.set_trace()
    # if sync_feature_path is not None and os.path.exists(sync_feature_path):
    #     sync_features = torch.load(sync_feature_path, weights_only=True, map_location='cpu').to(device)            
    # else:
    #     sync_features = torch.zeros(1, 240, 768).to(device)
    
    # try:
    #     sync_features = torch.load(sync_feature_path, weights_only=True, map_location='cpu').to(device)
    #     # import pdb;pdb.set_trace()
    # except:
    #     # import pdb;pdb.set_trace()
    #     sync_video_tensor = video_read_local(video_path, seek_time=seconds_start, duration=seconds_total, target_fps=25)
    #     sync_video=sync_transform(sync_video_tensor)
    #     sync_video = sync_video.unsqueeze(0).to(device)
    #     sync_features = sync_feature_extractor.encode_video_with_sync(sync_video) 



    audio_tensor=audio_tensor.to(device)
    seconds_input=sample_size/sample_rate
    print(f'video_path: {video_path}')
    print(f'audio_path: {audio_path}')

    
    # Use default or empty string if prompt is not provided
    if not prompt:
        prompt = ""
    # import pdb; pdb.set_trace()
    
    conditioning = [{
        # "video_prompt": [Video_tensors.unsqueeze(0)],        
        "video_prompt": {"video_tensors":Video_tensors.unsqueeze(0), "video_sync_frames": sync_features},        
        "text_prompt": prompt,
        "audio_prompt": audio_tensor.unsqueeze(0),
        "seconds_start": seconds_start,
        "seconds_total": seconds_input
    }] * batch_size
    # import pdb; pdb.set_trace()
    if negative_prompt:
        negative_conditioning = [{
            "video_prompt": [Video_tensors.unsqueeze(0)],        
            "text_prompt": negative_prompt,
            "audio_prompt": audio_tensor.unsqueeze(0),
            "seconds_start": seconds_start,
            "seconds_total": seconds_total
        }] * batch_size
    else:
        negative_conditioning = None

    print(f"Model type: {model_type}")

    try:
        device = next(model.parameters()).device 
    except Exception as e:
        device = next(current_model.parameters()).device

    seed = int(seed)

    if not use_init:
        init_audio = None

    input_sample_size = sample_size

    if init_audio is not None:
        in_sr, init_audio = init_audio
        init_audio = torch.from_numpy(init_audio).float().div(32767)
        
        if init_audio.dim() == 1:
            init_audio = init_audio.unsqueeze(0)  # [1, n]
        elif init_audio.dim() == 2:
            init_audio = init_audio.transpose(0, 1)  # [n, 2] -> [2, n]

        if in_sr != sample_rate:
            resample_tf = T.Resample(in_sr, sample_rate).to(init_audio.device)
            init_audio = resample_tf(init_audio)

        audio_length = init_audio.shape[-1]

        if audio_length > sample_size:
            input_sample_size = audio_length + (model.min_input_length - (audio_length % model.min_input_length)) % model.min_input_length

        init_audio = (sample_rate, init_audio)

    def progress_callback(callback_info):
        nonlocal preview_images
        denoised = callback_info["denoised"]
        current_step = callback_info["i"]
        sigma = callback_info["sigma"]

        if (current_step - 1) % preview_every == 0:
            if model.pretransform is not None:
                denoised = model.pretransform.decode(denoised)
            denoised = rearrange(denoised, "b d n -> d (b n)")
            denoised = denoised.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_spectrogram = audio_spectrogram_image(denoised, sample_rate=sample_rate)
            preview_images.append((audio_spectrogram, f"Step {current_step} sigma={sigma:.3f})"))

    if mask_cropfrom is not None: 
        mask_args = {
            "cropfrom": mask_cropfrom,
            "pastefrom": mask_pastefrom,
            "pasteto": mask_pasteto,
            "maskstart": mask_maskstart,
            "maskend": mask_maskend,
            "softnessL": mask_softnessL,
            "softnessR": mask_softnessR,
            "marination": mask_marination,
        }
    else:
        mask_args = None 

    # 根据模型类型进行音频生成
    if model_type == "diffusion_cond":
      
        audio = generate_diffusion_cond(
            model, 
            conditioning=conditioning,
            negative_conditioning=negative_conditioning,
            steps=steps,
            cfg_scale=cfg_scale,
            batch_size=batch_size,
            sample_size=input_sample_size,
            sample_rate=sample_rate,
            seed=seed,
            device=device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
            mask_args=mask_args,
            callback=progress_callback if preview_every is not None else None,
            scale_phi=cfg_rescale
        )
        # import pdb; pdb.set_trace()
    elif model_type == "diffusion_uncond":
        audio = generate_diffusion_uncond(
            model, 
            steps=steps,
            batch_size=batch_size,
            sample_size=input_sample_size,
            seed=seed,
            device=device,
            sampler_type=sampler_type,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            init_audio=init_audio,
            init_noise_level=init_noise_level,
            callback=progress_callback if preview_every is not None else None
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 转换为 WAV 文件
    audio = rearrange(audio, "b d n -> d (b n)")
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    torchaudio.save("output.wav", audio, sample_rate)

    file_name = os.path.basename(video_path) if video_path else "output"

    output_dir = f"demo_result/{model_name}_prompt_{prompt[:10]}"
    output_video_path = f"{output_dir}/{file_name}"

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if video_path:
        merge_video_audio(video_path, "output.wav", output_video_path, seconds_start, seconds_total)
        
    audio_spectrogram = audio_spectrogram_image(audio, sample_rate=sample_rate)
    
    # 释放模型和显存
    del model
    torch.cuda.empty_cache()
    gc.collect()

    return (output_video_path, "output.wav", [audio_spectrogram, *preview_images])

def toggle_custom_model(selected_model):
    return gr.Row.update(visible=(selected_model == "Custom Model"))



def create_sampling_ui(model_options, model_config_map, inpainting=False):
    with gr.Blocks() as demo:
        # 模型选择
        with gr.Row():
            with gr.Column(scale=2):
                model_dropdown = gr.Dropdown(
                    choices=list(model_options.keys()) + ["Custom Model"],  # 添加 "Custom Model" 选项
                    label="Select Model",
                    value=list(model_options.keys())[0]
                )
            with gr.Column(scale=5):
                pass  # 保持空白或添加其他内容

        # 输入提示
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(show_label=False, placeholder="Enter your prompt")
                negative_prompt = gr.Textbox(show_label=False, placeholder="Negative prompt")
                
                # 视频输入选项
                video_file = gr.File(label="Upload Video File")
                video_path = gr.Textbox(label="Video Path", placeholder="Enter video file path")
                sync_feature_path = gr.Textbox(label="Sync feature Path", placeholder="Enter sync feature file path")                
                # 音频输入选项
                audio_prompt_file = gr.File(label="Upload Audio Prompt File")
                audio_prompt_path = gr.Textbox(label="Audio Prompt Path", placeholder="Enter audio file path")

        # 自定义模型输入字段，默认隐藏
        with gr.Row(visible=False) as custom_model_row:
            with gr.Column():
                custom_model_config = gr.Textbox(label="Custom Model Config Path", placeholder="Path to model_config.json")
                custom_ckpt_path = gr.Textbox(label="Custom Model Checkpoint Path", placeholder="Path to model.ckpt")

        # 绑定模型选择下拉菜单，控制自定义模型输入字段的显示
        model_dropdown.change(
            fn=toggle_custom_model,
            inputs=[model_dropdown],
            outputs=[custom_model_row]
        )

        # Timing controls
        with gr.Row():
            with gr.Column(scale=6):
                seconds_start_slider = gr.Slider(minimum=0, maximum=512, step=1, value=0, label="Seconds Start")
                seconds_total_slider = gr.Slider(minimum=1, maximum=300, step=1, value=10, label="Seconds Total")

        # 生成参数
        with gr.Row():
            with gr.Column(scale=4):
                steps_slider = gr.Slider(minimum=1, maximum=500, step=1, value=100, label="Steps")
                preview_every_slider = gr.Slider(minimum=0, maximum=100, step=1, value=0, label="Preview Every")
                cfg_scale_slider = gr.Slider(minimum=0.0, maximum=25.0, step=0.1, value=7.0, label="CFG Scale")

        # Sampler 参数
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Sampler Params", open=False):
                    seed_textbox = gr.Textbox(label="Seed (set to -1 for random seed)", value="-1")
                    sampler_type_dropdown = gr.Dropdown(
                        ["dpmpp-2m-sde", "dpmpp-3m-sde", "k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast"],
                        label="Sampler Type",
                        value="dpmpp-3m-sde"
                    )
                    sigma_min_slider = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.03, label="Sigma Min")
                    sigma_max_slider = gr.Slider(minimum=0.0, maximum=1000.0, step=0.1, value=500, label="Sigma Max")
                    cfg_rescale_slider = gr.Slider(minimum=0.0, maximum=1, step=0.01, value=0.0, label="CFG Rescale Amount")

        # Init Audio 参数
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Accordion("Init Audio", open=False):
                    init_audio_checkbox = gr.Checkbox(label="Use Init Audio")
                    init_audio_input = gr.Audio(label="Init Audio")
                    init_noise_level_slider = gr.Slider(minimum=0.1, maximum=100.0, step=0.01, value=0.1, label="Init Noise Level")

        if inpainting: 
            # Inpainting 参数
            with gr.Accordion("Inpainting", open=False):
                mask_cropfrom_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Crop From %")
                mask_pastefrom_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Paste From %")
                mask_pasteto_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Paste To %")
                mask_maskstart_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=50, label="Mask Start %")
                mask_maskend_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=100, label="Mask End %")
                mask_softnessL_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Left Crossfade Length %")
                mask_softnessR_slider = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, value=0, label="Softmask Right Crossfade Length %")
                mask_marination_slider = gr.Slider(minimum=0.0, maximum=1, step=0.0001, value=0, label="Marination Level", visible=False)

        # Generate按钮居中对齐
        with gr.Row():
            generate_button = gr.Button("Generate", variant='primary', scale=1)

        # 输出组件
        with gr.Row():
            with gr.Column(scale=6):
                video_output = gr.Video(label="Output Video", interactive=False)
                audio_output = gr.Audio(label="Output Audio", interactive=False)
                audio_spectrogram_output = gr.Gallery(label="Output Spectrogram", show_label=False)
                send_to_init_button = gr.Button("Send to Init Audio", scale=1)

        # 绑定发送按钮，将生成的音频传递给 init_audio_input
        send_to_init_button.click(
            fn=lambda audio: audio,
            inputs=[audio_output],
            outputs=[init_audio_input]
        )

        # 绑定生成按钮
        if inpainting:
            inputs = [
                prompt, 
                model_dropdown,
                negative_prompt,
                video_file,
                video_path,
                audio_prompt_file,
                audio_prompt_path,
                seconds_start_slider, 
                seconds_total_slider, 
                cfg_scale_slider, 
                steps_slider, 
                preview_every_slider, 
                seed_textbox, 
                sampler_type_dropdown, 
                sigma_min_slider, 
                sigma_max_slider,
                cfg_rescale_slider,
                init_audio_checkbox,
                init_audio_input,
                init_noise_level_slider,
                mask_cropfrom_slider,
                mask_pastefrom_slider,
                mask_pasteto_slider,
                mask_maskstart_slider,
                mask_maskend_slider,
                mask_softnessL_slider,
                mask_softnessR_slider,
                mask_marination_slider,
                custom_model_config,
                custom_ckpt_path
            ]
        else:
            # 默认生成标签页
            inputs = [
                prompt, 
                model_dropdown,
                negative_prompt,
                video_file,
                video_path,
                sync_feature_path,
                audio_prompt_file,
                audio_prompt_path,
                seconds_start_slider, 
                seconds_total_slider, 
                cfg_scale_slider, 
                steps_slider, 
                preview_every_slider, 
                seed_textbox, 
                sampler_type_dropdown, 
                sigma_min_slider, 
                sigma_max_slider,
                cfg_rescale_slider,
                init_audio_checkbox,
                init_audio_input,
                init_noise_level_slider,
                custom_model_config,
                custom_ckpt_path
            ]

        generate_button.click(
            fn=generate_cond, 
            inputs=inputs,
            outputs=[
                video_output,
                audio_output, 
                audio_spectrogram_output
            ], 
            api_name="generate"
        )

        examples = [
                        [
                            "A happy music",                   # prompt
                            list(model_options.keys())[0],     # model_dropdown
                            "",                                 # negative_prompt
                            "./demo_result/a6f086e4-epoch=17-step=45000 (music only)_prompt_music/wgKb7Q9hyp0.mp4",              # video_file，提供一个实际存在的本地视频文件路径
                            "",                                 # video_path，不提供则空字符串
                            None,                               # audio_prompt_file不提供文件则None
                            "",                                 # audio_prompt_path
                            0,                                  # seconds_start_slider
                            10,                                 # seconds_total_slider
                            7.0,                                # cfg_scale_slider
                            100,                                # steps_slider
                            0,                                  # preview_every_slider
                            "-1",                               # seed_textbox
                            "dpmpp-3m-sde",                     # sampler_type_dropdown
                            0.03,                                # sigma_min_slider
                            500,                                 # sigma_max_slider
                            0.0,                                # cfg_rescale_slider
                            False,                              # init_audio_checkbox
                            None,                               # init_audio_input
                            0.1,                                # init_noise_level_slider
                            None,                               # custom_model_config
                            None                                # custom_ckpt_path
                        ],
                    ]
            
        # gr.Examples(
        #     examples=examples,
        #     inputs=inputs,
        #     outputs=[
        #         video_output,
        #         audio_output,
        #         audio_spectrogram_output
        #     ],
        #     label="Examples"
        # )
        
        return demo
    
def create_txt2audio_ui(model_options, model_config_map):
    with gr.Blocks() as ui:
        with gr.Tab("Generation"):
            create_sampling_ui(model_options, model_config_map)
        with gr.Tab("Inpainting"):
            create_sampling_ui(model_options, model_config_map, inpainting=True)    
    return ui

def toggle_custom_model(selected_model):
    return gr.Row.update(visible=(selected_model == "Custom Model"))

def create_ui(model_config_path=None, ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, model_half=False):
    global model_configurations
    global device

    # 确保只有预设模型或提供的模型被加载
    # assert (pretrained_name is not None) ^ (model_config_path is not None and ckpt_path is not None), "Must specify either pretrained name or provide a model config and checkpoint, but not both"

    if model_config_path is not None:
        # Load config from json file
        with open(model_config_path) as f:
            model_configs = json.load(f)
    else:
        model_configs = None

    try:
        has_mps = platform.system() == "Darwin" and torch.backends.mps.is_available()
    except Exception:
        # In case this version of Torch doesn't even have `torch.backends.mps`...
        has_mps = False

    if has_mps:
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    # 假设 model_configs 是一个字典，包含多个模型的信息
    if isinstance(model_configs, dict):
        model_options = model_configs
    elif isinstance(model_configs, list):
        # 如果是列表，每个元素是一个模型的配置
        model_options = {f"model_{i}": cfg for i, cfg in enumerate(model_configs)}
    else:
        model_options = {"default": {"model_config": model_config_path, "ckpt_path": ckpt_path, "pretrained_name": pretrained_name}}

    # 加载所有模型的配置，但不实际加载模型
    for model_name, cfg in model_options.items():
        model_config = cfg.get("model_config")
        ckpt_path = cfg.get("ckpt_path")
        pretrained_name = cfg.get("pretrained_name")
        pretransform_ckpt_path = cfg.get("pretransform_ckpt_path")
        model_type = cfg.get("model_type", "diffusion_cond")  # 默认模型类型

        model_configurations[model_name] = {
            "model_config": model_config,
            "ckpt_path": ckpt_path,
            "pretrained_name": pretrained_name,
            "pretransform_ckpt_path": pretransform_ckpt_path,
            "model_type": model_type
        }

    # 创建 Gradio 界面
    ui = create_txt2audio_ui(model_options, model_configurations)
    return ui

# 入口点
if __name__ == "__main__":
    # Provide the model config via AUDIOX_TURBO_DEMO_CONFIG, or pass --config / a JSON
    # launch config. Defaults to the packaged 4-step inference config.
    model_config_path = os.environ.get(
        "AUDIOX_TURBO_DEMO_CONFIG",
        "configs/audiox_turbo_infer_4step.json",
    )
    ui = create_ui(
        model_config_path=model_config_path,
    )
    ui.launch(share=True)

import torch


def get_custom_metadata(info: dict, audio: torch.Tensor) -> dict:
    """Build the conditioning dict for a ``text_condition-audio`` sample.

    Returns a dict with:
      - ``video_prompt``: zero video tensor + zero sync features (text-only mode).
      - ``text_prompt``: the audio caption string.
      - ``audio_prompt``: mono audio reference padded to 10 s.
      - ``video_sync_feature``: zero sync features.
    """
    seek_time  = info["seconds_start"]
    fps        = info["video_fps"]
    audio_prompt = info["audio_prompt"]

    length_set  = 10
    sample_rate = 44100

    # Extract the audio caption for the current time window.
    caption_raw = info["text_prompt"].get("audio_caption", "")
    if isinstance(caption_raw, dict):
        key = f"{int(seek_time)}-{int(seek_time) + length_set}"
        caption = caption_raw.get(key, next(iter(caption_raw.values()), ""))
    else:
        caption = str(caption_raw)

    text_prompt  = caption
    video_tensor = torch.zeros(length_set * fps, 3, 224, 224)
    sync_frames  = torch.zeros(1, 240, 768)

    # Normalise audio_prompt to mono and pad / trim to exactly 10 s.
    if audio_prompt.shape[0] == 2:
        audio_prompt = audio_prompt.mean(dim=0, keepdim=True)
    target_len = sample_rate * length_set
    if audio_prompt.shape[1] < target_len:
        pad = target_len - audio_prompt.shape[1]
        audio_prompt = torch.nn.functional.pad(audio_prompt, (0, pad))
    else:
        audio_prompt = audio_prompt[:, :target_len]

    return {
        "video_prompt": {
            "video_tensors": video_tensor.unsqueeze(0),
            "video_sync_frames": sync_frames,
        },
        "text_prompt": text_prompt,
        "audio_prompt": audio_prompt.unsqueeze(0),
        "video_sync_feature": sync_frames,
    }

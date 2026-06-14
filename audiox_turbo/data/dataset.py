import importlib
import io
import json
import numpy as np
import os
import posixpath
import random
import re
import subprocess
import time
import torch
import torchaudio
import webdataset as wds

from aeiou.core import is_silence
from os import path
from pedalboard.io import AudioFile
from torchaudio import transforms as T
from typing import Optional, Callable, List
from torch.utils.data import Sampler

from .utils import Stereo, Mono, PhaseFlipper, PadCrop_Normalized_T


AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")


class RandomEpochSampler(Sampler):
    """Sampler that draws a fresh random subset each epoch."""

    def __init__(self, dataset_size, epoch_size, replacement=True):
        self.dataset_size = dataset_size
        self.epoch_size = epoch_size
        self.replacement = replacement
        if not replacement and epoch_size > dataset_size:
            raise ValueError("epoch_size cannot be greater than dataset_size when replacement=False")

    def __iter__(self):
        if self.replacement:
            indices = torch.randint(0, self.dataset_size, (self.epoch_size,), dtype=torch.long)
        else:
            indices = torch.randperm(self.dataset_size)[:self.epoch_size]
        return iter(indices.tolist())

    def __len__(self):
        return self.epoch_size


def fast_scandir(dir: str, ext: list):
    """Recursively scan a directory for files with the given extensions."""
    subfolders, files = [], []
    ext = ['.' + x if x[0] != '.' else x for x in ext]
    try:
        for f in os.scandir(dir):
            try:
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")
                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass
    except:
        pass
    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def extract_audio_paths(jsonl_file, exts):
    audio_paths, video_paths, video_sync_frames, text_prompts, data_types = [], [], [], [], []
    with open(jsonl_file, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                p = data.get('path', '')
                video_path = data.get('video_path', '')
                text_prompt = data.get('caption', '')
                data_type = data.get('type', None)
                video_sync_features = data.get('sync_feature', None)
                if any(p.endswith(ext) for ext in exts):
                    audio_paths.append(p)
                    video_paths.append(video_path)
                    text_prompts.append(text_prompt)
                    data_types.append(data_type)
                    video_sync_frames.append(video_sync_features)
            except json.JSONDecodeError:
                print(f"Error decoding JSON line: {line.strip()}")
    return audio_paths, video_paths, video_sync_frames, text_prompts, data_types


def keyword_scandir(dir: str, ext: list, keywords: list):
    """Recursively scan a directory for files matching given extensions and keywords."""
    subfolders, files = [], []
    keywords = [keyword.lower() for keyword in keywords]
    ext = ['.' + x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:
        for f in os.scandir(dir):
            try:
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(keyword in name_lower for keyword in keywords)
                    has_banned = any(banned_word in name_lower for banned_word in banned_words)
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass
    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files


def get_audio_filenames(paths: list, keywords=None,
                        exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']):
    """Recursively get a list of audio filenames."""
    filenames, video_filenames, video_sync_frames, text_prompts, data_types = [], [], [], [], []

    if type(paths) is str:
        paths = [paths]

    if os.path.isdir(paths[0]):
        for p in paths:
            if keywords is not None:
                subfolders, files = keyword_scandir(p, exts, keywords)
            else:
                subfolders, files = fast_scandir(p, exts)
            filenames.extend(files)
        return filenames

    elif os.path.isfile(paths[0]):
        assert paths[0].endswith('.jsonl')
        for p in paths:
            audio_paths, video_paths, vsf, text_prompt, data_type = extract_audio_paths(p, exts)
            filenames.extend(audio_paths)
            video_filenames.extend(video_paths)
            video_sync_frames.extend(vsf)
            text_prompts.extend(text_prompt)
            data_types.extend(data_type)
        return filenames, video_filenames, video_sync_frames, text_prompts, data_types


def random_silence(audio_tensor, sample_rate, min_second, max_second):
    """Randomly zero out a segment of an audio tensor."""
    total_samples = audio_tensor.size(1)
    zero_duration_seconds = random.uniform(min_second, max_second)
    zero_duration_samples = int(zero_duration_seconds * sample_rate)
    max_start_index = total_samples - zero_duration_samples
    if max_start_index > 0:
        start_index = random.randint(0, max_start_index)
        audio_tensor[:, start_index:start_index + zero_duration_samples] = 0
    return audio_tensor


class LocalDatasetConfig:
    def __init__(self, id: str, path: str, video_fps: int,
                 custom_metadata_fn: Optional[Callable[[str], str]] = None):
        self.id = id
        self.path = path
        self.video_fps = video_fps
        self.custom_metadata_fn = custom_metadata_fn


class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, configs, sample_size=65536, sample_rate=48000, keywords=None,
                 random_crop=True, force_channels="stereo", video_fps=5):
        super().__init__()
        self.filenames = []
        self.video_filenames = []
        self.video_sync_frames = []
        self.text_prompts = []
        self.data_types = []
        self.augs = torch.nn.Sequential(PhaseFlipper())
        self.root_paths = []
        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)
        self.force_channels = force_channels
        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )
        self.sr = sample_rate
        self.sample_size = sample_size
        self.custom_metadata_fns = {}

        for config in configs:
            self.video_fps = config.video_fps
            self.root_paths.append(config.path)
            audio_files, video_files, vsf, text_prompt, data_types = get_audio_filenames(config.path, keywords)
            self.filenames.extend(audio_files)
            self.video_filenames.extend(video_files)
            self.video_sync_frames.extend(vsf)
            self.text_prompts.extend(text_prompt)
            self.data_types.extend(data_types)
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn

        print(f'Found {len(self.filenames)} files')

    def load_file(self, filename):
        ext = filename.split(".")[-1]
        if ext == "mp3":
            try:
                with AudioFile(filename) as f:
                    audio = torch.from_numpy(f.read(f.frames))
                    in_sr = f.samplerate
            except Exception as e:
                print(f"Couldn't load file {filename}: {e}, use zero audio")
                audio = torch.zeros((2, self.sample_size))
                in_sr = self.sr
        else:
            audio, in_sr = torchaudio.load(filename, format=ext)
        if in_sr != self.sr:
            audio = T.Resample(in_sr, self.sr)(audio)
        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        video_filename = self.video_filenames[idx]
        video_sync_frames = self.video_sync_frames[idx]
        text_prompt = self.text_prompts[idx]
        data_type = self.data_types[idx]

        try:
            start_time = time.time()

            try:
                audio = self.load_file(audio_filename)
            except Exception as e:
                print(f"Couldn't load file {audio_filename}: {e}, use zero audio")
                audio = torch.zeros((2, self.sample_size))

            if data_type in ["text_condition-audio", "text_condition-music",
                             "video_condition-audio", "video_condition-music",
                             "text+video_condition-audio", "text+video_condition-music"]:
                if_audio_contition = False
                audio_prompt = torch.zeros((2, self.sr * 10))
            elif data_type in ["audio_condition-audio", "audio_condition-music",
                               "uni_condition-audio", "uni_condition-music"]:
                if_audio_contition = True

            if if_audio_contition:
                audio_org = audio.clamp(-1, 1)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            if if_audio_contition:
                if data_type.split("-")[-1] == "audio":
                    start_index = max(0, int(seconds_start * self.sr))
                    end_index = int((seconds_start + 10) * self.sr)
                    audio_prompt = audio_org[:, start_index:end_index]
                elif data_type.split("-")[-1] == "music":
                    if seconds_start < 10:
                        start_index = 0
                        end_index = int(10 * self.sr)
                    else:
                        start_index = max(0, int((seconds_start - 10) * self.sr))
                        end_index = int(seconds_start * self.sr)
                    audio_prompt = audio_org[:, start_index:end_index]

            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}
            info["path"] = audio_filename
            info["video_path"] = video_filename
            info["video_sync_frames"] = video_sync_frames
            info["text_prompt"] = text_prompt
            info["audio_prompt"] = audio_prompt
            info["data_type"] = data_type

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)

            info["timestamps"] = (t_start, t_end)
            info["seconds_start"] = seconds_start
            info["seconds_total"] = seconds_total
            info["padding_mask"] = padding_mask
            info["video_fps"] = self.video_fps
            info["load_time"] = time.time() - start_time

            for custom_md_path in self.custom_metadata_fns.keys():
                if os.path.isdir(custom_md_path):
                    if custom_md_path in audio_filename:
                        custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                        custom_metadata = custom_metadata_fn(info, audio)
                        info.update(custom_metadata)
                elif os.path.isfile(custom_md_path):
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

            return (audio, info)
        except Exception as e:
            print(f"Couldn't load file {audio_filename}: {e}")
            return self[random.randrange(len(self))]


def group_by_keys(data, keys=wds.tariterators.base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples."""
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if wds.tariterators.trace:
            print(prefix, suffix, current_sample.keys() if isinstance(current_sample, dict) else None)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        if current_sample is None or prefix != current_sample["__key__"]:
            if wds.tariterators.valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffix in current_sample:
            print(f"{fname}: duplicate file name in tar file {suffix} {current_sample.keys()}")
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if wds.tariterators.valid_sample(current_sample):
        yield current_sample


wds.tariterators.group_by_keys = group_by_keys


def get_s3_contents(dataset_path, s3_url_prefix=None, filter='', recursive=True, debug=False, profile=None):
    """Return a list of full S3 paths to files in a given S3 bucket and directory path."""
    if dataset_path != '' and not dataset_path.endswith('/'):
        dataset_path += '/'
    bucket_path = posixpath.join(s3_url_prefix or '', dataset_path)
    cmd = ['aws', 's3', 'ls', bucket_path]
    if profile is not None:
        cmd.extend(['--profile', profile])
    if recursive:
        cmd.append('--recursive')
    run_ls = subprocess.run(cmd, capture_output=True, check=True)
    contents = run_ls.stdout.decode('utf-8').split('\n')
    contents = [x.strip() for x in contents if x]
    contents = [re.sub(r'^\S+\s+\S+\s+\d+\s+', '', x)
                if re.match(r'^\S+\s+\S+\s+\d+\s+', x) else x for x in contents]
    contents = [posixpath.join(s3_url_prefix or '', x) for x in contents if not x.endswith('/')]
    if filter:
        contents = [x for x in contents if filter in x]
    if recursive:
        main_dir = "/".join(bucket_path.split('/')[3:])
        contents = [x.replace(f'{main_dir}', '').replace('//', '/') for x in contents]
    if debug:
        print("contents = \n", contents)
    return contents


def get_all_s3_urls(names=[], subsets=[''], s3_url_prefix=None, recursive=True,
                    filter_str='tar', debug=False, profiles={}):
    """Get URLs of shards (tar files) for multiple datasets in one S3 bucket."""
    urls = []
    for name in names:
        contents_str = name if s3_url_prefix is None else posixpath.join(s3_url_prefix, name)
        if debug:
            print(f"get_all_s3_urls: {contents_str}:")
        for subset in subsets:
            subset_str = posixpath.join(contents_str, subset)
            if debug:
                print(f"subset_str = {subset_str}")
            profile = profiles.get(name, None)
            tar_list = get_s3_contents(subset_str, s3_url_prefix=None, recursive=recursive,
                                       filter=filter_str, debug=debug, profile=profile)
            for tar in tar_list:
                tar = tar.replace(" ", r"\ ").replace("(", r"\(").replace(")", r"\)")
                s3_path = posixpath.join(name, subset, tar) + " -"
                if s3_url_prefix is None:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {s3_path}"
                else:
                    request_str = f"pipe:aws s3 --cli-connect-timeout 0 cp {posixpath.join(s3_url_prefix, s3_path)}"
                if profiles.get(name):
                    request_str += f" --profile {profiles.get(name)}"
                if debug:
                    print("request_str = ", request_str)
                urls.append(request_str)
    return urls


def log_and_continue(exn):
    """Ignore any webdataset exception and continue."""
    print(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def is_valid_sample(sample):
    has_json = "json" in sample
    has_audio = "audio" in sample
    is_silent = is_silence(sample["audio"])
    is_rejected = "__reject__" in sample["json"] and sample["json"]["__reject__"]
    return has_json and has_audio and not is_silent and not is_rejected


class S3DatasetConfig:
    def __init__(self, id: str, s3_path: str,
                 custom_metadata_fn: Optional[Callable[[str], str]] = None,
                 profile: Optional[str] = None):
        self.id = id
        self.path = s3_path
        self.custom_metadata_fn = custom_metadata_fn
        self.profile = profile
        self.urls = []

    def load_data_urls(self):
        self.urls = get_all_s3_urls(
            names=[self.path], s3_url_prefix=None, recursive=True,
            profiles={self.path: self.profile} if self.profile else {},
        )
        return self.urls


class LocalWebDatasetConfig:
    def __init__(self, id: str, path: str,
                 custom_metadata_fn: Optional[Callable[[str], str]] = None,
                 profile: Optional[str] = None):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.urls = []

    def load_data_urls(self):
        self.urls = fast_scandir(self.path, ["tar"])[1]
        return self.urls


def audio_decoder(key, value):
    ext = key.split(".")[-1]
    if ext in AUDIO_KEYS:
        return torchaudio.load(io.BytesIO(value))
    return None


def collation_fn(samples):
    batched = list(zip(*samples))
    result = []
    for b in batched:
        if isinstance(b[0], (int, float)):
            b = np.array(b)
        elif isinstance(b[0], torch.Tensor):
            b = torch.stack(b)
        elif isinstance(b[0], np.ndarray):
            b = np.array(b)
        result.append(b)
    return result


class WebDatasetDataLoader():
    def __init__(self, datasets: List[S3DatasetConfig], batch_size, sample_size,
                 sample_rate=48000, num_workers=8, epoch_steps=1000, random_crop=True,
                 force_channels="stereo", augment_phase=True, **data_loader_kwargs):
        self.datasets = datasets
        self.sample_size = sample_size
        self.sample_rate = sample_rate
        self.random_crop = random_crop
        self.force_channels = force_channels
        self.augment_phase = augment_phase

        urls = [url for dataset in datasets for url in dataset.load_data_urls()]
        random.shuffle(urls)

        self.dataset = wds.DataPipeline(
            wds.ResampledShards(urls),
            wds.tarfile_to_samples(handler=log_and_continue),
            wds.decode(audio_decoder, handler=log_and_continue),
            wds.map(self.wds_preprocess, handler=log_and_continue),
            wds.select(is_valid_sample),
            wds.to_tuple("audio", "json", handler=log_and_continue),
            wds.batched(batch_size, partial=False, collation_fn=collation_fn),
        ).with_epoch(epoch_steps // num_workers if num_workers > 0 else epoch_steps)

        self.data_loader = wds.WebLoader(self.dataset, num_workers=num_workers, **data_loader_kwargs)

    def wds_preprocess(self, sample):
        found_key, rewrite_key = '', ''
        for k in sample:
            for akey in AUDIO_KEYS:
                if k.endswith(akey):
                    found_key, rewrite_key = k, akey
                    break
            if found_key:
                break
        if not found_key:
            return None

        audio, in_sr = sample[found_key]
        if in_sr != self.sample_rate:
            audio = T.Resample(in_sr, self.sample_rate)(audio)

        if self.sample_size is not None:
            pad_crop = PadCrop_Normalized_T(self.sample_size, randomize=self.random_crop,
                                            sample_rate=self.sample_rate)
            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = pad_crop(audio)
            sample["json"]["seconds_start"] = seconds_start
            sample["json"]["seconds_total"] = seconds_total
            sample["json"]["padding_mask"] = padding_mask
        else:
            t_start, t_end = 0, 1

        if audio.shape[-1] == 0:
            audio = torch.zeros(1, 1)

        augs = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
            PhaseFlipper() if self.augment_phase else torch.nn.Identity(),
        )
        audio = augs(audio)

        sample["json"]["timestamps"] = (t_start, t_end)

        if "text" in sample["json"]:
            sample["json"]["prompt"] = sample["json"]["text"]

        for dataset in self.datasets:
            if dataset.custom_metadata_fn is None:
                continue
            if dataset.path in sample["__url__"]:
                custom_metadata = dataset.custom_metadata_fn(sample["json"], audio)
                sample["json"].update(custom_metadata)

        if found_key != rewrite_key:
            del sample[found_key]

        sample["audio"] = audio
        sample["json"]["audio"] = audio
        return sample


def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate,
                                  audio_channels=2, num_workers=4, video_fps=5):
    dataset_type = dataset_config.get("dataset_type", None)
    assert dataset_type is not None, "Dataset type must be specified in dataset config"
    force_channels = "mono" if audio_channels == 1 else "stereo"

    if dataset_type == "audio_dir":
        audio_dir_configs = dataset_config.get("datasets", None)
        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        configs = []
        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"
            custom_metadata_fn = None
            custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)
            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)
                custom_metadata_fn = metadata_module.get_custom_metadata
            configs.append(LocalDatasetConfig(
                id=audio_dir_config["id"], path=audio_dir_path,
                custom_metadata_fn=custom_metadata_fn, video_fps=video_fps,
            ))

        train_set = SampleDataset(
            configs, sample_rate=sample_rate, sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels, video_fps=video_fps,
        )

        epoch_size = dataset_config["datasets"][0].get("epoch_size", None)
        if epoch_size is not None and epoch_size > 0:
            print(f"Epoch sampling mode: {epoch_size} samples/epoch (total: {len(train_set)})")
            sampler = RandomEpochSampler(dataset_size=len(train_set), epoch_size=epoch_size, replacement=True)
            return torch.utils.data.DataLoader(
                train_set, batch_size, sampler=sampler, shuffle=False,
                num_workers=num_workers, persistent_workers=False,
                pin_memory=False, drop_last=True, collate_fn=collation_fn,
            )
        else:
            print(f"Default mode: {len(train_set)} samples/epoch")
            return torch.utils.data.DataLoader(
                train_set, batch_size, shuffle=True,
                num_workers=num_workers, persistent_workers=False,
                pin_memory=False, drop_last=True, collate_fn=collation_fn,
            )

    elif dataset_type in ["s3", "wds"]:
        wds_configs = []
        for wds_config in dataset_config["datasets"]:
            custom_metadata_fn = None
            custom_metadata_module_path = wds_config.get("custom_metadata_module", None)
            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)
                custom_metadata_fn = metadata_module.get_custom_metadata
            if "s3_path" in wds_config:
                wds_configs.append(S3DatasetConfig(
                    id=wds_config["id"], s3_path=wds_config["s3_path"],
                    custom_metadata_fn=custom_metadata_fn,
                    profile=wds_config.get("profile", None),
                ))
            elif "path" in wds_config:
                wds_configs.append(LocalWebDatasetConfig(
                    id=wds_config["id"], path=wds_config["path"],
                    custom_metadata_fn=custom_metadata_fn,
                ))
        return WebDatasetDataLoader(
            wds_configs, sample_rate=sample_rate, sample_size=sample_size,
            batch_size=batch_size, random_crop=dataset_config.get("random_crop", True),
            num_workers=num_workers, persistent_workers=True,
            force_channels=force_channels,
            epoch_steps=dataset_config.get("epoch_steps", 2000),
        ).data_loader


def create_dataloader_from_config_valid(dataset_config, batch_size, sample_size, sample_rate,
                                        audio_channels=2, num_workers=4):
    dataset_type = dataset_config.get("dataset_type", None)
    assert dataset_type is not None, "Dataset type must be specified in dataset config"
    force_channels = "mono" if audio_channels == 1 else "stereo"

    if dataset_type == "audio_dir":
        audio_dir_configs = dataset_config.get("datasets", None)
        assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

        configs = []
        for audio_dir_config in audio_dir_configs:
            audio_dir_path = audio_dir_config.get("path", None)
            assert audio_dir_path is not None, "Path must be set for local audio directory configuration"
            custom_metadata_fn = None
            custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)
            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)
                custom_metadata_fn = metadata_module.get_custom_metadata
            configs.append(LocalDatasetConfig(
                id=audio_dir_config["id"], path=audio_dir_path,
                custom_metadata_fn=custom_metadata_fn,
            ))

        valid_set = SampleDataset(
            configs, sample_rate=sample_rate, sample_size=sample_size,
            random_crop=dataset_config.get("random_crop", True),
            force_channels=force_channels,
        )
        return torch.utils.data.DataLoader(
            valid_set, batch_size, shuffle=False,
            num_workers=num_workers, persistent_workers=False,
            pin_memory=True, drop_last=True, collate_fn=collation_fn,
        )

    elif dataset_type in ["s3", "wds"]:
        wds_configs = []
        for wds_config in dataset_config["datasets"]:
            custom_metadata_fn = None
            custom_metadata_module_path = wds_config.get("custom_metadata_module", None)
            if custom_metadata_module_path is not None:
                spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
                metadata_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(metadata_module)
                custom_metadata_fn = metadata_module.get_custom_metadata
            if "s3_path" in wds_config:
                wds_configs.append(S3DatasetConfig(
                    id=wds_config["id"], s3_path=wds_config["s3_path"],
                    custom_metadata_fn=custom_metadata_fn,
                    profile=wds_config.get("profile", None),
                ))
            elif "path" in wds_config:
                wds_configs.append(LocalWebDatasetConfig(
                    id=wds_config["id"], path=wds_config["path"],
                    custom_metadata_fn=custom_metadata_fn,
                ))
        return WebDatasetDataLoader(
            wds_configs, sample_rate=sample_rate, sample_size=sample_size,
            batch_size=batch_size, random_crop=dataset_config.get("random_crop", True),
            num_workers=num_workers, persistent_workers=True,
            force_channels=force_channels,
            epoch_steps=dataset_config.get("epoch_steps", 2000),
        ).data_loader

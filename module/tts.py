import io
import importlib
import importlib.util
import ctypes
import json
import locale
import logging
import os
import pathlib
import queue
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Any, Generator, Optional, Protocol, cast
from urllib.parse import unquote, urlparse

import numpy as np

from .config import load_config

log = logging.getLogger(__name__)


_MODULE_CONFIG = load_config()
SAMPLE_RATE = _MODULE_CONFIG.get('playout', {}).get('sample_rate', 16000)
CHANNELS = _MODULE_CONFIG.get('playout', {}).get('channels', 1)
BYTES_PER_SAMPLE = 2

_DICT_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'dictionary.json'
_READERS_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'configs' / 'readers.xml'
_VOICES_PATH = pathlib.Path(__file__).parent.parent / 'voices'
_KOKORO_ROOT = _VOICES_PATH / 'kokoro'
_KOKORO_HF_HOME = _KOKORO_ROOT / '.hf'
_KOKORO_REPO_ID = 'hexgrad/Kokoro-82M'
_KOKORO_MODEL_FILE = 'kokoro-v1_0.pth'
_KOKORO_CONFIG_FILE = 'config.json'
_KOKORO_SAMPLE_RATE = 24000
_KOKORO_LANG_ALIASES = {
    'a': 'a',
    'en': 'a',
    'en-us': 'a',
    'en-ca': 'a',
    'b': 'b',
    'en-gb': 'b',
    'en-uk': 'b',
    'e': 'e',
    'es': 'e',
    'es-es': 'e',
    'f': 'f',
    'fr': 'f',
    'fr-fr': 'f',
    'fr-ca': 'f',
    'h': 'h',
    'hi': 'h',
    'i': 'i',
    'it': 'i',
    'it-it': 'i',
    'j': 'j',
    'ja': 'j',
    'ja-jp': 'j',
    'p': 'p',
    'pt': 'p',
    'pt-br': 'p',
    'z': 'z',
    'zh': 'z',
    'zh-cn': 'z',
}
_KOKORO_DEFAULT_VOICES = {
    'a': {'male': 'am_adam', 'female': 'af_heart'},
    'b': {'male': 'bm_george', 'female': 'bf_emma'},
    'e': {'male': 'em_alex', 'female': 'ef_dora'},
    'f': {'male': 'ff_siwis', 'female': 'ff_siwis'},
    'h': {'male': 'hm_omega', 'female': 'hf_alpha'},
    'i': {'male': 'im_nicola', 'female': 'if_sara'},
    'j': {'male': 'jm_kumo', 'female': 'jf_alpha'},
    'p': {'male': 'pm_alex', 'female': 'pf_dora'},
    'z': {'male': 'zm_yunjian', 'female': 'zf_xiaobei'},
}
_DICT_MB_RE = re.compile(r'(\d+)\s+MB\b')
_loaded_dictionary: dict[str, dict[str, str]] | None = None
_compiled_dictionary: dict[str, list[tuple[re.Pattern[str], str]]] = {}
_readers_cache: list[dict[str, str]] | None = None


def _load_dictionary() -> dict[str, dict[str, str]]:
    global _loaded_dictionary
    if _loaded_dictionary is None:
        with open(_DICT_PATH, encoding='utf-8') as f:
            _loaded_dictionary = json.load(f)
    assert _loaded_dictionary is not None
    return _loaded_dictionary


def _dictionary_entries(lang: str) -> list[tuple[str, str]]:
    data = _load_dictionary()
    lang_prefix = lang[:2] + '-*'
    combined: dict[str, str] = {}
    for key, entries in data.items():
        if key == lang_prefix:
            combined.update(entries)
        elif key == 'en-*' and lang_prefix == 'en-*':
            combined.update(entries)
    if not combined:
        combined.update(data.get('en-*', {}))
    return list(combined.items())


def _compiled_entries(lang: str) -> list[tuple[re.Pattern[str], str]]:
    if lang not in _compiled_dictionary:
        entries = _dictionary_entries(lang)
        entries.sort(key=lambda item: len(item[0]), reverse=True)
        _compiled_dictionary[lang] = [
            (re.compile(r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])'), replacement)
            for term, replacement in entries
        ]
    return _compiled_dictionary[lang]


def apply_dictionary(text: str, lang: str = 'en-CA') -> str:
    text = _DICT_MB_RE.sub(r'\1 millibars', text)
    for pattern, replacement in _compiled_entries(lang):
        text = pattern.sub(replacement, text)
    return text


def apply(text: str, lang: str = 'en-CA') -> str:
    return apply_dictionary(text, lang)


def generate_package(
    config: dict[str, Any],
    feed_id: str,
    package_type: str,
    weather_data: Optional[dict[str, Any]] = None,
) -> str:
    from module.packages import climate_summary_package, current_conditions_package, date_time_package, station_id

    feed = next((f for f in config.get('feeds', []) if f['id'] == feed_id), None)
    lang = feed.get('language', 'en-CA') if feed else 'en-CA'
    tz = feed.get('timezone', 'UTC') if feed else 'UTC'

    log.info('Generating TTS package for feed %s, type %s, language %s', feed_id, package_type, lang)

    if package_type == 'date_time':
        return date_time_package(tz, lang)
    if package_type == 'station_id':
        return station_id(config, feed_id, lang)
    if package_type == 'current_conditions':
        loc_name = None
        if feed:
            for block in feed.get('locations', []):
                if not isinstance(block, dict):
                    continue
                for entry in block.get('observationLocations', []):
                    if isinstance(entry, dict):
                        loc_name = entry.get('name_override') or entry.get('name')
                        break
                if loc_name:
                    break
        return current_conditions_package(weather_data, loc_name, lang)
    if package_type == 'climate_summary':
        loc_name = None
        if feed:
            for block in feed.get('locations', []):
                if not isinstance(block, dict):
                    continue
                for entry in block.get('climateLocations', []):
                    if isinstance(entry, dict):
                        loc_name = entry.get('name_override') or entry.get('name')
                        break
                if loc_name:
                    break
        return climate_summary_package(weather_data, loc_name, lang)
    return ''

available_providers = []

try:
    pyttsx3 = importlib.import_module('pyttsx3')
except ImportError:
    log.warning("pyttsx3 not available, pyttsx3 TTS provider will be disabled.")
    pyttsx3 = None

try:
    _piper_module = importlib.import_module('piper')
    PiperVoice = _piper_module.PiperVoice
    SynthesisConfig = _piper_module.SynthesisConfig
    available_providers.append('piper')
except ImportError:
    log.warning("Piper not available, Piper TTS provider will be disabled.")
    PiperVoice = None
    SynthesisConfig = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

KModel = None
KPipeline = None
hf_hub_download = None
kokoro_torch = None

_kokoro_runtime_checked = False
_hf_hub_download_checked = False

_rocm_libs_preloaded = False

def _preload_rocm_libs() -> bool:
    global _rocm_libs_preloaded
    if _rocm_libs_preloaded:
        return True
    try:
        import torch
        if torch.__file__ is None:
            log.debug('torch.__file__ is None; skipping ROCm lib preload')
            return False
        torch_lib_dir = pathlib.Path(torch.__file__).parent / 'lib'
    except ImportError:
        log.debug('torch not available; skipping ROCm lib preload')
        return False
    _ROCM_PRELOAD = [
        'libhipblas.so', 'libhipblaslt.so', 'libhipsolver.so',
        'librocblas.so', 'libMIOpen.so', 'librccl.so',
    ]
    loaded_any = False
    for name in _ROCM_PRELOAD:
        candidate = torch_lib_dir / name
        if not candidate.exists():
            continue
        try:
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            loaded_any = True
        except OSError as exc:
            log.debug('ROCm preload skipped %s: %s', name, exc)
    if ort is not None:
        capi_dir = pathlib.Path(ort.__file__).parent / 'capi'
        rocm_provider = capi_dir / 'libonnxruntime_providers_rocm.so'
        if rocm_provider.exists():
            try:
                ctypes.CDLL(str(rocm_provider), mode=ctypes.RTLD_GLOBAL)
                loaded_any = True
                log.debug('Preloaded libonnxruntime_providers_rocm.so')
            except OSError as exc:
                log.warning(
                    'Could not preload ROCm onnxruntime provider: %s — '
                    'ensure onnxruntime-rocm matches your ROCm/hipBLAS version. '
                    'onnxruntime-rocm on PyPI is built for ROCm 5.x (hipBLAS.so.2); '
                    'ROCm 6+ requires a build from AMD\'s repo or compiled from source.',
                    exc,
                )
    _rocm_libs_preloaded = True
    return loaded_any

from module.events import shutdown_event, tts_queue


_FILE_URI_RE = re.compile(r'^file://', re.IGNORECASE)
_VOICE_CACHE_MAX = 2
_voice_cache: OrderedDict[str, Any] = OrderedDict()
_kokoro_model_cache: OrderedDict[str, Any] = OrderedDict()
_kokoro_pipeline_cache: OrderedDict[str, Any] = OrderedDict()
_logged_piper_runtime: set[tuple[str, str]] = set()
_logged_kokoro_runtime: set[tuple[str, str]] = set()
_logged_kokoro_voice_fallbacks: set[tuple[str, str]] = set()

_TTS_AUDIO_FILTERS = (
    'acompressor=threshold=-20dB:ratio=4:attack=5:release=50:makeup=10dB,'
    'loudnorm=I=-9.0:LRA=7:TP=0.0'
)

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_PCM_EDGE_FADE_MS = 8


def smooth_pcm_edges(pcm_s16le: bytes, fade_ms: int = _PCM_EDGE_FADE_MS) -> bytes:
    if not pcm_s16le:
        return b''

    samples = np.frombuffer(pcm_s16le, dtype=np.int16)
    if samples.size == 0:
        return b''

    channel_count = max(int(CHANNELS), 1)
    frame_count = samples.size // channel_count
    if frame_count <= 1:
        return pcm_s16le

    usable_samples = frame_count * channel_count
    fade_frames = min(max(int(round(SAMPLE_RATE * fade_ms / 1000.0)), 0), frame_count // 2)
    if fade_frames == 0:
        return pcm_s16le

    shaped = samples[:usable_samples].astype(np.float32).reshape(frame_count, channel_count)
    fade_in = np.linspace(0.0, 1.0, fade_frames, endpoint=True, dtype=np.float32)[:, None]
    fade_out = np.linspace(1.0, 0.0, fade_frames, endpoint=True, dtype=np.float32)[:, None]
    shaped[:fade_frames] *= fade_in
    shaped[-fade_frames:] *= fade_out
    output = np.clip(shaped.reshape(-1), -32768, 32767).astype(np.int16).tobytes()
    if usable_samples == samples.size:
        return output
    return output + samples[usable_samples:].tobytes()


def _resample_pcm(pcm_s16le: bytes, from_sr: int) -> bytes:
    if not pcm_s16le or from_sr <= 0:
        return pcm_s16le
    if from_sr == SAMPLE_RATE:
        return pcm_s16le
    samples = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return b''
    target_len = int(round(len(samples) * SAMPLE_RATE / from_sr))
    resampled = np.interp(
        np.linspace(0, len(samples) - 1, target_len),
        np.arange(len(samples)),
        samples,
    ).astype(np.float32)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def _wav_to_pcm_s16le(wav_bytes: bytes) -> tuple[bytes, int]:
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, 'rb') as w:
        n_channels = w.getnchannels()
        src_sr = w.getframerate()
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16)
    if n_channels == 2 and CHANNELS == 1:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif n_channels == 1 and CHANNELS == 2:
        samples = np.column_stack([samples, samples]).ravel()
    return samples.tobytes(), src_sr


def _to_system_pcm(wav_bytes: bytes) -> bytes | None:
    try:
        raw, src_sr = _wav_to_pcm_s16le(wav_bytes)
        system_pcm = _resample_pcm(raw, src_sr)
        return _normalize_pcm(system_pcm) or smooth_pcm_edges(system_pcm)
    except Exception as e:
        log.error("WAV decode failed: %s", e)
        return None


def _decode_file_pcm(file_path: pathlib.Path) -> bytes | None:
    try:
        proc = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-i', str(file_path),
             '-af', _TTS_AUDIO_FILTERS,
             '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
             'pipe:1'],
            capture_output=True, check=True,
        )
        return smooth_pcm_edges(proc.stdout) if proc.stdout else None
    except Exception as e:
        log.warning('TTS normalization failed for %s, retrying without filters: %s', file_path.name, e)
        try:
            proc = subprocess.run(
                ['ffmpeg', '-loglevel', 'error', '-i', str(file_path),
                 '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
                 'pipe:1'],
                capture_output=True, check=True,
            )
            return smooth_pcm_edges(proc.stdout) if proc.stdout else None
        except Exception as exc:
            log.error("Failed to decode %s: %s", file_path.name, exc)
            return None


def decode_audio_file_pcm(file_path: pathlib.Path) -> bytes | None:
    return _decode_file_pcm(file_path)


def decode_audio_bytes_pcm(audio_bytes: bytes, suffix: str = '.bin') -> bytes | None:
    if not audio_bytes:
        return None
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    tmp_path = pathlib.Path(tmp_name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(audio_bytes)
        return _decode_file_pcm(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _normalize_pcm(pcm_s16le: bytes) -> bytes | None:
    if not pcm_s16le:
        return None
    try:
        proc = subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
             '-i', 'pipe:0',
             '-af', _TTS_AUDIO_FILTERS,
             '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
             'pipe:1'],
            input=pcm_s16le,
            capture_output=True,
            check=True,
        )
        return smooth_pcm_edges(proc.stdout) if proc.stdout else None
    except Exception:
        return smooth_pcm_edges(pcm_s16le)


def _transcode(src: pathlib.Path, dst: pathlib.Path) -> bool:
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src),
            '-af', _TTS_AUDIO_FILTERS,
            '-ar', str(SAMPLE_RATE),
            '-ac', str(CHANNELS),
            '-sample_fmt', 's16', str(dst)],
            capture_output=True,
            check=True,
        )
        log.info("Transcoded %s to %s", src.name, dst.name)
        return True
    except Exception as e:
        log.warning('TTS normalization failed for %s, retrying without filters: %s', src.name, e)
        try:
            subprocess.run(
                ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src),
                '-ar', str(SAMPLE_RATE),
                '-ac', str(CHANNELS),
                '-sample_fmt', 's16', str(dst)],
                capture_output=True,
                check=True,
            )
            log.info("Transcoded %s to %s", src.name, dst.name)
            return True
        except Exception as exc:
            log.error("Transcode failed %s: %s", src.name, exc)
            return False


def _unlink_with_retries(path: pathlib.Path, attempts: int = 8, delay: float = 0.1) -> None:
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt == attempts - 1:
                log.debug('Temp file cleanup failed for %s', path, exc_info=True)
                return
            time.sleep(delay)
        except OSError:
            log.debug('Temp file cleanup failed for %s', path, exc_info=True)
            return

def _normalize_piper_acceleration(value: Any) -> str:
    normalized = str(value or 'auto').strip().lower()
    aliases = {
        '': 'auto',
        'gpu': 'rocm',
        'amd': 'rocm',
        'none': 'cpu',
        'off': 'cpu',
        'false': 'cpu',
        'true': 'auto',
    }
    return aliases.get(normalized, normalized)


def _ensure_kokoro_cache_env() -> None:
    _KOKORO_ROOT.mkdir(parents=True, exist_ok=True)
    hub_cache = _KOKORO_HF_HOME / 'hub'
    transformers_cache = _KOKORO_HF_HOME / 'transformers'
    for cache_dir in (_KOKORO_HF_HOME, hub_cache, transformers_cache):
        cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(_KOKORO_HF_HOME)
    os.environ['HF_HUB_CACHE'] = str(hub_cache)
    os.environ['HUGGINGFACE_HUB_CACHE'] = str(hub_cache)
    os.environ['TRANSFORMERS_CACHE'] = str(transformers_cache)


def _load_hf_hub_download() -> Any:
    global hf_hub_download, _hf_hub_download_checked
    if _hf_hub_download_checked:
        return hf_hub_download

    _hf_hub_download_checked = True
    try:
        hf_hub_download = importlib.import_module('huggingface_hub').hf_hub_download
    except Exception as exc:
        log.warning('huggingface_hub not available, Kokoro asset download will be disabled: %s', exc)
        hf_hub_download = None
    return hf_hub_download


def _load_kokoro_runtime() -> tuple[Any, Any, Any]:
    global KModel, KPipeline, kokoro_torch, _kokoro_runtime_checked
    if _kokoro_runtime_checked:
        return KModel, KPipeline, kokoro_torch

    _kokoro_runtime_checked = True
    _ensure_kokoro_cache_env()
    if sys.version_info >= (3, 13):
        log.warning('Kokoro TTS is disabled on Python %s; upstream Kokoro dependencies currently require Python < 3.13', sys.version.split()[0])
        KModel = None
        KPipeline = None
        kokoro_torch = None
        return KModel, KPipeline, kokoro_torch
    try:
        kokoro_module = importlib.import_module('kokoro')
        KModel = kokoro_module.KModel
        KPipeline = kokoro_module.KPipeline
        kokoro_torch = importlib.import_module('torch')
        if 'kokoro' not in available_providers:
            available_providers.append('kokoro')
    except Exception as exc:
        log.warning('Kokoro not available, Kokoro TTS provider will be disabled: %s', exc)
        KModel = None
        KPipeline = None
        kokoro_torch = None
    return KModel, KPipeline, kokoro_torch


def _resolve_existing_path(path_value: str, roots: list[pathlib.Path]) -> str | None:
    candidate = pathlib.Path(path_value)
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.extend(root / path_value for root in roots)
        candidates.extend(root / candidate.name for root in roots)
    for item in candidates:
        if item.exists():
            return str(item)
    return None


def _select_reader(provider_name: str, lang: str, voice: str | None) -> dict[str, str] | None:
    readers = [reader for reader in _load_readers() if reader.get('provider') == provider_name]
    if not readers:
        return None

    normalized_lang = str(lang or '').strip().lower().replace('_', '-')
    lang_prefix = normalized_lang.split('-', 1)[0]
    desired_gender = str(voice or 'male').strip().lower()
    desired_gender = 'female' if desired_gender == 'female' else 'male'

    exact_lang = [reader for reader in readers if reader['language'] == normalized_lang]
    prefix_lang = [
        reader for reader in readers
        if reader['language'] and reader['language'].split('-', 1)[0] == lang_prefix
    ]
    unspecified_lang = [reader for reader in readers if not reader['language']]

    for group in (exact_lang, prefix_lang, unspecified_lang, readers):
        if not group:
            continue
        gender_match = [reader for reader in group if reader.get('gender') == desired_gender]
        return gender_match[0] if gender_match else group[0]

    return None


def _load_readers() -> list[dict[str, str]]:
    global _readers_cache
    if _readers_cache is not None:
        return _readers_cache

    readers: list[dict[str, str]] = []
    if not _READERS_PATH.exists():
        _readers_cache = readers
        return readers

    try:
        root = ET.parse(_READERS_PATH).getroot()
    except Exception as exc:
        log.warning('Failed to parse readers config %s: %s', _READERS_PATH, exc)
        _readers_cache = readers
        return readers

    for reader_el in root.findall('reader'):
        provider = str(reader_el.get('provider') or '').strip().lower()
        if provider not in {'piper', 'kokoro'}:
            continue
        language = str(reader_el.findtext('language') or '').strip().lower().replace('_', '-')
        gender = str(reader_el.findtext('gender') or 'male').strip().lower()
        path = str(reader_el.findtext('path') or '').strip()
        if not path:
            continue
        readers.append({
            'provider': provider,
            'language': language,
            'gender': 'female' if gender == 'female' else 'male',
            'path': path,
        })

    _readers_cache = readers
    return readers


def _resolve_reader_model(path_value: str) -> tuple[str, str | None]:
    base = pathlib.Path(path_value)
    if base.suffix.lower() == '.onnx':
        model_candidates = [base, pathlib.Path('voices') / base.name, pathlib.Path('piper-tts') / base.name]
    else:
        model_candidates = [
            base,
            pathlib.Path(f'{path_value}.onnx'),
            pathlib.Path('voices') / f'{path_value}.onnx',
            pathlib.Path('piper-tts') / f'{path_value}.onnx',
        ]

    model_path = ''
    for candidate in model_candidates:
        if candidate.exists():
            model_path = str(candidate)
            break

    if not model_path:
        fallback = pathlib.Path('voices') / f'{path_value}.onnx'
        model_path = str(fallback)

    model_file = pathlib.Path(model_path)
    cfg_candidates = [
        pathlib.Path(f'{model_path}.json'),
        model_file.with_suffix('.onnx.json'),
        model_file.with_suffix('.json'),
    ]
    cfg_path: str | None = None
    for cfg in cfg_candidates:
        if cfg.exists():
            cfg_path = str(cfg)
            break

    return model_path, cfg_path


def _reader_piper_cfg(lang: str, voice: str | None) -> dict[str, Any] | None:
    selected = _select_reader('piper', lang, voice)
    if not selected:
        return None
    model_path, config_path = _resolve_reader_model(str(selected['path']))
    if not pathlib.Path(model_path).exists():
        return None
    result: dict[str, Any] = {'model': model_path, 'speaker': 0}
    if config_path:
        result['config'] = config_path
    return result


def _normalize_kokoro_lang_code(value: Any) -> str | None:
    normalized = str(value or '').strip().lower().replace('_', '-')
    if not normalized:
        return None
    return _KOKORO_LANG_ALIASES.get(normalized)


def _infer_kokoro_voice_lang_code(voice_value: Any) -> str | None:
    voice_name = pathlib.Path(str(voice_value or '').strip()).stem.lower()
    if not voice_name:
        return None
    prefix = voice_name.split('_', 1)[0]
    if len(prefix) == 2 and prefix[0] in _KOKORO_DEFAULT_VOICES and prefix[1] in {'f', 'm'}:
        return prefix[0]
    return _normalize_kokoro_lang_code(voice_name)


def _default_kokoro_voice(lang_code: Any, voice_type: Any) -> str:
    resolved_lang = _normalize_kokoro_lang_code(lang_code) or 'a'
    voice_map = _KOKORO_DEFAULT_VOICES.get(resolved_lang) or _KOKORO_DEFAULT_VOICES['a']
    slot = 'female' if str(voice_type or 'male').strip().lower() == 'female' else 'male'
    return voice_map.get(slot) or next(iter(voice_map.values()))


def _resolve_kokoro_lang_code(language: Any, voice_value: Any, configured: Any) -> str:
    return (
        _normalize_kokoro_lang_code(configured)
        or _infer_kokoro_voice_lang_code(voice_value)
        or _normalize_kokoro_lang_code(language)
        or 'a'
    )


def _reader_kokoro_cfg(lang: str, voice: str | None) -> dict[str, Any] | None:
    selected = _select_reader('kokoro', lang, voice)
    if not selected:
        return None
    voice_value = str(selected.get('path') or '').strip()
    if not voice_value:
        return None
    return {
        'voice': voice_value,
        'lang_code': _resolve_kokoro_lang_code(lang, voice_value, None),
    }


def _resolve_kokoro_support_path(path_value: Any, default_filename: str) -> str:
    normalized = str(path_value or '').strip()
    if normalized:
        resolved = _resolve_existing_path(normalized, [_KOKORO_ROOT, _VOICES_PATH])
        if resolved:
            return resolved
        if pathlib.Path(normalized).is_absolute():
            raise FileNotFoundError(f'Kokoro asset not found: {normalized}')
        return _ensure_kokoro_asset(pathlib.Path(normalized).as_posix())
    return _ensure_kokoro_asset(default_filename)


def _ensure_kokoro_asset(relative_path: str) -> str:
    downloader = _load_hf_hub_download()
    if downloader is None:
        raise RuntimeError('huggingface_hub not available')

    _ensure_kokoro_cache_env()
    local_path = _KOKORO_ROOT / pathlib.Path(relative_path)
    if local_path.exists():
        return str(local_path)

    downloaded = downloader(
        repo_id=_KOKORO_REPO_ID,
        filename=relative_path.replace('\\', '/'),
        cache_dir=_KOKORO_HF_HOME / 'hub',
        local_dir=_KOKORO_ROOT,
    )
    return str(pathlib.Path(downloaded))


def _resolve_kokoro_voice_path(voice_value: Any, lang_code: str, voice_type: str) -> tuple[str, str]:
    requested_value = str(voice_value or '').strip()
    requested_name = pathlib.Path(requested_value).stem if requested_value else ''

    if requested_name:
        requested_path = requested_value if pathlib.Path(requested_value).suffix.lower() == '.pt' else f'{requested_name}.pt'
        resolved = _resolve_existing_path(requested_path, [_KOKORO_ROOT / 'voices', _KOKORO_ROOT, _VOICES_PATH])
        if resolved:
            return resolved, requested_name
        try:
            return _ensure_kokoro_asset(f'voices/{requested_name}.pt'), requested_name
        except Exception:
            fallback_name = _default_kokoro_voice(lang_code, voice_type)
            fallback_key = (requested_name, fallback_name)
            if fallback_key not in _logged_kokoro_voice_fallbacks:
                _logged_kokoro_voice_fallbacks.add(fallback_key)
                log.warning('Kokoro voice %s unavailable, using %s instead', requested_name, fallback_name)
            return _ensure_kokoro_asset(f'voices/{fallback_name}.pt'), fallback_name

    fallback_name = _default_kokoro_voice(lang_code, voice_type)
    return _ensure_kokoro_asset(f'voices/{fallback_name}.pt'), fallback_name


def _resolve_kokoro_device(v_cfg: dict[str, Any]) -> str:
    _, _, torch_module = _load_kokoro_runtime()
    requested = str(v_cfg.get('device') or 'auto').strip().lower()
    aliases = {
        '': 'auto',
        'gpu': 'cuda',
        'none': 'cpu',
    }
    requested = aliases.get(requested, requested)
    has_cuda = bool(torch_module is not None and torch_module.cuda.is_available())

    if requested == 'cuda':
        device = 'cuda' if has_cuda else 'cpu'
    elif requested == 'auto':
        device = 'cuda' if has_cuda else 'cpu'
    else:
        device = 'cpu'

    log_key = (requested, device)
    if log_key not in _logged_kokoro_runtime:
        _logged_kokoro_runtime.add(log_key)
        log.info('Kokoro runtime selected: requested=%s device=%s', requested, device)
    return device


def _coerce_kokoro_cfg(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        return {'voice': value.strip()}
    return {}


def _resolve_kokoro_config(ctx: dict) -> dict[str, Any] | None:
    tts_cfg = ctx.get('config', {}).get('tts', {})
    global_cfg = dict(tts_cfg.get('kokoro', {})) if isinstance(tts_cfg.get('kokoro'), dict) else {}
    lang = str(ctx.get('lang') or 'en-CA')
    voice_type = ctx.get('voice') if ctx.get('voice') in ('male', 'female') else 'male'

    lang_backend = tts_cfg.get('lang', {}).get(lang, {}).get('backend', {}).get('kokoro', {})
    selected_cfg = {}
    if isinstance(lang_backend, dict):
        selected_cfg = _coerce_kokoro_cfg(lang_backend.get(voice_type))
        if not selected_cfg:
            selected_cfg = _coerce_kokoro_cfg(lang_backend.get('voice'))
        if not selected_cfg and voice_type != 'male':
            selected_cfg = _coerce_kokoro_cfg(lang_backend.get('male'))
        if not selected_cfg and voice_type != 'female':
            selected_cfg = _coerce_kokoro_cfg(lang_backend.get('female'))

    if not selected_cfg:
        reader_cfg = _reader_kokoro_cfg(lang, voice_type)
        if reader_cfg:
            selected_cfg = dict(reader_cfg)

    merged = dict(global_cfg)
    merged.update(selected_cfg)

    lang_code = _resolve_kokoro_lang_code(lang, merged.get('voice'), merged.get('lang_code'))
    voice_name = str(merged.get('voice') or '').strip()
    if not voice_name:
        voice_name = _default_kokoro_voice(lang_code, voice_type)

    voice_path, resolved_voice = _resolve_kokoro_voice_path(voice_name, lang_code, voice_type)
    merged['voice'] = resolved_voice
    merged['voice_path'] = voice_path
    merged['lang_code'] = lang_code
    merged['model'] = _resolve_kokoro_support_path(merged.get('model'), _KOKORO_MODEL_FILE)
    merged['config'] = _resolve_kokoro_support_path(merged.get('config'), _KOKORO_CONFIG_FILE)
    try:
        merged['speed'] = float(merged.get('speed') or 1.0)
    except (TypeError, ValueError):
        merged['speed'] = 1.0
    merged['device'] = _resolve_kokoro_device(merged)
    return merged


def _load_kokoro_model(v_cfg: dict[str, Any]) -> Any:
    model_class, _, _ = _load_kokoro_runtime()
    if model_class is None:
        raise RuntimeError('Kokoro not available')

    cache_key = f"{v_cfg['device']}:{v_cfg['config']}:{v_cfg['model']}"
    if cache_key in _kokoro_model_cache:
        _kokoro_model_cache.move_to_end(cache_key)
        return _kokoro_model_cache[cache_key]

    loaded = cast(Any, model_class)(config=v_cfg['config'], model=v_cfg['model']).to(v_cfg['device']).eval()
    _kokoro_model_cache[cache_key] = loaded
    if len(_kokoro_model_cache) > _VOICE_CACHE_MAX:
        _kokoro_model_cache.popitem(last=False)
    return loaded


def _load_kokoro_pipeline(v_cfg: dict[str, Any]) -> Any:
    _, pipeline_class, _ = _load_kokoro_runtime()
    if pipeline_class is None:
        raise RuntimeError('Kokoro not available')

    cache_key = str(v_cfg.get('lang_code') or 'a')
    if cache_key in _kokoro_pipeline_cache:
        _kokoro_pipeline_cache.move_to_end(cache_key)
        return _kokoro_pipeline_cache[cache_key]

    loaded = cast(Any, pipeline_class)(lang_code=cache_key, model=False)
    _kokoro_pipeline_cache[cache_key] = loaded
    if len(_kokoro_pipeline_cache) > _VOICE_CACHE_MAX:
        _kokoro_pipeline_cache.popitem(last=False)
    return loaded


def _resolve_piper_runtime(config: dict[str, Any]) -> tuple[bool, str]:
    requested = _normalize_piper_acceleration(config.get('tts', {}).get('piper', {}).get('acceleration'))
    available = list(ort.get_available_providers()) if ort is not None else []

    has_cuda = 'CUDAExecutionProvider' in available
    has_rocm = 'ROCMExecutionProvider' in available

    if requested == 'cuda':
        use_cuda = has_cuda
        provider = 'CUDAExecutionProvider' if has_cuda else 'CPUExecutionProvider'
    elif requested == 'rocm':
        use_cuda = False
        provider = 'ROCMExecutionProvider'
    elif requested == 'auto':
        if has_cuda:
            use_cuda = True
            provider = 'CUDAExecutionProvider'
        elif has_rocm:
            use_cuda = False
            provider = 'ROCMExecutionProvider'
        else:
            use_cuda = False
            provider = 'CPUExecutionProvider'
    else:
        use_cuda = False
        provider = 'CPUExecutionProvider'

    if requested == 'cuda' and not has_cuda:
        key = ('cuda-unavailable', ','.join(available) or 'none')
        if key not in _logged_piper_runtime:
            _logged_piper_runtime.add(key)
            log.warning(
                'Piper CUDA acceleration requested but unavailable; using CPUExecutionProvider (available providers: %s)',
                ', '.join(available) or 'none',
            )

    log_key = (requested, provider)
    if log_key not in _logged_piper_runtime:
        _logged_piper_runtime.add(log_key)
        log.info(
            'Piper runtime selected: requested=%s provider=%s available=%s',
            requested,
            provider,
            ', '.join(available) or 'none',
        )

    return use_cuda, provider


def _load_piper_voice(
    model_path: str,
    config: dict[str, Any],
    config_path: Optional[str] = None,
) -> Any:
    use_cuda, provider = _resolve_piper_runtime(config)
    cache_key = f'{provider}:{model_path}'
    if cache_key in _voice_cache:
        _voice_cache.move_to_end(cache_key)
        return _voice_cache[cache_key]
    if PiperVoice is None:
        raise RuntimeError('Piper not available')
    loaded = cast(Any, PiperVoice).load(model_path, config_path=config_path, use_cuda=use_cuda)
    if provider == 'ROCMExecutionProvider' and ort is not None:
        _preload_rocm_libs()
        try:
            loaded.session = ort.InferenceSession(
                model_path,
                sess_options=ort.SessionOptions(),
                providers=['ROCMExecutionProvider', 'CPUExecutionProvider'],
            )
            actual = loaded.session.get_providers()
            if 'ROCMExecutionProvider' in actual:
                log.info('Piper ROCm session active (providers: %s)', ', '.join(actual))
            else:
                log.warning(
                    'ROCMExecutionProvider not active after session creation (got: %s); '
                    'check that onnxruntime-rocm is installed and torch ROCm libs are accessible',
                    ', '.join(actual),
                )
        except Exception as exc:
            log.warning('Piper ROCm session failed (%s); falling back to CPU', exc)
    _voice_cache[cache_key] = loaded
    if len(_voice_cache) > _VOICE_CACHE_MAX:
        _voice_cache.popitem(last=False)
    return loaded


def _resolve_piper_config(ctx: dict) -> dict | None:
    tts_cfg = ctx['config'].get('tts', {})
    lang_map = tts_cfg.get('lang', {}).get(ctx['lang'], {}).get('backend', {}).get('piper', {})
    voice_type = ctx['voice'] if ctx['voice'] in ('male', 'female') else 'male'
    v_cfg = lang_map.get(voice_type)
    if not isinstance(v_cfg, dict):
        v_cfg = lang_map.get('male') if isinstance(lang_map.get('male'), dict) else lang_map.get('female')
    if isinstance(v_cfg, dict) and v_cfg.get('model'):
        return v_cfg
    return _reader_piper_cfg(str(ctx.get('lang') or 'en-CA'), ctx.get('voice'))


def _kokoro_audio_to_pcm(audio: Any) -> bytes | None:
    if audio is None:
        return None

    if hasattr(audio, 'detach'):
        samples = np.asarray(audio.detach().cpu().numpy(), dtype=np.float32)
    else:
        samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        return None

    flattened = np.clip(samples.reshape(-1), -1.0, 1.0)
    pcm = (flattened * 32767.0).astype(np.int16)
    if CHANNELS == 2:
        pcm = np.column_stack([pcm, pcm]).ravel().astype(np.int16)
    return _resample_pcm(pcm.tobytes(), _KOKORO_SAMPLE_RATE)


def _piper_spec(ctx: dict, v_cfg: dict) -> Any:
    global_cfg = ctx['config'].get('tts', {}).get('piper', {})
    if SynthesisConfig is None:
        raise RuntimeError('Piper not available')
    return cast(Any, SynthesisConfig)(
        speaker_id=int(v_cfg.get('speaker', global_cfg.get('speaker', 0))),
    )


def _piper_synthesize_wav_bytes(text: str, ctx: dict) -> bytes | None:
    v_cfg = _resolve_piper_config(ctx)
    if not v_cfg:
        return None
    try:
        voice = _load_piper_voice(v_cfg['model'], ctx['config'], v_cfg.get('config'))
        spec = _piper_spec(ctx, v_cfg)
        chunks = list(cast(Any, voice).synthesize(text, syn_config=spec))
        if not chunks:
            return None
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            first = chunks[0]
            wf.setframerate(first.sample_rate)
            wf.setsampwidth(first.sample_width)
            wf.setnchannels(first.sample_channels)
            for chunk in chunks:
                wf.writeframes(chunk.audio_int16_bytes)
        return buf.getvalue()
    except Exception as e:
        log.error("Piper synthesis failed: %s", e)
        return None


def _chunk_to_system_pcm(audio_chunk: Any) -> bytes:
    raw = bytes(audio_chunk.audio_int16_bytes)
    samples = np.frombuffer(raw, dtype=np.int16)
    sample_channels = int(getattr(audio_chunk, 'sample_channels', 1) or 1)
    sample_rate = int(getattr(audio_chunk, 'sample_rate', SAMPLE_RATE) or SAMPLE_RATE)
    if sample_channels == 2 and CHANNELS == 1:
        samples = samples.reshape(-1, 2).mean(axis=1).astype(np.int16)
    elif sample_channels == 1 and CHANNELS == 2:
        samples = np.column_stack([samples, samples]).ravel().astype(np.int16)
    return _resample_pcm(samples.tobytes(), sample_rate)


def _piper_stream_pcm(text: str, ctx: dict) -> Generator[bytes, None, None]:
    """Yield normalized s16le PCM for one synthesized text block."""
    v_cfg = _resolve_piper_config(ctx)
    if not v_cfg:
        return
    try:
        voice = _load_piper_voice(v_cfg['model'], ctx['config'], v_cfg.get('config'))
        spec = _piper_spec(ctx, v_cfg)
        chunks: list[bytes] = []
        for audio_chunk in cast(Any, voice).synthesize(text, syn_config=spec):
            pcm = _chunk_to_system_pcm(audio_chunk)
            if pcm:
                chunks.append(pcm)
        normalized = _normalize_pcm(b''.join(chunks))
        if normalized:
            yield normalized
    except Exception as e:
        log.error("Piper stream failed: %s", e)


def _provide_piper_pcm(text: str, ctx: dict) -> bytes | None:
    chunks = list(_piper_stream_pcm(text, ctx))
    return b''.join(chunks) if chunks else None


def _resolve_pyttsx3_voice_value(ctx: dict) -> str | None:
    tts_cfg = ctx.get('config', {}).get('tts', {})
    lang = ctx.get('lang')

    lang_backend = (
        tts_cfg.get('lang', {}).get(lang, {})
        .get('backend', {}).get('pyttsx3', {})
    ) if lang is not None else {}

    voice_type = ctx.get('voice') if ctx.get('voice') in ('male', 'female') else 'male'
    if isinstance(lang_backend, dict):
        v_cfg = lang_backend.get(voice_type)
        if isinstance(v_cfg, dict) and v_cfg.get('voice'):
            return v_cfg.get('voice')

        if isinstance(lang_backend.get('voice'), str):
            return lang_backend.get('voice')

    global_pyttsx3 = tts_cfg.get('pyttsx3', {})
    if isinstance(global_pyttsx3, dict) and global_pyttsx3.get('voice'):
        return global_pyttsx3.get('voice')
    return None


def _select_and_set_pyttsx3_voice(engine: Any, desired: str) -> str | None:
    if not desired:
        return None
    try:
        engine.setProperty('voice', desired)
        return desired
    except Exception:
        pass

    voices: list[dict[str, Any]] = []
    try:
        voices = [
            {
                'id': str(getattr(v, 'id', '') or ''),
                'name': str(getattr(v, 'name', '') or ''),
                'lang': _normalize_voice_languages(getattr(v, 'languages', [])),
            }
            for v in (engine.getProperty('voices') or [])
        ]
    except Exception:
        if os.name == 'nt':
            voices = _get_windows_sapi5_voices(log_skipped=False)
        else:
            return None

    desired_norm = str(desired).strip().lower()

    for voice in voices:
        voice_id = str(voice.get('id') or '')
        if voice_id and voice_id == desired:
            try:
                engine.setProperty('voice', voice_id)
                return voice_id
            except Exception:
                pass

    for voice in voices:
        name = str(voice.get('name') or '')
        if name and name.lower() == desired_norm:
            try:
                engine.setProperty('voice', str(voice.get('id') or name))
                return str(voice.get('id') or name)
            except Exception:
                pass

    for voice in voices:
        name = str(voice.get('name') or '')
        if name and desired_norm in name.lower():
            try:
                engine.setProperty('voice', str(voice.get('id') or name))
                return str(voice.get('id') or name)
            except Exception:
                pass

    for voice in voices:
        for token in voice.get('lang') or []:
            token_text = str(token).lower()
            if desired_norm in token_text or desired_norm == token_text:
                try:
                    resolved = str(voice.get('id') or voice.get('name') or '')
                    engine.setProperty('voice', resolved)
                    return resolved
                except Exception:
                    pass

    return None


def _provide_pyttsx3_pcm(text: str, ctx: dict) -> bytes | None:
    if pyttsx3 is None:
        return None
    cfg = ctx['config'].get('tts', {}).get('pyttsx3', {})
    if not cfg.get('enabled'):
        return None
    engine = None
    tmp_file = None
    try:
        engine = pyttsx3.init()

        try:
            desired = _resolve_pyttsx3_voice_value(ctx)
            if desired:
                _select_and_set_pyttsx3_voice(engine, desired)

            if cfg.get('voice'):
                _select_and_set_pyttsx3_voice(engine, cfg.get('voice'))
        except Exception:
            pass
        if cfg.get('rate'):
            engine.setProperty('rate', cfg['rate'])
        if cfg.get('volume'):
            engine.setProperty('volume', cfg['volume'])
        fd, tmp_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        tmp_file = pathlib.Path(tmp_path)
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        pcm = _decode_file_pcm(tmp_file)
        return pcm
    except Exception as e:
        log.error("Pyttsx3 PCM provider failed: %s", e)
        return None
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        if tmp_file is not None:
            _unlink_with_retries(tmp_file)


def _provide_kokoro_pcm(text: str, ctx: dict) -> bytes | None:
    try:
        v_cfg = _resolve_kokoro_config(ctx)
        if not v_cfg:
            return None

        model = _load_kokoro_model(v_cfg)
        pipeline = _load_kokoro_pipeline(v_cfg)
        chunks = []
        for result in cast(Any, pipeline)(text, voice=v_cfg['voice_path'], speed=v_cfg['speed'], model=model):
            pcm = _kokoro_audio_to_pcm(getattr(result, 'audio', None))
            if pcm:
                chunks.append(pcm)

        if not chunks:
            return None

        combined = b''.join(chunks)
        return _normalize_pcm(combined) or smooth_pcm_edges(combined)
    except Exception as e:
        log.error('Kokoro PCM provider failed: %s', e)
        return None


PCM_PROVIDERS: dict[str, Any] = {
    'kokoro': _provide_kokoro_pcm,
    'piper': _provide_piper_pcm,
    'pyttsx3': _provide_pyttsx3_pcm,
}

_PCM_STREAM_PROVIDERS: dict[str, Any] = {
    'piper': _piper_stream_pcm,
}


def synthesize_pcm(
    config: dict[str, Any],
    text: str,
    lang: str = 'en-CA',
    voice: Optional[str] = None,
) -> bytes | None:
    return b''.join(synthesize_pcm_stream(config, text, lang, voice)) or None


def synthesize_pcm_stream(
    config: dict[str, Any],
    text: str,
    lang: str = 'en-CA',
    voice: Optional[str] = None,
) -> Generator[bytes, None, None]:
    """Yield normalized s16le PCM chunks, one per synthesized line.
    File URI lines are decoded and normalized whole after ffmpeg processing."""
    ctx = {'config': config, 'lang': lang, 'voice': voice}
    fallback_order = config.get('tts', {}).get('fallback_order', ['piper'])

    for line in text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        if _FILE_URI_RE.match(line):
            file_path = pathlib.Path(unquote(urlparse(line).path))
            pcm = _decode_file_pcm(file_path)
            if pcm:
                yield pcm
        else:
            clean = apply_dictionary(line, lang)
            for provider_name in fallback_order:
                stream_fn = _PCM_STREAM_PROVIDERS.get(provider_name)
                if stream_fn:
                    yielded = False
                    for chunk in stream_fn(clean, ctx):
                        yield chunk
                        yielded = True
                    if yielded:
                        break
                else:
                    batch_fn = PCM_PROVIDERS.get(provider_name)
                    if batch_fn:
                        pcm = batch_fn(clean, ctx)
                        if pcm:
                            yield pcm
                            break


def _provide_piper(text: str, ctx: dict, out: pathlib.Path) -> bool:
    wav_bytes = _piper_synthesize_wav_bytes(text, ctx)
    if not wav_bytes:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(wav_bytes)
    raw_path = out.with_suffix('.bin.wav')
    out.rename(raw_path)
    success = _transcode(raw_path, out)
    raw_path.unlink(missing_ok=True)
    return success


def _write_pcm_wav(path: pathlib.Path, pcm_s16le: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(BYTES_PER_SAMPLE)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_s16le)


def write_pcm_wav(path: pathlib.Path, pcm_s16le: bytes) -> None:
    _write_pcm_wav(path, pcm_s16le)


def _normalize_voice_languages(raw_languages: Any) -> list[str]:
    if raw_languages is None:
        return []
    if isinstance(raw_languages, (list, tuple, set)):
        items = list(raw_languages)
    else:
        items = [raw_languages]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, bytes):
            text = item.decode('utf-8', errors='ignore')
        else:
            text = str(item)
        text = text.strip().strip('\x05')
        if not text:
            continue
        lowered = text.lower().replace('_', '-')
        if lowered not in seen:
            seen.add(lowered)
            normalized.append(lowered)
    return normalized


def _windows_locale_list(language_attr: Any) -> list[str]:
    raw = str(language_attr or '').strip()
    if not raw:
        return []

    locales: list[str] = []
    seen: set[str] = set()
    for token in raw.split(';'):
        token = token.strip()
        if not token:
            continue
        try:
            locale_id = int(token, 16)
        except ValueError:
            try:
                locale_id = int(token)
            except ValueError:
                normalized = token.lower().replace('_', '-')
            else:
                normalized = locale.windows_locale.get(locale_id, token).replace('_', '-')
        else:
            normalized = locale.windows_locale.get(locale_id, token).replace('_', '-')
        if normalized not in seen:
            seen.add(normalized)
            locales.append(normalized)
    return locales


def _windows_sapi5_voice_from_token(token: Any, log_skipped: bool = True) -> dict[str, Any] | None:
    try:
        voice_id = str(token.Id)
    except Exception as e:
        if log_skipped:
            log.warning('Skipping Windows SAPI voice with unreadable id: %s', e)
        return None

    try:
        name = str(token.GetDescription())
    except Exception as e:
        if log_skipped:
            log.warning('Skipping malformed Windows SAPI voice token %s: %s', voice_id, e)
        return None

    try:
        lang = _windows_locale_list(token.GetAttribute('Language'))
    except Exception:
        lang = []

    return {
        'id': voice_id,
        'name': name,
        'lang': lang,
    }


def _get_windows_sapi5_voices(log_skipped: bool = True) -> list[dict[str, Any]]:
    try:
        comtypes_client = importlib.import_module('comtypes.client')
        speech = comtypes_client.CreateObject('SAPI.SPVoice')
        tokens = speech.GetVoices()
        voices = []
        for idx in range(int(tokens.Count)):
            try:
                token = tokens.Item(idx)
            except Exception as e:
                if log_skipped:
                    log.warning('Skipping Windows SAPI voice token at index %d: %s', idx, e)
                continue
            voice = _windows_sapi5_voice_from_token(token, log_skipped=log_skipped)
            if voice is not None:
                voices.append(voice)
        return voices
    except Exception as e:
        log.error('Failed to enumerate Windows SAPI voices directly: %s', e)
        return []


def get_available_pyttsx3_voices() -> list[dict[str, Any]]:
    if pyttsx3 is None:
        return []
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        return [
            {
                'id': str(v.id),
                'name': str(v.name),
                'lang': _normalize_voice_languages(getattr(v, 'languages', [])),
            }
            for v in voices
        ]
    except Exception as e:
        if os.name == 'nt':
            log.warning('Failed to get pyttsx3 voices, falling back to direct Windows SAPI enumeration: %s', e)
            return _get_windows_sapi5_voices()
        log.error('Failed to get pyttsx3 voices: %s', e)
        return []

def _provide_pyttsx3(text: str, ctx: dict, out: pathlib.Path) -> bool:
    if pyttsx3 is None: return False
    cfg = ctx['config'].get('tts', {}).get('pyttsx3', {})
    if not cfg.get('enabled'): return False

    engine = None
    raw_tmp = out.with_suffix('.bin.wav')
    try:
        engine = pyttsx3.init()

        try:
            desired = _resolve_pyttsx3_voice_value(ctx)
            if desired:
                _select_and_set_pyttsx3_voice(engine, desired)
            if cfg.get('voice'):
                _select_and_set_pyttsx3_voice(engine, cfg.get('voice'))
        except Exception:
            pass
        if cfg.get('rate'): engine.setProperty('rate', cfg['rate'])
        if cfg.get('volume'): engine.setProperty('volume', cfg['volume'])

        _unlink_with_retries(raw_tmp)
        engine.save_to_file(text, str(raw_tmp))
        engine.runAndWait()

        success = _transcode(raw_tmp, out)
        return success
    except Exception as e:
        log.error("Pyttsx3 provider failed: %s", e)
        return False
    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass
        _unlink_with_retries(raw_tmp)


def _provide_kokoro(text: str, ctx: dict, out: pathlib.Path) -> bool:
    pcm = _provide_kokoro_pcm(text, ctx)
    if not pcm:
        return False
    try:
        _write_pcm_wav(out, pcm)
        return True
    except Exception as e:
        log.error('Kokoro provider failed: %s', e)
        return False

PROVIDERS = {
    'kokoro': _provide_kokoro,
    'piper': _provide_piper,
    'pyttsx3': _provide_pyttsx3
}

def synthesize(
    config: dict[str, Any],
    text: str,
    feed_id: str,
    package_id: str,
    lang: str = 'en-CA',
    voice: Optional[str] = None,
) -> Optional[pathlib.Path]:

    out_dir = pathlib.Path("audio") / feed_id / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{package_id}.wav"
    
    segments = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line: continue
        kind = 'file' if _FILE_URI_RE.match(line) else 'text'
        segments.append((kind, line))

    if not segments: return None

    part_dir = out_dir / f".{package_id}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    processed_parts = []

    ctx = {'config': config, 'lang': lang, 'voice': voice}
    fallback_order = config.get('tts', {}).get('fallback_order', ['piper'])

    try:
        for i, (kind, content) in enumerate(segments):
            part_path = part_dir / f"{i}.wav"
            
            if kind == 'file':
                file_path = pathlib.Path(unquote(urlparse(content).path))
                if _transcode(file_path, part_path):
                    processed_parts.append(part_path)
            else:

                clean_text = apply_dictionary(content, lang)
                for provider_name in fallback_order:
                    if provider_name in PROVIDERS and PROVIDERS[provider_name](clean_text, ctx, part_path):
                        processed_parts.append(part_path)
                        break

        if not processed_parts: return None

        if len(processed_parts) == 1:
            _transcode(processed_parts[0], final_path)
        else:
            inputs = []
            filter_str = ""
            for i, p in enumerate(processed_parts):
                inputs.extend(['-i', str(p)])

                filter_str += f"[{i}:a]aresample={SAMPLE_RATE},pan={'stereo|c0=c0|c1=c0' if CHANNELS==2 else 'mono|c0=c0'}[a{i}];"
            
            filter_str += "".join([f"[a{i}]" for i in range(len(processed_parts))])
            filter_str += f"concat=n={len(processed_parts)}:v=0:a=1,{_TTS_AUDIO_FILTERS}[outa]"
            
            cmd = ['ffmpeg', '-y', '-loglevel', 'error'] + inputs + [
                '-filter_complex', filter_str,
                '-map', '[outa]',
                '-ac', str(CHANNELS),
                '-ar', str(SAMPLE_RATE),
                '-sample_fmt', 's16',
                str(final_path)
            ]
            proc = subprocess.run(cmd, capture_output=True)

            if proc.returncode != 0 or not final_path.exists():
                log.error("FFmpeg concat failed: %s", proc.stderr.decode(errors="ignore"))
                return None

    finally:
        shutil.rmtree(part_dir, ignore_errors=True)

    return final_path if final_path.exists() else None

def tts_thread_worker(config: dict[str, Any]) -> None:
    while not shutdown_event.is_set():
        try:
            item = tts_queue.get(timeout=1)
            if item is None: break
            
            feed_id, pkg_id, text, lang, *rest = item
            voice = rest[0] if rest else None
            
            log.info('[%s/%s] Synthesizing: %s', feed_id, lang, pkg_id)
            if synthesize(config, text, feed_id, pkg_id, lang, voice):
                log.debug('Completed %s', pkg_id)
                
            tts_queue.task_done()
        except queue.Empty:
            continue
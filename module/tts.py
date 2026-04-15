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
import tempfile
import time
import wave
from collections import OrderedDict
from typing import Any, Generator, Optional, Protocol, cast
from urllib.parse import unquote, urlparse

import numpy as np
import yaml

log = logging.getLogger(__name__)

def load_config(config_path: str | None = None) -> dict[str, Any]:
    selected_path = config_path or os.environ.get('CONFIG_PATH')
    if selected_path:
        config_file = pathlib.Path(selected_path)
    else:
        config_file = pathlib.Path(__file__).parent.parent / 'config.yaml'
    with open(config_file, encoding='utf-8') as f:
        return yaml.safe_load(f)


_MODULE_CONFIG = load_config()
SAMPLE_RATE = _MODULE_CONFIG.get('playout', {}).get('sample_rate', 16000)
CHANNELS = _MODULE_CONFIG.get('playout', {}).get('channels', 1)
BYTES_PER_SAMPLE = 2

_DICT_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'dictionary.json'
_DICT_MB_RE = re.compile(r'(\d+)\s+MB\b')
_loaded_dictionary: dict[str, dict[str, str]] | None = None
_compiled_dictionary: dict[str, list[tuple[re.Pattern[str], str]]] = {}


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
    from managed.packages import climate_summary_package, current_conditions_package, date_time_package, station_id

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

_rocm_libs_preloaded = False

def _preload_rocm_libs() -> bool:
    global _rocm_libs_preloaded
    if _rocm_libs_preloaded:
        return True
    try:
        import torch
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

from managed.events import shutdown_event, tts_queue


_FILE_URI_RE = re.compile(r'^file://', re.IGNORECASE)
_VOICE_CACHE_MAX = 2
_voice_cache: OrderedDict[str, Any] = OrderedDict()
_logged_piper_runtime: set[tuple[str, str]] = set()

_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _resample_pcm(pcm_s16le: bytes, from_sr: int) -> bytes:
    if from_sr == SAMPLE_RATE:
        return pcm_s16le
    samples = np.frombuffer(pcm_s16le, dtype=np.int16).astype(np.float32)
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
        return _resample_pcm(raw, src_sr)
    except Exception as e:
        log.error("WAV decode failed: %s", e)
        return None


def _decode_file_pcm(file_path: pathlib.Path) -> bytes | None:
    try:
        proc = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-i', str(file_path),
             '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
             'pipe:1'],
            capture_output=True, check=True,
        )
        return proc.stdout if proc.stdout else None
    except Exception as e:
        log.error("Failed to decode %s: %s", file_path.name, e)
        return None


def _transcode(src: pathlib.Path, dst: pathlib.Path) -> bool:
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(src),
             '-ar', str(SAMPLE_RATE),
             '-ac', str(CHANNELS),
             '-sample_fmt', 's16', str(dst)],
            check=True,
        )
        log.info("Transcoded %s to %s", src.name, dst.name)
        return True
    except Exception as e:
        log.error("Transcode failed %s: %s", src.name, e)
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
    if not isinstance(v_cfg, dict) or not v_cfg.get('model'):
        return None
    return v_cfg


def _piper_spec(ctx: dict, v_cfg: dict) -> Any:
    global_cfg = ctx['config'].get('tts', {}).get('piper', {})
    if SynthesisConfig is None:
        raise RuntimeError('Piper not available')
    return cast(Any, SynthesisConfig)(
        speaker_id=int(v_cfg.get('speaker', global_cfg.get('speaker', 0))),
        volume=1.5,
        length_scale=0.95,
        noise_scale=0.75,
        noise_w_scale=0.75,
        normalize_audio=True,
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
    """Yield s16le PCM chunks sentence-by-sentence. First chunk arrives within
    ~one sentence of synthesis latency (~200 ms typical)."""
    v_cfg = _resolve_piper_config(ctx)
    if not v_cfg:
        return
    try:
        voice = _load_piper_voice(v_cfg['model'], ctx['config'], v_cfg.get('config'))
        spec = _piper_spec(ctx, v_cfg)
        for audio_chunk in cast(Any, voice).synthesize(text, syn_config=spec):
            pcm = _chunk_to_system_pcm(audio_chunk)
            if pcm:
                yield pcm
    except Exception as e:
        log.error("Piper stream failed: %s", e)


def _provide_piper_pcm(text: str, ctx: dict) -> bytes | None:
    chunks = list(_piper_stream_pcm(text, ctx))
    return b''.join(chunks) if chunks else None


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


PCM_PROVIDERS: dict[str, Any] = {
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
    """Yield s16le PCM chunks as they are synthesized, one per sentence.
    File URI lines are yielded whole after ffmpeg decode."""
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


def _get_windows_sapi5_voices() -> list[dict[str, Any]]:
    try:
        comtypes_client = importlib.import_module('comtypes.client')
        speech = comtypes_client.CreateObject('SAPI.SPVoice')
        return [
            {
                'id': str(token.Id),
                'name': str(token.GetDescription()),
                'lang': _windows_locale_list(token.GetAttribute('Language')),
            }
            for token in speech.GetVoices()
        ]
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
        log.error('Failed to get pyttsx3 voices: %s', e)
        if os.name == 'nt':
            return _get_windows_sapi5_voices()
        return []

def _provide_pyttsx3(text: str, ctx: dict, out: pathlib.Path) -> bool:
    if pyttsx3 is None: return False
    cfg = ctx['config'].get('tts', {}).get('pyttsx3', {})
    if not cfg.get('enabled'): return False

    engine = None
    raw_tmp = out.with_suffix('.bin.wav')
    try:
        engine = pyttsx3.init()
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

PROVIDERS = {
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
            filter_str += f"concat=n={len(processed_parts)}:v=0:a=1[outa]"
            
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
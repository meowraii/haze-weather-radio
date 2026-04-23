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
        volume=2.0,
    )


def _piper_synthesize_wav_bytes(text: str, ctx: dict) -> bytes | None:
    v_cfg = _resolve_piper_config(ctx)
    if not v_cfg:
        return None
    try:
        voice = _load_piper_voice(v_cfg['model'], ctx['config'], v_cfg.get('config'))
        spec = _piper_spec(ctx, v_cfg)
        chunks = list(cast(Any, voice).synthesize(text, syn_config=spec))[
  {
    "identifier": "urn:oid:2.49.0.1.124.3610638895.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:41:56.992611+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - rainfall - ended",
      "event": "rainfall",
      "category": "",
      "responseType": "",
      "urgency": "Past",
      "severity": "Minor",
      "certainty": "Observed",
      "audience": "general public",
      "effective": "2026-04-23T01:02:36+00:00",
      "onset": null,
      "expires": "2026-04-23T02:02:36+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-23T01:02:36+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.1306487224.2026,2026-04-21T21:45:36-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.1623400869.2026,2026-04-22T10:11:22-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.0301200268.2026,2026-04-22T10:15:00-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.2432562057.2026,2026-04-23T00:21:40-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.1724836808.2026,2026-04-23T00:26:15-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260423010236197/WW_13_73_CWWG/RFW/2054248571079554968202604210501_WW_13_73_CWWG/actual/en_proper_complete_c-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17593 M:91128 C:102644"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "ended"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - rainfall"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "ended"
      }
    ],
    "text": {
      "description": "\nSignificant rainfall is no longer expected.\n\n\n###\n",
      "instruction": ""
    },
    "areas": [
      {
        "areaDesc": "R.M. of Meadow Lake including Waterhen Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717055"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717805"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717816"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Beaver River including Pierceland and Goodsoil",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717810"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717811"
          }
        ]
      },
      {
        "areaDesc": "Green Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718090"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loon Lake including Loon Lake and Makwa",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066140"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717807"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717808"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717809"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717815"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big River including Big River and Chitek Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066150"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716863"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Frenchman Butte including St. Walburg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717801"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mervin including Turtleford Mervin and Spruce Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717803"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717804"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Britannia including Hillmond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wilton including Lashburn Marshall and Lone Rock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eldon including Maidstone and Waseca",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Manitou Lake including Marsden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713091"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.1724836808.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:41:58.046665+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - rainfall - in effect",
      "event": "rainfall",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-23T00:21:40+00:00",
      "onset": "2026-04-23T00:15:00+00:00",
      "expires": "2026-04-23T04:59:40+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-23T00:26:15+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.1306487224.2026,2026-04-21T21:45:36-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.1623400869.2026,2026-04-22T10:11:22-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.0301200268.2026,2026-04-22T10:15:00-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.2432562057.2026,2026-04-23T00:21:40-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260423002140986/WW_13_73_CWWG/RFW/2054248571079554968202604210501_WW_13_73_CWWG/actual/en_proper_complete_u-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17593 M:91127 C:102643"
      },
      {
        "valueName": "profile:CAP-CP:0.4:MinorChange",
        "value": "text"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - rainfall"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-23T03:00:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\nHeavy rainfall with total amounts of 15 to 25 mm is expected. \n\nRain is beginning to change to snow. The full changeover to snow is expected in a few hours, and at that point rain will end.\n\n###\n\nWater will likely pool on roads and in low-lying areas.\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": "\nAvoid low-lying areas. Watch for washouts near rivers, creeks and culverts.\n"
    },
    "areas": [
      {
        "areaDesc": "R.M. of Meadow Lake including Waterhen Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717055"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717805"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717816"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Beaver River including Pierceland and Goodsoil",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717810"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717811"
          }
        ]
      },
      {
        "areaDesc": "Green Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718090"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loon Lake including Loon Lake and Makwa",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066140"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717807"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717808"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717809"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717815"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big River including Big River and Chitek Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066150"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716863"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Frenchman Butte including St. Walburg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717801"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mervin including Turtleford Mervin and Spruce Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717803"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717804"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Britannia including Hillmond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wilton including Lashburn Marshall and Lone Rock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eldon including Maidstone and Waseca",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Manitou Lake including Marsden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713091"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.2432562057.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:41:59.794312+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - rainfall - in effect",
      "event": "rainfall",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-23T00:21:40+00:00",
      "onset": "2026-04-23T00:15:00+00:00",
      "expires": "2026-04-23T04:59:40+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-23T00:21:40+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.1306487224.2026,2026-04-21T21:45:36-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.1623400869.2026,2026-04-22T10:11:22-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.0301200268.2026,2026-04-22T10:15:00-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260423002140986/WW_13_73_CWWG/RFW/2054248571079554968202604210501_WW_13_73_CWWG/actual/en_proper_complete_c-fr_not_present_u/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17593 M:91127 C:102642"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - rainfall"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-23T03:00:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\n###\n\nWater will likely pool on roads and in low-lying areas.\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": "\nAvoid low-lying areas. Watch for washouts near rivers, creeks and culverts.\n"
    },
    "areas": [
      {
        "areaDesc": "R.M. of Meadow Lake including Waterhen Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717055"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717805"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717816"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Beaver River including Pierceland and Goodsoil",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717810"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717811"
          }
        ]
      },
      {
        "areaDesc": "Green Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718090"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loon Lake including Loon Lake and Makwa",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066140"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717807"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717808"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717809"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717815"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big River including Big River and Chitek Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066150"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716863"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Frenchman Butte including St. Walburg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717801"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mervin including Turtleford Mervin and Spruce Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717803"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717804"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Britannia including Hillmond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wilton including Lashburn Marshall and Lone Rock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eldon including Maidstone and Waseca",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Manitou Lake including Marsden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713091"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.2595904495.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:42:07.603683+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "special weather statement in effect",
      "event": "weather",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Minor",
      "certainty": "Possible",
      "audience": "general public",
      "effective": "2026-04-22T21:22:22+00:00",
      "onset": "2026-04-22T21:16:00+00:00",
      "expires": "2026-04-23T13:22:22+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T21:22:22+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.3129929343.2026,2026-04-22T10:52:32-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.1699486493.2026,2026-04-22T18:08:20-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "statement"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422211716383/WS_13_73_CWWG/SPS/55244495581609489202604220501_WS_13_73_CWWG/actual/en_proper_complete_c-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17592 M:91106 C:102618"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "special weather statement"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WS_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-24T06:26:00.000Z"
      }
    ],
    "text": {
      "description": "\nA major late season storm is poised to affect Saskatchewan beginning later today and lasting into the weekend.\n\nPrecipitation will initially start as rain for portions of western and southern Saskatchewan before transitioning to snow overnight tonight  There will be a risk of thunderstorms, especially over southern regions this evening.\n\nThis will be a prolonged snowfall event, with snow lasting into Saturday.  Total snowfall accumulations of 10 to 15 cm are possible, and even up to 20 in areas from Outlook to Saskatoon to Prince Albert.\n\nThere will also be a risk of freezing rain tonight and Thursday, particularly over east-central regions of the province.\n\nStrong northerly winds will give reduced visibilities in falling snow.\n\nCooler, below seasonal weather is forecast in the wake of this system into next week.\n\n###\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": ""
    },
    "areas": [
      {
        "areaDesc": "R.M. of Lacadena including Kyle Tyner and Sanctuary",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Victory including Beechy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707063"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Canaan including Lucky Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Riverside including Cabri Pennant and Success",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Saskatchewan Landing including Stewart Valley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Webb including Webb and Antelope lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Swift Current including Swift Current and Wymark",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lac Pelletier including Blumenhof",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061225"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704061"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Excelsior including Waldeck Rush Lake and Main Centre",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707020"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707023"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Morse including Herbert Morse Ernfold and Gouldtown",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Coulee including Neidpath and McMahon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707018"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lawtonia including Hodgeville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Glen Bain including Glen Bain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061235"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Whiska Creek including Vanguard Neville and Pambrun",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061236"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Maple Bush including Riverhurst and Douglas Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707074"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Huron including Tugaske",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707077"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Enfield including Central Butte",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eyebrow including Eyebrow and Brownlee",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062214"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Craik including Craik and Aylesbury",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707093"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sarnia including Holdfast Chamberlain and Dilke",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Marquis including Tuxford Keeler and Buffalo Pound",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707051"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Dufferin including Bethune and Findlater",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706081"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Chaplin including Chaplin",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707031"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wheatlands including Mortlach and Parkbeg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707034"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Shamrock including Shamrock and Kelstern",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rodgers including Coderre and Courval",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Caron including Caronport and Caron",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707037"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pense including Pense Belle Plaine and Stony Beach",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062243"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706023"
          }
        ]
      },
      {
        "areaDesc": "City of Moose Jaw",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062244"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hillsborough including Crestwynd and Old Wives lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062245"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Redburn including Rouleau and Hearne",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062246"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Baildon including Briercrest",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062247"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707001"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Moose Jaw east of Highway 2 including Pasqua",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062248"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Moose Jaw west of Highway 2 including Bushell Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062249"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hudson Bay including Shoal Lake and Red Earth Reserves",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714839"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714840"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714846"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hudson Bay including Hudson Bay and Reserve",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Porcupine including Porcupine Plain and Weekes",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hazel Dell including Lintlaw and Hazel Dell",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709061"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Preeceville including Preeceville and Sturgis",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Invermay including Invermay and Rama",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buchanan including Buchanan Amsterdam and Tadmore",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064214"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709053"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Insinger including Theodore and Sheho",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Good Lake including Canora and Good Spirit Lake Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Keys including The Key Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709821"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709830"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709832"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Philips including Pelly and St Philips",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709820"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709822"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709826"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sliding Hills including Veregin Mikado and Hamton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709033"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Cote including Kamsack Togo and Duck Mountain Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Clayton including Norquay Stenen and Swan Plain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709069"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Livingston including Arran",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064242"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709076"
          }
        ]
      },
      {
        "areaDesc": "City of Saskatoon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065100"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Spiritwood including Spiritwood and Leoville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716862"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716880"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716882"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716894"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Canwood including Debden and Big River Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716858"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716860"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meeting Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716870"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716872"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Leask including Leask Mistawasis Res. and Parkside",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716854"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716855"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716886"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716888"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716890"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716891"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Shellbrook including Sturgeon Lake Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065230"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716856"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Duck Lake including Duck Lake and Beardy's Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065240"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715859"
          }
        ]
      },
      {
        "areaDesc": "District of Lakeland including Emma Lake and Anglin Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715076"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Paddockwood including Candle Lake and Paddockwood",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715070"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715098"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715099"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715851"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715853"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716857"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buckland including Wahpeton Res. and Spruce Home",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065261"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715085"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715094"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715848"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Garden River including Meath Park and Albertville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065262"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715085"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715092"
          }
        ]
      },
      {
        "areaDesc": "City of Prince Albert",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065263"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715846"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prince Albert including Davis",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Birch Hills including Muskoday Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715847"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Louis including One Arrow Res. and Domremy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715844"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715861"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Torch River including Choiceland and White Fox",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065310"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714093"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715849"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Nipawin including Nipawin Aylsham and Pontrilas",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065321"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714073"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714076"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Moose Range including Carrot River and Tobin Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065322"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kinistino including Kinistino and James Smith Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065331"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715849"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715850"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Invergordon including Yellow Creek and Tarnopol",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065332"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Flett's Springs including Beatty Ethelton and Pathlow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065333"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715052"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Three Lakes including Middle Lake and St Benedict",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065334"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715047"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lake Lenore including St. Brieux",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065335"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Willow Creek including Gronlid and Fairy Glen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065341"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714053"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Connaught including Ridgedale and New Osgoode",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065342"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Star City including Melfort and Star City",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065343"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714051"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Tisdale including Tisdale Eldersley and Sylvania",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065344"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pleasantdale including Naicam and Pleasantdale",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065345"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714030"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714035"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714842"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Barrier Valley including Archerwill and McKague",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065346"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Arborfield including Arborfield and Zenon Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065351"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bjorkdale including Greenwater Lake Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065352"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Redberry including Hafford and Krydor",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065411"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Blaine Lake including Blaine Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065412"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Great Bend including Radisson and Borden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065413"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716011"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Laird including Waldheim Hepburn and Laird",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065421"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosthern including Rosthern Hague and Carlton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065422"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715859"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eagle Creek including Arelee and Sonningdale",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065431"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Perdue including Perdue and Kinley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065433"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712050"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712052"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Vanscoy including Delisle Asquith and Vanscoy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065434"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Corman Park northeast of the Yellowhead Highway incl. Martensville Warman and Langham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065435"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711070"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711073"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711075"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Corman Park southwest of the Yellowhead Highway",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065436"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Aberdeen including Aberdeen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065441"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fish Creek including Alvena",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065442"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715857"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715862"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hoodoo including Wakaw and Cudworth",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065443"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715043"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grant including Vonda Prud'homme and Smuts",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065444"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715017"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bayne including Bruno Peterson and Dana",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065445"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Colonsay including Colonsay and Meacham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065451"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711078"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711079"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Viscount including Viscount and Plunkett",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065452"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Blucher including Allan Clavet Bradwell and Elstow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065453"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711069"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711077"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Harris including Harris and Tessier",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065511"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Montrose including Donovan and Swanson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065512"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Milden including Dinsmore Milden and Wiseton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065513"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712012"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fertile Valley including Conquest Macrorie and Bounty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065514"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712020"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of King George northwest of Lucky Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065515"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Coteau including Birsay and Danielson Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065516"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Dundurn including Dundurn and Blackstrap Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065521"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711828"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rudy including Outlook and Glenside",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065522"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosedale including Hanley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065523"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loreburn including Elbow Loreburn and Hawarden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065524"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711024"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lost River including South Allan and the Allan Hills",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065531"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Morris including Watrous Young and Zelma",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065532"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of McCraney including Kenaston and Bladworth",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065533"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wood Creek including Simpson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065534"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Arm River including Davidson and Girvin",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065541"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711014"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Willner west of Davidson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065542"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big Arm including Imperial and Liberty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065543"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Humboldt including Humboldt Carmel and Fulda",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065611"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Peter including Muenster and Lake Lenore",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065612"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715005"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715006"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wolverine including Burr",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065613"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711096"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Leroy including Leroy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065614"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Spalding including Spalding",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065621"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Ponass Lake including Rose Valley Fosston and Nora",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065622"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714025"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lakeside including Watson and Quill Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065623"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lakeview including Wadena and Clair",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065624"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Usborne including Lanigan Drake and Guernsey",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065631"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prairie Rose including Jansen and Esk",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065632"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wreford including Nokomis and Venn",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065633"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mount Hope including Semans",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065634"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710024"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Last Mountain Valley including Govan and Duval",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065635"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big Quill including Wynyard Dafoe and Kandahar",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065641"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710828"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Elfros including Elfros Leslie and Mozart",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065642"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710043"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kutawa including Raymore Punnichy and Poorman Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065643"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710824"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710825"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Emerald including Wishart and Bankend",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065644"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710852"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Touchwood including Serath and Touchwood Hills Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065645"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710823"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kellross including Kelliher and Lestock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065646"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710012"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710822"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710832"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710834"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710836"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710838"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710840"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710842"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710843"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710844"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710846"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710847"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710848"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710849"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710850"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710851"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kelvington including Yellowquill Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065651"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714841"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sasman including Margo Kylemore and Nut Mountain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065652"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Foam Lake including Foam Lake and Fishing Lake Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065653"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710035"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710826"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710854"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Ituna Bon Accord including Ituna and Hubbard",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065654"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meadow Lake including Waterhen Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717055"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717805"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717816"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Beaver River including Pierceland and Goodsoil",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717810"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717811"
          }
        ]
      },
      {
        "areaDesc": "Green Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718090"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loon Lake including Loon Lake and Makwa",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066140"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717807"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717808"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717809"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717815"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big River including Big River and Chitek Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066150"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716863"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Frenchman Butte including St. Walburg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717801"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mervin including Turtleford Mervin and Spruce Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717803"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717804"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Turtle River including Edam and Vawn",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717011"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Britannia including Hillmond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wilton including Lashburn Marshall and Lone Rock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eldon including Maidstone and Waseca",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Paynton including Paynton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717825"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Manitou Lake including Marsden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713091"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hillsdale including Neilburg and Baldwinton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Senlac including Senlac",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713078"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Round Valley including Unity",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713074"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Cut Knife including Cut Knife",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713096"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713098"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713835"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713836"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Battle River including Sweet Grass Res. and Delmas",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066242"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712078"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712829"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712830"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712832"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712833"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712837"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buffalo including Phippen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066243"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Parkdale including Glaslyn and Fairholme",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717048"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Medstead including Medstead Belbutte and Birch Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716861"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meota including Meota and The Battlefords Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066253"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717005"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717812"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717813"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Round Hill including Rabbit Lake and Whitkow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066254"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716033"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716034"
          }
        ]
      },
      {
        "areaDesc": "The Battlefords",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066260"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of North Battleford northwest of The Battlefords",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Douglas including Speers Richard and Alticane",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716892"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mayfield including Maymont Denholm and Fielding",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716005"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Glenside north of Biggar",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066280"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eye Hill including Macklin Denzil and Evesham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066311"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grass Lake including Salvador and Reward",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066312"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713056"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Heart's Hill including Cactus Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066313"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713046"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Progress including Kerrobert and Luseland",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066314"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Tramping Lake including Scott and Revenue",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066321"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Reford including Wilkie Landis and Leipzig",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066322"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mariposa including Tramping Lake and Broadacres",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066323"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grandview including Handel and Kelfield",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066324"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713033"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosemount including Cando and Traynor",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066331"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Biggar including Biggar and Springwater",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066332"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712046"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Antelope Park including Loverna and Hoosier",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066341"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prairiedale including Major and Smiley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066342"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Milton including Alsask and Marengo",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066343"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713014"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Oakdale including Coleville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066351"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Winslow including Dodsland and Plenty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066352"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713031"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kindersley including Kindersley Brock and Flaxcombe",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066353"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mountain View including Herschel and Stranraer",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066361"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Marriott south of Biggar",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066362"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712034"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pleasant Valley including McGee and Fiske",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066363"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712001"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Andrews including Rosetown and Zealandia",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066364"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Chesterfield including Eatonia and Mantario",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066371"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Newcombe including Glidden and Madison",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066372"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708071"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Monet including Elrose Wartime and Forgan",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066381"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Snipe Lake including Eston and Plato",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066382"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708076"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.2966475486.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:42:10.434008+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - rainfall - in effect",
      "event": "rainfall",
      "category": "",
      "responseType": "",
      "urgency": "Immediate",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-22T21:20:48+00:00",
      "onset": "2026-04-22T21:18:00+00:00",
      "expires": "2026-04-23T08:01:48+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T21:20:48+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.3937344710.2026,2026-04-21T21:49:08-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.2355305607.2026,2026-04-22T08:14:13-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.2124846331.2026,2026-04-22T12:26:32-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.0700039062.2026,2026-04-22T16:54:17-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422211852904/WW_16_76_CWWG/RFW/2022071891763871591202604210501_WW_16_76_CWWG/actual/en_proper_complete_c-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17592 M:91104 C:102616"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - rainfall"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Northern Alberta"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_16_76_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-23T06:00:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\nRainfall, combined with melting snow, continues. The ground, already near saturation, has little ability to absorb further rainfall.\n\nAn additional 5 mm of rain is expected this evening, bringing rainfall totals to 25 mm in some locations.\n\nRain will change over to snow this evening.\n\n###\n\nWater will likely pool on roads and in low-lying areas.\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to ABstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #ABStorm.\n",
      "instruction": "\nDon't drive through flooded roadways. Avoid low-lying areas. Watch for washouts near rivers, creeks and culverts.\n"
    },
    "areas": [
      {
        "areaDesc": "City of Lloydminster",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066400-075260"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810039"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Plamondon Hylo and Avenir",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075111"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Heart Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075112"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812840"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Lac La Biche and Square Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075113"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812828"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Lakeland Prov. Park and Rec. Area",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075114"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Fork Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075115"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville including Cold Lake Air Weapons Range",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075116"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4816037"
          }
        ]
      },
      {
        "areaDesc": "Smoky Lake Co. near Buffalo Lake and Kikino Smts",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075121"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812022"
          }
        ]
      },
      {
        "areaDesc": "Smoky Lake Co. near Vilna Saddle Lake and Whitefish Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075122"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812808"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near Ashmont St. Vincent and St. Lina",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075131"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812020"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near St. Paul and Lafond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075132"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812018"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near Elk Point and St. Edouard",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075133"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812016"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near Lindbergh and Frog Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075134"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812804"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near La Corey Wolf Lake and Truman",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075141"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Glendon and Moose Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075142"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812012"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Bonnyville Ardmore and Kehewin Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075143"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812811"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Cold Lake and City of Cold Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075144"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812813"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812815"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Beaverdam",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075145"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812810"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Fishing Lake Smt",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075146"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          }
        ]
      },
      {
        "areaDesc": "Co. of Two Hills near Two Hills and Brosseau",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810052"
          }
        ]
      },
      {
        "areaDesc": "Co. of Two Hills near Myrnam and Derwent",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810051"
          }
        ]
      },
      {
        "areaDesc": "Co. of Minburn near Innisfree Lavoy and Ranfurly",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810031"
          }
        ]
      },
      {
        "areaDesc": "Co. of Minburn near Minburn and Mannville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810034"
          }
        ]
      },
      {
        "areaDesc": "Beaver Co. near Viking and Kinsella",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075230"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810022"
          }
        ]
      },
      {
        "areaDesc": "Flagstaff Co. near Killam and Sedgewick",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807044"
          }
        ]
      },
      {
        "areaDesc": "Flagstaff Co. near Lougheed and Hardisty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075242"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807048"
          }
        ]
      },
      {
        "areaDesc": "Flagstaff Co. near Alliance and Bellshill Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075243"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807032"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Vermilion",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810042"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Islay and McNabb Sanctuary",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Dewberry and Clandonald",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075253"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810046"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Tulliby Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075254"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810805"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Kitscoty and Marwayne",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075255"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810044"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Paradise Valley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075256"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810038"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Irma",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807056"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Wainwright",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807054"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Edgerton and Koroluk Landslide",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807052"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Chauvin Dillberry Lake and Roros",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075274"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807051"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Provost near Hughenden Amisk and Kessler",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075281"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807008"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Provost near Czar Metiskow and Cadogan",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075282"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807004"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Provost near Provost and Hayter",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075283"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807002"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.3952465859.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:42:12.641605+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - snowfall - in effect",
      "event": "snowfall",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-22T21:12:09+00:00",
      "onset": "2026-04-23T09:50:00+00:00",
      "expires": "2026-04-23T13:12:09+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        },
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T21:17:40+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.0272271273.2026,2026-04-22T21:12:09-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422211209287/WW_13_73_CWWG/SFW/1858203161489275794202604220501_WW_13_73_CWWG/actual/en_proper_complete_u-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17592 M:91098 C:102610"
      },
      {
        "valueName": "profile:CAP-CP:0.4:MinorChange",
        "value": "text"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - snowfall"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Newly_Active_Areas",
        "value": "065100,065211,065212,065221,065222,065230,065240,065251,065252,065261,065262,065263,065271,065272,065273,065411,065412,065413,065421,065422,065431,065433,065434,065435,065436,065441,065442,065443,065444,065445,065451,065452,065453,065511,065512,065513,065514,065515,065516,065521,065522,065523,065524,065531,065532,065533,065534,065541,065542,065543"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-24T00:50:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\nSnowfall with total amounts of 15 to 20 cm is expected.\n\nSnowfall of 15 to 20 cm is expected by Thursday evening.\n\nA strong low pressure system will bring heavy snowfall to the region on Thursday. Snow will begin overnight tonight and be particularly heavy through the day, ending Thursday evening.  15 to 20 cm are expected.\n\n###\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": ""
    },
    "areas": [
      {
        "areaDesc": "City of Saskatoon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065100"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Spiritwood including Spiritwood and Leoville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716862"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716880"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716882"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716894"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Canwood including Debden and Big River Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716858"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716860"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meeting Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716870"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716872"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Leask including Leask Mistawasis Res. and Parkside",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716854"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716855"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716886"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716888"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716890"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716891"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Shellbrook including Sturgeon Lake Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065230"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716856"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Duck Lake including Duck Lake and Beardy's Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065240"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715859"
          }
        ]
      },
      {
        "areaDesc": "District of Lakeland including Emma Lake and Anglin Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715076"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Paddockwood including Candle Lake and Paddockwood",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715070"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715098"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715099"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715851"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715853"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716857"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buckland including Wahpeton Res. and Spruce Home",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065261"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715085"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715094"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715848"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Garden River including Meath Park and Albertville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065262"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715085"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715092"
          }
        ]
      },
      {
        "areaDesc": "City of Prince Albert",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065263"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715846"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prince Albert including Davis",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Birch Hills including Muskoday Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715847"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Louis including One Arrow Res. and Domremy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715844"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715861"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Redberry including Hafford and Krydor",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065411"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Blaine Lake including Blaine Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065412"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Great Bend including Radisson and Borden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065413"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716011"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Laird including Waldheim Hepburn and Laird",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065421"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosthern including Rosthern Hague and Carlton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065422"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715859"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eagle Creek including Arelee and Sonningdale",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065431"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Perdue including Perdue and Kinley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065433"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712050"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712052"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Vanscoy including Delisle Asquith and Vanscoy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065434"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Corman Park northeast of the Yellowhead Highway incl. Martensville Warman and Langham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065435"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711070"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711073"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711075"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Corman Park southwest of the Yellowhead Highway",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065436"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Aberdeen including Aberdeen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065441"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fish Creek including Alvena",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065442"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715857"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715862"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hoodoo including Wakaw and Cudworth",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065443"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715043"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grant including Vonda Prud'homme and Smuts",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065444"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715017"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bayne including Bruno Peterson and Dana",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065445"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Colonsay including Colonsay and Meacham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065451"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711078"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711079"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Viscount including Viscount and Plunkett",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065452"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Blucher including Allan Clavet Bradwell and Elstow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065453"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711069"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711077"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Harris including Harris and Tessier",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065511"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Montrose including Donovan and Swanson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065512"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Milden including Dinsmore Milden and Wiseton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065513"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712012"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fertile Valley including Conquest Macrorie and Bounty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065514"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712020"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of King George northwest of Lucky Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065515"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Coteau including Birsay and Danielson Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065516"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Dundurn including Dundurn and Blackstrap Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065521"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711828"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rudy including Outlook and Glenside",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065522"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosedale including Hanley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065523"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loreburn including Elbow Loreburn and Hawarden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065524"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711024"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lost River including South Allan and the Allan Hills",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065531"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Morris including Watrous Young and Zelma",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065532"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of McCraney including Kenaston and Bladworth",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065533"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wood Creek including Simpson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065534"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Arm River including Davidson and Girvin",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065541"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711014"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Willner west of Davidson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065542"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big Arm including Imperial and Liberty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065543"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711009"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.1010643242.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:42:15.023949+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - wind - in effect",
      "event": "wind",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-22T21:12:40+00:00",
      "onset": "2026-04-23T03:00:00+00:00",
      "expires": "2026-04-23T13:12:40+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        },
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T21:15:30+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.4124017873.2026,2026-04-22T10:13:52-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.1100028534.2026,2026-04-22T21:12:40-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422211240544/WW_13_73_CWWG/WDW/54426281581609489202604220501_WW_13_73_CWWG/actual/en_proper_complete_u-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17592 M:91098 C:102609"
      },
      {
        "valueName": "profile:CAP-CP:0.4:MinorChange",
        "value": "text"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - wind"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Newly_Active_Areas",
        "value": "061111,061112,061113,061114,061115,061121,061122,061123,061131,061132,061133,061211,061212,061213,061221,061222,061223,061224,061225,061231,061232,061233,061234,061235,061236,062411,062412,062413,062414,062421,062422,062431,062432,062433,062434,062441,062442,062443,062451,062452,062461,062462,062463,062464,062511,062512,062513,062514,062521,062522,062523,062524,062531,062532,062533,062534,062541,062542,062543,062544"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-23T15:00:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\nStrong winds that may cause damage are expected.\n\nWhat:  90 km/h wind gusts.\n\nWhen:  Overnight Wednesday night and lasting until Thursday afternoon\n\nWhere:  Southern Saskatchewan.\n\nRemarks:  A major spring storm will give severe wind gusts to portions of southern Saskatchewan tonight.\n\n###\n\nLocal utility outages are possible. Some property damage is possible.\n\nWind warnings are issued when there is a significant risk of damaging winds.\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": ""
    },
    "areas": [
      {
        "areaDesc": "R.M. of Deer Forks including Burstall and Estuary",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061111"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Happyland including Leader Prelate and Mendham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061112"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Enterprise including Richmound",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061113"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fox Valley including Fox Valley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061114"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big Stick including Golden Prairie",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061115"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708018"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Clinworth including Sceptre Lemsford and Portreeve",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061121"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Miry Creek including Abbey Lancer and Shackleton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061122"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pittville including Hazlet",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061123"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Piapot including Piapot",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061131"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704050"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Gull Lake including Gull Lake and Tompkins",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061132"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Carmichael including Carmichael",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061133"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704056"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lacadena including Kyle Tyner and Sanctuary",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Victory including Beechy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707063"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Canaan including Lucky Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Riverside including Cabri Pennant and Success",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Saskatchewan Landing including Stewart Valley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Webb including Webb and Antelope lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Swift Current including Swift Current and Wymark",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lac Pelletier including Blumenhof",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061225"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704061"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Excelsior including Waldeck Rush Lake and Main Centre",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707020"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707023"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Morse including Herbert Morse Ernfold and Gouldtown",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Coulee including Neidpath and McMahon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707018"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lawtonia including Hodgeville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Glen Bain including Glen Bain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061235"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Whiska Creek including Vanguard Neville and Pambrun",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061236"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Maple Creek including Maple Creek",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061311"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704048"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Maple Creek including Cypress Hills Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061312"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704802"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Reno including Consul Robsart and Willow Creek",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061313"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704021"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Arlington including Dollard",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061321"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of White Valley including Eastend and Ravenscrag",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061322"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Frontier including Frontier and Claydon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061323"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bone Creek including Simmie and Scotsguard",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061331"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grassy Creek including Shaunavon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061332"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wise Creek including Cadillac and Admiral",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061333"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lone Tree including Climax and Bracken",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061341"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Val Marie including Val Marie Orkney and Monchy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061342"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704003"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Auvergne including Ponteix and Aneroid",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061351"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pinto Creek including Kincaid and Hazenmore",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061352"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703052"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Glen McPherson west of Mankota",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061353"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mankota including Mankota and Ferland",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061354"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Gravelbourg including Gravelbourg and Bateman",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062411"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703071"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sutton including Mazenod Palmer and Vantage",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062412"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703074"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wood River including Lafleche Woodrow and Melaval",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062413"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Stonehenge including Limerick and Congress",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062414"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Waverley including Glentworth and Fir Mountain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062421"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703801"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Old Post including Wood Mountain and Killdeer",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062422"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lake Johnson including Mossbank and Ardill",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062431"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703093"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Terrell including Spring Valley and Cardross",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062432"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703096"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lake of The Rivers including Assiniboia",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062433"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Excel including Viceroy Ormiston and Verwood",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062434"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Willow Bunch including Willow Bunch and St Victor",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062441"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Poplar Valley including Rockglen and Fife Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062442"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hart Butte including Coronach",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062443"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Elmsthorpe including Avonlea and Truax",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062451"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702800"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Key West including Ogema and Kayville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062452"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702800"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bengough including Bengough",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062461"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702024"
          }
        ]
      },
      {
        "areaDesc": "R.M. of The Gap including Ceylon and Hardy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062462"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Happy Valley including Big Beaver",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062463"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702018"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Surprise Valley including Minton and Regway",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062464"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702015"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Caledonia including Milestone and Parry",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062511"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702800"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Scott including Yellow Grass Lang and Lewvan",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062512"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702069"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Norton including Pangman and Khedive",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062513"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Brokenshell including Trossachs",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062514"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702051"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wellington including Cedoux Colfax and Tyvan",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062521"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702073"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fillmore including Fillmore Creelman and Osage",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062522"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702078"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702079"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Weyburn including Weyburn and McTaggart",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062523"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702048"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Griffin including Griffin and Froude",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062524"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702042"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Laurier including Radville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062531"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702031"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lomond including Colgate and Goodwater",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062532"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702033"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lake Alma including Lake Alma and Beaubier",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062533"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Souris Valley including Tribune and Oungre",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062534"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Cymri including Midale Macoun and Halbrite",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062541"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Benson including Benson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062542"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4701027"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Cambria including Torquay and Outram",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062543"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702002"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Estevan including Estevan and Hitchcock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062544"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4701022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4701024"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.1100028534.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:42:16.771544+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - wind - in effect",
      "event": "wind",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-22T21:12:40+00:00",
      "onset": "2026-04-23T03:00:00+00:00",
      "expires": "2026-04-23T13:12:40+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        },
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T21:12:40+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.4124017873.2026,2026-04-22T10:13:52-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422211240544/WW_13_73_CWWG/WDW/54426281581609489202604220501_WW_13_73_CWWG/actual/en_proper_complete_c-fr_not_present_u/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17592 M:91098 C:102607"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - wind"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Newly_Active_Areas",
        "value": "061111,061112,061113,061114,061115,061121,061122,061123,061131,061132,061133,061211,061212,061213,061221,061222,061223,061224,061225,061231,061232,061233,061234,061235,061236,062411,062412,062413,062414,062421,062422,062431,062432,062433,062434,062441,062442,062443,062451,062452,062461,062462,062463,062464,062511,062512,062513,062514,062521,062522,062523,062524,062531,062532,062533,062534,062541,062542,062543,062544"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-23T15:00:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\nStrong winds that may cause damage are expected.\n\n###\n\nLocal utility outages are possible. Some property damage is possible.\n\nWind warnings are issued when there is a significant risk of damaging winds.\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": ""
    },
    "areas": [
      {
        "areaDesc": "R.M. of Deer Forks including Burstall and Estuary",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061111"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Happyland including Leader Prelate and Mendham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061112"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Enterprise including Richmound",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061113"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fox Valley including Fox Valley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061114"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big Stick including Golden Prairie",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061115"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708018"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Clinworth including Sceptre Lemsford and Portreeve",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061121"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Miry Creek including Abbey Lancer and Shackleton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061122"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pittville including Hazlet",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061123"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Piapot including Piapot",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061131"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704050"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Gull Lake including Gull Lake and Tompkins",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061132"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Carmichael including Carmichael",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061133"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704056"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lacadena including Kyle Tyner and Sanctuary",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Victory including Beechy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707063"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Canaan including Lucky Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Riverside including Cabri Pennant and Success",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Saskatchewan Landing including Stewart Valley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Webb including Webb and Antelope lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Swift Current including Swift Current and Wymark",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lac Pelletier including Blumenhof",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061225"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704061"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Excelsior including Waldeck Rush Lake and Main Centre",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707020"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707023"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Morse including Herbert Morse Ernfold and Gouldtown",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Coulee including Neidpath and McMahon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707018"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lawtonia including Hodgeville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Glen Bain including Glen Bain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061235"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Whiska Creek including Vanguard Neville and Pambrun",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061236"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Maple Creek including Maple Creek",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061311"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704048"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Maple Creek including Cypress Hills Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061312"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704802"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Reno including Consul Robsart and Willow Creek",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061313"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704021"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Arlington including Dollard",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061321"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of White Valley including Eastend and Ravenscrag",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061322"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Frontier including Frontier and Claydon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061323"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bone Creek including Simmie and Scotsguard",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061331"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grassy Creek including Shaunavon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061332"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wise Creek including Cadillac and Admiral",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061333"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lone Tree including Climax and Bracken",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061341"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Val Marie including Val Marie Orkney and Monchy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061342"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704003"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Auvergne including Ponteix and Aneroid",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061351"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pinto Creek including Kincaid and Hazenmore",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061352"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703052"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Glen McPherson west of Mankota",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061353"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mankota including Mankota and Ferland",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061354"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Gravelbourg including Gravelbourg and Bateman",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062411"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703071"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sutton including Mazenod Palmer and Vantage",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062412"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703074"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wood River including Lafleche Woodrow and Melaval",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062413"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Stonehenge including Limerick and Congress",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062414"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Waverley including Glentworth and Fir Mountain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062421"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703801"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Old Post including Wood Mountain and Killdeer",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062422"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lake Johnson including Mossbank and Ardill",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062431"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703093"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Terrell including Spring Valley and Cardross",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062432"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703096"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lake of The Rivers including Assiniboia",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062433"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Excel including Viceroy Ormiston and Verwood",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062434"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Willow Bunch including Willow Bunch and St Victor",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062441"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Poplar Valley including Rockglen and Fife Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062442"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hart Butte including Coronach",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062443"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Elmsthorpe including Avonlea and Truax",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062451"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702800"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Key West including Ogema and Kayville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062452"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702800"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bengough including Bengough",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062461"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702024"
          }
        ]
      },
      {
        "areaDesc": "R.M. of The Gap including Ceylon and Hardy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062462"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Happy Valley including Big Beaver",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062463"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702018"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Surprise Valley including Minton and Regway",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062464"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702015"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Caledonia including Milestone and Parry",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062511"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702800"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Scott including Yellow Grass Lang and Lewvan",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062512"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702069"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Norton including Pangman and Khedive",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062513"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Brokenshell including Trossachs",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062514"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702051"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wellington including Cedoux Colfax and Tyvan",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062521"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702073"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fillmore including Fillmore Creelman and Osage",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062522"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702078"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702079"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Weyburn including Weyburn and McTaggart",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062523"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702048"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Griffin including Griffin and Froude",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062524"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702042"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Laurier including Radville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062531"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702031"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lomond including Colgate and Goodwater",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062532"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702033"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lake Alma including Lake Alma and Beaubier",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062533"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Souris Valley including Tribune and Oungre",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062534"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Cymri including Midale Macoun and Halbrite",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062541"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Benson including Benson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062542"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4701027"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Cambria including Torquay and Outram",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062543"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4702002"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Estevan including Estevan and Hitchcock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062544"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4701022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4701024"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.0272271273.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:42:18.463188+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - snowfall - in effect",
      "event": "snowfall",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-22T21:12:09+00:00",
      "onset": "2026-04-23T09:50:00+00:00",
      "expires": "2026-04-23T13:12:09+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        },
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T21:12:09+00:00",
      "status": "Actual",
      "msgType": "Alert",
      "scope": "Public",
      "references": null,
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422211209287/WW_13_73_CWWG/SFW/1858203161489275794202604220501_WW_13_73_CWWG/actual/en_proper_complete_c-fr_not_present_u/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17592 M:91097 C:102606"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - snowfall"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Newly_Active_Areas",
        "value": "065100,065211,065212,065221,065222,065230,065240,065251,065252,065261,065262,065263,065271,065272,065273,065411,065412,065413,065421,065422,065431,065433,065434,065435,065436,065441,065442,065443,065444,065445,065451,065452,065453,065511,065512,065513,065514,065515,065516,065521,065522,065523,065524,065531,065532,065533,065534,065541,065542,065543"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-24T00:50:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\nSnowfall with total amounts of 15 to 20 cm is expected.\n\n###\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": ""
    },
    "areas": [
      {
        "areaDesc": "City of Saskatoon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065100"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Spiritwood including Spiritwood and Leoville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716862"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716880"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716882"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716894"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Canwood including Debden and Big River Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716858"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716860"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meeting Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716870"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716872"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Leask including Leask Mistawasis Res. and Parkside",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716854"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716855"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716886"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716888"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716890"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716891"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Shellbrook including Sturgeon Lake Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065230"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716856"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Duck Lake including Duck Lake and Beardy's Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065240"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715859"
          }
        ]
      },
      {
        "areaDesc": "District of Lakeland including Emma Lake and Anglin Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715076"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Paddockwood including Candle Lake and Paddockwood",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715070"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715098"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715099"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715851"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715853"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716857"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buckland including Wahpeton Res. and Spruce Home",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065261"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715085"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715094"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715848"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Garden River including Meath Park and Albertville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065262"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715085"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715092"
          }
        ]
      },
      {
        "areaDesc": "City of Prince Albert",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065263"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715846"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prince Albert including Davis",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Birch Hills including Muskoday Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715847"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Louis including One Arrow Res. and Domremy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715844"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715861"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Redberry including Hafford and Krydor",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065411"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Blaine Lake including Blaine Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065412"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Great Bend including Radisson and Borden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065413"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716011"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Laird including Waldheim Hepburn and Laird",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065421"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosthern including Rosthern Hague and Carlton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065422"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715859"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eagle Creek including Arelee and Sonningdale",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065431"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Perdue including Perdue and Kinley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065433"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712050"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712052"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Vanscoy including Delisle Asquith and Vanscoy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065434"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Corman Park northeast of the Yellowhead Highway incl. Martensville Warman and Langham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065435"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711070"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711073"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711075"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Corman Park southwest of the Yellowhead Highway",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065436"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Aberdeen including Aberdeen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065441"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fish Creek including Alvena",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065442"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715857"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715862"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hoodoo including Wakaw and Cudworth",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065443"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715043"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grant including Vonda Prud'homme and Smuts",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065444"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715017"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bayne including Bruno Peterson and Dana",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065445"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Colonsay including Colonsay and Meacham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065451"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711078"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711079"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Viscount including Viscount and Plunkett",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065452"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Blucher including Allan Clavet Bradwell and Elstow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065453"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711069"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711077"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Harris including Harris and Tessier",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065511"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Montrose including Donovan and Swanson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065512"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Milden including Dinsmore Milden and Wiseton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065513"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712012"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fertile Valley including Conquest Macrorie and Bounty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065514"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712020"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of King George northwest of Lucky Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065515"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Coteau including Birsay and Danielson Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065516"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Dundurn including Dundurn and Blackstrap Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065521"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711828"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rudy including Outlook and Glenside",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065522"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosedale including Hanley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065523"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loreburn including Elbow Loreburn and Hawarden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065524"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711024"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lost River including South Allan and the Allan Hills",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065531"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Morris including Watrous Young and Zelma",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065532"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of McCraney including Kenaston and Bladworth",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065533"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wood Creek including Simpson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065534"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Arm River including Davidson and Girvin",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065541"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711014"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Willner west of Davidson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065542"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big Arm including Imperial and Liberty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065543"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711009"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.0700039062.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:42:48.231831+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - rainfall - in effect",
      "event": "rainfall",
      "category": "",
      "responseType": "",
      "urgency": "Immediate",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-22T16:54:17+00:00",
      "onset": "2026-04-22T16:52:00+00:00",
      "expires": "2026-04-23T08:54:17+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T16:54:17+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.3937344710.2026,2026-04-21T21:49:08-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.2355305607.2026,2026-04-22T08:14:13-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.2124846331.2026,2026-04-22T12:26:32-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422165302986/WW_16_76_CWWG/RFW/2022071891763871591202604210501_WW_16_76_CWWG/actual/en_proper_complete_c-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17584 M:91044 C:102550"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - rainfall"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Northern Alberta"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_16_76_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-23T09:00:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\nRainfall, combined with melting snow, continues. The ground, already near saturation, has little ability to absorb further rainfall.\n\nTotal rainfall amounts will reach 25 mm in some locations. \n\nRain will change over to snow this evening.\n\n###\n\nWater will likely pool on roads and in low-lying areas.\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to ABstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #ABStorm.\n",
      "instruction": "\nDon't drive through flooded roadways. Avoid low-lying areas. Watch for washouts near rivers, creeks and culverts.\n"
    },
    "areas": [
      {
        "areaDesc": "City of Lloydminster",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066400-075260"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810039"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Plamondon Hylo and Avenir",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075111"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Heart Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075112"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812840"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Lac La Biche and Square Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075113"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812828"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Lakeland Prov. Park and Rec. Area",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075114"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Fork Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075115"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville including Cold Lake Air Weapons Range",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075116"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4816037"
          }
        ]
      },
      {
        "areaDesc": "Smoky Lake Co. near Buffalo Lake and Kikino Smts",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075121"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812022"
          }
        ]
      },
      {
        "areaDesc": "Smoky Lake Co. near Vilna Saddle Lake and Whitefish Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075122"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812808"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near Ashmont St. Vincent and St. Lina",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075131"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812020"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near St. Paul and Lafond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075132"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812018"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near Elk Point and St. Edouard",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075133"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812016"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near Lindbergh and Frog Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075134"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812804"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near La Corey Wolf Lake and Truman",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075141"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Glendon and Moose Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075142"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812012"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Bonnyville Ardmore and Kehewin Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075143"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812811"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Cold Lake and City of Cold Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075144"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812813"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812815"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Beaverdam",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075145"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812810"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Fishing Lake Smt",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075146"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          }
        ]
      },
      {
        "areaDesc": "Co. of Two Hills near Two Hills and Brosseau",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810052"
          }
        ]
      },
      {
        "areaDesc": "Co. of Two Hills near Myrnam and Derwent",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810051"
          }
        ]
      },
      {
        "areaDesc": "Co. of Minburn near Innisfree Lavoy and Ranfurly",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810031"
          }
        ]
      },
      {
        "areaDesc": "Co. of Minburn near Minburn and Mannville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810034"
          }
        ]
      },
      {
        "areaDesc": "Beaver Co. near Viking and Kinsella",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075230"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810022"
          }
        ]
      },
      {
        "areaDesc": "Flagstaff Co. near Killam and Sedgewick",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807044"
          }
        ]
      },
      {
        "areaDesc": "Flagstaff Co. near Lougheed and Hardisty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075242"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807048"
          }
        ]
      },
      {
        "areaDesc": "Flagstaff Co. near Alliance and Bellshill Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075243"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807032"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Vermilion",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810042"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Islay and McNabb Sanctuary",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Dewberry and Clandonald",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075253"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810046"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Tulliby Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075254"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810805"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Kitscoty and Marwayne",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075255"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810044"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Paradise Valley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075256"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810038"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Irma",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807056"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Wainwright",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807054"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Edgerton and Koroluk Landslide",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807052"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Chauvin Dillberry Lake and Roros",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075274"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807051"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Provost near Hughenden Amisk and Kessler",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075281"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807008"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Provost near Czar Metiskow and Cadogan",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075282"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807004"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Provost near Provost and Hayter",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075283"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807002"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.2124846331.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:42:54.032965+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - rainfall - in effect",
      "event": "rainfall",
      "category": "",
      "responseType": "",
      "urgency": "Immediate",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-22T12:26:32+00:00",
      "onset": "2026-04-22T12:21:00+00:00",
      "expires": "2026-04-23T04:26:32+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T12:26:32+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.3937344710.2026,2026-04-21T21:49:08-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.2355305607.2026,2026-04-22T08:14:13-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422122226899/WW_16_76_CWWG/RFW/2022071891763871591202604210501_WW_16_76_CWWG/actual/en_proper_complete_c-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17584 M:91023 C:102529"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - rainfall"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Northern Alberta"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_16_76_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-23T09:00:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\nHeavy rain is expected today, with total amounts of 25 to 30 mm.\n\nRain will intensify this morning. Localized flooding is possible as the frozen ground is less able to absorb falling rain. \n\nRain will change over to snow late tonight.\n\n###\n\nWater will likely pool on roads and in low-lying areas.\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to ABstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #ABStorm.\n",
      "instruction": "\nAvoid low-lying areas. Watch for washouts near rivers, creeks and culverts.\n"
    },
    "areas": [
      {
        "areaDesc": "City of Lloydminster",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066400-075260"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810039"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Plamondon Hylo and Avenir",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075111"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Heart Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075112"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812840"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Lac La Biche and Square Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075113"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812828"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Lakeland Prov. Park and Rec. Area",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075114"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          }
        ]
      },
      {
        "areaDesc": "Lac La Biche Co. near Fork Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075115"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville including Cold Lake Air Weapons Range",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075116"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4816037"
          }
        ]
      },
      {
        "areaDesc": "Smoky Lake Co. near Buffalo Lake and Kikino Smts",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075121"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812022"
          }
        ]
      },
      {
        "areaDesc": "Smoky Lake Co. near Vilna Saddle Lake and Whitefish Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075122"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812808"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near Ashmont St. Vincent and St. Lina",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075131"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812020"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near St. Paul and Lafond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075132"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812018"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near Elk Point and St. Edouard",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075133"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812016"
          }
        ]
      },
      {
        "areaDesc": "Co. of St. Paul near Lindbergh and Frog Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075134"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812804"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near La Corey Wolf Lake and Truman",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075141"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Glendon and Moose Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075142"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812012"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Bonnyville Ardmore and Kehewin Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075143"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812811"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Cold Lake and City of Cold Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075144"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812813"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812815"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Beaverdam",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075145"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812810"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Bonnyville near Fishing Lake Smt",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075146"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4812004"
          }
        ]
      },
      {
        "areaDesc": "Co. of Two Hills near Two Hills and Brosseau",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810052"
          }
        ]
      },
      {
        "areaDesc": "Co. of Two Hills near Myrnam and Derwent",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810051"
          }
        ]
      },
      {
        "areaDesc": "Co. of Minburn near Innisfree Lavoy and Ranfurly",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810031"
          }
        ]
      },
      {
        "areaDesc": "Co. of Minburn near Minburn and Mannville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810034"
          }
        ]
      },
      {
        "areaDesc": "Beaver Co. near Viking and Kinsella",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075230"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810022"
          }
        ]
      },
      {
        "areaDesc": "Flagstaff Co. near Killam and Sedgewick",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807044"
          }
        ]
      },
      {
        "areaDesc": "Flagstaff Co. near Lougheed and Hardisty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075242"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807048"
          }
        ]
      },
      {
        "areaDesc": "Flagstaff Co. near Alliance and Bellshill Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075243"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807032"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Vermilion",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810042"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Islay and McNabb Sanctuary",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Dewberry and Clandonald",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075253"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810046"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Tulliby Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075254"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810805"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Kitscoty and Marwayne",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075255"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810044"
          }
        ]
      },
      {
        "areaDesc": "Co. of Vermilion River near Paradise Valley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075256"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4810038"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Irma",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807056"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Wainwright",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807054"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Edgerton and Koroluk Landslide",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807052"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Wainwright near Chauvin Dillberry Lake and Roros",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075274"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807051"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Provost near Hughenden Amisk and Kessler",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075281"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807008"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Provost near Czar Metiskow and Cadogan",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075282"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807004"
          }
        ]
      },
      {
        "areaDesc": "M.D. of Provost near Provost and Hayter",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "075283"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4807002"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.0301200268.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:43:03.494480+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - rainfall - in effect",
      "event": "rainfall",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-22T10:11:22+00:00",
      "onset": "2026-04-22T10:05:00+00:00",
      "expires": "2026-04-23T02:11:22+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T10:15:00+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.1306487224.2026,2026-04-21T21:45:36-00:00 cap-pac@canada.ca,urn:oid:2.49.0.1.124.1623400869.2026,2026-04-22T10:11:22-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422101122306/WW_13_73_CWWG/RFW/2054248571079554968202604210501_WW_13_73_CWWG/actual/en_proper_complete_u-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17568 M:91001 C:102506"
      },
      {
        "valueName": "profile:CAP-CP:0.4:MinorChange",
        "value": "text"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - rainfall"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-23T03:00:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\nHeavy rainfall with total amounts of 15 to 25 mm is expected. \n\nRain will begin overnight on Tuesday and change over to snow Wednesday evening. With the ground still frozen over most of the area, localized flooding is more likely to occur as the ground has a reduced ability to absorb falling rain. \n\nSnow is then expected to persist over this area until Saturday afternoon with total snowfall accumulations of 10 to 20 cm likely. Additional snowfall warnings may be issued at a later date.\n\n###\n\nWater will likely pool on roads and in low-lying areas.\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": "\nAvoid low-lying areas. Watch for washouts near rivers, creeks and culverts.\n"
    },
    "areas": [
      {
        "areaDesc": "R.M. of Meadow Lake including Waterhen Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717055"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717805"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717816"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Beaver River including Pierceland and Goodsoil",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717810"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717811"
          }
        ]
      },
      {
        "areaDesc": "Green Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718090"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loon Lake including Loon Lake and Makwa",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066140"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717807"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717808"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717809"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717815"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big River including Big River and Chitek Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066150"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716863"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Frenchman Butte including St. Walburg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717801"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mervin including Turtleford Mervin and Spruce Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717803"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717804"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Britannia including Hillmond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wilton including Lashburn Marshall and Lone Rock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eldon including Maidstone and Waseca",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Manitou Lake including Marsden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713091"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.1623400869.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T01:43:05.645945+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "yellow warning - rainfall - in effect",
      "event": "rainfall",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Moderate",
      "certainty": "Likely",
      "audience": "general public",
      "effective": "2026-04-22T10:11:22+00:00",
      "onset": "2026-04-22T10:05:00+00:00",
      "expires": "2026-04-23T02:11:22+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T10:11:22+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.1306487224.2026,2026-04-21T21:45:36-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "warning"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422101122306/WW_13_73_CWWG/RFW/2054248571079554968202604210501_WW_13_73_CWWG/actual/en_proper_complete_c-fr_not_present_u/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17566 M:90999 C:102503"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "yellow warning - rainfall"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WW_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-23T03:00:00.000Z"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Risk_Tiered_Ranking",
        "value": "8"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Impact",
        "value": "moderate"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:MSC_Confidence",
        "value": "high"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Colour",
        "value": "yellow"
      }
    ],
    "text": {
      "description": "\n###\n\nWater will likely pool on roads and in low-lying areas.\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": "\nAvoid low-lying areas. Watch for washouts near rivers, creeks and culverts.\n"
    },
    "areas": [
      {
        "areaDesc": "R.M. of Meadow Lake including Waterhen Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717055"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717805"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717816"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Beaver River including Pierceland and Goodsoil",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717810"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717811"
          }
        ]
      },
      {
        "areaDesc": "Green Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718090"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loon Lake including Loon Lake and Makwa",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066140"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717807"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717808"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717809"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717815"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big River including Big River and Chitek Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066150"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716863"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Frenchman Butte including St. Walburg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717801"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mervin including Turtleford Mervin and Spruce Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717803"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717804"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Britannia including Hillmond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wilton including Lashburn Marshall and Lone Rock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eldon including Maidstone and Waseca",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Manitou Lake including Marsden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713091"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.1699486493.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T02:27:05.217640+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "special weather statement in effect",
      "event": "weather",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Minor",
      "certainty": "Possible",
      "audience": "general public",
      "effective": "2026-04-22T18:08:20+00:00",
      "onset": "2026-04-22T18:00:00+00:00",
      "expires": "2026-04-23T10:08:20+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T18:08:20+00:00",
      "status": "Actual",
      "msgType": "Update",
      "scope": "Public",
      "references": "cap-pac@canada.ca,urn:oid:2.49.0.1.124.3129929343.2026,2026-04-22T10:52:32-00:00",
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "statement"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422180151285/WS_13_73_CWWG/SPS/55244495581609489202604220501_WS_13_73_CWWG/actual/en_proper_complete_c-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17584 M:91062 C:102568"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "special weather statement"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WS_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-24T03:10:00.000Z"
      }
    ],
    "text": {
      "description": "\nA major late season storm is poised to affect Saskatchewan beginning later today and lasting into the weekend.\n\nPrecipitation will initially start as rain for portions of western and southern Saskatchewan before transitioning to snow overnight tonight  There will be a risk of thunderstorms, especially over southern regions this evening.\n\nThis will be a prolonged snowfall event, with snow lasting into Saturday.  Total snowfall accumulations of 10 to 15 cm are possible.  Snowfall warnings may be required as confidence increases closer to the event.\n\nThere will also be a risk of freezing rain tonight and Thursday, particularly over east-central regions.\n\nStrong northerly winds will give reduced visibilities in falling snow.\n\nCooler, below seasonal weather is forecast in the wake of this system into next week.\n\n###\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": ""
    },
    "areas": [
      {
        "areaDesc": "R.M. of Lacadena including Kyle Tyner and Sanctuary",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Victory including Beechy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707063"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Canaan including Lucky Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Riverside including Cabri Pennant and Success",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Saskatchewan Landing including Stewart Valley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Webb including Webb and Antelope lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Swift Current including Swift Current and Wymark",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lac Pelletier including Blumenhof",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061225"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704061"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Excelsior including Waldeck Rush Lake and Main Centre",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707020"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707023"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Morse including Herbert Morse Ernfold and Gouldtown",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Coulee including Neidpath and McMahon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707018"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lawtonia including Hodgeville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Glen Bain including Glen Bain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061235"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Whiska Creek including Vanguard Neville and Pambrun",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061236"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Maple Bush including Riverhurst and Douglas Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707074"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Huron including Tugaske",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707077"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Enfield including Central Butte",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eyebrow including Eyebrow and Brownlee",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062214"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Craik including Craik and Aylesbury",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707093"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sarnia including Holdfast Chamberlain and Dilke",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Marquis including Tuxford Keeler and Buffalo Pound",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707051"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Dufferin including Bethune and Findlater",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706081"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Chaplin including Chaplin",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707031"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wheatlands including Mortlach and Parkbeg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707034"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Shamrock including Shamrock and Kelstern",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rodgers including Coderre and Courval",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Caron including Caronport and Caron",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707037"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pense including Pense Belle Plaine and Stony Beach",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062243"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706023"
          }
        ]
      },
      {
        "areaDesc": "City of Moose Jaw",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062244"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hillsborough including Crestwynd and Old Wives lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062245"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Redburn including Rouleau and Hearne",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062246"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Baildon including Briercrest",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062247"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707001"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Moose Jaw east of Highway 2 including Pasqua",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062248"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Moose Jaw west of Highway 2 including Bushell Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062249"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hudson Bay including Shoal Lake and Red Earth Reserves",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714839"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714840"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714846"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hudson Bay including Hudson Bay and Reserve",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Porcupine including Porcupine Plain and Weekes",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hazel Dell including Lintlaw and Hazel Dell",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709061"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Preeceville including Preeceville and Sturgis",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Invermay including Invermay and Rama",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buchanan including Buchanan Amsterdam and Tadmore",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064214"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709053"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Insinger including Theodore and Sheho",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Good Lake including Canora and Good Spirit Lake Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Keys including The Key Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709821"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709830"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709832"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Philips including Pelly and St Philips",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709820"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709822"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709826"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sliding Hills including Veregin Mikado and Hamton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709033"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Cote including Kamsack Togo and Duck Mountain Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Clayton including Norquay Stenen and Swan Plain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709069"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Livingston including Arran",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064242"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709076"
          }
        ]
      },
      {
        "areaDesc": "City of Saskatoon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065100"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Spiritwood including Spiritwood and Leoville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716862"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716880"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716882"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716894"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Canwood including Debden and Big River Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716858"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716860"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meeting Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716870"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716872"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Leask including Leask Mistawasis Res. and Parkside",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716854"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716855"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716886"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716888"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716890"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716891"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Shellbrook including Sturgeon Lake Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065230"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716856"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Duck Lake including Duck Lake and Beardy's Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065240"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715859"
          }
        ]
      },
      {
        "areaDesc": "District of Lakeland including Emma Lake and Anglin Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715076"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Paddockwood including Candle Lake and Paddockwood",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715070"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715098"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715099"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715851"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715853"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716857"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buckland including Wahpeton Res. and Spruce Home",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065261"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715085"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715094"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715848"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Garden River including Meath Park and Albertville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065262"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715085"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715092"
          }
        ]
      },
      {
        "areaDesc": "City of Prince Albert",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065263"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715846"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prince Albert including Davis",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Birch Hills including Muskoday Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715847"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Louis including One Arrow Res. and Domremy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715844"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715861"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Torch River including Choiceland and White Fox",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065310"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714093"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715849"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Nipawin including Nipawin Aylsham and Pontrilas",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065321"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714073"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714076"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Moose Range including Carrot River and Tobin Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065322"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kinistino including Kinistino and James Smith Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065331"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715849"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715850"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Invergordon including Yellow Creek and Tarnopol",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065332"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Flett's Springs including Beatty Ethelton and Pathlow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065333"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715052"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Three Lakes including Middle Lake and St Benedict",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065334"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715047"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lake Lenore including St. Brieux",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065335"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Willow Creek including Gronlid and Fairy Glen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065341"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714053"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Connaught including Ridgedale and New Osgoode",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065342"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Star City including Melfort and Star City",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065343"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714051"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Tisdale including Tisdale Eldersley and Sylvania",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065344"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pleasantdale including Naicam and Pleasantdale",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065345"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714030"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714035"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714842"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Barrier Valley including Archerwill and McKague",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065346"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Arborfield including Arborfield and Zenon Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065351"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bjorkdale including Greenwater Lake Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065352"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Redberry including Hafford and Krydor",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065411"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Blaine Lake including Blaine Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065412"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Great Bend including Radisson and Borden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065413"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716011"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Laird including Waldheim Hepburn and Laird",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065421"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosthern including Rosthern Hague and Carlton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065422"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715859"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eagle Creek including Arelee and Sonningdale",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065431"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Perdue including Perdue and Kinley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065433"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712050"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712052"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Vanscoy including Delisle Asquith and Vanscoy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065434"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Corman Park northeast of the Yellowhead Highway incl. Martensville Warman and Langham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065435"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711070"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711073"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711075"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Corman Park southwest of the Yellowhead Highway",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065436"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Aberdeen including Aberdeen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065441"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fish Creek including Alvena",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065442"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715857"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715862"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hoodoo including Wakaw and Cudworth",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065443"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715043"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grant including Vonda Prud'homme and Smuts",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065444"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715017"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bayne including Bruno Peterson and Dana",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065445"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Colonsay including Colonsay and Meacham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065451"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711078"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711079"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Viscount including Viscount and Plunkett",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065452"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Blucher including Allan Clavet Bradwell and Elstow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065453"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711069"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711077"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Harris including Harris and Tessier",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065511"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Montrose including Donovan and Swanson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065512"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Milden including Dinsmore Milden and Wiseton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065513"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712012"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fertile Valley including Conquest Macrorie and Bounty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065514"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712020"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of King George northwest of Lucky Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065515"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Coteau including Birsay and Danielson Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065516"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Dundurn including Dundurn and Blackstrap Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065521"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711828"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rudy including Outlook and Glenside",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065522"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosedale including Hanley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065523"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loreburn including Elbow Loreburn and Hawarden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065524"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711024"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lost River including South Allan and the Allan Hills",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065531"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Morris including Watrous Young and Zelma",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065532"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of McCraney including Kenaston and Bladworth",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065533"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wood Creek including Simpson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065534"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Arm River including Davidson and Girvin",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065541"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711014"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Willner west of Davidson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065542"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big Arm including Imperial and Liberty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065543"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Humboldt including Humboldt Carmel and Fulda",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065611"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Peter including Muenster and Lake Lenore",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065612"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715005"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715006"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wolverine including Burr",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065613"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711096"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Leroy including Leroy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065614"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Spalding including Spalding",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065621"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Ponass Lake including Rose Valley Fosston and Nora",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065622"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714025"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lakeside including Watson and Quill Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065623"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lakeview including Wadena and Clair",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065624"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Usborne including Lanigan Drake and Guernsey",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065631"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prairie Rose including Jansen and Esk",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065632"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wreford including Nokomis and Venn",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065633"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mount Hope including Semans",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065634"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710024"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Last Mountain Valley including Govan and Duval",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065635"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big Quill including Wynyard Dafoe and Kandahar",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065641"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710828"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Elfros including Elfros Leslie and Mozart",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065642"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710043"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kutawa including Raymore Punnichy and Poorman Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065643"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710824"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710825"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Emerald including Wishart and Bankend",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065644"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710852"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Touchwood including Serath and Touchwood Hills Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065645"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710823"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kellross including Kelliher and Lestock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065646"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710012"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710822"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710832"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710834"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710836"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710838"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710840"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710842"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710843"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710844"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710846"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710847"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710848"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710849"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710850"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710851"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kelvington including Yellowquill Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065651"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714841"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sasman including Margo Kylemore and Nut Mountain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065652"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Foam Lake including Foam Lake and Fishing Lake Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065653"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710035"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710826"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710854"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Ituna Bon Accord including Ituna and Hubbard",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065654"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meadow Lake including Waterhen Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717055"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717805"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717816"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Beaver River including Pierceland and Goodsoil",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717810"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717811"
          }
        ]
      },
      {
        "areaDesc": "Green Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718090"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loon Lake including Loon Lake and Makwa",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066140"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717807"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717808"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717809"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717815"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big River including Big River and Chitek Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066150"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716863"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Frenchman Butte including St. Walburg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717801"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mervin including Turtleford Mervin and Spruce Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717803"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717804"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Turtle River including Edam and Vawn",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717011"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Britannia including Hillmond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wilton including Lashburn Marshall and Lone Rock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eldon including Maidstone and Waseca",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Paynton including Paynton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717825"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Manitou Lake including Marsden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713091"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hillsdale including Neilburg and Baldwinton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Senlac including Senlac",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713078"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Round Valley including Unity",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713074"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Cut Knife including Cut Knife",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713096"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713098"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713835"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713836"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Battle River including Sweet Grass Res. and Delmas",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066242"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712078"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712829"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712830"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712832"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712833"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712837"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buffalo including Phippen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066243"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Parkdale including Glaslyn and Fairholme",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717048"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Medstead including Medstead Belbutte and Birch Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716861"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meota including Meota and The Battlefords Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066253"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717005"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717812"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717813"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Round Hill including Rabbit Lake and Whitkow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066254"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716033"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716034"
          }
        ]
      },
      {
        "areaDesc": "The Battlefords",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066260"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of North Battleford northwest of The Battlefords",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Douglas including Speers Richard and Alticane",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716892"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mayfield including Maymont Denholm and Fielding",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716005"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Glenside north of Biggar",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066280"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eye Hill including Macklin Denzil and Evesham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066311"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grass Lake including Salvador and Reward",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066312"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713056"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Heart's Hill including Cactus Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066313"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713046"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Progress including Kerrobert and Luseland",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066314"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Tramping Lake including Scott and Revenue",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066321"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Reford including Wilkie Landis and Leipzig",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066322"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mariposa including Tramping Lake and Broadacres",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066323"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grandview including Handel and Kelfield",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066324"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713033"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosemount including Cando and Traynor",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066331"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Biggar including Biggar and Springwater",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066332"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712046"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Antelope Park including Loverna and Hoosier",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066341"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prairiedale including Major and Smiley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066342"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Milton including Alsask and Marengo",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066343"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713014"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Oakdale including Coleville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066351"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Winslow including Dodsland and Plenty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066352"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713031"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kindersley including Kindersley Brock and Flaxcombe",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066353"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mountain View including Herschel and Stranraer",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066361"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Marriott south of Biggar",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066362"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712034"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pleasant Valley including McGee and Fiske",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066363"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712001"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Andrews including Rosetown and Zealandia",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066364"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Chesterfield including Eatonia and Mantario",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066371"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Newcombe including Glidden and Madison",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066372"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708071"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Monet including Elrose Wartime and Forgan",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066381"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Snipe Lake including Eston and Plato",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066382"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708076"
          }
        ]
      }
    ]
  },
  {
    "identifier": "urn:oid:2.49.0.1.124.3129929343.2026",
    "feed_id": "sk-saskatoon",
    "received_at": "2026-04-23T02:27:40.672349+00:00",
    "metadata": {
      "language": "en-CA",
      "senderName": "Environment Canada",
      "headline": "special weather statement in effect",
      "event": "weather",
      "category": "",
      "responseType": "",
      "urgency": "Future",
      "severity": "Minor",
      "certainty": "Possible",
      "audience": "general public",
      "effective": "2026-04-22T10:52:32+00:00",
      "onset": "2026-04-22T10:47:00+00:00",
      "expires": "2026-04-23T02:52:32+00:00",
      "web": "https://weather.gc.ca/",
      "eventCodes": [
        {
          "valueName": "",
          "value": ""
        }
      ]
    },
    "source": {
      "sender": "cap-pac@canada.ca",
      "sent": "2026-04-22T10:52:32+00:00",
      "status": "Actual",
      "msgType": "Alert",
      "scope": "Public",
      "references": null,
      "sourceStr": "Env. Can. - Can. Met. Ctr. \u2013 Montr\u00e9al",
      "note": "Service Notice \u2013 February 2026: The Environment and Climate Change Canada (ECCC) CAP Service undergoes changes from time to time as the business of alerting evolves. For 2026, changes are expected to include... 1) ECCC's initiative to introduce free-form polygons to represent the true threat area of a weather hazard is tentatively set for Spring 2026 deployment. Data changes will appear in CAP following this deployment and 2) other minor improvements and corrections. For more information on these changes: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/ | Notification de service \u2013 f\u00e9vrier 2026: Le service du PAC d\u2019Environnement et Changement climatique Canada (ECCC) subit p\u00e9riodiquement des changements \u00e0 mesure que le syst\u00e8me d\u2019alerte \u00e9volue. Pour 2026, il y aura des changements incluant... 1) l\u2019initiative d\u2019ECCC visant \u00e0 introduire des polygones libres pour repr\u00e9senter la v\u00e9ritable zone de menace d\u2019un danger m\u00e9t\u00e9orologique devrait \u00eatre d\u00e9ploy\u00e9e provisoirement au printemps 2026. Des changements de donn\u00e9es appara\u00eetront dans le PAC \u00e0 la suite de ce d\u00e9ploiement et 2) d\u2019autres am\u00e9liorations et corrections mineures. Pour plus d\u2019informations sur ces changements: https://comm.collab.science.gc.ca/mailman3/hyperkitty/list/dd_info@comm.collab.science.gc.ca/"
    },
    "parameters": [
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Type",
        "value": "statement"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Broadcast_Intrusive",
        "value": "no"
      },
      {
        "valueName": "layer:SOREM:1.0:Broadcast_Immediately",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Parent_URI",
        "value": "msc/alert/environment/hazard/alert-3.0-ascii/consolidated-xml-2.0/20260422104753389/WS_13_73_CWWG/SPS/55244495581609489202604220501_WS_13_73_CWWG/actual/en_proper_complete_c-fr_proper_complete_c/NinJo"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:CAP_count",
        "value": "A:17570 M:91004 C:102510"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Name",
        "value": "special weather statement"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.0:Alert_Coverage",
        "value": "Southern Saskatchewan"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Designation_Code",
        "value": "WS_13_73_CWWG"
      },
      {
        "valueName": "layer:SOREM:2.0:WirelessImmediate",
        "value": "No"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Alert_Location_Status",
        "value": "active"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Newly_Active_Areas",
        "value": "061211,061212,061213,061221,061222,061223,061224,061225,061231,061232,061233,061234,061235,061236,062211,062212,062213,062214,062221,062222,062223,062224,062231,062232,062233,062234,062241,062243,062244,062245,062246,062247,062248,062249,064110,064120,064130,064211,064212,064213,064214,064221,064222,064231,064232,064233,064234,064241,064242,065100,065211,065212,065221,065222,065230,065240,065251,065252,065261,065262,065263,065271,065272,065273,065310,065321,065322,065331,065332,065333,065334,065335,065341,065342,065343,065344,065345,065346,065351,065352,065411,065412,065413,065421,065422,065431,065433,065434,065435,065436,065441,065442,065443,065444,065445,065451,065452,065453,065511,065512,065513,065514,065515,065516,065521,065522,065523,065524,065531,065532,065533,065534,065541,065542,065543,065611,065612,065613,065614,065621,065622,065623,065624,065631,065632,065633,065634,065635,065641,065642,065643,065644,065645,065646,065651,065652,065653,065654,066110,066120,066130,066140,066150,066211,066212,066213,066221,066222,066223,066224,066231,066232,066233,066234,066241,066242,066243,066251,066252,066253,066254,066260,066271,066272,066273,066280,066311,066312,066313,066314,066321,066322,066323,066324,066331,066332,066341,066342,066343,066351,066352,066353,066361,066362,066363,066364,066371,066372,066381,066382"
      },
      {
        "valueName": "layer:EC-MSC-SMC:1.1:Event_End_Time",
        "value": "2026-04-23T19:57:00.000Z"
      }
    ],
    "text": {
      "description": "\nA major late season storm is poised to impact Saskatchewan beginning Wednesday and lasting into the weekend.\n\nPrecipitation will initially start as rain on Wednesday for portions of western and southern Saskatchewan before transitioning to snow overnight into Thursday morning.  There will be a risk of thunderstorms, especially over southern regions on Wednesday evening.\n\nThis will be a prolonged snowfall event, with snow lasting into Saturday.  Total snowfall accumulations of 10 to 15 cm are possible.  Snowfall warnings may be required as confidence increases closer to the event.\n\nThere will also be a risk of freezing rain on Wednesday night into THursday, particularly over east-central regions.\n\nStrong northerly winds will give reduced visibilities in falling snow.\n\nCooler, below seasonal weather is forecast in the wake of this system into next week.\n\n###\n\nPlease continue to monitor alerts and forecasts issued by Environment Canada. To report severe weather, send an email to SKstorm@ec.gc.ca, call 1-800-239-0484 or post reports on X using #SKStorm.\n",
      "instruction": ""
    },
    "areas": [
      {
        "areaDesc": "R.M. of Lacadena including Kyle Tyner and Sanctuary",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Victory including Beechy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707063"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Canaan including Lucky Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Riverside including Cabri Pennant and Success",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Saskatchewan Landing including Stewart Valley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Webb including Webb and Antelope lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Swift Current including Swift Current and Wymark",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lac Pelletier including Blumenhof",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061225"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4704061"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Excelsior including Waldeck Rush Lake and Main Centre",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707020"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707023"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Morse including Herbert Morse Ernfold and Gouldtown",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Coulee including Neidpath and McMahon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707018"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lawtonia including Hodgeville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Glen Bain including Glen Bain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061235"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Whiska Creek including Vanguard Neville and Pambrun",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "061236"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4703062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Maple Bush including Riverhurst and Douglas Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707074"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Huron including Tugaske",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707077"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Enfield including Central Butte",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eyebrow including Eyebrow and Brownlee",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062214"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Craik including Craik and Aylesbury",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707093"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sarnia including Holdfast Chamberlain and Dilke",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Marquis including Tuxford Keeler and Buffalo Pound",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707051"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Dufferin including Bethune and Findlater",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706081"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Chaplin including Chaplin",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707031"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wheatlands including Mortlach and Parkbeg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707034"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Shamrock including Shamrock and Kelstern",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rodgers including Coderre and Courval",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Caron including Caronport and Caron",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707037"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pense including Pense Belle Plaine and Stony Beach",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062243"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706023"
          }
        ]
      },
      {
        "areaDesc": "City of Moose Jaw",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062244"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hillsborough including Crestwynd and Old Wives lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062245"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Redburn including Rouleau and Hearne",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062246"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4706019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Baildon including Briercrest",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062247"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707001"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Moose Jaw east of Highway 2 including Pasqua",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062248"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Moose Jaw west of Highway 2 including Bushell Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "062249"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hudson Bay including Shoal Lake and Red Earth Reserves",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714839"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714840"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714846"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hudson Bay including Hudson Bay and Reserve",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Porcupine including Porcupine Plain and Weekes",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hazel Dell including Lintlaw and Hazel Dell",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709061"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Preeceville including Preeceville and Sturgis",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Invermay including Invermay and Rama",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buchanan including Buchanan Amsterdam and Tadmore",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064214"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709053"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Insinger including Theodore and Sheho",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Good Lake including Canora and Good Spirit Lake Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Keys including The Key Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709821"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709830"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709832"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Philips including Pelly and St Philips",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709820"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709822"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709826"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sliding Hills including Veregin Mikado and Hamton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709033"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Cote including Kamsack Togo and Duck Mountain Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709037"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Clayton including Norquay Stenen and Swan Plain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709069"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Livingston including Arran",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "064242"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4709076"
          }
        ]
      },
      {
        "areaDesc": "City of Saskatoon",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065100"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Spiritwood including Spiritwood and Leoville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716862"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716880"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716882"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716894"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Canwood including Debden and Big River Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716858"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716860"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meeting Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716870"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716872"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Leask including Leask Mistawasis Res. and Parkside",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716854"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716855"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716886"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716888"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716890"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716891"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Shellbrook including Sturgeon Lake Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065230"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716856"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Duck Lake including Duck Lake and Beardy's Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065240"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715859"
          }
        ]
      },
      {
        "areaDesc": "District of Lakeland including Emma Lake and Anglin Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715076"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Paddockwood including Candle Lake and Paddockwood",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715070"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715098"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715099"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715851"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715853"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716857"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buckland including Wahpeton Res. and Spruce Home",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065261"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715085"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715094"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715848"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Garden River including Meath Park and Albertville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065262"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715085"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715092"
          }
        ]
      },
      {
        "areaDesc": "City of Prince Albert",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065263"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715846"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prince Albert including Davis",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Birch Hills including Muskoday Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715847"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Louis including One Arrow Res. and Domremy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715844"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715861"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Torch River including Choiceland and White Fox",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065310"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714093"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715849"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Nipawin including Nipawin Aylsham and Pontrilas",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065321"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714073"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714076"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Moose Range including Carrot River and Tobin Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065322"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kinistino including Kinistino and James Smith Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065331"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715849"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715850"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Invergordon including Yellow Creek and Tarnopol",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065332"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Flett's Springs including Beatty Ethelton and Pathlow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065333"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715052"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Three Lakes including Middle Lake and St Benedict",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065334"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715044"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715047"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lake Lenore including St. Brieux",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065335"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Willow Creek including Gronlid and Fairy Glen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065341"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714053"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Connaught including Ridgedale and New Osgoode",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065342"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Star City including Melfort and Star City",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065343"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714051"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Tisdale including Tisdale Eldersley and Sylvania",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065344"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714043"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pleasantdale including Naicam and Pleasantdale",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065345"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714030"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714035"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714842"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Barrier Valley including Archerwill and McKague",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065346"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714036"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Arborfield including Arborfield and Zenon Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065351"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bjorkdale including Greenwater Lake Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065352"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Redberry including Hafford and Krydor",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065411"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Blaine Lake including Blaine Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065412"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Great Bend including Radisson and Borden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065413"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716011"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Laird including Waldheim Hepburn and Laird",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065421"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosthern including Rosthern Hague and Carlton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065422"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715859"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eagle Creek including Arelee and Sonningdale",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065431"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Perdue including Perdue and Kinley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065433"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712050"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712052"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Vanscoy including Delisle Asquith and Vanscoy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065434"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Corman Park northeast of the Yellowhead Highway incl. Martensville Warman and Langham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065435"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711070"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711073"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711075"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Corman Park southwest of the Yellowhead Highway",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065436"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711065"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Aberdeen including Aberdeen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065441"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fish Creek including Alvena",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065442"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715857"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715862"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hoodoo including Wakaw and Cudworth",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065443"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715043"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grant including Vonda Prud'homme and Smuts",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065444"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715016"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715017"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Bayne including Bruno Peterson and Dana",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065445"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715012"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Colonsay including Colonsay and Meacham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065451"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711078"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711079"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Viscount including Viscount and Plunkett",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065452"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711091"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Blucher including Allan Clavet Bradwell and Elstow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065453"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711069"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711077"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Harris including Harris and Tessier",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065511"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Montrose including Donovan and Swanson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065512"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Milden including Dinsmore Milden and Wiseton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065513"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712012"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Fertile Valley including Conquest Macrorie and Bounty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065514"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712020"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of King George northwest of Lucky Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065515"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707066"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Coteau including Birsay and Danielson Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065516"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4707068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Dundurn including Dundurn and Blackstrap Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065521"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711060"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711828"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rudy including Outlook and Glenside",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065522"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosedale including Hanley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065523"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711032"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loreburn including Elbow Loreburn and Hawarden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065524"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711024"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lost River including South Allan and the Allan Hills",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065531"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711059"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Morris including Watrous Young and Zelma",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065532"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of McCraney including Kenaston and Bladworth",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065533"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wood Creek including Simpson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065534"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711041"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Arm River including Davidson and Girvin",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065541"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711014"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Willner west of Davidson",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065542"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big Arm including Imperial and Liberty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065543"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Humboldt including Humboldt Carmel and Fulda",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065611"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715007"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Peter including Muenster and Lake Lenore",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065612"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715005"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4715006"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wolverine including Burr",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065613"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711096"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Leroy including Leroy",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065614"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710058"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Spalding including Spalding",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065621"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Ponass Lake including Rose Valley Fosston and Nora",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065622"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714025"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lakeside including Watson and Quill Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065623"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710061"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710064"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Lakeview including Wadena and Clair",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065624"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Usborne including Lanigan Drake and Guernsey",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065631"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711048"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711049"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prairie Rose including Jansen and Esk",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065632"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wreford including Nokomis and Venn",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065633"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mount Hope including Semans",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065634"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710024"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Last Mountain Valley including Govan and Duval",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065635"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4711004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big Quill including Wynyard Dafoe and Kandahar",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065641"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710046"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710828"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Elfros including Elfros Leslie and Mozart",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065642"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710043"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kutawa including Raymore Punnichy and Poorman Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065643"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710824"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710825"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Emerald including Wishart and Bankend",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065644"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710031"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710852"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Touchwood including Serath and Touchwood Hills Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065645"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710823"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kellross including Kelliher and Lestock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065646"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710009"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710012"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710822"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710832"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710834"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710836"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710838"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710840"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710842"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710843"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710844"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710845"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710846"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710847"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710848"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710849"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710850"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710851"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kelvington including Yellowquill Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065651"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4714841"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Sasman including Margo Kylemore and Nut Mountain",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065652"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710071"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Foam Lake including Foam Lake and Fishing Lake Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065653"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710035"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710826"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710854"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Ituna Bon Accord including Ituna and Hubbard",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "065654"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4710004"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meadow Lake including Waterhen Res.",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066110"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717052"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717054"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717055"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717805"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717806"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717816"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717819"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Beaver River including Pierceland and Goodsoil",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066120"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717066"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717810"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717811"
          }
        ]
      },
      {
        "areaDesc": "Green Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066130"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4718090"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Loon Lake including Loon Lake and Makwa",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066140"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717056"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717057"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717058"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717807"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717808"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717809"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717815"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Big River including Big River and Chitek Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066150"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716075"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716077"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716863"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Frenchman Butte including St. Walburg",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066211"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717034"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717036"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717801"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717802"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717820"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mervin including Turtleford Mervin and Spruce Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066212"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717039"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717045"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717803"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717804"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Turtle River including Edam and Vawn",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066213"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717011"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Britannia including Hillmond",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066221"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Wilton including Lashburn Marshall and Lone Rock",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066222"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717022"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eldon including Maidstone and Waseca",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066223"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717017"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717018"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717019"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Paynton including Paynton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066224"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717013"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717014"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717825"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Manitou Lake including Marsden",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066231"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713091"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Hillsdale including Neilburg and Baldwinton",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066232"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Senlac including Senlac",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066233"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713076"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713078"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Round Valley including Unity",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066234"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713072"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713074"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Cut Knife including Cut Knife",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066241"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713096"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713098"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713835"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713836"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Battle River including Sweet Grass Res. and Delmas",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066242"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712078"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712829"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712830"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712832"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712833"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712837"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Buffalo including Phippen",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066243"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713068"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Parkdale including Glaslyn and Fairholme",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066251"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717047"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717048"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Medstead including Medstead Belbutte and Birch Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066252"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716062"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716063"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716861"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Meota including Meota and The Battlefords Prov. Park",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066253"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717001"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717005"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717812"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4717813"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Round Hill including Rabbit Lake and Whitkow",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066254"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716033"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716034"
          }
        ]
      },
      {
        "areaDesc": "The Battlefords",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066260"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712079"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716027"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716029"
          }
        ]
      },
      {
        "areaDesc": "R.M. of North Battleford northwest of The Battlefords",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066271"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716028"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Douglas including Speers Richard and Alticane",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066272"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716023"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716026"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716892"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mayfield including Maymont Denholm and Fielding",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066273"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716003"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4716005"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Glenside north of Biggar",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066280"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Eye Hill including Macklin Denzil and Evesham",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066311"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713049"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713051"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713053"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713054"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grass Lake including Salvador and Reward",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066312"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713056"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Heart's Hill including Cactus Lake",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066313"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713046"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Progress including Kerrobert and Luseland",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066314"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713041"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713044"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Tramping Lake including Scott and Revenue",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066321"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713059"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713062"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Reford including Wilkie Landis and Leipzig",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066322"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713064"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713067"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713069"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mariposa including Tramping Lake and Broadacres",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066323"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713038"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713039"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Grandview including Handel and Kelfield",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066324"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713032"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713033"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Rosemount including Cando and Traynor",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066331"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712072"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Biggar including Biggar and Springwater",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066332"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712042"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712046"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Antelope Park including Loverna and Hoosier",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066341"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713016"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Prairiedale including Major and Smiley",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066342"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713019"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713021"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713022"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Milton including Alsask and Marengo",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066343"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713011"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713014"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Oakdale including Coleville",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066351"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713024"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713026"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Winslow including Dodsland and Plenty",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066352"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713028"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713029"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713031"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Kindersley including Kindersley Brock and Flaxcombe",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066353"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713002"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713008"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4713009"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Mountain View including Herschel and Stranraer",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066361"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712038"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Marriott south of Biggar",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066362"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712034"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Pleasant Valley including McGee and Fiske",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066363"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712001"
          }
        ]
      },
      {
        "areaDesc": "R.M. of St Andrews including Rosetown and Zealandia",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066364"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712004"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712006"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4712008"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Chesterfield including Eatonia and Mantario",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066371"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708065"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708068"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Newcombe including Glidden and Madison",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066372"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708071"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Monet including Elrose Wartime and Forgan",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066381"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708092"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708094"
          }
        ]
      },
      {
        "areaDesc": "R.M. of Snipe Lake including Eston and Plato",
        "polygon": null,
        "geocodes": [
          {
            "valueName": "layer:EC-MSC-SMC:1.0:CLC",
            "value": "066382"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708074"
          },
          {
            "valueName": "profile:CAP-CP:Location:0.3",
            "value": "4708076"
          }
        ]
      }
    ]
  }
]
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
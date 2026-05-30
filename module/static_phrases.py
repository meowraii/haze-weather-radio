from __future__ import annotations

import hashlib
import copy
import json
import logging
import pathlib
import wave
from datetime import datetime
from typing import Any
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

from module.buffer import CHANNELS, SAMPLE_RATE
from module.packages import Package_Config
from module.tts import synthesize_pcm

log = logging.getLogger(__name__)

_CACHE_PATH = pathlib.Path('managed') / 'staticPhrases.json'
_STATIC_ROOT = pathlib.Path('audio') / 'static'
_READERS_PATH = pathlib.Path('managed') / 'configs' / 'readers.xml'

_HOURS_EN: dict[int, str] = {
    1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
    11: 'eleven', 12: 'twelve',
}

_MINUTE_ONES = [
    'zero', 'one', 'two', 'three', 'four', 'five', 'six',
    'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
    'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen',
    'eighteen', 'nineteen',
]
_MINUTE_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty']

_ORDINALS: list[str] = [
    '', 'first', 'second', 'third', 'fourth', 'fifth', 'sixth',
    'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth',
    'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth',
    'eighteenth', 'nineteenth', 'twentieth', 'twenty-first', 'twenty-second',
    'twenty-third', 'twenty-fourth', 'twenty-fifth', 'twenty-sixth',
    'twenty-seventh', 'twenty-eighth', 'twenty-ninth', 'thirtieth', 'thirty-first',
]

_TZ_SPOKEN: dict[str, str] = {
    'CDT': 'Central Daylight Time',
    'CST': 'Central Standard Time',
    'MDT': 'Mountain Daylight Time',
    'MST': 'Mountain Standard Time',
    'EDT': 'Eastern Daylight Time',
    'EST': 'Eastern Standard Time',
    'PDT': 'Pacific Daylight Time',
    'PST': 'Pacific Standard Time',
    'ADT': 'Atlantic Daylight Time',
    'AST': 'Atlantic Standard Time',
    'NDT': 'Newfoundland Daylight Time',
    'NST': 'Newfoundland Standard Time',
    'UTC': 'Coordinated Universal Time',
    'GMT': 'Greenwich Mean Time',
}


def _minute_word(m: int) -> str:
    if m == 0:
        return "o'clock"
    if 1 <= m <= 9:
        return f'oh {_MINUTE_ONES[m]}'
    if m < 20:
        return _MINUTE_ONES[m]
    tens, ones = divmod(m, 10)
    return _MINUTE_TENS[tens] if ones == 0 else f'{_MINUTE_TENS[tens]} {_MINUTE_ONES[ones]}'


def _tens_ones_words(n: int) -> str:
    if n < 20:
        return _MINUTE_ONES[n]
    tens, ones = divmod(n, 10)
    return _MINUTE_TENS[tens] if ones == 0 else f'{_MINUTE_TENS[tens]}-{_MINUTE_ONES[ones]}'


def _year_to_words(year: int) -> str:
    century, remainder = divmod(year, 100)
    if remainder == 0:
        return f'{_tens_ones_words(century)} hundred'
    if century == 20:
        return f'two thousand {_tens_ones_words(remainder)}'
    if century == 19:
        return f'nineteen {_tens_ones_words(remainder)}'
    return str(year)


def _date_text_en(dt: datetime) -> str:
    return f'{dt.strftime("%A")}, {dt.strftime("%B")} {_ORDINALS[dt.day]}, {_year_to_words(dt.year)}'


def _build_en_ca_catalog() -> dict[str, str]:
    phrases: dict[str, str] = {
        'greeting.morning':   'Good morning.',
        'greeting.afternoon': 'Good afternoon.',
        'greeting.evening':   'Good evening.',
        'greeting.night':     'Good night.',
        'time.current_is':    'The current time is',
        'time.am':            'A.M.',
        'time.pm':            'P.M.',
    }
    for h, word in _HOURS_EN.items():
        phrases[f'time.hour.{h}'] = word
    for m in range(60):
        phrases[f'time.minute.{m:02d}'] = _minute_word(m)
    for abbr, spoken in _TZ_SPOKEN.items():
        phrases[f'tz.{abbr}'] = spoken
    return phrases


_LANG_CATALOGS: dict[str, dict[str, str]] = {
    'en-CA': _build_en_ca_catalog(),
}

_cache: dict[str, Any] = {}
_date_clip_cache: dict[tuple[str, str], bytes] = {}
_readers_cache: dict[str, dict[str, Any]] | None = None


def _readers() -> dict[str, dict[str, Any]]:
    global _readers_cache
    if _readers_cache is not None:
        return _readers_cache

    readers: dict[str, dict[str, Any]] = {}
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
        reader_id = str(reader_el.get('id') or '').strip()
        if not reader_id:
            continue
        provider = str(reader_el.get('provider') or 'piper').strip().lower() or 'piper'
        gender = str((reader_el.findtext('gender') or 'male')).strip().lower() or 'male'
        language = str((reader_el.findtext('language') or '')).strip()
        path = str((reader_el.findtext('path') or '')).strip()
        if not path:
            continue
        readers[reader_id] = {
            'id': reader_id,
            'provider': provider,
            'gender': 'female' if gender == 'female' else 'male',
            'language': language,
            'path': path,
        }

    _readers_cache = readers
    return readers


def _resolve_piper_reader_path(base_path: str) -> tuple[str, str | None]:
    candidate = pathlib.Path(base_path)
    model_candidates: list[pathlib.Path]

    if candidate.suffix.lower() == '.onnx':
        model_candidates = [candidate]
    else:
        model_candidates = [
            candidate,
            pathlib.Path(f'{base_path}.onnx'),
            pathlib.Path('piper-tts') / f'{base_path}.onnx',
            pathlib.Path('voices') / f'{base_path}.onnx',
        ]

    model_path: pathlib.Path = model_candidates[-1]
    for item in model_candidates:
        if item.exists():
            model_path = item
            break

    config_candidates = [
        model_path.with_suffix(model_path.suffix + '.json') if model_path.suffix else pathlib.Path(f'{model_path}.json'),
        model_path.with_suffix('.onnx.json') if model_path.suffix != '.onnx' else model_path.with_suffix('.onnx.json'),
        model_path.with_suffix('.json'),
    ]
    config_path: str | None = None
    for cfg_path in config_candidates:
        if cfg_path.exists():
            config_path = str(cfg_path)
            break

    return str(model_path), config_path


def _reader_for_package(pkg_id: str, lang: str) -> dict[str, Any] | None:
    Package_Config.ensure_loaded()
    profile = Package_Config.get_profile(pkg_id)
    reader_id = str(profile.get('reader_id') or '').strip()
    if not reader_id:
        default_profile = Package_Config.get_profile('defaults')
        reader_id = str(default_profile.get('reader_id') or '').strip()
    if not reader_id:
        return None

    reader = _readers().get(reader_id)
    if not reader:
        log.warning('Reader id %s not found in %s', reader_id, _READERS_PATH)
        return None

    reader_lang = str(reader.get('language') or '').strip().lower()
    if reader_lang and not lang.lower().startswith(reader_lang):
        log.warning('Reader %s language %s does not match phrase language %s', reader_id, reader_lang, lang)
    return reader


def _reader_tts_config(config: dict[str, Any], lang: str, reader: dict[str, Any] | None) -> dict[str, Any]:
    if not reader:
        return config

    cfg = copy.deepcopy(config)
    tts_cfg = cfg.setdefault('tts', {})
    lang_cfg = tts_cfg.setdefault('lang', {}).setdefault(lang, {}).setdefault('backend', {})
    provider = str(reader.get('provider') or 'piper').strip().lower()
    gender = str(reader.get('gender') or 'male').strip().lower()
    voice_slot = 'female' if gender == 'female' else 'male'
    reader_path = str(reader.get('path') or '').strip()

    if provider == 'pyttsx3':
        pyttsx3_cfg = tts_cfg.setdefault('pyttsx3', {})
        pyttsx3_cfg['enabled'] = True
        if reader_path:
            pyttsx3_cfg['voice'] = reader_path
            backend = lang_cfg.setdefault('pyttsx3', {})
            backend[voice_slot] = {'voice': reader_path}
        tts_cfg['fallback_order'] = ['pyttsx3']
        return cfg

    if provider == 'kokoro':
        kokoro_cfg = tts_cfg.setdefault('kokoro', {})
        if reader_path:
            kokoro_cfg['voice'] = reader_path
            backend = lang_cfg.setdefault('kokoro', {})
            backend[voice_slot] = {'voice': reader_path}
        tts_cfg['fallback_order'] = ['kokoro']
        return cfg

    model_path, model_cfg = _resolve_piper_reader_path(reader_path)
    piper_backend = lang_cfg.setdefault('piper', {})
    slot_cfg = dict(piper_backend.get(voice_slot) or {})
    slot_cfg['model'] = model_path
    if model_cfg:
        slot_cfg['config'] = model_cfg
    slot_cfg.setdefault('speaker', 0)
    piper_backend[voice_slot] = slot_cfg
    tts_cfg['fallback_order'] = ['piper']
    return cfg


def _clip_path(lang: str, key: str) -> pathlib.Path:
    parts = key.split('.')
    return _STATIC_ROOT / lang / pathlib.Path(*parts[:-1]) / f'{parts[-1]}.wav'


def _write_pcm_as_wav(path: pathlib.Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm)


def _read_wav_pcm(path: pathlib.Path) -> bytes | None:
    try:
        with wave.open(str(path), 'rb') as w:
            return w.readframes(w.getnframes())
    except Exception:
        return None


def _date_clip_disk_path(lang: str, date_str: str) -> pathlib.Path:
    return _STATIC_ROOT / lang / 'date' / f'{date_str}.wav'


def _get_date_pcm(config: dict[str, Any], lang: str, now: datetime) -> bytes | None:
    date_str = now.strftime('%Y-%m-%d')
    cache_key = (lang, date_str)

    if cache_key in _date_clip_cache:
        return _date_clip_cache[cache_key]

    path = _date_clip_disk_path(lang, date_str)
    if path.exists():
        pcm = _read_wav_pcm(path)
        if pcm:
            _date_clip_cache[cache_key] = pcm
            return pcm

    if lang == 'en-CA':
        text = _date_text_en(now)
    else:
        text = now.strftime('%A, %B ') + _ORDINALS[now.day] + ', ' + _year_to_words(now.year)

    pcm = synthesize_pcm(config, text, lang=lang)
    if not pcm:
        log.warning('Failed to synthesize daily date clip [%s] %s', lang, date_str)
        return None

    _write_pcm_as_wav(path, pcm)
    _date_clip_cache[cache_key] = pcm
    log.debug('Synthesized daily date clip [%s] %s: "%s"', lang, date_str, text)
    return pcm


def _load_cache() -> dict[str, Any]:
    if not _CACHE_PATH.exists():
        return {'tts_config': {}, 'lang_clips': {}, 'feed_clips': {}, 'feed_clip_config': {}}
    try:
        with open(_CACHE_PATH, encoding='utf-8') as f:
            data = json.load(f)
        for key in ('tts_config', 'lang_clips', 'feed_clips', 'feed_clip_config'):
            if not isinstance(data.get(key), dict):
                data[key] = {}
        return data
    except Exception:
        return {'tts_config': {}, 'lang_clips': {}, 'feed_clips': {}, 'feed_clip_config': {}}


def _save_cache(data: dict[str, Any]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def _tts_fingerprint(config: dict[str, Any], lang: str, reader: dict[str, Any] | None) -> dict[str, Any]:
    effective_cfg = _reader_tts_config(config, lang, reader)
    tts_cfg = effective_cfg.get('tts', {})
    provider = str((tts_cfg.get('fallback_order') or ['piper'])[0])
    backend_cfg = tts_cfg.get('lang', {}).get(lang, {}).get('backend', {}).get(provider, {})
    voice_slot = 'female' if str((reader or {}).get('gender') or 'male').strip().lower() == 'female' else 'male'

    if provider == 'pyttsx3':
        pyttsx3_cfg = tts_cfg.get('pyttsx3', {}) if isinstance(tts_cfg.get('pyttsx3'), dict) else {}
        voice = str(pyttsx3_cfg.get('voice', ''))
        return {
            'provider': provider,
            'voice': voice,
            'reader_id': str((reader or {}).get('id', '')),
        }

    if provider == 'kokoro':
        kokoro_cfg = tts_cfg.get('kokoro', {}) if isinstance(tts_cfg.get('kokoro'), dict) else {}
        voice_cfg = backend_cfg.get(voice_slot) or backend_cfg.get('male') or backend_cfg.get('female') or {}
        if isinstance(voice_cfg, str):
            voice_cfg = {'voice': voice_cfg}
        if not isinstance(voice_cfg, dict):
            voice_cfg = {}
        return {
            'provider': provider,
            'voice': str(voice_cfg.get('voice') or kokoro_cfg.get('voice', '')),
            'lang_code': str(voice_cfg.get('lang_code') or kokoro_cfg.get('lang_code', '')),
            'model': str(voice_cfg.get('model') or kokoro_cfg.get('model', '')),
            'config': str(voice_cfg.get('config') or kokoro_cfg.get('config', '')),
            'device': str(voice_cfg.get('device') or kokoro_cfg.get('device', '')),
            'speed': str(voice_cfg.get('speed') or kokoro_cfg.get('speed', '')),
            'reader_id': str((reader or {}).get('id', '')),
            'voice_slot': voice_slot,
        }

    voice_cfg = backend_cfg.get(voice_slot) or backend_cfg.get('male') or {}
    if not isinstance(voice_cfg, dict):
        voice_cfg = {}
    return {
        'provider': provider,
        'model': str(voice_cfg.get('model', '')),
        'config': str(voice_cfg.get('config', '')),
        'speaker': int(voice_cfg.get('speaker') or 0),
        'reader_id': str((reader or {}).get('id', '')),
        'voice_slot': voice_slot,
    }


def _langs_for_feed(feed: dict[str, Any]) -> list[str]:
    languages_cfg = feed.get('languages')
    if isinstance(languages_cfg, dict) and languages_cfg:
        return list(languages_cfg.keys())
    return [str(feed.get('language', 'en-CA'))]


def _init_lang_clips(config: dict[str, Any], cache: dict[str, Any], lang: str, catalog: dict[str, str]) -> None:
    reader = _reader_for_package('date_time', lang)
    synth_cfg = _reader_tts_config(config, lang, reader)
    fingerprint = _tts_fingerprint(config, lang, reader)
    cached_fingerprint = cache['tts_config'].get(lang, {})
    force_rebuild = cached_fingerprint != fingerprint

    if force_rebuild:
        log.info('TTS config changed for %s — full static phrase rebuild', lang)

    lang_clips = cache['lang_clips'].setdefault(lang, {})
    synthesized = 0
    ready = 0

    for key, text in catalog.items():
        path = _clip_path(lang, key)
        lang_clips[key] = str(path)

        if force_rebuild or not path.exists():
            pcm = synthesize_pcm(synth_cfg, text, lang=lang, voice=reader.get('gender') if reader else None)
            if pcm:
                _write_pcm_as_wav(path, pcm)
                synthesized += 1
                ready += 1
            else:
                log.warning('Failed to synthesize static phrase [%s] %s', lang, key)
        else:
            ready += 1

    cache['tts_config'][lang] = fingerprint
    log.info('Static phrases [%s]: %d ready, %d synthesized', lang, ready, synthesized)


def _init_feed_clips(config: dict[str, Any], cache: dict[str, Any], feed: dict[str, Any], lang: str) -> None:
    from module.packages import station_id as gen_station_id

    feed_id = feed.get('id', '')
    reader = _reader_for_package('station_id', lang)
    synth_cfg = _reader_tts_config(config, lang, reader)
    text = gen_station_id(config, feed_id, lang)
    text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
    fingerprint = {**_tts_fingerprint(config, lang, reader), 'text_hash': text_hash}

    cached = cache.get('feed_clip_config', {}).get(feed_id, {}).get(lang, {})
    sid_path = _STATIC_ROOT / 'feeds' / feed_id / lang / 'station_id.wav'

    feed_clips = cache.setdefault('feed_clips', {}).setdefault(feed_id, {}).setdefault(lang, {})
    feed_clip_cfg = cache.setdefault('feed_clip_config', {}).setdefault(feed_id, {})
    feed_clips['station_id'] = str(sid_path)

    if cached != fingerprint or not sid_path.exists():
        if cached != fingerprint:
            log.info('Station ID config/text changed for %s [%s] — rebuilding', feed_id, lang)
        pcm = synthesize_pcm(synth_cfg, text, lang=lang, voice=reader.get('gender') if reader else None)
        if pcm:
            _write_pcm_as_wav(sid_path, pcm)
            feed_clip_cfg[lang] = fingerprint
        else:
            log.warning('Failed to synthesize station_id for %s [%s]', feed_id, lang)
    else:
        feed_clip_cfg[lang] = fingerprint


def init_static_phrases(config: dict[str, Any], feeds: list[dict[str, Any]]) -> None:
    global _cache
    _cache = _load_cache()

    for lang, catalog in _LANG_CATALOGS.items():
        _init_lang_clips(config, _cache, lang, catalog)

    for feed in feeds:
        for lang in _langs_for_feed(feed):
            if lang in _LANG_CATALOGS:
                _init_feed_clips(config, _cache, feed, lang)

    _save_cache(_cache)


def get_clip_path(lang: str, key: str) -> pathlib.Path | None:
    rel = _cache.get('lang_clips', {}).get(lang, {}).get(key)
    if not rel:
        return None
    p = pathlib.Path(rel)
    return p if p.exists() else None


def get_feed_clip_path(feed_id: str, lang: str, key: str) -> pathlib.Path | None:
    rel = _cache.get('feed_clips', {}).get(feed_id, {}).get(lang, {}).get(key)
    if not rel:
        return None
    p = pathlib.Path(rel)
    return p if p.exists() else None


def splice_date_time(config: dict[str, Any], lang: str, tz_abbr: str, now: datetime) -> bytes | None:
    hour_12 = now.hour % 12 or 12
    h = now.hour
    greeting = (
        'morning' if 5 <= h < 12 else
        'afternoon' if h < 17 else
        'evening' if h < 21 else
        'night'
    )
    time_keys = [
        f'greeting.{greeting}',
        'time.current_is',
        f'time.hour.{hour_12}',
        f'time.minute.{now.minute:02d}',
        'time.am' if h < 12 else 'time.pm',
        f'tz.{tz_abbr}',
    ]

    missing = [k for k in time_keys if get_clip_path(lang, k) is None]
    if missing:
        log.warning('Cannot splice date_time [%s]: missing clips %s', lang, missing)
        return None

    time_pcm = b''.join(
        pcm for k in time_keys
        if (pcm := _read_wav_pcm(get_clip_path(lang, k))) is not None  # type: ignore[arg-type]
    )

    date_pcm = _get_date_pcm(config, lang, now)
    if date_pcm is None:
        return None

    return time_pcm + date_pcm


def splice_station_id(feed_id: str, lang: str) -> bytes | None:
    path = get_feed_clip_path(feed_id, lang, 'station_id')
    if path is None:
        return None
    return _read_wav_pcm(path)

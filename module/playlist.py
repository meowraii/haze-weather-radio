import hashlib
import json
import logging
import os
import pathlib
from typing import Any

from managed.events import data_ready, initial_synthesis_done, read_data_pool, shutdown_event, tts_queue, update_playout_sequence
from module.tts import synthesize
from managed.packages import (
    Package_Config,
    alerts_package,
    current_conditions_package,
    date_time_package,
    eccc_discussion_package,
    forecast_package,
    geophysical_alert_package,
    station_id,
    user_bulletin_package,
)

_BULLETINS_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'userbulletins.json'
_REGISTRY_PATH = pathlib.Path(__file__).parent.parent / 'data' / 'alertsRegistry.json'

log = logging.getLogger(__name__)

_last_text_hash: dict[str, str] = {}
_bulletins_cache: list[Any] = []
_bulletins_cached_mtime: float = -1.0
_registry_cache: list[Any] = []
_registry_cached_mtime: float = -1.0


def _get_bulletins_mtime() -> float:
    try:
        return os.path.getmtime(_BULLETINS_PATH)
    except OSError:
        return 0.0


def _needs_synthesis(pkg_id: str, feed_id: str, lang: str, text: str) -> bool:
    key = f"{feed_id}:{lang}:{pkg_id}"
    return _last_text_hash.get(key) != hashlib.md5(text.encode()).hexdigest()


def _mark_synthesized(pkg_id: str, feed_id: str, lang: str, text: str) -> None:
    _last_text_hash[f"{feed_id}:{lang}:{pkg_id}"] = hashlib.md5(text.encode()).hexdigest()


def _load_bulletins() -> list[Any]:
    global _bulletins_cache, _bulletins_cached_mtime
    current_mtime = _get_bulletins_mtime()
    if current_mtime != _bulletins_cached_mtime:
        try:
            with open(_BULLETINS_PATH, encoding='utf-8') as f:
                _bulletins_cache = json.load(f)
        except Exception:
            _bulletins_cache = []
        _bulletins_cached_mtime = current_mtime
    return _bulletins_cache


def _load_registry() -> list[Any]:
    global _registry_cache, _registry_cached_mtime
    try:
        current_mtime = os.path.getmtime(_REGISTRY_PATH)
    except OSError:
        return []
    if current_mtime != _registry_cached_mtime:
        try:
            with open(_REGISTRY_PATH, encoding='utf-8') as f:
                _registry_cache = json.load(f)
        except Exception:
            _registry_cache = []
        _registry_cached_mtime = current_mtime
    return _registry_cache

def playlist_thread_worker(config: dict[str, Any]) -> None:
    feeds = [f for f in config.get('feeds', []) if f.get('enabled', True)]
    first_cycle = True
    while not shutdown_event.is_set():
        data_ready.wait(timeout=10)
        data_ready.clear()
        for feed in feeds:
            if shutdown_event.is_set():
                break
            _play_feed_cycle(config, feed)
        if first_cycle:
            initial_synthesis_done.set()
            first_cycle = False


def _play_feed_cycle(config: dict[str, Any], feed: dict[str, Any]) -> None:
    feed_id = feed['id']
    tz = feed.get('timezone', 'UTC')

    languages_cfg = feed.get('languages')
    if isinstance(languages_cfg, dict) and languages_cfg:
        languages = list(languages_cfg.keys())
    else:
        languages = [feed.get('language', 'en-CA')]

    locations = feed.get('locations', [])
    primary_loc = next(
        (loc for loc in locations if isinstance(loc, dict)),
        None,
    )
    loc_name = primary_loc.get('name') if primary_loc else None
    primary_loc_id = primary_loc.get('eccc_id') if primary_loc else None

    all_locs = [loc for loc in locations if isinstance(loc, dict)]
    forecast_locs = [loc for loc in all_locs if loc.get('generate_forecast', False)]

    focn45_bulletin = read_data_pool("focn45")
    discussion_text: str | None = None
    if focn45_bulletin is not None:
        discussion_text = getattr(focn45_bulletin, 'text', None) or (
            focn45_bulletin if isinstance(focn45_bulletin, str) else None
        )

    bulletins = _load_bulletins()
    registry = _load_registry()
    raw_order = config.get('playout', {}).get('playlist_order') or []
    parsed_order: list[str] = []
    for entry in raw_order:
        if isinstance(entry, str):
            parsed_order.append(entry)
        elif isinstance(entry, dict):
            pkg_id_entry = next((k for k in entry if k != 'ttl'), None)
            if pkg_id_entry:
                parsed_order.append(pkg_id_entry)
    max_items = feed.get('playlist', {}).get('max_items', None)
    sequence: list[pathlib.Path] = []
    ready_seq: list[pathlib.Path] = []
    out_root = pathlib.Path('output') / feed_id

    for lang in languages:
        if shutdown_event.is_set():
            break

        conditions_parts: list[str] = []
        for i, loc in enumerate(all_locs):
            loc_id = loc.get('eccc_id')
            data = read_data_pool(f"{feed_id}:{loc_id}") if loc_id else None
            text = current_conditions_package(data, loc.get('name'), lang, secondary=(i > 0))
            if text:
                conditions_parts.append(text)

        forecast_parts: list[str] = []
        for loc in forecast_locs:
            loc_id = loc.get('eccc_id')
            forecast_data = read_data_pool(f"{feed_id}:forecast:{loc_id}") if loc_id else None
            part = forecast_package(forecast_data, loc.get('name'), lang)
            if part:
                forecast_parts.append(part)

        wwv_text: str | None = read_data_pool("wwv")

        pkg_lookup: dict[str, str] = {
            'date_time': date_time_package(tz, lang),
            'station_id': station_id(config, feed_id, lang),
            'current_conditions': '  '.join(conditions_parts),
            'forecast': '  '.join(forecast_parts),
            'eccc_discussion': eccc_discussion_package(discussion_text, loc_name, lang),
            'geophysical_alert': geophysical_alert_package(wwv_text),
            'user_bulletin': user_bulletin_package(bulletins, lang, tz),
            'alerts': alerts_package(registry, lang, tz, feed),
        }
        default_order = list(pkg_lookup.keys())
        playlist_order: list[str] = parsed_order or default_order

        limit = max_items if max_items is not None else len(playlist_order)
        for pkg_id in playlist_order[:limit]:
            if shutdown_event.is_set():
                break
            text = pkg_lookup.get(pkg_id, '')
            if not text:
                continue
            allowed_langs = Package_Config.per_package.get(pkg_id, {}).get('lang')
            if allowed_langs is not None and lang not in allowed_langs:
                continue
            out_wav = out_root / lang / f'{pkg_id}.wav'
            sequence.append(out_wav)
            if not _needs_synthesis(pkg_id, feed_id, lang, text):
                if out_wav.exists():
                    ready_seq.append(out_wav)
                continue
            pkg_cfg = Package_Config.per_package.get(pkg_id, {})
            use_voice: str | None = pkg_cfg.get('use_voice')
            if out_wav.exists():
                ready_seq.append(out_wav)
                tts_queue.put((feed_id, pkg_id, text, lang, use_voice))
            else:
                result = synthesize(config, text, feed_id, pkg_id, lang, use_voice)
                if result:
                    ready_seq.append(out_wav)
                    update_playout_sequence(feed_id, ready_seq[:])
            _mark_synthesized(pkg_id, feed_id, lang, text)
    update_playout_sequence(feed_id, ready_seq)

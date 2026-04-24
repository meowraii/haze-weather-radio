import json
import logging
import os
import pathlib
from typing import Any

from managed.events import NowPlayingMetadata, PlayableItem, data_ready, initial_synthesis_done, read_data_pool, shutdown_event, update_playout_sequence
from managed.packages import (
    Package_Config,
    air_quality_package,
    alerts_package,
    climate_summary_package,
    current_conditions_package,
    date_time_package,
    eccc_discussion_package,
    forecast_package,
    geophysical_alert_package,
    station_id,
    user_bulletin_package,
)
from module.feed_util import (
    air_quality_locations as _air_quality_locations,
    air_quality_name as _air_quality_name,
    climate_locations as _climate_locations,
    climate_name as _climate_name,
    current_conditions_name as _current_conditions_name,
    forecast_locations as _forecast_locations,
    forecast_name as _forecast_name,
    location_label as _location_label,
    observation_locations as _observation_locations,
)

_BULLETINS_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'userbulletins.json'
_REGISTRY_PATH = pathlib.Path(__file__).parent.parent / 'data' / 'alertsRegistry.json'

log = logging.getLogger(__name__)

_PACKAGE_METADATA_LABELS: dict[str, str] = {
    'alerts': 'Alerts',
    'date_time': 'Date and Time',
    'station_id': 'Station Identification',
    'user_bulletin': 'User Bulletin',
    'current_conditions': 'Current Conditions',
    'forecast': 'Forecast',
    'air_quality': 'Air Quality',
    'eccc_discussion': 'Discussion',
    'geophysical_alert': 'Geophysical Alert',
    'climate_summary': 'Climate Summary',
    'chime_8': 'Half-Hour Chime',
    'chime_16': 'Top-of-Hour Chime',
}

_bulletins_cache: list[Any] = []
_bulletins_cached_mtime: float = -1.0
_registry_cache: list[Any] = []
_registry_cached_mtime: float = -1.0


def _localized_name(value: Any, lang: str) -> str | None:
    if isinstance(value, dict):
        lang_short = lang[:2]
        for key in (lang, lang_short, 'en-CA', 'en', 'fr-CA', 'fr'):
            raw = value.get(key)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                return text
        for raw in value.values():
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                return text
        return None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _conditions_station_name(data: dict[str, Any] | None, lang: str) -> str | None:
    if not isinstance(data, dict):
        return None
    return _localized_name(data.get('station'), lang)


def _forecast_station_name(data: dict[str, Any] | None, lang: str) -> str | None:
    if not isinstance(data, dict):
        return None
    return _localized_name(data.get('name'), lang)


def _climate_station_name(data: dict[str, Any] | None, lang: str) -> str | None:
    if not isinstance(data, dict):
        return None
    return _localized_name(data.get('station'), lang)


def _metadata_title(
    pkg_id: str,
    area_name: str | None,
    station_name: str | None,
) -> str:
    if pkg_id == 'current_conditions' and station_name:
        return f'Conditions at {station_name}'
    package_name = _PACKAGE_METADATA_LABELS.get(pkg_id, pkg_id.replace('_', ' ').title())
    if area_name:
        return f'{area_name} Area {package_name}'
    return package_name


def _get_bulletins_mtime() -> float:
    try:
        return os.path.getmtime(_BULLETINS_PATH)
    except OSError:
        return 0.0


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
            try:
                _play_feed_cycle(config, feed)
            except Exception:
                log.exception('Playlist cycle failed for feed %s', feed.get('id', '?'))
        if first_cycle:
            initial_synthesis_done.set()
            first_cycle = False


def _feed_playout_cfg(feed: dict[str, Any]) -> dict[str, Any]:
    raw = feed.get('playout', {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        merged: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    return {}


def _play_feed_cycle(config: dict[str, Any], feed: dict[str, Any]) -> None:
    feed_id = feed['id']
    playout_cfg = _feed_playout_cfg(feed)
    if not playout_cfg.get('routine', True):
        return

    tz = feed.get('timezone', 'UTC')

    languages_cfg = feed.get('languages')
    if isinstance(languages_cfg, dict) and languages_cfg:
        languages = list(languages_cfg.keys())
    else:
        languages = [feed.get('language', 'en-CA')]

    all_locs = _observation_locations(feed)
    forecast_locs = _forecast_locations(feed)
    climate_locs = _climate_locations(feed)
    aqhi_locs = _air_quality_locations(feed)
    primary_loc = all_locs[0] if all_locs else (forecast_locs[0] if forecast_locs else None)
    loc_name = primary_loc.get('name') or primary_loc.get('id') if primary_loc else None

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
    sequence: list[PlayableItem] = []
    track_number = 0

    for lang in languages:
        if shutdown_event.is_set():
            break

        conditions_parts: list[str] = []
        current_station_name: str | None = None
        area_name: str | None = None
        for i, loc in enumerate(all_locs):
            loc_id = loc.get('id')
            data = read_data_pool(f"{feed_id}:{loc_id}") if loc_id else None
            text = current_conditions_package(data, _current_conditions_name(loc), lang, secondary=(i > 0))
            resolved_name = (
                _current_conditions_name(loc)
                or _conditions_station_name(data, lang)
                or _location_label(loc, loc_id)
            )
            if i == 0:
                current_station_name = resolved_name
            if area_name is None:
                area_name = resolved_name
            if text:
                conditions_parts.append(text)

        forecast_parts: list[str] = []
        for loc in forecast_locs:
            loc_id = loc.get('id')
            forecast_data = read_data_pool(f"{feed_id}:forecast:{loc_id}") if loc_id else None
            part = forecast_package(forecast_data, _forecast_name(loc), lang)
            if area_name is None:
                area_name = (
                    _forecast_name(loc)
                    or _forecast_station_name(forecast_data, lang)
                    or _location_label(loc, loc_id)
                )
            if part:
                forecast_parts.append(part)

        climate_parts: list[str] = []
        for loc in climate_locs:
            loc_id = loc.get('id')
            climate_data = read_data_pool(f"{feed_id}:climate:{loc_id}") if loc_id else None
            part = climate_summary_package(climate_data, _climate_name(loc), lang)
            if area_name is None:
                area_name = (
                    _climate_name(loc)
                    or _climate_station_name(climate_data, lang)
                    or _location_label(loc, loc_id)
                )
            if part:
                climate_parts.append(part)

        aqhi_parts: list[str] = []
        for loc in aqhi_locs:
            loc_id = loc.get('id')
            aqhi_data = read_data_pool(f"{feed_id}:aqhi:{loc_id}") if loc_id else None
            part = air_quality_package(aqhi_data, _air_quality_name(loc), lang)
            if part:
                aqhi_parts.append(part)

        if area_name is None:
            area_name = feed.get('name') or loc_name or feed_id
        if current_station_name is None:
            current_station_name = area_name

        wwv_text: str | None = read_data_pool("wwv")

        pkg_lookup: dict[str, str] = {
            'date_time': date_time_package(tz, lang),
            'station_id': station_id(config, feed_id, lang),
            'current_conditions': '  '.join(conditions_parts),
            'forecast': '  '.join(forecast_parts),
            'air_quality': '  '.join(aqhi_parts),
            'climate_summary': '  '.join(climate_parts),
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
            pkg_cfg = Package_Config.per_package.get(pkg_id, {})
            use_voice: str | None = pkg_cfg.get('use_voice')
            track_number += 1
            sequence.append(PlayableItem(
                pkg_id=pkg_id,
                text=text,
                lang=lang,
                voice=use_voice,
                metadata=NowPlayingMetadata(
                    title=_metadata_title(pkg_id, area_name, current_station_name),
                    track=str(track_number),
                ),
            ))

    update_playout_sequence(feed_id, sequence)

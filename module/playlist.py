from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import pathlib
import re
from typing import Any

from module.events import (
    NowPlayingMetadata,
    PlayableItem,
    append_playout_items,
    data_ready,
    get_last_played_item_id,
    get_playout_sequence,
    get_sequence_version,
    initial_synthesis_done,
    read_data_pool,
    shutdown_event,
    update_playout_sequence,
)
from module.packages import (
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

log = logging.getLogger(__name__)

_BULLETINS_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'userbulletins.json'
_REGISTRY_DIR = pathlib.Path(__file__).parent.parent / 'data' / 'alerts'
_WORD_RE = re.compile(r'\b\w+\b')

_DEFAULT_DYNAMIC_PLAYOUT_CFG: dict[str, Any] = {
    'target_depth': 8,
    'low_water_mark': 3,
    'max_fill_seconds': 180.0,
    'boundary_guard_seconds': 2.0,
}

_DEFAULT_PACING_CFG: dict[str, float] = {
    'package_gap_s': 0.35,
}

_DEFAULT_STATION_ID_SCHEDULE_CFG: dict[str, Any] = {
    'enabled': True,
    'minutes': [0, 15, 30, 45],
}

_DEFAULT_DATE_TIME_SCHEDULE_CFG: dict[str, Any] = {
    'enabled': True,
    'minutes': [5, 15, 25, 35, 45, 55],
}

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

_CHIME_DURATION_S: dict[str, float] = {
    'top': 10.0,
    'half': 5.0,
}

_bulletins_cache: list[Any] = []
_bulletins_cached_mtime = -1.0
_registry_cache: dict[str, tuple[float, list[Any]]] = {}
_scheduled_chime_boundaries: dict[str, str] = {}


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


def _feed_display_name(feed: dict[str, Any]) -> str:
    name = feed.get('name')
    if isinstance(name, str) and name.strip():
        return name.strip()
    return str(feed.get('id', 'feed'))


def _metadata_title(pkg_id: str, area_name: str | None, station_name: str | None) -> str:
    if pkg_id == 'current_conditions' and station_name:
        return f'Conditions at {station_name}'
    package_name = _PACKAGE_METADATA_LABELS.get(pkg_id, pkg_id.replace('_', ' ').title())
    if area_name:
        return f'{area_name} Area {package_name}'
    return package_name


def _load_bulletins() -> list[Any]:
    global _bulletins_cache, _bulletins_cached_mtime

    try:
        current_mtime = os.path.getmtime(_BULLETINS_PATH)
    except OSError:
        current_mtime = 0.0

    if current_mtime != _bulletins_cached_mtime:
        try:
            with open(_BULLETINS_PATH, encoding='utf-8') as file_handle:
                _bulletins_cache = json.load(file_handle)
        except Exception:
            _bulletins_cache = []
        _bulletins_cached_mtime = current_mtime

    return _bulletins_cache


def _load_registry(feed_id: str) -> list[Any]:
    path = _REGISTRY_DIR / f'{feed_id}.json'
    try:
        current_mtime = os.path.getmtime(path)
    except OSError:
        return []

    cached_mtime, cached_items = _registry_cache.get(feed_id, (-1.0, []))
    if current_mtime != cached_mtime:
        try:
            with open(path, encoding='utf-8') as file_handle:
                cached_items = json.load(file_handle)
        except Exception:
            cached_items = []
        _registry_cache[feed_id] = (current_mtime, cached_items)

    return list(cached_items)


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


def _dynamic_playout_cfg(config: dict[str, Any], feed: dict[str, Any]) -> dict[str, Any]:
    merged = dict(_DEFAULT_DYNAMIC_PLAYOUT_CFG)
    global_cfg = config.get('playout', {}).get('dynamic_playlist', {})
    feed_cfg = _feed_playout_cfg(feed).get('dynamic_playlist', {})
    if isinstance(global_cfg, dict):
        merged.update(global_cfg)
    if isinstance(feed_cfg, dict):
        merged.update(feed_cfg)
    return merged


def _pacing_cfg(config: dict[str, Any], feed: dict[str, Any]) -> dict[str, float]:
    merged: dict[str, Any] = dict(_DEFAULT_PACING_CFG)

    global_playout = config.get('playout', {})
    if isinstance(global_playout, dict):
        for key in _DEFAULT_PACING_CFG:
            value = global_playout.get(key)
            if value is not None:
                merged[key] = value
        global_cfg = global_playout.get('pacing', {})
        if isinstance(global_cfg, dict):
            for key in _DEFAULT_PACING_CFG:
                value = global_cfg.get(key)
                if value is not None:
                    merged[key] = value

    feed_playout = _feed_playout_cfg(feed)
    for key in _DEFAULT_PACING_CFG:
        value = feed_playout.get(key)
        if value is not None:
            merged[key] = value
    feed_cfg = feed_playout.get('pacing', {})
    if isinstance(feed_cfg, dict):
        for key in _DEFAULT_PACING_CFG:
            value = feed_cfg.get(key)
            if value is not None:
                merged[key] = value

    return {
        key: max(0.0, float(merged.get(key, default) or 0.0))
        for key, default in _DEFAULT_PACING_CFG.items()
    }


def _playlist_order(config: dict[str, Any]) -> list[str]:
    raw_order = config.get('playout', {}).get('playlist_order') or []
    order: list[str] = []
    for entry in raw_order:
        if isinstance(entry, str):
            order.append(entry)
            continue
        if isinstance(entry, dict):
            pkg_id = next((key for key in entry if key != 'ttl'), None)
            if pkg_id:
                order.append(pkg_id)
    return [pkg_id for pkg_id in order if pkg_id != 'date_time']


def _station_id_schedule_cfg(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(_DEFAULT_STATION_ID_SCHEDULE_CFG)
    raw = config.get('playout', {}).get('station_id_schedule', {})
    if isinstance(raw, dict):
        merged.update(raw)

    raw_minutes = merged.get('minutes', _DEFAULT_STATION_ID_SCHEDULE_CFG['minutes'])
    if isinstance(raw_minutes, str):
        tokens = [part.strip() for part in raw_minutes.split(',') if part.strip()]
    elif isinstance(raw_minutes, (list, tuple, set)):
        tokens = list(raw_minutes)
    else:
        tokens = [raw_minutes]

    minutes: list[int] = []
    for token in tokens:
        try:
            minute = int(token)
        except (TypeError, ValueError):
            continue
        if 0 <= minute <= 59 and minute not in minutes:
            minutes.append(minute)

    return {
        'enabled': bool(merged.get('enabled', True)),
        'minutes': minutes or list(_DEFAULT_STATION_ID_SCHEDULE_CFG['minutes']),
    }


def _date_time_schedule_cfg(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(_DEFAULT_DATE_TIME_SCHEDULE_CFG)
    raw = config.get('playout', {}).get('date_time_schedule', {})
    if isinstance(raw, dict):
        merged.update(raw)

    raw_minutes = merged.get('minutes', _DEFAULT_DATE_TIME_SCHEDULE_CFG['minutes'])
    if isinstance(raw_minutes, str):
        tokens = [part.strip() for part in raw_minutes.split(',') if part.strip()]
    elif isinstance(raw_minutes, (list, tuple, set)):
        tokens = list(raw_minutes)
    else:
        tokens = [raw_minutes]

    minutes: list[int] = []
    for token in tokens:
        try:
            minute = int(token)
        except (TypeError, ValueError):
            continue
        if 0 <= minute <= 59 and minute not in minutes:
            minutes.append(minute)

    return {
        'enabled': bool(merged.get('enabled', True)),
        'minutes': minutes or list(_DEFAULT_DATE_TIME_SCHEDULE_CFG['minutes']),
    }


def _pkg_voice(pkg_id: str) -> str | None:
    pkg_cfg = Package_Config.per_package.get(pkg_id, {})
    voice = pkg_cfg.get('use_voice')
    if isinstance(voice, str) and voice:
        return voice
    return None


def _pkg_allowed_in_lang(pkg_id: str, lang: str) -> bool:
    allowed_langs = Package_Config.per_package.get(pkg_id, {}).get('lang')
    return allowed_langs is None or lang in allowed_langs


def _estimate_duration(text: str) -> float:
    words = len(_WORD_RE.findall(text))
    if words <= 0:
        return 2.0
    return max(2.0, (words / 155.0) * 60.0 + 0.35)


def _make_item(
    pkg_id: str,
    text: str,
    lang: str,
    item_id: str,
    group_id: str,
    title: str,
    gap_after_s: float,
) -> PlayableItem:
    return PlayableItem(
        pkg_id=pkg_id,
        text=text,
        lang=lang,
        voice=_pkg_voice(pkg_id),
        metadata=NowPlayingMetadata(title=title),
        item_id=item_id,
        group_id=group_id,
        gap_after_s=max(0.0, gap_after_s),
        estimated_duration=_estimate_duration(text),
    )


def _make_text_items(
    pkg_id: str,
    text: str,
    lang: str,
    group_id: str,
    title: str,
    item_prefix: str,
    *,
    package_gap_s: float,
) -> list[PlayableItem]:
    normalized = ' '.join(str(text or '').split())
    if not normalized:
        return []
    return [
        _make_item(pkg_id, normalized, lang, item_prefix, group_id, title, package_gap_s)
    ]


def _seconds_until_next_boundary(config: dict[str, Any]) -> float | None:
    now = _dt.datetime.now().astimezone()
    minutes: set[int] = set()

    station_id_cfg = _station_id_schedule_cfg(config)
    if station_id_cfg['enabled']:
        minutes.update(station_id_cfg['minutes'])

    date_time_cfg = _date_time_schedule_cfg(config)
    if date_time_cfg['enabled']:
        minutes.update(date_time_cfg['minutes'])

    if not minutes:
        return None

    candidates: list[_dt.datetime] = []
    for minute in sorted(minutes):
        candidate = now.replace(minute=minute, second=0, microsecond=0)
        if candidate <= now:
            candidate += _dt.timedelta(hours=1)
        candidates.append(candidate)

    return min((candidate - now).total_seconds() for candidate in candidates)


def _next_chime_boundary(config: dict[str, Any]) -> tuple[_dt.datetime, str] | None:
    chimes_cfg = config.get('playout', {}).get('chimes', {})
    if not chimes_cfg.get('enabled', False):
        return None

    now = _dt.datetime.now().astimezone()
    candidates: list[tuple[_dt.datetime, str]] = []

    if chimes_cfg.get('top_of_hour', {}).get('enabled', True):
        candidate = now.replace(minute=0, second=0, microsecond=0)
        if candidate <= now:
            candidate += _dt.timedelta(hours=1)
        candidates.append((candidate, 'top'))

    if chimes_cfg.get('half_hour', {}).get('enabled', True):
        candidate = now.replace(minute=30, second=0, microsecond=0)
        if candidate <= now:
            candidate += _dt.timedelta(hours=1)
        candidates.append((candidate, 'half'))

    return min(candidates, key=lambda x: x[0]) if candidates else None


def _build_chime_items(config: dict[str, Any], feed: dict[str, Any], chime_type: str, boundary_key: str) -> list[PlayableItem]:
    chimes_cfg = config.get('playout', {}).get('chimes', {})
    chime_key = 'top_of_hour' if chime_type == 'top' else 'half_hour'
    chime_steps = 16 if chime_type == 'top' else 8
    chime_pkg_id = f'chime_{chime_steps}'
    default_seq = ['chime']
    chime_seq: list[str] = chimes_cfg.get(chime_key, {}).get('sequence', default_seq)

    languages_cfg = feed.get('languages')
    lang = (
        str(next(iter(languages_cfg)))
        if isinstance(languages_cfg, dict) and languages_cfg
        else str(feed.get('language', 'en-CA'))
    )
    timezone_name = str(feed.get('timezone', 'UTC'))
    package_gap_s = _pacing_cfg(config, feed)['package_gap_s']

    items: list[PlayableItem] = []
    for ci in chime_seq:
        if ci == 'chime':
            items.append(PlayableItem(
                pkg_id=chime_pkg_id,
                text='',
                lang=lang,
                metadata=NowPlayingMetadata(title=_PACKAGE_METADATA_LABELS[chime_pkg_id]),
                item_id=f'{boundary_key}:chime',
                group_id=boundary_key,
                estimated_duration=_CHIME_DURATION_S[chime_type],
                gap_after_s=0.0,
            ))
        elif ci == 'date_time':
            text = date_time_package(timezone_name, lang)
            items.extend(_make_text_items(
                'date_time', text, lang,
                boundary_key,
                _PACKAGE_METADATA_LABELS['date_time'],
                f'{boundary_key}:date_time',
                package_gap_s=package_gap_s,
            ))
        elif ci == 'station_id':
            text = station_id(config, str(feed['id']), lang)
            items.extend(_make_text_items(
                'station_id', text, lang,
                boundary_key,
                _PACKAGE_METADATA_LABELS['station_id'],
                f'{boundary_key}:station_id',
                package_gap_s=0.0,
            ))
    return items


def _closest_insert_index(sequence: list[PlayableItem], target_offset_s: float) -> int:
    elapsed = 0.0
    best_index = len(sequence)
    best_delta = float('inf')

    for index in range(len(sequence) + 1):
        delta = abs(target_offset_s - elapsed)
        if delta < best_delta:
            best_delta = delta
            best_index = index
        if index < len(sequence):
            elapsed += _queue_cost(sequence[index])

    return best_index


def _inject_chime_items(
    feed_id: str,
    config: dict[str, Any],
    feed: dict[str, Any],
    sequence: list[PlayableItem],
    max_fill_seconds: float,
) -> list[PlayableItem]:
    chime_boundary = _next_chime_boundary(config)
    if chime_boundary is None:
        return sequence

    boundary_at, chime_type = chime_boundary
    boundary_secs = (boundary_at - _dt.datetime.now().astimezone()).total_seconds()
    if boundary_secs > max_fill_seconds:
        return sequence

    boundary_key = f'chime:{chime_type}:{boundary_at.astimezone(_dt.UTC).strftime("%Y%m%dT%H%M")}'
    if _scheduled_chime_boundaries.get(feed_id) == boundary_key:
        return sequence
    if any(item.pkg_id in ('chime_8', 'chime_16') for item in sequence):
        return sequence

    chime_items = _build_chime_items(config, feed, chime_type, boundary_key)
    if not chime_items:
        return sequence

    insert_index = _closest_insert_index(sequence, max(0.0, boundary_secs))
    _scheduled_chime_boundaries[feed_id] = boundary_key
    return sequence[:insert_index] + chime_items + sequence[insert_index:]


def _current_conditions_context(
    feed_id: str,
    location: dict[str, Any],
    lang: str,
    secondary: bool,
) -> tuple[str, str | None, str | None]:
    loc_id = location.get('id')
    data = read_data_pool(f'{feed_id}:{loc_id}') if loc_id else None
    text = current_conditions_package(data, _current_conditions_name(location), lang, secondary=secondary)
    resolved_name = _current_conditions_name(location) or _conditions_station_name(data, lang) or _location_label(location, loc_id)
    return text, resolved_name, loc_id


def _forecast_context(feed_id: str, location: dict[str, Any], lang: str) -> tuple[str, str | None, str | None]:
    loc_id = location.get('id')
    data = read_data_pool(f'{feed_id}:forecast:{loc_id}') if loc_id else None
    text = forecast_package(data, _forecast_name(location), lang)
    resolved_name = _forecast_name(location) or _forecast_station_name(data, lang) or _location_label(location, loc_id)
    return text, resolved_name, loc_id


def _climate_context(feed_id: str, location: dict[str, Any], lang: str) -> tuple[str, str | None, str | None]:
    loc_id = location.get('id')
    data = read_data_pool(f'{feed_id}:climate:{loc_id}') if loc_id else None
    text = climate_summary_package(data, _climate_name(location), lang)
    resolved_name = _climate_name(location) or _climate_station_name(data, lang) or _location_label(location, loc_id)
    return text, resolved_name, loc_id


def _air_quality_context(feed_id: str, location: dict[str, Any], lang: str) -> tuple[str, str | None, str | None]:
    loc_id = location.get('id')
    data = read_data_pool(f'{feed_id}:aqhi:{loc_id}') if loc_id else None
    text = air_quality_package(data, _air_quality_name(location), lang)
    resolved_name = _air_quality_name(location) or _location_label(location, loc_id)
    return text, resolved_name, loc_id


def _build_cycle_items(config: dict[str, Any], feed: dict[str, Any]) -> list[PlayableItem]:
    feed_id = str(feed['id'])
    feed_name = _feed_display_name(feed)
    timezone_name = str(feed.get('timezone', 'UTC'))
    languages_cfg = feed.get('languages')
    if isinstance(languages_cfg, dict) and languages_cfg:
        languages = [str(lang) for lang in languages_cfg.keys()]
    else:
        languages = [str(feed.get('language', 'en-CA'))]

    order = _playlist_order(config)
    if not order:
        return []

    dynamic_cfg = _dynamic_playout_cfg(config, feed)
    pacing_cfg = _pacing_cfg(config, feed)
    package_gap_s = pacing_cfg['package_gap_s']

    observation_locations = _observation_locations(feed)
    forecast_locations = _forecast_locations(feed)
    climate_locations = _climate_locations(feed)
    air_quality_locations = _air_quality_locations(feed)
    bulletins = _load_bulletins()
    registry = _load_registry(feed_id)
    focn45_bulletin = read_data_pool('focn45')
    discussion_text = getattr(focn45_bulletin, 'text', None) or (focn45_bulletin if isinstance(focn45_bulletin, str) else None)
    wwv_text = read_data_pool('wwv')

    sequence: list[PlayableItem] = []

    for lang in languages:
        if shutdown_event.is_set():
            break

        for pkg_id in order:
            if not _pkg_allowed_in_lang(pkg_id, lang):
                continue

            group_id = f'{lang}:{pkg_id}'

            if pkg_id == 'date_time':
                continue

            if pkg_id == 'station_id':
                continue

            if pkg_id == 'current_conditions':
                for index, location in enumerate(observation_locations):
                    text, resolved_name, loc_id = _current_conditions_context(feed_id, location, lang, secondary=(index > 0))
                    if not text:
                        continue
                    title = _metadata_title(pkg_id, resolved_name, resolved_name)
                    prefix = f'{group_id}:{loc_id or index}'
                    sequence.extend(
                        _make_text_items(
                            pkg_id,
                            text,
                            lang,
                            group_id,
                            title,
                            prefix,
                            package_gap_s=package_gap_s,
                        )
                    )
                continue

            if pkg_id == 'forecast':
                for index, location in enumerate(forecast_locations):
                    text, resolved_name, loc_id = _forecast_context(feed_id, location, lang)
                    if not text:
                        continue
                    title = _metadata_title(pkg_id, resolved_name, resolved_name)
                    prefix = f'{group_id}:{loc_id or index}'
                    sequence.extend(
                        _make_text_items(
                            pkg_id,
                            text,
                            lang,
                            group_id,
                            title,
                            prefix,
                            package_gap_s=package_gap_s,
                        )
                    )
                continue

            if pkg_id == 'climate_summary':
                for index, location in enumerate(climate_locations):
                    text, resolved_name, loc_id = _climate_context(feed_id, location, lang)
                    if not text:
                        continue
                    title = _metadata_title(pkg_id, resolved_name, resolved_name)
                    prefix = f'{group_id}:{loc_id or index}'
                    sequence.extend(
                        _make_text_items(
                            pkg_id,
                            text,
                            lang,
                            group_id,
                            title,
                            prefix,
                            package_gap_s=package_gap_s,
                        )
                    )
                continue

            if pkg_id == 'air_quality':
                for index, location in enumerate(air_quality_locations):
                    text, resolved_name, loc_id = _air_quality_context(feed_id, location, lang)
                    if not text:
                        continue
                    title = _metadata_title(pkg_id, resolved_name, resolved_name)
                    prefix = f'{group_id}:{loc_id or index}'
                    sequence.extend(
                        _make_text_items(
                            pkg_id,
                            text,
                            lang,
                            group_id,
                            title,
                            prefix,
                            package_gap_s=package_gap_s,
                        )
                    )
                continue

            if pkg_id == 'eccc_discussion':
                text = eccc_discussion_package(discussion_text, None, lang)
            elif pkg_id == 'geophysical_alert':
                text = geophysical_alert_package(wwv_text)
            elif pkg_id == 'user_bulletin':
                text = user_bulletin_package(bulletins, lang, timezone_name)
            elif pkg_id == 'alerts':
                text = alerts_package(registry, lang, timezone_name, feed)
            else:
                text = ''

            if not text:
                continue

            sequence.extend(
                _make_text_items(
                    pkg_id,
                    text,
                    lang,
                    group_id,
                    _metadata_title(pkg_id, feed_name, feed_name),
                    group_id,
                    package_gap_s=package_gap_s,
                )
            )

    return sequence


def _queue_cost(item: PlayableItem) -> float:
    return (item.estimated_duration or _estimate_duration(item.text)) + max(0.0, item.gap_after_s)


def _queued_duration(items: list[PlayableItem]) -> float:
    return sum(_queue_cost(item) for item in items)


def _refill_feed_queue(config: dict[str, Any], feed: dict[str, Any], *, force: bool = False) -> None:
    feed_id = str(feed['id'])
    playout_cfg = _feed_playout_cfg(feed)
    if not playout_cfg.get('routine', True):
        if get_playout_sequence(feed_id):
            update_playout_sequence(feed_id, [])
        return

    sequence_version = get_sequence_version(feed_id)
    dynamic_cfg = _dynamic_playout_cfg(config, feed)
    target_depth = max(1, int(dynamic_cfg.get('target_depth', 8) or 8))
    low_water_mark = max(0, int(dynamic_cfg.get('low_water_mark', 3) or 3))
    max_fill_seconds = max(15.0, float(dynamic_cfg.get('max_fill_seconds', 180.0) or 15.0))
    boundary_guard_seconds = max(0.0, float(dynamic_cfg.get('boundary_guard_seconds', 2.0) or 0.0))

    pending = get_playout_sequence(feed_id)
    if not force and len(pending) >= low_water_mark:
        return

    cycle = _build_cycle_items(config, feed)
    if not cycle:
        if force and pending:
            update_playout_sequence(feed_id, [])
        return

    existing_ids = {item.item_id for item in pending if item.item_id}
    anchor_id = pending[-1].item_id if pending else get_last_played_item_id(feed_id)
    start_index = 0
    if anchor_id:
        for index, item in enumerate(cycle):
            if item.item_id == anchor_id:
                start_index = (index + 1) % len(cycle)
                break

    remaining_time_budget = max_fill_seconds - _queued_duration(pending)
    boundary_budget = _seconds_until_next_boundary(config)
    if boundary_budget is not None:
        boundary_budget = max(0.0, boundary_budget - boundary_guard_seconds - _queued_duration(pending))

    additions: list[PlayableItem] = []
    for offset in range(len(cycle)):
        if len(pending) + len(additions) >= target_depth:
            break

        item = cycle[(start_index + offset) % len(cycle)]
        if item.item_id and item.item_id in existing_ids:
            continue

        queue_cost = _queue_cost(item)
        if remaining_time_budget < queue_cost:
            break
        if boundary_budget is not None and boundary_budget < queue_cost:
            break

        additions.append(item)
        existing_ids.add(item.item_id)
        remaining_time_budget -= queue_cost
        if boundary_budget is not None:
            boundary_budget -= queue_cost

    planned_sequence = pending + additions
    planned_sequence = _inject_chime_items(feed_id, config, feed, planned_sequence, max_fill_seconds)

    if not additions and planned_sequence == pending:
        return

    if get_sequence_version(feed_id) != sequence_version:
        if additions:
            append_playout_items(feed_id, additions)
        return

    update_playout_sequence(feed_id, planned_sequence)


def playlist_thread_worker(config: dict[str, Any]) -> None:
    feeds = [feed for feed in config.get('feeds', []) if feed.get('enabled', True)]
    if not feeds:
        initial_synthesis_done.set()
        return

    awaiting_initial_refresh = True
    first_cycle = True

    while not shutdown_event.is_set():
        timeout = 0.25 if awaiting_initial_refresh else 1.0
        refreshed = data_ready.wait(timeout=timeout)
        if awaiting_initial_refresh and not refreshed:
            continue
        if refreshed:
            data_ready.clear()
            awaiting_initial_refresh = False

        for feed in feeds:
            if shutdown_event.is_set():
                break
            try:
                _refill_feed_queue(config, feed, force=bool(refreshed or first_cycle))
            except Exception:
                log.exception('Playlist refill failed for feed %s', feed.get('id', '?'))

        if first_cycle and not awaiting_initial_refresh:
            initial_synthesis_done.set()
            first_cycle = False
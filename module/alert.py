from __future__ import annotations

import asyncio
import csv
import datetime as _dt
import functools
import json
import logging
import os
import pathlib
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, cast

import numpy as np

_REGISTRY_DIR = pathlib.Path('data') / 'alerts'
_registry_lock = threading.Lock()


def _registry_path(feed_id: str) -> pathlib.Path:
    return _REGISTRY_DIR / f'{feed_id}.json'

_MAX_ALERT_SCRIPT_WORDS = 400
_ALERT_SENT_RE = re.compile(r'(?<=[.!?])\s+')


def _truncate_at_sentence_boundary(text: str, max_words: int) -> str:
    if len(text.split()) <= max_words:
        return text
    parts = _ALERT_SENT_RE.split(text)
    result: list[str] = []
    count = 0
    for part in parts:
        w = len(part.split())
        if count + w > max_words and result:
            break
        result.append(part)
        count += w
    return ' '.join(result) if result else ' '.join(text.split()[:max_words])


def _fit_sentences_to_word_budget(sentences: list[str]) -> list[str]:
    if sum(len(s.split()) for s in sentences) <= _MAX_ALERT_SCRIPT_WORDS:
        return sentences
    out = list(sentences)
    for i in range(len(out) - 1, -1, -1):
        if sum(len(s.split()) for s in out) <= _MAX_ALERT_SCRIPT_WORDS:
            break
        budget = _MAX_ALERT_SCRIPT_WORDS - sum(len(out[j].split()) for j in range(i))
        out[i] = _truncate_at_sentence_boundary(out[i], budget) if budget > 0 else ''
    return [s for s in out if s]


def _parse_cap_references(references: str) -> list[str]:
    ids: list[str] = []
    for ref in references.strip().split():
        parts = ref.split(',')
        if len(parts) >= 2:
            ids.append(parts[1])
    return ids


def _slugify_alert_filename(value: str) -> str:
    normalized = unicodedata.normalize('NFKD', value)
    ascii_only = normalized.encode('ascii', 'ignore').decode('ascii')
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', ascii_only).strip('._-')
    return slug or 'alert'


def _alert_audio_dir(feed_id: str) -> pathlib.Path:
    return pathlib.Path('audio') / 'alerts' / _slugify_alert_filename(feed_id)


def _alert_audio_path(feed_id: str, identifier: str) -> pathlib.Path:
    return _alert_audio_dir(feed_id) / f'{_slugify_alert_filename(identifier)}.wav'


def _cap_audio_resources(alert: CAPAlert, languages: list[str]) -> list[tuple[CAPInfo, CAPResource]]:
    ordered_infos: list[CAPInfo] = []
    seen_info_ids: set[int] = set()
    for lang in languages:
        info = alert.info_for_lang(lang)
        if info is None:
            continue
        key = id(info)
        if key in seen_info_ids:
            continue
        seen_info_ids.add(key)
        ordered_infos.append(info)
    for info in alert.infos:
        key = id(info)
        if key in seen_info_ids:
            continue
        seen_info_ids.add(key)
        ordered_infos.append(info)

    matches: list[tuple[CAPInfo, CAPResource]] = []
    for info in ordered_infos:
        for resource in info.resources:
            mime_type = (resource.mime_type or '').lower()
            description = (resource.description or '').lower()
            uri = (resource.uri or '').lower()
            if mime_type.startswith('audio/') or 'broadcast audio' in description or uri.endswith(('.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac')):
                matches.append((info, resource))
    return matches


def _cap_audio_extension(resource: CAPResource) -> str:
    mime_type = (resource.mime_type or '').lower().split(';', 1)[0].strip()
    if mime_type == 'audio/mpeg':
        return '.mp3'
    if mime_type == 'audio/wav' or mime_type == 'audio/x-wav':
        return '.wav'
    if mime_type == 'audio/ogg':
        return '.ogg'
    if mime_type == 'audio/flac':
        return '.flac'
    if mime_type == 'audio/aac':
        return '.aac'
    uri_path = pathlib.Path((resource.uri or '').split('?', 1)[0])
    if uri_path.suffix:
        return uri_path.suffix.lower()
    return '.bin'


def _cap_audio_bytes(resource: CAPResource) -> bytes | None:
    if resource.data:
        return resource.data
    uri = (resource.uri or '').strip()
    if not uri:
        return None
    try:
        with urllib.request.urlopen(uri, timeout=15) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        log.warning('CAP audio fetch failed (%s): %s', uri, exc)
        return None


def _resolve_cap_audio_pcm(
    alert: CAPAlert,
    feed_langs: list[str],
) -> tuple[bytes | None, str | None]:
    from module.tts import decode_audio_bytes_pcm

    for info, resource in _cap_audio_resources(alert, feed_langs):
        resource_bytes = _cap_audio_bytes(resource)
        if not resource_bytes:
            continue
        pcm = decode_audio_bytes_pcm(resource_bytes, suffix=_cap_audio_extension(resource))
        if not pcm:
            log.warning('[%s] CAP audio decode failed for %s', alert.identifier, resource.uri or resource.description or 'resource')
            continue
        label_parts = ['cap']
        if info.language:
            label_parts.append(info.language)
        if resource.description:
            label_parts.append(resource.description)
        if resource.uri:
            label_parts.append(resource.uri)
        return pcm, ':'.join(part.strip() for part in label_parts if part and part.strip())
    return None, None


def _save_alert_audio(feed_id: str, identifier: str, alert_pcm: bytes) -> pathlib.Path:
    from module.tts import write_pcm_wav

    output_path = _alert_audio_path(feed_id, identifier)
    write_pcm_wav(output_path, alert_pcm)
    return output_path


def save_alert_audio(feed_id: str, identifier: str, alert_pcm: bytes) -> pathlib.Path:
    return _save_alert_audio(feed_id, identifier, alert_pcm)


def _build_alert_entry(feed_id: str, alert: CAPAlert | Any) -> dict[str, Any]:
    identifier = alert.identifier if hasattr(alert, 'identifier') else str(alert)
    info = alert.infos[0] if hasattr(alert, 'infos') and alert.infos else None
    sent = getattr(alert, 'sent', None)

    areas: list[dict[str, Any]] = []
    if info:
        for area in getattr(info, 'areas', None) or []:
            geocodes = [
                {'valueName': g.name, 'value': g.value}
                for g in (getattr(area, 'geocodes', None) or [])
            ]
            polygon = getattr(area, 'polygon', None)
            areas.append({
                'areaDesc': getattr(area, 'area_desc', '') or getattr(area, 'description', ''),
                'polygon': polygon if isinstance(polygon, str) else (' '.join(polygon) if polygon else None),
                'geocodes': geocodes,
            })

    event_codes: list[dict[str, str]] = []
    if info:
        for ec in getattr(info, 'event_codes', None) or []:
            event_codes.append({'valueName': getattr(ec, 'name', ''), 'value': getattr(ec, 'value', '')})

    return {
        'identifier': identifier,
        'feed_id': feed_id,
        'received_at': _dt.datetime.now(_dt.UTC).isoformat(),
        'metadata': {
            'language': getattr(info, 'language', '') if info else '',
            'senderName': info.sender_name if info else '',
            'headline': info.headline if info else '',
            'event': info.event if info else '',
            'category': getattr(info, 'category', '') if info else '',
            'responseType': getattr(info, 'response_type', '') if info else '',
            'urgency': getattr(info, 'urgency', '') if info else '',
            'severity': getattr(info, 'severity', '') if info else '',
            'certainty': getattr(info, 'certainty', '') if info else '',
            'audience': getattr(info, 'audience', '') if info else '',
            'effective': info.effective.isoformat() if info and info.effective else None,
            'onset': info.onset.isoformat() if info and info.onset else None,
            'expires': info.expires.isoformat() if info and info.expires else None,
            'web': getattr(info, 'web', '') if info else '',
            'eventCodes': event_codes,
        },
        'source': {
            'sender': getattr(alert, 'sender', ''),
            'sent': sent.isoformat() if sent else None,
            'status': getattr(alert, 'status', ''),
            'msgType': alert.msg_type if hasattr(alert, 'msg_type') else 'Alert',
            'scope': getattr(alert, 'scope', ''),
            'references': getattr(alert, 'references', '') or None,
            'sourceStr': getattr(alert, 'source', '') or None,
            'note': getattr(alert, 'note', '') or None,
        },
        'parameters': [
            {'valueName': p.name, 'value': p.value}
            for p in (info.parameters if info else ())
        ],
        'text': {
            'description': info.description if info else '',
            'instruction': info.instruction if info else '',
        },
        'areas': areas,
    }


def _write_registry_entry(feed_id: str, alert: Any) -> bool:
    identifier = alert.identifier if hasattr(alert, 'identifier') else str(alert)
    if not identifier:
        return False
    reg_path = _registry_path(feed_id)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with _registry_lock:
        try:
            registry: list[dict] = json.loads(reg_path.read_text(encoding='utf-8')) if reg_path.exists() else []
        except Exception:
            registry = []
        if any(r.get('identifier') == identifier for r in registry):
            return False

        referenced_ids: set[str] = set()
        raw_refs = getattr(alert, 'references', '') or ''
        if raw_refs and getattr(alert, 'msg_type', '') in ('Update', 'Cancel'):
            referenced_ids = set(_parse_cap_references(raw_refs))

        if referenced_ids:
            before = len(registry)
            registry = [r for r in registry if r.get('identifier') not in referenced_ids]
            removed = before - len(registry)
            if removed:
                log.debug('Registry: removed %d superseded entry/entries for feed %s', removed, feed_id)
            remove_runtime_alert_entries(feed_id, referenced_ids)

        if getattr(alert, 'msg_type', '') == 'Cancel':
            try:
                reg_path.write_text(json.dumps(registry, indent=2), encoding='utf-8')
                log.debug('Registry: cancelled entries removed for %s', identifier)
            except Exception as e:
                log.error('Failed to write alert registry: %s', e)
            return False

        entry = _build_alert_entry(feed_id, alert)

        registry.append(entry)
        try:
            reg_path.write_text(json.dumps(registry, indent=2), encoding='utf-8')
            log.debug('Registry updated: %s', identifier)
            return True
        except Exception as e:
            log.error('Failed to write alert registry: %s', e)
            return False


_REGISTRY_EXPIRY_GRACE = _dt.timedelta(hours=24)


def purge_expired_alerts() -> int:
    if not _REGISTRY_DIR.exists():
        return 0
    cutoff = _dt.datetime.now(_dt.UTC) - _REGISTRY_EXPIRY_GRACE
    total_removed = 0
    with _registry_lock:
        for reg_path in _REGISTRY_DIR.glob('*.json'):
            try:
                registry: list[dict] = json.loads(reg_path.read_text(encoding='utf-8'))
            except Exception:
                continue

            before = len(registry)
            pruned: list[dict] = []
            for entry in registry:
                expires_raw: str | None = (entry.get('metadata') or {}).get('expires')
                if not expires_raw:
                    pruned.append(entry)
                    continue
                try:
                    expires_dt = _dt.datetime.fromisoformat(expires_raw)
                    if expires_dt.tzinfo is None:
                        expires_dt = expires_dt.replace(tzinfo=_dt.UTC)
                except ValueError:
                    pruned.append(entry)
                    continue
                if expires_dt >= cutoff:
                    pruned.append(entry)

            removed = before - len(pruned)
            if removed:
                try:
                    reg_path.write_text(json.dumps(pruned, indent=2), encoding='utf-8')
                except Exception as e:
                    log.error('Failed to write alert registry during purge: %s', e)
                    continue
                total_removed += removed
    return total_removed


from module.events import (
    append_runtime_event,
    enqueue_scheduled_package,
    push_alert,
    remove_runtime_alert_entries,
    store_runtime_alert_entry,
    update_feed_runtime,
)
from module.webhook import dispatch_webhook
from module.packages import (
    _AL_PH,
    _clean_alert_text,
    _format_datetime_spoken,
    _join_areas,
    _load_forecast_loc_db,
    _parse_eccc_subject,
)
from module.cap_specific.naads_tcp import CAPAlert, CAPInfo, naad_listener
from module.cap_specific.naadsatom_http import naad_archive_fetch
from module.cap_specific.nwsatom_http import nws_atom_listener
from module.buffer import CHANNELS, SAMPLE_RATE as BUS_SR
from module.same import SAMEHeader, SAME_SAMPLE_RATE, generate_same, resample, to_pcm16
from module.tts import synthesize_pcm

log = logging.getLogger(__name__)

_SAME_MAPPING_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'sameMapping.json'

_same_mapping_cache: dict[str, Any] | None = None
_EVENT_NORMALIZE_RE = re.compile(r'[^a-z0-9]+')


def _event_key(text: str) -> str:
    return _EVENT_NORMALIZE_RE.sub('', text.lower())


_WXR_CATEGORIES = frozenset({'met', 'geo'})
_CIV_CATEGORIES = frozenset({
    'security', 'law', 'cbrne', 'health', 'safety',
    'fire', 'rescue', 'env', 'transport', 'infra', 'other',
})


def _load_same_mapping() -> dict[str, Any]:
    global _same_mapping_cache
    if _same_mapping_cache is not None:
        return _same_mapping_cache
    with open(_SAME_MAPPING_PATH, encoding='utf-8') as f:
        mapping: dict[str, Any] = json.load(f)
    _same_mapping_cache = mapping
    naads_count = len(mapping.get('naadsToEas', {}))
    log.debug('Loaded SAME mapping: %d EAS codes, %d CAP-CP event mappings', len(mapping.get('eas', {})), naads_count)
    return mapping

_SEVERITY_PRIORITY: dict[str, int] = {
    "Extreme": 0,
    "Severe": 1,
    "Moderate": 2,
    "Minor": 3,
    "Unknown": 4,
}

_SEVERITY_POST_ALERT_ALERTS_PACKAGE = frozenset({'EXTREME', 'SEVERE', 'MODERATE'})
_IMPACT_POST_ALERT_ALERTS_PACKAGE = frozenset({'EXTREME', 'SEVERE', 'MODERATE', 'HIGH'})

_INFORMATIONAL_PRIORITY = 9

_MANAGED = pathlib.Path(__file__).parent.parent / "managed"
_CSV_DIR = _MANAGED / "csv"

_FORECAST_LOCATIONS_PATH = _CSV_DIR / "FORECAST_LOCATIONS.csv"
_CAP_CP_GEOCODES_PATH = _CSV_DIR / "CAP-CP_Geocodes.csv"
_CLC_BASE_ZONE_PATH = _CSV_DIR / "CLC_Base_Zone.csv"
_NWS_ZONE_COUNTY_CORRELATION_PATH = _CSV_DIR / "NWS_ZONE_COUNTY_CORRELATION.csv"

_geocode_db: dict[str, tuple[str, str]] | None = None
_forecast_db: dict[str, str] | None = None

_aggregate_clc_to_geocodes: dict[str, set[str]] | None = None
_aggregate_geocode_to_clcs: dict[str, set[str]] | None = None
_aggregate_clc_to_province: dict[str, str] | None = None
_geocode_db: dict[str, tuple[str, str]] | None = None
_CLC_LABEL_SPLIT_RE = re.compile(r'\bincluding\b', re.IGNORECASE)
_CLC_LABEL_PART_RE = re.compile(r'\s*(?:,|\band\b|\bet\b|/)\s*', re.IGNORECASE)
_CLC_ADMIN_PREFIX_RE = re.compile(
    r'^(?:r\.?m\.?|rm|m\.?r\.?|county|district|regional district|municipality|township|paroisse|parish|village|ville|city|town|district municipality|regional municipality)\s+of\s+',
    re.IGNORECASE,
)
_CLC_ADMIN_SUFFIX_RE = re.compile(
    r'\b(?:no\.?|number)\s+[0-9]+[a-z]?\b',
    re.IGNORECASE,
)
_CLC_NOISE_RE = re.compile(r'[^a-z0-9]+')


def _split_coverage_values(raw_value: str) -> list[str]:
    return [part.strip() for part in raw_value.split(',') if part.strip()]


@functools.lru_cache(maxsize=1)
def _load_forecast_coverage_db() -> dict[str, set[str]]:
    db: dict[str, set[str]] = {}
    try:
        with open(_FORECAST_LOCATIONS_PATH, newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            next(reader, None)
            header = next(reader, None)
            if header is None:
                return db
            normalized = [c.strip().upper() for c in header]
            code_idx = normalized.index('CODE')
            province_indices = [
                idx for idx, column in enumerate(normalized)
                if 'PROVINCE' in column or 'WATERBODY' in column
            ]
            for row in reader:
                if len(row) <= code_idx:
                    continue
                code = row[code_idx].strip().strip('"')
                if not code:
                    continue
                values: set[str] = set()
                for idx in province_indices:
                    if len(row) <= idx:
                        continue
                    values.update(value.upper() for value in _split_coverage_values(row[idx].strip().strip('"')))
                for value in values:
                    db.setdefault(value, set()).add(code)
    except (OSError, csv.Error) as e:
        log.error("Failed to load forecast coverage DB: %s", e)
    return db


@functools.lru_cache(maxsize=1)
def _load_nws_zone_county_db() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    zones: dict[str, set[str]] = {}
    counties: dict[str, set[str]] = {}
    try:
        with open(_NWS_ZONE_COUNTY_CORRELATION_PATH, newline='', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='|')
            for row in reader:
                if len(row) < 8:
                    continue
                state = row[0].strip().upper()
                zone_code = row[4].strip().upper()
                county_code = row[6].strip().upper().zfill(6)
                if state and zone_code:
                    zones.setdefault(state, set()).add(zone_code)
                if state and county_code:
                    counties.setdefault(state, set()).add(county_code)
    except (OSError, csv.Error) as e:
        log.error("Failed to load NWS zone/county DB: %s", e)
    return zones, counties


def _normalize_location_label(value: str) -> str:
    ascii_value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    lowered = ascii_value.lower().replace('&', ' and ')
    lowered = lowered.replace('saint', 'st').replace('sainte', 'st')
    lowered = _CLC_ADMIN_PREFIX_RE.sub('', lowered)
    lowered = _CLC_ADMIN_SUFFIX_RE.sub('', lowered)
    lowered = _CLC_NOISE_RE.sub(' ', lowered)
    return ' '.join(part for part in lowered.split() if part)


def _clc_name_aliases(name: str) -> set[str]:
    aliases: set[str] = set()
    raw_name = name.strip()
    if not raw_name:
        return aliases
    normalized_full = _normalize_location_label(raw_name)
    if normalized_full:
        aliases.add(normalized_full)

    parts = _CLC_LABEL_SPLIT_RE.split(raw_name, maxsplit=1)
    for part in parts:
        normalized_part = _normalize_location_label(part)
        if normalized_part:
            aliases.add(normalized_part)
        for subpart in _CLC_LABEL_PART_RE.split(part):
            normalized_subpart = _normalize_location_label(subpart)
            if normalized_subpart:
                aliases.add(normalized_subpart)
    return aliases


def _geocode_name_aliases(name: str) -> set[str]:
    aliases: set[str] = set()
    normalized = _normalize_location_label(name)
    if normalized:
        aliases.add(normalized)
    return aliases


def _load_aggregate_db() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    global _aggregate_clc_to_geocodes, _aggregate_geocode_to_clcs, _aggregate_clc_to_province
    if _aggregate_clc_to_geocodes is not None:
        return _aggregate_clc_to_geocodes, _aggregate_geocode_to_clcs, _aggregate_clc_to_province  # type: ignore[return-value]
    _aggregate_clc_to_geocodes = {}
    _aggregate_geocode_to_clcs = {}
    _aggregate_clc_to_province = _load_forecast_db()
    geocode_rows_by_region: dict[tuple[str, str], list[tuple[str, set[str]]]] = {}

    try:
        with open(_CAP_CP_GEOCODES_PATH, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = [column.strip().upper() for column in next(reader, [])]
            name_idx = header.index('NAME')
            nom_idx = header.index('NOM')
            code_idx = header.index('CAPCPGCODE')
            waterbody_idx = header.index('WATRBODY_C')
            province_idx = header.index('PROVINCE_C')
            country_idx = header.index('COUNTRY_C')
            for row in reader:
                if len(row) <= max(name_idx, nom_idx, code_idx, waterbody_idx, province_idx, country_idx):
                    continue
                geocode = row[code_idx].strip().strip('"')
                if not geocode:
                    continue
                region_key = (
                    row[country_idx].strip().strip('"').upper(),
                    row[waterbody_idx].strip().strip('"').upper() or row[province_idx].strip().strip('"').upper(),
                )
                aliases: set[str] = set()
                aliases.update(_geocode_name_aliases(row[name_idx].strip().strip('"')))
                aliases.update(_geocode_name_aliases(row[nom_idx].strip().strip('"')))
                if aliases:
                    geocode_rows_by_region.setdefault(region_key, []).append((geocode, aliases))
    except (OSError, csv.Error, ValueError) as e:
        log.error('Failed to load CAP-CP geocode relationships: %s', e)
        return _aggregate_clc_to_geocodes, _aggregate_geocode_to_clcs, _aggregate_clc_to_province

    try:
        with open(_CLC_BASE_ZONE_PATH, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = [column.strip().upper() for column in next(reader, [])]
            clc_idx = header.index('CLC')
            name_idx = header.index('NAME')
            nom_idx = header.index('NOM')
            waterbody_idx = header.index('WATRBODY_C')
            province_idx = header.index('PROVINCE_C')
            country_idx = header.index('COUNTRY_C')
            for row in reader:
                if len(row) <= max(clc_idx, name_idx, nom_idx, waterbody_idx, province_idx, country_idx):
                    continue
                clc = row[clc_idx].strip().strip('"')
                if not clc:
                    continue
                aliases: set[str] = set()
                aliases.update(_clc_name_aliases(row[name_idx].strip().strip('"')))
                aliases.update(_clc_name_aliases(row[nom_idx].strip().strip('"')))
                if not aliases:
                    continue
                region_key = (
                    row[country_idx].strip().strip('"').upper(),
                    row[waterbody_idx].strip().strip('"').upper() or row[province_idx].strip().strip('"').upper(),
                )
                for geocode, geocode_aliases in geocode_rows_by_region.get(region_key, []):
                    if aliases.isdisjoint(geocode_aliases):
                        continue
                    _aggregate_clc_to_geocodes.setdefault(clc, set()).add(geocode)
                    _aggregate_geocode_to_clcs.setdefault(geocode, set()).add(clc)
    except (OSError, csv.Error, ValueError) as e:
        log.error('Failed to load CLC base relationships: %s', e)
    return _aggregate_clc_to_geocodes, _aggregate_geocode_to_clcs, _aggregate_clc_to_province


def _load_geocode_db() -> dict[str, tuple[str, str]]:
    global _geocode_db
    if _geocode_db is not None:
        return _geocode_db
    _geocode_db = {}
    try:
        with open(_CAP_CP_GEOCODES_PATH, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = [column.strip().upper() for column in next(reader, [])]
            code_idx = header.index('CAPCPGCODE')
            waterbody_idx = header.index('WATRBODY_C')
            province_idx = header.index('PROVINCE_C')
            for row in reader:
                if len(row) <= max(code_idx, waterbody_idx, province_idx):
                    continue
                geocode = row[code_idx].strip().strip('"')
                if not geocode:
                    continue
                region_type = 'WB' if row[waterbody_idx].strip().strip('"') else 'LAND'
                region_code = row[waterbody_idx].strip().strip('"').upper() or row[province_idx].strip().strip('"').upper()
                _geocode_db[geocode] = (region_type, region_code)
    except (OSError, csv.Error, ValueError) as e:
        log.error('Failed to load CAP-CP geocode DB: %s', e)
    return _geocode_db


def _load_forecast_db() -> dict[str, str]:
    global _forecast_db
    if _forecast_db is not None:
        return _forecast_db
    db: dict[str, str] = {}
    try:
        with _CLC_BASE_ZONE_PATH.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = [c.strip().upper() for c in next(reader, [])]
            clc_idx = header.index("CLC")
            prov_idx = header.index("PROVINCE_C")
            for row in reader:
                if len(row) <= max(clc_idx, prov_idx):
                    continue
                clc = row[clc_idx].strip().strip('"')
                if clc:
                    db[clc] = row[prov_idx].strip().strip('"')
    except Exception:
        pass
    _forecast_db = db
    log.debug("Loaded %d CLC entries from CLC_Base_Zone.csv", len(db))
    return db


def _matches_feed_alert_code(pattern: str, raw_geocodes: set[str]) -> bool:
    pattern = pattern.strip()
    if not pattern:
        return False
    if pattern.endswith('*'):
        prefix = pattern[:-1]
        return any(code.startswith(prefix) for code in raw_geocodes)
    return pattern in raw_geocodes


def _geocodes_cover_clc(geocodes: tuple[str, ...] | list[str], clc_code: str) -> bool:
    clc_to_geo, _, _ = _load_aggregate_db()
    known_geocodes = clc_to_geo.get(clc_code, set())
    if known_geocodes:
        return bool(set(geocodes) & known_geocodes)
    for gc in geocodes:
        if gc == clc_code:
            return True
    return False


def _feed_clc_geocodes(feed_codes: list[str]) -> set[str]:
    clc_to_geo, _, _ = _load_aggregate_db()
    result: set[str] = set()
    for code in feed_codes:
        matched = clc_to_geo.get(code)
        if matched:
            result.update(matched)
    return result


def _is_national_geocode(geocodes: tuple[str, ...] | list[str]) -> bool:
    geo_db = _load_geocode_db()
    for gc in geocodes:
        if gc == '0':
            return True
        info = geo_db.get(gc)
        if info and info[0] == 'WB' and gc in ('001', '002', '003', '004', '005', '006', '007', '008'):
            continue
    return False


def _eccc_active_clcs_for_update(alert: CAPAlert) -> tuple[frozenset[str], bool]:
    """Returns (active_clc_codes, is_eccc_update).

    For ECCC msgType=Update: returns the set of CLC codes that are currently
    active in this update.  Prefers Newly_Active_Areas if populated; otherwise
    falls back to the CLC geocodes in the alert's area blocks.  Returns
    (frozenset(), False) for non-ECCC or non-Update alerts.
    """
    if alert.msg_type != 'Update':
        return frozenset(), False
    info = alert.english or (alert.infos[0] if alert.infos else None)
    if not info:
        return frozenset(), False
    pd = info.param_dict()
    if not any(k.startswith('layer:ec-msc-smc') for k in pd):
        return frozenset(), False
    raw_newly_active = pd.get('layer:ec-msc-smc:1.1:newly_active_areas', '')
    if raw_newly_active.strip():
        return frozenset(c.strip() for c in raw_newly_active.split(',') if c.strip()), True
    area_clcs: set[str] = set()
    for area in info.areas:
        for code in area.clc_codes:
            if code:
                area_clcs.add(code)
    return frozenset(area_clcs), True

def _normalize_strings(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        item = values.strip()
        return [item] if item else []

    normalized: list[str] = []
    if isinstance(values, (list, tuple, set)):
        for value in cast(list[Any] | tuple[Any, ...] | set[Any], values):
            if value is None:
                continue
            item = str(value).strip()
            if item:
                normalized.append(item)
    return normalized


def feed_same_codes(feed: dict[str, Any]) -> list[str]:
    db = _load_forecast_db()
    province_db = _load_forecast_coverage_db()
    nws_zone_db, nws_county_db = _load_nws_zone_county_db()
    codes: list[str] = []
    seen: set[str] = set()

    def _expand_codes(raw_value: str) -> list[str]:
        if not raw_value:
            return []
        normalized: list[str] = []
        for part in _split_coverage_values(raw_value):
            if '-' in part:
                left, right = part.split('-', 1)
                if left.replace('*', '').isdigit() and right.replace('*', '').isdigit():
                    part = right
            normalized.append(part)
        return normalized

    def _loc_text(loc: dict[str, Any], *keys: str) -> str:
        for key in keys:
            value = str(loc.get(key) or '').strip()
            if value:
                return value
        return ''

    def _raw_codes(loc: dict[str, Any]) -> list[str]:
        explicit_same = str(loc.get('same') or '').strip()
        if explicit_same:
            return _expand_codes(explicit_same)

        coverage_type = str(loc.get('coverage_type') or loc.get('kind') or loc.get('type') or '').strip().lower()
        coverage_key = _loc_text(loc, 'abbr', 'code', 'id', 'zone', 'county', 'same')
        if coverage_type == 'province':
            return sorted(province_db.get(coverage_key.upper(), set())) or ([coverage_key] if coverage_key else [])
        if coverage_type == 'state':
            zone_codes = nws_zone_db.get(coverage_key.upper(), set())
            county_codes = nws_county_db.get(coverage_key.upper(), set())
            return sorted(zone_codes | county_codes) or ([coverage_key] if coverage_key else [])
        if coverage_type == 'zone':
            return sorted(nws_zone_db.get(coverage_key.upper(), set())) or ([coverage_key] if coverage_key else [])
        if coverage_type == 'county':
            return sorted(nws_county_db.get(coverage_key.upper(), set())) or ([coverage_key] if coverage_key else [])

        raw = str(loc.get('id') or loc.get('forecast_region') or '').strip()
        if not raw:
            return []
        if '-' in raw:
            left, right = raw.split('-', 1)
            if left.replace('*', '').isdigit() and right.replace('*', '').isdigit():
                raw = right
        return _expand_codes(raw)

    def _append_target(target: str) -> None:
        if not target:
            return

        if '-' in target:
            left, right = target.split('-', 1)
            if left.replace('*', '').isdigit() and right.replace('*', '').isdigit():
                target = right

        if target.endswith('*') and target[:-1].isdigit():
            prefix = target[:-1]
        elif target.isdigit():
            prefix = target.rstrip('0')
        else:
            prefix = ''

        if prefix:
            for code in sorted(db):
                if code.startswith(prefix) and code not in seen:
                    seen.add(code)
                    codes.append(code)
            return

        if target not in seen:
            seen.add(target)
            codes.append(target)

    def _append_loc(loc: dict[str, Any]) -> None:
        for target in _raw_codes(loc):
            _append_target(target)
    coverage = feed.get('coverage')
    if isinstance(coverage, list) and coverage:
        for region in coverage:
            if not isinstance(region, dict):
                continue
            subregions = [subregion for subregion in region.get('subregions', []) if isinstance(subregion, dict)]
            if subregions:
                for subregion in subregions:
                    _append_loc(subregion)
            else:
                _append_loc(region)
        return codes

    for block in feed.get('locations', []):
        if not isinstance(block, dict):
            continue
        for loc in block.get('forecastLocations', []):
            if isinstance(loc, dict):
                _append_loc(loc)
    return codes

def _matching_feed_codes(alert: CAPAlert, feed: dict[str, Any]) -> list[str]:
    feed_codes = feed_same_codes(feed)
    if not feed_codes:
        return []

    alert_codes = set(alert.all_geocodes) | set(alert.clc_codes)

    direct_matches = [code for code in feed_codes if _matches_feed_alert_code(code, alert_codes)]
    if direct_matches:
        return direct_matches[:31]

    waterbody_matches = [
        code for code in feed_codes
        if code.isdigit() and _geocodes_cover_clc(alert.all_geocodes, code)
    ]
    return waterbody_matches[:31]


def _format_log_codes(values: Any, limit: int = 12) -> str:
    codes = list(dict.fromkeys(_normalize_strings(values)))
    if not codes:
        return '-'
    if len(codes) <= limit:
        return ','.join(codes)
    shown = ','.join(codes[:limit])
    return f"{shown},+{len(codes) - limit} more"


def _match_log_context(alert: CAPAlert, feed: dict[str, Any], cap_filter: dict[str, Any]) -> str:
    return (
        f"use_feed_locations={bool(cap_filter.get('use_feed_locations', True))} "
        f"alert_geocodes={_format_log_codes(alert.all_geocodes)} "
        f"alert_clc={_format_log_codes(alert.clc_codes)} "
        f"feed_codes={_format_log_codes(feed_same_codes(feed))} "
        f"filter_geocodes={_format_log_codes(cap_filter.get('geocodes'))}"
    )


def _header_locations(alert: CAPAlert, feed: dict[str, Any]) -> list[str]:
    feed_codes = feed_same_codes(feed)

    if not feed_codes:
        return ["000000"]

    active_clcs, is_eccc_update = _eccc_active_clcs_for_update(alert)
    if is_eccc_update:
        matched = [c for c in feed_codes if c in active_clcs]
        return matched[:31] if matched else []

    all_geocodes = alert.all_geocodes
    if _is_national_geocode(all_geocodes):
        return feed_codes[:31]

    matched_codes = _matching_feed_codes(alert, feed)
    if matched_codes:
        return matched_codes

    return []


def _normalize_blocklist(raw_blocklist: Any) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "severity": set(),
        "certainty": set(),
        "urgency": set(),
        "status": set(),
        "scope": set(),
        "naads_events": set(),
        "other": [],
    }

    sources: list[dict[str, Any]]
    if isinstance(raw_blocklist, dict):
        sources = [cast(dict[str, Any], raw_blocklist)]
    elif isinstance(raw_blocklist, list):
        sources = [
            cast(dict[str, Any], raw_entry)
            for raw_entry in cast(list[Any], raw_blocklist)
            if isinstance(raw_entry, dict)
        ]
    else:
        sources = []

    for source in sources:
        for key in ("severity", "certainty", "urgency", "status", "scope", "naads_events"):
            normalized[key].update(_normalize_strings(source.get(key)))

        raw_other = source.get("other")
        if isinstance(raw_other, list):
            for entry in raw_other:
                if isinstance(entry, dict) and "value_name" in entry and "value" in entry:
                    normalized["other"].append(entry)

    return normalized


def _matches_geocode_pattern(pattern: str, raw_geocodes: set[str]) -> bool:
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        return any(code.startswith(prefix) for code in raw_geocodes)

    return pattern in raw_geocodes


def _alert_blocked(alert: CAPAlert, cap_filter: dict[str, Any]) -> tuple[bool, str | None]:
    blocklist = _normalize_blocklist(cap_filter.get("blocklist"))

    scalar_checks = {
        "severity": alert.severity,
        "certainty": alert.certainty,
        "urgency": alert.urgency,
        "status": alert.status,
        "scope": alert.scope,
    }
    for field, value in scalar_checks.items():
        blocked_values: set[str] = blocklist.get(field, set())
        if value and value in blocked_values:
            return True, f"{field}={value}"

    blocked_events: set[str] = blocklist.get("naads_events", set())
    if blocked_events:
        cap_event = alert.cap_cp_event or ""
        alert_event = alert.event
        for blocked in blocked_events:
            blocked_key = _event_key(blocked)
            if (cap_event and _event_key(cap_event) == blocked_key) or \
               (alert_event and _event_key(alert_event) == blocked_key):
                return True, f"naads_event={blocked}"

    other_rules: list[dict[str, str]] = blocklist.get("other", [])
    if other_rules:
        params = alert.param_dict()
        for rule in other_rules:
            vn = rule.get("value_name", "").lower()
            rv = rule.get("value", "")
            if vn and rv:
                actual = params.get(vn, "")
                if actual and actual.lower() == rv.lower():
                    return True, f"other({vn}={rv})"

    return False, None


def _feed_cap_filter(cap_cfg: dict[str, Any]) -> dict[str, Any]:
    raw_filter = cap_cfg.get('filter')
    if isinstance(raw_filter, dict):
        return dict(cast(dict[str, Any], raw_filter))
    return {}


def cap_event_to_same(alert: CAPAlert, event: str) -> str:
    mapping = _load_same_mapping()

    if alert.same_event:
        return alert.same_event

    naads_map = cast(dict[str, str], mapping.get('naadsToEas', {}))
    if event in naads_map:
        return naads_map[event]

    event_key = _event_key(event)
    for code, label in cast(dict[str, str], mapping.get('eas', {})).items():
        if _event_key(label) == event_key:
            return code

    return "ADR"


def _resolve_same_event(alert: CAPAlert) -> str:
    if alert.same_event:
        return alert.same_event

    mapping = _load_same_mapping()
    naads_map = cast(dict[str, str], mapping.get('naadsToEas', {}))

    cap_event = alert.cap_cp_event
    if cap_event and cap_event in naads_map:
        return naads_map[cap_event]

    if alert.infos:
        return cap_event_to_same(alert, alert.infos[0].event)

    return 'ADR'


def _same_originator(alert: CAPAlert) -> str:
    for info in alert.infos:
        eas_org = info.param_dict().get('eas-org', '')
        if eas_org:
            return eas_org.upper()[:3]

    info = alert.infos[0] if alert.infos else None
    if info:
        categories = {c.lower() for c in info.categories}
        if categories & _WXR_CATEGORIES:
            return 'WXR'
        if categories & _CIV_CATEGORIES:
            return 'CIV'

    return 'CIV'


def _duration_code(alert: CAPAlert) -> str:
    info = alert.infos[0] if alert.infos else None
    if not info or not info.expires:
        return "0100"
    try:
        sent = alert.sent or _dt.datetime.now(_dt.UTC)
        expires = info.expires
        delta = expires - sent
        total_min = max(int(delta.total_seconds() / 60), 15)
        hours = min(total_min // 60, 99)
        mins = min(total_min % 60, 59)
        return f"{hours:02d}{mins:02d}"
    except Exception:
        return "0100"


def _attention_tone(alert: CAPAlert, same_event: str, same_cfg: dict[str, Any]) -> str:
    if alert.broadcast_immediately and same_cfg.get('alert_ready_on_NAADS_BIP', False):
        return 'NPAS'

    for override in same_cfg.get('attention_tone_override', []):
        if isinstance(override, dict) and same_event in override:
            return str(override[same_event])

    return str(same_cfg.get('default_attention_tone', 'WXR'))


def _alert_priority(alert: CAPAlert) -> int:
    if alert.broadcast_immediately:
        return 0
    return _SEVERITY_PRIORITY.get(alert.severity, _INFORMATIONAL_PRIORITY)


def _should_schedule_post_alert_package(alert: CAPAlert) -> bool:
    severity = str(alert.severity or '').strip().upper()
    if severity in _SEVERITY_POST_ALERT_ALERTS_PACKAGE:
        return True
    info = alert.infos[0] if alert.infos else None
    if info is None:
        return False
    impact = str(info.param_dict().get('layer:ec-msc-smc:1.1:msc_impact', '') or '').strip().upper()
    return impact in _IMPACT_POST_ALERT_ALERTS_PACKAGE


def _pcm_to_voice_array(pcm: bytes, same_sr: int) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    if CHANNELS == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return resample(samples, BUS_SR, same_sr)


def _info_param(info: CAPInfo, name: str) -> str:
    return str(info.param_dict().get(name.lower(), ''))


def _cap_alert_source(info: CAPInfo) -> str:
    pd = info.param_dict()
    if any(k.startswith('layer:ec-msc-smc') for k in pd):
        return 'eccc'
    sender_lower = info.sender_name.lower()
    if 'weather.gov' in sender_lower or 'national weather service' in sender_lower:
        return 'nws'
    if 'eas-org' in pd:
        return 'nws'
    return 'civil'


def _resolve_cap_areas(
    info: CAPInfo,
    feed: dict[str, Any],
    lang_short: str,
) -> list[str]:
    raw_newly_active = _info_param(info, 'layer:ec-msc-smc:1.1:newly_active_areas')
    if raw_newly_active:
        alert_clcs = [c.strip() for c in raw_newly_active.split(',') if c.strip()]
    else:
        seen_clcs: set[str] = set()
        alert_clcs = []
        for area in info.areas:
            for code in area.clc_codes:
                if code and code not in seen_clcs:
                    seen_clcs.add(code)
                    alert_clcs.append(code)

    feed_patterns = feed_same_codes(feed)

    if not feed_patterns:
        return []

    seen_matched: set[str] = set()
    matched_clcs: list[str] = []
    for clc in alert_clcs:
        for pattern in feed_patterns:
            if pattern.endswith('*'):
                if clc.startswith(pattern[:-1]) and clc not in seen_matched:
                    seen_matched.add(clc)
                    matched_clcs.append(clc)
                    break
            elif clc == pattern and clc not in seen_matched:
                seen_matched.add(clc)
                matched_clcs.append(clc)
                break

    if not matched_clcs:
        return []

    db = _load_forecast_loc_db()
    lang_idx = 1 if lang_short == 'fr' else 0
    seen_names: set[str] = set()
    names: list[str] = []
    for clc in matched_clcs:
        entry = db.get(clc)
        if entry:
            name = entry[lang_idx].replace(' - ', ', ')
            if name and name not in seen_names:
                seen_names.add(name)
                names.append(name)
    return names


def _build_alert_tts_script(
    alert: CAPAlert,
    feed: dict[str, Any],
    lang: str,
) -> str:
    info = alert.info_for_lang(lang)
    if not info:
        return ""

    lang_short = lang[:2]
    ph = _AL_PH.get(lang_short, _AL_PH['en'])
    pd = info.param_dict()
    source_type = _cap_alert_source(info)
    msg_type = alert.msg_type
    now = _dt.datetime.now(_dt.UTC)
    sentences: list[str] = []

    if source_type == 'eccc':
        alert_name = str(pd.get('layer:ec-msc-smc:1.0:alert_name', ''))
        colour     = str(pd.get('layer:ec-msc-smc:1.1:colour', ''))
        alert_type = str(pd.get('layer:ec-msc-smc:1.0:alert_type', ''))
        coverage   = str(pd.get('layer:ec-msc-smc:1.0:alert_coverage', '') or info.event)
        confidence = str(pd.get('layer:ec-msc-smc:1.1:msc_confidence', ''))
        impact     = str(pd.get('layer:ec-msc-smc:1.1:msc_impact', ''))
        loc_status = str(
            pd.get('layer:ec-msc-smc:1.1:alert_location_status', '') or
            pd.get('layer:ec-msc-smc:1.0:alert_location_status', '')
        ).lower()

        subject = _parse_eccc_subject(alert_name, colour, alert_type) if alert_name else info.event.title()
        sender_name = info.sender_name

        areas = _resolve_cap_areas(info, feed, lang_short)
        area_str = _join_areas(areas, lang_short) if areas else coverage

        if loc_status == 'ended':
            sentences.append(ph['eccc_ended'].format(subject=subject, areas=area_str))
        elif msg_type == 'Cancel':
            sentences.append(ph['eccc_cancelled'].format(sender=sender_name, subject=subject, areas=area_str))
        elif msg_type == 'Update':
            sentences.append(ph['eccc_updated'].format(sender=sender_name, subject=subject, areas=area_str))
        else:
            sentences.append(ph['eccc_issued'].format(sender=sender_name, subject=subject, areas=area_str))

        if loc_status != 'ended' and msg_type != 'Cancel':
            onset_dt = info.onset or info.effective
            onset_future = onset_dt is not None and onset_dt > now
            if onset_future and info.expires:
                sentences.append(ph['timing_span'].format(
                    onset=_format_datetime_spoken(onset_dt, lang_short),
                    expires=_format_datetime_spoken(info.expires, lang_short),
                ))
            elif info.expires:
                sentences.append(ph['timing_expires'].format(expires=_format_datetime_spoken(info.expires, lang_short)))
            elif onset_future:
                sentences.append(ph['timing_onset'].format(onset=_format_datetime_spoken(onset_dt, lang_short)))

            if confidence and impact:
                sentences.append(ph['confidence_impact'].format(confidence=confidence, impact=impact))

            if info.description:
                sentences.append(_clean_alert_text(info.description))
            if info.instruction:
                sentences.append(_clean_alert_text(info.instruction))

    elif source_type == 'nws':
        nws_sender = info.sender_name or 'The National Weather Service'
        event_name = info.event.title() if info.event else ''
        nws_area = next((a.description for a in info.areas if a.description), '')
        if nws_area:
            sentences.append(ph['nws_header'].format(sender=nws_sender, event=event_name, areas=nws_area))
        elif event_name:
            sentences.append(ph['generic_issued'].format(sender=nws_sender, event=event_name))
        if info.expires:
            sentences.append(ph['timing_expires'].format(expires=_format_datetime_spoken(info.expires, lang_short)))
        if info.description:
            sentences.append(_clean_alert_text(info.description))
        if info.instruction:
            sentences.append(_clean_alert_text(info.instruction))

    else:
        event_name = str(info.event or info.headline or '').title()
        civil_area = next((a.description for a in info.areas if a.description), '')
        civil_sender = info.sender_name or 'Alert Ready'
        area_str_civil = civil_area or event_name

        if msg_type == 'Cancel':
            sentences.append(ph['civil_cancelled'].format(sender=civil_sender, event=event_name, areas=area_str_civil))
        elif msg_type == 'Update':
            sentences.append(ph['civil_updated'].format(sender=civil_sender, event=event_name, areas=area_str_civil))
        else:
            sentences.append(ph['civil_issued'].format(sender=civil_sender, event=event_name, areas=area_str_civil))

        if info.expires:
            sentences.append(ph['civil_timing'].format(expires=_format_datetime_spoken(info.expires, lang_short)))

        if info.description:
            sentences.append(_clean_alert_text(info.description))
        if info.instruction:
            sentences.append(_clean_alert_text(info.instruction))

    return '  '.join(filter(None, _fit_sentences_to_word_budget(sentences)))


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


def _generate_alert_audio(
    alert: CAPAlert,
    feed: dict[str, Any],
    config: dict[str, Any],
) -> None:
    feed_id = feed['id']
    runtime_entry = _build_alert_entry(feed_id, alert)
    same_cfg = config.get('same', {})
    callsign = str(same_cfg.get('sender') or os.environ.get('SAME_ID', 'HAZE0000'))
    same_sr = SAME_SAMPLE_RATE

    playout_cfg = _feed_playout_cfg(feed)
    include_same = playout_cfg.get('same', True)

    same_event = _resolve_same_event(alert)
    originator = _same_originator(alert)
    duration = _duration_code(alert)
    locations = _header_locations(alert, feed)
    if not locations:
        log.info(
            '[%s] Alert skipped — no matching locations: %s (event=%s)',
            feed_id,
            alert.identifier,
            same_event,
        )
        return
    tone_type = _attention_tone(alert, same_event, same_cfg)
    priority = _alert_priority(alert)
    schedule_post_alert_package = _should_schedule_post_alert_package(alert)

    feed_langs = list(feed.get('languages', {}).keys()) or ['en-CA']

    voice_pcm_parts: list[bytes] = []
    cap_audio_pcm, cap_audio_label = _resolve_cap_audio_pcm(alert, feed_langs)
    if cap_audio_pcm is None:
        for lang in feed_langs:
            msg = _build_alert_tts_script(alert, feed, lang)
            if not msg:
                continue
            pcm = synthesize_pcm(config, msg, lang=lang)
            if pcm:
                voice_pcm_parts.append(pcm)

    if not include_same:
        source_pcm = cap_audio_pcm or (b''.join(voice_pcm_parts) if voice_pcm_parts else None)
        if not source_pcm:
            return
        samples = np.frombuffer(source_pcm, dtype=np.int16).astype(np.float32) / 32767.0
        if CHANNELS == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        alert_pcm = to_pcm16(resample(samples, BUS_SR, BUS_SR))
        if push_alert(feed_id, priority, alert_pcm, alert.identifier):
            store_runtime_alert_entry(feed_id, alert.identifier, runtime_entry)
        saved_path = _save_alert_audio(feed_id, alert.identifier, alert_pcm)
        threading.Thread(
            target=dispatch_webhook,
            args=(feed_id, runtime_entry, same_event, saved_path, config),
            daemon=True,
        ).start()
        if schedule_post_alert_package:
            enqueue_scheduled_package(feed_id, 'alerts')
        log.info(
            '[%s] Alert queued (no SAME): %s (event=%s, priority=%d, audio_source=%s, saved=%s)',
            feed_id,
            alert.identifier,
            same_event,
            priority,
            cap_audio_label or 'tts',
            saved_path,
        )
        return

    voice_array: np.ndarray | None = None
    audio_label = cap_audio_label
    if cap_audio_pcm:
        voice_array = _pcm_to_voice_array(cap_audio_pcm, same_sr)
    elif voice_pcm_parts:
        voice_array = _pcm_to_voice_array(b''.join(voice_pcm_parts), same_sr)
        audio_label = 'tts'

    header = SAMEHeader(
        originator=originator,
        event=same_event,
        locations=tuple(locations),
        duration=duration,
        callsign=callsign,
    )
    runtime_entry.setdefault('source', {})['sameHeader'] = header.encoded

    full_signal = generate_same(
        header=header,
        tone_type=cast(Any, tone_type),
        audio_msg_array=voice_array,
        audio_label=audio_label,
        attn_duration_s=8.0,
    )

    alert_pcm = to_pcm16(resample(full_signal, same_sr, BUS_SR))
    if push_alert(feed_id, priority, alert_pcm, alert.identifier):
        store_runtime_alert_entry(feed_id, alert.identifier, runtime_entry)
    saved_path = _save_alert_audio(feed_id, alert.identifier, alert_pcm)
    threading.Thread(
        target=dispatch_webhook,
        args=(feed_id, runtime_entry, same_event, saved_path, config),
        daemon=True,
    ).start()
    if schedule_post_alert_package:
        enqueue_scheduled_package(feed_id, 'alerts')
    log.info(
        '[%s] SAME queued: %s (event=%s, tone=%s, priority=%d, audio_source=%s, saved=%s)',
        feed_id,
        header.encoded,
        same_event,
        tone_type,
        priority,
        audio_label or 'none',
        saved_path,
    )


class AlertDedup:
    def __init__(
        self,
        window_s: float = 300.0,
        key_fields: list[str] | None = None,
    ) -> None:
        self._window = window_s
        self._key_fields = key_fields or ["identifier", "sent"]
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def _build_key(self, alert: CAPAlert) -> str:
        parts: list[str] = []
        for field in self._key_fields:
            value = getattr(alert, field, "")
            parts.append(value.isoformat() if isinstance(value, _dt.datetime) else str(value))
        return ":".join(parts)

    def is_new(self, alert: CAPAlert) -> bool:
        now = time.monotonic()
        with self._lock:
            self._seen = {k: v for k, v in self._seen.items() if now - v < self._window}
            key = self._build_key(alert)
            if key in self._seen:
                return False
            self._seen[key] = now
            return True


def _alert_geo_matches_feed(
    alert: CAPAlert,
    feed: dict[str, Any],
    cap_filter: dict[str, Any],
) -> tuple[bool, str]:
    all_geocodes = alert.all_geocodes

    if _is_national_geocode(all_geocodes):
        return True, "national coverage"

    use_feed_locs = cap_filter.get("use_feed_locations", True)

    if use_feed_locs:
        if alert.msg_type == "Update":
            active_clcs, is_eccc_update = _eccc_active_clcs_for_update(alert)
            if is_eccc_update:
                feed_codes = feed_same_codes(feed)
                matched = [c for c in feed_codes if c in active_clcs]
                if matched:
                    return True, "eccc update CLC coverage"
                return False, "eccc update outside feed CLC coverage"

        matched_feed_codes = _matching_feed_codes(alert, feed)
        if matched_feed_codes:
            return True, "feed code coverage"
        return False, "outside feed coverage"

    filter_geocodes = _normalize_strings(cap_filter.get("geocodes"))
    if filter_geocodes:
        alert_geocodes = set(all_geocodes)
        if any(_matches_geocode_pattern(pattern, alert_geocodes) for pattern in filter_geocodes):
            return True, "configured geocode filter"
        return False, "no geocode filter match"

    return True, "location filtering disabled"


def _covers_feed_area(
    alert: CAPAlert,
    feed: dict[str, Any],
    cap_filter: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if alert.status not in ("Actual", "Test"):
        return False, f"status={alert.status}"
    if alert.msg_type not in ("Alert", "Update"):
        return False, f"msg_type={alert.msg_type}"
    return _alert_geo_matches_feed(alert, feed, cap_filter or {})


def _alert_expired(alert: CAPAlert) -> bool:
    expiries = [i.expires for i in alert.infos if i.expires is not None]
    return bool(expiries) and all(e <= _dt.datetime.now(_dt.UTC) for e in expiries)


def matches_feed(
    alert: CAPAlert,
    feed: dict[str, Any],
    cap_filter: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    covers, reason = _covers_feed_area(alert, feed, cap_filter)
    if not covers:
        return False, reason
    blocked, block_reason = _alert_blocked(alert, cap_filter or {})
    if blocked:
        return False, block_reason or "blocked"
    return True, reason


async def alert_worker(
    config: dict[str, Any],
    feeds: list[dict[str, Any]],
    shutdown: asyncio.Event,
) -> None:
    def _dispatch_alert(
        alert: CAPAlert,
        feed_alert_key: str,
        dedup: AlertDedup,
        source_name: str,
        suppress_audio: bool = False,
    ) -> None:
        if not dedup.is_new(alert):
            log.debug("Duplicate %s alert skipped: %s", source_name, alert.identifier)
            return

        if _alert_expired(alert):
            log.debug("Expired %s alert skipped: %s", source_name, alert.identifier)
            return

        if alert.msg_type == 'Cancel' and alert.references:
            for feed in feeds:
                cap_cfg = feed.get("alerts", {}).get(feed_alert_key, {})
                if not cap_cfg.get("enabled", True):
                    continue
                effective_filter = _feed_cap_filter(cap_cfg)
                geo_match, _ = _alert_geo_matches_feed(alert, feed, effective_filter)
                if not geo_match:
                    continue
                _write_registry_entry(feed["id"], alert)
            return

        area_matched_any = False
        for feed in feeds:
            feed_id = feed["id"]

            cap_cfg = feed.get("alerts", {}).get(feed_alert_key, {})
            if not cap_cfg.get("enabled", True):
                continue

            effective_filter = _feed_cap_filter(cap_cfg)

            matched, match_reason = matches_feed(alert, feed, effective_filter)
            if not matched:
                log.info(
                    "[%s] %s alert filtered: %s — %s (%s; %s)",
                    feed_id,
                    source_name,
                    alert.event,
                    alert.headline,
                    match_reason,
                    _match_log_context(alert, feed, effective_filter),
                )
                continue

            area_matched_any = True
            is_new = _write_registry_entry(feed_id, alert)
            update_feed_runtime(feed_id, {
                'last_alert_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'last_alert_event': alert.event,
                'last_alert_headline': alert.headline,
                'last_alert_severity': alert.severity,
            })
            if is_new:
                append_runtime_event(
                    'alert',
                    f'{alert.event}: {alert.headline}',
                    feed_id,
                    {'severity': alert.severity},
                )
                log.info("[%s] %s alert registered: %s — %s (%s)", feed_id, source_name, alert.event, alert.headline, match_reason)
                if not suppress_audio:
                    try:
                        _generate_alert_audio(alert, feed, config)
                    except Exception:
                        log.exception('[%s] SAME generation failed for %s', feed_id, alert.identifier)
                else:
                    log.debug('[%s] Archive alert registered without audio: %s', feed_id, alert.identifier)
            else:
                log.debug("[%s] %s alert already in registry, skipping SAME: %s", feed_id, source_name, alert.identifier)

        if not area_matched_any:
            log.info(
                "No %s feeds accepted alert %s (%s, severity=%s, urgency=%s, certainty=%s, scope=%s, alert_geocodes=%s, alert_clc=%s)",
                source_name,
                alert.identifier,
                alert.event,
                alert.severity,
                alert.urgency,
                alert.certainty,
                alert.scope,
                _format_log_codes(alert.all_geocodes),
                _format_log_codes(alert.clc_codes),
            )

    cap_cp_cfg = config.get("cap", {}).get("cap_cp", {})
    nws_cfg = config.get("cap", {}).get("nws_cap", {})
    tasks: list[asyncio.Task[None]] = []

    if cap_cp_cfg.get("enabled", False):
        dedup_cfg = cap_cp_cfg.get("dedup", {})
        cap_cp_dedup = AlertDedup(
            window_s=dedup_cfg.get("window_seconds", 300),
            key_fields=dedup_cfg.get("key_fields"),
        )
        async def on_cap_cp_alert(alert: CAPAlert) -> None:
            await asyncio.to_thread(_dispatch_alert, alert, 'cap_cp', cap_cp_dedup, 'CAP-CP')

        async def on_cap_cp_alert_archive(alert: CAPAlert) -> None:
            await asyncio.to_thread(_dispatch_alert, alert, 'cap_cp', cap_cp_dedup, 'CAP-CP', True)

        naads_sources = cap_cp_cfg.get("sources", [])
        if naads_sources:
            try:
                count = await naad_archive_fetch(naads_sources, on_cap_cp_alert_archive)
                log.info("NAADS startup archive: %d active alert(s) fetched (audio suppressed)", count)
            except Exception as exc:
                log.warning("NAADS startup archive fetch failed: %s", exc)

        tasks.append(asyncio.create_task(naad_listener(on_cap_cp_alert, shutdown), name='cap_cp_listener'))
    else:
        log.info("CAP-CP alerting disabled")

    if nws_cfg.get("enabled", False):
        dedup_cfg = nws_cfg.get("dedup", {})
        nws_dedup = AlertDedup(
            window_s=dedup_cfg.get("window_seconds", 300),
            key_fields=dedup_cfg.get("key_fields"),
        )
        async def on_nws_alert(alert: CAPAlert, source_cfg: dict[str, Any]) -> None:
            _ = source_cfg
            await asyncio.to_thread(_dispatch_alert, alert, 'nws', nws_dedup, 'NWS CAP')

        tasks.append(asyncio.create_task(nws_atom_listener(config, on_nws_alert, shutdown), name='nws_cap_listener'))
    else:
        log.info("NWS CAP alerting disabled")

    if not tasks:
        return

    await asyncio.gather(*tasks)
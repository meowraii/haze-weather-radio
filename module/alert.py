from __future__ import annotations

import asyncio
import csv
import datetime as _dt
import json
import logging
import pathlib
import re
import threading
import time
from datetime import datetime
from typing import Any, cast

import numpy as np

_REGISTRY_PATH = pathlib.Path('data') / 'alertsRegistry.json'
_registry_lock = threading.Lock()


def _parse_cap_references(references: str) -> list[str]:
    ids: list[str] = []
    for ref in references.strip().split():
        parts = ref.split(',')
        if len(parts) >= 2:
            ids.append(parts[1])
    return ids


def _write_registry_entry(feed_id: str, alert: Any) -> bool:
    identifier = alert.identifier if hasattr(alert, 'identifier') else str(alert)
    if not identifier:
        return False
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _registry_lock:
        try:
            registry: list[dict] = json.loads(_REGISTRY_PATH.read_text(encoding='utf-8')) if _REGISTRY_PATH.exists() else []
        except Exception:
            registry = []
        if any(r.get('identifier') == identifier and r.get('feed_id') == feed_id for r in registry):
            return False

        referenced_ids: set[str] = set()
        raw_refs = getattr(alert, 'references', '') or ''
        if raw_refs and getattr(alert, 'msg_type', '') in ('Update', 'Cancel'):
            referenced_ids = set(_parse_cap_references(raw_refs))

        if referenced_ids:
            before = len(registry)
            registry = [
                r for r in registry
                if not (r.get('feed_id') == feed_id and r.get('identifier') in referenced_ids)
            ]
            removed = before - len(registry)
            if removed:
                log.debug('Registry: removed %d superseded entry/entries for feed %s', removed, feed_id)

        if getattr(alert, 'msg_type', '') == 'Cancel':
            try:
                _REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding='utf-8')
                log.debug('Registry: cancelled entries removed for %s', identifier)
            except Exception as e:
                log.error('Failed to write alert registry: %s', e)
            return False

        info = None
        if hasattr(alert, 'infos') and alert.infos:
            info = alert.infos[0]

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

        entry: dict[str, Any] = {
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

        registry.append(entry)
        try:
            _REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding='utf-8')
            log.debug('Registry updated: %s', identifier)
            return True
        except Exception as e:
            log.error('Failed to write alert registry: %s', e)
            return False


from managed.events import append_runtime_event, push_alert, update_feed_runtime
from module.cap_specific.naads_tcp import CAPAlert, naad_listener
from module.cap_specific.naadsatom_http import naad_archive_fetch
from module.cap_specific.nwsatom_http import nws_atom_listener
from module.buffer import CHANNELS, SAMPLE_RATE as BUS_SR
from module.same import SAMEHeader, generate_same, resample, to_pcm16
from module.tts import synthesize_pcm

log = logging.getLogger(__name__)

_SAME_MAPPING_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'sameMapping.json'

_same_mapping_cache: dict[str, Any] | None = None
_EVENT_NORMALIZE_RE = re.compile(r'[^a-z0-9]+')

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

_INFORMATIONAL_PRIORITY = 9

_MANAGED = pathlib.Path(__file__).parent.parent / "managed"

_geocode_db: dict[str, tuple[str, str]] | None = None
_forecast_db: dict[str, str] | None = None

_aggregate_clc_to_geocodes: dict[str, set[str]] | None = None
_aggregate_geocode_to_clcs: dict[str, set[str]] | None = None
_aggregate_clc_to_province: dict[str, str] | None = None


def _load_aggregate_db() -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, str]]:
    global _aggregate_clc_to_geocodes, _aggregate_geocode_to_clcs, _aggregate_clc_to_province
    if _aggregate_clc_to_geocodes is not None:
        return _aggregate_clc_to_geocodes, _aggregate_geocode_to_clcs, _aggregate_clc_to_province  # type: ignore[return-value]
    path = _MANAGED / "AGGREGATE_LOCATION_CODES.csv"
    clc_to_geo: dict[str, set[str]] = {}
    geo_to_clc: dict[str, set[str]] = {}
    clc_to_prov: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 10:
                continue
            clc = row[6].strip().strip('"')
            province = row[5].strip().strip('"')
            geocode = row[9].strip().strip('"')
            if not clc or not geocode:
                continue
            clc_to_geo.setdefault(clc, set()).add(geocode)
            geo_to_clc.setdefault(geocode, set()).add(clc)
            if clc not in clc_to_prov and province:
                clc_to_prov[clc] = province
    _aggregate_clc_to_geocodes = clc_to_geo
    _aggregate_geocode_to_clcs = geo_to_clc
    _aggregate_clc_to_province = clc_to_prov
    log.debug("Loaded aggregate location DB: %d CLCs, %d geocodes", len(clc_to_geo), len(geo_to_clc))
    return clc_to_geo, geo_to_clc, clc_to_prov


def _load_geocode_db() -> dict[str, tuple[str, str]]:
    global _geocode_db
    if _geocode_db is not None:
        return _geocode_db
    _geocode_db = {}
    log.debug("_load_geocode_db: stub — use _load_aggregate_db()")
    return _geocode_db


def _load_forecast_db() -> dict[str, str]:
    global _forecast_db
    if _forecast_db is not None:
        return _forecast_db
    _, _, clc_to_prov = _load_aggregate_db()
    _forecast_db = clc_to_prov
    log.debug("Loaded %d forecast locations from aggregate DB", len(_forecast_db))
    return _forecast_db


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
    codes: list[str] = []
    seen: set[str] = set()

    def _raw_codes(loc: dict[str, Any]) -> list[str]:
        explicit_same = str(loc.get('same') or '').strip()
        if explicit_same:
            return [part.strip() for part in explicit_same.split(',') if part.strip()]

        raw = str(loc.get('forecast_region') or '').strip()
        if not raw:
            return []
        if '-' in raw:
            left, right = raw.split('-', 1)
            if left.replace('*', '').isdigit() and right.replace('*', '').isdigit():
                raw = right
        return [part.strip() for part in raw.split(',') if part.strip()]

    for block in feed.get("locations", []):
        if not isinstance(block, dict):
            continue
        for loc in block.get("forecastLocations", []):
            if not isinstance(loc, dict):
                continue
            for target in _raw_codes(loc):
                if target.endswith("*") and target[:-1].isdigit():
                    prefix = target[:-1]
                    for code in sorted(db):
                        if code.startswith(prefix) and code not in seen:
                            seen.add(code)
                            codes.append(code)
                    continue
                if target and target not in seen:
                    seen.add(target)
                    codes.append(target)
    return codes


def _event_key(value: str) -> str:
    return _EVENT_NORMALIZE_RE.sub('', value.lower())


def _matches_feed_alert_code(pattern: str, alert_codes: set[str]) -> bool:
    if not pattern:
        return False
    if pattern.endswith('*'):
        prefix = pattern[:-1]
        return any(code.startswith(prefix) for code in alert_codes)
    return pattern in alert_codes


def _matching_feed_codes(alert: CAPAlert, feed: dict[str, Any]) -> list[str]:
    feed_codes = feed_same_codes(feed)
    if not feed_codes:
        return []

    alert_codes = set(alert.all_geocodes) | set(alert.clc_codes)

    direct_matches = [code for code in feed_codes if _matches_feed_alert_code(code, alert_codes)]
    if direct_matches:
        return direct_matches[:31]

    feed_geocodes = _feed_clc_geocodes(feed_codes)
    if feed_geocodes:
        alert_geocodes = set(alert.all_geocodes)
        if alert_geocodes & feed_geocodes:
            _, geo_to_clc, _ = _load_aggregate_db()
            matched_clcs: list[str] = []
            seen: set[str] = set()
            for gc in alert_geocodes & feed_geocodes:
                for clc in geo_to_clc.get(gc, ()):
                    if clc in set(feed_codes) and clc not in seen:
                        seen.add(clc)
                        matched_clcs.append(clc)
            if matched_clcs:
                return matched_clcs[:31]
            return feed_codes[:31]

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

    all_geocodes = alert.all_geocodes
    if _is_national_geocode(all_geocodes):
        return feed_codes[:31]

    matched_codes = _matching_feed_codes(alert, feed)
    if matched_codes:
        return matched_codes

    return feed_codes[:31]


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
    current_values = {
        "severity": alert.severity,
        "certainty": alert.certainty,
        "urgency": alert.urgency,
        "status": alert.status,
        "scope": alert.scope,
    }
    for key in ("severity", "certainty", "urgency", "status", "scope"):
        blocked_values = blocklist.get(key, set())
        value = current_values.get(key, "")
        if value and value in blocked_values:
            return True, f"{key}={value}"

    blocked_events: set[str] = blocklist.get("naads_events", set())
    if blocked_events:
        cap_event = alert.cap_cp_event or ""
        alert_event = alert.event
        for blocked in blocked_events:
            blocked_lower = blocked.lower()
            if (cap_event and _event_key(cap_event) == _event_key(blocked_lower)) or \
               (alert_event and _event_key(alert_event) == _event_key(blocked_lower)):
                return True, f"naads_event={blocked}"

    other_rules: list[dict[str, str]] = blocklist.get("other", [])
    if other_rules:
        params = alert.param_dict()
        for rule in other_rules:
            vn = rule.get("value_name", "")
            rv = rule.get("value", "")
            if vn and rv:
                actual = params.get(vn, "")
                if actual and actual.lower() == rv.lower():
                    return True, f"other({vn}={rv})"

    return False, None


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


def _pcm_to_voice_array(pcm: bytes, same_sr: int) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    if CHANNELS == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return resample(samples, BUS_SR, same_sr)


def _generate_alert_audio(
    alert: CAPAlert,
    feed: dict[str, Any],
    config: dict[str, Any],
) -> None:
    feed_id = feed['id']
    same_cfg = config.get('same', {})
    callsign = str(same_cfg.get('sender', 'HAZE0000'))
    same_sr = int(same_cfg.get('sample_rate_hz', 22050))

    same_event = _resolve_same_event(alert)
    originator = _same_originator(alert)
    duration = _duration_code(alert)
    locations = _header_locations(alert, feed)
    tone_type = _attention_tone(alert, same_event, same_cfg)
    priority = _alert_priority(alert)

    feed_langs = list(feed.get('languages', {}).keys()) or ['en-CA']

    voice_pcm_parts: list[bytes] = []
    for lang in feed_langs:
        info = alert.info_for_lang(lang)
        if not info:
            continue
        msg = info.description or info.headline
        if not msg:
            continue
        pcm = synthesize_pcm(config, msg, lang=lang)
        if pcm:
            voice_pcm_parts.append(pcm)

    voice_array: np.ndarray | None = None
    if voice_pcm_parts:
        voice_array = _pcm_to_voice_array(b''.join(voice_pcm_parts), same_sr)

    header = SAMEHeader(
        originator=originator,
        event=same_event,
        locations=locations,
        duration=duration,
        callsign=callsign,
    )

    full_signal = generate_same(
        header=header,
        tone_type=tone_type,
        audio_msg_array=voice_array,
        sample_rate=same_sr,
        attn_duration_s=8.0,
    )

    alert_pcm = to_pcm16(resample(full_signal, same_sr, BUS_SR))
    push_alert(feed_id, priority, alert_pcm, alert.identifier)
    log.info(
        '[%s] SAME queued: %s (event=%s, tone=%s, priority=%d)',
        feed_id, header.encoded, same_event, tone_type, priority,
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

    def _build_key(self, alert: CAPAlert) -> str:
        parts: list[str] = []
        for field in self._key_fields:
            value = getattr(alert, field, "")
            parts.append(value.isoformat() if isinstance(value, _dt.datetime) else str(value))
        return ":".join(parts)

    def is_new(self, alert: CAPAlert) -> bool:
        now = time.monotonic()
        self._seen = {k: v for k, v in self._seen.items() if now - v < self._window}
        key = self._build_key(alert)
        if key in self._seen:
            return False
        self._seen[key] = now
        return True


def _covers_feed_area(
    alert: CAPAlert,
    feed: dict[str, Any],
    cap_filter: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    if alert.status not in ("Actual", "Test"):
        return False, f"status={alert.status}"
    if alert.msg_type not in ("Alert", "Update"):
        return False, f"msg_type={alert.msg_type}"

    if cap_filter is None:
        cap_filter = {}

    use_feed_locs = cap_filter.get("use_feed_locations", True)
    all_geocodes = alert.all_geocodes
    alert_geocodes = set(all_geocodes)

    if _is_national_geocode(all_geocodes):
        return True, "national coverage"

    if use_feed_locs:
        matched_feed_codes = _matching_feed_codes(alert, feed)
        if matched_feed_codes:
            return True, "feed code coverage"

        feed_geocodes = _feed_clc_geocodes(feed_same_codes(feed))
        if feed_geocodes and alert_geocodes & feed_geocodes:
            return True, "aggregate geocode coverage"

        return False, "outside feed coverage"

    filter_geocodes = _normalize_strings(cap_filter.get("geocodes"))
    if filter_geocodes:
        if any(_matches_geocode_pattern(pattern, alert_geocodes) for pattern in filter_geocodes):
            return True, "configured geocode filter"
        return False, "no geocode filter match"

    return True, "filters disabled"


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
        cap_filter: dict[str, Any],
        dedup: AlertDedup,
        source_name: str,
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
                _write_registry_entry(feed["id"], alert)
            return

        area_matched_any = False
        for feed in feeds:
            feed_id = feed["id"]

            cap_cfg = feed.get("alerts", {}).get(feed_alert_key, {})
            if not cap_cfg.get("enabled", True):
                continue

            covers, cover_reason = _covers_feed_area(alert, feed, cap_filter)
            if not covers:
                log.info(
                    "[%s] %s alert outside area: %s — %s (%s; %s)",
                    feed_id,
                    source_name,
                    alert.event,
                    alert.headline,
                    cover_reason,
                    _match_log_context(alert, feed, cap_filter),
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
                log.info("[%s] %s alert registered: %s — %s (%s)", feed_id, source_name, alert.event, alert.headline, cover_reason)
                try:
                    _generate_alert_audio(alert, feed, config)
                except Exception:
                    log.exception('[%s] SAME generation failed for %s', feed_id, alert.identifier)
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
        cap_filter = cap_cp_cfg.get("filter", {})

        async def on_cap_cp_alert(alert: CAPAlert) -> None:
            await asyncio.to_thread(_dispatch_alert, alert, 'cap_cp', cap_filter, cap_cp_dedup, 'CAP-CP')

        naads_sources = cap_cp_cfg.get("sources", [])
        if naads_sources:
            try:
                count = await naad_archive_fetch(naads_sources, on_cap_cp_alert)
                log.info("NAADS startup archive: %d active alert(s) fetched", count)
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
        nws_filter = nws_cfg.get("filter", {"use_feed_locations": True})

        async def on_nws_alert(alert: CAPAlert, source_cfg: dict[str, Any]) -> None:
            source_filter = dict(nws_filter)
            if 'use_feed_locations' in source_cfg:
                source_filter['use_feed_locations'] = bool(source_cfg.get('use_feed_locations'))
            await asyncio.to_thread(_dispatch_alert, alert, 'nws', source_filter, nws_dedup, 'NWS CAP')

        tasks.append(asyncio.create_task(nws_atom_listener(config, on_nws_alert, shutdown), name='nws_cap_listener'))
    else:
        log.info("NWS CAP alerting disabled")

    if not tasks:
        return

    await asyncio.gather(*tasks)
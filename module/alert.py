from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import pathlib
import time
import wave
from datetime import datetime
from typing import Any, cast

from managed.events import append_runtime_event, push_alert, update_feed_runtime
from module.naads import CAPAlert, CAPInfo, naad_listener
from module.queue import SAMPLE_RATE as BUS_SR
from module.same import (
    SAMEHeader,
    convert_time_code,
    generate_same,
    to_pcm16,
)
from module.tts import synthesize

log = logging.getLogger(__name__)

_SAME_MAPPING_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'sameMapping.json'

_same_mapping_cache: dict[str, Any] | None = None
_cap_to_same_cache: dict[str, str] | None = None


def _load_same_mapping() -> tuple[dict[str, Any], dict[str, str]]:
    global _same_mapping_cache, _cap_to_same_cache
    if _same_mapping_cache is not None and _cap_to_same_cache is not None:
        return _same_mapping_cache, _cap_to_same_cache
    with open(_SAME_MAPPING_PATH, encoding='utf-8') as f:
        mapping: dict[str, Any] = json.load(f)
    cap_to_same: dict[str, str] = {}
    for code, entry in mapping.items():
        for event in entry.get('naadsEvent', []):
            cap_to_same[event] = code
    _same_mapping_cache = mapping
    _cap_to_same_cache = cap_to_same
    log.debug('Loaded SAME mapping: %d codes, %d CAP events', len(mapping), len(cap_to_same))
    return mapping, cap_to_same

_SEVERITY_PRIORITY: dict[str, int] = {
    "Extreme": 0,
    "Severe": 1,
    "Moderate": 2,
    "Minor": 3,
    "Unknown": 4,
}

_MANAGED = pathlib.Path(__file__).parent.parent / "managed"

_PT_TO_PROVINCE: dict[str, str] = {
    '10': 'NL', '11': 'PE', '12': 'NS', '13': 'NB',
    '24': 'QC', '35': 'ON', '46': 'MB', '47': 'SK',
    '48': 'AB', '59': 'BC', '60': 'YT', '61': 'NT', '62': 'NU',
}

_geocode_db: dict[str, tuple[str, str]] | None = None
_forecast_db: dict[str, str] | None = None


def _load_geocode_db() -> dict[str, tuple[str, str]]:
    global _geocode_db
    if _geocode_db is not None:
        return _geocode_db
    path = _MANAGED / "CAP-CP_GEOCODES.csv"
    db: dict[str, tuple[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 3:
                code = row[0].strip()
                scale = row[1].strip()
                name = row[2].strip()
                if code and scale:
                    db[code] = (scale, name)
    _geocode_db = db
    log.debug("Loaded %d CAP-CP geocodes", len(db))
    return db


def _load_forecast_db() -> dict[str, str]:
    global _forecast_db
    if _forecast_db is not None:
        return _forecast_db
    path = _MANAGED / "FORECAST_LOCATIONS.csv"
    db: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        next(reader, None)
        for row in reader:
            if len(row) >= 6:
                code = row[0].strip()
                prov = row[5].strip().strip('"').split(",")[0].strip()
                if code and prov and code not in db:
                    db[code] = prov
    _forecast_db = db
    log.debug("Loaded %d forecast locations", len(db))
    return db


def _geocode_province(geocode: str) -> str | None:
    if not geocode or geocode[0] == '0':
        return None
    for pt_code, abbr in _PT_TO_PROVINCE.items():
        if geocode.startswith(pt_code):
            return abbr
    return None


def _clc_province(clc_code: str) -> str | None:
    db = _load_forecast_db()
    return db.get(clc_code)


def _geocode_provinces(geocodes: tuple[str, ...] | list[str]) -> set[str]:
    provinces: set[str] = set()
    for gc in geocodes:
        prov = _geocode_province(gc)
        if prov:
            provinces.add(prov)
    return provinces


def _geocodes_cover_clc(geocodes: tuple[str, ...] | list[str], clc_code: str) -> bool:
    clc_prov = _clc_province(clc_code)
    if not clc_prov:
        return False
    for gc in geocodes:
        prov = _geocode_province(gc)
        if prov and prov == clc_prov:
            return True
        if gc.startswith('0'):
            wbd_clc = gc + '0' if len(gc) == 5 else None
            if wbd_clc and wbd_clc == clc_code:
                return True
            if len(gc) <= 3:
                wb_prefix = gc.zfill(3)
                if clc_code.startswith(wb_prefix):
                    return True
    return False


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
    for loc in feed.get("locations", []):
        same = loc.get("same")
        if not same:
            continue
        raw = str(same).strip()
        if raw.endswith("*"):
            prefix = raw[:-1]
            for code in sorted(db):
                if code.startswith(prefix) and code not in seen:
                    seen.add(code)
                    codes.append(code)
        else:
            if raw not in seen:
                seen.add(raw)
                codes.append(raw)
    return codes


def _feed_provinces(feed: dict[str, Any]) -> set[str]:
    provinces: set[str] = set()
    for loc in feed.get("locations", []):
        ps = loc.get("province_state")
        if ps:
            provinces.add(str(ps).upper())
    return provinces


def _header_locations(alert: CAPAlert, feed: dict[str, Any]) -> list[str]:
    feed_codes = feed_same_codes(feed)

    if not feed_codes:
        return ["000000"]

    all_geocodes = alert.all_geocodes
    if _is_national_geocode(all_geocodes):
        return feed_codes[:31]

    alert_clc = alert.clc_codes
    if alert_clc:
        feed_set = set(feed_codes)
        matched = [code for code in alert_clc if code in feed_set]
        if matched:
            return matched[:31]

    alert_provs = _geocode_provinces(all_geocodes)
    feed_provs = _feed_provinces(feed)

    if alert_provs & feed_provs:
        matched = [
            code for code in feed_codes
            if _geocodes_cover_clc(all_geocodes, code)
               or _clc_province(code) in alert_provs
        ]
        return matched[:31] if matched else feed_codes[:31]

    return feed_codes[:31]


def _normalize_blocklist(raw_blocklist: Any) -> dict[str, set[str]]:
    normalized: dict[str, set[str]] = {
        "severity": set(),
        "certainty": set(),
        "urgency": set(),
        "status": set(),
        "scope": set(),
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
        for key in normalized:
            normalized[key].update(_normalize_strings(source.get(key)))

    return normalized


def _matches_geocode_pattern(pattern: str, raw_geocodes: set[str], alert_provinces: set[str]) -> bool:
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        return any(code.startswith(prefix) for code in raw_geocodes)

    if pattern in raw_geocodes:
        return True

    pattern_prov = _geocode_province(pattern)
    if pattern_prov and pattern_prov in alert_provinces:
        return True

    return False


def _alert_blocked(alert: CAPAlert, cap_filter: dict[str, Any]) -> tuple[bool, str | None]:
    blocklist = _normalize_blocklist(cap_filter.get("blocklist"))
    current_values = {
        "severity": alert.severity,
        "certainty": alert.certainty,
        "urgency": alert.urgency,
        "status": alert.status,
        "scope": alert.scope,
    }
    for key, blocked_values in blocklist.items():
        value = current_values.get(key, "")
        if value and value in blocked_values:
            return True, f"{key}={value}"

    return False, None


def cap_event_to_same(event: str) -> str:
    _, cap_to_same = _load_same_mapping()
    return cap_to_same.get(event, 'CEM')


def _resolve_same_event(alert: CAPAlert) -> str:
    if alert.same_event:
        return alert.same_event
    return cap_event_to_same(alert.infos[0].event) if alert.infos else 'CEM'


def _same_originator(alert: CAPAlert) -> str:
    info = alert.infos[0] if alert.infos else None
    if not info:
        return "WXR"
    same_code = _resolve_same_event(alert)
    wx_codes = {
        "SVR", "TOR", "FFW", "FLW", "WSW", "BZW", "HUW", "TRW", "TSW", "SSW",
        "AVW", "DSW", "EQW", "VOW", "WFW", "SPS", "SVS", "HWW", "SMW", "FSW",
        "IBW", "LSW",
    }
    if same_code in wx_codes:
        return "WXR"
    ev = info.event
    if ev in ("civilEmerg", "civil", "amber"):
        return "CIV"
    return "EAS"


def _duration_code(alert: CAPAlert) -> str:
    info = alert.infos[0] if alert.infos else None
    if not info or not info.expires:
        return "0100"
    try:
        sent = datetime.fromisoformat(alert.sent)
        expires = datetime.fromisoformat(info.expires)
        delta = expires - sent
        total_min = max(int(delta.total_seconds() / 60), 15)
        hours = min(total_min // 60, 99)
        mins = min(total_min % 60, 59)
        return f"{hours:02d}{mins:02d}"
    except Exception:
        return "0100"


def _get_audio_resource(info: CAPInfo) -> bytes | None:
    for res in info.resources:
        if res.mime_type.startswith("audio/") and res.data:
            return res.data
    return None


def _attention_tone_type(alert: CAPAlert, config: dict[str, Any]) -> str:
    same_cfg = config.get("same", {})

    if same_cfg.get("alert_ready_on_NAADS_BIP", False) and alert.broadcast_immediately:
        return "NPAS"

    same_event = _resolve_same_event(alert)
    overrides = cast(list[dict[str, Any]], same_cfg.get("attention_tone_override") or [])
    for entry in overrides:
        for code, tone in entry.items():
            if code == same_event:
                return str(tone).upper()

    return same_cfg.get("default_attention_tone", "WXR").upper()


def build_alert_audio(
    alert: CAPAlert,
    config: dict[str, Any],
    feed: dict[str, Any],
) -> pathlib.Path | None:
    feed_id = feed["id"]
    info = alert.infos[0] if alert.infos else None
    if not info:
        return None

    same_event = _resolve_same_event(alert)
    clc_codes = _header_locations(alert, feed)

    same_cfg = config.get("same", {})
    callsign = same_cfg.get("sender", "HAZE0000")

    header = SAMEHeader(
        originator=_same_originator(alert),
        event=same_event,
        locations=clc_codes,
        duration=_duration_code(alert),
        callsign=callsign,
        issue_time=convert_time_code(alert.sent),
    )

    tone_type = _attention_tone_type(alert, config)
    voice_path = _generate_voice_wav(alert, config, feed)

    same_sr = config.get('same', {}).get('sample_rate_hz', 22050)
    full_signal = generate_same(
        header=header,
        tone_type=tone_type,
        audio_msg_fp32=voice_path,
        sample_rate=same_sr,
        attn_duration_s=8.0,
    )

    out_dir = pathlib.Path("output") / feed_id / "alerts"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = alert.identifier.replace("/", "_").replace("\\", "_")[:80]
    out_path = out_dir / f"{safe_id}.wav"
    tmp_path = out_dir / f"{safe_id}.tmp.wav"

    with wave.open(str(tmp_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(same_sr)
        wf.writeframes(to_pcm16(full_signal))

    os.replace(str(tmp_path), str(out_path))
    log.info("[%s] Alert audio generated: %s (%.1fs)", feed_id, out_path.name, len(full_signal) / same_sr)
    return out_path


def _generate_voice_wav(
    alert: CAPAlert,
    config: dict[str, Any],
    feed: dict[str, Any],
) -> pathlib.Path | None:
    feed_id = feed["id"]
    langs = list(feed.get("languages", {}).keys())
    primary_lang = langs[0] if langs else "en-CA"

    if alert.broadcast_immediately:
        bi_infos = [i for i in alert.infos if i.parameters.get("layer:sorem:1.0:broadcast_immediately", "").lower() == "yes" and i.language.startswith(primary_lang[:2])]
        if not bi_infos:
            bi_infos = [i for i in alert.infos if i.parameters.get("layer:sorem:1.0:broadcast_immediately", "").lower() == "yes"]
        info = bi_infos[0] if bi_infos else alert.info_for_lang(primary_lang[:2])
    else:
        info = alert.info_for_lang(primary_lang[:2])
    if not info:
        return None

    embedded = _get_audio_resource(info)
    if embedded:
        try:
            return _write_embedded_audio(embedded, feed_id, alert.identifier)
        except Exception:
            log.warning("[%s] Failed to decode embedded alert audio, falling back to TTS", feed_id)

    text_parts: list[str] = []
    if info.headline:
        text_parts.append(info.headline)
    if info.description:
        text_parts.append(info.description)
    if info.instruction:
        text_parts.append(info.instruction)
    if info.areas:
        text_parts.append("Affected areas: " + ", ".join(info.areas) + ".")

    if not text_parts:
        return None

    text = " ".join(text_parts)
    return synthesize(config, text, feed_id, f"alert_{alert.identifier[:40]}", lang=primary_lang)


def _write_embedded_audio(data: bytes, feed_id: str, identifier: str) -> pathlib.Path:
    import subprocess
    out_dir = pathlib.Path("output") / feed_id / "alerts"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = identifier.replace("/", "_").replace("\\", "_")[:60]
    out_path = out_dir / f"{safe_id}.voice.wav"
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", "pipe:0",
            "-ar", str(BUS_SR), "-ac", "1",
            str(out_path),
        ],
        input=data,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr[:200]}")
    return out_path


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
            parts.append(getattr(alert, field, ""))
        return ":".join(parts)

    def is_new(self, alert: CAPAlert) -> bool:
        now = time.monotonic()
        self._seen = {k: v for k, v in self._seen.items() if now - v < self._window}
        key = self._build_key(alert)
        if key in self._seen:
            return False
        self._seen[key] = now
        return True


def matches_feed(
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

    blocked, reason = _alert_blocked(alert, cap_filter)
    if blocked:
        return False, reason or "blocked"

    use_feed_locs = cap_filter.get("use_feed_locations", True)
    all_geocodes = alert.all_geocodes
    alert_geocodes = set(all_geocodes)
    alert_provinces = _geocode_provinces(all_geocodes)

    if _is_national_geocode(all_geocodes):
        return True, "national coverage"

    if use_feed_locs:
        feed_provs = _feed_provinces(feed)
        if feed_provs and alert_provinces & feed_provs:
            return True, "feed province coverage"
        feed_codes = feed_same_codes(feed)
        if feed_codes and any(_geocodes_cover_clc(all_geocodes, code) for code in feed_codes):
            return True, "feed SAME coverage"

    filter_geocodes = _normalize_strings(cap_filter.get("geocodes"))
    if filter_geocodes:
        if any(_matches_geocode_pattern(pattern, alert_geocodes, alert_provinces) for pattern in filter_geocodes):
            return True, "configured geocode filter"
        return False, "no geocode filter match"

    if not use_feed_locs and not filter_geocodes:
        return True, "filters disabled"

    return False, "outside feed coverage"


async def alert_worker(
    config: dict[str, Any],
    feeds: list[dict[str, Any]],
    shutdown: asyncio.Event,
) -> None:
    dedup_cfg = config.get("cap", {}).get("cap_cp", {}).get("dedup", {})
    dedup = AlertDedup(
        window_s=dedup_cfg.get("window_seconds", 300),
        key_fields=dedup_cfg.get("key_fields"),
    )
    cap_filter = config.get("cap", {}).get("cap_cp", {}).get("filter", {})

    async def on_alert(alert: CAPAlert) -> None:
        if not dedup.is_new(alert):
            log.debug("Duplicate alert skipped: %s", alert.identifier)
            return

        matched_any = False
        for feed in feeds:
            feed_id = feed["id"]

            cap_cfg = feed.get("alerts", {}).get("cap_cp", {})
            if not cap_cfg.get("enabled", False):
                continue

            matched, reason = matches_feed(alert, feed, cap_filter)
            if not matched:
                log.info("[%s] Alert skipped: %s — %s (%s)", feed_id, alert.event, alert.headline, reason)
                continue

            matched_any = True
            log.info("[%s] Alert matched: %s — %s (%s)", feed_id, alert.event, alert.headline, reason)

            wav_path = await asyncio.to_thread(build_alert_audio, alert, config, feed)
            if wav_path:
                priority = _SEVERITY_PRIORITY.get(alert.severity, 3)
                push_alert(feed_id, priority, wav_path)
                update_feed_runtime(feed_id, {
                    'last_alert_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'last_alert_event': alert.event,
                    'last_alert_headline': alert.headline,
                    'last_alert_severity': alert.severity,
                    'last_alert_audio': str(wav_path),
                })
                append_runtime_event(
                    'alert',
                    f'{alert.event}: {alert.headline}',
                    feed_id,
                    {
                        'severity': alert.severity,
                        'priority': priority,
                        'audio': str(wav_path),
                    },
                )
                log.info("[%s] Alert queued: %s (priority %d)", feed_id, wav_path.name, priority)

        if not matched_any:
            log.info(
                "No CAP-CP feeds accepted alert %s (%s, severity=%s, urgency=%s, certainty=%s, scope=%s)",
                alert.identifier,
                alert.event,
                alert.severity,
                alert.urgency,
                alert.certainty,
                alert.scope,
            )

    cap_cp_cfg = config.get("cap", {}).get("cap_cp", {})
    if not cap_cp_cfg.get("enabled", False):
        log.info("CAP-CP alerting disabled")
        return

    await naad_listener(on_alert, shutdown)

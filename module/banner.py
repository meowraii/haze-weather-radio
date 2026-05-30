from __future__ import annotations

import csv
import datetime as dt
import json
import locale
import os
import pathlib
import re
import zoneinfo
from typing import Any

from module.events import get_runtime_alert_entries

_SEVERITY_PRIORITY: list[str] = ['Extreme', 'Severe', 'Moderate', 'Minor', 'Unknown']

_SEVERITY_HEX_DEFAULT: dict[str, str] = {
    'Extreme': '#B91C1C',
    'Severe': '#B91C1C',
    'Moderate': '#B45309',
    'Minor': '#0B3810',
    'Unknown': '#0B3810',
}

_EAS_CRAWL_COLORS: dict[str, list[str]] = {
    'warning': [
        "#931102",
        '#370c16',
    ],
    'watch': [
        '#929301',
        "#37380b",
    ],
    'advisory': [
        "#019310",
        '#0b3810',
    ],
}

_HEADLINE_TRAIL_RE = re.compile(
    r'\s*-\s*(in effect|ended|updated|cancelled|statement)\s*$',
    re.IGNORECASE,
)

_SAME_MAPPING_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'sameMapping.json'
_FORECAST_LOCATIONS_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'csv' / 'FORECAST_LOCATIONS.csv'

_ORIGINATOR_LABELS: dict[str, str] = {
    'EAS': 'An EAS Participant',
    'CIV': 'A Civil Authority',
    'PEP': 'A Primary Entry Point System',
}

_WEATHER_SERVICES_BY_REGION: dict[str, str] = {
    'AU': 'The Australian Bureau of Meteorology',
    'BR': 'The National Institute of Meteorology of Brazil',
    'CA': 'Environment Canada',
    'DE': 'Deutscher Wetterdienst',
    'ES': 'Agencia Estatal de Meteorologia',
    'FR': 'Meteo-France',
    'GB': 'The Met Office',
    'HK': 'The Hong Kong Observatory',
    'IE': 'Met Eireann',
    'IN': 'The India Meteorological Department',
    'IT': 'The Italian Meteorological Service',
    'JP': 'The Japan Meteorological Agency',
    'KR': 'The Korea Meteorological Administration',
    'MX': 'Servicio Meteorologico Nacional',
    'NZ': 'MetService New Zealand',
    'PH': 'PAGASA',
    'SG': 'The Meteorological Service Singapore',
    'US': 'The National Weather Service',
    'ZA': 'The South African Weather Service',
}

_SAME_CATEGORY_TERMINAL: dict[str, str] = {
    'warning': 'warning',
    'watch': 'watch',
    'advisory': 'advisory',
    'statement': 'advisory',
    'outlook': 'advisory',
}

MIMIC_ENDEC = 'SAGE'

_ENDEC_MODE_ALIASES: dict[str, str] = {
    '': 'NONE',
    'NONE': 'NONE',
    'TFT': 'TFT',
    'SAGE': 'SAGE EAS',
    'SAGE EAS': 'SAGE EAS',
    'SAGE DIGITAL': 'SAGE DIGITAL',
    'DIGITAL': 'SAGE DIGITAL',
    'TRILITHIC': 'TRILITHIC',
    'VIAVI': 'TRILITHIC',
    'EASY': 'TRILITHIC',
    'BURK': 'BURK',
    'DAS': 'DASDEC',
    'DASDEC': 'DASDEC',
    'MONROE': 'DASDEC',
}

_ENDEC_ORIGINATORS: dict[str, dict[str, str]] = {
    'NONE': {
        'EAS': 'An EAS Participant',
        'CIV': 'A Civil Authority',
        'PEP': 'A Primary Entry Point System',
    },
    'TFT': {
        'EAS': 'An EAS Participant',
        'CIV': 'A Civil Authority',
        'PEP': 'A Primary Entry Point System',
    },
    'SAGE EAS': {
        'EAS': 'A Broadcast station or cable system',
        'CIV': 'The Civil Authorities',
        'PEP': 'A Primary Entry Point System',
    },
    'SAGE DIGITAL': {
        'EAS': 'An EAS Participant',
        'CIV': 'The Civil Authorities',
        'PEP': 'A Primary Entry Point System',
    },
    'TRILITHIC': {
        'EAS': 'An EAS Participant',
        'CIV': 'Civil Authorities',
        'PEP': 'A Primary Entry Point System',
    },
    'BURK': {
        'EAS': 'A Broadcast station or cable system',
        'CIV': 'Civil Authorities',
        'PEP': 'A Primary Entry Point System',
    },
    'DASDEC': {
        'EAS': 'An EAS Participant',
        'CIV': 'A Civil Authority',
        'PEP': 'A Primary Entry Point System',
    },
}

_same_event_labels_cache: dict[str, str] | None = None
_location_labels_cache: dict[str, str] | None = None
_system_locale_tags_cache: list[str] | None = None


def _clean_fragment(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def _resolve_endec_mode(raw: str | None = None) -> str:
    normalized = _clean_fragment(raw or MIMIC_ENDEC).replace('_', ' ').replace('-', ' ').upper()
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return _ENDEC_MODE_ALIASES.get(normalized, 'NONE')


def _with_indefinite_article(value: Any) -> str:
    text = _clean_fragment(value)
    if not text:
        return 'an Alert'
    if re.match(r'^(a|an)\s+', text, re.IGNORECASE):
        return text
    article = 'an' if text[0].lower() in {'a', 'e', 'i', 'o', 'u'} else 'a'
    return f'{article} {text}'


def _strip_leading_article(value: Any, include_the: bool = False) -> str:
    text = _clean_fragment(value)
    if not text:
        return ''
    pattern = r'^(a|an)\s+'
    if include_the:
        pattern = r'^(a|an|the)\s+'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)


def _display_dt(value: Any, tz_name: str = 'UTC') -> dt.datetime | None:
    parsed = _parse_dt(value)
    if parsed is None:
        return None
    try:
        return parsed.astimezone(zoneinfo.ZoneInfo(tz_name))
    except Exception:
        return parsed


def _format_endec_time(value: dt.datetime | None, mode: str, *, role: str, other: dt.datetime | None = None) -> str:
    if value is None:
        return ''
    if mode in {'SAGE EAS', 'SAGE DIGITAL'}:
        text = value.strftime('%I:%M %p').lower()
        if other is not None and (value.date() != other.date() or value.year != other.year):
            suffix = value.strftime(' %a %b %d')
            if value.year != other.year:
                suffix += value.strftime(', %Y')
            text += suffix.lower()
        return text
    if mode == 'TRILITHIC':
        zone_abbr = value.strftime('%Z') or 'UTC'
        return f"{value.strftime('%m/%d/%y %H:%M:00')} {zone_abbr}"
    if mode == 'BURK':
        if role == 'start':
            return value.strftime('%B %d, %Y at %I:%M %p')
        return value.strftime('%I:%M %p, %B %d, %Y')
    if mode == 'DASDEC':
        if role == 'start':
            return value.strftime('%I:%M %p ON %b %d, %Y').upper()
        return value.strftime('%I:%M %p %b %d, %Y').upper()
    if mode == 'TFT':
        if role == 'end' and other is not None and value.date() == other.date() and value.year == other.year:
            return value.strftime('%I:%M %p').upper()
        return value.strftime('%I:%M %p ON %b %d, %Y').upper()
    if other is not None and value.date() == other.date() and value.year == other.year:
        return value.strftime('%I:%M %p')
    if other is not None and value.year == other.year:
        return value.strftime('%I:%M %p %B %d')
    return value.strftime('%I:%M %p %B %d, %Y')


def _join_semicolon_parts(parts: list[str]) -> str:
    cleaned = [part for part in (_clean_fragment(part) for part in parts) if part]
    if not cleaned:
        return ''
    if len(cleaned) > 1:
        cleaned[-1] = f'and {cleaned[-1]}'
    return '; '.join(cleaned) + ';'


def _format_endec_areas(area_names: list[str], mode: str) -> str:
    if mode == 'TRILITHIC':
        return ' - '.join(name for name in (_clean_fragment(part) for part in area_names) if name)
    if mode in {'BURK', 'DASDEC'}:
        return _join_semicolon_parts(area_names)
    return _join_parts(area_names)


def _normalize_locale_tag(raw: Any) -> str:
    text = _clean_fragment(raw)
    if not text:
        return ''
    text = text.split('.', 1)[0].split('@', 1)[0].replace('_', '-').lower()
    return '' if text in {'c', 'posix'} else text


def _system_locale_tags() -> list[str]:
    global _system_locale_tags_cache
    if _system_locale_tags_cache is not None:
        return _system_locale_tags_cache

    candidates: list[str] = []
    for env_key in ('LC_ALL', 'LC_MESSAGES', 'LANG'):
        normalized = _normalize_locale_tag(os.environ.get(env_key))
        if normalized:
            candidates.append(normalized)

    locale_specs: list[tuple[Any, Any]] = []
    try:
        locale_specs.append(locale.getlocale())
    except Exception:
        pass
    for attr_name in ('LC_MESSAGES', 'LC_TIME', 'LC_CTYPE'):
        category = getattr(locale, attr_name, None)
        if category is None:
            continue
        try:
            locale_specs.append(locale.getlocale(category))
        except Exception:
            continue

    for language, _encoding in locale_specs:
        normalized = _normalize_locale_tag(language)
        if normalized:
            candidates.append(normalized)

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    _system_locale_tags_cache = deduped
    return deduped


def _weather_service_label() -> str:
    for locale_tag in _system_locale_tags():
        parts = [part for part in locale_tag.split('-') if part]
        for part in parts[1:]:
            region = part.upper()
            if len(region) == 2 and region in _WEATHER_SERVICES_BY_REGION:
                return _WEATHER_SERVICES_BY_REGION[region]
    return 'The National Weather Service'


def _join_parts(parts: list[str]) -> str:
    cleaned = [part for part in (_clean_fragment(part) for part in parts) if part]
    if not cleaned:
        return ''
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f'{cleaned[0]} and {cleaned[1]}'
    return ', '.join(cleaned[:-1]) + f', and {cleaned[-1]}'


def _load_same_event_labels() -> dict[str, str]:
    global _same_event_labels_cache
    if _same_event_labels_cache is not None:
        return _same_event_labels_cache
    try:
        with open(_SAME_MAPPING_PATH, encoding='utf-8') as file_handle:
            data = json.load(file_handle)
        labels = data.get('eas', {}) if isinstance(data, dict) else {}
        _same_event_labels_cache = {
            str(code).upper(): _clean_fragment(label)
            for code, label in labels.items()
            if _clean_fragment(label)
        }
    except Exception:
        _same_event_labels_cache = {}
    return _same_event_labels_cache


def _load_location_labels() -> dict[str, str]:
    global _location_labels_cache
    if _location_labels_cache is not None:
        return _location_labels_cache

    labels: dict[str, str] = {}
    try:
        with open(_FORECAST_LOCATIONS_PATH, newline='', encoding='utf-8') as file_handle:
            reader = csv.reader(file_handle)
            for row in reader:
                if len(row) < 2:
                    continue
                code = row[0].strip().strip('"')
                label = _clean_fragment(row[1].strip().strip('"'))
                if code.isdigit() and label and code not in labels:
                    labels[code] = label
    except Exception:
        labels = {}
    _location_labels_cache = labels
    return labels


def _ordinal_suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')


def _parse_dt(value: Any) -> dt.datetime | None:
    text = _clean_fragment(value)
    if not text:
        return None
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed


def _format_dt(dt_str: str | None, tz_name: str = 'UTC') -> str:
    if not dt_str:
        return ''
    try:
        parsed = dt.datetime.fromisoformat(dt_str)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.UTC)
        try:
            parsed = parsed.astimezone(zoneinfo.ZoneInfo(tz_name))
        except Exception:
            pass
        hour = parsed.hour % 12 or 12
        am_pm = 'A.M.' if parsed.hour < 12 else 'P.M.'
        suffix = _ordinal_suffix(parsed.day)
        return f'{hour}:{parsed.minute:02d} {am_pm} on {parsed.strftime("%B")} {parsed.day}{suffix}, {parsed.year}'
    except Exception:
        return dt_str


def _resolve_area_name(area: dict[str, Any]) -> str:
    area_desc = _clean_fragment(area.get('areaDesc'))
    if area_desc:
        return area_desc

    same_code = _clean_fragment(area.get('sameCode') or area.get('code'))
    if not same_code:
        for geocode in area.get('geocodes') or []:
            value = _clean_fragment((geocode or {}).get('value'))
            if value.isdigit() and len(value) == 6:
                same_code = value
                break

    if same_code:
        return _load_location_labels().get(same_code, same_code)
    return ''


def _normalize_headline(value: Any) -> str:
    headline = _HEADLINE_TRAIL_RE.sub('', _clean_fragment(value)).strip(' -')
    if not headline:
        return 'Alert'

    normalized_parts: list[str] = []
    for part in headline.split(' - '):
        clean_part = _clean_fragment(part)
        lowered = clean_part.lower()
        if lowered == 'yellow warning':
            normalized_parts.append('Yellow Advisory')
        else:
            normalized_parts.append(clean_part.title())
    return ' - '.join(part for part in normalized_parts if part)


def _message_id(entry: dict[str, Any]) -> str:
    meta = entry.get('metadata') or {}
    source = entry.get('source') or {}

    explicit = _clean_fragment(entry.get('display_id') or meta.get('displayId'))
    if explicit:
        return explicit

    timestamp_raw = _clean_fragment(source.get('sent') or entry.get('received_at'))
    if timestamp_raw:
        try:
            parsed = dt.datetime.fromisoformat(timestamp_raw)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.UTC)
            return f'MSG{parsed.astimezone(dt.UTC).strftime("%H%M%S")}'
        except ValueError:
            pass

    raw_identifier = _clean_fragment(entry.get('identifier'))
    if raw_identifier.startswith(('manual_', 'test_')):
        return f'MSG{raw_identifier.split("_")[-1][-6:]}'
    return ''


def _is_manual(entry: dict[str, Any]) -> bool:
    source = entry.get('source') or {}
    kind = _clean_fragment(source.get('kind')).lower()
    return kind in {'manual', 'test'}


def _manual_originator_label(entry: dict[str, Any]) -> str:
    source = entry.get('source') or {}
    originator = _clean_fragment(source.get('originator') or 'EAS').upper()[:3]
    if originator == 'WXR':
        return _weather_service_label()
    return _ORIGINATOR_LABELS.get(originator, 'An EAS Participant')


def _manual_event_label(entry: dict[str, Any]) -> str:
    meta = entry.get('metadata') or {}
    source = entry.get('source') or {}
    event_code = _clean_fragment(meta.get('event') or source.get('eventCode') or 'ADR').upper()[:3]
    return _load_same_event_labels().get(event_code, event_code)


def _alert_originator_code(entry: dict[str, Any]) -> str:
    meta = entry.get('metadata') or {}
    source = entry.get('source') or {}
    explicit = _clean_fragment(source.get('originator') or meta.get('originator')).upper()[:3]
    if explicit:
        return explicit
    sender_name = _clean_fragment(meta.get('senderName'))
    lowered = sender_name.lower()
    if any(token in lowered for token in ('weather', 'meteorolog', 'met office', 'observatory', 'environment canada')):
        return 'WXR'
    if 'civil' in lowered or 'alert ready' in lowered:
        return 'CIV'
    return ''


def _alert_source_label(entry: dict[str, Any]) -> str:
    source = entry.get('source') or {}
    if _is_manual(entry):
        manual_label = _clean_fragment(source.get('callsign') or source.get('sender_id') or source.get('sender'))
        if manual_label:
            return manual_label
    return _message_id(entry)


def _alert_subject(entry: dict[str, Any], mode: str) -> str:
    meta = entry.get('metadata') or {}
    originator = _alert_originator_code(entry)
    if not _is_manual(entry):
        sender_name = _clean_fragment(meta.get('senderName') or meta.get('event') or 'Alert')
        if mode == 'BURK':
            return _strip_leading_article(sender_name, include_the=True) or sender_name
        return sender_name
    if originator == 'WXR':
        weather_label = _weather_service_label()
        if mode == 'BURK':
            return _strip_leading_article(weather_label, include_the=True) or weather_label
        return weather_label
    labels = _ENDEC_ORIGINATORS.get(mode, _ENDEC_ORIGINATORS['NONE'])
    if originator in labels:
        return labels[originator]
    if originator:
        return f'Unknown Originator {originator}'
    return _ORIGINATOR_LABELS['EAS']


def _alert_event_phrase(entry: dict[str, Any], headline: str, mode: str) -> str:
    raw_phrase = _with_indefinite_article(headline)
    if mode == 'BURK':
        return _strip_leading_article(raw_phrase).upper()
    if mode == 'DASDEC':
        return raw_phrase.upper()
    return raw_phrase


def _alert_verb(entry: dict[str, Any], mode: str) -> str:
    if mode not in {'SAGE EAS', 'SAGE DIGITAL', 'TRILITHIC'}:
        return 'has'
    return 'have' if _alert_originator_code(entry) == 'CIV' else 'has'


def _build_endec_lead(
    entry: dict[str, Any],
    headline: str,
    area_names: list[str],
    onset_at: dt.datetime | None,
    expires_at: dt.datetime | None,
    mimic_endec: str | None,
) -> str:
    mode = _resolve_endec_mode(mimic_endec)
    subject = _alert_subject(entry, mode)
    event_phrase = _alert_event_phrase(entry, headline, mode)
    areas = _format_endec_areas(area_names, mode)
    source_label = _alert_source_label(entry)
    start_text = _format_endec_time(onset_at, mode, role='start', other=expires_at)
    end_text = _format_endec_time(expires_at, mode, role='end', other=onset_at)

    if mode in {'SAGE EAS', 'SAGE DIGITAL'}:
        lead = f'{subject} {_alert_verb(entry, mode)} issued {event_phrase}'
        if areas:
            lead = f'{lead} for {areas}'
        if start_text and end_text:
            lead = f'{lead} beginning at {start_text} and ending at {end_text}'
        elif end_text:
            lead = f'{lead} ending at {end_text}'
        if source_label:
            lead = f'{lead} ({source_label})'
        return f'{lead}.'

    if mode == 'TRILITHIC':
        area_clause = f'for the following counties: {areas}' if areas else 'for'
        lead = f'{subject} {_alert_verb(entry, mode)} issued {event_phrase} {area_clause}'.strip()
        if end_text:
            lead = f'{lead}. Effective Until {end_text}'
        if source_label:
            lead = f'{lead}. ({source_label})'
        return f'{lead}.'

    if mode == 'BURK':
        lead = f'{subject} has issued {event_phrase}'
        if areas:
            lead = f'{lead} for the following counties/areas: {areas}'
        if start_text and end_text:
            lead = f'{lead} on {start_text} effective until {end_text}'
        elif end_text:
            lead = f'{lead} effective until {end_text}'
        return f'{lead}.'

    if mode == 'DASDEC':
        lead = f'{subject.upper()} HAS ISSUED {event_phrase}'
        if areas:
            lead = f'{lead} FOR THE FOLLOWING COUNTIES/AREAS: {areas}'
        if start_text and end_text:
            lead = f'{lead} AT {start_text} EFFECTIVE UNTIL {end_text}'
        elif end_text:
            lead = f'{lead} EFFECTIVE UNTIL {end_text}'
        if source_label:
            lead = f'{lead}. MESSAGE FROM {source_label.upper()}'
        return f'{lead}.'

    if mode == 'TFT':
        event_code = _clean_fragment((entry.get('source') or {}).get('eventCode') or (entry.get('metadata') or {}).get('event')).upper()[:3]
        if _alert_originator_code(entry) == 'EAS' or event_code in {'NPT', 'EAN'}:
            lead = f'{event_phrase} has been issued'
        else:
            lead = f'{subject} has issued {event_phrase}'
        if areas:
            lead = f'{lead} for the following counties/areas: {areas}'
        if start_text and end_text:
            lead = f'{lead} at {start_text} effective until {end_text}'
        elif end_text:
            lead = f'{lead} effective until {end_text}'
        if source_label:
            lead = f'{lead}. message from {source_label}'
        return f'{lead}.'.upper()

    lead = f'{subject} has issued {event_phrase}'
    if areas:
        lead = f'{lead} for {areas}'
    if start_text and end_text:
        lead = f'{lead} beginning at {start_text} and ending at {end_text}'
    elif end_text:
        lead = f'{lead} effective until {end_text}'
    if source_label:
        lead = f'{lead}. Message from {source_label}'
    return f'{lead}.'


def _derive_same_category(event: str) -> str:
    words = event.lower().strip().split()
    last = words[-1] if words else ''
    if last in _SAME_CATEGORY_TERMINAL:
        return _SAME_CATEGORY_TERMINAL[last]
    raw = event.strip()
    if len(raw) == 3:
        lowered = raw[-1].lower()
        if lowered == 'w':
            return 'warning'
        if lowered == 'a':
            return 'watch'
        if lowered in {'s', 'y', 'e'}:
            return 'advisory'
    return 'advisory'


def _pick_same_category(entry: dict[str, Any]) -> str:
    meta = entry.get('metadata') or {}
    source = entry.get('source') or {}
    return _derive_same_category(str(meta.get('event') or source.get('eventCode') or ''))


def pick_banner_gradient(alerts: list[dict[str, Any]]) -> list[str]:
    for severity in _SEVERITY_PRIORITY:
        for entry in alerts:
            meta = entry.get('metadata') or {}
            entry_severity = _clean_fragment(meta.get('severity')).title() or 'Unknown'
            if entry_severity != severity:
                continue
            category = _pick_same_category(entry)
            gradient = _EAS_CRAWL_COLORS.get(category)
            if gradient:
                return gradient
            fallback = _SEVERITY_HEX_DEFAULT.get(severity, '#1F2937')
            return [fallback, fallback]

    for entry in alerts:
        category = _pick_same_category(entry)
        gradient = _EAS_CRAWL_COLORS.get(category)
        if gradient:
            return gradient

    return _EAS_CRAWL_COLORS['advisory']


def pick_banner_color(alerts: list[dict[str, Any]]) -> str:
    return pick_banner_gradient(alerts)[0]


def build_alert_message(entry: dict[str, Any], tz_name: str = 'UTC', mimic_endec: str | None = None) -> str:
    if not entry:
        return 'Alert'

    meta = entry.get('metadata') or {}
    text_block = entry.get('text') or {}
    areas = entry.get('areas') or []

    area_names = [name for name in (_resolve_area_name(area) for area in areas) if name]
    onset_at = _display_dt(meta.get('onset') or meta.get('effective'), tz_name)
    expires_at = _display_dt(meta.get('expires'), tz_name)

    if _is_manual(entry):
        headline = _manual_event_label(entry)
    else:
        headline = _normalize_headline(meta.get('headline') or meta.get('event') or 'Alert')

    parts: list[str] = [_build_endec_lead(entry, headline, area_names, onset_at, expires_at, mimic_endec)]

    description = _clean_fragment(text_block.get('description'))
    instruction = _clean_fragment(text_block.get('instruction'))
    if description:
        parts.append(description)
    if instruction:
        parts.append(instruction)

    return ' '.join(parts)


def serialize_alert(entry: dict[str, Any], tz_name: str = 'UTC', mimic_endec: str | None = None) -> dict[str, Any]:
    meta = entry.get('metadata') or {}
    source = entry.get('source') or {}
    text_block = entry.get('text') or {}
    areas = entry.get('areas') or []

    area_names = [name for name in (_resolve_area_name(area) for area in areas) if name]
    headline = _manual_event_label(entry) if _is_manual(entry) else _normalize_headline(meta.get('headline') or meta.get('event') or 'Alert')
    issuer = _manual_originator_label(entry) if _is_manual(entry) else _clean_fragment(meta.get('senderName') or meta.get('event') or 'Alert')
    message = build_alert_message(entry, tz_name, mimic_endec)
    effective_at = _clean_fragment(meta.get('effective'))
    onset_at = _clean_fragment(meta.get('onset') or meta.get('effective'))
    expires_at = _clean_fragment(meta.get('expires'))

    return {
        'identifier': _clean_fragment(entry.get('identifier')),
        'feed_id': _clean_fragment(entry.get('feed_id')),
        'display_id': _message_id(entry),
        'headline': headline,
        'issuer': issuer,
        'event': _clean_fragment(meta.get('event') or source.get('eventCode')).upper()[:3] or headline,
        'severity': _clean_fragment(meta.get('severity')).title() or 'Unknown',
        'urgency': _clean_fragment(meta.get('urgency')).title(),
        'certainty': _clean_fragment(meta.get('certainty')).title(),
        'areas': area_names,
        'area_text': _join_parts(area_names),
        'received_at': _clean_fragment(entry.get('received_at') or source.get('sent')),
        'effective_at': effective_at,
        'onset_at': onset_at,
        'expires_at': expires_at,
        'onset_display': _format_dt(onset_at, tz_name),
        'expires_display': _format_dt(expires_at, tz_name),
        'description': _clean_fragment(text_block.get('description')),
        'instruction': _clean_fragment(text_block.get('instruction')),
        'message': message,
        'source_kind': _clean_fragment(source.get('kind') or source.get('status')).lower(),
        'background_color': pick_banner_color([entry]),
        'background_gradient': pick_banner_gradient([entry]),
    }


def _alert_sort_key(entry: dict[str, Any]) -> tuple[int, float, str]:
    meta = entry.get('metadata') or {}
    severity = _clean_fragment(meta.get('severity')).title() or 'Unknown'
    try:
        severity_rank = _SEVERITY_PRIORITY.index(severity)
    except ValueError:
        severity_rank = len(_SEVERITY_PRIORITY)
    active_at = _parse_dt(meta.get('effective') or entry.get('received_at'))
    timestamp = active_at.timestamp() if active_at is not None else 0.0
    return severity_rank, -timestamp, _clean_fragment(entry.get('identifier'))


def get_active_alerts(feed_id: str) -> list[dict[str, Any]]:
    return sorted(get_runtime_alert_entries(feed_id), key=_alert_sort_key)
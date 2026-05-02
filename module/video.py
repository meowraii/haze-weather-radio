from __future__ import annotations

import asyncio
import csv
import datetime as _dt
import json
import locale as _locale
import logging
import os as _os
import pathlib
import re
import subprocess
import tempfile
import threading
import textwrap as _textwrap
import zoneinfo as _zoneinfo
from typing import Any

from module.events import get_runtime_alert_entries, shutdown_event
from module.buffer import CHANNELS, SAMPLE_RATE

log = logging.getLogger(__name__)

_VIDEO_FORMATS: frozenset[str] = frozenset({'vp8', 'vp9', 'theora'})

_VIDEO_CODEC_MAP: dict[str, tuple[str, str, str]] = {
    'vp8':    ('libvpx',     'webm', 'video/webm'),
    'vp9':    ('libvpx-vp9', 'webm', 'video/webm'),
    'theora': ('libtheora',  'ogg',  'video/ogg'),
}


def _video_codec_args(codec: str) -> list[str]:
    normalized = str(codec or '').strip().lower()
    if normalized == 'libvpx':
        return ['-deadline', 'realtime', '-cpu-used', '8', '-lag-in-frames', '0']
    if normalized == 'libvpx-vp9':
        return [
            '-deadline', 'realtime',
            '-cpu-used', '8',
            '-row-mt', '1',
            '-tile-columns', '2',
            '-frame-parallel', '1',
            '-lag-in-frames', '0',
            '-auto-alt-ref', '0',
        ]
    return []

_SEVERITY_PRIORITY: list[str] = ['Extreme', 'Severe', 'Moderate', 'Minor', 'Unknown']

_SEVERITY_HEX_DEFAULT: dict[str, str] = {
    'Extreme': '#FF0000',
    'Severe':  '#FF6600',
    'Moderate': '#FFCC00',
    'Minor':   '#00CC00',
    'Unknown': '#CCCCCC',
}

_IDLE_BANNER_HEX = '#003366'

_RECONNECT_DELAY_S = 3.0

_HEADLINE_TRAIL_RE = re.compile(
    r'\s*-\s*(in effect|ended|updated|cancelled|statement)\s*$',
    re.IGNORECASE,
)

_SOREM_PREFIX = 'layer:sorem'
_ECCC_PREFIX = 'layer:ec-msc'
_SAME_MAPPING_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'sameMapping.json'
_FORECAST_LOCATIONS_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'csv' / 'FORECAST_LOCATIONS.csv'

_ORIGINATOR_LABELS: dict[str, str] = {
    'EAS': 'An EAS Participant',
    'CIV': 'Civil Authorities',
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

_same_event_labels_cache: dict[str, str] | None = None
_location_labels_cache: dict[str, str] | None = None
_system_locale_tags_cache: list[str] | None = None


def _feed_playout_mapping(feed: dict[str, Any]) -> dict[str, Any]:
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


def _ordinal_suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')


def _clean_overlay_fragment(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value or '')).strip()


def _normalize_locale_tag(raw: Any) -> str:
    text = _clean_overlay_fragment(raw)
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
        normalized = _normalize_locale_tag(_os.environ.get(env_key))
        if normalized:
            candidates.append(normalized)

    locale_specs: list[tuple[Any, Any]] = []
    try:
        locale_specs.append(_locale.getlocale())
    except Exception:
        pass
    for attr_name in ('LC_MESSAGES', 'LC_TIME', 'LC_CTYPE'):
        category = getattr(_locale, attr_name, None)
        if category is None:
            continue
        try:
            locale_specs.append(_locale.getlocale(category))
        except Exception:
            continue

    for language, _encoding in locale_specs:
        normalized = _normalize_locale_tag(language)
        if normalized:
            candidates.append(normalized)

    normalized_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            normalized_candidates.append(candidate)
    _system_locale_tags_cache = normalized_candidates
    return _system_locale_tags_cache


def _weather_service_label() -> str:
    for locale_tag in _system_locale_tags():
        parts = [part for part in locale_tag.split('-') if part]
        for part in parts[1:]:
            region = part.upper()
            if len(region) == 2 and region in _WEATHER_SERVICES_BY_REGION:
                return _WEATHER_SERVICES_BY_REGION[region]
    return 'The Weather Service'


def _join_overlay_parts(parts: list[str]) -> str:
    cleaned = [part for part in (_clean_overlay_fragment(part) for part in parts) if part]
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
            str(code).upper(): _clean_overlay_fragment(label)
            for code, label in labels.items()
            if _clean_overlay_fragment(label)
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
                label = _clean_overlay_fragment(row[1].strip().strip('"'))
                if code.isdigit() and label and code not in labels:
                    labels[code] = label
    except Exception:
        labels = {}
    _location_labels_cache = labels
    return _location_labels_cache


def _format_overlay_dt(dt_str: str | None, tz_name: str = 'UTC') -> str:
    if not dt_str:
        return ''
    try:
        dt = _dt.datetime.fromisoformat(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.UTC)
        try:
            dt = dt.astimezone(_zoneinfo.ZoneInfo(tz_name))
        except Exception:
            pass
        hour = dt.hour % 12 or 12
        am_pm = 'A.M.' if dt.hour < 12 else 'P.M.'
        suffix = _ordinal_suffix(dt.day)
        return f'{hour}:{dt.minute:02d} {am_pm} on {dt.strftime("%B")} {dt.day}{suffix}, {dt.year}'
    except Exception:
        return dt_str


def _resolve_overlay_area_name(area: dict[str, Any]) -> str:
    area_desc = _clean_overlay_fragment(area.get('areaDesc'))
    if area_desc:
        return area_desc

    same_code = _clean_overlay_fragment(area.get('sameCode') or area.get('code'))
    if not same_code:
        for geocode in area.get('geocodes') or []:
            value = _clean_overlay_fragment((geocode or {}).get('value'))
            if value.isdigit() and len(value) == 6:
                same_code = value
                break

    if same_code:
        return _load_location_labels().get(same_code, same_code)
    return ''


def _normalize_cap_headline(value: Any) -> str:
    headline = _HEADLINE_TRAIL_RE.sub('', _clean_overlay_fragment(value)).strip(' -')
    if not headline:
        return 'Alert'

    normalized_parts: list[str] = []
    for part in headline.split(' - '):
        clean_part = _clean_overlay_fragment(part)
        lowered = clean_part.lower()
        if lowered == 'yellow warning':
            normalized_parts.append('Yellow Advisory')
        else:
            normalized_parts.append(clean_part.title())
    return ' - '.join(part for part in normalized_parts if part)


def _overlay_message_id(entry: dict[str, Any]) -> str:
    meta = entry.get('metadata') or {}
    source = entry.get('source') or {}

    explicit = _clean_overlay_fragment(entry.get('display_id') or meta.get('displayId'))
    if explicit:
        return explicit

    timestamp_raw = _clean_overlay_fragment(source.get('sent') or entry.get('received_at'))
    if timestamp_raw:
        try:
            ts = _dt.datetime.fromisoformat(timestamp_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=_dt.UTC)
            return f'MSG{ts.astimezone(_dt.UTC).strftime("%H%M%S")}'
        except ValueError:
            pass

    raw_identifier = _clean_overlay_fragment(entry.get('identifier'))
    if raw_identifier.startswith(('manual_', 'test_')):
        return f'MSG{raw_identifier.split("_")[-1][-6:]}'
    return ''


def _is_manual_overlay(entry: dict[str, Any]) -> bool:
    source = entry.get('source') or {}
    kind = _clean_overlay_fragment(source.get('kind')).lower()
    return kind in {'manual', 'test'}


def _manual_originator_label(entry: dict[str, Any]) -> str:
    source = entry.get('source') or {}
    originator = _clean_overlay_fragment(source.get('originator') or 'EAS').upper()[:3]
    if originator == 'WXR':
        return _weather_service_label()
    return _ORIGINATOR_LABELS.get(originator, 'An EAS Participant')


def _manual_event_label(entry: dict[str, Any]) -> str:
    meta = entry.get('metadata') or {}
    source = entry.get('source') or {}
    event_code = _clean_overlay_fragment(meta.get('event') or source.get('eventCode') or 'ADR').upper()[:3]
    return _load_same_event_labels().get(event_code, event_code)


def _build_overlay_text(entry: dict[str, Any], tz_name: str = 'UTC') -> str:
    if not entry:
        return 'Alert'

    meta = entry.get('metadata') or {}
    text_block = entry.get('text') or {}
    areas = entry.get('areas') or []

    area_names = [name for name in (_resolve_overlay_area_name(area) for area in areas) if name]
    area_clause = f' for {_join_overlay_parts(area_names)}' if area_names else ''
    onset_str = _format_overlay_dt(meta.get('onset') or meta.get('effective'), tz_name)
    expires_str = _format_overlay_dt(meta.get('expires'), tz_name)

    if _is_manual_overlay(entry):
        issuer = _manual_originator_label(entry)
        headline = _manual_event_label(entry)
    else:
        issuer = _clean_overlay_fragment(meta.get('senderName') or meta.get('event') or 'Alert')
        headline = _normalize_cap_headline(meta.get('headline') or meta.get('event') or 'Alert')

    main = f'{issuer} has issued a {headline}{area_clause}'

    if onset_str and expires_str:
        main = f'{main}. Beginning at {onset_str} and effective until {expires_str}'
    elif expires_str:
        main = f'{main}. Effective until {expires_str}'
    else:
        main = f'{main}'

    message_id = _overlay_message_id(entry)
    if message_id:
        main = f'{main} ({message_id})'
    main = f'{main}.'

    parts: list[str] = [main]

    description = _clean_overlay_fragment(text_block.get('description'))
    instruction = _clean_overlay_fragment(text_block.get('instruction'))
    if description:
        parts.append(description)
    if instruction:
        parts.append(instruction)

    return ' '.join(parts)


def _format_overlay_display_text(text: str, style: dict[str, Any], width: int) -> str:
    if str(style.get('format', 'crawl')).lower() != 'fullscreen':
        return text

    normalized = text.replace('\r\n', '\n').replace('\r', '\n')
    font_size = max(16, int(style.get('font_size', 48) or 48))
    border_cfg = _coerce_mapping(style.get('fullscreen_border', {}))
    border_width = int(border_cfg.get('width', 0) or 0) if border_cfg.get('enabled') else 0
    usable_width = max(240, width - (border_width * 2) - max(64, width // 10))
    approx_char_width = max(8.0, font_size * 0.55)
    max_chars = max(16, int(usable_width / approx_char_width))
    wrapper = _textwrap.TextWrapper(
        width=max_chars,
        break_long_words=True,
        break_on_hyphens=True,
        replace_whitespace=True,
        drop_whitespace=True,
    )

    wrapped_blocks: list[str] = []
    for block in normalized.split('\n'):
        clean_block = _clean_overlay_fragment(block)
        if not clean_block:
            if wrapped_blocks and wrapped_blocks[-1] != '':
                wrapped_blocks.append('')
            continue
        wrapped_blocks.extend(wrapper.wrap(clean_block) or [''])
    return '\n'.join(wrapped_blocks).strip() or text

_IDLE_MODE = 'idle'
_ALERT_MODE = 'alert'

_SAME_CATEGORY_TERMINAL: dict[str, str] = {
    'warning':  'warning',
    'watch':    'watch',
    'advisory': 'advisory',
    'statement': 'advisory',
    'outlook':  'advisory',
}


def _derive_same_category(event: str) -> str:
    words = event.lower().strip().split()
    last = words[-1] if words else ''
    if last in _SAME_CATEGORY_TERMINAL:
        return _SAME_CATEGORY_TERMINAL[last]
    raw = event.strip()
    if len(raw) == 3:
        lc = raw[-1].lower()
        if lc == 'w': return 'warning'
        if lc == 'a': return 'watch'
        if lc == 'y': return 'advisory'
    return 'unknown'


def _to_ffmpeg_color(hex_color: str) -> str:
    return hex_color.strip().replace('#', '0x')


def _escape_drawtext_value(value: Any) -> str:
    return (
        str(value)
        .replace('\\', r'\\')
        .replace(':', r'\\:')
        .replace(',', r'\\,')
        .replace("'", r"\\'")
    )


def _coerce_mapping(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        merged: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    return {}


def _pick_severity_color(alerts: list[dict[str, Any]], style: dict[str, Any]) -> str:
    colors = _coerce_mapping(style.get('colors', {}))
    same_palette = _coerce_mapping(colors.get('same', {}))
    naads = _coerce_mapping(colors.get('naads', {}))
    sev_map = _coerce_mapping(naads.get('severity', {}))

    for sev in _SEVERITY_PRIORITY:
        for entry in alerts:
            meta = entry.get('metadata') or {}
            entry_sev = (meta.get('severity') or '').strip()
            if entry_sev.lower() != sev.lower():
                continue
            cat = _derive_same_category(str(meta.get('event') or ''))
            if cat != 'unknown' and same_palette.get(cat):
                return _to_ffmpeg_color(str(same_palette[cat]))
            color = sev_map.get(sev.lower()) or sev_map.get(sev) or _SEVERITY_HEX_DEFAULT.get(sev, '#FFCC00')
            return _to_ffmpeg_color(str(color))
    return _to_ffmpeg_color(_IDLE_BANNER_HEX)


def _get_active_alerts(feed_id: str) -> list[dict[str, Any]]:
    active = get_runtime_alert_entries(feed_id)
    reg_path = pathlib.Path('data') / 'alerts' / f'{feed_id}.json'
    if not reg_path.exists():
        return active
    try:
        entries: list[dict[str, Any]] = json.loads(reg_path.read_text(encoding='utf-8'))
    except Exception:
        return active
    now = _dt.datetime.now(_dt.UTC)
    for entry in entries:
        expires_raw: str | None = (entry.get('metadata') or {}).get('expires')
        if expires_raw:
            try:
                exp = _dt.datetime.fromisoformat(expires_raw)
                if exp.tzinfo is None:
                    exp = exp.replace(tzinfo=_dt.UTC)
                if exp < now:
                    continue
            except ValueError:
                pass
        active.append(entry)
    return active


def _build_drawtext_filter(
    text_file: pathlib.Path,
    banner_color: str,
    width: int,
    height: int,
    style: dict[str, Any],
    *,
    reload_text: bool = False,
) -> str:
    fmt = str(style.get('format', 'crawl'))

    font = str(style.get('font', 'Arial'))
    font_weight = str(style.get('font_weight', '')).strip().lower()
    if font_weight and font_weight not in ('normal', 'regular'):
        font = f'{font}:style={font_weight.title()}'
    font = _escape_drawtext_value(font)
    text_file_value = _escape_drawtext_value(text_file)

    font_size = int(style.get('font_size', 48))
    font_color = _to_ffmpeg_color(str(style.get('font_color', '#FFFFFF')))
    opacity = float(style.get('opacity', 0.9))
    shadow_cfg = _coerce_mapping(style.get('text_shadow', {}))
    stroke_cfg = _coerce_mapping(style.get('text_stroke', {}))
    reload_args = ['reload=1'] if reload_text else []

    def _shadow_params() -> list[str]:
        if not shadow_cfg.get('enabled'):
            return []
        sc = _to_ffmpeg_color(str(shadow_cfg.get('color', '#000000')))
        sx = int(shadow_cfg.get('offset_x', 2))
        sy = int(shadow_cfg.get('offset_y', 2))
        return [f'shadowcolor={sc}', f'shadowx={sx}', f'shadowy={sy}']

    def _stroke_params() -> list[str]:
        if not stroke_cfg.get('enabled'):
            return []
        bc = _to_ffmpeg_color(str(stroke_cfg.get('color', '#000000')))
        bw = int(stroke_cfg.get('width', 2))
        return [f'bordercolor={bc}', f'borderw={bw}']

    if fmt == 'crawl':
        crawl_h = font_size + 40
        crawl_y = int(height * 0.10)
        text_y = crawl_y + (crawl_h - font_size) // 2
        scroll_speed = font_size * 6

        box_filter = (
            f'drawbox=x=0:y={crawl_y}:w=iw:h={crawl_h}'
            f':color={banner_color}@{opacity:.2f}:t=fill'
        )

        dt_opts = [
            f'textfile={text_file_value}',
            *reload_args,
            f'font={font}',
            f'fontsize={font_size}',
            f'fontcolor={font_color}',
            f'y={text_y}',
            f'x=w-mod(t*{scroll_speed}\\,w+tw)',
            *_shadow_params(),
            *_stroke_params(),
        ]
        return f'{box_filter},drawtext={":".join(dt_opts)}'

    border_cfg = _coerce_mapping(style.get('fullscreen_border', {}))
    bg_filter = f'drawbox=x=0:y=0:w=iw:h=ih:color={banner_color}@{opacity:.2f}:t=fill'

    filters: list[str] = [bg_filter]
    if border_cfg.get('enabled'):
        bc = _to_ffmpeg_color(str(border_cfg.get('color', '#FF0000')))
        bw = int(border_cfg.get('width', 10))
        filters.append(f'drawbox=x=0:y=0:w=iw:h=ih:color={bc}@1:t={bw}')

    text_box_w = max(160, width - max(80, width // 10))
    text_box_h = max(font_size * 4, int(height * 0.78))
    line_spacing = max(6, font_size // 5)

    dt_opts = [
        f'textfile={text_file_value}',
        *reload_args,
        f'font={font}',
        f'fontsize={font_size}',
        f'fontcolor={font_color}',
        f'x=(w-{text_box_w})/2',
        f'y=(h-{text_box_h})/2',
        f'boxw={text_box_w}',
        f'boxh={text_box_h}',
        f'line_spacing={line_spacing}',
        'text_align=center+middle',
        'fix_bounds=1',
        *_shadow_params(),
        *_stroke_params(),
    ]
    filters.append(f'drawtext={":".join(dt_opts)}')
    return ','.join(filters)


def _postprocess_video_filter(
    filter_chain: str,
    *,
    interlace: bool,
    output_pix_fmt: str | None = 'yuv420p',
) -> str:
    filters = [filter_chain]
    if interlace:
        filters.append('format=yuv420p')
        filters.extend([
            'interlace=scan=tff:lowpass=0',
            'fieldorder=tff',
            'setfield=tff',
        ])
        if output_pix_fmt and output_pix_fmt.lower() != 'yuv420p':
            filters.append(f'format={output_pix_fmt}')
    elif output_pix_fmt:
        filters.append(f'format={output_pix_fmt}')
    return ','.join(filters)


class VideoIcecastSink:
    bus_queue_limit = 24
    bus_drop_oldest = True
    bus_clocked = True
    bus_prefill_chunks = 2
    bus_fill_silence = True

    def __init__(self, feed: dict[str, Any], video_cfg: dict[str, Any], stream_cfg: dict[str, Any]) -> None:
        self._feed = feed
        self._feed_id: str = feed['id']
        self._video_cfg = video_cfg
        self._stream_cfg = stream_cfg
        self._closed = False
        self._ssl: bool = bool(stream_cfg.get('ssl', False))
        self._tz_name: str = str(feed.get('timezone', 'UTC'))

        self._width: int = int(video_cfg.get('width', 1920))
        self._height: int = int(video_cfg.get('height', 1080))
        self._fps: float = float(video_cfg.get('fps', 29.97))
        self._style: dict[str, Any] = _coerce_mapping(video_cfg.get('style', {}))

        fmt = str(stream_cfg.get('format', 'vp9'))
        self._v_codec, self._container, self._content_type = _VIDEO_CODEC_MAP.get(fmt, _VIDEO_CODEC_MAP['vp9'])
        self._audio_bitrate: int = int(stream_cfg.get('bitrate_kbps', 32))

        fd, tmp = tempfile.mkstemp(suffix='.txt', prefix=f'haze_vt_{self._feed_id}_')
        _os.close(fd)
        self._text_file = pathlib.Path(tmp)
        self._text_file.write_text('', encoding='utf-8')

        self._pcm_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=64)
        self._writer_task: asyncio.Task | None = None

        self._proc_lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._rebuilding = False

        self._current_color: str = _to_ffmpeg_color(_IDLE_BANNER_HEX)
        self._last_text: str = ''

        self._start_ffmpeg()
        self._writer_task = asyncio.create_task(self._pcm_writer())

    def _icecast_url(self) -> str:
        cfg = self._stream_cfg
        user = cfg.get('username', 'source')
        pw = cfg.get('password', '')
        host = cfg.get('host', 'localhost')
        port = int(cfg.get('port', 8000))
        mount = cfg.get('mount') or f'/{self._feed_id}'
        return f'icecast://{user}:{pw}@{host}:{port}{mount}'

    def _write_proc(self, pcm: bytes) -> None:
        with self._proc_lock:
            proc = self._proc
            if proc is None or proc.stdin is None or proc.poll() is not None:
                raise BrokenPipeError('video icecast ffmpeg stdin unavailable')
            proc.stdin.write(pcm)

    def _build_cmd(self) -> list[str]:
        fps_str = str(self._fps)
        interlace = bool(self._style.get('interlace', False))
        bg_image = self._style.get('background_image')

        if bg_image:
            video_input = ['-loop', '1', '-framerate', fps_str, '-i', str(bg_image)]
            src_filter = f'[1:v]scale={self._width}:{self._height}:force_original_aspect_ratio=increase,crop={self._width}:{self._height},fps={fps_str}'
        else:
            video_input = ['-f', 'lavfi', '-i', f'color=c=black:size={self._width}x{self._height}:r={fps_str}']
            src_filter = '[1:v]null'

        overlay = _build_drawtext_filter(
            self._text_file,
            self._current_color,
            self._width,
            self._height,
            self._style,
        )
        filter_complex = f'{_postprocess_video_filter(f"{src_filter},{overlay}", interlace=interlace)}[vout]'

        return [
            'ffmpeg',
            '-loglevel', 'warning',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-thread_queue_size', '512',
            '-f', 's16le',
            '-ar', str(SAMPLE_RATE),
            '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            *video_input,
            '-filter_complex', filter_complex,
            '-map', '0:a',
            '-map', '[vout]',
            '-c:a', 'libopus',
            '-b:a', f'{self._audio_bitrate}k',
            '-ar', str(SAMPLE_RATE),
            '-ac', str(CHANNELS),
            '-c:v', self._v_codec,
            '-pix_fmt', 'yuv420p',
            *_video_codec_args(self._v_codec),
            '-g', '30',
            '-b:v', '400k',
            *(['-tls', '1'] if self._ssl else []),
            '-content_type', self._content_type,
            '-f', self._container,
            self._icecast_url(),
        ]

    def _start_ffmpeg(self) -> None:
        cmd = self._build_cmd()
        with self._proc_lock:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def _rebuild_proc(self, banner_color: str) -> None:
        self._rebuilding = True
        try:
            with self._proc_lock:
                old = self._proc
                try:
                    if old and old.stdin:
                        old.stdin.close()
                except Exception:
                    pass
                if old:
                    try:
                        old.terminate()
                    except Exception:
                        pass
                    try:
                        old.wait(timeout=1.0)
                    except Exception:
                        try:
                            old.kill()
                            old.wait(timeout=1.0)
                        except Exception:
                            pass
                self._current_color = banner_color
                self._proc = subprocess.Popen(self._build_cmd(), stdin=subprocess.PIPE)
        finally:
            self._rebuilding = False

    async def _recover_from_disconnect(self, reason: Exception) -> None:
        if self._closed:
            return
        log.warning('[%s] Video Icecast sink disconnected (%s), reconnecting in %.1fs', self._feed_id, reason, _RECONNECT_DELAY_S)
        await asyncio.sleep(_RECONNECT_DELAY_S)
        if self._closed:
            return
        await asyncio.to_thread(self._rebuild_proc, self._current_color)

    async def _pcm_writer(self) -> None:
        while not self._closed:
            pcm = await self._pcm_queue.get()
            try:
                await asyncio.to_thread(self._write_proc, pcm)
            except RuntimeError as exc:
                if 'cannot schedule new futures after shutdown' in str(exc).lower():
                    return
                if self._rebuilding:
                    continue
                await self._recover_from_disconnect(exc)
            except (BrokenPipeError, OSError, ValueError) as exc:
                if self._rebuilding:
                    continue
                await self._recover_from_disconnect(exc)
            except Exception as exc:
                if self._rebuilding:
                    continue
                await self._recover_from_disconnect(exc)

    async def on_alert_start(self, identifier: str) -> None:
        if self._closed:
            return

        alerts = _get_active_alerts(self._feed_id)
        entry = next((a for a in alerts if a.get('identifier') == identifier), None) or {}

        overlay_text = _format_overlay_display_text(
            _build_overlay_text(entry, self._tz_name),
            self._style,
            self._width,
        )

        if overlay_text != self._last_text:
            self._last_text = overlay_text
            try:
                self._text_file.write_text(overlay_text, encoding='utf-8')
            except Exception:
                pass

        new_color = _pick_severity_color([entry], self._style)
        await asyncio.to_thread(self._rebuild_proc, new_color)

    async def on_alert_end(self) -> None:
        if self._closed:
            return

        self._last_text = ''
        try:
            self._text_file.write_text('', encoding='utf-8')
        except Exception:
            pass

        idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
        await asyncio.to_thread(self._rebuild_proc, idle_color)

    async def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        try:
            self._pcm_queue.put_nowait(pcm)
        except asyncio.QueueFull:
            try:
                _ = self._pcm_queue.get_nowait()
                self._pcm_queue.put_nowait(pcm)
            except Exception:
                pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._writer_task:
            self._writer_task.cancel()

        with self._proc_lock:
            proc = self._proc
            self._proc = None

        if proc:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            self._text_file.unlink(missing_ok=True)
        except Exception:
            pass
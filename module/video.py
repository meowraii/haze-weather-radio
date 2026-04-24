from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import pathlib
import re
import subprocess
import tempfile
import threading
import zoneinfo as _zoneinfo
from typing import Any

from managed.events import shutdown_event
from module.buffer import CHANNELS, SAMPLE_RATE

log = logging.getLogger(__name__)

_VIDEO_FORMATS: frozenset[str] = frozenset({'vp8', 'vp9', 'theora'})

_VIDEO_CODEC_MAP: dict[str, tuple[str, str, str]] = {
    'vp8':    ('libvpx',     'webm', 'video/webm'),
    'vp9':    ('libvpx-vp9', 'webm', 'video/webm'),
    'theora': ('libtheora',  'ogg',  'video/ogg'),
}

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
        am_pm = 'AM' if dt.hour < 12 else 'PM'
        return f'{hour}:{dt.minute:02d} {am_pm} {dt.strftime("%A %B")} {dt.day}, {dt.year}'
    except Exception:
        return dt_str


def _build_overlay_text(entry: dict[str, Any], tz_name: str = 'UTC') -> str:
    meta = entry.get('metadata') or {}
    text_block = entry.get('text') or {}
    params = entry.get('parameters') or []
    areas = entry.get('areas') or []

    sender = str(meta.get('senderName') or meta.get('event') or 'Alert').strip()
    raw_headline = str(meta.get('headline') or meta.get('event') or 'Alert').strip()
    headline = _HEADLINE_TRAIL_RE.sub('', raw_headline).strip()

    area_names = [a.get('areaDesc', '').strip() for a in areas if a.get('areaDesc', '').strip()]
    onset_str = _format_overlay_dt(meta.get('onset') or meta.get('effective'), tz_name)
    expires_str = _format_overlay_dt(meta.get('expires'), tz_name)

    param_names = {str(p.get('valueName', '')).lower() for p in params}
    if any(n.startswith(_SOREM_PREFIX) or n.startswith(_ECCC_PREFIX) for n in param_names):
        source_tag = 'Alert Ready (CAP-CP)'
    else:
        source_tag = sender

    parts: list[str] = []
    areas_str = '; '.join(area_names) if area_names else ''

    if areas_str:
        main = f'{sender} has issued a {headline} for the following areas: {areas_str}'
    else:
        main = f'{sender} has issued a {headline}'

    if onset_str and expires_str:
        main = f'{main}, beginning at {onset_str} and effective until {expires_str}.'
    elif expires_str:
        main = f'{main}, effective until {expires_str}.'
    else:
        main = f'{main}.'

    parts.append(main)
    parts.append(f'({source_tag})')

    description = str(text_block.get('description') or '').strip()
    instruction = str(text_block.get('instruction') or '').strip()
    if description:
        parts.append(description)
    if instruction:
        parts.append(instruction)

    return '  '.join(parts)

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
    reg_path = pathlib.Path('data') / 'alerts' / f'{feed_id}.json'
    if not reg_path.exists():
        return []
    try:
        entries: list[dict[str, Any]] = json.loads(reg_path.read_text(encoding='utf-8'))
    except Exception:
        return []
    now = _dt.datetime.now(_dt.UTC)
    active: list[dict[str, Any]] = []
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
            'reload=1',
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

    text_align = str(style.get('text_alignment', 'center'))
    text_x = {'center': '(w-tw)/2', 'left': '10', 'right': 'w-tw-10'}.get(text_align, '(w-tw)/2')

    dt_opts = [
        f'textfile={text_file_value}',
        'reload=1',
        f'font={font}',
        f'fontsize={font_size}',
        f'fontcolor={font_color}',
        f'x={text_x}',
        'y=(h-th)/2',
        *_shadow_params(),
        *_stroke_params(),
    ]
    filters.append(f'drawtext={":".join(dt_opts)}')
    return ','.join(filters)


class VideoIcecastSink:
    """AudioPipeline sink that muxes PCM audio with a live video overlay and streams to Icecast.

    The overlay shows a crawl banner or fullscreen card ONLY while an alert is actively being
    toned — matching the behaviour of real television EAS. In idle state the audio stream
    carries regular broadcast content with no video overlay. Call on_alert_start(identifier)
    when alert audio begins and on_alert_end() when it finishes.  A background thread watches
    for unexpected ffmpeg exits and restarts the process if needed.
    """

    bus_queue_limit = 64
    bus_drop_oldest = False

    def __init__(
        self,
        feed: dict[str, Any],
        video_cfg: dict[str, Any],
        stream_cfg: dict[str, Any],
    ) -> None:
        self._feed = feed
        self._feed_id: str = feed['id']
        self._video_cfg = video_cfg
        self._stream_cfg = stream_cfg
        self._closed = False
        self._tz_name: str = str(feed.get('timezone', 'UTC'))

        self._width: int = int(video_cfg.get('width', 1920))
        self._height: int = int(video_cfg.get('height', 1080))
        self._fps: float = float(video_cfg.get('fps', 29.97))
        self._style: dict[str, Any] = _coerce_mapping(video_cfg.get('style', {}))

        fmt = str(stream_cfg.get('format', 'vp9'))
        self._v_codec, self._container, self._content_type = _VIDEO_CODEC_MAP.get(
            fmt, _VIDEO_CODEC_MAP['vp9']
        )
        self._audio_bitrate: int = int(stream_cfg.get('bitrate_kbps', 32))

        import os as _os
        fd, tmp = tempfile.mkstemp(suffix='.txt', prefix=f'haze_vt_{self._feed_id}_')
        _os.close(fd)
        self._text_file = pathlib.Path(tmp)
        self._text_file.write_text(' ', encoding='utf-8')

        self._crawl_repeat: int = max(0, int(self._style.get('crawl_repeat', 1)))
        self._alert_text: str = ' '
        self._clear_task: asyncio.Task | None = None

        self._proc_lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._current_color: str = _to_ffmpeg_color(_IDLE_BANNER_HEX)

        self._mode: str = _IDLE_MODE

        self._health_stop = threading.Event()
        self._health_watcher = threading.Thread(
            target=self._proc_health_watcher,
            name=f'video-health:{self._feed_id}',
            daemon=True,
        )
        self._start_ffmpeg(self._current_color)
        self._health_watcher.start()

    def _icecast_url(self) -> str:
        cfg = self._stream_cfg
        user = cfg.get('username', 'source')
        pw = cfg.get('password', '')
        host = cfg.get('host', 'localhost')
        port = int(cfg.get('port', 8000))
        mount = cfg.get('mount') or f'/{self._feed_id}'
        return f'icecast://{user}:{pw}@{host}:{port}{mount}'

    def _build_idle_cmd(self) -> list[str]:
        fps_str = str(self._fps)
        interlace = bool(self._style.get('interlace', False))
        bg_image = self._style.get('background_image')

        passthrough_cfg = _coerce_mapping(self._video_cfg.get('passthrough', {}))
        use_video_pt = passthrough_cfg.get('video', False)
        input_urls: list[str] = passthrough_cfg.get('input_urls') or []
        if isinstance(input_urls, str):
            input_urls = [input_urls]

        use_alpha = self._v_codec == 'libvpx-vp9' and not bg_image and not use_video_pt

        if use_video_pt and input_urls:
            video_input: list[str] = ['-i', str(input_urls[0])]
            src_filter = f'[1:v]scale={self._width}:{self._height},fps={fps_str}'
        elif bg_image:
            bg_path = pathlib.Path(str(bg_image))
            if bg_path.exists():
                video_input = ['-loop', '1', '-framerate', fps_str, '-i', str(bg_image)]
                src_filter = f'[1:v]scale={self._width}:{self._height}:force_original_aspect_ratio=increase,crop={self._width}:{self._height},fps={fps_str}'
            else:
                video_input = ['-f', 'lavfi', '-i', str(bg_image)]
                src_filter = f'[1:v]scale={self._width}:{self._height},fps={fps_str}'
        else:
            video_input = [
                '-f', 'lavfi',
                '-i', f'color=c=black:size={self._width}x{self._height}:r={fps_str}',
            ]
            src_filter = '[1:v]format=yuva420p,colorchannelmixer=aa=0' if use_alpha else '[1:v]null'

        filter_chain = src_filter
        if interlace:
            filter_chain = f'{filter_chain},setfield=tff'
        filter_complex = f'{filter_chain}[vout]'

        codec_extra = ['-pix_fmt', 'yuva420p', '-auto-alt-ref', '0'] if use_alpha else []

        return [
            'ffmpeg', '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS), '-i', 'pipe:0',
            *video_input,
            '-filter_complex', filter_complex,
            '-map', '0:a',
            '-map', '[vout]',
            '-c:a', 'libopus', '-b:a', f'{self._audio_bitrate}k',
            '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-c:v', self._v_codec,
            *codec_extra,
            '-b:v', '200k',
            '-deadline', 'realtime',
            '-cpu-used', '8',
            '-r', fps_str,
            '-content_type', self._content_type,
            '-f', self._container,
            self._icecast_url(),
        ]

    def _build_alert_cmd(self, banner_color: str) -> list[str]:
        bg_image = self._style.get('background_image')
        fps_str = str(self._fps)
        interlace = bool(self._style.get('interlace', False))

        passthrough_cfg = _coerce_mapping(self._video_cfg.get('passthrough', {}))
        use_video_pt = passthrough_cfg.get('video', False)
        input_urls: list[str] = passthrough_cfg.get('input_urls') or []
        if isinstance(input_urls, str):
            input_urls = [input_urls]

        video_input: list[str]
        src_filter: str

        if use_video_pt and input_urls:
            video_input = ['-i', str(input_urls[0])]
            src_filter = f'[1:v]scale={self._width}:{self._height},fps={fps_str}'
        elif bg_image:
            bg_path = pathlib.Path(str(bg_image))
            if bg_path.exists():
                video_input = ['-loop', '1', '-framerate', fps_str, '-i', str(bg_image)]
                src_filter = f'[1:v]scale={self._width}:{self._height}:force_original_aspect_ratio=increase,crop={self._width}:{self._height},fps={fps_str}'
            else:
                video_input = ['-f', 'lavfi', '-i', str(bg_image)]
                src_filter = f'[1:v]scale={self._width}:{self._height},fps={fps_str}'
        else:
            video_input = [
                '-f', 'lavfi',
                '-i', f'color=c=black:size={self._width}x{self._height}:r={fps_str}',
            ]
            src_filter = '[1:v]null'

        overlay = _build_drawtext_filter(
            self._text_file, banner_color, self._width, self._height, self._style
        )
        filter_chain = f'{src_filter},{overlay}'
        if interlace:
            filter_chain = f'{filter_chain},setfield=tff'
        filter_complex = f'{filter_chain}[vout]'

        return [
            'ffmpeg', '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            *video_input,
            '-filter_complex', filter_complex,
            '-map', '0:a',
            '-map', '[vout]',
            '-c:a', 'libopus', '-b:a', f'{self._audio_bitrate}k',
            '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-c:v', self._v_codec,
            '-b:v', '500k',
            '-deadline', 'realtime',
            '-cpu-used', '8',
            '-r', fps_str,
            '-content_type', self._content_type,
            '-f', self._container,
            self._icecast_url(),
        ]

    def _build_cmd(self, banner_color: str) -> list[str]:
        return self._build_idle_cmd() if self._mode == _IDLE_MODE else self._build_alert_cmd(banner_color)

    def _start_ffmpeg(self, banner_color: str) -> None:
        cmd = self._build_cmd(banner_color)
        with self._proc_lock:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self._current_color = banner_color
        log.info('[%s] Video: ffmpeg started (mode=%s, color=%s, codec=%s)', self._feed_id, self._mode, banner_color, self._v_codec)

    def _stop_ffmpeg(self) -> None:
        with self._proc_lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=4)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        except Exception:
            pass

    def _restart_ffmpeg(self, banner_color: str) -> None:
        self._stop_ffmpeg()
        self._start_ffmpeg(banner_color)

    def _proc_health_watcher(self) -> None:
        while not self._health_stop.is_set() and not shutdown_event.is_set():
            with self._proc_lock:
                proc = self._proc
            if proc is not None and proc.poll() is not None:
                log.warning(
                    '[%s] Video: ffmpeg exited unexpectedly (rc=%d), restarting in %.0fs',
                    self._feed_id, proc.returncode, _RECONNECT_DELAY_S,
                )
                self._health_stop.wait(timeout=_RECONNECT_DELAY_S)
                if not self._health_stop.is_set():
                    self._restart_ffmpeg(self._current_color)
            self._health_stop.wait(timeout=2.0)

    async def on_alert_start(self, identifier: str) -> None:
        """Call when an alert begins toning. Shows the crawl/overlay for that specific alert."""
        if self._closed:
            return
        if self._clear_task and not self._clear_task.done():
            self._clear_task.cancel()
            self._clear_task = None

        alerts = _get_active_alerts(self._feed_id)
        entry = next((a for a in alerts if a.get('identifier') == identifier), None)
        if entry is None:
            entry = {}

        meta = entry.get('metadata') or {}
        overlay_text = _build_overlay_text(entry, self._tz_name)
        self._alert_text = overlay_text
        try:
            self._text_file.write_text(overlay_text, encoding='utf-8')
        except Exception:
            pass

        new_color = (
            _pick_severity_color([entry], self._style)
            if meta.get('severity')
            else _to_ffmpeg_color('#FFCC00')
        )
        prev_mode = self._mode
        self._mode = _ALERT_MODE
        if prev_mode == _IDLE_MODE or new_color != self._current_color:
            log.info('[%s] Video: alert starting — restarting ffmpeg (mode→alert, color=%s)', self._feed_id, new_color)
            await asyncio.to_thread(self._restart_ffmpeg, new_color)
        else:
            log.debug('[%s] Video: alert starting — text updated', self._feed_id)

    async def on_alert_end(self) -> None:
        """Call when an alert finishes toning. Returns the overlay to idle state."""
        if self._closed:
            return

        crawl_repeat = self._crawl_repeat
        fmt = str(self._style.get('format', 'crawl'))
        if crawl_repeat > 0 and fmt == 'crawl' and self._alert_text.strip():
            font_size = int(self._style.get('font_size', 48))
            scroll_speed = font_size * 6
            text_px = len(self._alert_text) * font_size * 0.55
            pass_duration = (self._width + text_px) / scroll_speed
            delay = pass_duration * crawl_repeat

            async def _delayed_clear() -> None:
                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    return
                if self._closed:
                    return
                try:
                    self._text_file.write_text(' ', encoding='utf-8')
                except Exception:
                    pass
                idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
                self._mode = _IDLE_MODE
                log.info('[%s] Video: post-crawl — returning to idle', self._feed_id)
                await asyncio.to_thread(self._restart_ffmpeg, idle_color)

            self._clear_task = asyncio.ensure_future(_delayed_clear())
            log.debug('[%s] Video: crawl repeat — clearing in %.1fs', self._feed_id, delay)
        else:
            try:
                self._text_file.write_text(' ', encoding='utf-8')
            except Exception:
                pass
            idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
            self._mode = _IDLE_MODE
            log.info('[%s] Video: alert ended — returning to idle', self._feed_id)
            await asyncio.to_thread(self._restart_ffmpeg, idle_color)

    async def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        with self._proc_lock:
            proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            await asyncio.to_thread(proc.stdin.write, pcm)
        except BrokenPipeError:
            log.warning('[%s] Video: broken pipe — restarting ffmpeg', self._feed_id)
            self._restart_ffmpeg(self._current_color)
        except Exception as exc:
            log.error('[%s] Video: write error: %s', self._feed_id, exc)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._clear_task and not self._clear_task.done():
            self._clear_task.cancel()
        self._health_stop.set()
        self._stop_ffmpeg()
        try:
            self._text_file.unlink(missing_ok=True)
        except Exception:
            pass
        log.info('[%s] Video: closed', self._feed_id)

from __future__ import annotations

import asyncio
import base64
import logging
import os
import pathlib
import select
import shlex
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

try:
    import sounddevice as sd
except Exception:
    sd = None

from managed.events import NowPlayingMetadata
from module.buffer import CHANNELS, SAMPLE_RATE

log = logging.getLogger(__name__)

_CODEC_MAP: dict[str, tuple[str, str, str]] = {
    'opus': ('libopus',    'audio/ogg',  'ogg'),
    'flac':  ('flac',       'audio/flac', 'flac'),
    'ogg':  ('libvorbis',  'audio/ogg',  'ogg'),
    'mp3':  ('libmp3lame', 'audio/mpeg', 'mp3'),
    'aac':  ('aac',        'audio/aac',  'adts'),
}

_STREAM_DYNAMICS = (
    'equalizer=f=125:t=q:w=0.7:g=4,'
    'equalizer=f=200:t=q:w=1:g=2,'
    'equalizer=f=350:t=q:w=1:g=3,'
    'equalizer=f=500:t=q:w=1:g=2,'
    'equalizer=f=1200:t=q:w=1.2:g=1.5,'
    'equalizer=f=2500:t=q:w=1.5:g=2,'
    'acompressor=threshold=-22dB:ratio=16:attack=5:release=70:makeup=8dB,'
)

class IcecastSink:
    bus_queue_limit = 64
    bus_drop_oldest = False

    def __init__(self, config: dict[str, Any]) -> None:
        password = config.get('password', '') or ''
        username = config.get('username') or 'source'
        mount = config.get('mount') or f"/{config.get('feed_id', 'stream')}"
        self._host: str = config['host']
        self._port: int = config['port']
        self._mount: str = mount
        self._username: str = username
        self._password: str = password
        self._ssl: bool = config.get('ssl', False)
        self._stream_name: str = str(config.get('stream_name') or config.get('feed_id', 'stream')).strip()
        self._stream_description: str = str(config.get('stream_description') or '').strip()
        self._stream_genre: str = str(config.get('stream_genre') or 'Weather Radio').strip() or 'Weather Radio'
        self._stream_album: str = str(config.get('stream_album') or self._stream_name).strip() or self._stream_name
        self._stream_creator: str = str(config.get('stream_creator') or '').strip()
        self._stream_artist: str = str(config.get('stream_artist') or self._stream_name).strip() or self._stream_name

        url = (
            f"icecast://{username}:{password}@"
            f"{config['host']}:{config['port']}{mount}"
        )

        fmt = config.get('format', 'opus')
        codec, content_type, container = _CODEC_MAP.get(fmt, ('libopus', 'audio/ogg', 'ogg'))
        bitrate = config.get('bitrate_kbps', 32)

        self._cmd: list[str] = [
            'ffmpeg', '-loglevel', 'error',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            '-af', _STREAM_DYNAMICS,
            '-c:a', codec, '-b:a', f'{bitrate}k',
            '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-ice_name', self._stream_name,
            '-ice_description', self._stream_description,
            '-ice_genre', self._stream_genre,
            '-metadata', f'artist={self._stream_artist}',
            '-metadata', f'album={self._stream_album}',
            '-metadata', f'creator={self._stream_creator}',
            '-metadata', f'genre={self._stream_genre}',
            '-metadata', f'title={self._stream_name}',
            *(['-tls', '1'] if self._ssl else []),
            '-content_type', content_type,
            '-f', container,
            url,
        ]

        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE)
        self._closed = False
        self._reconnect_delay = 2.0
        self._max_reconnect_delay = 60.0
        self._consecutive_failures = 0

    def _restart_proc(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE)

    async def write(self, pcm: bytes) -> None:
        if self._closed or self._proc.stdin is None or not pcm:
            return
        try:
            await asyncio.to_thread(self._proc.stdin.write, pcm)
            self._consecutive_failures = 0
            self._reconnect_delay = 2.0
        except BrokenPipeError:
            self._consecutive_failures += 1
            delay = min(self._reconnect_delay, self._max_reconnect_delay)
            log.warning('Icecast: pipe broken (attempt %d), reconnecting in %.1fs',
                       self._consecutive_failures, delay)
            await asyncio.sleep(delay)
            self._reconnect_delay = min(self._reconnect_delay * 1.5, self._max_reconnect_delay)
            try:
                await asyncio.to_thread(self._restart_proc)
                log.info('Icecast reconnected to %s', self._mount)
            except Exception as e:
                if self._consecutive_failures >= 10:
                    log.error('Icecast reconnect failed after %d attempts: %s — stream disabled',
                             self._consecutive_failures, e)
                    self._closed = True
                else:
                    log.warning('Icecast reconnect failed: %s — will retry', e)
        except Exception as e:
            log.error('Icecast write error: %s', e)
            self._closed = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin:
            self._proc.stdin.close()
        await asyncio.to_thread(self._proc.wait)

    async def set_metadata(self, metadata: NowPlayingMetadata | str) -> None:
        if self._closed:
            return
        if isinstance(metadata, str):
            title = metadata.strip()
        else:
            title = str(metadata.title).strip()
        if not title:
            title = self._stream_name
        scheme = 'https' if self._ssl else 'http'
        params = {
            'mount': self._mount,
            'mode': 'updinfo',
            'song': metadata.title if isinstance(metadata, NowPlayingMetadata) else title,
            'artist': self._stream_artist,
            'genre': self._stream_genre,
            'name': self._stream_name,
            'description': self._stream_description,
        }
        url = (
            f"{scheme}://{self._host}:{self._port}/admin/metadata"
            f"?{urllib.parse.urlencode({k: v for k, v in params.items() if v})}"
        )
        credentials = base64.b64encode(
            f"{self._username}:{self._password}".encoded()
        ).decode()
        req = urllib.request.Request(
            url, headers={'Authorization': f'Basic {credentials}'}
        )
        for attempt in range(3):
            try:
                await asyncio.to_thread(urllib.request.urlopen, req, timeout=3)
                log.info('Icecast metadata updated: %s', title)
                return
            except urllib.error.HTTPError as e:
                if e.code == 404 and attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                log.warning('Icecast metadata update failed (%s): %s', url, e)
                return
            except Exception as e:
                log.warning('Icecast metadata update failed (%s): %s', url, e)
                return


def _build_video_overlay_inputs(
    video_cfg: dict[str, Any],
    width: int,
    height: int,
    fps: float,
    text_file: 'pathlib.Path',
    banner_color: str,
    *,
    idle: bool = False,
) -> tuple[list[str], str]:
    """Return (extra_ffmpeg_input_args, filter_complex_prefix) for a video overlay.

    Caller appends ``[vout]`` to the returned filter string and maps it.
    """
    from module.video import _build_drawtext_filter, _coerce_mapping
    import pathlib as _pathlib

    style = _coerce_mapping(video_cfg.get('style', {}))
    bg_image = style.get('background_image')
    fps_str = str(fps)

    passthrough_cfg = _coerce_mapping(video_cfg.get('passthrough', {}))
    use_video_pt = passthrough_cfg.get('video', False)
    input_urls = passthrough_cfg.get('input_urls') or []
    if isinstance(input_urls, str):
        input_urls = [input_urls]

    if use_video_pt and input_urls:
        extra_args = ['-i', str(input_urls[0])]
        scale_src = f'[1:v]scale={width}:{height},fps={fps_str}'
    elif bg_image:
        bg_path = _pathlib.Path(str(bg_image))
        if bg_path.exists():
            extra_args = ['-loop', '1', '-framerate', fps_str, '-i', str(bg_image)]
            scale_src = f'[1:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},fps={fps_str}'
        else:
            extra_args = ['-f', 'lavfi', '-i', str(bg_image)]
            scale_src = f'[1:v]scale={width}:{height},fps={fps_str}'
    else:
        extra_args = ['-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:r={fps_str}']
        scale_src = '[1:v]null'

    if idle:
        return extra_args, scale_src
    overlay = _build_drawtext_filter(text_file, banner_color, width, height, style)
    return extra_args, f'{scale_src},{overlay}'


class _VideoStreamSink:
    """Generic ffmpeg-backed sink that accepts PCM on write() and optionally shows an alert overlay.

    Pass a ``cmd_factory(banner_color: str) -> list[str]`` callable so the process can be
    restarted with a new overlay color when ``on_alert_start`` / ``on_alert_end`` are called.
    """

    bus_queue_limit = 64
    bus_drop_oldest = False

    def __init__(
        self,
        feed_id: str,
        cmd_factory: 'Callable[[str], list[str]]',
        text_file: 'pathlib.Path | None',
        initial_color: str,
        label: str,
        video_cfg: 'dict[str, Any] | None',
        *,
        tz_name: str = 'UTC',
        drop_oldest: bool = False,
        idle_factory: 'Callable[[], list[str]] | None' = None,
    ) -> None:
        self._feed_id = feed_id
        self._cmd_factory = cmd_factory
        self._idle_factory = idle_factory
        self._text_file = text_file
        self._current_color = initial_color
        self._label = label
        self._video_cfg = video_cfg
        self._tz_name = tz_name
        self._closed = False
        self.bus_drop_oldest = drop_oldest
        self._mode = 'idle' if idle_factory else 'alert'

        self._proc_lock = threading.Lock()
        initial_cmd = idle_factory() if idle_factory else cmd_factory(initial_color)
        self._proc = subprocess.Popen(initial_cmd, stdin=subprocess.PIPE)
        log.info('[%s] %s started (mode=%s)', feed_id, label, self._mode)

    def _rebuild_proc(self, banner_color: str) -> None:
        with self._proc_lock:
            old = self._proc
        try:
            if old.stdin:
                old.stdin.close()
        except Exception:
            pass
        try:
            old.wait(timeout=3)
        except subprocess.TimeoutExpired:
            old.kill()
            old.wait()
        cmd = self._idle_factory() if (self._mode == 'idle' and self._idle_factory) else self._cmd_factory(banner_color)
        with self._proc_lock:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            self._current_color = banner_color

    async def on_alert_start(self, identifier: str) -> None:
        if self._closed:
            return
        from module.video import (
            _get_active_alerts, _pick_severity_color, _to_ffmpeg_color, _coerce_mapping,
            _build_overlay_text,
        )
        alerts = _get_active_alerts(self._feed_id)
        entry = next((a for a in alerts if a.get('identifier') == identifier), None) or {}
        meta = entry.get('metadata') or {}
        overlay_text = _build_overlay_text(entry, self._tz_name)
        if self._text_file:
            try:
                self._text_file.write_text(overlay_text, encoding='utf-8')
            except Exception:
                pass
        style = _coerce_mapping((self._video_cfg or {}).get('style', {}))
        new_color = (
            _pick_severity_color([entry], style)
            if meta.get('severity')
            else _to_ffmpeg_color('#FFCC00')
        )
        prev_mode = self._mode
        self._mode = 'alert'
        if prev_mode == 'idle' or new_color != self._current_color:
            log.info('[%s] %s: alert start — restarting (mode→alert, color=%s)', self._feed_id, self._label, new_color)
            await asyncio.to_thread(self._rebuild_proc, new_color)
        else:
            log.debug('[%s] %s: alert start — text updated', self._feed_id, self._label)

    async def on_alert_end(self) -> None:
        if self._closed:
            return
        if self._text_file:
            try:
                self._text_file.write_text(' ', encoding='utf-8')
            except Exception:
                pass
        from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
        idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
        if self._idle_factory:
            self._mode = 'idle'
            log.info('[%s] %s: alert end — returning to idle (no overlay)', self._feed_id, self._label)
            await asyncio.to_thread(self._rebuild_proc, idle_color)
        elif idle_color != self._current_color:
            log.info('[%s] %s: alert end — returning to idle color', self._feed_id, self._label)
            await asyncio.to_thread(self._rebuild_proc, idle_color)

    async def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        with self._proc_lock:
            proc = self._proc
        if proc.stdin is None:
            return
        try:
            await asyncio.to_thread(proc.stdin.write, pcm)
        except BrokenPipeError:
            log.warning('[%s] %s: broken pipe, restarting', self._feed_id, self._label)
            await asyncio.to_thread(self._rebuild_proc, self._current_color)
        except Exception as exc:
            log.error('[%s] %s: write error: %s', self._feed_id, self._label, exc)
            self._closed = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._proc_lock:
            proc = self._proc
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        await asyncio.to_thread(proc.wait)
        if self._text_file is not None:
            try:
                self._text_file.unlink(missing_ok=True)
            except Exception:
                pass
        log.info('[%s] %s closed', self._feed_id, self._label)


def _build_video_stream_cmd(
    feed_id: str,
    audio_bitrate_kbps: int,
    video_bitrate_kbps: int,
    vcodec: str,
    acodec: str,
    container: str,
    output_url: str,
    extra_audio_inputs: list[str],
    video_cfg: 'dict[str, Any] | None',
    width: int,
    height: int,
    fps: float,
    text_file: 'pathlib.Path | None' = None,
    banner_color: str | None = None,
    extra_output_args: list[str] | None = None,
    idle: bool = False,
) -> list[str]:
    """Build an ffmpeg command that accepts PCM audio on stdin and streams to output_url."""
    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    effective_color = banner_color or _to_ffmpeg_color(_IDLE_BANNER_HEX)
    has_video = bool(vcodec)

    video_inputs: list[str] = []
    filter_args: list[str] = []

    if has_video and video_cfg and text_file:
        video_inputs, filter_complex = _build_video_overlay_inputs(
            video_cfg, width, height, fps, text_file, effective_color, idle=idle,
        )
        filter_args = ['-filter_complex', f'{filter_complex}[vout]', '-map', '0:a', '-map', '[vout]']
    elif has_video:
        fps_str = str(fps)
        video_inputs = ['-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:r={fps_str}']
        filter_args = ['-map', '0:a', '-map', '1:v']

    a_codec_args: list[str]
    if acodec == 'pcm' or not acodec:
        a_codec_args = ['-c:a', 'pcm_s16le']
    else:
        a_codec_args = ['-c:a', acodec, '-b:a', f'{audio_bitrate_kbps}k']

    v_codec_args: list[str] = []
    if has_video:
        v_codec_args = ['-c:v', vcodec, '-b:v', f'{video_bitrate_kbps}k', '-r', str(fps)]

    return [
        'ffmpeg', '-loglevel', 'warning',
        '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
        '-i', 'pipe:0',
        *extra_audio_inputs,
        *video_inputs,
        *filter_args,
        *a_codec_args,
        *v_codec_args,
        *(extra_output_args or []),
        '-f', container,
        output_url,
    ]


def _make_video_text_file(feed_id: str, label: str) -> 'pathlib.Path':
    fd, tmp = tempfile.mkstemp(suffix='.txt', prefix=f'haze_vt_{feed_id}_{label}_')
    os.close(fd)
    tf = pathlib.Path(tmp)
    tf.write_text(' ', encoding='utf-8')
    return tf


def UdpSink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    ip = config.get('ip', '127.0.0.1')
    port = int(config.get('port', 8899))
    fmt = str(config.get('format', 'mpegts'))
    audio_br = int(config.get('bitrate_kbps', 32))
    video_br = int(config.get('vrate_kbps', 1000))
    vcodec = str(config.get('vcodec') or '')
    acodec = str(config.get('acodec') or 'aac')
    width = int(config.get('width', 1920))
    height = int(config.get('height', 1080))
    fps = float(config.get('fps', 29.97))
    url = f'udp://{ip}:{port}?pkt_size=1316'

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'udp') if vcodec and video_cfg else None

    def build(color: str) -> list[str]:
        return _build_video_stream_cmd(
            feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
            width, height, fps, text_file=tf, banner_color=color,
        )
    idle_factory = None
    if tf and video_cfg:
        def build_idle_udp() -> list[str]:
            return _build_video_stream_cmd(
                feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
                width, height, fps, text_file=tf, banner_color=idle_color, idle=True,
            )
        idle_factory = build_idle_udp
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'UDP({ip}:{port})', video_cfg, tz_name=tz_name, idle_factory=idle_factory)


def RtpSink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    ip = config.get('ip', '127.0.0.1')
    port = int(config.get('port', 8899))
    fmt = str(config.get('format', 'rtp_mpegts'))
    audio_br = int(config.get('bitrate_kbps', 32))
    video_br = int(config.get('vrate_kbps', 1000))
    vcodec = str(config.get('vcodec') or '')
    acodec = str(config.get('acodec') or 'aac')
    width = int(config.get('width', 1920))
    height = int(config.get('height', 1080))
    fps = float(config.get('fps', 29.97))
    url = f'rtp://{ip}:{port}?pkt_size=1316'

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'rtp') if vcodec and video_cfg else None

    def build(color: str) -> list[str]:
        return _build_video_stream_cmd(
            feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
            width, height, fps, text_file=tf, banner_color=color,
        )
    idle_factory = None
    if tf and video_cfg:
        def build_idle_rtp() -> list[str]:
            return _build_video_stream_cmd(
                feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
                width, height, fps, text_file=tf, banner_color=idle_color, idle=True,
            )
        idle_factory = build_idle_rtp
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'RTP({ip}:{port})', video_cfg, tz_name=tz_name, idle_factory=idle_factory)


def RtmpSink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    url = str(config.get('url', 'rtmp://localhost/live/stream'))
    audio_br = int(config.get('bitrate_kbps', 32))
    video_br = int(config.get('vrate_kbps', 1000))
    vcodec = str(config.get('vcodec') or '')
    acodec = str(config.get('acodec') or 'aac')
    width = int(config.get('width', 1920))
    height = int(config.get('height', 1080))
    fps = float(config.get('fps', 29.97))

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'rtmp') if vcodec and video_cfg else None

    def build(color: str) -> list[str]:
        return _build_video_stream_cmd(
            feed_id, audio_br, video_br, vcodec, acodec, 'flv', url, [], video_cfg,
            width, height, fps, text_file=tf, banner_color=color,
        )
    idle_factory = None
    if tf and video_cfg:
        def build_idle_rtmp() -> list[str]:
            return _build_video_stream_cmd(
                feed_id, audio_br, video_br, vcodec, acodec, 'flv', url, [], video_cfg,
                width, height, fps, text_file=tf, banner_color=idle_color, idle=True,
            )
        idle_factory = build_idle_rtmp
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'RTMP({url})', video_cfg, tz_name=tz_name, idle_factory=idle_factory)


def SrtSink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    url = str(config.get('url', 'srt://localhost:12345'))
    fmt = str(config.get('format', 'mpegts'))
    audio_br = int(config.get('bitrate_kbps', 32))
    video_br = int(config.get('vrate_kbps', 1000))
    vcodec = str(config.get('vcodec') or '')
    acodec = str(config.get('acodec') or 'aac')
    width = int(config.get('width', 1920))
    height = int(config.get('height', 1080))
    fps = float(config.get('fps', 29.97))

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'srt') if vcodec and video_cfg else None

    def build(color: str) -> list[str]:
        return _build_video_stream_cmd(
            feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
            width, height, fps, text_file=tf, banner_color=color,
        )
    idle_factory = None
    if tf and video_cfg:
        def build_idle_srt() -> list[str]:
            return _build_video_stream_cmd(
                feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
                width, height, fps, text_file=tf, banner_color=idle_color, idle=True,
            )
        idle_factory = build_idle_srt
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'SRT({url})', video_cfg, tz_name=tz_name, idle_factory=idle_factory)


def RtspSink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    url = str(config.get('url', 'rtsp://localhost:8554/stream'))
    fmt = str(config.get('format', 'rtsp'))
    audio_br = int(config.get('bitrate_kbps', 32))
    video_br = int(config.get('vrate_kbps', 1000))
    vcodec = str(config.get('vcodec') or '')
    acodec = str(config.get('acodec') or 'aac')
    width = int(config.get('width', 1920))
    height = int(config.get('height', 1080))
    fps = float(config.get('fps', 29.97))

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'rtsp') if vcodec and video_cfg else None

    def build(color: str) -> list[str]:
        return _build_video_stream_cmd(
            feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
            width, height, fps, text_file=tf, banner_color=color,
            extra_output_args=['-rtsp_transport', 'tcp'],
        )
    idle_factory = None
    if tf and video_cfg:
        def build_idle_rtsp() -> list[str]:
            return _build_video_stream_cmd(
                feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
                width, height, fps, text_file=tf, banner_color=idle_color, idle=True,
                extra_output_args=['-rtsp_transport', 'tcp'],
            )
        idle_factory = build_idle_rtsp
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'RTSP({url})', video_cfg, tz_name=tz_name, idle_factory=idle_factory)


def FramebufferSink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    path = str(config.get('path', '/dev/fb0'))
    width = int(config.get('width', 1920))
    height = int(config.get('height', 1080))
    fps = float(config.get('fps', 29.97))
    fps_str = str(fps)

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'fb') if video_cfg else None

    def build(color: str) -> list[str]:
        if video_cfg and tf:
            vi, fc = _build_video_overlay_inputs(video_cfg, width, height, fps, tf, color)
            filter_args = [
                '-filter_complex',
                f'{fc},scale={width}:{height},format=bgr0[vout]',
                '-map', '[vout]',
            ]
        else:
            vi = ['-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:r={fps_str}']
            filter_args = ['-map', '1:v']
        return [
            'ffmpeg', '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            *vi, *filter_args,
            *([] if video_cfg and tf else ['-vf', f'scale={width}:{height},format=bgr0']),
            '-r', fps_str,
            '-f', 'fbdev', path,
        ]
    idle_factory = None
    if video_cfg and tf:
        def build_idle_fb() -> list[str]:
            vi, fc = _build_video_overlay_inputs(video_cfg, width, height, fps, tf, idle_color, idle=True)
            return [
                'ffmpeg', '-loglevel', 'warning',
                '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
                '-i', 'pipe:0',
                *vi,
                '-filter_complex', f'{fc},scale={width}:{height},format=bgr0[vout]',
                '-map', '[vout]',
                '-r', fps_str,
                '-f', 'fbdev', path,
            ]
        idle_factory = build_idle_fb
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'Framebuffer({path})', video_cfg, drop_oldest=True, tz_name=tz_name, idle_factory=idle_factory)


def DriSink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    path = str(config.get('path', '/dev/dri/card0'))
    width = int(config.get('width', 1920))
    height = int(config.get('height', 1080))
    fps = float(config.get('fps', 29.97))
    fps_str = str(fps)

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'dri') if video_cfg else None

    def build(color: str) -> list[str]:
        if video_cfg and tf:
            vi, fc = _build_video_overlay_inputs(video_cfg, width, height, fps, tf, color)
            filter_args = [
                '-filter_complex',
                f'{fc},scale={width}:{height},format=yuv420p[vout]',
                '-map', '[vout]',
            ]
        else:
            vi = ['-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:r={fps_str}']
            filter_args = ['-map', '1:v']
        return [
            'ffmpeg', '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            *vi, *filter_args,
            *([] if video_cfg and tf else ['-vf', f'scale={width}:{height},format=yuv420p']),
            '-r', fps_str,
            '-f', 'drm_output',
            '-device', path,
            '-',
        ]
    idle_factory = None
    if video_cfg and tf:
        def build_idle_dri() -> list[str]:
            vi, fc = _build_video_overlay_inputs(video_cfg, width, height, fps, tf, idle_color, idle=True)
            return [
                'ffmpeg', '-loglevel', 'warning',
                '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
                '-i', 'pipe:0',
                *vi,
                '-filter_complex', f'{fc},scale={width}:{height},format=yuv420p[vout]',
                '-map', '[vout]',
                '-r', fps_str,
                '-f', 'drm_output',
                '-device', path,
                '-',
            ]
        idle_factory = build_idle_dri
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'DRI({path})', video_cfg, drop_oldest=True, tz_name=tz_name, idle_factory=idle_factory)


def V4L2Sink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    device = str(config.get('device', '/dev/video0'))
    width = int(config.get('width', 1920))
    height = int(config.get('height', 1080))
    fps = float(config.get('fps', 29.97))
    fps_str = str(fps)

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'v4l2') if video_cfg else None

    def build(color: str) -> list[str]:
        if video_cfg and tf:
            vi, fc = _build_video_overlay_inputs(video_cfg, width, height, fps, tf, color)
            filter_args = [
                '-filter_complex',
                f'{fc},scale={width}:{height},format=yuv420p[vout]',
                '-map', '[vout]',
            ]
        else:
            vi = ['-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:r={fps_str}']
            filter_args = ['-map', '1:v']
        return [
            'ffmpeg', '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            *vi, *filter_args,
            *([] if video_cfg and tf else ['-vf', f'scale={width}:{height},format=yuv420p']),
            '-r', fps_str,
            '-f', 'v4l2', device,
        ]
    idle_factory = None
    if video_cfg and tf:
        def build_idle_v4l2() -> list[str]:
            vi, fc = _build_video_overlay_inputs(video_cfg, width, height, fps, tf, idle_color, idle=True)
            return [
                'ffmpeg', '-loglevel', 'warning',
                '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
                '-i', 'pipe:0',
                *vi,
                '-filter_complex', f'{fc},scale={width}:{height},format=yuv420p[vout]',
                '-map', '[vout]',
                '-r', fps_str,
                '-f', 'v4l2', device,
            ]
        idle_factory = build_idle_v4l2
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'V4L2({device})', video_cfg, drop_oldest=True, tz_name=tz_name, idle_factory=idle_factory)


class AudioDeviceSink:
    bus_queue_limit = 8
    bus_drop_oldest = True

    def __init__(self, device: str | int | None = None) -> None:
        if sd is None:
            raise RuntimeError('sounddevice is unavailable; install PortAudio and the sounddevice package to use the audio device sink')
        self._stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            device=device,
            latency='high',
        )
        self._stream.start()

    async def write(self, pcm: bytes) -> None:
        await asyncio.to_thread(self._stream.write, pcm)

    async def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class FileSink:
    def __init__(self, path: str) -> None:
        self._f = open(path, 'wb')

    async def write(self, pcm: bytes) -> None:
        await asyncio.to_thread(self._f.write, pcm)

    async def close(self) -> None:
        self._f.close()


_RADIO_DYNAMICS = (
    _STREAM_DYNAMICS +
    'lowpass=f=2800,'
    'highpass=f=120,'
    'lowpass=f=3000,'
    'volume=2,'
)

_PIFMADV_PREFILL_CHUNKS = 10
_WRITE_STALL_TIMEOUT = 1.0


class PiFmAdvSink:
    bus_queue_limit = 48
    bus_drop_oldest = False
    bus_clocked = True
    bus_prefill_chunks = _PIFMADV_PREFILL_CHUNKS
    bus_fill_silence = True

    def __init__(self, config: dict[str, Any]) -> None:
        freq_mhz: str = str(config['frequency_mhz'])
        dev: int = config.get('deviation_hz', 5000)
        bw: int = config.get('bandwidth_hz', 10000)
        bin_root: str = config.get('bin_root', '/home/pi/PiFmAdv/src')
        pi_fm_adv_bin: str = f"{bin_root}/pi_fm_adv"
        alt_freqs: list = config.get('alternative_frequencies', [])
        use_sudo: bool = config.get('use_sudo', True)
        tx_power: int = config.get('tx_power', 4)
        ssh_cfg: dict = config.get('ssh', {})
        self._use_ssh: bool = ssh_cfg.get('enabled', False)
        ssh_bin: str = ssh_cfg.get('ssh_bin', 'ssh')
        ssh_port: int = int(ssh_cfg.get('port') or 22)
        ssh_host: str = ssh_cfg.get('host', '')
        ssh_key: str = os.path.expanduser(str(ssh_cfg.get('public_key_path', '~/.ssh/id_rsa')))
        ssh_user: str = ssh_cfg.get('username', 'pi')
        ffmpeg_bin: str = ssh_cfg.get('ffmpeg_bin') or config.get('ffmpeg_bin') or 'ffmpeg'

        fm_args: list[str] = []
        if use_sudo:
            fm_args += ['sudo', '-n']
        fm_args += [
            pi_fm_adv_bin,
            '--audio', '-',
            '--freq', freq_mhz,
            '--dev', str(dev),
            '--power', str(tx_power),
            '--preemph', '75us',
            '--rds', '0',
        ]
        if alt_freqs:
            fm_args += ['--af', ','.join(str(f) for f in alt_freqs)]

        self._cleanup_cmd: list[str] = []
        if self._use_ssh:
            udp_port: int = int(ssh_cfg.get('udp_port') or 5000)
            remote_ffmpeg_bin: str = str(ssh_cfg.get('remote_ffmpeg_bin') or 'ffmpeg')
            _ssh_base: list[str] = [
                ssh_bin, '-T',
                '-i', ssh_key,
                '-p', str(ssh_port),
                '-l', ssh_user,
                '-o', 'BatchMode=yes',
                ssh_host,
            ]
            sudo = 'sudo -n ' if use_sudo else ''
            self._cleanup_cmd = _ssh_base + [
                f'{sudo}fuser -k {udp_port}/udp 2>/dev/null || true; '
                f'{sudo}pkill -x pi_fm_adv 2>/dev/null || true; '
                'sleep 1'
            ]
            receiver_script = (
                f'{shlex.quote(remote_ffmpeg_bin)} -loglevel warning '
                f'-rtbufsize 64M '
                f'-i {shlex.quote(f"rtp://0.0.0.0:{udp_port}?buffer_size=4194304&fifo_size=4194304&pkt_size=1316")} '
                f'-vn -af aresample=async=1000:min_hard_comp=0.100 -ac 1 -ar 8000 -f wav - | '
                + shlex.join(fm_args)
            )
            self._fm_cmd: list[str] = _ssh_base + [receiver_script]
            self._ffmpeg_cmd: list[str] = [
                ffmpeg_bin, '-loglevel', 'warning',
                '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
                '-i', 'pipe:0',
                '-af', _RADIO_DYNAMICS,
                '-c:a', 'libopus', '-b:a', f'32k', '-application', 'audio',
                '-f', 'rtp_mpegts',
                f'rtp://{ssh_host}:{udp_port}?pkt_size=1316',
            ]
        else:
            self._fm_cmd = fm_args
            self._ffmpeg_cmd = [
                ffmpeg_bin, '-loglevel', 'warning',
                '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
                '-i', 'pipe:0',
                '-ac', str(CHANNELS), '-ar', str(SAMPLE_RATE),
                '-af', _RADIO_DYNAMICS,
                '-f', 'wav', 'pipe:1',
            ]

        self._ffmpeg: subprocess.Popen | None = None
        self._fm: subprocess.Popen | None = None
        self._closed = False
        self._label = f"{freq_mhz} MHz dev=±{dev} Hz bw={bw} Hz"
        self._start()
        log.info('PiFmAdv sink started: %s ssh=%s', self._label, ssh_host if self._use_ssh else 'disabled')

    def _start(self) -> None:
        if self._cleanup_cmd:
            subprocess.run(
                self._cleanup_cmd,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=15,
            )
        if self._use_ssh:
            self._fm = subprocess.Popen(
                self._fm_cmd,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            time.sleep(0.5)
            self._ffmpeg = subprocess.Popen(
                self._ffmpeg_cmd,
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        else:
            self._ffmpeg = subprocess.Popen(
                self._ffmpeg_cmd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            self._fm = subprocess.Popen(
                self._fm_cmd,
                stdin=self._ffmpeg.stdout,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if self._ffmpeg.stdout is not None:
                self._ffmpeg.stdout.close()
            time.sleep(0.2)

    def _kill(self) -> None:
        for proc in (self._ffmpeg, self._fm):
            if proc is None:
                continue
            try:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        self._ffmpeg = None
        self._fm = None

    def _restart(self) -> None:
        self._kill()
        time.sleep(0.1)
        self._start()

    def _subprocesses_alive(self) -> bool:
        if self._ffmpeg is not None and self._ffmpeg.poll() is not None:
            log.warning('PiFmAdv: ffmpeg exited (rc=%s) on %s', self._ffmpeg.returncode, self._label)
            return False
        if self._fm is not None and self._fm.poll() is not None:
            log.warning('PiFmAdv: fm process exited (rc=%s) on %s', self._fm.returncode, self._label)
            return False
        return True

    async def write(self, pcm: bytes) -> None:
        if self._closed or self._ffmpeg is None or self._ffmpeg.stdin is None or not pcm:
            return
        if not self._subprocesses_alive():
            log.warning('PiFmAdv: dead subprocess detected on %s, restarting', self._label)
            await asyncio.to_thread(self._restart)
            raise RuntimeError('PiFmAdv: subprocess died; restarted')
        ffmpeg = self._ffmpeg
        stdin = ffmpeg.stdin
        if stdin is None:
            return
        try:
            fd = stdin.fileno()
        except Exception:
            return
        _, writable, _ = await asyncio.to_thread(select.select, [], [fd], [], _WRITE_STALL_TIMEOUT)
        if not writable:
            log.warning('PiFmAdv: write stall on %s, restarting', self._label)
            await asyncio.to_thread(self._restart)
            raise RuntimeError('PiFmAdv: write stall; restarted')
        try:
            await asyncio.to_thread(stdin.write, pcm)
        except BrokenPipeError as exc:
            log.warning('PiFmAdv: broken pipe on %s, restarting', self._label)
            await asyncio.to_thread(self._restart)
            raise RuntimeError('PiFmAdv pipe broken') from exc
        except OSError as exc:
            log.warning('PiFmAdv: OS error on %s, restarting: %s', self._label, exc)
            await asyncio.to_thread(self._restart)
            raise RuntimeError(f'PiFmAdv OS error: {exc}') from exc
        except Exception as exc:
            log.error('PiFmAdv: unexpected write error on %s: %s', self._label, exc)
            self._closed = True
            raise RuntimeError(f'PiFmAdv write error: {exc}') from exc

    async def close(self) -> None:
        if self._ffmpeg is None and self._fm is None:
            return
        self._closed = True
        await asyncio.to_thread(self._kill)
        log.info('PiFmAdv sink closed')

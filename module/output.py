from __future__ import annotations

import asyncio
import base64
import datetime
import logging
import os
import pathlib
import select
import shlex
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zoneinfo
from typing import Any, Callable

try:
    import sounddevice as sd
except Exception:
    sd = None

from managed.events import NowPlayingMetadata
from module.buffer import CHANNELS, SAMPLE_RATE

log = logging.getLogger(__name__)

_STANDARD_STREAM_QUEUE_LIMIT = 24
_LOW_LATENCY_STREAM_QUEUE_LIMIT = 12
_LOW_LATENCY_STREAM_PREFILL_CHUNKS = 2

_CODEC_MAP: dict[str, tuple[str, str, str]] = {
    'opus': ('libopus',    'audio/ogg',  'ogg'),
    'flac':  ('flac',       'audio/flac', 'flac'),
    'ogg':  ('libvorbis',  'audio/ogg',  'ogg'),
    'mp3':  ('libmp3lame', 'audio/mpeg', 'mp3'),
    'aac':  ('aac',        'audio/aac',  'adts'),
}

_STREAM_DYNAMICS = (
    'highpass=f=90,'
    'equalizer=f=120:t=q:w=1.2:g=4,'
    'equalizer=f=600:t=q:w=1.5:g=-4,'
    'equalizer=f=2500:t=q:w=0.6:g=2,'
    'lowpass=f=9500,'
    'acompressor=threshold=-15dB:ratio=4:attack=5:release=120:makeup=6dB,'
)


class IcecastSink:
    bus_queue_limit = _STANDARD_STREAM_QUEUE_LIMIT
    bus_drop_oldest = True

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
            '-af', _STREAM_DYNAMICS + 'alimiter=limit=0.50,',
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

        self._proc_lock = threading.Lock()
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE)
        self._closed = False
        self._reconnect_delay = 2.0
        self._max_reconnect_delay = 60.0
        self._consecutive_failures = 0

    def _restart_proc(self) -> None:
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
            self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE)

    def _write_proc(self, pcm: bytes) -> None:
        with self._proc_lock:
            if self._proc.stdin is None:
                raise BrokenPipeError('icecast ffmpeg stdin unavailable')
            self._proc.stdin.write(pcm)

    async def _recover_from_write_failure(self, reason: Exception) -> None:
        self._consecutive_failures += 1
        delay = min(self._reconnect_delay, self._max_reconnect_delay)
        log.warning(
            'Icecast write failed (%s, attempt %d), reconnecting in %.1fs',
            reason,
            self._consecutive_failures,
            delay,
        )
        await asyncio.sleep(delay)
        self._reconnect_delay = min(self._reconnect_delay * 1.5, self._max_reconnect_delay)
        try:
            await asyncio.to_thread(self._restart_proc)
            log.info('Icecast reconnected to %s', self._mount)
        except Exception as exc:
            if self._consecutive_failures >= 10:
                log.error(
                    'Icecast reconnect failed after %d attempts: %s — stream disabled',
                    self._consecutive_failures,
                    exc,
                )
                self._closed = True
            else:
                log.warning('Icecast reconnect failed: %s — will retry', exc)

    async def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        try:
            await asyncio.to_thread(self._write_proc, pcm)
            self._consecutive_failures = 0
            self._reconnect_delay = 2.0
        except RuntimeError as exc:
            if 'cannot schedule new futures after shutdown' in str(exc).lower():
                return
            await self._recover_from_write_failure(exc)
        except (BrokenPipeError, OSError, ValueError) as exc:
            await self._recover_from_write_failure(exc)
        except Exception as exc:
            await self._recover_from_write_failure(exc)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._proc_lock:
            proc = self._proc
        if proc.stdin:
            proc.stdin.close()
        await asyncio.to_thread(proc.wait)

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
            f"{self._username}:{self._password}".encode()
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
    raw_idle_details = style.get('idle_details_channel', False)
    if isinstance(raw_idle_details, dict):
        idle_details_enabled = bool(raw_idle_details.get('enabled', False))
    else:
        idle_details_enabled = bool(raw_idle_details)
    show_idle_details = idle and str(style.get('format', 'crawl')).lower() == 'fullscreen' and idle_details_enabled
    bg_image = style.get('background_image')
    fps_str = str(fps)

    passthrough_cfg = _coerce_mapping(video_cfg.get('passthrough', {}))
    use_video_pt = passthrough_cfg.get('video', False)
    input_urls = passthrough_cfg.get('input_urls') or []
    if isinstance(input_urls, str):
        input_urls = [input_urls]

    if idle and not show_idle_details:
        extra_args = ['-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:r={fps_str}']
        return extra_args, '[1:v]null'

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
        if show_idle_details:
            overlay = _build_drawtext_filter(text_file, banner_color, width, height, style)
            return extra_args, f'{scale_src},{overlay}'
        return extra_args, scale_src
    overlay = _build_drawtext_filter(text_file, banner_color, width, height, style)
    return extra_args, f'{scale_src},{overlay}'


def _resolve_idle_details_cfg(video_cfg: dict[str, Any] | None) -> dict[str, Any]:
    from module.video import _coerce_mapping

    style = _coerce_mapping((video_cfg or {}).get('style', {}))
    raw = style.get('idle_details_channel', False)
    if isinstance(raw, dict):
        cfg: dict[str, Any] = dict(raw)
    else:
        cfg = {'enabled': bool(raw)}
    cfg.setdefault('enabled', False)
    cfg.setdefault('title', 'EAS Details Channel')
    cfg.setdefault('refresh_seconds', 1.0)
    return cfg


def _resolve_video_output_params(
    sink_cfg: dict[str, Any],
    video_cfg: dict[str, Any] | None,
) -> tuple[int, int, float, bool]:
    from module.video import _coerce_mapping

    video_map = video_cfg if isinstance(video_cfg, dict) else {}
    style = _coerce_mapping(video_map.get('style', {}))
    width = int(sink_cfg.get('width') or video_map.get('width') or 1920)
    height = int(sink_cfg.get('height') or video_map.get('height') or 1080)
    fps = float(sink_cfg.get('fps') or video_map.get('fps') or 29.97)
    if 'interlace' in sink_cfg:
        interlace = bool(sink_cfg.get('interlace'))
    else:
        interlace = bool(style.get('interlace', False))
    return width, height, fps, interlace


def _host_identity() -> tuple[str, str]:
    hostname = socket.gethostname()
    ip_address = '127.0.0.1'
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(('8.8.8.8', 80))
            ip_address = sock.getsockname()[0]
    except OSError:
        try:
            resolved = socket.gethostbyname(hostname)
            if resolved:
                ip_address = resolved
        except OSError:
            pass
    return hostname, ip_address


def _low_latency_video_args(vcodec: str, fps: float) -> list[str]:
    codec = str(vcodec or '').strip().lower()
    if not codec:
        return []

    gop = max(1, round(fps))
    if codec == 'libx264':
        return ['-preset', 'veryfast', '-tune', 'zerolatency', '-g', str(gop)]
    if codec == 'libx265':
        return [
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-x265-params', f'bframes=0:rc-lookahead=0:keyint={gop}:min-keyint={gop}:scenecut=0',
        ]
    if codec == 'libvpx':
        return ['-deadline', 'realtime', '-cpu-used', '8', '-lag-in-frames', '0']
    if codec == 'libvpx-vp9':
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
        queue_limit: int = _STANDARD_STREAM_QUEUE_LIMIT,
        drop_oldest: bool = False,
        idle_factory: 'Callable[[], list[str]] | None' = None,
        frame_width: int | None = None,
        clocked: bool = False,
        prefill_chunks: int = 0,
        fill_silence: bool = False,
    ) -> None:
        self._feed_id = feed_id
        self._cmd_factory = cmd_factory
        self._idle_factory = idle_factory
        self._text_file = text_file
        self._current_color = initial_color
        self._label = label
        self._video_cfg = video_cfg
        self._tz_name = tz_name
        self._frame_width = int(frame_width or ((video_cfg or {}).get('width') or 1920))
        self._closed = False
        self.bus_queue_limit = max(1, int(queue_limit))
        self.bus_drop_oldest = drop_oldest
        self.bus_clocked = clocked
        self.bus_prefill_chunks = max(0, int(prefill_chunks))
        self.bus_fill_silence = fill_silence
        self._mode = 'idle' if idle_factory else 'alert'
        self._suspend_writes = False
        self._idle_details_cfg = _resolve_idle_details_cfg(video_cfg)
        self._idle_details_enabled = bool(self._idle_details_cfg.get('enabled', False))
        self._idle_details_refresh_s = max(0.5, float(self._idle_details_cfg.get('refresh_seconds', 1.0) or 1.0))
        self._hostname, self._ip_address = _host_identity()
        self._last_idle_text_refresh = 0.0

        if self._mode == 'idle':
            self._refresh_idle_text(force=True)

        self._proc_lock = threading.Lock()
        self._proc = subprocess.Popen(self._build_cmd(initial_color), stdin=subprocess.PIPE)
        log.info('[%s] %s started', feed_id, label)

    def _build_cmd(self, banner_color: str) -> list[str]:
        if self._mode == 'idle' and self._idle_factory is not None:
            return self._idle_factory()
        return self._cmd_factory(banner_color)

    def _build_idle_details_text(self) -> str:
        try:
            now = datetime.datetime.now(zoneinfo.ZoneInfo(self._tz_name))
        except Exception:
            now = datetime.datetime.now().astimezone()
        title = str(self._idle_details_cfg.get('title') or 'EAS Details Channel').strip() or 'EAS Details Channel'
        return '\n'.join([
            title,
            f'Feed: {self._feed_id}',
            f'Time: {now.strftime("%Y-%m-%d %I:%M:%S %p %Z")}',
            f'Host: {self._hostname}',
            f'IP: {self._ip_address}',
        ])

    def _refresh_idle_text(self, *, force: bool = False) -> None:
        if not self._text_file or not self._idle_details_enabled or self._mode != 'idle':
            return
        now = time.monotonic()
        if not force and now - self._last_idle_text_refresh < self._idle_details_refresh_s:
            return
        self._text_file.write_text(self._build_idle_details_text(), encoding='utf-8')
        self._last_idle_text_refresh = now

    def _rebuild_proc(self, banner_color: str) -> None:
        with self._proc_lock:
            old = self._proc
            try:
                if old.stdin:
                    old.stdin.close()
            except Exception:
                pass
            if old.poll() is None:
                try:
                    old.terminate()
                except Exception:
                    pass
                try:
                    old.wait(timeout=0.35)
                except subprocess.TimeoutExpired:
                    old.kill()
                    old.wait(timeout=1.0)
            self._proc = subprocess.Popen(self._build_cmd(banner_color), stdin=subprocess.PIPE)
            self._current_color = banner_color

    def _write_proc(self, pcm: bytes) -> None:
        with self._proc_lock:
            proc = self._proc
            if proc.stdin is None:
                raise BrokenPipeError('ffmpeg stdin unavailable')
            proc.stdin.write(pcm)

    async def on_alert_start(self, identifier: str) -> None:
        if self._closed:
            return
        from module.video import (
            _get_active_alerts, _pick_severity_color, _to_ffmpeg_color, _coerce_mapping,
            _build_overlay_text, _format_overlay_display_text,
        )
        alerts = _get_active_alerts(self._feed_id)
        entry = next((a for a in alerts if a.get('identifier') == identifier), None) or {}
        meta = entry.get('metadata') or {}
        overlay_text = _format_overlay_display_text(
            _build_overlay_text(entry, self._tz_name),
            _coerce_mapping((self._video_cfg or {}).get('style', {})),
            self._frame_width,
        )
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
        previous_mode = self._mode
        self._mode = 'alert'
        if previous_mode != 'alert' or new_color != self._current_color:
            self._suspend_writes = True
            try:
                await asyncio.to_thread(self._rebuild_proc, new_color)
            finally:
                self._suspend_writes = False
        else:
            self._current_color = new_color
        log.info('[%s] %s: alert start — overlay updated (color=%s)', self._feed_id, self._label, new_color)

    async def on_alert_end(self) -> None:
        if self._closed:
            return
        previous_mode = self._mode
        if self._text_file:
            try:
                if self._idle_details_enabled:
                    self._mode = 'idle'
                    self._refresh_idle_text(force=True)
                else:
                    self._text_file.write_text(' ', encoding='utf-8')
            except Exception:
                pass
        from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
        self._mode = 'idle'
        idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
        if previous_mode != 'idle' and self._idle_factory is not None:
            self._suspend_writes = True
            try:
                await asyncio.to_thread(self._rebuild_proc, idle_color)
            finally:
                self._suspend_writes = False
        else:
            self._current_color = idle_color
        log.info('[%s] %s: alert end — overlay cleared', self._feed_id, self._label)

    async def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        if self._suspend_writes:
            return
        if self._mode == 'idle':
            try:
                self._refresh_idle_text()
            except Exception:
                pass
        try:
            await asyncio.to_thread(self._write_proc, pcm)
        except RuntimeError as exc:
            if 'cannot schedule new futures after shutdown' in str(exc).lower():
                return
            log.error('[%s] %s: write error: %s', self._feed_id, self._label, exc)
            self._closed = True
        except (BrokenPipeError, OSError, ValueError) as exc:
            if self._closed:
                return
            log.warning('[%s] %s: write failed (%s), restarting', self._feed_id, self._label, exc)
            try:
                await asyncio.to_thread(self._rebuild_proc, self._current_color)
            except Exception as restart_exc:
                log.error('[%s] %s: restart failed after write error: %s', self._feed_id, self._label, restart_exc)
                self._closed = True
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
    low_latency: bool = False,
    interlace: bool = False,
    output_pix_fmt: str | None = 'yuv420p',
) -> list[str]:
    """Build an ffmpeg command that accepts PCM audio on stdin and streams to output_url."""
    from module.video import _IDLE_BANNER_HEX, _postprocess_video_filter, _to_ffmpeg_color
    effective_color = banner_color or _to_ffmpeg_color(_IDLE_BANNER_HEX)
    has_video = bool(vcodec)

    video_inputs: list[str] = []
    filter_args: list[str] = []

    if has_video and video_cfg and text_file:
        video_inputs, filter_complex = _build_video_overlay_inputs(
            video_cfg, width, height, fps, text_file, effective_color, idle=idle,
        )
        filter_complex = _postprocess_video_filter(
            filter_complex,
            interlace=interlace,
            output_pix_fmt=output_pix_fmt,
        )
        filter_args = ['-filter_complex', f'{filter_complex}[vout]', '-map', '0:a', '-map', '[vout]']
    elif has_video:
        fps_str = str(fps)
        video_inputs = ['-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:r={fps_str}']
        filter_complex = _postprocess_video_filter(
            f'[1:v]fps={fps_str},scale={width}:{height}',
            interlace=interlace,
            output_pix_fmt=output_pix_fmt,
        )
        filter_args = ['-filter_complex', f'{filter_complex}[vout]', '-map', '0:a', '-map', '[vout]']

    a_codec_args: list[str]
    if acodec == 'pcm' or not acodec:
        a_codec_args = ['-c:a', 'pcm_s16le']
    else:
        a_codec_args = ['-c:a', acodec, '-b:a', f'{audio_bitrate_kbps}k']

    v_codec_args: list[str] = []
    if has_video:
        codec_tuning_args = _low_latency_video_args(vcodec, fps) if low_latency else []
        v_codec_args = [
            '-c:v', vcodec,
            *(['-pix_fmt', output_pix_fmt] if output_pix_fmt else []),
            *codec_tuning_args,
            '-b:v', f'{video_bitrate_kbps}k',
        ]

    output_args = list(extra_output_args or [])
    if low_latency:
        output_args = ['-flush_packets', '1', '-muxdelay', '0', '-muxpreload', '0', *output_args]

    return [
        'ffmpeg', '-loglevel', 'warning',
        '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
        '-i', 'pipe:0',
        *extra_audio_inputs,
        *video_inputs,
        *filter_args,
        *a_codec_args,
        *v_codec_args,
        *output_args,
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
    width, height, fps, interlace = _resolve_video_output_params(config, video_cfg)
    url = f'udp://{ip}:{port}?pkt_size=1316&buffer_size=65536'

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'udp') if vcodec and video_cfg else None

    def build(color: str) -> list[str]:
        return _build_video_stream_cmd(
            feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
            width, height, fps, text_file=tf, banner_color=color, low_latency=True, interlace=interlace,
        )
    idle_factory = None
    if tf and video_cfg:
        def build_idle_udp() -> list[str]:
            return _build_video_stream_cmd(
                feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
                width, height, fps, text_file=tf, banner_color=idle_color, idle=True, low_latency=True, interlace=interlace,
            )
        idle_factory = build_idle_udp
    return _VideoStreamSink(
        feed_id,
        build,
        tf,
        idle_color,
        f'UDP({ip}:{port})',
        video_cfg,
        tz_name=tz_name,
        queue_limit=_LOW_LATENCY_STREAM_QUEUE_LIMIT,
        drop_oldest=True,
        idle_factory=idle_factory,
        frame_width=width,
        clocked=True,
        prefill_chunks=_LOW_LATENCY_STREAM_PREFILL_CHUNKS,
    )


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
    width, height, fps, interlace = _resolve_video_output_params(config, video_cfg)
    url = f'rtp://{ip}:{port}?pkt_size=1316&buffer_size=65536'

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'rtp') if vcodec and video_cfg else None

    def build(color: str) -> list[str]:
        return _build_video_stream_cmd(
            feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
            width, height, fps, text_file=tf, banner_color=color, low_latency=True, interlace=interlace,
        )
    idle_factory = None
    if tf and video_cfg:
        def build_idle_rtp() -> list[str]:
            return _build_video_stream_cmd(
                feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
                width, height, fps, text_file=tf, banner_color=idle_color, idle=True, low_latency=True, interlace=interlace,
            )
        idle_factory = build_idle_rtp
    return _VideoStreamSink(
        feed_id,
        build,
        tf,
        idle_color,
        f'RTP({ip}:{port})',
        video_cfg,
        tz_name=tz_name,
        queue_limit=_LOW_LATENCY_STREAM_QUEUE_LIMIT,
        drop_oldest=True,
        idle_factory=idle_factory,
        frame_width=width,
        clocked=True,
        prefill_chunks=_LOW_LATENCY_STREAM_PREFILL_CHUNKS,
    )


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
    width, height, fps, interlace = _resolve_video_output_params(config, video_cfg)

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'rtmp') if vcodec and video_cfg else None

    def build(color: str) -> list[str]:
        return _build_video_stream_cmd(
            feed_id, audio_br, video_br, vcodec, acodec, 'flv', url, [], video_cfg,
            width, height, fps, text_file=tf, banner_color=color, interlace=interlace,
        )
    idle_factory = None
    if tf and video_cfg:
        def build_idle_rtmp() -> list[str]:
            return _build_video_stream_cmd(
                feed_id, audio_br, video_br, vcodec, acodec, 'flv', url, [], video_cfg,
                width, height, fps, text_file=tf, banner_color=idle_color, idle=True, interlace=interlace,
            )
        idle_factory = build_idle_rtmp
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'RTMP({url})', video_cfg, tz_name=tz_name, queue_limit=_STANDARD_STREAM_QUEUE_LIMIT, drop_oldest=True, idle_factory=idle_factory, frame_width=width)


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
    width, height, fps, interlace = _resolve_video_output_params(config, video_cfg)

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'srt') if vcodec and video_cfg else None

    def build(color: str) -> list[str]:
        return _build_video_stream_cmd(
            feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
            width, height, fps, text_file=tf, banner_color=color, interlace=interlace,
        )
    idle_factory = None
    if tf and video_cfg:
        def build_idle_srt() -> list[str]:
            return _build_video_stream_cmd(
                feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
                width, height, fps, text_file=tf, banner_color=idle_color, idle=True, interlace=interlace,
            )
        idle_factory = build_idle_srt
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'SRT({url})', video_cfg, tz_name=tz_name, queue_limit=_STANDARD_STREAM_QUEUE_LIMIT, drop_oldest=True, idle_factory=idle_factory, frame_width=width)


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
    width, height, fps, interlace = _resolve_video_output_params(config, video_cfg)

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'rtsp') if vcodec and video_cfg else None

    def build(color: str) -> list[str]:
        return _build_video_stream_cmd(
            feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
            width, height, fps, text_file=tf, banner_color=color,
            extra_output_args=['-rtsp_transport', 'tcp'], interlace=interlace,
        )
    idle_factory = None
    if tf and video_cfg:
        def build_idle_rtsp() -> list[str]:
            return _build_video_stream_cmd(
                feed_id, audio_br, video_br, vcodec, acodec, fmt, url, [], video_cfg,
                width, height, fps, text_file=tf, banner_color=idle_color, idle=True,
                extra_output_args=['-rtsp_transport', 'tcp'], interlace=interlace,
            )
        idle_factory = build_idle_rtsp
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'RTSP({url})', video_cfg, tz_name=tz_name, queue_limit=_STANDARD_STREAM_QUEUE_LIMIT, drop_oldest=True, idle_factory=idle_factory, frame_width=width)


def FramebufferSink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    path = str(config.get('path', '/dev/fb0'))
    from module.video import _postprocess_video_filter

    width, height, fps, interlace = _resolve_video_output_params(config, video_cfg)
    fps_str = str(fps)

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'fb') if video_cfg else None

    def build(color: str) -> list[str]:
        if video_cfg and tf:
            vi, fc = _build_video_overlay_inputs(video_cfg, width, height, fps, tf, color)
            filter_args = [
                '-filter_complex',
                f'{_postprocess_video_filter(fc, interlace=interlace, output_pix_fmt="bgr0")}[vout]',
                '-map', '[vout]',
            ]
        else:
            vi = ['-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:r={fps_str}']
            filter_args = [
                '-filter_complex',
                f'{_postprocess_video_filter(f"[1:v]fps={fps_str},scale={width}:{height}", interlace=interlace, output_pix_fmt="bgr0")}[vout]',
                '-map', '[vout]',
            ]
        return [
            'ffmpeg', '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            *vi, *filter_args,
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
                '-filter_complex', f'{_postprocess_video_filter(fc, interlace=interlace, output_pix_fmt="bgr0")}[vout]',
                '-map', '[vout]',
                '-r', fps_str,
                '-f', 'fbdev', path,
            ]
        idle_factory = build_idle_fb
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'Framebuffer({path})', video_cfg, queue_limit=_STANDARD_STREAM_QUEUE_LIMIT, drop_oldest=True, tz_name=tz_name, idle_factory=idle_factory, frame_width=width)


def DriSink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    path = str(config.get('path', '/dev/dri/card0'))
    from module.video import _postprocess_video_filter

    width, height, fps, interlace = _resolve_video_output_params(config, video_cfg)
    fps_str = str(fps)

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'dri') if video_cfg else None

    def build(color: str) -> list[str]:
        if video_cfg and tf:
            vi, fc = _build_video_overlay_inputs(video_cfg, width, height, fps, tf, color)
            filter_args = [
                '-filter_complex',
                f'{_postprocess_video_filter(fc, interlace=interlace, output_pix_fmt="yuv420p")}[vout]',
                '-map', '[vout]',
            ]
        else:
            vi = ['-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:r={fps_str}']
            filter_args = [
                '-filter_complex',
                f'{_postprocess_video_filter(f"[1:v]fps={fps_str},scale={width}:{height}", interlace=interlace, output_pix_fmt="yuv420p")}[vout]',
                '-map', '[vout]',
            ]
        return [
            'ffmpeg', '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            *vi, *filter_args,
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
                '-filter_complex', f'{_postprocess_video_filter(fc, interlace=interlace, output_pix_fmt="yuv420p")}[vout]',
                '-map', '[vout]',
                '-r', fps_str,
                '-f', 'drm_output',
                '-device', path,
                '-',
            ]
        idle_factory = build_idle_dri
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'DRI({path})', video_cfg, queue_limit=_STANDARD_STREAM_QUEUE_LIMIT, drop_oldest=True, tz_name=tz_name, idle_factory=idle_factory, frame_width=width)


def V4L2Sink(
    config: dict[str, Any],
    feed_id: str,
    video_cfg: dict[str, Any] | None = None,
    tz_name: str = 'UTC',
) -> _VideoStreamSink:
    device = str(config.get('device', '/dev/video0'))
    from module.video import _postprocess_video_filter

    width, height, fps, interlace = _resolve_video_output_params(config, video_cfg)
    fps_str = str(fps)

    from module.video import _to_ffmpeg_color, _IDLE_BANNER_HEX
    idle_color = _to_ffmpeg_color(_IDLE_BANNER_HEX)
    tf = _make_video_text_file(feed_id, 'v4l2') if video_cfg else None

    def build(color: str) -> list[str]:
        if video_cfg and tf:
            vi, fc = _build_video_overlay_inputs(video_cfg, width, height, fps, tf, color)
            filter_args = [
                '-filter_complex',
                f'{_postprocess_video_filter(fc, interlace=interlace, output_pix_fmt="yuv420p")}[vout]',
                '-map', '[vout]',
            ]
        else:
            vi = ['-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:r={fps_str}']
            filter_args = [
                '-filter_complex',
                f'{_postprocess_video_filter(f"[1:v]fps={fps_str},scale={width}:{height}", interlace=interlace, output_pix_fmt="yuv420p")}[vout]',
                '-map', '[vout]',
            ]
        return [
            'ffmpeg', '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            *vi, *filter_args,
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
                '-filter_complex', f'{_postprocess_video_filter(fc, interlace=interlace, output_pix_fmt="yuv420p")}[vout]',
                '-map', '[vout]',
                '-r', fps_str,
                '-f', 'v4l2', device,
            ]
        idle_factory = build_idle_v4l2
    return _VideoStreamSink(feed_id, build, tf, idle_color, f'V4L2({device})', video_cfg, queue_limit=_STANDARD_STREAM_QUEUE_LIMIT, drop_oldest=True, tz_name=tz_name, idle_factory=idle_factory, frame_width=width)


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

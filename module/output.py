from __future__ import annotations

import asyncio
import base64
import logging
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    import sounddevice as sd
except Exception:
    sd = None

from module.buffer import CHANNELS, SAMPLE_RATE
from module.events import NowPlayingMetadata

log = logging.getLogger(__name__)

_STANDARD_STREAM_QUEUE_LIMIT = 48
_LOW_LATENCY_STREAM_QUEUE_LIMIT = 16
_LOW_LATENCY_STREAM_PREFILL_CHUNKS = 2

_CODEC_MAP: dict[str, tuple[str, str, str]] = {
    'opus': ('libopus', 'audio/ogg', 'ogg'),
    'flac': ('flac', 'audio/flac', 'flac'),
    'ogg': ('libvorbis', 'audio/ogg', 'ogg'),
    'mp3': ('libmp3lame', 'audio/mpeg', 'mp3'),
    'aac': ('aac', 'audio/aac', 'adts'),
}

_STREAM_DYNAMICS = (
    'volume=0.65,'
    'highpass=f=110,'
    'equalizer=f=140:t=q:w=1.0:g=2.5,'
    'equalizer=f=220:t=q:w=1.0:g=2.0,'
    'equalizer=f=350:t=q:w=1.2:g=1.2,'
    'equalizer=f=500:t=q:w=1.4:g=-2.5,'
    'equalizer=f=900:t=q:w=1.6:g=-1.5,'
    'equalizer=f=2400:t=q:w=0.8:g=1.5,'
    'lowpass=f=3400,'
    'acompressor=threshold=-18dB:ratio=6:attack=8:release=140:makeup=2dB,'
    'alimiter=limit=0.80,'
)


def _audio_codec_args(acodec: str, bitrate_kbps: int) -> list[str]:
    normalized = str(acodec or '').strip()
    if normalized in {'', 'pcm', 'pcm_s16le'}:
        return ['-c:a', 'pcm_s16le']
    return ['-c:a', normalized, '-b:a', f'{bitrate_kbps}k']


def _metadata_args(stream_metadata: dict[str, Any] | None) -> list[str]:
    args: list[str] = []
    if not isinstance(stream_metadata, dict):
        return args
    for key, value in stream_metadata.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        args.extend(['-metadata', f'{key}={text}'])
    return args


def _build_stream_cmd(
    *,
    audio_bitrate_kbps: int,
    acodec: str,
    container: str,
    output_url: str,
    extra_output_args: list[str] | None = None,
    stream_metadata: dict[str, Any] | None = None,
    low_latency: bool = False,
) -> list[str]:
    output_args = list(extra_output_args or [])
    if low_latency:
        output_args = ['-flush_packets', '1', '-muxdelay', '0', '-muxpreload', '0', *output_args]

    return [
        'ffmpeg', '-loglevel', 'warning',
        '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
        '-i', 'pipe:0',
        '-af', _STREAM_DYNAMICS + 'alimiter=limit=0.50,',
        *_audio_codec_args(acodec, audio_bitrate_kbps),
        *_metadata_args(stream_metadata),
        *output_args,
        '-f', container,
        output_url,
    ]


class IcecastSink:
    bus_queue_limit = _STANDARD_STREAM_QUEUE_LIMIT
    bus_drop_oldest = True

    def __init__(self, config: dict[str, Any]) -> None:
        password = config.get('password', '') or ''
        username = config.get('username') or 'source'
        mount = config.get('mount') or f"/{config.get('feed_id', 'stream')}"
        self._host: str = str(config['host'])
        self._port: int = int(config['port'])
        self._mount: str = str(mount)
        self._username: str = str(username)
        self._password: str = str(password)
        self._ssl: bool = bool(config.get('ssl', False))
        self._stream_name: str = str(config.get('stream_name') or config.get('feed_id', 'stream')).strip()
        self._stream_description: str = str(config.get('stream_description') or '').strip()
        self._stream_genre: str = str(config.get('stream_genre') or 'Weather Radio').strip() or 'Weather Radio'
        self._stream_album: str = str(config.get('stream_album') or self._stream_name).strip() or self._stream_name
        self._stream_creator: str = str(config.get('stream_creator') or '').strip()
        self._stream_artist: str = str(config.get('stream_artist') or self._stream_name).strip() or self._stream_name

        url = f"icecast://{username}:{password}@{config['host']}:{config['port']}{mount}"
        fmt = str(config.get('format', 'opus')).strip().lower()
        codec, content_type, container = _CODEC_MAP.get(fmt, ('libopus', 'audio/ogg', 'ogg'))
        bitrate = int(config.get('bitrate_kbps', 32))

        self._cmd = [
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
            if self._proc.stdin is None or self._proc.poll() is not None:
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
                    'Icecast reconnect failed after %d attempts: %s - stream disabled',
                    self._consecutive_failures,
                    exc,
                )
                self._closed = True
            else:
                log.warning('Icecast reconnect failed: %s - will retry', exc)

    async def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        try:
            await asyncio.to_thread(self._write_proc, pcm)
            if self._proc.poll() is not None:
                raise BrokenPipeError('icecast ffmpeg exited after write')
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
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
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
            'song': title,
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
            f'{self._username}:{self._password}'.encode()
        ).decode()
        request = urllib.request.Request(
            url,
            headers={'Authorization': f'Basic {credentials}'},
        )

        for attempt in range(3):
            try:
                await asyncio.to_thread(urllib.request.urlopen, request, timeout=3)
                log.info('Icecast metadata updated: %s', title)
                return
            except urllib.error.HTTPError as exc:
                if exc.code == 404 and attempt < 2:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                log.warning('Icecast metadata update failed (%s): %s', url, exc)
                return
            except Exception as exc:
                log.warning('Icecast metadata update failed (%s): %s', url, exc)
                return


class _FfmpegStreamSink:
    bus_queue_limit = _STANDARD_STREAM_QUEUE_LIMIT
    bus_drop_oldest = False
    bus_clocked = False
    bus_prefill_chunks = 0
    bus_fill_silence = False

    def __init__(
        self,
        cmd: list[str],
        label: str,
        *,
        queue_limit: int = _STANDARD_STREAM_QUEUE_LIMIT,
        drop_oldest: bool = False,
        clocked: bool = False,
        prefill_chunks: int = 0,
        fill_silence: bool = False,
    ) -> None:
        self._cmd = list(cmd)
        self._label = label
        self._closed = False
        self.bus_queue_limit = max(1, int(queue_limit))
        self.bus_drop_oldest = bool(drop_oldest)
        self.bus_clocked = bool(clocked)
        self.bus_prefill_chunks = max(0, int(prefill_chunks))
        self.bus_fill_silence = bool(fill_silence)
        self._proc_lock = threading.Lock()
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE)
        log.info('%s started', self._label)

    def _restart_proc(self) -> None:
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
                    old.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    old.kill()
                    old.wait(timeout=1.0)
            self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE)

    def _write_proc(self, pcm: bytes) -> None:
        with self._proc_lock:
            proc = self._proc
            if proc.stdin is None or proc.poll() is not None:
                raise BrokenPipeError('ffmpeg stdin unavailable')
            proc.stdin.write(pcm)

    async def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        try:
            await asyncio.to_thread(self._write_proc, pcm)
        except RuntimeError as exc:
            if 'cannot schedule new futures after shutdown' in str(exc).lower():
                return
            log.error('%s write error: %s', self._label, exc)
            self._closed = True
        except (BrokenPipeError, OSError, ValueError) as exc:
            if self._closed:
                return
            log.warning('%s write failed (%s), restarting', self._label, exc)
            try:
                await asyncio.to_thread(self._restart_proc)
            except Exception as restart_exc:
                log.error('%s restart failed after write error: %s', self._label, restart_exc)
                self._closed = True
        except Exception as exc:
            log.error('%s write error: %s', self._label, exc)
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
        log.info('%s closed', self._label)


def _make_stream_sink(
    *,
    sink_label: str,
    audio_bitrate_kbps: int,
    acodec: str,
    container: str,
    url: str,
    queue_limit: int,
    low_latency: bool = False,
    clocked: bool = False,
    prefill_chunks: int = 0,
    fill_silence: bool = False,
    extra_output_args: list[str] | None = None,
    stream_metadata: dict[str, Any] | None = None,
) -> _FfmpegStreamSink:
    return _FfmpegStreamSink(
        _build_stream_cmd(
            audio_bitrate_kbps=audio_bitrate_kbps,
            acodec=acodec,
            container=container,
            output_url=url,
            extra_output_args=extra_output_args,
            stream_metadata=stream_metadata,
            low_latency=low_latency,
        ),
        sink_label,
        queue_limit=queue_limit,
        drop_oldest=False,
        clocked=clocked,
        prefill_chunks=prefill_chunks,
        fill_silence=fill_silence,
    )


def UdpSink(config: dict[str, Any], feed_id: str) -> _FfmpegStreamSink:
    ip = config.get('ip', '127.0.0.1')
    port = int(config.get('port', 8899))
    return _make_stream_sink(
        sink_label=f'UDP({ip}:{port})',
        audio_bitrate_kbps=int(config.get('bitrate_kbps', 32)),
        acodec=str(config.get('acodec') or 'aac'),
        container=str(config.get('format', 'mpegts')),
        url=f'udp://{ip}:{port}?pkt_size=1316&buffer_size=65536',
        queue_limit=_LOW_LATENCY_STREAM_QUEUE_LIMIT,
        low_latency=True,
        clocked=True,
        prefill_chunks=_LOW_LATENCY_STREAM_PREFILL_CHUNKS,
        fill_silence=True,
        stream_metadata=config.get('stream_metadata') if isinstance(config.get('stream_metadata'), dict) else None,
    )


def RtpSink(config: dict[str, Any], feed_id: str) -> _FfmpegStreamSink:
    ip = config.get('ip', '127.0.0.1')
    port = int(config.get('port', 8899))
    return _make_stream_sink(
        sink_label=f'RTP({ip}:{port})',
        audio_bitrate_kbps=int(config.get('bitrate_kbps', 32)),
        acodec=str(config.get('acodec') or 'aac'),
        container=str(config.get('format', 'rtp_mpegts')),
        url=f'rtp://{ip}:{port}?pkt_size=1316&buffer_size=65536',
        queue_limit=_LOW_LATENCY_STREAM_QUEUE_LIMIT,
        low_latency=True,
        clocked=True,
        prefill_chunks=_LOW_LATENCY_STREAM_PREFILL_CHUNKS,
        fill_silence=True,
        stream_metadata=config.get('stream_metadata') if isinstance(config.get('stream_metadata'), dict) else None,
    )


def RtmpSink(config: dict[str, Any], feed_id: str) -> _FfmpegStreamSink:
    return _make_stream_sink(
        sink_label=f'RTMP({config.get("url", "")})',
        audio_bitrate_kbps=int(config.get('bitrate_kbps', 32)),
        acodec=str(config.get('acodec') or 'aac'),
        container='flv',
        url=str(config.get('url', 'rtmp://localhost/live/stream')),
        queue_limit=_STANDARD_STREAM_QUEUE_LIMIT,
        stream_metadata=config.get('stream_metadata') if isinstance(config.get('stream_metadata'), dict) else None,
    )


def SrtSink(config: dict[str, Any], feed_id: str) -> _FfmpegStreamSink:
    return _make_stream_sink(
        sink_label=f'SRT({config.get("url", "")})',
        audio_bitrate_kbps=int(config.get('bitrate_kbps', 32)),
        acodec=str(config.get('acodec') or 'aac'),
        container=str(config.get('format', 'mpegts')),
        url=str(config.get('url', 'srt://localhost:12345')),
        queue_limit=_STANDARD_STREAM_QUEUE_LIMIT,
        stream_metadata=config.get('stream_metadata') if isinstance(config.get('stream_metadata'), dict) else None,
    )


def RtspSink(config: dict[str, Any], feed_id: str) -> _FfmpegStreamSink:
    return _make_stream_sink(
        sink_label=f'RTSP({config.get("url", "")})',
        audio_bitrate_kbps=int(config.get('bitrate_kbps', 32)),
        acodec=str(config.get('acodec') or 'aac'),
        container=str(config.get('format', 'rtsp')),
        url=str(config.get('url', 'rtsp://localhost:8554/stream')),
        queue_limit=_STANDARD_STREAM_QUEUE_LIMIT,
        extra_output_args=['-rtsp_transport', 'tcp'],
        stream_metadata=config.get('stream_metadata') if isinstance(config.get('stream_metadata'), dict) else None,
    )


class AudioDeviceSink:
    bus_queue_limit = 20
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
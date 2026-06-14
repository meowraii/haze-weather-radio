from __future__ import annotations

import asyncio
import base64
import logging
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

try:
    import sounddevice as sd
except Exception:
    sd = None

from module.audio import PyAVStreamWriter
from module.buffer import CHANNELS, CHUNK_BYTES, CHUNK_SAMPLES, SAMPLE_RATE
from module.events import NowPlayingMetadata

log = logging.getLogger(__name__)

_STANDARD_STREAM_QUEUE_LIMIT = 48
_LOW_LATENCY_STREAM_QUEUE_LIMIT = 16
_LOW_LATENCY_STREAM_PREFILL_CHUNKS = 2

_CODEC_MAP: dict[str, tuple[str, str, str]] = {
    'opus': ('opus', 'audio/ogg', 'ogg'),
    'flac': ('flac', 'audio/flac', 'flac'),
    'ogg': ('opus', 'audio/ogg', 'ogg'),
    'mp3': ('libmp3lame', 'audio/mpeg', 'mp3'),
    'aac': ('aac', 'audio/aac', 'adts'),
}


def _normalize_codec(acodec: str, container: str) -> str:
    normalized = str(acodec or '').strip()
    if normalized in {'', 'pcm', 'pcm_s16le'}:
        return 'pcm_s16le'
    aliases = {
        'libopus': 'opus',
        'libvorbis': 'opus' if container == 'ogg' else 'vorbis',
        'vorbis': 'opus' if container == 'ogg' else 'vorbis',
        'mp3': 'libmp3lame',
    }
    return aliases.get(normalized, normalized)


def _bit_rate(codec: str, bitrate_kbps: int) -> int | None:
    if codec in {'pcm_s16le', 'flac'}:
        return None
    return max(1, int(bitrate_kbps)) * 1000


def _clean_metadata(stream_metadata: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(stream_metadata, dict):
        return {}
    return {
        str(key): str(value)
        for key, value in stream_metadata.items()
        if value is not None and str(value).strip()
    }


def _stream_options(
    *,
    low_latency: bool = False,
    ssl: bool = False,
    content_type: str | None = None,
    ice_name: str | None = None,
    ice_description: str | None = None,
    ice_genre: str | None = None,
    rtsp_transport: str | None = None,
) -> dict[str, str]:
    options: dict[str, str] = {}
    if low_latency:
        options.update({'flush_packets': '1', 'muxdelay': '0', 'muxpreload': '0'})
    if ssl:
        options['tls'] = '1'
    if content_type:
        options['content_type'] = content_type
    if ice_name:
        options['ice_name'] = ice_name
    if ice_description:
        options['ice_description'] = ice_description
    if ice_genre:
        options['ice_genre'] = ice_genre
    if rtsp_transport:
        options['rtsp_transport'] = rtsp_transport
    return options


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
        codec, content_type, container = _CODEC_MAP.get(fmt, ('opus', 'audio/ogg', 'ogg'))
        bitrate = int(config.get('bitrate_kbps', 32))

        codec = _normalize_codec(codec, container)
        metadata = {
            'artist': self._stream_artist,
            'album': self._stream_album,
            'creator': self._stream_creator,
            'genre': self._stream_genre,
            'title': self._stream_name,
        }

        self._proc_lock = threading.Lock()
        self._writer = PyAVStreamWriter(
            url=url,
            container_format=container,
            codec=codec,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            bit_rate=_bit_rate(codec, bitrate),
            metadata=metadata,
            options=_stream_options(
                ssl=self._ssl,
                content_type=content_type,
                ice_name=self._stream_name,
                ice_description=self._stream_description,
                ice_genre=self._stream_genre,
            ),
        )
        self._writer.open()
        self._closed = False
        self._reconnect_delay = 2.0
        self._max_reconnect_delay = 60.0
        self._consecutive_failures = 0

    def _restart_proc(self) -> None:
        with self._proc_lock:
            self._writer.restart()

    def _write_proc(self, pcm: bytes) -> None:
        with self._proc_lock:
            self._writer.write(pcm)

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
        await asyncio.to_thread(self._writer.close)

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


class _PyAVStreamSink:
    bus_queue_limit = _STANDARD_STREAM_QUEUE_LIMIT
    bus_drop_oldest = False
    bus_clocked = False
    bus_prefill_chunks = 0
    bus_fill_silence = False

    def __init__(
        self,
        writer: PyAVStreamWriter,
        label: str,
        *,
        queue_limit: int = _STANDARD_STREAM_QUEUE_LIMIT,
        drop_oldest: bool = False,
        clocked: bool = False,
        prefill_chunks: int = 0,
        fill_silence: bool = False,
    ) -> None:
        self._writer = writer
        self._label = label
        self._closed = False
        self.bus_queue_limit = max(1, int(queue_limit))
        self.bus_drop_oldest = bool(drop_oldest)
        self.bus_clocked = bool(clocked)
        self.bus_prefill_chunks = max(0, int(prefill_chunks))
        self.bus_fill_silence = bool(fill_silence)
        self._proc_lock = threading.Lock()
        self._writer.open()
        log.info('%s started', self._label)

    def _restart_proc(self) -> None:
        with self._proc_lock:
            self._writer.restart()

    def _write_proc(self, pcm: bytes) -> None:
        with self._proc_lock:
            self._writer.write(pcm)

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
        await asyncio.to_thread(self._writer.close)
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
) -> _PyAVStreamSink:
    if extra_output_args:
        log.warning('%s ignores legacy CLI-style extra_output_args under PyAV: %s', sink_label, extra_output_args)
    codec = _normalize_codec(acodec, container)
    options = _stream_options(
        low_latency=low_latency,
        rtsp_transport='tcp' if container == 'rtsp' else None,
    )
    return _PyAVStreamSink(
        PyAVStreamWriter(
            url=url,
            container_format=container,
            codec=codec,
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
            bit_rate=_bit_rate(codec, audio_bitrate_kbps),
            metadata=_clean_metadata(stream_metadata),
            options=options,
        ),
        sink_label,
        queue_limit=queue_limit,
        drop_oldest=False,
        clocked=clocked,
        prefill_chunks=prefill_chunks,
        fill_silence=fill_silence,
    )


def UdpSink(config: dict[str, Any], feed_id: str) -> _PyAVStreamSink:
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


def RtpSink(config: dict[str, Any], feed_id: str) -> _PyAVStreamSink:
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


def RtmpSink(config: dict[str, Any], feed_id: str) -> _PyAVStreamSink:
    return _make_stream_sink(
        sink_label=f'RTMP({config.get("url", "")})',
        audio_bitrate_kbps=int(config.get('bitrate_kbps', 32)),
        acodec=str(config.get('acodec') or 'aac'),
        container='flv',
        url=str(config.get('url', 'rtmp://localhost/live/stream')),
        queue_limit=_STANDARD_STREAM_QUEUE_LIMIT,
        stream_metadata=config.get('stream_metadata') if isinstance(config.get('stream_metadata'), dict) else None,
    )


def SrtSink(config: dict[str, Any], feed_id: str) -> _PyAVStreamSink:
    return _make_stream_sink(
        sink_label=f'SRT({config.get("url", "")})',
        audio_bitrate_kbps=int(config.get('bitrate_kbps', 32)),
        acodec=str(config.get('acodec') or 'aac'),
        container=str(config.get('format', 'mpegts')),
        url=str(config.get('url', 'srt://localhost:12345')),
        queue_limit=_STANDARD_STREAM_QUEUE_LIMIT,
        stream_metadata=config.get('stream_metadata') if isinstance(config.get('stream_metadata'), dict) else None,
    )


def RtspSink(config: dict[str, Any], feed_id: str) -> _PyAVStreamSink:
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
    bus_queue_limit = 6
    bus_drop_oldest = True

    def __init__(self, config: dict[str, Any] | str | int | None = None) -> None:
        if sd is None:
            raise RuntimeError('sounddevice is unavailable; install PortAudio and the sounddevice package to use the audio device sink')
        cfg = config if isinstance(config, dict) else {'device': config}
        device = _resolve_audio_device(cfg.get('device', cfg.get('name')))
        self._label = _audio_device_label(device)
        self._blocksize = max(128, int(cfg.get('blocksize') or CHUNK_SAMPLES))
        self._max_buffer_bytes = max(
            CHUNK_BYTES * 2,
            CHUNK_BYTES * int(cfg.get('buffer_chunks') or 8),
        )
        self._buffer = bytearray()
        self._lock = threading.Lock()
        self._closed = False
        self._underruns = 0
        self._overruns = 0
        self._callback_errors = 0
        self._stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='int16',
            device=device,
            blocksize=self._blocksize,
            latency=cfg.get('latency', 'low'),
            callback=self._callback,
        )
        self._stream.start()
        log.info(
            'Audio device sink started: %s (%d Hz, %d channel%s, blocksize=%d)',
            self._label,
            SAMPLE_RATE,
            CHANNELS,
            '' if CHANNELS == 1 else 's',
            self._blocksize,
        )

    def _callback(self, outdata: Any, frames: int, time_info: Any, status: Any) -> None:
        needed = int(frames) * CHANNELS * 2
        chunk = bytes(needed)
        underrun_happened = False
        try:
            if status:
                self._callback_errors += 1
                if self._callback_errors in {1, 8, 32} or self._callback_errors % 128 == 0:
                    log.warning('Audio device status on %s: %s', self._label, status)
            with self._lock:
                available = min(len(self._buffer), needed)
                if available:
                    chunk = bytes(self._buffer[:available])
                    del self._buffer[:available]
                    if available < needed:
                        self._underruns += 1
                        underrun_happened = True
                        chunk += bytes(needed - available)
                else:
                    self._underruns += 1
                    underrun_happened = True
            if underrun_happened and (self._underruns in {1, 8, 32} or self._underruns % 128 == 0):
                log.debug('Audio device underrun on %s: %d', self._label, self._underruns)
            outdata[:] = chunk
        except Exception:
            self._callback_errors += 1
            outdata[:] = bytes(needed)
            if self._callback_errors in {1, 8, 32} or self._callback_errors % 128 == 0:
                log.exception('Audio device callback failed on %s', self._label)

    async def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        with self._lock:
            self._buffer.extend(pcm)
            overflow = len(self._buffer) - self._max_buffer_bytes
            if overflow > 0:
                frame_width = CHANNELS * 2
                drop = overflow
                remainder = drop % frame_width
                if remainder:
                    drop += frame_width - remainder
                del self._buffer[:drop]
                self._overruns += 1
                if self._overruns in {1, 8, 32} or self._overruns % 128 == 0:
                    log.warning(
                        'Audio device buffer overrun on %s; dropped %.1f ms (%d total)',
                        self._label,
                        (drop / (SAMPLE_RATE * CHANNELS * 2)) * 1000.0,
                        self._overruns,
                    )

    def drop_pending(self) -> int:
        with self._lock:
            dropped = len(self._buffer)
            self._buffer.clear()
        return dropped // max(CHUNK_BYTES, 1)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        with self._lock:
            self._buffer.clear()
        await asyncio.to_thread(self._stream.stop)
        await asyncio.to_thread(self._stream.close)
        log.info('Audio device sink closed: %s', self._label)


def _resolve_audio_device(value: Any) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if sd is None:
        return text
    try:
        devices = sd.query_devices()
    except Exception:
        return text
    lowered = text.lower()
    for idx, device in enumerate(devices):
        name = str(device.get('name', ''))
        if name.lower() == lowered and int(device.get('max_output_channels') or 0) >= CHANNELS:
            return idx
    for idx, device in enumerate(devices):
        name = str(device.get('name', ''))
        if lowered in name.lower() and int(device.get('max_output_channels') or 0) >= CHANNELS:
            return idx
    return text


def _audio_device_label(device: str | int | None) -> str:
    if sd is None:
        return str(device or 'default')
    try:
        info = sd.query_devices(device, 'output')
        name = str(info.get('name') or device or 'default')
        return f'{device}: {name}' if device is not None else name
    except Exception:
        return str(device or 'default')


class FileSink:
    def __init__(self, path: str) -> None:
        self._writer = PyAVStreamWriter(
            url=path,
            container_format='wav',
            codec='pcm_s16le',
            sample_rate=SAMPLE_RATE,
            channels=CHANNELS,
        )
        self._writer.open()

    async def write(self, pcm: bytes) -> None:
        if pcm:
            await asyncio.to_thread(self._writer.write, pcm)

    async def close(self) -> None:
        await asyncio.to_thread(self._writer.close)

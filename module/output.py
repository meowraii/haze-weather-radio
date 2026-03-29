from __future__ import annotations

import asyncio
import base64
import logging
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from module.queue import CHANNELS, SAMPLE_RATE

import sounddevice as sd


log = logging.getLogger(__name__)

_CODEC_MAP: dict[str, tuple[str, str, str]] = {
    'opus': ('libopus',   'audio/ogg',   'ogg'),
    'ogg':  ('libvorbis', 'audio/ogg',   'ogg'),
    'mp3':  ('libmp3lame','audio/mpeg',  'mp3'),
    'aac':  ('aac',       'audio/aac',   'adts'),
}

_STREAM_DYNAMICS = (
    'acompressor=threshold=-50dB:ratio=30:attack=0.1:release=50:makeup=46dB,'
    'loudnorm=i=-6.0:lra=1.0:tp=-0.1,'
    'alimiter=limit=1.0:attack=0.1:release=10:level=1'
)


class IcecastSink:
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
        url = (
            f"icecast://{username}:{password}@"
            f"{config['host']}:{config['port']}{mount}"
        )
        fmt = config.get('format', 'opus')
        codec, content_type, container = _CODEC_MAP.get(fmt, ('libopus', 'audio/ogg', 'ogg'))
        bitrate = config.get('bitrate_kbps', 96)
        self._cmd = [
            'ffmpeg', '-loglevel', 'error', '-re',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            '-af', _STREAM_DYNAMICS,
            '-c:a', codec, '-b:a', f'{bitrate}k',
            '-content_type', content_type,
            '-f', container,
            url,
        ]
        self._proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE)
        self._closed = False

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
        if self._closed or self._proc.stdin is None:
            return
        try:
            await asyncio.to_thread(self._proc.stdin.write, pcm)
        except BrokenPipeError:
            log.warning('Icecast stream pipe broken, reconnecting in 5s')
            await asyncio.sleep(5.0)
            try:
                await asyncio.to_thread(self._restart_proc)
                log.info('Icecast stream reconnected to %s', self._mount)
            except Exception as e:
                log.error('Icecast reconnect failed: %s — stream disabled', e)
                self._closed = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._proc.stdin:
            self._proc.stdin.close()
        await asyncio.to_thread(self._proc.wait)

    async def set_metadata(self, title: str) -> None:
        if self._closed:
            return
        scheme = 'https' if self._ssl else 'http'
        song_encoded = urllib.parse.quote(title, safe='')
        url = f"{scheme}://{self._host}:{self._port}/admin/metadata?mount={self._mount}&mode=updinfo&song={song_encoded}"
        credentials = base64.b64encode(f"{self._username}:{self._password}".encode()).decode()
        req = urllib.request.Request(url, headers={'Authorization': f'Basic {credentials}'})
        for attempt in range(5):
            try:
                await asyncio.to_thread(urllib.request.urlopen, req, timeout=3)
                log.info('Icecast metadata updated: %s', title)
                return
            except urllib.error.HTTPError as e:
                if e.code == 404 and attempt < 4:
                    await asyncio.sleep(1.5)
                    continue
                log.warning('Icecast metadata update failed (%s): %s', url, e)
                return
            except Exception as e:
                log.warning('Icecast metadata update failed (%s): %s', url, e)
                return


class AudioDeviceSink:
    def __init__(self, device: str | int | None = None) -> None:
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


class PiFmAdvSink:
    def __init__(self, config: dict[str, Any]) -> None:
        freq = config['frequency_hz']
        dev = config.get('deviation_hz', 5000)
        bw = config.get('bandwidth_hz', 12500)
        host = config.get('host')
        bin_root = config.get('bin_root', '/home/pi/PiFmAdv/src')
        pi_fm_adv_path = f"{bin_root}/pi_fm_adv"

        lpf = min(bw // 2 - 100, 7500)

        audio_filter = (
            f'highpass=f=100,'
            f'acompressor=threshold=-24dB:ratio=8:attack=5:release=200:makeup=4dB,'
            f'loudnorm=I=-23:LRA=7:TP=-3,'
            f'lowpass=f={lpf},'
            f'alimiter=limit=0.70:attack=0.5:release=5'
        )

        ffmpeg_cmd = [
            'ffmpeg', '-loglevel', 'error',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            '-af', audio_filter,
            '-ac', '1', '-ar', str(out_sr),
            '-f', 'wav', 'pipe:1',
        ]

        fm_args = [
            'sudo', pi_fm_adv_path,
            '--audio', '-',
            '--freq', freq,
            '--dev', str(dev),
            '--preemph', '75us',
            '--rds', '0',
        ]

        if host:
            remote_cmd = ' '.join(fm_args)
            fm_cmd = ['ssh', '-o', 'StrictHostKeyChecking=accept-new',
                      '-o', 'ServerAliveInterval=15', host, remote_cmd]
        else:
            fm_cmd = fm_args

        self._ffmpeg = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        self._fm = subprocess.Popen(
            fm_cmd,
            stdin=self._ffmpeg.stdout,
        )
        self._ffmpeg.stdout.close()
        self._closed = False
        log.info('pi_fm_adv sink started: %.4f MHz dev=±%d Hz', freq / 1_000_000.0, dev)

    async def write(self, pcm: bytes) -> None:
        if self._closed or self._ffmpeg.stdin is None:
            return
        try:
            await asyncio.to_thread(self._ffmpeg.stdin.write, pcm)
        except BrokenPipeError:
            log.warning('pi_fm_adv pipe broken')
            self._closed = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ffmpeg.stdin:
            self._ffmpeg.stdin.close()
        await asyncio.to_thread(self._ffmpeg.wait)
        await asyncio.to_thread(self._fm.wait)

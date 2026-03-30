from __future__ import annotations

import asyncio
import base64
import logging
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import sounddevice as sd

from module.queue import CHANNELS, SAMPLE_RATE

log = logging.getLogger(__name__)

_CODEC_MAP: dict[str, tuple[str, str, str]] = {
    'opus': ('libopus',    'audio/ogg',  'ogg'),
    'ogg':  ('libvorbis',  'audio/ogg',  'ogg'),
    'mp3':  ('libmp3lame', 'audio/mpeg', 'mp3'),
    'aac':  ('aac',        'audio/aac',  'adts'),
}

_STREAM_DYNAMICS = (
    'loudnorm=I=-12:LRA=6:TP=-1,'
    'alimiter=limit=0.96'
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

        self._cmd: list[str] = [
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
        url = (
            f"{scheme}://{self._host}:{self._port}/admin/metadata"
            f"?mount={self._mount}&mode=updinfo&song={song_encoded}"
        )

        credentials = base64.b64encode(
            f"{self._username}:{self._password}".encode()
        ).decode()

        req = urllib.request.Request(
            url, headers={'Authorization': f'Basic {credentials}'}
        )

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
        freq_mhz: str = str(config['frequency_mhz'])
        dev: int = config.get('deviation_hz', 5000)
        bw: int = config.get('bandwidth_hz', 12000)

        bin_root: str = config.get('bin_root', '/home/pi/PiFmAdv/src')
        pi_fm_adv_bin: str = f"{bin_root}/pi_fm_adv"

        alt_freqs: list = config.get('alternative_frequencies', [])

        ssh_cfg: dict = config.get('ssh', {})
        use_ssh: bool = ssh_cfg.get('enabled', False)

        ssh_bin: str = ssh_cfg.get('ssh_bin', 'ssh')
        ssh_host: str = ssh_cfg.get('host', '')
        ssh_key: str = ssh_cfg.get('public_key_path', '~/.ssh/id_rsa')
        ssh_user: str = ssh_cfg.get('username', 'pi')

        use_sudo: bool = config.get('use_sudo', True)

        lpf: int = min(bw // 2 - 100, 7500)

        fm_args: list[str] = []
        if use_sudo:
            fm_args += ['sudo', '-n']

        fm_args += [
            pi_fm_adv_bin,
            '--audio', '-',
            '--freq', freq_mhz,
            '--dev', str(dev),
            '--cutoff', str(lpf),
            '--preemph', '75us',
            '--rds', '0',
        ]

        if alt_freqs:
            fm_args += ['--af', ','.join(str(f) for f in alt_freqs)]

        ffmpeg_local_cmd: list[str] = [
            'ffmpeg', '-loglevel', 'error',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            '-af', _STREAM_DYNAMICS,
            '-ac', '1', '-ar', '11025',
            '-f', 'wav', 'pipe:1',
        ]

        self._stderr_task: asyncio.Task | None = None

        if use_ssh:
            remote_cmd = f"{' '.join(fm_args)}"

            self._fm = subprocess.Popen(
                [
                    ssh_bin,
                    '-T',
                    '-i', ssh_key,
                    '-l', ssh_user,
                    '-o', 'StrictHostKeyChecking=accept-new',
                    '-o', 'ServerAliveInterval=15',
                    '-o', 'ServerAliveCountMax=3',
                    '-o', 'Compression=no',
                    ssh_host,
                    remote_cmd,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            self._ffmpeg = subprocess.Popen(
                ffmpeg_local_cmd,
                stdin=subprocess.PIPE,
                stdout=self._fm.stdin,
                stderr=subprocess.DEVNULL,
            )

            self._fm.stdin.close()

        else:
            self._ffmpeg = subprocess.Popen(
                ffmpeg_local_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )

            self._fm = subprocess.Popen(
                fm_args,
                stdin=self._ffmpeg.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )

            self._ffmpeg.stdout.close()

        self._closed = False

        log.info(
            'PiFmAdv sink started: %s MHz dev=±%d Hz lpf=%d Hz ssh=%s',
            freq_mhz, dev, lpf, ssh_host if use_ssh else 'disabled',
        )

    async def write(self, pcm: bytes) -> None:
        if self._closed or self._ffmpeg.stdin is None:
            return
        try:
            await asyncio.to_thread(self._ffmpeg.stdin.write, pcm)
        except BrokenPipeError:
            log.warning('PiFmAdv: pipe broken')
            self._closed = True

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._ffmpeg.stdin:
            try:
                self._ffmpeg.stdin.close()
            except OSError:
                pass

        await asyncio.to_thread(self._ffmpeg.wait)
        await asyncio.to_thread(self._fm.wait)

        log.info('PiFmAdv sink closed')
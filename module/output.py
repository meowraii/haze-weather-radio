from __future__ import annotations

import asyncio
import base64
import logging
import os
import queue as _stdlib_queue
import subprocess
import threading
import time
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
    'volume=1.2,'
    'alimiter=limit=0.98:level=0'
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
            'ffmpeg', '-loglevel', 'error',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            '-af', _STREAM_DYNAMICS,
            '-c:a', codec, '-b:a', f'{bitrate}k',
            '-ar', str(SAMPLE_RATE),
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
        if self._closed or self._proc.stdin is None or not pcm:
            return
        try:
            await asyncio.to_thread(self._proc.stdin.write, pcm)
        except BrokenPipeError:
            log.warning('Icecast: pipe broken, reconnecting in 5s')
            await asyncio.sleep(5.0)
            try:
                await asyncio.to_thread(self._restart_proc)
                log.info('Icecast reconnected to %s', self._mount)
            except Exception as e:
                log.error('Icecast reconnect failed: %s — stream disabled', e)
                self._closed = True
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
        self._stream = sd.binOutputStream(
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


_PIFM_RECONNECT_DELAYS = (2.0, 5.0, 10.0, 30.0)
_PIFM_QUEUE_LIMIT = 128

_RADIO_DYNAMICS = (
    'acompressor=threshold=-18dB:ratio=10:attack=20:release=200:makeup=16dB,'
    'equalizer=f=800:t=q:w=1.2:g=2,'
    'equalizer=f=1500:t=q:w=1.2:g=6,'
    'highpass=f=200,'
    'lowpass=f=3000,'
    'alimiter=limit=0.99:level=0'
)


class PiFmAdvSink:

    def __init__(self, config: dict[str, Any]) -> None:
        freq_mhz: str = str(config['frequency_mhz'])
        dev: int = config.get('deviation_hz', 5000)
        bw: int = config.get('bandwidth_hz', 10000)
        bin_root: str = config.get('bin_root', '/home/pi/PiFmAdv/src')
        pi_fm_adv_bin: str = f"{bin_root}/pi_fm_adv"
        alt_freqs: list = config.get('alternative_frequencies', [])
        use_sudo: bool = config.get('use_sudo', True)
        lpf: int = min(bw // 2 - 100, 7500)
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
            '--cutoff', str(lpf),
            '--power', str(tx_power),
            '--preemph', '75us',
            '--rds', '0',
        ]
        if alt_freqs:
            fm_args += ['--af', ','.join(str(f) for f in alt_freqs)]

        self._ffmpeg_cmd: list[str] = [
            ffmpeg_bin, '-loglevel', 'warning',
            '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
            '-i', 'pipe:0',
            '-ac', '1',
            '-ar', '6000',
            '-af', _RADIO_DYNAMICS,
            '-flush_packets', '1',
            '-f', 'wav',
            'pipe:1',
        ]

        if self._use_ssh:
            self._fm_cmd: list[str] = [
                ssh_bin,
                '-T',
                '-i', ssh_key,
                '-p', str(ssh_port),
                '-l', ssh_user,
                '-o', 'BatchMode=yes',
                '-o', 'StrictHostKeyChecking=accept-new',
                '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=3',
                '-o', 'Compression=no',
                ssh_host,
                ' '.join(fm_args),
            ]
        else:
            self._fm_cmd = fm_args

        self._ffmpeg: subprocess.Popen | None = None
        self._fm: subprocess.Popen | None = None
        self._closed = False
        self._queue: _stdlib_queue.Queue[bytes | None] = _stdlib_queue.Queue(
            maxsize=_PIFM_QUEUE_LIMIT,
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f'pifmadv:{freq_mhz}',
            daemon=True,
        )

        self._start_pipeline()

        self._label = f"{freq_mhz} MHz dev=±{dev} Hz lpf={lpf} Hz sr={SAMPLE_RATE}"
        self._writer_thread.start()
        ssh_label = ssh_host if self._use_ssh else 'disabled'
        log.info('PiFmAdv sink started: %s ssh=%s', self._label, ssh_label)

    def _start_pipeline(self) -> None:
        if self._use_ssh:
            self._fm = subprocess.Popen(
                self._fm_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._ffmpeg = subprocess.Popen(
                self._ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=self._fm.stdin,
                stderr=subprocess.DEVNULL,
            )
            if self._fm.stdin is not None:
                self._fm.stdin.close()
        else:
            self._ffmpeg = subprocess.Popen(
                self._ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            self._fm = subprocess.Popen(
                self._fm_cmd,
                stdin=self._ffmpeg.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if self._ffmpeg.stdout is not None:
                self._ffmpeg.stdout.close()
        time.sleep(0.4)
        ff_rc = self._ffmpeg.poll() if self._ffmpeg else -1
        fm_rc = self._fm.poll() if self._fm else -1
        if ff_rc is not None:
            raise RuntimeError(f'ffmpeg exited immediately (rc={ff_rc})')
        if fm_rc is not None:
            raise RuntimeError(f'fm/ssh process exited immediately (rc={fm_rc})')

    def _pipeline_alive(self) -> bool:
        return (
            self._ffmpeg is not None and self._ffmpeg.poll() is None
            and self._fm is not None and self._fm.poll() is None
        )

    def _drain_queue(self) -> None:
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
                drained += 1
            except _stdlib_queue.Empty:
                break
        if drained:
            log.debug('PiFmAdv: drained %d stale chunks', drained)

    def _reconnect_with_drain(self) -> None:
        attempt = 0
        while not self._closed:
            delay = _PIFM_RECONNECT_DELAYS[min(attempt, len(_PIFM_RECONNECT_DELAYS) - 1)]
            log.info('PiFmAdv: reconnect attempt %d in %.0fs…', attempt + 1, delay)
            time.sleep(delay)
            if self._closed:
                return
            try:
                self._start_pipeline()
                self._drain_queue()
                log.info('PiFmAdv: pipeline restarted (attempt %d) — %s', attempt + 1, self._label)
                return
            except Exception as exc:
                log.warning('PiFmAdv: restart attempt %d failed: %s', attempt + 1, exc)
                self._kill_pipeline()
                attempt += 1

    def _kill_pipeline(self) -> None:
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

    def _writer_loop(self) -> None:
        while not self._closed:
            try:
                chunk = self._queue.get(timeout=0.5)
            except _stdlib_queue.Empty:
                if not self._pipeline_alive():
                    log.warning('PiFmAdv: pipeline died silently — reconnecting')
                    self._kill_pipeline()
                    self._reconnect_with_drain()
                continue
            if chunk is None:
                break
            if not self._pipeline_alive():
                log.warning('PiFmAdv: pipeline dead before write — reconnecting')
                self._kill_pipeline()
                self._reconnect_with_drain()
                if self._closed:
                    break
                continue
            try:
                if self._ffmpeg and self._ffmpeg.stdin and not self._ffmpeg.stdin.closed:
                    self._ffmpeg.stdin.write(chunk)
                    self._ffmpeg.stdin.flush()
                else:
                    raise BrokenPipeError('ffmpeg stdin unavailable')
            except (BrokenPipeError, OSError, ValueError) as exc:
                if self._closed:
                    break
                log.warning('PiFmAdv: pipe write failed: %s', exc)
                self._kill_pipeline()
                self._reconnect_with_drain()

    async def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        try:
            self._queue.put_nowait(pcm)
        except _stdlib_queue.Full:
            pass

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except _stdlib_queue.Full:
            pass
        await asyncio.to_thread(self._kill_pipeline)
        await asyncio.to_thread(self._writer_thread.join, 5.0)
        log.info('PiFmAdv sink closed')
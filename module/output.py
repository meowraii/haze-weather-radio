from __future__ import annotations

import asyncio
import base64
import logging
import os
import select
import shlex
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

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
    'acompressor=threshold=-22dB:ratio=16:attack=5:release=70:makeup=18dB,'
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
            'song': f"{self._stream_name} - {metadata.title}" if isinstance(metadata, NowPlayingMetadata) else title,
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

    async def write(self, pcm: bytes) -> None:
        if self._closed or self._ffmpeg is None or self._ffmpeg.stdin is None or not pcm:
            return
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

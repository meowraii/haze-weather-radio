from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import os
import pathlib
import re
import secrets
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
try:
    import av
    from aiortc import RTCPeerConnection, RTCSessionDescription
    try:
        from aiortc import RTCRtpSender
    except Exception:
        RTCRtpSender = None
except Exception as exc:
    av = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    RTCRtpSender = None
    _WEBRTC_IMPORT_ERROR = exc
else:
    _WEBRTC_IMPORT_ERROR = None

pifm_bin = "/home/rai/PiFmAdv/src/pi_fm_adv"
DEFAULT_DEVIATION_HZ = 7000
DEFAULT_AUDIO_FILTERS = (
    'agate=threshold=-42dB:ratio=18:attack=4:release=110:makeup=1,'
    'highpass=f=30,'
    'lowpass=f=10000,'
    'equalizer=f=1800:t=q:w=1.1:g=2.5,'
    'acompressor=threshold=-18dB:ratio=3:attack=5:release=120:makeup=2,'
    'volume=14dB,'
    'alimiter=limit=0.995:level=disabled'
)

log = logging.getLogger(__name__)


class ReceiverHttpError(RuntimeError):
    def __init__(self, url: str, status: int, detail: str) -> None:
        super().__init__(f'{url} failed with HTTP {status}: {detail}')
        self.url = url
        self.status = status
        self.detail = detail


def _ensure_webrtc_dependencies() -> None:
    if _WEBRTC_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        'hazeReceiver.py requires aiortc for secure WebRTC. '
        'With av>=17.1.0, install the current aiortc wheel without dependencies '
        'until upstream metadata supports PyAV 17: python -m pip install --no-deps aiortc==1.14.0'
    ) from _WEBRTC_IMPORT_ERROR


@dataclass(frozen=True)
class ReceiverConfig:
    server_url: str
    feed_id: str
    receiver_api_base: str
    pair_token: str
    state_file: pathlib.Path
    allow_insecure_dev: bool
    output_sample_rate: int
    channels: int
    webrtc_input_sample_rate: int
    ffmpeg_bin: str
    ffmpeg_log_level: str
    audio_filters: str
    pifmadv_bin: str
    pifm_extra_args: tuple[str, ...]
    reconnect_initial_delay_s: float
    reconnect_max_delay_s: float
    reconnect_backoff: float
    stream_stall_timeout_s: float
    write_chunk_size: int
    audio_frame_ms: int


def _safe_feed_name(feed_id: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', feed_id).strip('._') or 'default'


def _default_state_file(feed_id: str) -> pathlib.Path:
    base = pathlib.Path.home() / '.haze_receiver'
    return base / f'{_safe_feed_name(feed_id)}.json'


def _normalize_server_url(raw: str, allow_insecure_dev: bool) -> str:
    value = str(raw or '').strip()
    if not value:
        raise ValueError('--server is required')
    parsed = urlparse(value)
    if not parsed.scheme:
        value = f'http://{value}'
        parsed = urlparse(value)
    if parsed.scheme in {'ws', 'wss'}:
        parsed = parsed._replace(scheme='https' if parsed.scheme == 'wss' else 'http')
    if parsed.scheme not in {'http', 'https'}:
        raise ValueError('--server must use http:// or https://')
    if not parsed.netloc:
        raise ValueError('--server must include a hostname or IP address')
    return urlunparse((parsed.scheme, parsed.netloc, '', '', '', '')).rstrip('/')


def _api_url(config: ReceiverConfig, path: str) -> str:
    base = '/' + config.receiver_api_base.strip('/') + '/'
    return urljoin(config.server_url + base, path.lstrip('/'))


def _panel_ws_url(config: ReceiverConfig) -> str:
    parsed = urlparse(config.server_url)
    scheme = 'wss' if parsed.scheme == 'https' else 'ws'
    base = urlunparse((scheme, parsed.netloc, '', '', '', '')).rstrip('/')
    return f'{base}/api/public/v1/panel/ws?feeds=1'


def _receiver_proof_message(kind: str, values: dict[str, Any]) -> bytes:
    payload = {'kind': kind, **{str(k): str(v) for k, v in values.items()}}
    return json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')


def _receiver_hmac(secret: str, message: bytes) -> str:
    return hmac.new(secret.encode('utf-8'), message, hashlib.sha256).hexdigest()


def _sdp_audio_codecs(sdp: str) -> list[str]:
    codecs: list[str] = []
    for line in str(sdp or '').splitlines():
        line = line.strip()
        if not line.lower().startswith('a=rtpmap:'):
            continue
        _, _, payload = line.partition(' ')
        codec = payload.split('/', 1)[0].strip()
        if codec and codec not in codecs:
            codecs.append(codec)
    return codecs


def _transmitter_label(transmitter: dict[str, Any]) -> str:
    relationship = str(transmitter.get('relationship') or 'unknown').strip() or 'unknown'
    frequency = str(transmitter.get('frequency_mhz') or 'unknown').strip() or 'unknown'
    return f'{relationship}@{frequency}MHz'


def _load_state(path: pathlib.Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open('r', encoding='utf-8') as handle:
            data = json.load(handle)
    except Exception:
        log.warning('Receiver state file is unreadable; starting unpaired')
        return {}
    return data if isinstance(data, dict) else {}


def _save_state(path: pathlib.Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write('\n')
    os.replace(tmp, path)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


async def _post_json(session: aiohttp.ClientSession, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with session.post(url, json=payload) as response:
        body = await response.text()
        if response.status >= 400:
            detail = body
            with contextlib.suppress(Exception):
                parsed = json.loads(body)
                detail = str(parsed.get('detail') or body)
            raise ReceiverHttpError(url, response.status, detail)
        try:
            parsed_body = json.loads(body or '{}')
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'{url} returned invalid JSON') from exc
        if not isinstance(parsed_body, dict):
            raise RuntimeError(f'{url} returned a non-object JSON payload')
        return parsed_body


class ReceiverSupervisor:
    def __init__(self, config: ReceiverConfig) -> None:
        self.config = config
        self.stop_event = asyncio.Event()
        self.last_audio_ts = 0.0
        self.receiver_hostname = socket.gethostname()
        self.state = _load_state(config.state_file)
        self.receiver_id = str(self.state.get('receiver_id') or uuid.uuid4())
        self.state['receiver_id'] = self.receiver_id
        self.state['feed_id'] = config.feed_id
        self.state['server_url'] = config.server_url
        last_transmitter = self.state.get('last_transmitter')
        self.last_transmitter = last_transmitter if isinstance(last_transmitter, dict) else None
        self.fallback_pifm_proc: asyncio.subprocess.Process | None = None
        self.fallback_ffmpeg_proc: asyncio.subprocess.Process | None = None
        self.fallback_pipe_task: asyncio.Task[Any] | None = None
        self.fallback_log_tasks: list[asyncio.Task[Any]] = []
        _save_state(config.state_file, self.state)

    async def run_forever(self) -> None:
        self._install_signal_handlers()
        delay = self.config.reconnect_initial_delay_s
        try:
            while not self.stop_event.is_set():
                try:
                    reason = await self._run_once()
                    delay = self.config.reconnect_initial_delay_s
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    reason = f'receiver failure: {exc}'
                if self.stop_event.is_set():
                    break
                log.warning('Receiver reconnect requested: %s', reason)
                await self._ensure_fallback_carrier()
                await asyncio.sleep(delay)
                delay = min(self.config.reconnect_max_delay_s, delay * self.config.reconnect_backoff)
        finally:
            await self._stop_fallback_carrier()
            log.info('Receiver stopped')

    async def _run_once(self) -> str:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=4, sock_read=None)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            ws_url = _panel_ws_url(self.config)
            async with session.ws_connect(
                ws_url,
                heartbeat=15,
                receive_timeout=None,
            ) as ws:
                feed = await self._wait_for_feed(ws)
                transmitter = feed.get('transmitter') if isinstance(feed.get('transmitter'), dict) else {}
                if transmitter.get('frequency_mhz') is not None:
                    self.last_transmitter = dict(transmitter)
                    self.state['last_transmitter'] = self.last_transmitter
                    _save_state(self.config.state_file, self.state)
                return await self._run_webrtc_session(ws, transmitter)

    async def _wait_for_feed(self, ws: aiohttp.ClientWebSocketResponse) -> dict[str, Any]:
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            msg = await ws.receive(timeout=max(0.1, deadline - time.monotonic()))
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f'public panel websocket error: {ws.exception()}')
            if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE}:
                raise RuntimeError('public panel websocket closed before feed metadata arrived')
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if payload.get('type') != 'public_state':
                continue
            data = payload.get('data') if isinstance(payload.get('data'), dict) else {}
            summary = data.get('summary') if isinstance(data.get('summary'), dict) else {}
            feeds = data.get('feeds') if isinstance(data.get('feeds'), list) else []
            if not feeds:
                feeds = summary.get('feeds') if isinstance(summary.get('feeds'), list) else []
            for feed in feeds:
                if not isinstance(feed, dict):
                    continue
                if str(feed.get('id') or '').strip() == self.config.feed_id:
                    if not bool(feed.get('webrtc_enabled')):
                        raise RuntimeError(f'feed {self.config.feed_id} does not have WebRTC enabled')
                    return feed
        raise RuntimeError(f'feed {self.config.feed_id} was not advertised by the public panel websocket')

    async def _receiver_auth(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        return await _post_json(session, _api_url(self.config, 'session'), {
            'feed_id': self.config.feed_id,
            'receiver_id': self.receiver_id,
            'receiver_hostname': self.receiver_hostname,
        })

    async def _pair_receiver(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        nonce = secrets.token_urlsafe(24)
        challenge = await _post_json(session, _api_url(self.config, 'pair/challenge'), {
            'feed_id': self.config.feed_id,
            'receiver_id': self.receiver_id,
            'receiver_hostname': self.receiver_hostname,
            'nonce': nonce,
        })
        challenge_id = str(challenge.get('challenge_id') or '')
        server_nonce = str(challenge.get('server_nonce') or '')
        if not challenge_id or not server_nonce:
            raise RuntimeError('pairing challenge was missing required fields')
        proof = _receiver_hmac(
            self.config.pair_token,
            _receiver_proof_message('pair-v1', {
                'challenge_id': challenge_id,
                'feed_id': self.config.feed_id,
                'receiver_id': self.receiver_id,
                'receiver_hostname': self.receiver_hostname,
                'receiver_nonce': nonce,
                'server_nonce': server_nonce,
            }),
        )
        completed = await _post_json(session, _api_url(self.config, 'pair/complete'), {
            'challenge_id': challenge_id,
            'feed_id': self.config.feed_id,
            'receiver_id': self.receiver_id,
            'receiver_hostname': self.receiver_hostname,
            'nonce': nonce,
            'proof': proof,
        })
        credential_id = str(completed.get('credential_id') or '')
        credential_secret = str(completed.get('credential_secret') or '')
        if not credential_id or not credential_secret:
            raise RuntimeError('pairing response did not include receiver credentials')
        self.state.update({
            'credential_id': credential_id,
            'credential_secret': credential_secret,
            'paired_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        })
        _save_state(self.config.state_file, self.state)
        log.info('Receiver paired with %s for feed %s', self.config.server_url, self.config.feed_id)
        return completed

    async def _request_session_cookie(
        self,
        session: aiohttp.ClientSession,
        credential_id: str,
        credential_secret: str,
    ) -> dict[str, Any]:
        nonce = secrets.token_urlsafe(24)
        proof = _receiver_hmac(
            credential_secret,
            _receiver_proof_message('session-v1', {
                'credential_id': credential_id,
                'feed_id': self.config.feed_id,
                'receiver_id': self.receiver_id,
                'receiver_hostname': self.receiver_hostname,
                'nonce': nonce,
            }),
        )
        return await _post_json(session, _api_url(self.config, 'session'), {
            'feed_id': self.config.feed_id,
            'receiver_id': self.receiver_id,
            'receiver_hostname': self.receiver_hostname,
            'credential_id': credential_id,
            'nonce': nonce,
            'proof': proof,
        })

    async def _run_webrtc_session(self, ws: aiohttp.ClientWebSocketResponse, transmitter: dict[str, Any]) -> str:
        if transmitter.get('frequency_mhz') is None:
            return 'receiver transmitter parameters did not include frequency_mhz'
        await self._stop_fallback_carrier()
        pifm_proc: asyncio.subprocess.Process | None = None
        ffmpeg_proc: asyncio.subprocess.Process | None = None
        pc: RTCPeerConnection | None = None
        try:
            pifm_proc = await self._start_pifmadv(transmitter)
            ffmpeg_proc = await self._start_audio_processor(transmitter)
            assert ffmpeg_proc.stdin is not None
            assert ffmpeg_proc.stdout is not None
            assert ffmpeg_proc.stderr is not None
            assert pifm_proc.stdin is not None
            assert pifm_proc.stderr is not None

            pc = RTCPeerConnection()
            track_future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()

            @pc.on('track')
            def _on_track(track: Any) -> None:
                if getattr(track, 'kind', '') == 'audio' and not track_future.done():
                    log.info('Receiver WebRTC audio track started: id=%s', getattr(track, 'id', 'unknown'))
                    track_future.set_result(track)

            @pc.on('iceconnectionstatechange')
            def _on_ice_state_change() -> None:
                log.info('Receiver WebRTC ICE state: %s', getattr(pc, 'iceConnectionState', 'unknown'))

            @pc.on('connectionstatechange')
            def _on_connection_state_change() -> None:
                log.info('Receiver WebRTC connection state: %s', getattr(pc, 'connectionState', 'unknown'))

            transceiver = pc.addTransceiver('audio', direction='recvonly')
            self._prefer_receiver_codecs(transceiver)
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            log.info(
                'Sending receiver WebRTC offer with audio codecs: %s',
                ', '.join(_sdp_audio_codecs(pc.localDescription.sdp)) or 'unknown',
            )
            await ws.send_json({
                'type': 'webrtc_offer',
                'feed_id': self.config.feed_id,
                'sdp': pc.localDescription.sdp,
                'sdp_type': pc.localDescription.type,
                'require_opus': True,
            })

            while True:
                msg = await ws.receive_json()
                msg_type = str(msg.get('type') or '')
                if msg_type == 'webrtc_answer':
                    answer_sdp = str(msg.get('sdp') or '')
                    log.info(
                        'Received receiver WebRTC answer with audio codecs: %s',
                        ', '.join(_sdp_audio_codecs(answer_sdp)) or 'unknown',
                    )
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type=str(msg.get('sdp_type') or 'answer')))
                    break
                if msg_type == 'webrtc_error':
                    return f'webrtc negotiation failed: {msg.get("detail")}'

            track = await asyncio.wait_for(track_future, timeout=10.0)
            self.last_audio_ts = time.monotonic()
            pump_audio_task = asyncio.create_task(
                self._pump_track_to_processor(track, ffmpeg_proc.stdin),
                name='webrtc_to_ffmpeg',
            )
            pipe_task = asyncio.create_task(
                self._pump_processor_to_pifm(ffmpeg_proc.stdout, pifm_proc.stdin),
                name='ffmpeg_to_pifm',
            )
            ffmpeg_wait_task = asyncio.create_task(self._wait_process(ffmpeg_proc, 'ffmpeg'), name='ffmpeg_wait')
            pifm_wait_task = asyncio.create_task(self._wait_process(pifm_proc, 'piFmAdv'), name='pifm_wait')
            ffmpeg_err_task = asyncio.create_task(self._log_stream(ffmpeg_proc.stderr, 'ffmpeg'), name='ffmpeg_stderr')
            pifm_err_task = asyncio.create_task(self._log_stream(pifm_proc.stderr, 'piFmAdv'), name='pifm_stderr')
            monitor_task = asyncio.create_task(self._monitor_health(ffmpeg_proc, pifm_proc, pc), name='monitor_health')
            ws_task = asyncio.create_task(self._watch_control_ws(ws), name='receiver_ws')

            done, pending = await asyncio.wait(
                {pump_audio_task, pipe_task, ffmpeg_wait_task, pifm_wait_task, monitor_task, ws_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            await self._cancel_tasks((*pending, ffmpeg_err_task, pifm_err_task), timeout_s=2.0)
            reason = 'receiver session stopped'
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    result = await task
                    if isinstance(result, str) and result:
                        reason = result
            return reason
        finally:
            if pc is not None:
                await self._close_peer_connection(pc)
            await self._terminate_process(ffmpeg_proc, 'ffmpeg')
            await self._terminate_process(pifm_proc, 'piFmAdv')

    def _prefer_receiver_codecs(self, transceiver: Any) -> None:
        if RTCRtpSender is None or not hasattr(transceiver, 'setCodecPreferences'):
            return
        with contextlib.suppress(Exception):
            capabilities = RTCRtpSender.getCapabilities('audio')
            preferred = [
                codec
                for codec in getattr(capabilities, 'codecs', [])
                if str(getattr(codec, 'mimeType', '')).lower() == 'audio/opus'
            ]
            preferred.sort(key=lambda codec: {
                'audio/opus': 0,
            }.get(str(getattr(codec, 'mimeType', '')).lower(), 99))
            if preferred:
                transceiver.setCodecPreferences(preferred)
                names = [str(getattr(codec, 'mimeType', '')).split('/', 1)[-1] for codec in preferred]
                log.info('Receiver preferred WebRTC codecs: %s', ', '.join(names))

    async def _start_audio_processor(self, transmitter: dict[str, Any]) -> asyncio.subprocess.Process:
        cmd = [
            self.config.ffmpeg_bin,
            '-hide_banner',
            '-nostats',
            '-loglevel',
            self.config.ffmpeg_log_level,
            '-f',
            's16le',
            '-ar',
            str(self.config.webrtc_input_sample_rate),
            '-ac',
            str(self.config.channels),
            '-i',
            'pipe:0',
            '-af',
            self.config.audio_filters,
            '-ar',
            str(self.config.output_sample_rate),
            '-ac',
            str(self.config.channels),
            '-f',
            'wav',
            'pipe:1',
        ]
        log.info(
            'Starting WebRTC audio processor for %s (%d Hz -> %d Hz)',
            _transmitter_label(transmitter),
            self.config.webrtc_input_sample_rate,
            self.config.output_sample_rate,
        )
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **self._process_kwargs(),
        )

    async def _start_silence_processor(self, transmitter: dict[str, Any]) -> asyncio.subprocess.Process:
        layout = 'mono' if self.config.channels == 1 else 'stereo'
        cmd = [
            self.config.ffmpeg_bin,
            '-hide_banner',
            '-nostats',
            '-loglevel',
            self.config.ffmpeg_log_level,
            '-f',
            'lavfi',
            '-i',
            f'anullsrc=r={self.config.output_sample_rate}:cl={layout}',
            '-af',
            self.config.audio_filters,
            '-ar',
            str(self.config.output_sample_rate),
            '-ac',
            str(self.config.channels),
            '-f',
            'wav',
            'pipe:1',
        ]
        log.info('Starting silent reconnect carrier for %s', _transmitter_label(transmitter))
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **self._process_kwargs(),
        )

    def _fallback_carrier_running(self) -> bool:
        return (
            self.fallback_pifm_proc is not None
            and self.fallback_pifm_proc.returncode is None
            and self.fallback_ffmpeg_proc is not None
            and self.fallback_ffmpeg_proc.returncode is None
            and self.fallback_pipe_task is not None
            and not self.fallback_pipe_task.done()
        )

    async def _ensure_fallback_carrier(self) -> None:
        if self._fallback_carrier_running():
            return
        await self._stop_fallback_carrier()
        transmitter = self.last_transmitter
        if not isinstance(transmitter, dict) or transmitter.get('frequency_mhz') is None:
            log.warning('No cached transmitter metadata; reconnect will not hold a silent carrier')
            return
        try:
            self.fallback_pifm_proc = await self._start_pifmadv(transmitter)
            self.fallback_ffmpeg_proc = await self._start_silence_processor(transmitter)
            assert self.fallback_ffmpeg_proc.stdout is not None
            assert self.fallback_ffmpeg_proc.stderr is not None
            assert self.fallback_pifm_proc.stdin is not None
            assert self.fallback_pifm_proc.stderr is not None
            self.fallback_pipe_task = asyncio.create_task(
                self._pump_processor_to_pifm(self.fallback_ffmpeg_proc.stdout, self.fallback_pifm_proc.stdin),
                name='fallback_silence_to_pifm',
            )
            self.fallback_log_tasks = [
                asyncio.create_task(self._log_stream(self.fallback_ffmpeg_proc.stderr, 'fallback_ffmpeg'), name='fallback_ffmpeg_stderr'),
                asyncio.create_task(self._log_stream(self.fallback_pifm_proc.stderr, 'fallback_piFmAdv'), name='fallback_pifm_stderr'),
            ]
            log.info('Silent reconnect carrier is active')
        except Exception as exc:
            log.warning('Could not start silent reconnect carrier: %s', exc)
            await self._stop_fallback_carrier()

    async def _stop_fallback_carrier(self) -> None:
        tasks: list[asyncio.Task[Any]] = []
        if self.fallback_pipe_task is not None:
            tasks.append(self.fallback_pipe_task)
        tasks.extend(self.fallback_log_tasks)
        if tasks:
            await self._cancel_tasks(tuple(tasks), timeout_s=1.0)
        await self._terminate_process(self.fallback_ffmpeg_proc, 'fallback ffmpeg')
        await self._terminate_process(self.fallback_pifm_proc, 'fallback piFmAdv')
        self.fallback_pipe_task = None
        self.fallback_log_tasks = []
        self.fallback_ffmpeg_proc = None
        self.fallback_pifm_proc = None

    async def _start_pifmadv(self, transmitter: dict[str, Any]) -> asyncio.subprocess.Process:
        frequency_mhz = float(transmitter['frequency_mhz'])
        deviation_hz = int(transmitter.get('deviation_hz') or DEFAULT_DEVIATION_HZ)
        preemphasis = str(transmitter.get('preemphasis') or 'none').strip().lower()
        cmd = [
            self.config.pifmadv_bin,
            '--audio',
            '-',
            '--freq',
            str(frequency_mhz),
            '--dev',
            str(deviation_hz),
            '--rds',
            '0',
            *self.config.pifm_extra_args,
        ]
        if preemphasis in {'50', '50us', 'eu'}:
            cmd.extend(['--preemph', 'eu'])
        elif preemphasis in {'75', '75us', 'us'}:
            cmd.extend(['--preemph', 'us'])
        log.info(
            'Starting piFmAdv transmitter %.3f MHz relationship=%s deviation=%dHz preemphasis=%s (%s)',
            frequency_mhz,
            transmitter.get('relationship') or 'unknown',
            deviation_hz,
            preemphasis,
            self.config.pifmadv_bin,
        )
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            **self._process_kwargs(),
        )

    async def _pump_track_to_processor(self, track: Any, ffmpeg_stdin: asyncio.StreamWriter) -> str:
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        producer = asyncio.create_task(self._read_track_audio(track, queue), name='webrtc_audio_reader')
        writer = asyncio.create_task(self._write_paced_audio(queue, ffmpeg_stdin), name='ffmpeg_audio_writer')
        try:
            done, pending = await asyncio.wait({producer, writer}, return_when=asyncio.FIRST_COMPLETED)
            await self._cancel_tasks(tuple(pending), timeout_s=1.0)
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    result = await task
                    if isinstance(result, str) and result:
                        return result
            return 'audio pump stopped'
        finally:
            with contextlib.suppress(Exception):
                track.stop()
            await self._cancel_tasks((producer, writer), timeout_s=1.0)

    async def _read_track_audio(self, track: Any, queue: asyncio.Queue[bytes]) -> str:
        layout = 'mono' if self.config.channels == 1 else 'stereo'
        resampler = av.AudioResampler(format='s16', layout=layout, rate=self.config.webrtc_input_sample_rate)
        logged_frame = False
        while not self.stop_event.is_set():
            try:
                frame = await track.recv()
            except Exception as exc:
                return f'webrtc audio track ended: {exc}'
            if not logged_frame:
                log.info(
                    'Receiver WebRTC audio frame: rate=%s layout=%s samples=%s format=%s',
                    getattr(frame, 'sample_rate', 'unknown'),
                    getattr(getattr(frame, 'layout', None), 'name', 'unknown'),
                    getattr(frame, 'samples', 'unknown'),
                    getattr(getattr(frame, 'format', None), 'name', 'unknown'),
                )
                logged_frame = True
            chunks: list[bytes] = []
            try:
                for out_frame in resampler.resample(frame):
                    needed = int(out_frame.samples) * self.config.channels * 2
                    raw = bytes(out_frame.planes[0])
                    if len(raw) < needed:
                        raw += b'\x00' * (needed - len(raw))
                    chunks.append(raw[:needed])
            except Exception as exc:
                return f'webrtc audio resample failed: {exc}'
            chunk = b''.join(chunks)
            if not chunk:
                continue
            self.last_audio_ts = time.monotonic()
            self._push_audio_chunk(queue, chunk)
        return 'shutdown requested'

    async def _write_paced_audio(self, queue: asyncio.Queue[bytes], ffmpeg_stdin: asyncio.StreamWriter) -> str:
        frame_ms = max(10, min(100, self.config.audio_frame_ms))
        frame_samples = max(1, int(self.config.webrtc_input_sample_rate * frame_ms / 1000))
        frame_bytes = frame_samples * self.config.channels * 2
        silence = b'\x00' * frame_bytes
        buffered = bytearray()
        next_tick = time.monotonic()
        transport = getattr(ffmpeg_stdin, 'transport', None)
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.set_write_buffer_limits(high=frame_bytes * 50, low=frame_bytes * 10)
        while not self.stop_event.is_set():
            if len(buffered) < frame_bytes:
                try:
                    buffered.extend(await asyncio.wait_for(queue.get(), timeout=frame_ms / 1000))
                except asyncio.TimeoutError:
                    pass
            while len(buffered) < frame_bytes:
                try:
                    buffered.extend(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if len(buffered) >= frame_bytes:
                chunk = bytes(buffered[:frame_bytes])
                del buffered[:frame_bytes]
            elif buffered:
                chunk = bytes(buffered) + silence[len(buffered):]
                buffered.clear()
            else:
                chunk = silence
            try:
                ffmpeg_stdin.write(chunk)
                await ffmpeg_stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return 'ffmpeg stdin closed'
            except Exception as exc:
                return f'ffmpeg write failed: {exc}'
            next_tick += frame_ms / 1000
            sleep_for = next_tick - time.monotonic()
            if sleep_for < -0.5:
                next_tick = time.monotonic()
                sleep_for = 0
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        return 'shutdown requested'

    def _push_audio_chunk(self, queue: asyncio.Queue[bytes], chunk: bytes) -> None:
        while queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(chunk)

    async def _pump_processor_to_pifm(
        self,
        ffmpeg_stdout: asyncio.StreamReader,
        pifm_stdin: asyncio.StreamWriter,
    ) -> str:
        while not self.stop_event.is_set():
            chunk = await ffmpeg_stdout.read(self.config.write_chunk_size)
            if not chunk:
                return 'ffmpeg output ended'
            try:
                pifm_stdin.write(chunk)
                await pifm_stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return 'piFmAdv stdin closed'
            except Exception as exc:
                return f'piFmAdv write failed: {exc}'
        return 'shutdown requested'

    async def _wait_process(self, proc: asyncio.subprocess.Process, name: str) -> str:
        code = await proc.wait()
        return f'{name} exited with code {code}'

    async def _watch_control_ws(self, ws: aiohttp.ClientWebSocketResponse) -> str:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.ERROR:
                return f'receiver websocket error: {ws.exception()}'
            if msg.type == aiohttp.WSMsgType.CLOSED:
                return 'receiver websocket closed'
            if msg.type == aiohttp.WSMsgType.TEXT:
                with contextlib.suppress(Exception):
                    payload = json.loads(msg.data)
                    if payload.get('type') == 'webrtc_error':
                        return f"receiver websocket reported error: {payload.get('detail')}"
        return 'receiver websocket ended'

    async def _monitor_health(
        self,
        ffmpeg_proc: asyncio.subprocess.Process,
        pifm_proc: asyncio.subprocess.Process,
        pc: RTCPeerConnection,
    ) -> str:
        while not self.stop_event.is_set():
            if ffmpeg_proc.returncode is not None:
                return f'ffmpeg exited with code {ffmpeg_proc.returncode}'
            if pifm_proc.returncode is not None:
                return f'piFmAdv exited with code {pifm_proc.returncode}'
            if pc.connectionState in {'failed', 'closed'}:
                return f'webrtc peer connection {pc.connectionState}'
            idle_for = time.monotonic() - self.last_audio_ts
            if idle_for >= self.config.stream_stall_timeout_s:
                return f'webrtc audio stalled for {idle_for:.1f}s'
            await asyncio.sleep(1.0)
        return 'shutdown requested'

    async def _log_stream(self, stream: asyncio.StreamReader, name: str) -> None:
        while not self.stop_event.is_set():
            line = await stream.readline()
            if not line:
                return
            text = line.decode(errors='replace').rstrip()
            if text:
                logging.info('[%s] %s', name, text)

    async def _close_peer_connection(self, pc: RTCPeerConnection) -> None:
        with contextlib.suppress(Exception):
            for receiver in pc.getReceivers():
                track = getattr(receiver, 'track', None)
                if track is not None:
                    track.stop()
        try:
            await asyncio.wait_for(pc.close(), timeout=3.0)
        except asyncio.TimeoutError:
            log.warning('WebRTC peer close timed out')
        except Exception as exc:
            log.warning('WebRTC peer close failed: %s', exc)

    async def _cancel_tasks(self, tasks: tuple[asyncio.Task[Any], ...], timeout_s: float) -> None:
        live = [task for task in tasks if not task.done()]
        for task in live:
            task.cancel()
        if not live:
            return
        done, pending = await asyncio.wait(live, timeout=timeout_s)
        for task in done:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        for task in pending:
            log.warning('Task %s did not stop cleanly', task.get_name())

    def _process_kwargs(self) -> dict[str, Any]:
        if os.name == 'nt':
            return {}
        return {'start_new_session': True}

    async def _terminate_process(self, proc: asyncio.subprocess.Process | None, name: str) -> None:
        if proc is None:
            return
        stdin = getattr(proc, 'stdin', None)
        if stdin is not None:
            with contextlib.suppress(Exception):
                stdin.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stdin.wait_closed(), timeout=1.0)
        if proc.returncode is None:
            self._signal_process(proc, signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self._signal_process(proc, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
        log.info('%s stopped with code %s', name, proc.returncode)

    def _signal_process(self, proc: asyncio.subprocess.Process, sig: signal.Signals) -> None:
        if proc.returncode is not None:
            return
        try:
            if os.name != 'nt':
                os.killpg(proc.pid, sig)
            elif sig == signal.SIGKILL:
                proc.kill()
            else:
                proc.terminate()
        except ProcessLookupError:
            return
        except Exception as exc:
            log.warning('Could not signal process %s with %s: %s', proc.pid, sig, exc)


    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()

        def _trigger_stop() -> None:
            if not self.stop_event.is_set():
                log.info('Shutdown signal received')
                self.stop_event.set()

        for sig_name in ('SIGINT', 'SIGTERM'):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, _trigger_stop)
            except NotImplementedError:
                signal.signal(sig, lambda _s, _f: _trigger_stop())


def _parse_args() -> ReceiverConfig:
    parser = argparse.ArgumentParser(
        prog='hazeReceiver.py',
        description='Haze feed receiver using WebSocket signaling, WebRTC audio, and piFmAdv output.',
    )
    parser.add_argument('--server', required=True, help='Haze server URL, for example http://haze-host:8086')
    parser.add_argument('--feed-id', required=True, help='Feed ID to receive, for example sk-0001')
    parser.add_argument('--receiver-api-base', default='/api/receiver/v1')
    parser.add_argument('--pair-token', default='', help=argparse.SUPPRESS)
    parser.add_argument('--pair-token-env', default='', help=argparse.SUPPRESS)
    parser.add_argument('--state-file', default='')
    parser.add_argument('--allow-insecure-dev', action='store_true')

    parser.add_argument('--output-sample-rate', type=int, default=48000)
    parser.add_argument('--channels', type=int, choices=[1, 2], default=1)
    parser.add_argument('--webrtc-input-sample-rate', type=int, default=48000)
    parser.add_argument('--ffmpeg-bin', default='ffmpeg')
    parser.add_argument('--ffmpeg-log-level', default='warning')
    parser.add_argument(
        '--audio-filters',
        default=DEFAULT_AUDIO_FILTERS,
    )
    parser.add_argument('--pifmadv-bin', default=pifm_bin)
    parser.add_argument('--pi-extra-arg', action='append', default=[])

    parser.add_argument('--stall-timeout', type=float, default=12.0)
    parser.add_argument('--reconnect-initial-delay', type=float, default=1.0)
    parser.add_argument('--reconnect-max-delay', type=float, default=8.0)
    parser.add_argument('--reconnect-backoff', type=float, default=1.5)
    parser.add_argument('--chunk-size', type=int, default=4096)
    parser.add_argument('--audio-frame-ms', type=int, default=20)
    args = parser.parse_args()

    pair_token = args.pair_token
    if not pair_token and args.pair_token_env:
        pair_token = os.environ.get(args.pair_token_env, '')

    try:
        server_url = _normalize_server_url(args.server, args.allow_insecure_dev)
    except ValueError as exc:
        parser.error(str(exc))

    state_file = pathlib.Path(args.state_file).expanduser() if args.state_file else _default_state_file(args.feed_id)
    return ReceiverConfig(
        server_url=server_url,
        feed_id=str(args.feed_id).strip(),
        receiver_api_base=str(args.receiver_api_base or '/api/receiver/v1'),
        pair_token=str(pair_token or ''),
        state_file=state_file,
        allow_insecure_dev=bool(args.allow_insecure_dev),
        output_sample_rate=max(8000, int(args.output_sample_rate)),
        channels=int(args.channels),
        webrtc_input_sample_rate=max(8000, int(args.webrtc_input_sample_rate)),
        ffmpeg_bin=str(args.ffmpeg_bin or 'ffmpeg'),
        ffmpeg_log_level=str(args.ffmpeg_log_level or 'warning'),
        audio_filters=str(args.audio_filters or 'anull'),
        pifmadv_bin=str(args.pifmadv_bin or pifm_bin),
        pifm_extra_args=tuple(args.pi_extra_arg),
        reconnect_initial_delay_s=max(0.1, float(args.reconnect_initial_delay)),
        reconnect_max_delay_s=max(0.1, float(args.reconnect_max_delay)),
        reconnect_backoff=max(1.0, float(args.reconnect_backoff)),
        stream_stall_timeout_s=max(2.0, float(args.stall_timeout)),
        write_chunk_size=max(512, int(args.chunk_size)),
        audio_frame_ms=max(10, min(100, int(args.audio_frame_ms))),
    )


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    for name in ('aioice', 'aiortc'):
        logging.getLogger(name).setLevel(logging.ERROR)


async def _main() -> None:
    _configure_logging()
    config = _parse_args()
    _ensure_webrtc_dependencies()
    supervisor = ReceiverSupervisor(config)
    await supervisor.run_forever()


if __name__ == '__main__':
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass

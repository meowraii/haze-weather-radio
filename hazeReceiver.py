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
import struct
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse, urlunparse

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
    'highpass=f=30,'
    'lowpass=f=10000,'
    'equalizer=f=1800:t=q:w=1.1:g=2.5,'
    'acompressor=threshold=-20dB:ratio=3.5:attack=5:release=120:makeup=5,'
    'volume=16dB,'
    'alimiter=limit=0.98:level=disabled'
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
    jitter_buffer_ms: int
    max_jitter_buffer_ms: int
    max_pacing_lag_ms: int
    preferred_codecs: tuple[str, ...]
    transport: str
    allow_webrtc_fallback: bool
    http_codec: str
    http_reconnect_delay_max_s: int
    http_read_timeout_s: float
    metrics_interval_s: float
    diagnose_audio: bool
    diagnose_duration_s: float
    diagnose_output: pathlib.Path | None


def _safe_feed_name(feed_id: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', feed_id).strip('._') or 'default'


def _default_state_file(feed_id: str) -> pathlib.Path:
    base = pathlib.Path.home() / '.haze_receiver'
    return base / f'{_safe_feed_name(feed_id)}.json'


def _streaming_wav_header(sample_rate: int, channels: int) -> bytes:
    bits_per_sample = 16
    byte_rate = int(sample_rate) * int(channels) * bits_per_sample // 8
    block_align = int(channels) * bits_per_sample // 8
    return b''.join((
        b'RIFF',
        struct.pack('<I', 0xFFFFFFFF),
        b'WAVE',
        b'fmt ',
        struct.pack('<IHHIIHH', 16, 1, int(channels), int(sample_rate), byte_rate, block_align, bits_per_sample),
        b'data',
        struct.pack('<I', 0xFFFFFFFF),
    ))


def _drop_pcm_frames_evenly(data: bytes, target_bytes: int, sample_bytes: int) -> bytes:
    """Return target_bytes by spreading dropped PCM sample frames across data."""
    excess_bytes = len(data) - target_bytes
    if excess_bytes <= 0:
        return data[:target_bytes]
    drop_frames = excess_bytes // sample_bytes
    total_frames = len(data) // sample_bytes
    target_frames = target_bytes // sample_bytes
    if drop_frames <= 0 or total_frames <= target_frames:
        return data[:target_bytes]
    drop_points = {
        min(total_frames - 1, max(0, int((index + 1) * total_frames / (drop_frames + 1))))
        for index in range(drop_frames)
    }
    out = bytearray(target_bytes)
    write_at = 0
    for frame_index in range(total_frames):
        if frame_index in drop_points:
            continue
        if write_at >= target_bytes:
            break
        start = frame_index * sample_bytes
        out[write_at:write_at + sample_bytes] = data[start:start + sample_bytes]
        write_at += sample_bytes
    if write_at < target_bytes:
        out[write_at:] = data[len(data) - (target_bytes - write_at):]
    return bytes(out)


def _pcm16_metrics(data: bytes, frame_samples: int = 960) -> dict[str, Any]:
    sample_count = len(data) // 2
    if sample_count <= 0:
        return {
            'samples': 0,
            'frames': 0,
            'peak': 0,
            'peak_dbfs': -120.0,
            'rms_avg_dbfs': -120.0,
            'rms_min_dbfs': -120.0,
            'rms_max_dbfs': -120.0,
            'near_silent_frames': 0,
            'repeated_frames': 0,
            'repeated_non_silent_frames': 0,
            'clipped_samples': 0,
            'zero_ratio': 0.0,
            'max_sample_jump': 0,
        }
    usable = sample_count * 2
    samples = struct.unpack('<' + 'h' * sample_count, data[:usable])
    frame_count = sample_count // frame_samples
    peak = 0
    clipped = 0
    zeros = 0
    max_jump = 0
    near_silent_frames = 0
    repeated_frames = 0
    repeated_non_silent_frames = 0
    rms_values: list[float] = []
    previous_frame: tuple[int, ...] | None = None
    previous_sample: int | None = None
    for sample in samples:
        abs_sample = abs(sample)
        peak = max(peak, abs_sample)
        if abs_sample >= 32760:
            clipped += 1
        if sample == 0:
            zeros += 1
        if previous_sample is not None:
            max_jump = max(max_jump, abs(sample - previous_sample))
        previous_sample = sample
    for start in range(0, frame_count * frame_samples, frame_samples):
        frame = tuple(samples[start:start + frame_samples])
        frame_peak = max((abs(sample) for sample in frame), default=0)
        if frame_peak <= 20:
            near_silent_frames += 1
        if previous_frame is not None and frame == previous_frame:
            repeated_frames += 1
            if frame_peak > 20:
                repeated_non_silent_frames += 1
        previous_frame = frame
        if frame:
            rms_values.append((sum(float(sample) * float(sample) for sample in frame) / len(frame)) ** 0.5)

    def dbfs(value: float) -> float:
        if value <= 0:
            return -120.0
        return 20.0 * math.log10(value / 32768.0)

    import math
    rms_avg = sum(rms_values) / len(rms_values) if rms_values else 0.0
    return {
        'samples': sample_count,
        'frames': frame_count,
        'peak': peak,
        'peak_dbfs': round(dbfs(float(peak)), 2),
        'rms_avg_dbfs': round(dbfs(rms_avg), 2),
        'rms_min_dbfs': round(dbfs(min(rms_values) if rms_values else 0.0), 2),
        'rms_max_dbfs': round(dbfs(max(rms_values) if rms_values else 0.0), 2),
        'near_silent_frames': near_silent_frames,
        'repeated_frames': repeated_frames,
        'repeated_non_silent_frames': repeated_non_silent_frames,
        'clipped_samples': clipped,
        'zero_ratio': round(zeros / sample_count, 6),
        'max_sample_jump': max_jump,
    }


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
            receiver_session = await self._start_receiver_session(session)
            ws_url = str(receiver_session.get('ws_url') or '').strip()
            if not ws_url:
                raise RuntimeError('receiver session did not include ws_url')
            async with session.ws_connect(
                ws_url,
                heartbeat=15,
                receive_timeout=None,
            ) as ws:
                transmitter = await self._wait_for_receiver_ready(ws)
                if transmitter.get('frequency_mhz') is not None:
                    self.last_transmitter = dict(transmitter)
                    self.state['last_transmitter'] = self.last_transmitter
                    _save_state(self.config.state_file, self.state)
                return await self._run_audio_session(session, ws, transmitter)

    async def _start_receiver_session(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        credential_id = str(self.state.get('credential_id') or '').strip()
        credential_secret = str(self.state.get('credential_secret') or '').strip()
        if credential_id and credential_secret:
            try:
                return await self._request_session_cookie(session, credential_id, credential_secret)
            except ReceiverHttpError as exc:
                if exc.status not in {401, 403}:
                    raise
                log.warning('Stored receiver credential was rejected; falling back to unpaired session')
        if self.config.pair_token:
            try:
                await self._pair_receiver(session)
                credential_id = str(self.state.get('credential_id') or '').strip()
                credential_secret = str(self.state.get('credential_secret') or '').strip()
                if credential_id and credential_secret:
                    return await self._request_session_cookie(session, credential_id, credential_secret)
            except ReceiverHttpError as exc:
                log.warning('Receiver pairing failed: %s', exc)
        return await self._receiver_auth(session)

    async def _wait_for_receiver_ready(self, ws: aiohttp.ClientWebSocketResponse) -> dict[str, Any]:
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            msg = await ws.receive(timeout=max(0.1, deadline - time.monotonic()))
            if msg.type == aiohttp.WSMsgType.ERROR:
                raise RuntimeError(f'receiver websocket error: {ws.exception()}')
            if msg.type in {aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE}:
                raise RuntimeError('receiver websocket closed before transmitter metadata arrived')
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                payload = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if payload.get('type') == 'receiver_error':
                raise RuntimeError(f"receiver websocket error: {payload.get('detail')}")
            if payload.get('type') != 'receiver_ready':
                continue
            transmitter = payload.get('transmitter') if isinstance(payload.get('transmitter'), dict) else {}
            if not transmitter:
                raise RuntimeError('receiver_ready did not include transmitter metadata')
            log.info(
                'Receiver ready for feed %s transmitter=%s',
                payload.get('feed_id') or self.config.feed_id,
                _transmitter_label(transmitter),
            )
            return transmitter
        raise RuntimeError(f'feed {self.config.feed_id} receiver metadata timed out')

    async def _run_audio_session(
        self,
        session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
        transmitter: dict[str, Any],
    ) -> str:
        transport = self.config.transport
        if transport in {'auto', 'http'}:
            if await self._http_audio_available(session):
                log.info(
                    'Receiver selected HTTP %s media transport for feed %s',
                    self.config.http_codec,
                    self.config.feed_id,
                )
                return await self._run_http_audio_session(session, ws, transmitter)
            if transport == 'http' or not self.config.allow_webrtc_fallback:
                return 'HTTP receiver audio stream is unavailable; WebRTC fallback is disabled'
            log.warning('HTTP receiver stream unavailable; falling back to WebRTC because --allow-webrtc-fallback was set')
        _ensure_webrtc_dependencies()
        log.info('Receiver selected WebRTC media transport for feed %s', self.config.feed_id)
        return await self._run_webrtc_session(ws, transmitter)

    def _http_audio_url(self) -> str:
        query = urlencode({
            'feed': self.config.feed_id,
            'codec': self.config.http_codec,
        })
        return f'{self.config.server_url}/api/public/v1/feed/audio?{query}'

    async def _http_audio_available(self, session: aiohttp.ClientSession) -> bool:
        url = self._http_audio_url()
        try:
            timeout = aiohttp.ClientTimeout(total=5, sock_connect=3, sock_read=3)
            async with session.head(url, timeout=timeout) as response:
                if response.status < 200 or response.status >= 300:
                    log.warning('HTTP receiver stream returned HTTP %s', response.status)
                    return False
                return True
        except Exception as exc:
            log.warning('HTTP receiver stream probe failed: %s', exc)
            return False

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
                'preferred_codec': self.config.preferred_codecs[0] if self.config.preferred_codecs else '',
                'require_opus': False,
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

    async def _run_http_audio_session(
        self,
        session: aiohttp.ClientSession,
        ws: aiohttp.ClientWebSocketResponse,
        transmitter: dict[str, Any],
    ) -> str:
        if transmitter.get('frequency_mhz') is None:
            return 'receiver transmitter parameters did not include frequency_mhz'
        await self._stop_fallback_carrier()
        pifm_proc: asyncio.subprocess.Process | None = None
        ffmpeg_proc: asyncio.subprocess.Process | None = None
        try:
            pifm_proc = await self._start_pifmadv(transmitter)
            if self._can_direct_http_raw_to_pifm():
                assert pifm_proc.stdin is not None
                assert pifm_proc.stderr is not None

                self.last_audio_ts = time.monotonic()
                pipe_task = asyncio.create_task(
                    self._pump_http_raw_to_pifm(session, pifm_proc.stdin),
                    name='http_raw_to_pifm',
                )
                pifm_wait_task = asyncio.create_task(self._wait_process(pifm_proc, 'piFmAdv'), name='http_raw_pifm_wait')
                pifm_err_task = asyncio.create_task(self._log_stream(pifm_proc.stderr, 'piFmAdv'), name='http_raw_pifm_stderr')
                monitor_task = asyncio.create_task(self._monitor_direct_http_health(pifm_proc), name='http_raw_monitor_health')
                ws_task = asyncio.create_task(self._watch_control_ws(ws), name='receiver_ws')
                ws_task.add_done_callback(self._log_http_control_ws_done)

                done, pending = await asyncio.wait(
                    {pipe_task, pifm_wait_task, monitor_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                await self._cancel_tasks((*pending, pifm_err_task, ws_task), timeout_s=2.0)
                reason = 'HTTP raw receiver session stopped'
                for task in done:
                    with contextlib.suppress(asyncio.CancelledError):
                        result = await task
                        if isinstance(result, str) and result:
                            reason = result
                return reason

            ffmpeg_proc = await self._start_http_audio_processor(transmitter)
            assert ffmpeg_proc.stdout is not None
            assert ffmpeg_proc.stderr is not None
            assert pifm_proc.stdin is not None
            assert pifm_proc.stderr is not None

            self.last_audio_ts = time.monotonic()
            pipe_task = asyncio.create_task(
                self._pump_processor_to_pifm(ffmpeg_proc.stdout, pifm_proc.stdin),
                name='http_ffmpeg_to_pifm',
            )
            ffmpeg_wait_task = asyncio.create_task(self._wait_process(ffmpeg_proc, 'ffmpeg'), name='http_ffmpeg_wait')
            pifm_wait_task = asyncio.create_task(self._wait_process(pifm_proc, 'piFmAdv'), name='http_pifm_wait')
            ffmpeg_err_task = asyncio.create_task(self._log_stream(ffmpeg_proc.stderr, 'http_ffmpeg'), name='http_ffmpeg_stderr')
            pifm_err_task = asyncio.create_task(self._log_stream(pifm_proc.stderr, 'piFmAdv'), name='http_pifm_stderr')
            monitor_task = asyncio.create_task(self._monitor_http_health(ffmpeg_proc, pifm_proc), name='http_monitor_health')
            ws_task = asyncio.create_task(self._watch_control_ws(ws), name='receiver_ws')
            ws_task.add_done_callback(self._log_http_control_ws_done)

            done, pending = await asyncio.wait(
                {pipe_task, ffmpeg_wait_task, pifm_wait_task, monitor_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            await self._cancel_tasks((*pending, ffmpeg_err_task, pifm_err_task, ws_task), timeout_s=2.0)
            reason = 'HTTP receiver session stopped'
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    result = await task
                    if isinstance(result, str) and result:
                        reason = result
            return reason
        finally:
            await self._terminate_process(ffmpeg_proc, 'http ffmpeg')
            await self._terminate_process(pifm_proc, 'piFmAdv')

    def _can_direct_http_raw_to_pifm(self) -> bool:
        codec = self.config.http_codec.strip().lower().replace('-', '_')
        filters = self.config.audio_filters.strip().lower()
        return (
            codec in {'raw', 'raw_pcm16', 's16le'}
            and self.config.output_sample_rate == 48000
            and self.config.channels == 1
            and filters in {'', 'anull'}
        )

    async def _pump_http_raw_to_pifm(
        self,
        session: aiohttp.ClientSession,
        pifm_stdin: asyncio.StreamWriter,
    ) -> str:
        queue_frames = max(
            32,
            int(self.config.max_jitter_buffer_ms / max(1, self.config.audio_frame_ms)) * 4,
        )
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_frames)
        reader = asyncio.create_task(self._read_http_raw_audio(session, queue), name='http_raw_reader')
        writer = asyncio.create_task(self._write_paced_raw_audio(queue, pifm_stdin), name='http_raw_writer')
        try:
            done, pending = await asyncio.wait({reader, writer}, return_when=asyncio.FIRST_COMPLETED)
            await self._cancel_tasks(tuple(pending), timeout_s=1.0)
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    result = await task
                    if isinstance(result, str) and result:
                        return result
            return 'HTTP raw receiver session stopped'
        finally:
            await self._cancel_tasks((reader, writer), timeout_s=1.0)

    async def _read_http_raw_audio(
        self,
        session: aiohttp.ClientSession,
        queue: asyncio.Queue[bytes],
    ) -> str:
        url = self._http_audio_url()
        timeout = aiohttp.ClientTimeout(
            total=None,
            sock_connect=5,
            sock_read=max(5.0, self.config.http_read_timeout_s),
        )
        try:
            async with session.get(url, timeout=timeout) as response:
                if response.status < 200 or response.status >= 300:
                    detail = (await response.text())[:200]
                    return f'HTTP raw receiver stream returned HTTP {response.status}: {detail}'
                log.info(
                    'Reading HTTP raw PCM for feed %s through paced receiver clock',
                    self.config.feed_id,
                )
                last_chunk_at: float | None = None
                async for chunk in response.content.iter_chunked(self.config.write_chunk_size):
                    if not chunk:
                        continue
                    now = time.monotonic()
                    if last_chunk_at is not None:
                        gap_ms = (now - last_chunk_at) * 1000
                        if gap_ms > self.config.max_pacing_lag_ms:
                            log.warning('HTTP raw receiver chunk gap %.1f ms', gap_ms)
                    last_chunk_at = now
                    self.last_audio_ts = time.monotonic()
                    self._push_audio_chunk(queue, bytes(chunk))
        except asyncio.TimeoutError:
            return 'HTTP raw receiver stream timed out'
        except Exception as exc:
            return f'HTTP raw receiver stream failed: {exc}'
        return 'HTTP raw receiver stream ended'

    async def _write_paced_raw_audio(
        self,
        queue: asyncio.Queue[bytes],
        pifm_stdin: asyncio.StreamWriter,
        label: str = 'HTTP raw receiver',
    ) -> str:
        frame_ms = max(10, min(100, self.config.audio_frame_ms))
        sample_bytes = self.config.channels * 2
        frame_bytes = max(
            sample_bytes,
            int(self.config.output_sample_rate * sample_bytes * frame_ms / 1000),
        )
        frame_bytes -= frame_bytes % sample_bytes
        frame_sample_count = max(1, frame_bytes // sample_bytes)
        byte_rate = self.config.output_sample_rate * sample_bytes
        target_buffer_bytes = max(
            frame_bytes,
            int(byte_rate * max(0, self.config.jitter_buffer_ms) / 1000),
        )
        max_buffer_bytes = max(
            target_buffer_bytes + frame_bytes,
            int(byte_rate * max(self.config.max_jitter_buffer_ms, self.config.jitter_buffer_ms) / 1000),
        )
        target_buffer_bytes -= target_buffer_bytes % sample_bytes
        max_buffer_bytes -= max_buffer_bytes % sample_bytes
        soft_buffer_bytes = min(
            max_buffer_bytes,
            max(
                target_buffer_bytes + frame_bytes,
                target_buffer_bytes + int(byte_rate * 80 / 1000),
            ),
        )
        soft_buffer_bytes -= soft_buffer_bytes % sample_bytes
        soft_trim_bytes = max(sample_bytes, frame_bytes // 10)
        soft_trim_bytes -= soft_trim_bytes % sample_bytes
        smooth_trim_max_bytes = max(sample_bytes, max(1, frame_sample_count // 240) * sample_bytes)
        max_pacing_lag_s = max(frame_ms * 2, self.config.max_pacing_lag_ms) / 1000
        buffered = bytearray()
        silence = b'\x00' * frame_bytes
        primed = target_buffer_bytes <= frame_bytes
        next_tick = time.monotonic()
        underflows = 0
        dropped_bytes = 0
        last_underflow_log = 0.0
        last_drop_log = 0.0
        last_pacing_log = 0.0
        trimmed_bytes = 0
        smooth_trim_budget_bytes = 0
        metrics_started_at = time.monotonic()
        next_metrics_at = metrics_started_at + self.config.metrics_interval_s
        bytes_written = 0
        frames_written = 0
        slow_drains = 0
        max_drain_ms = 0.0
        transport = getattr(pifm_stdin, 'transport', None)
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.set_write_buffer_limits(
                    high=frame_bytes * 12,
                    low=frame_bytes * 4,
                )
        log.info(
            '%s pacing target=%dms max=%dms frame=%dms frame_bytes=%d',
            label,
            int(self.config.jitter_buffer_ms),
            int(self.config.max_jitter_buffer_ms),
            frame_ms,
            frame_bytes,
        )
        try:
            pifm_stdin.write(_streaming_wav_header(self.config.output_sample_rate, self.config.channels))
            await pifm_stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return 'piFmAdv stdin closed'
        except Exception as exc:
            return f'piFmAdv write failed: {exc}'

        while not self.stop_event.is_set():
            while True:
                try:
                    buffered.extend(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if len(buffered) < frame_bytes:
                try:
                    buffered.extend(await asyncio.wait_for(queue.get(), timeout=frame_ms / 1000))
                except asyncio.TimeoutError:
                    pass
                while True:
                    try:
                        buffered.extend(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

            if not primed and len(buffered) > target_buffer_bytes:
                drop_bytes = len(buffered) - target_buffer_bytes
                drop_bytes -= drop_bytes % sample_bytes
                if drop_bytes > 0:
                    del buffered[:drop_bytes]
                    dropped_bytes += drop_bytes
                    trimmed_bytes += drop_bytes

            if primed and len(buffered) > soft_buffer_bytes:
                excess_bytes = max(0, len(buffered) - soft_buffer_bytes)
                trim_bytes = min(
                    len(buffered) - target_buffer_bytes,
                    max(soft_trim_bytes, excess_bytes),
                )
                trim_bytes -= trim_bytes % sample_bytes
                if trim_bytes > 0:
                    smooth_trim_budget_bytes = max(smooth_trim_budget_bytes, trim_bytes)

            if len(buffered) > max_buffer_bytes:
                drop_bytes = len(buffered) - target_buffer_bytes
                drop_bytes -= drop_bytes % sample_bytes
                if drop_bytes > 0:
                    del buffered[:drop_bytes]
                    dropped_bytes += drop_bytes
                    smooth_trim_budget_bytes = 0
                    now = time.monotonic()
                    if now - last_drop_log >= 5.0:
                        log.warning(
                            '%s buffer overflow; dropped %.0fms total buffered audio',
                            label,
                            dropped_bytes / byte_rate * 1000 if byte_rate else 0,
                    )
                        last_drop_log = now
                    primed = len(buffered) >= target_buffer_bytes

            if not primed and len(buffered) >= target_buffer_bytes:
                primed = True
                next_tick = time.monotonic()

            if primed and len(buffered) >= frame_bytes:
                trim_now = 0
                if smooth_trim_budget_bytes > 0:
                    trim_now = min(
                        smooth_trim_budget_bytes,
                        smooth_trim_max_bytes,
                        max(0, len(buffered) - frame_bytes),
                    )
                    trim_now -= trim_now % sample_bytes
                if trim_now > 0:
                    source_len = frame_bytes + trim_now
                    raw_chunk = bytes(buffered[:source_len])
                    del buffered[:source_len]
                    chunk = _drop_pcm_frames_evenly(raw_chunk, frame_bytes, sample_bytes)
                    smooth_trim_budget_bytes -= trim_now
                    trimmed_bytes += trim_now
                    dropped_bytes += trim_now
                else:
                    chunk = bytes(buffered[:frame_bytes])
                    del buffered[:frame_bytes]
            else:
                if primed:
                    underflows += 1
                    primed = False
                    now = time.monotonic()
                    if now - last_underflow_log >= 5.0:
                        log.warning(
                            '%s buffer underrun; outputting silence while rebuffering with %.0fms held audio after %d underrun(s)',
                            label,
                            len(buffered) / byte_rate * 1000 if byte_rate else 0,
                            underflows,
                        )
                        last_underflow_log = now
                chunk = silence
            try:
                pifm_stdin.write(chunk)
                drain_started = time.monotonic()
                await pifm_stdin.drain()
                drain_ms = (time.monotonic() - drain_started) * 1000
                max_drain_ms = max(max_drain_ms, drain_ms)
                if drain_ms > self.config.max_pacing_lag_ms:
                    slow_drains += 1
                    log.warning('piFmAdv raw stdin drain took %.1f ms', drain_ms)
                bytes_written += len(chunk)
                frames_written += 1
            except (BrokenPipeError, ConnectionResetError):
                return 'piFmAdv stdin closed'
            except Exception as exc:
                return f'piFmAdv write failed: {exc}'

            next_tick += frame_ms / 1000
            now = time.monotonic()
            sleep_for = next_tick - now
            if sleep_for < -max_pacing_lag_s:
                drop_bytes = max(0, len(buffered) - target_buffer_bytes)
                drop_bytes -= drop_bytes % sample_bytes
                if drop_bytes > 0:
                    del buffered[:drop_bytes]
                    dropped_bytes += drop_bytes
                    smooth_trim_budget_bytes = 0
                if now - last_pacing_log >= 5.0:
                    log.warning(
                        '%s writer lagged %.0fms; reset pacing clock and dropped %.0fms buffered audio',
                        label,
                        -sleep_for * 1000,
                        drop_bytes / byte_rate * 1000 if byte_rate else 0,
                    )
                    last_pacing_log = now
                next_tick = now
                sleep_for = 0
            if self.config.metrics_interval_s > 0 and now >= next_metrics_at:
                elapsed = max(0.001, now - metrics_started_at)
                log.info(
                    '%s metrics feed=%s bytes=%d frames=%d elapsed=%.1fs '
                    'buffered_ms=%.0f trimmed_ms=%.0f max_drain_ms=%.1f slow_drains=%d',
                    label,
                    self.config.feed_id,
                    bytes_written,
                    frames_written,
                    elapsed,
                    len(buffered) / byte_rate * 1000 if byte_rate else 0,
                    trimmed_bytes / byte_rate * 1000 if byte_rate else 0,
                    max_drain_ms,
                    slow_drains,
                )
                next_metrics_at = time.monotonic() + self.config.metrics_interval_s
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        return 'shutdown requested'

    def _prefer_receiver_codecs(self, transceiver: Any) -> None:
        if RTCRtpSender is None or not hasattr(transceiver, 'setCodecPreferences'):
            return
        with contextlib.suppress(Exception):
            capabilities = RTCRtpSender.getCapabilities('audio')
            available = list(getattr(capabilities, 'codecs', []) or [])
            ordered_mimes = [f'audio/{codec.lower()}' for codec in self.config.preferred_codecs]
            preferred = []
            for mime in ordered_mimes:
                preferred.extend(
                    codec
                    for codec in available
                    if str(getattr(codec, 'mimeType', '')).lower() == mime
                    and codec not in preferred
                )
            preferred.extend(codec for codec in available if codec not in preferred)
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
            '-filter_threads',
            '1',
            '-nostdin',
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
            '-acodec',
            'pcm_s16le',
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

    async def _start_http_audio_processor(self, transmitter: dict[str, Any]) -> asyncio.subprocess.Process:
        url = self._http_audio_url()
        input_args = []
        if self.config.http_codec.strip().lower().replace('-', '_') in {'raw', 'raw_pcm16', 's16le'}:
            input_args = [
                '-f',
                's16le',
                '-ar',
                '48000',
                '-ac',
                '1',
            ]
        cmd = [
            self.config.ffmpeg_bin,
            '-hide_banner',
            '-nostats',
            '-loglevel',
            self.config.ffmpeg_log_level,
            '-filter_threads',
            '1',
            '-nostdin',
            '-reconnect',
            '1',
            '-reconnect_streamed',
            '1',
            '-reconnect_at_eof',
            '1',
            '-reconnect_delay_max',
            str(self.config.http_reconnect_delay_max_s),
            '-rw_timeout',
            str(int(self.config.http_read_timeout_s * 1_000_000)),
            *input_args,
            '-i',
            url,
            '-af',
            self.config.audio_filters,
            '-ar',
            str(self.config.output_sample_rate),
            '-ac',
            str(self.config.channels),
            '-acodec',
            'pcm_s16le',
            '-f',
            'wav',
            'pipe:1',
        ]
        log.info(
            'Starting HTTP receiver audio processor for %s from %s (%d Hz -> %d Hz)',
            _transmitter_label(transmitter),
            url,
            48000,
            self.config.output_sample_rate,
        )
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
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
            '-filter_threads',
            '1',
            '-re',
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
            '-acodec',
            'pcm_s16le',
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
        queue_frames = max(
            32,
            int(self.config.max_jitter_buffer_ms / max(1, self.config.audio_frame_ms)) * 4,
        )
        queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=queue_frames)
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
        sample_bytes = self.config.channels * 2
        frame_bytes = frame_samples * sample_bytes
        byte_rate = self.config.webrtc_input_sample_rate * sample_bytes
        target_buffer_bytes = max(
            frame_bytes,
            int(byte_rate * max(0, self.config.jitter_buffer_ms) / 1000),
        )
        max_buffer_bytes = max(
            target_buffer_bytes + frame_bytes,
            int(byte_rate * max(self.config.max_jitter_buffer_ms, self.config.jitter_buffer_ms) / 1000),
        )
        target_buffer_bytes -= target_buffer_bytes % sample_bytes
        max_buffer_bytes -= max_buffer_bytes % sample_bytes
        soft_buffer_bytes = min(
            max_buffer_bytes,
            max(
                target_buffer_bytes + frame_bytes,
                target_buffer_bytes + int(byte_rate * 80 / 1000),
            ),
        )
        soft_buffer_bytes -= soft_buffer_bytes % sample_bytes
        soft_trim_bytes = max(sample_bytes, frame_bytes // 10)
        soft_trim_bytes -= soft_trim_bytes % sample_bytes
        smooth_trim_max_bytes = max(sample_bytes, max(1, frame_samples // 240) * sample_bytes)
        max_pacing_lag_s = max(frame_ms * 2, self.config.max_pacing_lag_ms) / 1000
        buffered = bytearray()
        primed = target_buffer_bytes <= frame_bytes
        next_tick = time.monotonic()
        underflows = 0
        dropped_bytes = 0
        smooth_trim_budget_bytes = 0
        last_underflow_log = 0.0
        last_drop_log = 0.0
        last_pacing_log = 0.0
        silence = b'\x00' * frame_bytes
        min_partial_frame_bytes = max(sample_bytes, int(frame_bytes * 3 / 4))
        min_partial_frame_bytes -= min_partial_frame_bytes % sample_bytes
        transport = getattr(ffmpeg_stdin, 'transport', None)
        if transport is not None:
            with contextlib.suppress(Exception):
                transport.set_write_buffer_limits(high=frame_bytes * 12, low=frame_bytes * 4)
        log.info(
            'Receiver audio jitter buffer target=%dms max=%dms frame=%dms pacing_reset=%dms',
            int(self.config.jitter_buffer_ms),
            int(self.config.max_jitter_buffer_ms),
            frame_ms,
            self.config.max_pacing_lag_ms,
        )
        while not self.stop_event.is_set():
            while True:
                try:
                    buffered.extend(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            if len(buffered) < frame_bytes:
                try:
                    buffered.extend(await asyncio.wait_for(queue.get(), timeout=frame_ms / 1000))
                except asyncio.TimeoutError:
                    pass
                while True:
                    try:
                        buffered.extend(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

            if not primed and len(buffered) > target_buffer_bytes:
                drop_bytes = len(buffered) - target_buffer_bytes
                drop_bytes -= drop_bytes % sample_bytes
                if drop_bytes > 0:
                    del buffered[:drop_bytes]
                    dropped_bytes += drop_bytes

            if primed and len(buffered) > soft_buffer_bytes:
                excess_bytes = max(0, len(buffered) - soft_buffer_bytes)
                trim_bytes = min(
                    len(buffered) - target_buffer_bytes,
                    max(soft_trim_bytes, excess_bytes),
                )
                trim_bytes -= trim_bytes % sample_bytes
                if trim_bytes > 0:
                    smooth_trim_budget_bytes = max(smooth_trim_budget_bytes, trim_bytes)

            if len(buffered) > max_buffer_bytes:
                drop_bytes = len(buffered) - target_buffer_bytes
                drop_bytes -= drop_bytes % sample_bytes
                if drop_bytes > 0:
                    del buffered[:drop_bytes]
                    dropped_bytes += drop_bytes
                    smooth_trim_budget_bytes = 0
                    now = time.monotonic()
                    if now - last_drop_log >= 5.0:
                        log.warning(
                            'Receiver audio jitter buffer overflow; dropped %.0fms total buffered audio',
                            dropped_bytes / byte_rate * 1000,
                    )
                        last_drop_log = now
                    primed = len(buffered) >= target_buffer_bytes

            if not primed and len(buffered) >= target_buffer_bytes:
                primed = True
                next_tick = time.monotonic()

            if primed and len(buffered) >= frame_bytes:
                trim_now = 0
                if smooth_trim_budget_bytes > 0:
                    trim_now = min(
                        smooth_trim_budget_bytes,
                        smooth_trim_max_bytes,
                        max(0, len(buffered) - frame_bytes),
                    )
                    trim_now -= trim_now % sample_bytes
                if trim_now > 0:
                    source_len = frame_bytes + trim_now
                    raw_chunk = bytes(buffered[:source_len])
                    del buffered[:source_len]
                    chunk = _drop_pcm_frames_evenly(raw_chunk, frame_bytes, sample_bytes)
                    smooth_trim_budget_bytes -= trim_now
                    dropped_bytes += trim_now
                else:
                    chunk = bytes(buffered[:frame_bytes])
                    del buffered[:frame_bytes]
            else:
                if primed:
                    underflows += 1
                    primed = False
                    now = time.monotonic()
                    if now - last_underflow_log >= 5.0:
                        log.warning(
                            'Receiver audio jitter buffer underrun; concealing while rebuffering with %.0fms held audio after %d underrun(s)',
                            len(buffered) / byte_rate * 1000,
                            underflows,
                        )
                        last_underflow_log = now
                chunk = silence
            try:
                ffmpeg_stdin.write(chunk)
                await ffmpeg_stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return 'ffmpeg stdin closed'
            except Exception as exc:
                return f'ffmpeg write failed: {exc}'
            next_tick += frame_ms / 1000
            now = time.monotonic()
            sleep_for = next_tick - now
            if sleep_for < -max_pacing_lag_s:
                drop_bytes = max(0, len(buffered) - target_buffer_bytes)
                drop_bytes -= drop_bytes % sample_bytes
                if drop_bytes > 0:
                    del buffered[:drop_bytes]
                    dropped_bytes += drop_bytes
                    smooth_trim_budget_bytes = 0
                if now - last_pacing_log >= 5.0:
                    log.warning(
                        'Receiver audio writer lagged %.0fms; reset pacing clock and dropped %.0fms buffered audio',
                        -sleep_for * 1000,
                        drop_bytes / byte_rate * 1000 if byte_rate else 0,
                    )
                    last_pacing_log = now
                next_tick = now
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
            self.last_audio_ts = time.monotonic()
            try:
                pifm_stdin.write(chunk)
                await pifm_stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return 'piFmAdv stdin closed'
            except Exception as exc:
                return f'piFmAdv write failed: {exc}'
        return 'shutdown requested'

    async def _read_processor_pcm(
        self,
        ffmpeg_stdout: asyncio.StreamReader,
        queue: asyncio.Queue[bytes],
    ) -> str:
        while not self.stop_event.is_set():
            chunk = await ffmpeg_stdout.read(self.config.write_chunk_size)
            if not chunk:
                return 'ffmpeg output ended'
            self.last_audio_ts = time.monotonic()
            self._push_audio_chunk(queue, chunk)
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

    def _log_http_control_ws_done(self, task: asyncio.Task[str]) -> None:
        if task.cancelled():
            return
        try:
            reason = task.result()
        except Exception as exc:
            log.warning('HTTP receiver control websocket failed while audio continues: %s', exc)
            return
        if reason:
            log.warning('HTTP receiver control websocket ended while audio continues: %s', reason)

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

    async def _monitor_http_health(
        self,
        ffmpeg_proc: asyncio.subprocess.Process,
        pifm_proc: asyncio.subprocess.Process,
    ) -> str:
        while not self.stop_event.is_set():
            if ffmpeg_proc.returncode is not None:
                return f'ffmpeg exited with code {ffmpeg_proc.returncode}'
            if pifm_proc.returncode is not None:
                return f'piFmAdv exited with code {pifm_proc.returncode}'
            idle_for = time.monotonic() - self.last_audio_ts
            if idle_for >= self.config.stream_stall_timeout_s:
                return f'HTTP receiver audio stalled for {idle_for:.1f}s'
            await asyncio.sleep(1.0)
        return 'shutdown requested'

    async def _monitor_direct_http_health(
        self,
        pifm_proc: asyncio.subprocess.Process,
    ) -> str:
        while not self.stop_event.is_set():
            if pifm_proc.returncode is not None:
                return f'piFmAdv exited with code {pifm_proc.returncode}'
            idle_for = time.monotonic() - self.last_audio_ts
            if idle_for >= self.config.stream_stall_timeout_s:
                return f'HTTP raw receiver audio stalled for {idle_for:.1f}s'
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
    parser.add_argument('--chunk-size', type=int, default=8192)
    parser.add_argument('--audio-frame-ms', type=int, default=20)
    parser.add_argument('--jitter-buffer-ms', type=int, default=320)
    parser.add_argument('--max-jitter-buffer-ms', type=int, default=1200)
    parser.add_argument('--max-pacing-lag-ms', type=int, default=120)
    parser.add_argument('--preferred-codecs', default='g722,opus,pcmu,pcma')
    parser.add_argument('--transport', choices=['auto', 'http', 'webrtc'], default='http')
    parser.add_argument(
        '--allow-webrtc-fallback',
        action='store_true',
        help='Allow --transport auto to fall back to the older WebRTC path if HTTP audio probing fails.',
    )
    parser.add_argument('--http-codec', default='raw_pcm16')
    parser.add_argument('--http-reconnect-delay-max', type=int, default=2)
    parser.add_argument('--http-read-timeout', type=float, default=8.0)
    parser.add_argument('--metrics-interval', type=float, default=10.0, help='Seconds between receiver pipe metric logs; set 0 to disable.')
    parser.add_argument('--diagnose-audio', action='store_true', help='Fetch the HTTP raw PCM stream and print audio integrity metrics without starting piFmAdv.')
    parser.add_argument('--diagnose-duration', type=float, default=60.0, help='Seconds of audio to sample in --diagnose-audio mode.')
    parser.add_argument('--diagnose-output', default='', help='Optional WAV file path to write during --diagnose-audio mode.')
    args = parser.parse_args()

    pair_token = args.pair_token
    if not pair_token and args.pair_token_env:
        pair_token = os.environ.get(args.pair_token_env, '')

    try:
        server_url = _normalize_server_url(args.server, args.allow_insecure_dev)
    except ValueError as exc:
        parser.error(str(exc))

    state_file = pathlib.Path(args.state_file).expanduser() if args.state_file else _default_state_file(args.feed_id)
    audio_frame_ms = max(10, min(100, int(args.audio_frame_ms)))
    jitter_buffer_ms = max(0, int(args.jitter_buffer_ms))
    max_jitter_buffer_ms = max(jitter_buffer_ms + audio_frame_ms, int(args.max_jitter_buffer_ms))
    max_pacing_lag_ms = max(audio_frame_ms * 2, int(args.max_pacing_lag_ms))
    preferred_codecs = tuple(
        codec
        for codec in (
            re.sub(r'[^a-z0-9]+', '', part.strip().lower())
            for part in str(args.preferred_codecs or '').split(',')
        )
        if codec
    )
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
        audio_frame_ms=audio_frame_ms,
        jitter_buffer_ms=jitter_buffer_ms,
        max_jitter_buffer_ms=max_jitter_buffer_ms,
        max_pacing_lag_ms=max_pacing_lag_ms,
        preferred_codecs=preferred_codecs or ('g722', 'opus', 'pcmu', 'pcma'),
        transport=str(args.transport or 'auto'),
        allow_webrtc_fallback=bool(args.allow_webrtc_fallback),
        http_codec=str(args.http_codec or 'raw_pcm16'),
        http_reconnect_delay_max_s=max(1, int(args.http_reconnect_delay_max)),
        http_read_timeout_s=max(2.0, float(args.http_read_timeout)),
        metrics_interval_s=max(0.0, float(args.metrics_interval)),
        diagnose_audio=bool(args.diagnose_audio),
        diagnose_duration_s=max(1.0, float(args.diagnose_duration)),
        diagnose_output=pathlib.Path(args.diagnose_output).expanduser() if args.diagnose_output else None,
    )


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    for name in ('aioice', 'aiortc'):
        logging.getLogger(name).setLevel(logging.ERROR)


async def _diagnose_audio(config: ReceiverConfig) -> None:
    codec = config.http_codec.strip().lower().replace('-', '_')
    if codec not in {'raw', 'raw_pcm16', 's16le'}:
        log.warning('--diagnose-audio uses raw PCM metrics; overriding HTTP codec %s with raw_pcm16', config.http_codec)
        codec = 'raw_pcm16'
    query = urlencode({'feed': config.feed_id, 'codec': codec})
    url = f'{config.server_url}/api/public/v1/feed/audio?{query}'
    timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=5,
        sock_read=max(5.0, config.http_read_timeout_s),
    )
    data = bytearray()
    chunk_count = 0
    max_chunk_gap_ms = 0.0
    first_chunk_at: float | None = None
    last_chunk_at: float | None = None
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                detail = (await response.text())[:300]
                raise RuntimeError(f'{url} returned HTTP {response.status}: {detail}')
            log.info('Diagnosing %s for %.1fs from %s', config.feed_id, config.diagnose_duration_s, url)
            deadline = time.monotonic() + config.diagnose_duration_s
            async for chunk in response.content.iter_chunked(config.write_chunk_size):
                now = time.monotonic()
                if first_chunk_at is None:
                    first_chunk_at = now
                if last_chunk_at is not None:
                    max_chunk_gap_ms = max(max_chunk_gap_ms, (now - last_chunk_at) * 1000)
                last_chunk_at = now
                if chunk:
                    data.extend(chunk)
                    chunk_count += 1
                if now >= deadline:
                    break
    if config.diagnose_output:
        config.diagnose_output.parent.mkdir(parents=True, exist_ok=True)
        config.diagnose_output.write_bytes(_streaming_wav_header(48000, 1) + bytes(data))
        log.info('Wrote diagnostic WAV capture to %s', config.diagnose_output)
    metrics = _pcm16_metrics(bytes(data))
    metrics.update({
        'feed_id': config.feed_id,
        'codec': codec,
        'bytes': len(data),
        'chunks': chunk_count,
        'wall_duration_s': round((last_chunk_at - first_chunk_at), 3) if first_chunk_at and last_chunk_at else 0.0,
        'max_chunk_gap_ms': round(max_chunk_gap_ms, 2),
        'expected_duration_s': config.diagnose_duration_s,
    })
    print(json.dumps(metrics, indent=2, sort_keys=True))


async def _main() -> None:
    _configure_logging()
    config = _parse_args()
    if config.diagnose_audio:
        await _diagnose_audio(config)
        return
    if config.transport == 'webrtc':
        _ensure_webrtc_dependencies()
    supervisor = ReceiverSupervisor(config)
    await supervisor.run_forever()


if __name__ == '__main__':
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass

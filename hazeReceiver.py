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
except Exception as exc:
    av = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    _WEBRTC_IMPORT_ERROR = exc
else:
    _WEBRTC_IMPORT_ERROR = None

pifm_bin = "/home/rai/PiFmAdv/src/pi_fm_adv"

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
        value = f'https://{value}'
        parsed = urlparse(value)
    if parsed.scheme in {'ws', 'wss'}:
        parsed = parsed._replace(scheme='https' if parsed.scheme == 'wss' else 'http')
    if parsed.scheme not in {'http', 'https'}:
        raise ValueError('--server must use http:// or https://')
    if parsed.scheme == 'http' and not allow_insecure_dev:
        raise ValueError('Receiver transport requires HTTPS; use --allow-insecure-dev only for local testing')
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
        _save_state(config.state_file, self.state)

    async def run_forever(self) -> None:
        self._install_signal_handlers()
        delay = self.config.reconnect_initial_delay_s
        while not self.stop_event.is_set():
            try:
                reason = await self._run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                reason = f'receiver failure: {exc}'
            if self.stop_event.is_set():
                break
            log.warning('Receiver reconnect requested: %s', reason)
            await asyncio.sleep(delay)
            delay = min(self.config.reconnect_max_delay_s, delay * self.config.reconnect_backoff)
        log.info('Receiver stopped')

    async def _run_once(self) -> str:
        timeout = aiohttp.ClientTimeout(total=12, sock_connect=4, sock_read=8)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            auth = await self._receiver_auth(session)
            ws_url = str(auth.get('ws_url') or '')
            cookie = str(auth.get('cookie') or '')
            if not ws_url or not cookie:
                return 'receiver session did not return websocket credentials'
            if ws_url.startswith('ws://') and not self.config.allow_insecure_dev:
                return 'receiver websocket URL is insecure'
            async with session.ws_connect(
                ws_url,
                headers={'Authorization': f'HazeReceiverCookie {cookie}'},
                heartbeat=15,
            ) as ws:
                ready_msg = await ws.receive_json()
                if ready_msg.get('type') != 'receiver_ready':
                    return f'unexpected receiver websocket message: {ready_msg.get("type")}'
                transmitter = ready_msg.get('transmitter') if isinstance(ready_msg.get('transmitter'), dict) else {}
                return await self._run_webrtc_session(ws, transmitter)

    async def _receiver_auth(self, session: aiohttp.ClientSession) -> dict[str, Any]:
        credential_id = str(self.state.get('credential_id') or '')
        credential_secret = str(self.state.get('credential_secret') or '')
        if credential_id and credential_secret:
            try:
                return await self._request_session_cookie(session, credential_id, credential_secret)
            except ReceiverHttpError as exc:
                if exc.status in {401, 403, 404}:
                    log.warning('Stored receiver credential was rejected by Haze: %s', exc)
                    self.state.pop('credential_id', None)
                    self.state.pop('credential_secret', None)
                    _save_state(self.config.state_file, self.state)
                else:
                    log.warning('Stored receiver credential could not be checked yet: %s', exc)
                    raise
            except Exception as exc:
                log.warning('Stored receiver credential check failed transiently: %s', exc)
                raise

        if not self.config.pair_token:
            raise RuntimeError('receiver is not paired and no pairing token was supplied')
        return await self._pair_receiver(session)

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
        frequency = transmitter.get('frequency_mhz')
        if frequency is None:
            return 'receiver transmitter parameters did not include frequency_mhz'
        pifm_proc: asyncio.subprocess.Process | None = None
        ffmpeg_proc: asyncio.subprocess.Process | None = None
        pc: RTCPeerConnection | None = None
        try:
            pifm_proc = await self._start_pifmadv(transmitter)
            ffmpeg_proc = await self._start_audio_processor()
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
                    track_future.set_result(track)

            pc.addTransceiver('audio', direction='recvonly')
            offer = await pc.createOffer()
            await pc.setLocalDescription(offer)
            await ws.send_json({
                'type': 'webrtc_offer',
                'sdp': pc.localDescription.sdp,
                'sdp_type': pc.localDescription.type,
            })

            while True:
                msg = await ws.receive_json()
                msg_type = str(msg.get('type') or '')
                if msg_type == 'webrtc_answer':
                    await pc.setRemoteDescription(RTCSessionDescription(sdp=str(msg.get('sdp') or ''), type=str(msg.get('sdp_type') or 'answer')))
                    break
                if msg_type == 'webrtc_error':
                    return f'webrtc negotiation failed: {msg.get("detail")}'

            track = await asyncio.wait_for(track_future, timeout=10.0)
            self.last_audio_ts = time.monotonic()
            pump_audio_task = asyncio.create_task(self._pump_track_to_processor(track, ffmpeg_proc.stdin), name='webrtc_to_ffmpeg')
            pipe_task = asyncio.create_task(self._pump_processor_to_pifm(ffmpeg_proc.stdout, pifm_proc.stdin), name='ffmpeg_to_pifm')
            ffmpeg_err_task = asyncio.create_task(self._log_stream(ffmpeg_proc.stderr, 'ffmpeg'), name='ffmpeg_stderr')
            pifm_err_task = asyncio.create_task(self._log_stream(pifm_proc.stderr, 'piFmAdv'), name='pifm_stderr')
            monitor_task = asyncio.create_task(self._monitor_health(ffmpeg_proc, pifm_proc, pc), name='monitor_health')
            ws_task = asyncio.create_task(self._watch_control_ws(ws), name='receiver_ws')

            done, pending = await asyncio.wait(
                {pump_audio_task, pipe_task, monitor_task, ws_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in (ffmpeg_err_task, pifm_err_task):
                task.cancel()
            for task in (*pending, ffmpeg_err_task, pifm_err_task):
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
            reason = 'receiver session stopped'
            for task in done:
                with contextlib.suppress(asyncio.CancelledError):
                    result = await task
                    if isinstance(result, str) and result:
                        reason = result
            return reason
        finally:
            if pc is not None:
                await pc.close()
            await self._terminate_process(ffmpeg_proc, 'ffmpeg')
            await self._terminate_process(pifm_proc, 'piFmAdv')

    async def _start_audio_processor(self) -> asyncio.subprocess.Process:
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
        log.info('Starting WebRTC audio processor (%d Hz -> %d Hz)', self.config.webrtc_input_sample_rate, self.config.output_sample_rate)
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _start_pifmadv(self, transmitter: dict[str, Any]) -> asyncio.subprocess.Process:
        frequency_mhz = float(transmitter['frequency_mhz'])
        deviation_hz = int(transmitter.get('deviation_hz') or 5000)
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
        if preemphasis == '50':
            cmd.extend(['--preemph', '50us'])
        elif preemphasis == '75':
            cmd.extend(['--preemph', '75us'])
        log.info('Starting piFmAdv transmitter %.3f MHz (%s)', frequency_mhz, self.config.pifmadv_bin)
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _pump_track_to_processor(self, track: Any, ffmpeg_stdin: asyncio.StreamWriter) -> str:
        layout = 'mono' if self.config.channels == 1 else 'stereo'
        resampler = av.AudioResampler(format='s16', layout=layout, rate=self.config.webrtc_input_sample_rate)
        silence_samples = max(1, int(self.config.webrtc_input_sample_rate * 0.02))
        silence = b'\x00' * silence_samples * self.config.channels * 2
        while not self.stop_event.is_set():
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=0.1)
            except asyncio.TimeoutError:
                chunk = silence
            except Exception as exc:
                return f'webrtc audio track ended: {exc}'
            else:
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
                if chunk:
                    self.last_audio_ts = time.monotonic()

            if not chunk:
                continue
            try:
                ffmpeg_stdin.write(chunk)
                await ffmpeg_stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return 'ffmpeg stdin closed'
            except Exception as exc:
                return f'ffmpeg write failed: {exc}'
        return 'shutdown requested'

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

    async def _terminate_process(self, proc: asyncio.subprocess.Process | None, name: str) -> None:
        if proc is None:
            return
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()
        log.info('%s stopped with code %s', name, proc.returncode)

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
        description='Secure Haze feed receiver using pairing, WebSocket signaling, WebRTC audio, and piFmAdv output.',
    )
    parser.add_argument('--server', required=True, help='Haze admin server URL, for example https://haze-host:6444')
    parser.add_argument('--feed-id', required=True, help='Feed ID to receive, for example sk-0001')
    parser.add_argument('--receiver-api-base', default='/api/receiver/v1')
    parser.add_argument('--pair-token', default='')
    parser.add_argument('--pair-token-env', default='')
    parser.add_argument('--state-file', default='')
    parser.add_argument('--allow-insecure-dev', action='store_true')

    parser.add_argument('--output-sample-rate', type=int, default=12000)
    parser.add_argument('--channels', type=int, choices=[1, 2], default=1)
    parser.add_argument('--webrtc-input-sample-rate', type=int, default=48000)
    parser.add_argument('--ffmpeg-bin', default='ffmpeg')
    parser.add_argument('--ffmpeg-log-level', default='warning')
    parser.add_argument(
        '--audio-filters',
        default='agate=threshold=-24dB:range=-40dB:attack=10:release=150,acompressor=threshold=-32dB:ratio=12:attack=5:release=100:makeup=20,highpass=f=120,lowpass=f=2600',
    )
    parser.add_argument('--pifmadv-bin', default=pifm_bin)
    parser.add_argument('--pi-extra-arg', action='append', default=[])

    parser.add_argument('--stall-timeout', type=float, default=12.0)
    parser.add_argument('--reconnect-initial-delay', type=float, default=1.0)
    parser.add_argument('--reconnect-max-delay', type=float, default=8.0)
    parser.add_argument('--reconnect-backoff', type=float, default=1.5)
    parser.add_argument('--chunk-size', type=int, default=4096)
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
    )


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


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

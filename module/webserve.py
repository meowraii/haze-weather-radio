from __future__ import annotations

import asyncio
import dataclasses
import hmac
import hashlib
import html
import io
import json
import logging
import os
from os import path
import pathlib
import queue
import secrets
import socket
import subprocess
import threading
import time
import uuid
import wave
import urllib.parse
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from fastapi import Depends, FastAPI, File, HTTPException, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from module.events import (
    append_runtime_event,
    read_data_pool,
    register_alert_audio_stream,
    register_feed_audio_stream,
    snapshot_alert_queues,
    snapshot_change_versions,
    snapshot_data_pool,
    snapshot_playout_sequences,
    snapshot_runtime,
    store_runtime_alert_entry,
    unregister_alert_audio_stream,
    unregister_feed_audio_stream,
    update_runtime_status,
)
from module.alert_templates import load_alert_templates, merge_alert_templates, write_alert_templates
from module.banner import get_active_alerts, pick_banner_color, pick_banner_gradient, serialize_alert
from module.packages import air_quality_package
from module.alert import feed_same_codes
from module.buffer import CHANNELS, SAMPLE_RATE
from module.scheduler import fire_test as _fire_test
from module.version import app_version, build_metadata
from module.webhook import dispatch_webhook_async

log = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    password: str = ''


class SessionRecord(BaseModel):
    expires_at: float
    created_at: float
    client_ip: str
    user_agent: str = ''


class FileWriteRequest(BaseModel):
    content: str = ''


class SAMAirRequest(BaseModel):
    feed_id: str = ''
    feed_ids: list[str] = []
    originator: str = 'WXR'
    event: str = 'CEM'
    locations: list[str] = []
    duration_hours: int = 1
    duration_minutes: int = 0
    callsign: str = ''
    tone_type: str = 'WXR'
    voice_message: str = ''
    prepend_same_translation: bool = False
    audio_file_path: str = ''
    air_on_all_feeds: bool = False


_ALL_PACKAGES = ('date_time', 'station_id', 'current_conditions', 'forecast', 'air_quality',
                 'climate_summary', 'eccc_discussion', 'geophysical_alert', 'user_bulletin')


_FORMAT_MEDIA_TYPES: dict[str, str] = {
    'raw':      'audio/raw',
    'wav':      'audio/wav',
    'mp3':      'audio/mpeg',
    'ogg':      'audio/ogg',
    'flac':     'audio/flac',
    'aac':      'audio/aac',
    'opus':     'audio/ogg; codecs=opus',
    'ulaw':     'audio/basic',
    'alaw':     'audio/x-alaw',
    'g722':     'audio/G722',
    'webm':     'audio/webm',
    'json':     'application/json',
    'xml':      'application/xml',
    'ssml':     'application/ssml+xml',
    'html':     'text/html; charset=utf-8',
    'markdown': 'text/markdown; charset=utf-8',
    'latex':    'application/x-latex',
}

_FORMAT_FFMPEG_ARGS: dict[str, list[str]] = {
    'mp3':  ['-c:a', 'libmp3lame', '-q:a', '2',           '-f', 'mp3'],
    'ogg':  ['-c:a', 'libvorbis',  '-q:a', '6',           '-f', 'ogg'],
    'flac': ['-c:a', 'flac',                               '-f', 'flac'],
    'aac':  ['-c:a', 'aac',        '-b:a', '128k',        '-f', 'adts'],
    'opus': ['-c:a', 'libopus',    '-b:a', '32k',         '-f', 'ogg'],
    'ulaw': ['-c:a', 'pcm_mulaw',  '-ar',  '8000', '-ac', '1', '-f', 'mulaw'],
    'alaw': ['-c:a', 'pcm_alaw',   '-ar',  '8000', '-ac', '1', '-f', 'alaw'],
    'g722': ['-c:a', 'g722',                               '-f', 'g722'],
    'webm': ['-c:a', 'libopus',    '-b:a', '32k',         '-f', 'webm'],
}

_TEXT_FORMATS: frozenset[str] = frozenset({'json', 'xml', 'ssml', 'html', 'markdown', 'latex'})



class WXGenerateRequest(BaseModel):
    locations: list[str] | str = []
    packages:  list[str] | str = 'all'
    source:    str | None      = None
    lang:      str             = 'en-CA'
    voice:     str | None      = None
    format:    str             = 'raw'


class WebRTCOfferRequest(BaseModel):
    sdp: str
    type: str = 'offer'


class ReceiverPairChallengeRequest(BaseModel):
    feed_id: str
    receiver_id: str
    receiver_hostname: str
    nonce: str


class ReceiverPairCompleteRequest(BaseModel):
    challenge_id: str
    feed_id: str
    receiver_id: str
    receiver_hostname: str
    nonce: str
    proof: str


class ReceiverSessionRequest(BaseModel):
    feed_id: str
    receiver_id: str
    receiver_hostname: str
    credential_id: str
    nonce: str
    proof: str


_MANAGED_ALLOWLIST: frozenset[str] = frozenset({'userbulletins.json', 'dictionary.json', 'packages.py', 'sameMapping.json', 'alertTemplates.xml'})

_WX_SOURCES: frozenset[str] = frozenset({'eccc', 'nws', 'twc'})


def _normalize_wx_source(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized or normalized in {'auto', 'any', 'all'}:
        return None
    aliases = {
        'weather.com': 'twc',
        'weatherdotcom': 'twc',
        'weather_dot_com': 'twc',
    }
    return aliases.get(normalized, normalized)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    if isinstance(value, list):
        for item in cast(list[Any], value):
            if isinstance(item, dict):
                return cast(dict[str, Any], item)
    return {}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in cast(dict[Any, Any], value).items():
            lowered = str(key).lower()
            if any(token in lowered for token in ('password', 'secret', 'token', 'key')):
                redacted[key] = '***'
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in cast(list[Any], value)]
    return value


def _serialize(value: Any, depth: int = 0) -> Any:
    if depth > 5:
        return str(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 600 else value[:597] + '...'
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return _serialize(dataclasses.asdict(cast(Any, value)), depth + 1)
    if isinstance(value, dict):
        return {
            str(key): _serialize(item, depth + 1)
            for key, item in cast(dict[Any, Any], value).items()
        }
    if isinstance(value, (list, tuple, set, deque)):
        return [_serialize(item, depth + 1) for item in cast(list[Any], list(value))]
    if hasattr(value, '__dict__'):
        return _serialize(vars(value), depth + 1)
    return str(value)


def _display_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ('text', 'name', 'value', 'operator_name', 'on_air_name'):
            if key in value:
                resolved = _display_text(value.get(key))
                if resolved:
                    return resolved
        for nested in value.values():
            resolved = _display_text(nested)
            if resolved:
                return resolved
        return ''
    if isinstance(value, (list, tuple)):
        for item in value:
            resolved = _display_text(item)
            if resolved:
                return resolved
        return ''
    return str(value).strip()


def _safe_local_path(value: Any, fallback: str = '/admin') -> str:
    text = str(value or '').strip()
    if not text.startswith('/') or text.startswith('//') or '\\' in text:
        return fallback
    return text


def _tail_lines(path: pathlib.Path, line_count: int) -> list[str]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', errors='replace') as handle:
        return [line.rstrip('\n') for line in deque(handle, maxlen=line_count)]


def _pcm_as_wav(pcm: bytes) -> bytes:
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm)
    return wav_buffer.getvalue()


def _wx_rendered_packages(pkg_lookup: dict[str, str], requested: list[str]) -> list[tuple[str, str]]:
    rendered: list[tuple[str, str]] = []
    for pkg_id in requested:
        text = str(pkg_lookup.get(pkg_id, '') or '')
        if text.strip():
            rendered.append((pkg_id, text))
    return rendered


def _wx_escape_xml_text(value: str) -> str:
    return html.escape(value, quote=False)


def _wx_escape_xml_attr(value: str) -> str:
    return html.escape(value, quote=True)


def _wx_escape_latex(value: str) -> str:
    replacements = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '#': r'\#',
        '$': r'\$',
        '%': r'\%',
        '&': r'\&',
        '_': r'\_',
        '^': r'\textasciicircum{}',
        '~': r'\textasciitilde{}',
    }
    return ''.join(replacements.get(char, char) for char in value)


def _feed_audio_track_class():
    from aiortc import MediaStreamTrack

    class FeedAudioTrack(MediaStreamTrack):
        kind = 'audio'

        def __init__(self, feed_id: str):
            super().__init__()
            self.feed_id = feed_id
            self._queue = register_feed_audio_stream(feed_id)
            self._sample_rate = SAMPLE_RATE
            self._frame_samples = max(1, int(self._sample_rate * 0.02))
            self._frame_bytes = self._frame_samples * CHANNELS * 2
            self._buffer = bytearray()
            self._timestamp = 0
            self._time_base = None
            self._layout = 'stereo' if CHANNELS == 2 else 'mono'
            self._started_at: float | None = None
            self._jitter_wait_s = 0.04

        async def recv(self):
            import av
            from fractions import Fraction

            if self._time_base is None:
                self._time_base = Fraction(1, self._sample_rate)
            loop = asyncio.get_running_loop()
            if self._started_at is None:
                self._started_at = loop.time()

            target_time = self._started_at + (self._timestamp / self._sample_rate)
            delay = target_time - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)

            while len(self._buffer) < self._frame_bytes:
                try:
                    pcm, _label = self._queue.get_nowait()
                except queue.Empty:
                    try:
                        pcm, _label = await asyncio.to_thread(self._queue.get, True, self._jitter_wait_s)
                    except queue.Empty:
                        break
                if pcm:
                    self._buffer.extend(pcm)

            if len(self._buffer) < self._frame_bytes:
                self._buffer.extend(b'\x00' * (self._frame_bytes - len(self._buffer)))

            chunk = bytes(self._buffer[:self._frame_bytes])
            del self._buffer[:self._frame_bytes]

            frame = av.AudioFrame(format='s16', layout=self._layout, samples=self._frame_samples)
            frame.sample_rate = self._sample_rate
            frame.pts = self._timestamp
            frame.time_base = self._time_base
            frame.planes[0].update(chunk)
            self._timestamp += self._frame_samples
            return frame

        def stop(self) -> None:
            unregister_feed_audio_stream(self.feed_id, self._queue)
            super().stop()

    return FeedAudioTrack


FeedAudioTrack = None
try:
    FeedAudioTrack = _feed_audio_track_class()
except Exception:
    FeedAudioTrack = None


def _ensure_feed_audio_track() -> Any:
    global FeedAudioTrack
    if FeedAudioTrack is not None:
        return FeedAudioTrack
    try:
        FeedAudioTrack = _feed_audio_track_class()
    except Exception:
        FeedAudioTrack = None
    return FeedAudioTrack


class WebServer:
    _shared_sessions: dict[str, SessionRecord] = {}
    _shared_token_lock = threading.Lock()
    _login_attempts: dict[str, deque[float]] = {}
    _session_cookie = 'haze_admin_session'
    _receiver_lock = threading.Lock()
    _receiver_challenges: dict[str, dict[str, Any]] = {}
    _receiver_cookies: dict[str, dict[str, Any]] = {}

    def __init__(
        self,
        config: dict[str, Any],
        feeds: list[dict[str, Any]] | None = None,
        *,
        mode: str = 'admin',
        started_at: float | None = None,
    ):
        self.config = config
        self.feeds = feeds or [f for f in config.get('feeds', []) if f.get('enabled', True)]
        self.mode = mode
        self.started_at = started_at if started_at is not None else time.time()
        self.root_dir = pathlib.Path(__file__).resolve().parent.parent
        self.webroot = self.root_dir / 'webroot'
        self._token_ttl_s = int(self._auth_cfg().get('session_ttl_seconds') or 12 * 60 * 60)
        self._peer_connections: set[Any] = set()
        self._config_payload_cache: dict[str, Any] | None = None
        self._feed_static_cache: list[dict[str, Any]] | None = None
        self._feed_static_signature: str = ''

        title = 'Haze Weather Radio Public' if self.mode == 'public' else 'Haze Weather Radio Panel'
        self.app = FastAPI(title=title, version=app_version(config))
        self._configure_security_middleware()
        self._configure_cors()
        self.app.mount('/assets', StaticFiles(directory=str(self.webroot)), name='assets')
        self._register_routes()

    def _panel_cfg(self) -> dict[str, Any]:
        return _coerce_mapping(self.config.get('webpanel', self.config.get('web', {})))

    def _admin_cfg(self) -> dict[str, Any]:
        panel_cfg = self._panel_cfg()
        cfg = _coerce_mapping(panel_cfg.get('admin', {}))
        if 'enabled' not in cfg:
            cfg['enabled'] = bool(panel_cfg.get('enabled', False))
        if not cfg.get('host'):
            cfg['host'] = panel_cfg.get('host') or '0.0.0.0'
        if not cfg.get('port'):
            cfg['port'] = panel_cfg.get('port') or 6444
        return cfg

    def _public_cfg(self) -> dict[str, Any]:
        panel_cfg = self._panel_cfg()
        cfg = _coerce_mapping(panel_cfg.get('public', {}))
        if 'enabled' not in cfg:
            cfg['enabled'] = bool(panel_cfg.get('enabled', False))
        if not cfg.get('host'):
            cfg['host'] = '0.0.0.0'
        if not cfg.get('port'):
            cfg['port'] = 8080
        return cfg

    def _auth_cfg(self) -> dict[str, Any]:
        cfg = _coerce_mapping(self._panel_cfg().get('authentication', {}))
        if not cfg.get('password') and os.environ.get('ADMIN_PASSWD'):
            cfg['password'] = os.environ.get('ADMIN_PASSWD')
        if not cfg.get('password_hash') and os.environ.get('ADMIN_PASSWD_HASH'):
            cfg['password_hash'] = os.environ.get('ADMIN_PASSWD_HASH')
        if bool(cfg.get('enabled', True)) and not cfg.get('password') and not cfg.get('password_hash'):
            generated = secrets.token_urlsafe(18)
            cfg['password'] = generated
            log.warning('Generated one-time admin panel password for this process: %s', generated)
        return cfg

    def _public_site_name(self) -> str:
        cfg = self._public_cfg()
        return str(cfg.get('site_name') or 'Haze Weather Radio').strip() or 'Haze Weather Radio'

    def _public_feeds_cfg(self) -> dict[str, Any]:
        return _coerce_mapping(self._public_cfg().get('feeds', {}))

    def _public_feeds_access(self) -> str:
        access = str(self._public_feeds_cfg().get('access') or 'disabled').strip().lower()
        return access if access in {'disabled', 'public', 'auth_required'} else 'disabled'

    def _public_webrtc_cfg(self) -> dict[str, Any]:
        return _coerce_mapping(self._public_feeds_cfg().get('webrtc', {}))

    def _receiver_cfg(self) -> dict[str, Any]:
        cfg = _coerce_mapping(self._panel_cfg().get('receiver', {}))
        if 'enabled' not in cfg:
            cfg['enabled'] = False
        if not cfg.get('base_path'):
            cfg['base_path'] = '/api/receiver/v1'
        if 'require_tls' not in cfg:
            cfg['require_tls'] = True
        return cfg

    def _receiver_enabled(self) -> bool:
        return bool(self._receiver_cfg().get('enabled', False))

    def _receiver_api_base(self) -> str:
        base_path = str(self._receiver_cfg().get('base_path') or '/api/receiver/v1')
        return '/' + base_path.strip('/')

    def _receiver_challenge_ttl_s(self) -> float:
        return max(5.0, float(self._receiver_cfg().get('challenge_ttl_seconds') or 60))

    def _receiver_cookie_ttl_s(self) -> float:
        return max(5.0, float(self._receiver_cfg().get('cookie_ttl_seconds') or 30))

    def _receiver_credential_ttl_s(self) -> float:
        return max(60.0, float(self._receiver_cfg().get('credential_ttl_seconds') or 365 * 24 * 60 * 60))

    def _receiver_credentials_path(self) -> pathlib.Path:
        configured = str(self._receiver_cfg().get('credentials_path') or '').strip()
        if configured:
            candidate = pathlib.Path(configured)
            return candidate if candidate.is_absolute() else (self.root_dir / candidate).resolve()
        return self.root_dir / 'managed' / 'receiver_credentials.json'

    def _receiver_request_secure(self, request: Request) -> bool:
        forwarded = str(request.headers.get('X-Forwarded-Proto') or '').split(',')[0].strip().lower()
        return request.url.scheme == 'https' or forwarded == 'https'

    def _receiver_websocket_secure(self, websocket: WebSocket) -> bool:
        forwarded = str(websocket.headers.get('X-Forwarded-Proto') or '').split(',')[0].strip().lower()
        return websocket.url.scheme == 'wss' or forwarded == 'https'

    def _require_receiver_https(self, request: Request) -> None:
        if bool(self._receiver_cfg().get('require_tls', True)) and not self._receiver_request_secure(request):
            raise HTTPException(status_code=403, detail='Receiver API requires HTTPS')

    def _receiver_store_load(self) -> dict[str, Any]:
        path = self._receiver_credentials_path()
        if not path.exists():
            return {'credentials': {}, 'consumed_pair_token_digests': {}}
        try:
            with path.open('r', encoding='utf-8') as handle:
                data = json.load(handle)
        except Exception:
            log.warning('Receiver credential store is unreadable; starting with an empty store')
            return {'credentials': {}, 'consumed_pair_token_digests': {}}
        if not isinstance(data, dict):
            return {'credentials': {}, 'consumed_pair_token_digests': {}}
        data.setdefault('credentials', {})
        data.setdefault('consumed_pair_token_digests', {})
        return data

    def _receiver_store_save(self, data: dict[str, Any]) -> None:
        path = self._receiver_credentials_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        with tmp.open('w', encoding='utf-8') as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write('\n')
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass

    def _receiver_pairing_tokens(self) -> list[dict[str, Any]]:
        raw_tokens = self._receiver_cfg().get('pairing_tokens') or self._receiver_cfg().get('tokens') or []
        if isinstance(raw_tokens, dict):
            raw_tokens = [raw_tokens]
        tokens: list[dict[str, Any]] = []
        for index, item in enumerate(raw_tokens if isinstance(raw_tokens, list) else []):
            if not isinstance(item, dict) or not bool(item.get('enabled', True)):
                continue
            token = str(item.get('token') or '').strip()
            token_env = str(item.get('token_env') or '').strip()
            if not token and token_env:
                token = str(os.environ.get(token_env) or '').strip()
            if not token:
                continue
            feed_ids_raw = item.get('feed_ids') or item.get('feeds') or []
            if isinstance(feed_ids_raw, str):
                feed_ids = {part.strip() for part in feed_ids_raw.split(',') if part.strip()}
            elif isinstance(feed_ids_raw, list):
                feed_ids = {str(part).strip() for part in feed_ids_raw if str(part).strip()}
            else:
                feed_ids = set()
            tokens.append({
                'id': str(item.get('id') or f'token-{index + 1}'),
                'token': token,
                'feed_ids': feed_ids,
                'digest': self._token_digest(token),
            })
        return tokens

    @staticmethod
    def _receiver_proof_message(kind: str, values: dict[str, Any]) -> bytes:
        payload = {'kind': kind, **{str(k): str(v) for k, v in values.items()}}
        return json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')

    @staticmethod
    def _receiver_hmac(secret: str, message: bytes) -> str:
        return hmac.new(secret.encode('utf-8'), message, hashlib.sha256).hexdigest()

    def _feed_by_id(self, feed_id: str) -> dict[str, Any] | None:
        wanted = str(feed_id or '').strip()
        return next((feed for feed in self.config.get('feeds', []) if str(feed.get('id') or '') == wanted), None)

    def _receiver_transmitter_params(self, feed: dict[str, Any]) -> dict[str, Any]:
        defaults = _coerce_mapping(self._receiver_cfg().get('transmitter_defaults', {}))
        transmitter = self._feed_transmitter_summary(feed)
        frequency = transmitter.get('frequency_mhz')
        return {
            'feed_id': str(feed.get('id') or ''),
            'feed_name': str(feed.get('name') or feed.get('id') or ''),
            'site_name': transmitter.get('site_name') or '',
            'site_names': transmitter.get('site_names') or [],
            'callsign': transmitter.get('callsign') or '',
            'frequency_mhz': frequency,
            'bandwidth_khz': float(defaults.get('bandwidth_khz') or 12.5),
            'deviation_hz': int(defaults.get('deviation_hz') or 5000),
            'preemphasis': str(defaults.get('preemphasis') or 'none'),
            'sample_rate': SAMPLE_RATE,
            'channels': CHANNELS,
        }

    def _make_receiver_cookie(self, receiver_hostname: str) -> str:
        timestamp_ms = int(time.time() * 1000) & ((1 << 48) - 1)
        receiver_hash = int.from_bytes(hashlib.sha256(receiver_hostname.encode('utf-8')).digest()[:2], 'big') & 0x0FFF
        server_hash = int.from_bytes(hashlib.sha256(socket.gethostname().encode('utf-8')).digest()[:2], 'big') & 0x3FFF
        random_tail = secrets.randbits(48)
        value = (
            (timestamp_ms << 80)
            | (0x8 << 76)
            | (receiver_hash << 64)
            | (0b10 << 62)
            | (server_hash << 48)
            | random_tail
        )
        return str(uuid.UUID(int=value))

    def _issue_receiver_cookie(self, credential_id: str, feed_id: str, receiver_id: str, receiver_hostname: str) -> tuple[str, str]:
        cookie = self._make_receiver_cookie(receiver_hostname)
        expires_at = time.time() + self._receiver_cookie_ttl_s()
        digest = self._token_digest(cookie)
        with self._receiver_lock:
            self._receiver_cookies[digest] = {
                'credential_id': credential_id,
                'feed_id': feed_id,
                'receiver_id': receiver_id,
                'receiver_hostname': receiver_hostname,
                'expires_at': expires_at,
            }
        return cookie, datetime.fromtimestamp(expires_at, timezone.utc).isoformat()

    def _consume_receiver_cookie(self, cookie: str | None) -> dict[str, Any] | None:
        if not cookie:
            return None
        digest = self._token_digest(cookie)
        now = time.time()
        with self._receiver_lock:
            expired = [key for key, value in self._receiver_cookies.items() if float(value.get('expires_at') or 0) <= now]
            for key in expired:
                self._receiver_cookies.pop(key, None)
            record = self._receiver_cookies.pop(digest, None)
        if not record or float(record.get('expires_at') or 0) <= now:
            return None
        return record

    def _receiver_ws_url(self, request: Request) -> str:
        forwarded = str(request.headers.get('X-Forwarded-Proto') or '').split(',')[0].strip().lower()
        scheme = 'wss' if request.url.scheme == 'https' or forwarded == 'https' else 'ws'
        host = request.headers.get('host') or request.url.hostname or '127.0.0.1'
        return f'{scheme}://{host}{self._receiver_api_base()}/ws'

    def _api_base(self) -> str:
        rest_cfg = _coerce_mapping(self._panel_cfg().get('rest_api', {}))
        base_path = str(rest_cfg.get('base_path') or '/api/v1')
        return '/' + base_path.strip('/')

    def _public_api_base(self) -> str:
        return '/api/public/v1'

    def _wx_base(self) -> str:
        wx_cfg = _coerce_mapping(self.config.get('wx_on_demand', {}))
        base = str(wx_cfg.get('endpoint-base') or '/api/wx-on-demand/v1')
        return '/' + base.strip('/')

    def _auth_enabled(self) -> bool:
        return bool(self._auth_cfg().get('enabled', True))

    def _secure_cookie(self, request: Request | None = None) -> bool:
        if bool(self._auth_cfg().get('secure_cookies', False)):
            return True
        return bool(request and request.url.scheme == 'https')

    def _configure_security_middleware(self) -> None:
        @self.app.middleware('http')
        async def security_middleware(request: Request, call_next):
            if self.mode == 'admin' and request.method not in {'GET', 'HEAD', 'OPTIONS'}:
                origin = request.headers.get('Origin') or request.headers.get('Referer')
                if origin:
                    parsed_origin = urllib.parse.urlparse(origin)
                    expected_host = request.headers.get('host', '')
                    origin_host = parsed_origin.netloc
                    if origin_host and not hmac.compare_digest(origin_host.lower(), expected_host.lower()):
                        return JSONResponse({'detail': 'Cross-origin request blocked'}, status_code=403)
            response = await call_next(request)
            response.headers.setdefault('X-Content-Type-Options', 'nosniff')
            response.headers.setdefault('X-Frame-Options', 'DENY')
            response.headers.setdefault('Referrer-Policy', 'same-origin')
            response.headers.setdefault('Permissions-Policy', 'camera=(), microphone=(self), geolocation=()')
            response.headers.setdefault(
                'Content-Security-Policy',
                "default-src 'self'; "
                "script-src 'self' https://unpkg.com 'unsafe-inline'; "
                "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data:; "
                "media-src 'self' blob:; "
                "connect-src 'self' ws: wss:; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'",
            )
            if request.url.scheme == 'https':
                response.headers.setdefault('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
            if request.url.path.startswith('/assets/'):
                response.headers['Cache-Control'] = 'no-store'
            return response

    def _configure_cors(self) -> None:
        cors_cfg = _coerce_mapping(self._panel_cfg().get('cors', {}))
        origins = cors_cfg.get('allow_origins') or []
        methods = cors_cfg.get('allow_methods') or ['GET', 'POST', 'PUT']
        headers = cors_cfg.get('allow_headers') or ['*']
        if origins:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_methods=methods,
                allow_headers=headers,
                allow_credentials=True,
            )

    @staticmethod
    def _token_digest(token: str) -> str:
        return hashlib.sha256(token.encode('utf-8')).hexdigest()

    def _issue_token(self, request: Request | None = None, websocket: WebSocket | None = None) -> str:
        token = secrets.token_urlsafe(32)
        digest = self._token_digest(token)
        now = time.time()
        headers = request.headers if request is not None else websocket.headers if websocket is not None else {}
        with self._shared_token_lock:
            self._shared_sessions[digest] = SessionRecord(
                expires_at=now + self._token_ttl_s,
                created_at=now,
                client_ip=self._client_host(request=request, websocket=websocket),
                user_agent=str(headers.get('User-Agent', ''))[:240],
            )
        return token

    def _prune_tokens(self) -> None:
        now = time.time()
        with self._shared_token_lock:
            expired = [digest for digest, session in self._shared_sessions.items() if session.expires_at <= now]
            for digest in expired:
                self._shared_sessions.pop(digest, None)

    def _check_token(self, token: str | None) -> bool:
        if not token:
            return False
        self._prune_tokens()
        digest = self._token_digest(token)
        with self._shared_token_lock:
            session = self._shared_sessions.get(digest)
            return bool(session and session.expires_at > time.time())

    def _revoke_token(self, token: str | None) -> None:
        if not token:
            return
        with self._shared_token_lock:
            self._shared_sessions.pop(self._token_digest(token), None)

    def _request_token(self, request: Request) -> str:
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            return auth_header[7:].strip()
        return (
            request.cookies.get(self._session_cookie)
            or request.headers.get('X-Session-Token')
            or request.query_params.get('token')
            or ''
        )

    def _check_request_auth(self, request: Request) -> bool:
        if not self._auth_enabled():
            return True
        return self._check_token(self._request_token(request))

    def _set_session_cookie(self, response: Response, token: str, request: Request) -> None:
        response.set_cookie(
            self._session_cookie,
            token,
            max_age=self._token_ttl_s,
            expires=self._token_ttl_s,
            httponly=True,
            secure=self._secure_cookie(request),
            samesite='lax',
            path='/',
        )

    def _clear_session_cookie(self, response: Response, request: Request | None = None) -> None:
        response.delete_cookie(
            self._session_cookie,
            httponly=True,
            secure=self._secure_cookie(request),
            samesite='lax',
            path='/',
        )

    def _verify_password(self, submitted: str) -> bool:
        auth_cfg = self._auth_cfg()
        password_hash = str(auth_cfg.get('password_hash') or '').strip()
        if password_hash:
            try:
                from pwdlib import PasswordHash

                return bool(PasswordHash.recommended().verify(submitted, password_hash))
            except Exception:
                log.exception('Unable to verify configured admin password hash')
                return False
        configured_password = str(auth_cfg.get('password') or '')
        if not configured_password:
            return True
        return hmac.compare_digest(submitted, configured_password)

    def _login_allowed(self, ip: str) -> bool:
        auth_cfg = self._auth_cfg()
        window_s = float(auth_cfg.get('login_rate_window_seconds') or 300)
        max_attempts = int(auth_cfg.get('login_rate_limit') or 8)
        now = time.time()
        with self._shared_token_lock:
            attempts = self._login_attempts.setdefault(ip, deque())
            while attempts and attempts[0] <= now - window_s:
                attempts.popleft()
            if len(attempts) >= max_attempts:
                return False
            attempts.append(now)
            return True

    def _clear_login_attempts(self, ip: str) -> None:
        with self._shared_token_lock:
            self._login_attempts.pop(ip, None)

    async def _require_auth(self, request: Request) -> None:
        if not self._check_request_auth(request):
            raise HTTPException(status_code=401, detail='Authentication required')

    async def _require_public_feed_access(self, request: Request) -> None:
        access = self._public_feeds_access()
        if access == 'disabled':
            raise HTTPException(status_code=404, detail='Public feed streaming disabled')
        if access == 'auth_required':
            await self._require_auth(request)

    def _client_host(self, request: Request | None = None, websocket: WebSocket | None = None) -> str:
        client = request.client if request is not None else websocket.client if websocket is not None else None
        return str(client.host) if client and getattr(client, 'host', None) else 'unknown'

    def _record_admin_connection(self, ip: str) -> None:
        update_runtime_status({
            'webpanel_last_connected_ip': ip,
            'webpanel_last_connected_at': datetime.now(timezone.utc).isoformat(),
        })

    async def _require_websocket_auth(self, websocket: WebSocket) -> bool:
        if not self._auth_enabled():
            return True
        auth_header = str(websocket.headers.get('Authorization') or '').strip()
        bearer_token = auth_header[7:].strip() if auth_header.startswith('Bearer ') else ''
        token = (
            websocket.cookies.get(self._session_cookie)
            or bearer_token
            or str(websocket.headers.get('X-Session-Token') or '').strip()
            or websocket.query_params.get('token')
        )
        if not self._check_token(token):
            await websocket.close(code=1008)
            return False
        return True

    def _config_payload(self) -> dict[str, Any]:
        if self._config_payload_cache is None:
            self._config_payload_cache = {
                'operator': _redact(self.config.get('operator', {})),
                'playout': _redact(self.config.get('playout', {})),
                'same': _redact(self.config.get('same', {})),
                'cap': _redact(self.config.get('cap', {})),
                'feeds': _redact(self.config.get('feeds', [])),
                'tts': _redact(self.config.get('tts', {})),
                'webpanel': _redact(self._panel_cfg()),
            }
        return self._config_payload_cache

    def _invalidate_panel_caches(self) -> None:
        self._config_payload_cache = None
        self._feed_static_cache = None
        self._feed_static_signature = ''

    def _feed_output_modes(self, feed: dict[str, Any]) -> list[str]:
        output_cfg = feed.get('output', {})
        enabled: list[str] = []
        for key in ('stream', 'audio_device', 'file', 'udp', 'rtp', 'rtmp', 'srt', 'rtsp'):
            if _coerce_mapping(output_cfg.get(key, {})).get('enabled'):
                enabled.append(key)
        return enabled

    def _feed_transmitter_summary(self, feed: dict[str, Any]) -> dict[str, Any]:
        tx_meta = feed.get('transmitter_metadata')
        primary: dict[str, Any] = {}
        site_names: list[str] = []
        if isinstance(tx_meta, list):
            for item in tx_meta:
                if not isinstance(item, dict):
                    continue
                site_name = str(item.get('site_name') or '').strip()
                if site_name and site_name not in site_names:
                    site_names.append(site_name)
            primary = next((item for item in tx_meta if isinstance(item, dict) and item.get('relationship') == 'primary'), {})
            if not primary:
                primary = next((item for item in tx_meta if isinstance(item, dict)), {})
        fallback_site = str(primary.get('site_name') or feed.get('name') or feed.get('id') or '').strip()
        if fallback_site and fallback_site not in site_names:
            site_names.insert(0, fallback_site)
        return {
            'callsign': str(primary.get('callsign') or feed.get('callsign') or '').strip(),
            'site_name': fallback_site,
            'site_names': site_names,
            'frequency_mhz': primary.get('frequency_mhz'),
        }

    def _operator_name(self) -> str:
        operator = self.config.get('operator', {})
        if isinstance(operator, dict):
            return (
                _display_text(operator.get('operator_name'))
                or _display_text(operator.get('name'))
                or _display_text(operator.get('on_air_name'))
            )
        return _display_text(operator)

    def _on_air_name(self) -> str:
        operator = self.config.get('operator', {})
        if isinstance(operator, dict):
            return (
                _display_text(operator.get('on_air_name'))
                or _display_text(operator.get('name'))
            )
        return _display_text(operator)

    def _public_admin_url(self, request: Request | None = None) -> str:
        admin_cfg = self._admin_cfg()
        host = str(admin_cfg.get('host') or '127.0.0.1').strip()
        port = int(admin_cfg.get('port') or 6444)
        scheme = request.url.scheme if request is not None else 'http'
        if host in {'0.0.0.0', '::', ''}:
            host = request.url.hostname if request is not None and request.url.hostname else '127.0.0.1'
        return f'{scheme}://{host}:{port}/admin'

    def _public_site_url(self, request: Request | None = None) -> str:
        public_cfg = self._public_cfg()
        host = str(public_cfg.get('host') or '127.0.0.1').strip()
        port = int(public_cfg.get('port') or 8080)
        scheme = request.url.scheme if request is not None else 'http'
        if host in {'0.0.0.0', '::', ''}:
            host = request.url.hostname if request is not None and request.url.hostname else '127.0.0.1'
        return f'{scheme}://{host}:{port}'

    def _feed_static_rows(self) -> list[dict[str, Any]]:
        feeds_config = [feed for feed in self.config.get('feeds', []) if isinstance(feed, dict)]
        signature = hashlib.sha1(
            json.dumps(feeds_config, sort_keys=True, default=str, separators=(',', ':')).encode('utf-8')
        ).hexdigest()
        if self._feed_static_cache is not None and self._feed_static_signature == signature:
            return self._feed_static_cache

        rows: list[dict[str, Any]] = []
        for feed in feeds_config:
            feed_id = feed.get('id', 'unknown')
            clc_codes = feed_same_codes(feed)
            location_count = 0
            for block in feed.get('locations', []):
                if not isinstance(block, dict):
                    continue
                location_count += len([entry for entry in block.get('observationLocations', []) if isinstance(entry, dict)])
                location_count += len([entry for entry in block.get('forecastLocations', []) if isinstance(entry, dict)])
            rows.append({
                'id': feed_id,
                'name': feed.get('name', feed_id),
                'enabled': bool(feed.get('enabled', True)),
                'timezone': feed.get('timezone', 'UTC'),
                'languages': list(feed.get('languages', {}).keys()) or [feed.get('language', 'en-CA')],
                'location_count': location_count,
                'outputs': self._feed_output_modes(feed),
                'clc_codes': clc_codes,
                'transmitter': self._feed_transmitter_summary(feed),
            })

        self._feed_static_cache = rows
        self._feed_static_signature = signature
        return rows

    def _feeds_payload(
        self,
        *,
        runtime_feeds: dict[str, Any] | None = None,
        sequences: dict[str, list[Any]] | None = None,
        queue_depths: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        runtime = runtime_feeds if runtime_feeds is not None else snapshot_runtime().get('feeds', {})
        sequences = sequences if sequences is not None else snapshot_playout_sequences()
        queue_depths = queue_depths if queue_depths is not None else snapshot_alert_queues()

        feeds: list[dict[str, Any]] = []
        for row in self._feed_static_rows():
            feed_id = str(row.get('id') or 'unknown')
            sequence = [item.pkg_id for item in sequences.get(feed_id, [])]
            feed_payload = dict(row)
            feed_payload.update({
                'playlist_items': sequence,
                'playlist_count': len(sequence),
                'alert_queue_depth': queue_depths.get(feed_id, 0),
                'runtime': _serialize(runtime.get(feed_id, {})),
            })
            feeds.append(feed_payload)
        return feeds

    def _summary_payload(
        self,
        *,
        runtime: dict[str, Any] | None = None,
        data_pool: dict[str, Any] | None = None,
        sequences: dict[str, list[Any]] | None = None,
        queue_depths: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        runtime = runtime if runtime is not None else snapshot_runtime()
        data_pool = data_pool if data_pool is not None else snapshot_data_pool()
        feeds = self._feeds_payload(
            runtime_feeds=_coerce_mapping(runtime.get('feeds', {})),
            sequences=sequences,
            queue_depths=queue_depths,
        )
        return {
            'name': 'Haze Weather Radio',
            'started_at': datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            'uptime_seconds': round(time.time() - self.started_at, 1),
            'shutdown_requested': bool(runtime.get('system', {}).get('shutdown_requested', False)),
            'auth_required': self._auth_enabled(),
            'feed_count': len(feeds),
            'enabled_feed_count': sum(1 for feed in feeds if feed['enabled']),
            'data_pool_key_count': len(data_pool),
            'feeds': feeds,
            'recent_events': runtime.get('events', [])[-12:],
        }

    def _public_feed_payload(
        self,
        *,
        runtime_feeds: dict[str, Any] | None = None,
        sequences: dict[str, list[Any]] | None = None,
        queue_depths: dict[str, int] | None = None,
    ) -> list[dict[str, Any]]:
        runtime = runtime_feeds if runtime_feeds is not None else snapshot_runtime().get('feeds', {})
        sequences = sequences if sequences is not None else snapshot_playout_sequences()
        queue_depths = queue_depths if queue_depths is not None else snapshot_alert_queues()
        payload: list[dict[str, Any]] = []

        feeds_by_id = {
            str(feed.get('id') or 'unknown'): feed
            for feed in self.config.get('feeds', [])
            if isinstance(feed, dict)
        }
        for row in self._feed_static_rows():
            feed_id = str(row.get('id') or 'unknown')
            feed = feeds_by_id.get(feed_id, {})
            feed_runtime = _coerce_mapping(runtime.get(feed_id, {}))
            transmitter = _coerce_mapping(row.get('transmitter', {}))
            payload.append({
                'id': feed_id,
                'name': str(row.get('name') or feed_id),
                'enabled': bool(row.get('enabled', True)),
                'timezone': str(row.get('timezone') or 'UTC'),
                'languages': row.get('languages') or [feed.get('language', 'en-CA')],
                'outputs': row.get('outputs') or [],
                'transmitter': transmitter,
                'queue_depth': len(sequences.get(feed_id, [])),
                'alert_queue_depth': queue_depths.get(feed_id, 0),
                'recent_items': list(feed_runtime.get('public_stream_recent_items') or []),
                'queue_stream': {
                    'now_playing': feed_runtime.get('public_stream_now_playing') or 'Idle',
                    'started_at': feed_runtime.get('public_stream_started_at'),
                    'queue_depth': feed_runtime.get('public_stream_queue_depth', len(sequences.get(feed_id, []))),
                },
                'on_air': {
                    'now_playing': feed_runtime.get('on_air_now_playing') or feed_runtime.get('now_playing') or 'Idle',
                    'last_played_at': feed_runtime.get('on_air_last_played_at') or feed_runtime.get('last_played_at'),
                },
            })
        return payload

    def _public_summary_payload(
        self,
        request: Request | None = None,
        include_feeds: bool = True,
        *,
        runtime: dict[str, Any] | None = None,
        sequences: dict[str, list[Any]] | None = None,
        queue_depths: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        runtime = runtime if runtime is not None else snapshot_runtime()
        feeds = self._public_feed_payload(
            runtime_feeds=_coerce_mapping(runtime.get('feeds', {})),
            sequences=sequences,
            queue_depths=queue_depths,
        ) if include_feeds else []
        configured_feeds = [feed for feed in self.config.get('feeds', []) if isinstance(feed, dict)]
        return {
            'name': self._public_site_name(),
            'hostname': socket.gethostname(),
            'operator': self._operator_name(),
            'started_at': datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            'uptime_seconds': round(time.time() - self.started_at, 1),
            'shutdown_requested': bool(runtime.get('system', {}).get('shutdown_requested', False)),
            'feed_count': len(configured_feeds),
            'enabled_feed_count': sum(1 for feed in configured_feeds if bool(feed.get('enabled', True))),
            'feeds_access': self._public_feeds_access(),
            'admin_url': self._public_admin_url(request),
            'webrtc_enabled': bool(self._public_webrtc_cfg().get('enabled', True)) and _ensure_feed_audio_track() is not None,
            'feeds': feeds,
        }

    def _datapool_payload(self, data_pool: dict[str, Any] | None = None) -> dict[str, Any]:
        data_pool = data_pool if data_pool is not None else snapshot_data_pool()
        return {
            key: _serialize(value)
            for key, value in sorted(data_pool.items(), key=lambda item: item[0])
        }

    def _log_file_for_source(self, source: str) -> pathlib.Path:
        log_cfg = _coerce_mapping(self.config.get('logging', {}))
        file_cfg = _coerce_mapping(log_cfg.get('file', {}))
        source_key = {
            'same': 'same_path',
            'web': 'web_path',
            'playout': 'playout_path',
        }.get(source, 'main_path')
        path = file_cfg.get(source_key) or file_cfg.get('path') or './logs/haze-weather-radio.log'
        text_path = str(path)
        if text_path.startswith('/'):
            return pathlib.Path(text_path)
        return (self.root_dir / text_path).resolve()

    def _logs_payload_data(
        self,
        source: str = 'app',
        lines: int = 120,
        *,
        log_path: pathlib.Path | None = None,
    ) -> dict[str, Any]:
        bounded_lines = max(10, min(lines, 500))
        path = log_path if log_path is not None else self._log_file_for_source(source)
        return {
            'source': source,
            'path': str(path),
            'lines': _tail_lines(path, bounded_lines),
        }

    def _managed_file_path(self, filename: str) -> pathlib.Path:
        if filename.endswith('.xml'):
            return self.root_dir / 'managed' / 'configs' / filename
        return self.root_dir / 'managed' / filename

    def _preview_audio_dir(self) -> pathlib.Path:
        return self.root_dir / 'audio' / '_previews'

    def _feed_timezone(self, feed_id: str) -> str:
        for feed in self.config.get('feeds', []):
            if isinstance(feed, dict) and str(feed.get('id') or '') == feed_id:
                return str(feed.get('timezone') or 'UTC')
        return 'UTC'

    def _manual_same_runtime_entry(
        self,
        payload: SAMAirRequest,
        feed_id: str,
        locations: list[str],
        header: Any,
        issued_at: datetime,
        expires_at: datetime,
        identifier: str,
        display_id: str,
        callsign: str,
        description: str,
    ) -> dict[str, Any]:
        same_event = payload.event.upper()[:3]
        return {
            'identifier': identifier,
            'feed_id': feed_id,
            'received_at': issued_at.isoformat(),
            'display_id': display_id,
            'metadata': {
                'event': same_event,
                'effective': issued_at.isoformat(),
                'onset': issued_at.isoformat(),
                'expires': expires_at.isoformat(),
            },
            'source': {
                'kind': 'manual',
                'originator': payload.originator.upper()[:3],
                'eventCode': same_event,
                'callsign': callsign,
                'sameHeader': header.encoded,
            },
            'text': {
                'description': description,
                'instruction': '',
            },
            'areas': [
                {'sameCode': location}
                for location in locations[:31]
            ],
        }

    def _manual_same_translation(
        self,
        payload: SAMAirRequest,
        feed_id: str,
        locations: list[str],
        header: Any,
        issued_at: datetime,
        expires_at: datetime,
        identifier: str,
        display_id: str,
        callsign: str,
    ) -> str:
        entry = self._manual_same_runtime_entry(
            payload,
            feed_id,
            locations,
            header,
            issued_at,
            expires_at,
            identifier,
            display_id,
            callsign,
            '',
        )
        return str(serialize_alert(entry, self._feed_timezone(feed_id)).get('message') or '').strip()

    def _build_same_audio(self, payload: SAMAirRequest) -> dict[str, Any]:
        from module.same import SAMEHeader, SAME_SAMPLE_RATE, generate_same, resample as _resample, to_pcm16

        feeds_cfg = self.config.get('feeds', [])
        same_cfg = self.config.get('same', {})
        callsign = same_cfg.get('sender') or os.environ.get('SAME_ID', 'HAZE0000')
        same_sr = SAME_SAMPLE_RATE

        if payload.air_on_all_feeds:
            target_ids = [f.get('id', 'default') for f in feeds_cfg if f.get('enabled', True)]
            if not target_ids:
                raise HTTPException(status_code=400, detail='No enabled feeds configured')
        elif payload.feed_ids:
            enabled_ids = {f.get('id') for f in feeds_cfg if f.get('enabled', True)}
            target_ids = [fid for fid in payload.feed_ids if fid in enabled_ids]
            if not target_ids:
                raise HTTPException(status_code=400, detail='None of the specified feeds are enabled')
        else:
            fid = payload.feed_id or (feeds_cfg[0].get('id', 'default') if feeds_cfg else 'default')
            target_ids = [fid]

        hours = max(0, min(payload.duration_hours, 99))
        minutes = max(0, min(payload.duration_minutes, 59))
        duration_code = f'{hours:02d}{minutes:02d}'
        if duration_code == '0000':
            duration_code = '0100'
        locations = [loc.strip() for loc in payload.locations if loc.strip()] or ['000000']
        header = SAMEHeader(
            originator=payload.originator.upper()[:3],
            event=payload.event.upper()[:3],
            locations=tuple(locations[:31]),
            duration=duration_code,
            callsign=callsign,
        )
        tone: str | None = payload.tone_type.upper() if payload.tone_type.upper() != 'NONE' else None
        issued_at = datetime.now(timezone.utc)
        duration_minutes_total = hours * 60 + minutes
        if duration_minutes_total <= 0:
            duration_minutes_total = 60
        expires_at = issued_at + timedelta(minutes=duration_minutes_total)
        identifier = f'manual_{int(time.time())}'
        display_id = f'MSG{issued_at.strftime("%H%M%S")}'
        custom_voice_message = payload.voice_message.strip()
        tts_parts: list[str] = []
        if payload.prepend_same_translation:
            tts_parts.append(self._manual_same_translation(
                payload,
                target_ids[0],
                locations,
                header,
                issued_at,
                expires_at,
                identifier,
                display_id,
                callsign,
            ))
        if custom_voice_message:
            tts_parts.append(custom_voice_message)
        tts_message = ' '.join(part for part in tts_parts if part).strip()

        voice_path: pathlib.Path | None = None
        voice_array = None
        prebuilt_alert_pcm: bytes | None = None
        if payload.audio_file_path.strip():
            upload_dir = (self.root_dir / 'audio' / '_uploads').resolve()
            preview_dir = self._preview_audio_dir().resolve()
            candidate = pathlib.Path(payload.audio_file_path)
            if not candidate.is_absolute():
                candidate = (self.root_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()
            in_preview_dir = False
            try:
                candidate.relative_to(upload_dir)
            except ValueError:
                try:
                    candidate.relative_to(preview_dir)
                    in_preview_dir = True
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail='Invalid audio file path') from exc
            if not candidate.exists():
                raise HTTPException(status_code=400, detail='Audio file not found')
            if in_preview_dir:
                from module.buffer import CHANNELS as _BUS_CH, SAMPLE_RATE as _BUS_SR
                with wave.open(str(candidate), 'rb') as wf:
                    channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    sample_rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                if sample_width != 2:
                    raise HTTPException(status_code=400, detail='Invalid preview audio format')
                if channels != _BUS_CH or sample_rate != _BUS_SR:
                    raise HTTPException(status_code=400, detail='Invalid preview audio sample rate or channel count')
                prebuilt_alert_pcm = frames
            else:
                voice_path = candidate
        elif tts_message:
            from module.tts import synthesize_pcm
            from module.buffer import CHANNELS, SAMPLE_RATE as BUS_SR
            import numpy as np
            pcm = synthesize_pcm(self.config, tts_message)
            if pcm:
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
                if CHANNELS == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1)
                voice_array = _resample(samples, BUS_SR, same_sr)
        if prebuilt_alert_pcm is not None:
            alert_pcm = prebuilt_alert_pcm
        else:
            full_signal = generate_same(
                header=header,
                tone_type=cast(Any, tone),
                audio_msg_path=voice_path,
                audio_msg_array=voice_array,
                attn_duration_s=8.0,
            )

            from module.buffer import SAMPLE_RATE as _BUS_SR
            alert_pcm = to_pcm16(_resample(full_signal, same_sr, _BUS_SR))

        return {
            'feeds_cfg': feeds_cfg,
            'target_ids': target_ids,
            'locations': locations,
            'header': header,
            'alert_pcm': alert_pcm,
            'issued_at': issued_at,
            'expires_at': expires_at,
            'identifier': identifier,
            'display_id': display_id,
            'callsign': callsign,
            'duration_code': duration_code,
            'same_sr': same_sr,
            'voice_message': custom_voice_message,
            'tts_message': tts_message,
        }

    def _register_routes(self) -> None:
        if self.mode == 'public':
            public_api_base = self._public_api_base()
            self.app.add_api_route('/', self.index, methods=['GET'])
            self.app.add_api_route('/feeds', self.public_feeds_page, methods=['GET'])
            self.app.add_api_route(f'{public_api_base}/health', self.public_health, methods=['GET'])
            self.app.add_api_route(
                f'{public_api_base}/feeds/{{feed_id}}/webrtc/offer',
                self.public_webrtc_offer,
                methods=['POST'],
            )
            self.app.add_api_websocket_route(
                f'{public_api_base}/panel/ws',
                self.public_panel_stream,
            )
            return

        api_base = self._api_base()
        self.app.add_api_route('/', self.admin_root, methods=['GET'])
        self.app.add_api_route('/admin', self.admin, methods=['GET'])
        self.app.add_api_route('/login', self.login_page, methods=['GET'])
        self.app.add_api_route('/banner', self.banner_page, methods=['GET'])
        self.app.add_api_route(f'{api_base}/health', self.health, methods=['GET'])
        self.app.add_api_route(f'{api_base}/auth/login', self.login, methods=['POST'])
        self.app.add_api_route(f'{api_base}/auth/logout', self.logout, methods=['POST'])
        self.app.add_api_route(f'{api_base}/auth/session', self.session_status, methods=['GET'])
        self.app.add_api_route('/same',   self.same_page,   methods=['GET'])
        self.app.add_api_route('/wx',     self.wx_root,     methods=['GET'])
        self.app.add_api_route(
            f'{api_base}/managed/{{filename}}',
            self.managed_read,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/managed/{{filename}}',
            self.managed_write,
            methods=['PUT'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/same/templates',
            self.same_templates_get,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/same/templates',
            self.same_templates_put,
            methods=['PUT'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/same/test',
            self.same_test,
            methods=['POST'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/same/generate',
            self.same_generate,
            methods=['POST'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/same/generated-audio',
            self.same_generated_audio,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/same/air',
            self.same_air,
            methods=['POST'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/same/upload-audio',
            self.upload_audio,
            methods=['POST'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/same/event-codes',
            self.same_event_codes,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/same/location-names',
            self.same_location_names,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/banner',
            self.banner_payload,
            methods=['GET'],
        )
        self.app.add_api_route(
            f'{api_base}/banner/stream',
            self.banner_stream,
            methods=['GET'],
        )
        self.app.add_api_websocket_route(
            f'{api_base}/banner/audio',
            self.banner_audio_stream,
        )
        self.app.add_api_websocket_route(
            f'{api_base}/panel/ws',
            self.admin_panel_stream,
        )
        if self._receiver_enabled():
            receiver_base = self._receiver_api_base()
            self.app.add_api_route(
                f'{receiver_base}/pair/challenge',
                self.receiver_pair_challenge,
                methods=['POST'],
            )
            self.app.add_api_route(
                f'{receiver_base}/pair/complete',
                self.receiver_pair_complete,
                methods=['POST'],
            )
            self.app.add_api_route(
                f'{receiver_base}/session',
                self.receiver_session,
                methods=['POST'],
            )
            self.app.add_api_websocket_route(
                f'{receiver_base}/ws',
                self.receiver_stream,
            )
        wx_base = self._wx_base()
        self.app.add_api_route(wx_base,                   self.wx_root,     methods=['GET'])
        self.app.add_api_route(f'{wx_base}/generate',     self.wx_generate, methods=['POST'])
        self.app.add_api_route(f'{wx_base}/packages',     self.wx_packages, methods=['GET'])

    async def index(self) -> FileResponse:
        return FileResponse(self.webroot / 'index.html')

    async def admin_root(self, request: Request) -> Response:
        return FileResponse(self.webroot / 'admin.html')

    async def admin(self, request: Request) -> Response:
        return FileResponse(self.webroot / 'admin.html')

    async def login_page(self, request: Request) -> Response:
        if self._auth_enabled() and self._check_request_auth(request):
            next_url = _safe_local_path(request.query_params.get('next'), '/admin')
            return RedirectResponse(url=next_url, status_code=303)
        return FileResponse(self.webroot / 'login.html')

    async def public_feeds_page(self, request: Request) -> FileResponse:
        if self._public_feeds_access() == 'disabled':
            raise HTTPException(status_code=404, detail='Public feed streaming disabled')
        return FileResponse(self.webroot / 'index.html')

    async def banner_page(self) -> FileResponse:
        return FileResponse(self.webroot / 'banner.html')

    async def health(self) -> dict[str, Any]:
        system = _coerce_mapping(snapshot_runtime().get('system', {}))
        metadata = build_metadata(self.config)
        return {
            'ok': True,
            'site_name': self._public_site_name(),
            'on_air_name': self._on_air_name(),
            **metadata,
            'auth_required': self._auth_enabled(),
            'wx_base': self._wx_base(),
            'public_url': self._public_site_url(),
            'started_at': datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            'uptime_seconds': round(time.time() - self.started_at, 1),
            'last_connected': {
                'ip': system.get('webpanel_last_connected_ip'),
                'at': system.get('webpanel_last_connected_at'),
            },
        }

    async def public_health(self, request: Request) -> dict[str, Any]:
        return {
            'ok': True,
            'started_at': datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            'uptime_seconds': round(time.time() - self.started_at, 1),
            'feeds_access': self._public_feeds_access(),
            'admin_url': self._public_admin_url(request),
            'webrtc_enabled': bool(self._public_webrtc_cfg().get('enabled', True)) and _ensure_feed_audio_track() is not None,
        }

    async def login(self, payload: LoginRequest, request: Request) -> JSONResponse:
        ip = self._client_host(request=request)
        if self._auth_enabled() and not self._login_allowed(ip):
            raise HTTPException(status_code=429, detail='Too many login attempts. Try again later.')
        if self._auth_enabled() and not self._verify_password(payload.password):
            raise HTTPException(status_code=401, detail='Invalid password')

        self._clear_login_attempts(ip)
        token = self._issue_token(request)
        self._record_admin_connection(ip)
        append_runtime_event('auth', 'Web session established')
        response = JSONResponse({
            'token': token,
            'expires_in_seconds': self._token_ttl_s,
            'auth_required': self._auth_enabled(),
        })
        self._set_session_cookie(response, token, request)
        return response

    async def logout(self, request: Request) -> JSONResponse:
        self._revoke_token(self._request_token(request))
        response = JSONResponse({'ok': True, 'public_url': self._public_site_url()})
        self._clear_session_cookie(response, request)
        append_runtime_event('auth', 'Web session ended')
        return response

    async def session_status(self, request: Request) -> dict[str, Any]:
        return {
            'authenticated': self._check_request_auth(request),
            'auth_required': self._auth_enabled(),
            'public_url': self._public_site_url(),
        }

    async def summary(self) -> dict[str, Any]:
        return self._summary_payload()

    async def public_summary(self, request: Request) -> dict[str, Any]:
        return self._public_summary_payload(request)

    async def config_payload(self) -> dict[str, Any]:
        return self._config_payload()

    async def public_feeds_payload(self, request: Request) -> list[dict[str, Any]]:
        await self._require_public_feed_access(request)
        return self._public_feed_payload()

    def _admin_panel_payload(
        self,
        log_source: str = 'app',
        lines: int = 120,
        *,
        runtime: dict[str, Any] | None = None,
        data_pool: dict[str, Any] | None = None,
        sequences: dict[str, list[Any]] | None = None,
        queue_depths: dict[str, int] | None = None,
        logs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        runtime = runtime if runtime is not None else snapshot_runtime()
        data_pool = data_pool if data_pool is not None else snapshot_data_pool()
        sequences = sequences if sequences is not None else snapshot_playout_sequences()
        queue_depths = queue_depths if queue_depths is not None else snapshot_alert_queues()
        system = _coerce_mapping(runtime.get('system', {}))
        return {
            'type': 'admin_state',
            'summary': self._summary_payload(
                runtime=runtime,
                data_pool=data_pool,
                sequences=sequences,
                queue_depths=queue_depths,
            ),
            'datapool': self._datapool_payload(data_pool),
            'config': self._config_payload(),
            'events': cast(list[dict[str, Any]], runtime.get('events', [])),
            'logs': logs if logs is not None else self._logs_payload_data(log_source, lines),
            'last_connected': {
                'ip': system.get('webpanel_last_connected_ip'),
                'at': system.get('webpanel_last_connected_at'),
            },
        }

    def _public_panel_payload(self, request: Request | None = None) -> dict[str, Any]:
        include_feeds = True
        if request is None:
            include_feeds = False
        return {
            'type': 'public_state',
            'summary': self._public_summary_payload(request, include_feeds=include_feeds),
        }

    async def _create_feed_webrtc_peer(self, feed_id: str, sdp: str, sdp_type: str) -> tuple[dict[str, str], Any, Any]:
        track_cls = _ensure_feed_audio_track()
        if track_cls is None:
            raise HTTPException(status_code=503, detail='aiortc is not installed')

        feed_cfg = self._feed_by_id(feed_id)
        if feed_cfg is None or not bool(feed_cfg.get('enabled', True)):
            raise HTTPException(status_code=404, detail='Feed not available')

        from aiortc import RTCPeerConnection, RTCSessionDescription

        peer = RTCPeerConnection()
        self._peer_connections.add(peer)
        track = track_cls(feed_id)
        peer.addTrack(track)

        @peer.on('connectionstatechange')
        async def _on_state_change() -> None:
            if peer.connectionState in {'failed', 'closed'}:
                track.stop()
                self._peer_connections.discard(peer)
                await peer.close()

        try:
            await peer.setRemoteDescription(RTCSessionDescription(sdp=sdp, type=sdp_type))
            answer = await peer.createAnswer()
            await peer.setLocalDescription(answer)
        except Exception:
            track.stop()
            self._peer_connections.discard(peer)
            try:
                await peer.close()
            except Exception:
                pass
            raise

        return {
            'sdp': peer.localDescription.sdp,
            'type': peer.localDescription.type,
        }, peer, track

    async def public_webrtc_offer(
        self,
        feed_id: str,
        payload: WebRTCOfferRequest,
        request: Request,
    ) -> dict[str, Any]:
        await self._require_public_feed_access(request)
        if not bool(self._public_webrtc_cfg().get('enabled', True)):
            raise HTTPException(status_code=404, detail='WebRTC disabled')

        try:
            answer, _peer, _track = await self._create_feed_webrtc_peer(feed_id, payload.sdp, payload.type)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f'WebRTC negotiation failed: {exc}') from exc
        return answer

    async def receiver_pair_challenge(self, payload: ReceiverPairChallengeRequest, request: Request) -> dict[str, Any]:
        if not self._receiver_enabled():
            raise HTTPException(status_code=404, detail='Receiver pairing disabled')
        self._require_receiver_https(request)
        feed_id = str(payload.feed_id or '').strip()
        receiver_id = str(payload.receiver_id or '').strip()
        receiver_hostname = str(payload.receiver_hostname or '').strip()
        receiver_nonce = str(payload.nonce or '').strip()
        if not feed_id or not receiver_id or not receiver_hostname or not receiver_nonce:
            raise HTTPException(status_code=400, detail='feed_id, receiver_id, receiver_hostname, and nonce are required')
        feed = self._feed_by_id(feed_id)
        if feed is None or not bool(feed.get('enabled', True)):
            raise HTTPException(status_code=404, detail='Feed not available')

        challenge_id = secrets.token_urlsafe(24)
        server_nonce = secrets.token_urlsafe(32)
        expires_at = time.time() + self._receiver_challenge_ttl_s()
        with self._receiver_lock:
            self._receiver_challenges[challenge_id] = {
                'feed_id': feed_id,
                'receiver_id': receiver_id,
                'receiver_hostname': receiver_hostname,
                'receiver_nonce': receiver_nonce,
                'server_nonce': server_nonce,
                'expires_at': expires_at,
            }
        return {
            'challenge_id': challenge_id,
            'server_nonce': server_nonce,
            'algorithm': 'hmac-sha256',
            'proof_kind': 'pair-v1',
            'expires_at': datetime.fromtimestamp(expires_at, timezone.utc).isoformat(),
        }

    async def receiver_pair_complete(self, payload: ReceiverPairCompleteRequest, request: Request) -> dict[str, Any]:
        if not self._receiver_enabled():
            raise HTTPException(status_code=404, detail='Receiver pairing disabled')
        self._require_receiver_https(request)
        challenge_id = str(payload.challenge_id or '').strip()
        feed_id = str(payload.feed_id or '').strip()
        receiver_id = str(payload.receiver_id or '').strip()
        receiver_hostname = str(payload.receiver_hostname or '').strip()
        receiver_nonce = str(payload.nonce or '').strip()
        proof = str(payload.proof or '').strip().lower()
        now = time.time()

        with self._receiver_lock:
            expired = [key for key, value in self._receiver_challenges.items() if float(value.get('expires_at') or 0) <= now]
            for key in expired:
                self._receiver_challenges.pop(key, None)
            challenge = self._receiver_challenges.pop(challenge_id, None)

        if not challenge:
            raise HTTPException(status_code=401, detail='Pairing challenge expired or invalid')
        if (
            challenge.get('feed_id') != feed_id
            or challenge.get('receiver_id') != receiver_id
            or challenge.get('receiver_hostname') != receiver_hostname
            or challenge.get('receiver_nonce') != receiver_nonce
        ):
            raise HTTPException(status_code=401, detail='Pairing challenge mismatch')

        feed = self._feed_by_id(feed_id)
        if feed is None or not bool(feed.get('enabled', True)):
            raise HTTPException(status_code=404, detail='Feed not available')

        message = self._receiver_proof_message('pair-v1', {
            'challenge_id': challenge_id,
            'feed_id': feed_id,
            'receiver_id': receiver_id,
            'receiver_hostname': receiver_hostname,
            'receiver_nonce': receiver_nonce,
            'server_nonce': challenge.get('server_nonce', ''),
        })

        with self._receiver_lock:
            store = self._receiver_store_load()
            consumed = _coerce_mapping(store.get('consumed_pair_token_digests', {}))
            matched_token: dict[str, Any] | None = None
            for token in self._receiver_pairing_tokens():
                feed_ids = cast(set[str], token.get('feed_ids') or set())
                if not feed_ids or (feed_id not in feed_ids and '*' not in feed_ids):
                    continue
                digest = str(token.get('digest') or '')
                if digest in consumed:
                    continue
                expected = self._receiver_hmac(str(token.get('token') or ''), message)
                if hmac.compare_digest(expected, proof):
                    matched_token = token
                    break
            if matched_token is None:
                raise HTTPException(status_code=401, detail='Invalid or consumed pairing token')

            credential_id = secrets.token_urlsafe(18)
            credential_secret = secrets.token_urlsafe(36)
            credential_expires = now + self._receiver_credential_ttl_s()
            credentials = _coerce_mapping(store.get('credentials', {}))
            credentials[credential_id] = {
                'credential_secret': credential_secret,
                'feed_id': feed_id,
                'receiver_id': receiver_id,
                'receiver_hostname': receiver_hostname,
                'server_hostname': socket.gethostname(),
                'created_at': datetime.fromtimestamp(now, timezone.utc).isoformat(),
                'expires_at': datetime.fromtimestamp(credential_expires, timezone.utc).isoformat(),
                'last_seen_at': None,
            }
            consumed[str(matched_token.get('digest') or '')] = {
                'token_id': str(matched_token.get('id') or ''),
                'feed_id': feed_id,
                'credential_id': credential_id,
                'consumed_at': datetime.fromtimestamp(now, timezone.utc).isoformat(),
            }
            store['credentials'] = credentials
            store['consumed_pair_token_digests'] = consumed
            self._receiver_store_save(store)

        cookie, cookie_expires_at = self._issue_receiver_cookie(credential_id, feed_id, receiver_id, receiver_hostname)
        append_runtime_event('receiver', f'Receiver paired for {feed_id}: {receiver_hostname}', feed_id=feed_id)
        return {
            'ok': True,
            'feed_id': feed_id,
            'receiver_id': receiver_id,
            'credential_id': credential_id,
            'credential_secret': credential_secret,
            'cookie': cookie,
            'cookie_expires_at': cookie_expires_at,
            'ws_url': self._receiver_ws_url(request),
            'transmitter': self._receiver_transmitter_params(feed),
        }

    async def receiver_session(self, payload: ReceiverSessionRequest, request: Request) -> dict[str, Any]:
        if not self._receiver_enabled():
            raise HTTPException(status_code=404, detail='Receiver sessions disabled')
        self._require_receiver_https(request)
        feed_id = str(payload.feed_id or '').strip()
        receiver_id = str(payload.receiver_id or '').strip()
        receiver_hostname = str(payload.receiver_hostname or '').strip()
        credential_id = str(payload.credential_id or '').strip()
        nonce = str(payload.nonce or '').strip()
        proof = str(payload.proof or '').strip().lower()
        if not all((feed_id, receiver_id, receiver_hostname, credential_id, nonce, proof)):
            raise HTTPException(status_code=400, detail='Missing receiver session fields')

        feed = self._feed_by_id(feed_id)
        if feed is None or not bool(feed.get('enabled', True)):
            raise HTTPException(status_code=404, detail='Feed not available')

        message = self._receiver_proof_message('session-v1', {
            'credential_id': credential_id,
            'feed_id': feed_id,
            'receiver_id': receiver_id,
            'receiver_hostname': receiver_hostname,
            'nonce': nonce,
        })

        with self._receiver_lock:
            store = self._receiver_store_load()
            credentials = _coerce_mapping(store.get('credentials', {}))
            credential = _coerce_mapping(credentials.get(credential_id, {}))
            if not credential:
                raise HTTPException(status_code=401, detail='Receiver credential not found')
            if credential.get('feed_id') != feed_id or credential.get('receiver_id') != receiver_id:
                raise HTTPException(status_code=401, detail='Receiver credential mismatch')
            expires_raw = str(credential.get('expires_at') or '')
            try:
                expires_at = datetime.fromisoformat(expires_raw)
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)
            except ValueError:
                raise HTTPException(status_code=401, detail='Receiver credential invalid')
            if expires_at <= datetime.now(timezone.utc):
                credentials.pop(credential_id, None)
                store['credentials'] = credentials
                self._receiver_store_save(store)
                raise HTTPException(status_code=401, detail='Receiver credential expired')
            expected = self._receiver_hmac(str(credential.get('credential_secret') or ''), message)
            if not hmac.compare_digest(expected, proof):
                raise HTTPException(status_code=401, detail='Invalid receiver credential proof')
            credential['receiver_hostname'] = receiver_hostname
            credential['last_seen_at'] = datetime.now(timezone.utc).isoformat()
            credentials[credential_id] = credential
            store['credentials'] = credentials
            self._receiver_store_save(store)

        cookie, cookie_expires_at = self._issue_receiver_cookie(credential_id, feed_id, receiver_id, receiver_hostname)
        return {
            'ok': True,
            'feed_id': feed_id,
            'receiver_id': receiver_id,
            'credential_id': credential_id,
            'cookie': cookie,
            'cookie_expires_at': cookie_expires_at,
            'ws_url': self._receiver_ws_url(request),
            'transmitter': self._receiver_transmitter_params(feed),
        }

    async def receiver_stream(self, websocket: WebSocket) -> None:
        if not self._receiver_enabled():
            await websocket.close(code=1008)
            return
        if bool(self._receiver_cfg().get('require_tls', True)) and not self._receiver_websocket_secure(websocket):
            await websocket.close(code=1008)
            return
        auth_header = str(websocket.headers.get('Authorization') or '').strip()
        cookie = ''
        if auth_header.lower().startswith('hazereceivercookie '):
            cookie = auth_header.split(' ', 1)[1].strip()
        session = self._consume_receiver_cookie(cookie)
        if not session:
            await websocket.close(code=1008)
            return

        feed_id = str(session.get('feed_id') or '')
        feed = self._feed_by_id(feed_id)
        if feed is None or not bool(feed.get('enabled', True)):
            await websocket.close(code=1008)
            return

        peers: list[tuple[Any, Any]] = []
        await websocket.accept()
        await websocket.send_json({
            'type': 'receiver_ready',
            'feed_id': feed_id,
            'receiver_id': session.get('receiver_id'),
            'credential_id': session.get('credential_id'),
            'transmitter': self._receiver_transmitter_params(feed),
            'webrtc': {'signaling': 'websocket', 'mode': 'recvonly-offer'},
        })
        append_runtime_event('receiver', f'Receiver connected for {feed_id}: {session.get("receiver_hostname")}', feed_id=feed_id)
        try:
            while True:
                message = await websocket.receive_json()
                msg_type = str(message.get('type') or '').strip()
                if msg_type == 'webrtc_offer':
                    sdp = str(message.get('sdp') or '')
                    sdp_type = str(message.get('sdp_type') or message.get('description_type') or 'offer')
                    try:
                        answer, peer, track = await self._create_feed_webrtc_peer(feed_id, sdp, sdp_type)
                    except Exception as exc:
                        await websocket.send_json({'type': 'webrtc_error', 'detail': str(exc)})
                        continue
                    peers.append((peer, track))
                    await websocket.send_json({
                        'type': 'webrtc_answer',
                        'sdp': answer['sdp'],
                        'sdp_type': answer['type'],
                    })
                elif msg_type == 'ping':
                    await websocket.send_json({'type': 'pong', 'time': datetime.now(timezone.utc).isoformat()})
                elif msg_type == 'close':
                    break
                else:
                    await websocket.send_json({'type': 'error', 'detail': 'Unknown receiver message type'})
        except WebSocketDisconnect:
            pass
        finally:
            for peer, track in peers:
                try:
                    track.stop()
                except Exception:
                    pass
                try:
                    self._peer_connections.discard(peer)
                    await peer.close()
                except Exception:
                    pass
            append_runtime_event('receiver', f'Receiver disconnected for {feed_id}', feed_id=feed_id)

    async def public_panel_stream(self, websocket: WebSocket) -> None:
        include_feeds = websocket.query_params.get('feeds') == '1'
        access = self._public_feeds_access()
        if include_feeds and access == 'disabled':
            await websocket.close(code=1008)
            return
        if include_feeds and access == 'auth_required' and not await self._require_websocket_auth(websocket):
            return

        await websocket.accept()
        last_signature = ''
        try:
            while True:
                versions = snapshot_change_versions()
                self._feed_static_rows()
                signature = self._hash_compact({
                    'versions': {
                        'runtime': versions.get('runtime'),
                        'sequences': versions.get('sequences') if include_feeds else {},
                        'alert_queues': versions.get('alert_queues') if include_feeds else 0,
                    },
                    'feed_static': self._feed_static_signature,
                    'feeds_access': self._public_feeds_access(),
                    'webrtc': bool(self._public_webrtc_cfg().get('enabled', True)) and _ensure_feed_audio_track() is not None,
                    'uptime_bucket': int((time.time() - self.started_at) // 10),
                })
                if signature != last_signature:
                    last_signature = signature
                    runtime = snapshot_runtime()
                    sequences = snapshot_playout_sequences() if include_feeds else {}
                    queue_depths = snapshot_alert_queues() if include_feeds else {}
                    await websocket.send_json({
                        'type': 'public_state',
                        'summary': self._public_summary_payload(
                            include_feeds=include_feeds,
                            runtime=runtime,
                            sequences=sequences,
                            queue_depths=queue_depths,
                        ),
                    })
                await asyncio.sleep(1.0)
        except WebSocketDisconnect:
            return

    @staticmethod
    def _hash_compact(value: Any) -> str:
        return hashlib.blake2s(
            json.dumps(value, sort_keys=True, default=str, separators=(',', ':')).encode('utf-8'),
            digest_size=12,
        ).hexdigest()

    def _log_file_stamp(self, source: str) -> tuple[pathlib.Path, int, int]:
        log_path = self._log_file_for_source(source)
        try:
            stat = log_path.stat()
        except FileNotFoundError:
            return log_path, 0, 0
        return log_path, int(stat.st_mtime_ns), int(stat.st_size)

    async def _handle_admin_panel_command(
        self,
        message: dict[str, Any],
        *,
        websocket: WebSocket,
        log_source: str,
        lines: int,
    ) -> dict[str, Any]:
        command = str(message.get('command') or '').strip()
        payload = _coerce_mapping(message.get('payload', {}))

        if command == 'health':
            return await self.health()
        if command == 'state':
            return self._admin_panel_payload(log_source=log_source, lines=lines)
        if command == 'same.test':
            return await self.same_test(websocket, str(payload.get('event_code') or 'RWT'))
        if command == 'same.generate':
            return await self.same_generate(SAMAirRequest(**payload))
        if command == 'same.air':
            return await self.same_air(SAMAirRequest(**payload))
        if command == 'same.templates.get':
            return await self.same_templates_get()
        if command == 'same.templates.put':
            return await self.same_templates_put(FileWriteRequest(**payload))
        if command == 'same.event_codes':
            return await self.same_event_codes()
        if command == 'same.location_names':
            return await self.same_location_names()
        if command == 'managed.read':
            return await self.managed_read(str(payload.get('filename') or ''))
        if command == 'managed.write':
            return await self.managed_write(
                str(payload.get('filename') or ''),
                FileWriteRequest(content=str(payload.get('content') or '')),
            )
        if command == 'wx.packages':
            return await self.wx_packages()

        raise HTTPException(status_code=400, detail=f'Unknown panel command: {command}')

    async def _send_panel_command_result(
        self,
        websocket: WebSocket,
        message: dict[str, Any],
        result: Any = None,
        *,
        error: str = '',
    ) -> None:
        reply_to = str(message.get('request_id') or '')
        if error:
            await websocket.send_json({'type': 'command_error', 'reply_to': reply_to, 'detail': error})
        else:
            await websocket.send_json({'type': 'command_ok', 'reply_to': reply_to, 'result': result})

    async def admin_panel_stream(self, websocket: WebSocket) -> None:
        log_source = str(websocket.query_params.get('source') or 'app')
        lines = int(websocket.query_params.get('lines') or 120)
        control_mode = str(websocket.query_params.get('mode') or '').strip().lower() == 'control'
        await websocket.accept()

        def _ws_token() -> str:
            auth_header = str(websocket.headers.get('Authorization') or '').strip()
            bearer_token = auth_header[7:].strip() if auth_header.startswith('Bearer ') else ''
            return (
                websocket.cookies.get(self._session_cookie)
                or bearer_token
                or str(websocket.headers.get('X-Session-Token') or '').strip()
                or websocket.query_params.get('token')
                or ''
            )

        async def _send_auth_state(authenticated: bool, reply_to: str = '') -> None:
            system = _coerce_mapping(snapshot_runtime().get('system', {}))
            await websocket.send_json({
                'type': 'auth_state',
                'reply_to': reply_to,
                'authenticated': authenticated,
                'auth_required': self._auth_enabled(),
                'public_url': self._public_site_url(),
                'started_at': datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
                'uptime_seconds': round(time.time() - self.started_at, 1),
                'last_connected': {
                    'ip': system.get('webpanel_last_connected_ip'),
                    'at': system.get('webpanel_last_connected_at'),
                },
                **build_metadata(self.config),
                'site_name': self._public_site_name(),
                'on_air_name': self._on_air_name(),
            })

        if self._auth_enabled() and not self._check_token(_ws_token()):
            await _send_auth_state(False)
            try:
                while True:
                    message = await websocket.receive_json()
                    msg_type = str(message.get('type') or '').strip()
                    if msg_type == 'auth_check':
                        await _send_auth_state(False, str(message.get('request_id') or ''))
                    elif msg_type == 'login':
                        ip = self._client_host(websocket=websocket)
                        reply_to = str(message.get('request_id') or '')
                        if not self._login_allowed(ip):
                            await websocket.send_json({'type': 'auth_error', 'reply_to': reply_to, 'detail': 'Too many login attempts. Try again later.'})
                            continue
                        password = str(message.get('password') or '')
                        if not self._verify_password(password):
                            await websocket.send_json({'type': 'auth_error', 'reply_to': reply_to, 'detail': 'Invalid password'})
                            continue
                        self._clear_login_attempts(ip)
                        token = self._issue_token(websocket=websocket)
                        self._record_admin_connection(ip)
                        append_runtime_event('auth', 'Web session established')
                        await websocket.send_json({
                            'type': 'auth_ok',
                            'reply_to': reply_to,
                            'token': token,
                            'expires_in_seconds': self._token_ttl_s,
                            'auth_required': self._auth_enabled(),
                            'public_url': self._public_site_url(),
                        })
                    elif msg_type == 'logout':
                        self._revoke_token(str(message.get('token') or ''))
                        append_runtime_event('auth', 'Web session ended')
                        await websocket.send_json({'type': 'logout_ok', 'reply_to': str(message.get('request_id') or ''), 'public_url': self._public_site_url()})
                    elif msg_type == 'ping':
                        await websocket.send_json({'type': 'pong', 'reply_to': str(message.get('request_id') or ''), 'time': datetime.now(timezone.utc).isoformat()})
                    else:
                        await websocket.send_json({'type': 'auth_required', 'reply_to': str(message.get('request_id') or ''), 'detail': 'Authentication required'})
            except WebSocketDisconnect:
                return

        self._record_admin_connection(self._client_host(websocket=websocket))
        await _send_auth_state(True)
        if control_mode:
            try:
                while True:
                    message = await websocket.receive_json()
                    msg_type = str(message.get('type') or '').strip()
                    if msg_type == 'auth_check':
                        await _send_auth_state(True, str(message.get('request_id') or ''))
                    elif msg_type == 'logout':
                        self._revoke_token(str(message.get('token') or _ws_token()))
                        append_runtime_event('auth', 'Web session ended')
                        await websocket.send_json({'type': 'logout_ok', 'reply_to': str(message.get('request_id') or ''), 'public_url': self._public_site_url()})
                    elif msg_type == 'ping':
                        await websocket.send_json({'type': 'pong', 'reply_to': str(message.get('request_id') or ''), 'time': datetime.now(timezone.utc).isoformat()})
                    elif msg_type == 'command':
                        try:
                            result = await self._handle_admin_panel_command(message, websocket=websocket, log_source=log_source, lines=lines)
                            await self._send_panel_command_result(websocket, message, result)
                        except HTTPException as exc:
                            await self._send_panel_command_result(websocket, message, error=str(exc.detail))
                        except Exception as exc:
                            log.exception('Panel command failed: %s', message.get('command'))
                            await self._send_panel_command_result(websocket, message, error=str(exc))
                    else:
                        await websocket.send_json({'type': 'error', 'reply_to': str(message.get('request_id') or ''), 'detail': 'Unknown panel control message type'})
            except WebSocketDisconnect:
                return

        async def _receive_commands() -> None:
            while True:
                message = await websocket.receive_json()
                msg_type = str(message.get('type') or '').strip()
                if msg_type == 'auth_check':
                    await _send_auth_state(True, str(message.get('request_id') or ''))
                elif msg_type == 'logout':
                    self._revoke_token(str(message.get('token') or _ws_token()))
                    append_runtime_event('auth', 'Web session ended')
                    await websocket.send_json({'type': 'logout_ok', 'reply_to': str(message.get('request_id') or ''), 'public_url': self._public_site_url()})
                    return
                elif msg_type == 'ping':
                    await websocket.send_json({'type': 'pong', 'reply_to': str(message.get('request_id') or ''), 'time': datetime.now(timezone.utc).isoformat()})
                elif msg_type == 'command':
                    try:
                        result = await self._handle_admin_panel_command(message, websocket=websocket, log_source=log_source, lines=lines)
                        await self._send_panel_command_result(websocket, message, result)
                    except HTTPException as exc:
                        await self._send_panel_command_result(websocket, message, error=str(exc.detail))
                    except Exception as exc:
                        log.exception('Panel command failed: %s', message.get('command'))
                        await self._send_panel_command_result(websocket, message, error=str(exc))

        async def _send_changed_state() -> None:
            last_signature = ''
            while True:
                versions = snapshot_change_versions()
                log_stamp = self._log_file_stamp(log_source)
                self._feed_static_rows()
                signature = self._hash_compact({
                    'versions': versions,
                    'log': [str(log_stamp[0]), log_stamp[1], log_stamp[2]],
                    'feed_static': self._feed_static_signature,
                    'config_id': id(self._config_payload()),
                })
                if signature != last_signature:
                    last_signature = signature
                    runtime = snapshot_runtime()
                    data_pool = snapshot_data_pool()
                    sequences = snapshot_playout_sequences()
                    queue_depths = snapshot_alert_queues()
                    payload = self._admin_panel_payload(
                        log_source=log_source,
                        lines=lines,
                        runtime=runtime,
                        data_pool=data_pool,
                        sequences=sequences,
                        queue_depths=queue_depths,
                        logs=self._logs_payload_data(log_source, lines, log_path=log_stamp[0]),
                    )
                    await websocket.send_json(payload)
                await asyncio.sleep(1.0)

        try:
            receiver = asyncio.create_task(_receive_commands())
            sender = asyncio.create_task(_send_changed_state())
            done, pending = await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            for task in done:
                task.result()
        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
            return

    async def feeds_payload(self) -> list[dict[str, Any]]:
        return self._feeds_payload()

    async def datapool_payload(self) -> dict[str, Any]:
        return self._datapool_payload()

    async def events_payload(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], snapshot_runtime().get('events', []))

    async def logs_payload(self, source: str = 'app', lines: int = 120) -> dict[str, Any]:
        return self._logs_payload_data(source, lines)

    async def editor_page(self) -> FileResponse:
        return FileResponse(self.webroot / 'editor.html')

    async def same_page(self) -> Response:
        return RedirectResponse(url='/admin#/same', status_code=303)

    async def managed_read(self, filename: str) -> dict[str, Any]:
        if filename not in _MANAGED_ALLOWLIST:
            raise HTTPException(status_code=404, detail='File not available')
        path = self._managed_file_path(filename)
        if not path.exists():
            raise HTTPException(status_code=404, detail='File not found')
        return {'filename': filename, 'content': path.read_text(encoding='utf-8')}

    async def managed_write(self, filename: str, payload: FileWriteRequest) -> dict[str, Any]:
        if filename not in _MANAGED_ALLOWLIST:
            raise HTTPException(status_code=404, detail='File not available')
        path = self._managed_file_path(filename)
        if filename.endswith('.json'):
            try:
                json.loads(payload.content)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f'Invalid JSON: {exc}') from exc
        elif filename.endswith('.xml'):
            import xml.etree.ElementTree as ET

            try:
                ET.fromstring(payload.content)
            except ET.ParseError as exc:
                raise HTTPException(status_code=400, detail=f'Invalid XML: {exc}') from exc
        path.write_text(payload.content, encoding='utf-8')
        self._invalidate_panel_caches()
        append_runtime_event('editor', f'Managed file saved via web panel: {filename}')
        return {'ok': True, 'filename': filename}

    async def same_event_codes(self) -> dict[str, Any]:
        path = self.root_dir / 'managed' / 'sameMapping.json'
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def _banner_payload_data(self, feed: str = '') -> dict[str, Any]:
        feeds_cfg = [
            cast(dict[str, Any], cfg)
            for cfg in self.config.get('feeds', [])
            if isinstance(cfg, dict)
        ]
        requested_feed = str(feed or '').strip()
        feed_cfg: dict[str, Any] | None
        if requested_feed:
            feed_cfg = next((cfg for cfg in feeds_cfg if str(cfg.get('id') or '') == requested_feed), None)
            if feed_cfg is None:
                raise HTTPException(status_code=404, detail='Feed not found')
        else:
            feed_cfg = next((cfg for cfg in feeds_cfg if bool(cfg.get('enabled', True))), None)
            if feed_cfg is None and feeds_cfg:
                feed_cfg = feeds_cfg[0]

        if feed_cfg is None:
            raise HTTPException(status_code=404, detail='No configured feeds available')

        feed_id = str(feed_cfg.get('id') or '').strip()
        tz_name = str(feed_cfg.get('timezone') or 'UTC')
        active_entries = get_active_alerts(feed_id)
        alerts = [serialize_alert(entry, tz_name) for entry in active_entries]
        combined_message = '   |   '.join(alert['message'] for alert in alerts if alert.get('message'))
        signature_source: dict[str, Any] = {
            'feed_id': feed_id,
            'alerts': [
                {
                    'identifier': alert.get('identifier'),
                    'display_id': alert.get('display_id'),
                    'effective_at': alert.get('effective_at'),
                    'expires_at': alert.get('expires_at'),
                    'description': alert.get('description'),
                    'instruction': alert.get('instruction'),
                }
                for alert in alerts
            ],
        }
        signature = hashlib.sha1(
            json.dumps(signature_source, sort_keys=True).encode('utf-8')
        ).hexdigest()

        return {
            'feed_id': feed_id,
            'feed_name': str(feed_cfg.get('name') or feed_id),
            'timezone': tz_name,
            'active': bool(alerts),
            'alert_count': len(alerts),
            'primary_color': pick_banner_color(active_entries),
            'primary_gradient': pick_banner_gradient(active_entries),
            'combined_message': combined_message,
            'signature': signature,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'alerts': alerts,
        }

    async def banner_payload(self, feed: str = '') -> dict[str, Any]:
        return self._banner_payload_data(feed)

    async def banner_stream(self, request: Request, feed: str = '') -> StreamingResponse:
        async def iterator():
            keepalive_ticks = 0
            last_signature = ''
            while True:
                if await request.is_disconnected():
                    break
                payload = self._banner_payload_data(feed)
                signature = str(payload.get('signature') or '')
                if signature != last_signature or not last_signature:
                    last_signature = signature
                    keepalive_ticks = 0
                    yield f"event: banner\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
                else:
                    keepalive_ticks += 1
                    if keepalive_ticks >= 15:
                        keepalive_ticks = 0
                        yield ':\n\n'
                await asyncio.sleep(1.0)

        return StreamingResponse(
            iterator(),
            media_type='text/event-stream',
            headers={
                'Cache-Control': 'no-store',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
            },
        )

    async def banner_audio_stream(self, websocket: WebSocket) -> None:
        feed = str(websocket.query_params.get('feed') or '').strip()
        try:
            payload = self._banner_payload_data(feed)
        except HTTPException:
            await websocket.close(code=1008)
            return

        feed_id = str(payload.get('feed_id') or '').strip()
        if not feed_id:
            await websocket.close(code=1008)
            return

        await websocket.accept()
        audio_stream = register_alert_audio_stream(feed_id)

        try:
            while True:
                if (
                    websocket.client_state == WebSocketState.DISCONNECTED
                    or websocket.application_state == WebSocketState.DISCONNECTED
                ):
                    break

                try:
                    pcm, _identifier = await asyncio.to_thread(audio_stream.get, True, 1.0)
                except queue.Empty:
                    continue

                await websocket.send_bytes(_pcm_as_wav(pcm))
        except WebSocketDisconnect:
            pass
        finally:
            unregister_alert_audio_stream(feed_id, audio_stream)

    async def same_location_names(self) -> dict[str, str]:
        import csv
        path = self.root_dir / 'managed' / 'csv' / 'FORECAST_LOCATIONS.csv'
        names: dict[str, str] = {}
        try:
            with open(path, encoding='utf-8', newline='') as f:
                reader = csv.reader(f)
                next(reader, None)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2 and row[0].strip() and row[0].strip() not in names:
                        names[row[0].strip()] = row[1].strip()
        except Exception:
            pass
        return names

    async def same_templates_get(self) -> dict[str, Any]:
        return load_alert_templates(self.root_dir / 'managed' / 'configs' / 'alertTemplates.xml')

    async def same_templates_put(self, payload: FileWriteRequest) -> dict[str, Any]:
        try:
            data = json.loads(payload.content)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f'Invalid JSON: {exc}') from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail='Templates must be a JSON object')
        path = self.root_dir / 'managed' / 'configs' / 'alertTemplates.xml'
        existing = load_alert_templates(path)
        merged = merge_alert_templates(existing, data)
        write_alert_templates(merged, path)
        append_runtime_event('editor', 'SAME templates saved via web panel')
        return {'ok': True}

    async def upload_audio(self, file: UploadFile = File(...)) -> dict[str, Any]:
        import subprocess
        from module.same import SAME_SAMPLE_RATE

        same_sr = SAME_SAMPLE_RATE
        upload_dir = self.root_dir / 'audio' / '_uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = secrets.token_hex(10)
        original_suffix = pathlib.Path(file.filename or 'audio.bin').suffix.lower() or '.bin'
        if original_suffix not in {'.wav', '.mp3', '.ogg', '.opus', '.flac', '.aac', '.m4a', '.webm', '.bin'}:
            raise HTTPException(status_code=400, detail='Unsupported audio file type')
        input_path = upload_dir / f'{safe_stem}_in{original_suffix}'
        output_path = upload_dir / f'{safe_stem}.wav'
        max_upload_bytes = int(self._auth_cfg().get('max_audio_upload_bytes') or 50 * 1024 * 1024)
        content = await file.read()

        audio_filters = (
            'acompressor=threshold=-20dB:ratio=4:attack=5:release=50:makeup=10dB,'
            'loudnorm=I=-9.0:LRA=7:TP=-2.0'
        )

        if len(content) > max_upload_bytes:
            raise HTTPException(status_code=413, detail=f'File too large (max {max_upload_bytes // (1024 * 1024)} MB)')
        input_path.write_bytes(content)
        result = subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-i', str(input_path),
             '-af', audio_filters,
             '-ar', str(same_sr), '-ac', '1', '-c:a', 'pcm_s16le',
             str(output_path)],
            capture_output=True,
            timeout=120,
        )
        input_path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise HTTPException(
                status_code=400,
                detail=f'Audio encoding failed: {result.stderr.decode("utf-8", errors="replace")[:300]}',
            )
        append_runtime_event('editor', f'Audio uploaded and encoded: {file.filename}')
        return {'path': str(output_path), 'filename': file.filename or 'uploaded', 'sample_rate': same_sr}

    async def same_test(self, request: Request, event_code: str = 'RWT') -> dict[str, Any]:
        code = event_code.upper()[:3]
        ok = _fire_test(self.config, self.feeds, code)
        if not ok:
            raise HTTPException(status_code=400, detail=f'No template found or no enabled feeds for {code}')
        return {'ok': True, 'event_code': code}

    async def same_air(self, payload: SAMAirRequest) -> dict[str, Any]:
        from module.alert import save_alert_audio
        from module.events import push_alert

        built = self._build_same_audio(payload)
        target_ids = built['target_ids']
        locations = built['locations']
        header = built['header']
        alert_pcm = built['alert_pcm']
        issued_at = built['issued_at']
        expires_at = built['expires_at']
        identifier = built['identifier']
        display_id = built['display_id']
        callsign = built['callsign']
        same_event = payload.event.upper()[:3]
        webhook_jobs: list[Coroutine[Any, Any, None]] = []

        for fid in target_ids:
            runtime_entry = self._manual_same_runtime_entry(
                payload,
                fid,
                locations,
                header,
                issued_at,
                expires_at,
                identifier,
                display_id,
                callsign,
                built.get('voice_message', ''),
            )
            if push_alert(fid, 0, alert_pcm, identifier):
                store_runtime_alert_entry(fid, identifier, runtime_entry)
                saved_path = save_alert_audio(fid, identifier, alert_pcm)
                webhook_jobs.append(dispatch_webhook_async(fid, runtime_entry, same_event, saved_path, self.config))

        if webhook_jobs:
            await asyncio.gather(*webhook_jobs)

        encoded = header.encoded
        append_runtime_event('manual-same', f'Manual SAME aired: {encoded} -> {", ".join(target_ids)}')
        return {'ok': True, 'header': encoded, 'feed_id': target_ids[0], 'feeds_aired': target_ids}

    async def same_generate(self, payload: SAMAirRequest) -> dict[str, Any]:
        built = self._build_same_audio(payload)
        preview_dir = self._preview_audio_dir()
        preview_dir.mkdir(parents=True, exist_ok=True)
        preview_path = preview_dir / f"{built['identifier']}.wav"

        with wave.open(str(preview_path), 'wb') as wf:
            from module.buffer import CHANNELS as _BUS_CH, SAMPLE_RATE as _BUS_SR
            wf.setnchannels(_BUS_CH)
            wf.setsampwidth(2)
            wf.setframerate(_BUS_SR)
            wf.writeframes(built['alert_pcm'])

        encoded = built['header'].encoded
        preview_url = f"{self._api_base()}/same/generated-audio?path={urllib.parse.quote(str(preview_path))}"
        append_runtime_event('manual-same', f'Manual SAME generated: {encoded}')
        return {
            'ok': True,
            'header': encoded,
            'feed_id': built['target_ids'][0],
            'feeds_aired': built['target_ids'],
            'path': str(preview_path),
            'download_url': preview_url,
            'sample_rate': built['same_sr'],
        }

    async def same_generated_audio(self, path: str) -> FileResponse:
        preview_dir = self._preview_audio_dir().resolve()
        upload_dir = (self.root_dir / 'audio' / '_uploads').resolve()
        candidate = pathlib.Path(path)
        if not candidate.is_absolute():
            candidate = (self.root_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        try:
            candidate.relative_to(preview_dir)
        except ValueError:
            try:
                candidate.relative_to(upload_dir)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail='Invalid audio file path') from exc
        if not candidate.exists():
            raise HTTPException(status_code=404, detail='Audio file not found')
        return FileResponse(candidate, media_type='audio/wav', filename=candidate.name)

    async def wx_packages(self) -> dict[str, Any]:
        return {'packages': list(_ALL_PACKAGES)}

    async def wx_root(self) -> Response:
        return RedirectResponse(url='/admin#/wx', status_code=303)

    async def wx_generate(self, payload: WXGenerateRequest) -> StreamingResponse:
        from module.tts import synthesize_pcm_stream
        from module.packages import (
            alerts_package, climate_summary_package, current_conditions_package, air_quality_package,
            date_time_package, eccc_discussion_package, forecast_package,
            geophysical_alert_package, station_id, user_bulletin_package,
        )
        import json as _json
        import pathlib as _pl

        fmt = (payload.format or 'raw').lower()
        if fmt not in _FORMAT_MEDIA_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid format {fmt!r}. Valid options: {", ".join(sorted(_FORMAT_MEDIA_TYPES))}',
            )

        requested_source = _normalize_wx_source(payload.source)
        if requested_source is not None and requested_source not in _WX_SOURCES:
            raise HTTPException(
                status_code=400,
                detail=f'Invalid source {payload.source!r}. Valid options: auto, eccc, nws, twc',
            )

        feeds_cfg = self.config.get('feeds', [])

        obs_index:  dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {}
        fcst_index: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        clim_index: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
        aqhi_index: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {}
        for _feed in feeds_cfg:
            _fid = _feed.get('id', '')
            _tz  = _feed.get('timezone', 'UTC')
            for _block in _feed.get('locations', []):
                if not isinstance(_block, dict):
                    continue
                for _e in _block.get('observationLocations', []):
                    _lid = _e.get('id', '')
                    _source = _normalize_wx_source(_e.get('source') or _feed.get('data_source') or 'eccc') or 'eccc'
                    if _lid:
                        obs_index.setdefault(_lid.lower(), []).append((_source, _fid, _tz, _e))
                for _e in _block.get('forecastLocations', []):
                    _lid = _e.get('id', '')
                    _source = _normalize_wx_source(_e.get('source') or _feed.get('data_source') or 'eccc') or 'eccc'
                    if _lid:
                        fcst_index.setdefault(_lid.lower(), []).append((_source, _fid, _e))
                for _e in _block.get('climateLocations', []):
                    _lid = _e.get('id', '')
                    _source = _normalize_wx_source(_e.get('source') or _feed.get('data_source') or 'eccc') or 'eccc'
                    if _lid:
                        clim_index.setdefault(_lid.lower(), []).append((_source, _fid, _e))
                for _e in _block.get('airQualityLocations', []):
                    _lid = _e.get('id', '')
                    _source = _normalize_wx_source(_e.get('source') or _feed.get('data_source') or 'eccc') or 'eccc'
                    if _lid:
                        aqhi_index.setdefault(_lid.lower(), []).append((_source, _fid, _tz, _e))

        def _pick_match[T](matches: list[T], source_getter) -> T | None:
            if not matches:
                return None
            if requested_source is None:
                return matches[0]
            for match in matches:
                if source_getter(match) == requested_source:
                    return match
            return None

        requested_locs = [
            str(location_id).strip()
            for location_id in (
                [payload.locations] if isinstance(payload.locations, str)
                else list(payload.locations)
            )
            if str(location_id).strip()
        ]
        requested_locs_lower = [lid.lower() for lid in requested_locs]
        resolved_obs = [
            match for lid in requested_locs_lower
            if (match := _pick_match(obs_index.get(lid, []), lambda item: item[0])) is not None
        ]
        resolved_fcst = [
            match for lid in requested_locs_lower
            if (match := _pick_match(fcst_index.get(lid, []), lambda item: item[0])) is not None
        ]
        resolved_clim = [
            match for lid in requested_locs_lower
            if (match := _pick_match(clim_index.get(lid, []), lambda item: item[0])) is not None
        ]
        resolved_aqhi = [
            match for lid in requested_locs_lower
            if (match := _pick_match(aqhi_index.get(lid, []), lambda item: item[0])) is not None
        ]

        tz              = 'UTC'
        primary_feed_id = feeds_cfg[0].get('id', '') if feeds_cfg else ''
        primary_feed    = feeds_cfg[0]                if feeds_cfg else {}
        if resolved_obs:
            _, primary_feed_id, tz, _ = resolved_obs[0]
        elif resolved_fcst:
            _, primary_feed_id, _ = resolved_fcst[0]
        elif resolved_clim:
            _, primary_feed_id, _ = resolved_clim[0]
        elif resolved_aqhi:
            _, primary_feed_id, tz, _ = resolved_aqhi[0]
        if primary_feed_id:
            primary_feed = next((f for f in feeds_cfg if f.get('id') == primary_feed_id), primary_feed)
            tz = primary_feed.get('timezone', tz)

        lang = payload.lang or primary_feed.get('language', self.config.get('language', 'en-CA'))

        requested_packages = (
            [payload.packages] if isinstance(payload.packages, str)
            else list(payload.packages)
        )
        requested: list[str] = (
            list(_ALL_PACKAGES)
            if len(requested_packages) == 1 and str(requested_packages[0]).strip().lower() == 'all'
            else [str(package_id) for package_id in requested_packages if str(package_id) in _ALL_PACKAGES]
        )
        if not requested:
            raise HTTPException(status_code=400, detail='No valid packages requested')

        focn45 = read_data_pool('focn45')
        discussion_text: str | None = (
            getattr(focn45, 'text', None) or (focn45 if isinstance(focn45, str) else None)
        )
        wwv_text: str | None = read_data_pool('wwv')

        try:
            bulletins_raw = (_pl.Path('managed') / 'userbulletins.json').read_text(encoding='utf-8')
            bulletins: list[Any] = _json.loads(bulletins_raw)
        except Exception:
            bulletins = []

        try:
            _reg_path = _pl.Path('data') / 'alerts' / f'{primary_feed_id}.json'
            registry: list[Any] = _json.loads(_reg_path.read_text(encoding='utf-8'))
        except Exception:
            registry = []

        conditions_parts = [
            current_conditions_package(
                read_data_pool(f"{fid}:{entry.get('id')}") if entry.get('id') else None,
                entry.get('name_override') or entry.get('name'),
                lang,
                secondary=(i > 0),
            )
            for i, (_, fid, _, entry) in enumerate(resolved_obs)
        ]
        forecast_parts = [
            forecast_package(
                read_data_pool(f"{fid}:forecast:{entry.get('id')}") if entry.get('id') else None,
                entry.get('name_override') or entry.get('name'),
                lang,
            )
            for _, fid, entry in resolved_fcst
        ]
        climate_parts = [
            climate_summary_package(
                read_data_pool(f"{fid}:climate:{entry.get('id')}") if entry.get('id') else None,
                entry.get('name_override') or entry.get('name'),
                lang,
            )
            for _, fid, entry in resolved_clim
        ]
        aqhi_parts = [
            air_quality_package(
                read_data_pool(f"{fid}:aqhi:{entry.get('id')}") if entry.get('id') else None,
                entry.get('name_override') or entry.get('name'),
                lang,
            )
            for _, fid, _, entry in resolved_aqhi
        ]

        pkg_lookup: dict[str, str] = {
            'date_time':          date_time_package(tz, lang),
            'station_id':         station_id(self.config, primary_feed_id, lang),
            'current_conditions': '  '.join(p for p in conditions_parts if p),
            'forecast':           '  '.join(p for p in forecast_parts if p),
            'climate_summary':    '  '.join(p for p in climate_parts if p),
            'air_quality':        '  '.join(p for p in aqhi_parts if p),
            'eccc_discussion':    eccc_discussion_package(discussion_text, None, lang) if discussion_text else '',
            'geophysical_alert':  geophysical_alert_package(wwv_text) if wwv_text else '',
            'user_bulletin':      user_bulletin_package(bulletins, lang, tz),
            'alerts':             alerts_package(registry, lang, tz, primary_feed),
        }

        voice   = payload.voice
        loc_str = ', '.join(requested_locs) if requested_locs else 'unspecified'
        source_label = requested_source or 'auto'
        append_runtime_event('wx-generate', f'On-demand WX: {requested} [{lang}] fmt={fmt} source={source_label} locs={loc_str}')

        rendered_packages = _wx_rendered_packages(pkg_lookup, requested)
        if not rendered_packages:
            raise HTTPException(
                status_code=404,
                detail='No weather content was available for the requested packages and locations',
            )
        rendered_package_ids = [pkg_id for pkg_id, _ in rendered_packages]

        if fmt in _TEXT_FORMATS:
            pkg_texts = dict(rendered_packages)
            if fmt == 'json':
                body = _json.dumps(
                    {
                        'packages': pkg_texts,
                        'locations': requested_locs,
                        'lang': lang,
                        'source': source_label,
                    },
                    ensure_ascii=False,
                    indent=2,
                ).encode('utf-8')
            elif fmt == 'xml':
                parts = [
                    f'  <package id="{_wx_escape_xml_attr(package_id)}">{_wx_escape_xml_text(text)}</package>'
                    for package_id, text in pkg_texts.items()
                ]
                body = ('<?xml version="1.0" encoding="UTF-8"?>\n<wx>\n' + '\n'.join(parts) + '\n</wx>').encode('utf-8')
            elif fmt == 'ssml':
                paras = '\n  '.join(f'<p>{_wx_escape_xml_text(text)}</p>' for text in pkg_texts.values())
                body = (
                    f'<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"'
                    f' xml:lang="{_wx_escape_xml_attr(lang)}">\n  {paras}\n</speak>'
                ).encode('utf-8')
            elif fmt == 'html':
                rows = '\n'.join(
                    (
                        f'<section id="{_wx_escape_xml_attr(package_id)}">'
                        f'<h2>{html.escape(package_id.replace("_", " ").title(), quote=False)}</h2>'
                        f'<pre>{html.escape(text, quote=False)}</pre>'
                        f'</section>'
                    )
                    for package_id, text in pkg_texts.items()
                )
                body = (
                    f'<!doctype html><html lang="{_wx_escape_xml_attr(lang)}"><head><meta charset="utf-8">'
                    f'<title>Weather Report</title></head><body>{rows}</body></html>'
                ).encode('utf-8')
            elif fmt == 'markdown':
                body = '\n\n'.join(
                    f'## {package_id.replace("_", " ").title()}\n\n{text}'
                    for package_id, text in pkg_texts.items()
                ).encode('utf-8')
            else:
                sections = '\n\n'.join(
                    (
                        f'\\section{{{_wx_escape_latex(package_id.replace("_", " ").title())}}}\n'
                        f'{_wx_escape_latex(text)}'
                    )
                    for package_id, text in pkg_texts.items()
                )
                body = (
                    '\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n'
                    '\\begin{document}\n\\title{Weather Report}\n\\maketitle\n'
                    f'{sections}\n\\end{{document}}'
                ).encode('utf-8')
            return StreamingResponse(
                iter([body]),
                media_type=_FORMAT_MEDIA_TYPES[fmt],
                headers={
                    'X-Packages':     ','.join(rendered_package_ids),
                    'X-Format':       fmt,
                    'X-Source':       source_label,
                    'Content-Length': str(len(body)),
                },
            )

        from module.buffer import SAMPLE_RATE as _BUS_SR, CHANNELS as _BUS_CH

        def _pcm_generator():
            for _, text in rendered_packages:
                yield from synthesize_pcm_stream(self.config, text, lang=lang, voice=voice)

        common_headers = {
            'X-Audio-Sample-Rate': str(_BUS_SR),
            'X-Audio-Channels':    str(_BUS_CH),
            'X-Packages':          ','.join(rendered_package_ids),
            'X-Format':            fmt,
            'X-Source':            source_label,
        }

        if fmt == 'raw':
            pcm_stream = iter(_pcm_generator())
            try:
                first_chunk = next(pcm_stream)
            except StopIteration as exc:
                raise HTTPException(status_code=502, detail='No audio was synthesized for the requested content') from exc

            def _raw_generator():
                yield first_chunk
                yield from pcm_stream

            return StreamingResponse(
                _raw_generator(),
                media_type='audio/raw',
                headers={'X-Audio-Encoding': 's16le', **common_headers},
            )

        pcm = b''.join(_pcm_generator())
        if not pcm:
            raise HTTPException(status_code=502, detail='No audio was synthesized for the requested content')

        if fmt == 'wav':
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(_BUS_CH)
                wf.setsampwidth(2)
                wf.setframerate(_BUS_SR)
                wf.writeframes(pcm)
            data = buf.getvalue()
        else:
            ffmpeg_extra = _FORMAT_FFMPEG_ARGS.get(fmt, [])
            result = subprocess.run(
                [
                    'ffmpeg', '-y', '-loglevel', 'error',
                    '-f', 's16le', '-ar', str(_BUS_SR), '-ac', str(_BUS_CH), '-i', 'pipe:0',
                    *ffmpeg_extra, 'pipe:1',
                ],
                input=pcm,
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise HTTPException(
                    status_code=500,
                    detail=f'Transcoding failed: {result.stderr.decode("utf-8", errors="replace")[:300]}',
                )
            data = result.stdout

        common_headers['Content-Length'] = str(len(data))

        async def _yield_once():
            yield data

        return StreamingResponse(
            _yield_once(),
            media_type=_FORMAT_MEDIA_TYPES[fmt],
            headers=common_headers,
        )

    def start(self) -> threading.Thread:
        mode_cfg = self._public_cfg() if self.mode == 'public' else self._admin_cfg()
        host = str(mode_cfg.get('host') or '0.0.0.0')
        port = int(mode_cfg.get('port') or (8080 if self.mode == 'public' else 6444))

        update_runtime_status({
            f'webpanel_{self.mode}_enabled': True,
            f'webpanel_{self.mode}_host': host,
            f'webpanel_{self.mode}_port': port,
        })
        append_runtime_event('web', f'{self.mode.title()} web panel listening on {host}:{port}')

        def _runner() -> None:
            try:
                import uvicorn
            except ModuleNotFoundError:
                log.exception('uvicorn is not installed for the active Python interpreter')
                append_runtime_event('web-error', 'uvicorn is not installed for the active Python interpreter')
                return

            uvicorn.run(self.app, host=host, port=port, log_level='info')

        return threading.Thread(target=_runner, name=f'webpanel:{self.mode}', daemon=True)


def start_web_server(config: dict[str, Any], feeds: list[dict[str, Any]] | None = None) -> list[threading.Thread]:
    panel_cfg = _coerce_mapping(config.get('webpanel', config.get('web', {})))
    if not panel_cfg.get('enabled', False):
        return []

    started_at = time.time()
    threads: list[threading.Thread] = []

    public_cfg = _coerce_mapping(panel_cfg.get('public', {}))
    public_enabled = bool(public_cfg.get('enabled', True))
    if public_enabled:
        public_server = WebServer(config, feeds, mode='public', started_at=started_at)
        log.info('Starting public web panel on %s:%s', public_server._public_cfg().get('host', '0.0.0.0'), public_server._public_cfg().get('port', 8080))
        threads.append(public_server.start())

    admin_server = WebServer(config, feeds, mode='admin', started_at=started_at)
    admin_cfg = admin_server._admin_cfg()
    if bool(admin_cfg.get('enabled', True)):
        log.info('Starting admin web panel on %s:%s', admin_cfg.get('host', '0.0.0.0'), admin_cfg.get('port', 6444))
        threads.append(admin_server.start())

    return threads

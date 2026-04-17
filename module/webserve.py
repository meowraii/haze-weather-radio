from __future__ import annotations

import dataclasses
import io
import json
import logging
import pathlib
import secrets
import subprocess
import threading
import time
import wave
from collections import deque
from datetime import datetime, timezone
from typing import Any, cast

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from managed.events import (
    append_runtime_event,
    read_data_pool,
    snapshot_alert_queues,
    snapshot_data_pool,
    snapshot_playout_sequences,
    snapshot_runtime,
    update_runtime_status,
)
from managed.packages import air_quality_package
from module.alert import feed_same_codes
from module.scheduler import fire_test as _fire_test

log = logging.getLogger(__name__)


class LoginRequest(BaseModel):
    password: str = ''


class FileWriteRequest(BaseModel):
    content: str = ''


class SAMAirRequest(BaseModel):
    feed_id: str = ''
    originator: str = 'WXR'
    event: str = 'CEM'
    locations: list[str] = []
    duration_hours: int = 1
    duration_minutes: int = 0
    callsign: str = ''
    tone_type: str = 'WXR'
    voice_message: str = ''
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


_MANAGED_ALLOWLIST: frozenset[str] = frozenset({'userbulletins.json', 'dictionary.json', 'packages.py', 'sameTemplate.json', 'sameTest.json', 'sameMapping.json'})

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


def _tail_lines(path: pathlib.Path, line_count: int) -> list[str]:
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8', errors='replace') as handle:
        return [line.rstrip('\n') for line in deque(handle, maxlen=line_count)]


class WebServer:
    def __init__(self, config: dict[str, Any], feeds: list[dict[str, Any]] | None = None):
        self.config = config
        self.feeds = feeds or [f for f in config.get('feeds', []) if f.get('enabled', True)]
        self.started_at = time.time()
        self.root_dir = pathlib.Path(__file__).resolve().parent.parent
        self.webroot = self.root_dir / 'webroot'
        self._tokens: dict[str, float] = {}
        self._token_ttl_s = 12 * 60 * 60

        self.app = FastAPI(title='Haze Weather Radio Panel', version='2026.3.27')
        self._configure_cors()
        self.app.mount('/assets', StaticFiles(directory=str(self.webroot)), name='assets')
        self._register_routes()

    def _panel_cfg(self) -> dict[str, Any]:
        return _coerce_mapping(self.config.get('webpanel', self.config.get('web', {})))

    def _auth_cfg(self) -> dict[str, Any]:
        return _coerce_mapping(self._panel_cfg().get('authentication', {}))

    def _api_base(self) -> str:
        rest_cfg = _coerce_mapping(self._panel_cfg().get('rest_api', {}))
        base_path = str(rest_cfg.get('base_path') or '/api/v1')
        return '/' + base_path.strip('/')

    def _wx_base(self) -> str:
        wx_cfg = _coerce_mapping(self.config.get('wx_on_demand', {}))
        base = str(wx_cfg.get('endpoint-base') or '/api/wx-on-demand/v1')
        return '/' + base.strip('/')

    def _auth_enabled(self) -> bool:
        return bool(self._auth_cfg().get('enabled', True))

    def _configure_cors(self) -> None:
        cors_cfg = _coerce_mapping(self._panel_cfg().get('cors', {}))
        origins = cors_cfg.get('allow_origins') or ['127.0.0.1', 'http://127.0.0.1']
        methods = cors_cfg.get('allow_methods') or ['GET', 'POST', 'PUT']
        headers = cors_cfg.get('allow_headers') or ['*']
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=methods,
            allow_headers=headers,
            allow_credentials=True,
        )

    def _issue_token(self) -> str:
        token = secrets.token_urlsafe(32)
        self._tokens[token] = time.time() + self._token_ttl_s
        return token

    def _prune_tokens(self) -> None:
        now = time.time()
        expired = [token for token, expires_at in self._tokens.items() if expires_at <= now]
        for token in expired:
            self._tokens.pop(token, None)

    def _check_token(self, token: str | None) -> bool:
        if not token:
            return False
        self._prune_tokens()
        expires_at = self._tokens.get(token)
        return bool(expires_at and expires_at > time.time())

    async def _require_auth(self, request: Request) -> None:
        if not self._auth_enabled():
            return
        auth_header = request.headers.get('Authorization', '')
        token = request.headers.get('X-Session-Token')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:].strip()
        if not self._check_token(token):
            raise HTTPException(status_code=401, detail='Authentication required')

    def _config_payload(self) -> dict[str, Any]:
        return {
            'operator': _redact(self.config.get('operator', {})),
            'playout': _redact(self.config.get('playout', {})),
            'same': _redact(self.config.get('same', {})),
            'cap': _redact(self.config.get('cap', {})),
            'feeds': _redact(self.config.get('feeds', [])),
            'tts': _redact(self.config.get('tts', {})),
            'webpanel': _redact(self._panel_cfg()),
        }

    def _feed_output_modes(self, feed: dict[str, Any]) -> list[str]:
        output_cfg = feed.get('output', {})
        enabled: list[str] = []
        for key in ('stream', 'audio_device', 'file', 'PiFmAdv'):
            if _coerce_mapping(output_cfg.get(key, {})).get('enabled'):
                enabled.append(key)
        return enabled

    def _feeds_payload(self) -> list[dict[str, Any]]:
        runtime = snapshot_runtime().get('feeds', {})
        sequences = snapshot_playout_sequences()
        queue_depths = snapshot_alert_queues()
        data_pool = snapshot_data_pool()

        feeds: list[dict[str, Any]] = []
        for feed in self.config.get('feeds', []):
            feed_id = feed.get('id', 'unknown')
            feed_data_keys = sorted(key for key in data_pool if str(key).startswith(f'{feed_id}:'))
            sequence = [item.pkg_id for item in sequences.get(feed_id, [])]
            clc_codes = feed_same_codes(feed)
            location_count = 0
            for block in feed.get('locations', []):
                if not isinstance(block, dict):
                    continue
                location_count += len([entry for entry in block.get('observationLocations', []) if isinstance(entry, dict)])
                location_count += len([entry for entry in block.get('forecastLocations', []) if isinstance(entry, dict)])
            feeds.append({
                'id': feed_id,
                'name': feed.get('name', feed_id),
                'enabled': bool(feed.get('enabled', True)),
                'timezone': feed.get('timezone', 'UTC'),
                'languages': list(feed.get('languages', {}).keys()) or [feed.get('language', 'en-CA')],
                'location_count': location_count,
                'outputs': self._feed_output_modes(feed),
                'playlist_items': sequence,
                'playlist_count': len(sequence),
                'alert_queue_depth': queue_depths.get(feed_id, 0),
                'data_keys': feed_data_keys,
                'clc_codes': clc_codes,
                'runtime': _serialize(runtime.get(feed_id, {})),
            })
        return feeds

    def _summary_payload(self) -> dict[str, Any]:
        runtime = snapshot_runtime()
        data_pool = snapshot_data_pool()
        feeds = self._feeds_payload()
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

    def _datapool_payload(self) -> dict[str, Any]:
        return {
            key: _serialize(value)
            for key, value in sorted(snapshot_data_pool().items(), key=lambda item: item[0])
        }

    def _log_file_for_source(self, source: str) -> pathlib.Path:
        if source == 'same':
            return self.root_dir / 'logs' / 'same.log'
        log_cfg = _coerce_mapping(self.config.get('logging', {}))
        file_cfg = _coerce_mapping(log_cfg.get('file', {}))
        path = file_cfg.get('path') or './logs/haze-weather-radio.log'
        text_path = str(path)
        if text_path.startswith('/'):
            return pathlib.Path(text_path)
        return (self.root_dir / text_path).resolve()

    def _register_routes(self) -> None:
        api_base = self._api_base()
        self.app.add_api_route('/', self.index, methods=['GET'])
        self.app.add_api_route('/panel', self.panel, methods=['GET'])
        self.app.add_api_route(f'{api_base}/health', self.health, methods=['GET'])
        self.app.add_api_route(f'{api_base}/auth/login', self.login, methods=['POST'])
        self.app.add_api_route(
            f'{api_base}/summary',
            self.summary,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/config',
            self.config_payload,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/feeds',
            self.feeds_payload,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/datapool',
            self.datapool_payload,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/events',
            self.events_payload,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route(
            f'{api_base}/logs',
            self.logs_payload,
            methods=['GET'],
            dependencies=[Depends(self._require_auth)],
        )
        self.app.add_api_route('/editor', self.editor_page, methods=['GET'])
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
        wx_base = self._wx_base()
        self.app.add_api_route(wx_base,                   self.wx_root,     methods=['GET'])
        self.app.add_api_route(f'{wx_base}/generate',     self.wx_generate, methods=['POST'])
        self.app.add_api_route(f'{wx_base}/packages',     self.wx_packages, methods=['GET'])

    async def index(self) -> FileResponse:
        return FileResponse(self.webroot / 'index.html')

    async def panel(self) -> FileResponse:
        return FileResponse(self.webroot / 'index.html')

    async def health(self) -> dict[str, Any]:
        return {
            'ok': True,
            'auth_required': self._auth_enabled(),
            'wx_base': self._wx_base(),
            'started_at': datetime.fromtimestamp(self.started_at, timezone.utc).isoformat(),
            'uptime_seconds': round(time.time() - self.started_at, 1),
        }

    async def login(self, payload: LoginRequest) -> dict[str, Any]:
        configured_password = str(self._auth_cfg().get('password') or '')
        if self._auth_enabled() and configured_password and payload.password != configured_password:
            raise HTTPException(status_code=401, detail='Invalid password')

        token = self._issue_token()
        append_runtime_event('auth', 'Web session established')
        return {
            'token': token,
            'expires_in_seconds': self._token_ttl_s,
            'auth_required': self._auth_enabled(),
        }

    async def summary(self) -> dict[str, Any]:
        return self._summary_payload()

    async def config_payload(self) -> dict[str, Any]:
        return self._config_payload()

    async def feeds_payload(self) -> list[dict[str, Any]]:
        return self._feeds_payload()

    async def datapool_payload(self) -> dict[str, Any]:
        return self._datapool_payload()

    async def events_payload(self) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]], snapshot_runtime().get('events', []))

    async def logs_payload(self, source: str = 'app', lines: int = 120) -> dict[str, Any]:
        bounded_lines = max(10, min(lines, 500))
        path = self._log_file_for_source(source)
        return {
            'source': source,
            'path': str(path),
            'lines': _tail_lines(path, bounded_lines),
        }

    async def editor_page(self) -> FileResponse:
        return FileResponse(self.webroot / 'editor.html')

    async def same_page(self) -> FileResponse:
        return FileResponse(self.webroot / 'same.html')

    async def managed_read(self, filename: str) -> dict[str, Any]:
        if filename not in _MANAGED_ALLOWLIST:
            raise HTTPException(status_code=404, detail='File not available')
        path = self.root_dir / 'managed' / filename
        if not path.exists():
            raise HTTPException(status_code=404, detail='File not found')
        return {'filename': filename, 'content': path.read_text(encoding='utf-8')}

    async def managed_write(self, filename: str, payload: FileWriteRequest) -> dict[str, Any]:
        if filename not in _MANAGED_ALLOWLIST:
            raise HTTPException(status_code=404, detail='File not available')
        path = self.root_dir / 'managed' / filename
        if filename.endswith('.json'):
            try:
                json.loads(payload.content)
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=400, detail=f'Invalid JSON: {exc}') from exc
        path.write_text(payload.content, encoding='utf-8')
        append_runtime_event('editor', f'Managed file saved via web panel: {filename}')
        return {'ok': True, 'filename': filename}

    async def same_event_codes(self) -> dict[str, Any]:
        path = self.root_dir / 'managed' / 'sameMapping.json'
        with open(path, encoding='utf-8') as f:
            same_mapping = json.load(f)
            return same_mapping.get('eas', {})

    async def same_templates_get(self) -> dict[str, Any]:
        path = self.root_dir / 'managed' / 'sameTemplate.json'
        if not path.exists():
            path = self.root_dir / 'managed' / 'sameTest.json'
        if not path.exists():
            return {}
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    async def same_templates_put(self, payload: FileWriteRequest) -> dict[str, Any]:
        try:
            data = json.loads(payload.content)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail=f'Invalid JSON: {exc}') from exc
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail='Templates must be a JSON object')
        path = self.root_dir / 'managed' / 'sameTemplate.json'
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        append_runtime_event('editor', 'SAME templates saved via web panel')
        return {'ok': True}

    async def upload_audio(self, file: UploadFile = File(...)) -> dict[str, Any]:
        import subprocess
        same_sr: int = int(self.config.get('same', {}).get('sample_rate_hz', 22050))
        upload_dir = self.root_dir / 'audio' / '_uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = secrets.token_hex(10)
        original_suffix = pathlib.Path(file.filename or 'audio.bin').suffix or '.bin'
        input_path = upload_dir / f'{safe_stem}_in{original_suffix}'
        output_path = upload_dir / f'{safe_stem}.wav'
        content = await file.read()

        audio_filters = (
            'acompressor=threshold=-20dB:ratio=4:attack=5:release=50:makeup=10dB,'
            'loudnorm=I=-9.0:LRA=7:TP=-2.0'
        )

        if len(content) > 150 * 1024 * 1024:
            raise HTTPException(status_code=413, detail='File too large (max 150 MB)')
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
        import os
        from module.same import SAMEHeader, generate_same, to_pcm16

        feeds_cfg = self.config.get('feeds', [])
        callsign = self.config.get('same', {}).get('sender', 'HAZE0000')

        if payload.air_on_all_feeds:
            target_ids = [f.get('id', 'default') for f in feeds_cfg if f.get('enabled', True)]
            if not target_ids:
                raise HTTPException(status_code=400, detail='No enabled feeds configured')
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
            locations=locations[:31],
            duration=duration_code,
            callsign=callsign,
        )
        tone: str | None = payload.tone_type.upper() if payload.tone_type.upper() != 'NONE' else None

        voice_path: pathlib.Path | None = None
        voice_array = None
        if payload.audio_file_path.strip():
            upload_dir = self.root_dir / 'audio' / '_uploads'
            candidate = pathlib.Path(payload.audio_file_path)
            try:
                candidate.relative_to(upload_dir)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail='Invalid audio file path') from exc
            if not candidate.exists():
                raise HTTPException(status_code=400, detail='Audio file not found')
            voice_path = candidate
        elif payload.voice_message.strip():
            from module.tts import synthesize_pcm
            from module.buffer import CHANNELS, SAMPLE_RATE as BUS_SR
            from module.same import resample as _resample
            import numpy as np
            pcm = synthesize_pcm(self.config, payload.voice_message.strip())
            if pcm:
                samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
                if CHANNELS == 2:
                    samples = samples.reshape(-1, 2).mean(axis=1)
                same_sr_v = int(self.config.get('same', {}).get('sample_rate_hz', 22050))
                voice_array = _resample(samples, BUS_SR, same_sr_v)

        same_sr: int = int(self.config.get('same', {}).get('sample_rate_hz', 22050))
        full_signal = generate_same(
            header=header,
            tone_type=tone,
            audio_msg_path=voice_path,
            audio_msg_array=voice_array,
            sample_rate=same_sr,
            attn_duration_s=8.0,
        )

        from module.buffer import SAMPLE_RATE as _BUS_SR
        from module.same import resample as _resample2
        alert_pcm = to_pcm16(_resample2(full_signal, same_sr, _BUS_SR))

        from managed.events import push_alert
        for fid in target_ids:
            push_alert(fid, 0, alert_pcm, f'manual_{int(time.time())}')

        encoded = header.encoded
        append_runtime_event('manual-same', f'Manual SAME aired: {encoded} → {", ".join(target_ids)}')
        return {'ok': True, 'header': encoded, 'feed_id': target_ids[0], 'feeds_aired': target_ids}

    async def wx_packages(self) -> dict[str, Any]:
        return {'packages': list(_ALL_PACKAGES)}

    async def wx_root(self) -> FileResponse:
        return FileResponse(self.webroot / 'wx.html')

    async def wx_generate(self, payload: WXGenerateRequest) -> StreamingResponse:
        from module.tts import synthesize_pcm_stream
        from managed.packages import (
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
            registry_raw = (_pl.Path('data') / 'alertsRegistry.json').read_text(encoding='utf-8')
            registry: list[Any] = _json.loads(registry_raw)
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

        if fmt in _TEXT_FORMATS:
            pkg_texts = {pkg_id: pkg_lookup.get(pkg_id, '') for pkg_id in requested}
            if fmt == 'json':
                body = _json.dumps({'packages': pkg_texts, 'locations': requested_locs, 'lang': lang}).encode()
            elif fmt == 'xml':
                parts = [f'  <package id="{p}">{t}</package>' for p, t in pkg_texts.items() if t]
                body = ('<?xml version="1.0" encoding="UTF-8"?>\n<wx>\n' + '\n'.join(parts) + '\n</wx>').encode()
            elif fmt == 'ssml':
                paras = '\n  '.join(f'<p>{t}</p>' for t in pkg_texts.values() if t)
                body = (
                    f'<speak version="1.1" xmlns="http://www.w3.org/2001/10/synthesis"'
                    f' xml:lang="{lang}">\n  {paras}\n</speak>'
                ).encode()
            elif fmt == 'html':
                rows = '\n'.join(
                    f'<section id="{p}"><h2>{p.replace("_", " ").title()}</h2><p>{t}</p></section>'
                    for p, t in pkg_texts.items() if t
                )
                body = (
                    f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
                    f'<title>Weather Report</title></head><body>{rows}</body></html>'
                ).encode()
            elif fmt == 'markdown':
                body = '\n\n'.join(
                    f'## {p.replace("_", " ").title()}\n\n{t}' for p, t in pkg_texts.items() if t
                ).encode()
            else:
                sections = '\n\n'.join(
                    f'\\section{{{p.replace("_", " ").title()}}}\n{t}' for p, t in pkg_texts.items() if t
                )
                body = (
                    '\\documentclass{article}\n\\usepackage[utf8]{inputenc}\n'
                    '\\begin{document}\n\\title{Weather Report}\n\\maketitle\n'
                    f'{sections}\n\\end{{document}}'
                ).encode()
            return StreamingResponse(
                iter([body]),
                media_type=_FORMAT_MEDIA_TYPES[fmt],
                headers={
                    'X-Packages':     ','.join(requested),
                    'X-Format':       fmt,
                    'X-Source':       source_label,
                    'Content-Length': str(len(body)),
                },
            )

        from module.buffer import SAMPLE_RATE as _BUS_SR, CHANNELS as _BUS_CH

        def _pcm_generator():
            for pkg_id in requested:
                text = pkg_lookup.get(pkg_id, '')
                if not text:
                    continue
                yield from synthesize_pcm_stream(self.config, text, lang=lang, voice=voice)

        common_headers = {
            'X-Audio-Sample-Rate': str(_BUS_SR),
            'X-Audio-Channels':    str(_BUS_CH),
            'X-Packages':          ','.join(requested),
            'X-Format':            fmt,
            'X-Source':            source_label,
        }

        if fmt == 'raw':
            return StreamingResponse(
                _pcm_generator(),
                media_type='audio/raw',
                headers={'X-Audio-Encoding': 's16le', **common_headers},
            )

        pcm = b''.join(_pcm_generator())

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
        panel_cfg = self._panel_cfg()
        host = str(panel_cfg.get('host') or '0.0.0.0')
        port = int(panel_cfg.get('port') or 8080)

        update_runtime_status({
            'webpanel_enabled': True,
            'webpanel_host': host,
            'webpanel_port': port,
        })
        append_runtime_event('web', f'Web panel listening on {host}:{port}')

        def _runner() -> None:
            try:
                import uvicorn
            except ModuleNotFoundError:
                log.exception('uvicorn is not installed for the active Python interpreter')
                append_runtime_event('web-error', 'uvicorn is not installed for the active Python interpreter')
                return

            uvicorn.run(self.app, host=host, port=port, log_level='info')

        return threading.Thread(target=_runner, name='webpanel', daemon=True)


def start_web_server(config: dict[str, Any], feeds: list[dict[str, Any]] | None = None) -> threading.Thread | None:
    panel_cfg = _coerce_mapping(config.get('webpanel', config.get('web', {})))
    if not panel_cfg.get('enabled', False):
        return None

    server = WebServer(config, feeds)
    log.info('Starting web panel on %s:%s', panel_cfg.get('host', '0.0.0.0'), panel_cfg.get('port', 8080))
    return server.start()
from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import secrets
import threading
import time
import wave
from collections import deque
from datetime import datetime, timezone
from typing import Any, cast

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from managed.events import (
    append_runtime_event,
    snapshot_alert_queues,
    snapshot_data_pool,
    snapshot_playout_sequences,
    snapshot_runtime,
    update_runtime_status,
)
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


_MANAGED_ALLOWLIST: frozenset[str] = frozenset({'userbulletins.json', 'dictionary.json', 'packages.py', 'sameTemplate.json'})


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
            sequence = [path.name for path in sequences.get(feed_id, [])]
            clc_codes = feed_same_codes(feed)
            feeds.append({
                'id': feed_id,
                'name': feed.get('name', feed_id),
                'enabled': bool(feed.get('enabled', True)),
                'timezone': feed.get('timezone', 'UTC'),
                'languages': list(feed.get('languages', {}).keys()) or [feed.get('language', 'en-CA')],
                'location_count': len(feed.get('locations', [])),
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
        self.app.add_api_route('/same', self.same_page, methods=['GET'])
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

    async def index(self) -> FileResponse:
        return FileResponse(self.webroot / 'index.html')

    async def panel(self) -> FileResponse:
        return FileResponse(self.webroot / 'index.html')

    async def health(self) -> dict[str, Any]:
        return {
            'ok': True,
            'auth_required': self._auth_enabled(),
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
            return json.load(f)

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
        upload_dir = self.root_dir / 'output' / '_uploads'
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = secrets.token_hex(10)
        original_suffix = pathlib.Path(file.filename or 'audio.bin').suffix or '.bin'
        input_path = upload_dir / f'{safe_stem}_in{original_suffix}'
        output_path = upload_dir / f'{safe_stem}.wav'
        content = await file.read()
        if len(content) > 150 * 1024 * 1024:
            raise HTTPException(status_code=413, detail='File too large (max 150 MB)')
        input_path.write_bytes(content)
        result = subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-i', str(input_path),
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
        if payload.audio_file_path.strip():
            upload_dir = self.root_dir / 'output' / '_uploads'
            candidate = pathlib.Path(payload.audio_file_path)
            try:
                candidate.relative_to(upload_dir)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail='Invalid audio file path') from exc
            if not candidate.exists():
                raise HTTPException(status_code=400, detail='Audio file not found')
            voice_path = candidate
        elif payload.voice_message.strip():
            from module.tts import synthesize
            voice_path = synthesize(self.config, payload.voice_message.strip(), target_ids[0], 'manual_same')

        same_sr: int = int(self.config.get('same', {}).get('sample_rate_hz', 22050))
        full_signal = generate_same(
            header=header,
            tone_type=tone,
            audio_msg_fp32=voice_path,
            sample_rate=same_sr,
            attn_duration_s=8.0,
        )
        out_dir = self.root_dir / 'output' / target_ids[0] / 'alerts'
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = out_dir / f'manual_{ts}.wav'
        tmp_path = out_dir / f'manual_{ts}.tmp.wav'
        with wave.open(str(tmp_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(same_sr)
            wf.writeframes(to_pcm16(full_signal))
        os.replace(str(tmp_path), str(out_path))

        from managed.events import push_alert
        for fid in target_ids:
            push_alert(fid, 0, out_path)

        encoded = header.encode()
        append_runtime_event('manual-same', f'Manual SAME aired: {encoded} → {", ".join(target_ids)}')
        return {'ok': True, 'header': encoded, 'feed_id': target_ids[0], 'feeds_aired': target_ids, 'path': str(out_path)}

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
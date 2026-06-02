from __future__ import annotations

import asyncio
import logging
import pathlib
import socket
import subprocess
import threading
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Coroutine

try:
    from discord_webhook import AsyncDiscordWebhook, DiscordEmbed
    _discord_webhook_available = True
except ImportError:
    AsyncDiscordWebhook = None  # type: ignore[assignment,misc]
    DiscordEmbed = None  # type: ignore[assignment,misc]
    _discord_webhook_available = False

from module.banner import serialize_alert
from module.feed_util import coverage_regions, forecast_locations
from module.packages import _load_forecast_loc_db

log = logging.getLogger(__name__)

_TEST_EVENT_CODES: frozenset[str] = frozenset({'RWT', 'RMT', 'DMO'})
_ADMIN_EVENT_CODES: frozenset[str] = frozenset({'NMN', 'ADR', 'TXP', 'TXF', 'TXO'})

_WEBHOOKS_XML = pathlib.Path(__file__).resolve().parent.parent / 'managed' / 'configs' / 'webhooks.xml'

_configs_lock = threading.Lock()
_configs_cache: list[WebhookConfig] | None = None
_configs_mtime: float = 0.0


@dataclass
class _EmbedAudio:
    enabled: bool = False
    codec: str = 'libopus'


@dataclass
class WebhookConfig:
    feed_id: str
    enabled: bool
    url: str
    username: str = 'Haze Weather Radio'
    icon_url: str = ''
    log_test_alerts: bool = False
    log_admin_alerts: bool = False
    embed_audio: _EmbedAudio = field(default_factory=_EmbedAudio)


def _load_configs() -> list[WebhookConfig]:
    global _configs_cache, _configs_mtime

    if not _WEBHOOKS_XML.exists():
        return []

    try:
        mtime = _WEBHOOKS_XML.stat().st_mtime
    except OSError:
        return []

    with _configs_lock:
        if _configs_cache is not None and mtime == _configs_mtime:
            return list(_configs_cache)

    try:
        tree = ET.parse(_WEBHOOKS_XML)
    except ET.ParseError as exc:
        log.warning('Failed to parse webhooks.xml: %s', exc)
        return []

    configs: list[WebhookConfig] = []
    for elem in tree.getroot().findall('Webhook'):
        feed_id = (elem.get('feed-id') or '').strip()
        if not feed_id:
            continue

        audio_elem = elem.find('EmbedAudio')
        embed_audio = _EmbedAudio()
        if audio_elem is not None:
            embed_audio.enabled = audio_elem.get('enabled', 'false').lower() == 'true'
            embed_audio.codec = (audio_elem.get('codec') or 'libopus').strip()

        configs.append(WebhookConfig(
            feed_id=feed_id,
            enabled=elem.get('enabled', 'false').lower() == 'true',
            url=(elem.findtext('WebhookURL') or '').strip(),
            username=(elem.findtext('Username') or 'Haze Weather Radio').strip(),
            icon_url=(elem.findtext('IconURL') or '').strip(),
            log_test_alerts=(elem.findtext('LogTestAlerts') or 'false').strip().lower() == 'true',
            log_admin_alerts=(elem.findtext('LogAdminAlerts') or 'false').strip().lower() == 'true',
            embed_audio=embed_audio,
        ))

    with _configs_lock:
        _configs_cache = configs
        _configs_mtime = mtime

    return list(configs)


_CODEC_EXT: dict[str, str] = {
    'libopus':    'ogg',
    'libmp3lame': 'mp3',
    'aac':        'aac',
    'libvorbis':  'ogg',
    'pcm_s16le':  'wav',
    'wav':        'wav',
}

_WAV_CODECS: frozenset[str] = frozenset({'pcm_s16le', 'wav'})

_STARTUP_COLOR = 0x2ECC71
_MAX_DISCORD_ATTACHMENT_BYTES = 8 * 1024 * 1024


def _banner_color_int(value: Any) -> int:
    text = str(value or '').strip().lstrip('#')
    if not text:
        return 0x888888
    try:
        return int(text, 16)
    except ValueError:
        return 0x888888


def _audio_bytes_for_attach(src: pathlib.Path, codec: str) -> tuple[bytes, str] | None:
    ext = _CODEC_EXT.get(codec, 'ogg')
    if codec in _WAV_CODECS:
        try:
            payload = src.read_bytes()
            if len(payload) > _MAX_DISCORD_ATTACHMENT_BYTES:
                log.warning('Webhook audio attachment skipped: %s exceeds %d bytes', src, _MAX_DISCORD_ATTACHMENT_BYTES)
                return None
            return payload, 'wav'
        except OSError as exc:
            log.warning('Webhook audio read failed: %s', exc)
            return None
    try:
        result = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-i', str(src),
             '-c:a', codec, '-f', ext, 'pipe:1'],
            capture_output=True, timeout=60,
        )
        if result.returncode != 0:
            log.warning(
                'Webhook audio transcode failed: %s',
                result.stderr.decode('utf-8', errors='replace')[:200],
            )
            return None
        if len(result.stdout) > _MAX_DISCORD_ATTACHMENT_BYTES:
            log.warning('Webhook audio attachment skipped: transcoded payload exceeds %d bytes', _MAX_DISCORD_ATTACHMENT_BYTES)
            return None
        return result.stdout, ext
    except Exception as exc:
        log.warning('Webhook audio transcode error: %s', exc)
        return None


def _clip_discord_text(value: Any, limit: int) -> str:
    text = str(value or '').strip()
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3].rstrip() + '...'


def _code_block_text(value: Any, limit: int = 1024) -> str:
    normalized = str(value or '').replace('```', "'''").strip()
    if not normalized:
        return ''
    inner_limit = max(1, limit - 8)
    return f"```\n{_clip_discord_text(normalized, inner_limit)}\n```"


def _response_status_codes(resp: Any) -> list[int | None]:
    responses = resp if isinstance(resp, list) else [resp]
    return [getattr(response, 'status_code', None) for response in responses]


def _response_detail_text(response: Any) -> str:
    text = getattr(response, 'text', None)
    if isinstance(text, str):
        return text
    content = getattr(response, 'content', None)
    if isinstance(content, bytes):
        return content.decode('utf-8', errors='replace')
    return str(content or '')


def _log_execute_result(cfg: WebhookConfig, resp: Any, label: str) -> None:
    responses = resp if isinstance(resp, list) else [resp]
    failing_responses = [
        response
        for response in responses
        if getattr(response, 'status_code', None) not in (None, 200, 204)
    ]
    if failing_responses:
        for response in failing_responses:
            status_code = getattr(response, 'status_code', None)
            detail = _clip_discord_text(_response_detail_text(response), 240)
            if detail:
                log.error('[%s] Discord webhook HTTP %s for %s: %s', cfg.feed_id, status_code, label, detail)
            else:
                log.error('[%s] Discord webhook HTTP %s for %s', cfg.feed_id, status_code, label)
        return
    log.info('[%s] Discord webhook dispatched: %s', cfg.feed_id, label)


def _run_async(coro: Coroutine[Any, Any, None]) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(coro)
        return
    loop.create_task(coro)


def _icecast_public_url(feed: dict[str, Any]) -> str | None:
    output_cfg = feed.get('output')
    if not isinstance(output_cfg, dict):
        return None
    stream_cfg = output_cfg.get('stream')
    if not (isinstance(stream_cfg, dict) and stream_cfg.get('enabled')):
        return None
    if str(stream_cfg.get('type') or 'icecast').strip().lower() != 'icecast':
        return None
    host = str(stream_cfg.get('host') or '').strip()
    if not host:
        return None
    scheme = 'https' if stream_cfg.get('ssl') else 'http'
    default_port = 443 if scheme == 'https' else 80
    try:
        port = int(stream_cfg.get('port') or default_port)
    except (TypeError, ValueError):
        port = default_port
    mount = str(stream_cfg.get('mount') or f"/{feed.get('id', 'stream')}").strip() or f"/{feed.get('id', 'stream')}"
    if not mount.startswith('/'):
        mount = f'/{mount}'
    port_text = '' if port == default_port else f':{port}'
    return f'{scheme}://{host}{port_text}{mount}'


def _host_identity() -> tuple[str, str]:
    hostname = socket.gethostname() or 'unknown-host'
    addresses: list[str] = []
    seen: set[str] = set()

    def _add(ip: str) -> None:
        text = str(ip).strip()
        if not text or text.startswith('127.') or text in seen:
            return
        seen.add(text)
        addresses.append(text)

    try:
        for ip in socket.gethostbyname_ex(hostname)[2]:
            _add(ip)
    except OSError:
        pass

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(('8.8.8.8', 80))
            _add(str(sock.getsockname()[0]))
    except OSError:
        pass

    if not addresses:
        addresses.append('127.0.0.1')

    return hostname, ', '.join(addresses)


def _feed_cfg(config: dict[str, Any] | None, feed_id: str) -> dict[str, Any] | None:
    if not isinstance(config, dict):
        return None
    feeds = config.get('feeds')
    if not isinstance(feeds, list):
        return None
    for feed in feeds:
        if isinstance(feed, dict) and str(feed.get('id') or '').strip() == feed_id:
            return feed
    return None


def _feed_timezone(config: dict[str, Any] | None, feed_id: str) -> str:
    feed = _feed_cfg(config, feed_id)
    if isinstance(feed, dict):
        return str(feed.get('timezone') or 'UTC')
    return 'UTC'


def _banner_page_url(config: dict[str, Any] | None, feed_id: str) -> str | None:
    if not isinstance(config, dict):
        return None
    panel_cfg = config.get('webpanel', config.get('web', {}))
    if not isinstance(panel_cfg, dict) or not panel_cfg.get('enabled', False):
        return None
    host = str(panel_cfg.get('host') or '').strip()
    hostname, ip_text = _host_identity()
    if host in {'', '0.0.0.0', '::'}:
        host = next((candidate.strip() for candidate in ip_text.split(',') if candidate.strip()), hostname)
    if not host:
        return None
    try:
        port = int(panel_cfg.get('port') or 8080)
    except (TypeError, ValueError):
        port = 8080
    port_text = '' if port == 80 else f':{port}'
    query = urllib.parse.urlencode({'feed': feed_id})
    return f'http://{host}{port_text}/banner?{query}'


def _summarize_values(values: list[str], *, limit: int = 6, max_len: int = 900) -> str:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)

    if not unique:
        return 'Not configured'

    shown = unique[:limit]
    summary = ', '.join(shown)
    remaining = len(unique) - len(shown)
    if remaining > 0:
        summary = f'{summary}, +{remaining} more'
    if len(summary) > max_len:
        summary = summary[: max_len - 3].rstrip(', ') + '...'
    return summary


def _area_labels(feed: dict[str, Any]) -> list[str]:
    forecast_db = _load_forecast_loc_db()
    labels: list[str] = []

    for region in coverage_regions(feed):
        if str(region.get('coverage_type') or '').strip().lower() != 'region':
            continue
        label = str(region.get('name_override') or '').strip()
        derive_forecast = str(region.get('derive_forecast') or '').strip()
        if not label and derive_forecast:
            label = forecast_db.get(derive_forecast, ('', ''))[0].strip()
        if not label:
            label = derive_forecast or str(region.get('id') or '').strip()
        if label:
            labels.append(label)

    if labels:
        return labels

    for loc in forecast_locations(feed):
        label = str(loc.get('name_override') or loc.get('name') or '').strip()
        if not label:
            loc_id = str(loc.get('id') or '').strip()
            if loc_id:
                label = forecast_db.get(loc_id, ('', ''))[0].strip() or loc_id
        if label:
            labels.append(label)

    if labels:
        return labels

    for transmitter in feed.get('transmitter_metadata', []):
        if not isinstance(transmitter, dict):
            continue
        site_name = str(transmitter.get('site_name') or '').strip()
        if site_name:
            labels.append(site_name)

    return labels


def _alert_embed_data(
    entry: dict[str, Any],
    same_event: str,
    feed_id: str,
    config: dict[str, Any] | None,
) -> dict[str, str]:
    metadata = entry.get('metadata') or {}
    source = entry.get('source') or {}
    text_block = entry.get('text') or {}
    tz_name = _feed_timezone(config, feed_id)
    serialized = serialize_alert(entry, tz_name)

    raw_description = str(text_block.get('description') or '').strip()
    generated_description = str(serialized.get('message') or '').strip()
    if raw_description and len(raw_description) <= 4096:
        description = raw_description
    else:
        description = generated_description or raw_description

    feed_cfg = _feed_cfg(config, feed_id)
    listen_url = _icecast_public_url(feed_cfg) if isinstance(feed_cfg, dict) else None

    return {
        'headline': str(serialized.get('headline') or metadata.get('headline') or metadata.get('event') or same_event).strip(),
        'event': str(serialized.get('event') or metadata.get('event') or same_event).strip(),
        'severity': str(serialized.get('severity') or metadata.get('severity') or 'Unknown').strip(),
        'expires': str(serialized.get('expires_display') or metadata.get('expires') or '').strip(),
        'description': description,
        'instruction': str(text_block.get('instruction') or '').strip(),
        'listen_url': str(listen_url or '').strip(),
        'banner_color': str(serialized.get('background_color') or '').strip(),
        'same_header': str(source.get('sameHeader') or metadata.get('sameHeader') or '').strip(),
        'same_message': str(serialized.get('message') or '').strip(),
    }


def _build_alert_webhook(
    cfg: WebhookConfig,
    feed_id: str,
    headline: str,
    event: str,
    severity: str,
    expires: str,
    description: str,
    instruction: str,
    listen_url: str,
    banner_color: str,
    same_header: str,
    same_message: str,
    identifier: str,
    same_event: str,
    audio_path: pathlib.Path | None,
) -> tuple[Any, bool]:
    embed_color = _banner_color_int(banner_color)

    wh = AsyncDiscordWebhook(url=cfg.url, username=cfg.username, avatar_url=cfg.icon_url or None, rate_limit_retry=True)
    embed = DiscordEmbed(
        title=_clip_discord_text(headline, 256),
        description=_clip_discord_text(description, 4096) or None,
        color=embed_color,
    )
    embed.add_embed_field(name='Feed', value=_clip_discord_text(feed_id or cfg.feed_id, 1024), inline=True)
    embed.add_embed_field(name='Event', value=_clip_discord_text(event, 1024), inline=True)
    embed.add_embed_field(name='Severity', value=_clip_discord_text(severity, 1024), inline=True)
    if expires:
        embed.add_embed_field(name='Expires', value=_clip_discord_text(expires, 1024), inline=True)
    if instruction:
        embed.add_embed_field(name='Instruction', value=_clip_discord_text(instruction, 1024), inline=False)
    if same_header:
        embed.add_embed_field(name='SAME Header', value=_code_block_text(same_header), inline=False)
    if same_message:
        embed.add_embed_field(name='SAME Message', value=_code_block_text(same_message), inline=False)
    if listen_url:
        embed.add_embed_field(name='Listen', value=_clip_discord_text(listen_url, 1024), inline=False)
    if identifier:
        embed.set_footer(text=_clip_discord_text(identifier, 2048))
    embed.set_timestamp()
    wh.add_embed(embed)

    attached_audio = False
    if cfg.embed_audio.enabled and audio_path and audio_path.exists():
        result = _audio_bytes_for_attach(audio_path, cfg.embed_audio.codec)
        if result:
            audio_bytes, ext = result
            wh.add_file(file=audio_bytes, filename=f'alert_{same_event.lower()}.{ext}')
            attached_audio = True

    return wh, attached_audio


async def _send(
    cfg: WebhookConfig,
    entry: dict[str, Any],
    same_event: str,
    audio_path: pathlib.Path | None,
    config: dict[str, Any] | None,
) -> None:
    if not _discord_webhook_available:
        log.error('discord_webhook package not installed — cannot send webhook. Run: pip install discord-webhook')
        return
    feed_id = str(entry.get('feed_id') or cfg.feed_id).strip() or cfg.feed_id
    identifier = entry.get('identifier') or ''
    embed_data = _alert_embed_data(entry, same_event, feed_id, config)

    wh, attached_audio = _build_alert_webhook(
        cfg,
        feed_id,
        embed_data['headline'],
        embed_data['event'],
        embed_data['severity'],
        embed_data['expires'],
        embed_data['description'],
        embed_data['instruction'],
        embed_data['listen_url'],
        embed_data['banner_color'],
        embed_data['same_header'],
        embed_data['same_message'],
        str(identifier),
        same_event,
        audio_path,
    )
    resp = await wh.execute()
    if attached_audio and any(status_code in {400, 413} for status_code in _response_status_codes(resp)):
        log.warning('[%s] Discord webhook rejected alert attachment, retrying without audio', cfg.feed_id)
        wh, _ = _build_alert_webhook(
            cfg,
            feed_id,
            embed_data['headline'],
            embed_data['event'],
            embed_data['severity'],
            embed_data['expires'],
            embed_data['description'],
            embed_data['instruction'],
            embed_data['listen_url'],
            embed_data['banner_color'],
            embed_data['same_header'],
            embed_data['same_message'],
            str(identifier),
            same_event,
            None,
        )
        resp = await wh.execute()

    _log_execute_result(cfg, resp, f"{same_event} ({identifier or embed_data['headline']})")


async def dispatch_startup_webhook_async(feed: dict[str, Any]) -> None:
    if not _discord_webhook_available:
        log.error('discord_webhook package not installed — cannot send webhook. Run: pip install discord-webhook')
        return

    feed_id = str(feed.get('id') or '').strip()
    if not feed_id:
        return

    targets = [cfg for cfg in _load_configs() if cfg.enabled and cfg.feed_id == feed_id and cfg.url]
    if not targets:
        return

    stream_url = _icecast_public_url(feed)
    hostname, ip_text = _host_identity()
    area_text = _summarize_values(_area_labels(feed))
    callsign = str(feed.get('callsign') or '').strip()

    for cfg in targets:
        wh = AsyncDiscordWebhook(url=cfg.url, username=cfg.username, avatar_url=cfg.icon_url or None, rate_limit_retry=True)
        embed = DiscordEmbed(
            title='Haze Weather Radio is online',
            description=f'Feed {feed_id} is now on the air.',
            color=_STARTUP_COLOR,
        )
        embed.add_embed_field(name='Feed', value=feed_id, inline=True)
        if callsign:
            embed.add_embed_field(name='Callsign', value=callsign, inline=True)
        if stream_url:
            embed.add_embed_field(name='Listen', value=stream_url, inline=False)
        embed.add_embed_field(name='Areas Served', value=area_text, inline=False)
        embed.add_embed_field(name='Hostname', value=hostname, inline=True)
        embed.add_embed_field(name='IP', value=ip_text, inline=True)
        embed.set_timestamp()
        wh.add_embed(embed)
        _log_execute_result(cfg, await wh.execute(), f'startup online ({feed_id})')


def dispatch_startup_webhook(feed: dict[str, Any]) -> None:
    _run_async(dispatch_startup_webhook_async(feed))


async def dispatch_webhook_async(
    feed_id: str,
    entry: dict[str, Any],
    same_event: str,
    audio_path: pathlib.Path | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    configs = _load_configs()
    targets = [c for c in configs if c.enabled and c.feed_id == feed_id and c.url]
    if not targets:
        return

    event_upper = same_event.upper()
    is_test = event_upper in _TEST_EVENT_CODES
    is_admin = event_upper in _ADMIN_EVENT_CODES

    for cfg in targets:
        if is_test and not cfg.log_test_alerts:
            continue
        if is_admin and not cfg.log_admin_alerts:
            continue
        try:
            await _send(cfg, entry, same_event, audio_path, config)
        except Exception as exc:
            log.error('[%s] Discord webhook dispatch failed: %s', feed_id, exc)


def dispatch_webhook(
    feed_id: str,
    entry: dict[str, Any],
    same_event: str,
    audio_path: pathlib.Path | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    _run_async(dispatch_webhook_async(feed_id, entry, same_event, audio_path, config))

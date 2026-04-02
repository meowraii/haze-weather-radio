from __future__ import annotations

import asyncio
import logging
import random
import urllib.error
import urllib.parse
import urllib.request
from collections import OrderedDict
from typing import Any, Awaitable, Callable
from xml.etree import ElementTree as ET

from module.cap_specific.naads_tcp import CAPAlert, parse_cap

log = logging.getLogger(__name__)

ATOM_NS = 'http://www.w3.org/2005/Atom'

BASE_RECONNECT_DELAY = 5.0
MAX_RECONNECT_DELAY = 60.0
ENTRY_CACHE_SIZE = 4096

_TITLE_CASE_QUERY_KEYS = {'severity', 'urgency', 'certainty'}
_LOWER_CASE_QUERY_KEYS = {'status', 'message_type'}


def _a(tag: str) -> str:
    return f'{{{ATOM_NS}}}{tag}'


def _normalize_query_part(part: str) -> tuple[str, str] | None:
    key, sep, value = part.partition('=')
    key = key.strip()
    if not key:
        return None
    if not sep:
        return key, ''
    raw_value = value.strip()
    if key in _TITLE_CASE_QUERY_KEYS:
        raw_value = ','.join(item.strip().title() for item in raw_value.split(',') if item.strip())
    elif key in _LOWER_CASE_QUERY_KEYS:
        raw_value = ','.join(item.strip().lower() for item in raw_value.split(',') if item.strip())
    return key, raw_value


def _source_url(source: dict[str, Any]) -> str:
    base_url = str(source.get('url') or 'https://api.weather.gov/alerts/active.atom').strip()
    query_parts = source.get('queries') or source.get('querys') or []
    if not query_parts:
        return base_url
    params: list[tuple[str, str]] = []
    for part in query_parts:
        normalized = _normalize_query_part(str(part).strip())
        if normalized is not None:
            params.append(normalized)
    if not params:
        return base_url
    separator = '&' if '?' in base_url else '?'
    return f'{base_url}{separator}{urllib.parse.urlencode(params, doseq=True)}'


def _user_agent(config: dict[str, Any]) -> str:
    operator = config.get('operator', {})
    contact = operator.get('email') or operator.get('operator_name') or 'unknown'
    version = config.get('version', 'dev')
    return f'haze-weather-radio/{version} ({contact})'


def _http_get(url: str, timeout_s: float, headers: dict[str, str]) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        if exc.code == 304:
            return 304, b'', dict(exc.headers.items())
        raise


def _cap_urls(entry: ET.Element) -> list[str]:
    urls: list[str] = []
    for link in entry.findall(_a('link')):
        href = (link.get('href') or '').strip()
        if not href:
            continue
        link_type = (link.get('type') or '').lower()
        if 'cap+xml' in link_type:
            urls.append(href)
    for link in entry.findall(_a('link')):
        href = (link.get('href') or '').strip()
        if href and href not in urls:
            urls.append(href)
    entry_id = (entry.findtext(_a('id')) or '').strip()
    if entry_id and entry_id not in urls:
        urls.append(entry_id)

    expanded: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            expanded.append(url)
        if url.startswith('http') and not url.endswith('.cap'):
            cap_url = url.rstrip('/') + '.cap'
            if cap_url not in seen:
                seen.add(cap_url)
                expanded.append(cap_url)
    return expanded


def _parse_atom_entries(raw: bytes) -> list[tuple[str, str, list[str]]]:
    root = ET.fromstring(raw)
    entries: list[tuple[str, str, list[str]]] = []
    for entry in root.findall(_a('entry')):
        entry_id = (entry.findtext(_a('id')) or '').strip()
        updated = (entry.findtext(_a('updated')) or '').strip()
        urls = _cap_urls(entry)
        if entry_id and urls:
            entries.append((entry_id, updated, urls))
    return entries


async def _fetch_cap_alert(urls: list[str], timeout_s: float, headers: dict[str, str]) -> CAPAlert | None:
    cap_headers = {
        **headers,
        'Accept': 'application/cap+xml, application/xml;q=0.9, */*;q=0.1',
    }
    for url in urls:
        try:
            status, body, _ = await asyncio.to_thread(_http_get, url, timeout_s, cap_headers)
        except Exception:
            continue
        if status != 200 or not body:
            continue
        alert = parse_cap(body)
        if alert is not None:
            return alert
    return None


async def _poll_source(
    config: dict[str, Any],
    source: dict[str, Any],
    on_alert: Callable[[CAPAlert, dict[str, Any]], Awaitable[None]],
    shutdown: asyncio.Event,
) -> None:
    source_id = str(source.get('id') or 'nws')
    url = _source_url(source)
    network_cfg = config.get('network', {}).get('requests', {})
    timeout_s = float(source.get('timeout_seconds') or network_cfg.get('read_timeout_seconds') or 15)
    poll_interval = float(source.get('poll_interval_seconds') or 30)
    headers = {
        'User-Agent': _user_agent(config),
        'Accept': 'application/atom+xml, application/xml;q=0.9, */*;q=0.1',
    }

    etag: str | None = None
    last_modified: str | None = None
    seen_entries: OrderedDict[str, str] = OrderedDict()
    attempt = 0

    while not shutdown.is_set():
        try:
            request_headers = dict(headers)
            if etag:
                request_headers['If-None-Match'] = etag
            if last_modified:
                request_headers['If-Modified-Since'] = last_modified

            status, raw, response_headers = await asyncio.to_thread(_http_get, url, timeout_s, request_headers)
            if status != 304:
                etag = response_headers.get('ETag') or etag
                last_modified = response_headers.get('Last-Modified') or last_modified
                for entry_id, updated, urls in _parse_atom_entries(raw):
                    if seen_entries.get(entry_id) == updated:
                        continue
                    seen_entries[entry_id] = updated
                    seen_entries.move_to_end(entry_id)
                    while len(seen_entries) > ENTRY_CACHE_SIZE:
                        seen_entries.popitem(last=False)
                    alert = await _fetch_cap_alert(urls, timeout_s, headers)
                    if alert is not None:
                        await on_alert(alert, source)
            attempt = 0
        except ET.ParseError as exc:
            attempt += 1
            log.warning('[%s] NWS Atom parse failed: %s', source_id, exc)
        except RuntimeError as exc:
            if 'cannot schedule new futures after shutdown' in str(exc).lower():
                break
            attempt += 1
            delay = min(MAX_RECONNECT_DELAY, BASE_RECONNECT_DELAY * (2 ** max(attempt - 1, 0)))
            delay += random.uniform(0, 1.5)
            log.warning('[%s] NWS Atom poll failed, retry in %.1fs: %s', source_id, delay, exc)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue
            break
        except Exception as exc:
            attempt += 1
            delay = min(MAX_RECONNECT_DELAY, BASE_RECONNECT_DELAY * (2 ** max(attempt - 1, 0)))
            delay += random.uniform(0, 1.5)
            log.warning('[%s] NWS Atom poll failed, retry in %.1fs: %s', source_id, delay, exc)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
            except asyncio.TimeoutError:
                continue
            break

        try:
            await asyncio.wait_for(shutdown.wait(), timeout=poll_interval)
        except asyncio.TimeoutError:
            pass


async def nws_atom_listener(
    config: dict[str, Any],
    on_alert: Callable[[CAPAlert, dict[str, Any]], Awaitable[None]],
    shutdown: asyncio.Event,
) -> None:
    nws_cfg = config.get('cap', {}).get('nws_cap', {})
    sources = nws_cfg.get('sources') or []
    if not isinstance(sources, list) or not sources:
        log.info('NWS CAP alerting enabled but no sources configured')
        return

    tasks = [
        asyncio.create_task(_poll_source(config, source, on_alert, shutdown), name=f"nws_atom:{source.get('id', 'source')}")
        for source in sources
        if isinstance(source, dict)
    ]
    if not tasks:
        return
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
from __future__ import annotations

import asyncio
import logging
import urllib.request
from typing import Any, Awaitable, Callable
from xml.etree import ElementTree as ET

from module.cap_specific.naads_tcp import CAPAlert, parse_cap

log = logging.getLogger(__name__)

_NAAD_RSS_URLS = [
    'https://rss.naad-adna.pelmorex.com/',
    'http://rss1.naad-adna.pelmorex.com/',
    'http://rss2.naad-adna.pelmorex.com/',
]

_ATOM_ENTRY = '{http://www.w3.org/2005/Atom}entry'
_ATOM_LINK  = '{http://www.w3.org/2005/Atom}link'
_ATOM_ID    = '{http://www.w3.org/2005/Atom}id'


def _rss_cap_links(raw: bytes) -> list[str]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return []
    links: list[str] = []
    seen: set[str] = set()
    for entry in root.iter(_ATOM_ENTRY):
        best: str | None = None
        for link_el in entry.findall(_ATOM_LINK):
            href = (link_el.get('href') or '').strip()
            if not href:
                continue
            if 'cap' in (link_el.get('type') or '').lower():
                best = href
                break
            if best is None:
                best = href
        if best is None:
            id_text = (entry.findtext(_ATOM_ID) or '').strip()
            if id_text and id_text.startswith('http'):
                best = id_text
        if best and best not in seen:
            seen.add(best)
            links.append(best)
    return links


def _rss_urls_from_sources(sources: list[dict[str, Any]]) -> list[str]:
    urls: list[str] = []
    for source in sources:
        for key in ('rss', 'rss1', 'rss2'):
            raw = str(source.get(key) or '').strip()
            if not raw:
                continue
            url = (raw if raw.startswith('http') else f'https://{raw}').rstrip('/') + '/'
            if url not in urls:
                urls.append(url)
    return urls or list(_NAAD_RSS_URLS)


async def naad_archive_fetch(
    sources: list[dict[str, Any]],
    on_alert: Callable[[CAPAlert], Awaitable[None]],
    timeout_s: float = 15.0,
) -> int:
    rss_urls = _rss_urls_from_sources(sources)

    def _http_get(url: str) -> bytes:
        req = urllib.request.Request(url, headers={
            'Accept': 'application/atom+xml, application/cap+xml, application/xml, */*',
        })
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read()

    processed = 0
    for rss_url in rss_urls:
        try:
            feed_raw = await asyncio.to_thread(_http_get, rss_url)
        except Exception as exc:
            log.warning('NAADS startup RSS unavailable (%s): %s', rss_url, exc)
            continue

        cap_links = _rss_cap_links(feed_raw)
        log.info('NAADS startup RSS (%s): %d unique CAP link(s)', rss_url, len(cap_links))

        for cap_url in cap_links:
            try:
                cap_raw = await asyncio.to_thread(_http_get, cap_url)
            except Exception as exc:
                log.warning('NAADS archive CAP fetch failed (%s): %s', cap_url, exc)
                continue
            alert = parse_cap(cap_raw)
            if alert:
                await on_alert(alert)
                processed += 1

        log.info('NAADS startup archive: %d alert(s) dispatched', processed)
        break

    return processed

from __future__ import annotations

import logging
import os
import pathlib
import xml.etree.ElementTree as ET
from typing import Any

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)

_DEFAULT_CONFIG_FILE = 'config.yaml'
_DEFAULT_FEEDS_FILE = 'managed/feeds.xml'


def _resolve_config_file(config_path: str | None = None) -> pathlib.Path:
    selected_path = config_path or os.environ.get('CONFIG_PATH')
    if selected_path:
        return pathlib.Path(selected_path)
    return pathlib.Path(__file__).parent.parent / _DEFAULT_CONFIG_FILE


def _resolve_sidecar_path(base_dir: pathlib.Path, raw_path: Any, default_path: str) -> pathlib.Path:
    raw_text = str(raw_path).strip() if raw_path is not None else ''
    candidate = pathlib.Path(raw_text) if raw_text else pathlib.Path(default_path)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate


def _load_yaml_file(path: pathlib.Path) -> Any:
    with open(path, encoding='utf-8') as file_handle:
        return yaml.safe_load(file_handle) or {}


def _coerce_bool(raw: str | None) -> bool | None:
    if raw is None:
        return None
    return raw.strip().lower() in ('true', 'yes', '1')


def _coerce_int(raw: str | None) -> int | None:
    if not raw or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _coerce_float(raw: str | None) -> float | None:
    if not raw or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        return None


_INT_OUTPUT_FIELDS: frozenset[str] = frozenset({'port', 'bitrate_kbps', 'vrate_kbps'})
_BOOL_OUTPUT_FIELDS: frozenset[str] = frozenset({'ssl'})


def _parse_output_sink(el: ET.Element) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if (en := el.get('enabled')) is not None:
        result['enabled'] = _coerce_bool(en)
    for child in el:
        raw = (child.text or '').strip()
        if child.tag in _BOOL_OUTPUT_FIELDS:
            result[child.tag] = _coerce_bool(raw) if raw else None
        elif child.tag in _INT_OUTPUT_FIELDS:
            result[child.tag] = _coerce_int(raw) if raw else None
        else:
            result[child.tag] = raw if raw else None
    return result


def _parse_transmitter_metadata(el: ET.Element) -> list[dict[str, Any]]:
    transmitters: list[dict[str, Any]] = []
    for t_el in el.findall('transmitter'):
        t: dict[str, Any] = {}
        for child in t_el:
            raw = (child.text or '').strip()
            t[child.tag] = _coerce_float(raw) if child.tag == 'frequency_mhz' else (raw if raw else None)
        transmitters.append(t)
    return transmitters


def _parse_coverage_entry(el: ET.Element) -> dict[str, Any]:
    entry: dict[str, Any] = {'coverage_type': el.tag}
    for key, value in el.attrib.items():
        if value:
            entry[key] = value
    if el.tag == 'region':
        subregions: list[dict[str, Any]] = []
        for subregion_el in el.findall('subregion'):
            subregion = _parse_coverage_entry(subregion_el)
            if subregion:
                subregions.append(subregion)
        if subregions:
            entry['subregions'] = subregions
    return entry


def _parse_coverage_block(el: ET.Element) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coverage: list[dict[str, Any]] = []
    forecast_locations: list[dict[str, Any]] = []

    for coverage_el in el:
        if not isinstance(coverage_el.tag, str):
            continue
        entry = _parse_coverage_entry(coverage_el)
        if not entry:
            continue
        coverage.append(entry)

        if coverage_el.tag != 'region':
            continue

        forecast_id = str(entry.get('derive_forecast') or '').strip() or str(entry.get('id') or '').strip()
        if forecast_id:
            forecast_location: dict[str, Any] = {'id': forecast_id}
            if (source := entry.get('source')):
                forecast_location['source'] = source
            if (region_id := str(entry.get('id') or '').strip()):
                forecast_location['forecast_region'] = region_id
            if (name_override := entry.get('name_override')):
                forecast_location['name_override'] = name_override
            forecast_locations.append(forecast_location)

    return coverage, forecast_locations


def _parse_forecast_locations_block(el: ET.Element) -> list[dict[str, Any]]:
    forecast_locations: list[dict[str, Any]] = []
    for loc_el in el.findall('location'):
        loc = {key: value for key, value in loc_el.attrib.items() if value}
        if loc:
            forecast_locations.append(loc)
    return forecast_locations


def _parse_feed_el(el: ET.Element) -> dict[str, Any]:
    feed: dict[str, Any] = {}

    for attr in ('id', 'callsign', 'timezone'):
        if (v := el.get(attr)) is not None:
            feed[attr] = v
    if (en := el.get('enabled')) is not None:
        feed['enabled'] = _coerce_bool(en)

    playout_el = el.find('playout')
    routine_enabled = True
    if playout_el is not None:
        playout: dict[str, Any] = {}
        for attr in ('priority', 'routine', 'same', 'chimes'):
            if (v := playout_el.get(attr)) is not None:
                playout[attr] = _coerce_bool(v)
        routine_enabled = bool(playout.get('routine', True))
        pkgs_el = playout_el.find('packages')
        if pkgs_el is not None:
            playout['packages'] = [p.text.strip() for p in pkgs_el.findall('package') if p.text]
        feed['playout'] = playout

    alerts_el = el.find('alerts')
    if alerts_el is not None:
        alerts: dict[str, Any] = {}
        cap_el = alerts_el.find('cap_cp')
        if cap_el is not None:
            cap: dict[str, Any] = {}
            if (en := cap_el.get('enabled')) is not None:
                cap['enabled'] = _coerce_bool(en)
            filter_el = cap_el.find('filter')
            if filter_el is not None:
                filt: dict[str, Any] = {}
                if (uf := filter_el.get('use_feed_locations')) is not None:
                    filt['use_feed_locations'] = _coerce_bool(uf)
                bl_el = filter_el.find('blocklist')
                if bl_el is not None:
                    filt['blocklist'] = {
                        'severity':    [e.text.strip() for e in bl_el.findall('severity')     if e.text],
                        'certainty':   [e.text.strip() for e in bl_el.findall('certainty')    if e.text],
                        'naads_events':[e.text.strip() for e in bl_el.findall('naads_event')  if e.text],
                        'urgency':     [e.text.strip() for e in bl_el.findall('urgency_item') if e.text],
                        'status':      [e.text.strip() for e in bl_el.findall('status_item')  if e.text],
                        'scope':       [e.text.strip() for e in bl_el.findall('scope_item')   if e.text],
                        'other':       [dict(o.attrib) for o in bl_el.findall('other')],
                    }
                cap['filter'] = filt
            alerts['cap_cp'] = cap
        nws_el = alerts_el.find('nws_cap')
        if nws_el is not None:
            nws: dict[str, Any] = {}
            if (en := nws_el.get('enabled')) is not None:
                nws['enabled'] = _coerce_bool(en)
            alerts['nws_cap'] = nws
        feed['alerts'] = alerts

    langs_el = el.find('languages')
    if langs_el is not None:
        langs: dict[str, Any] = {}
        for lang_el in langs_el.findall('lang'):
            if code := lang_el.get('code'):
                lang_cfg: dict[str, Any] = {}
                if (iv := lang_el.get('interval')) is not None:
                    lang_cfg['interval'] = _coerce_int(iv)
                langs[code] = lang_cfg
        feed['languages'] = langs

    desc_el = el.find('description')
    if desc_el is not None:
        desc: dict[str, Any] = {}
        for lang_el in desc_el.findall('lang'):
            if code := lang_el.get('code'):
                lang_desc: dict[str, Any] = {}
                if (text := lang_el.get('text')) is not None:
                    lang_desc['text'] = text
                if (suffix := lang_el.get('suffix')) is not None:
                    lang_desc['suffix'] = suffix
                desc[code] = lang_desc
        feed['description'] = desc

    locs_el = el.find('locations')
    if locs_el is not None:
        loc_blocks: list[dict[str, Any]] = []
        coverage_el = locs_el.find('coverage')
        if coverage_el is not None:
            coverage, forecast_locations = _parse_coverage_block(coverage_el)
            feed['coverage'] = coverage
            loc_blocks.append({'coverage': coverage})
            if routine_enabled and forecast_locations:
                loc_blocks.append({'forecastLocations': forecast_locations})
        else:
            forecast_el = locs_el.find('forecastLocations')
            if forecast_el is not None:
                forecast_locations = _parse_forecast_locations_block(forecast_el)
                if routine_enabled and forecast_locations:
                    loc_blocks.append({'forecastLocations': forecast_locations})

        for loc_type in ('observationLocations', 'airQualityLocations', 'climateLocations'):
            type_el = locs_el.find(loc_type)
            if type_el is not None:
                locs_list: list[dict[str, Any]] = [
                    {k: v for k, v in loc_el.attrib.items() if v}
                    for loc_el in type_el.findall('location')
                ]
                loc_blocks.append({loc_type: locs_list})
        feed['locations'] = loc_blocks

    tm_el = el.find('transmitter_metadata')
    transmitters: list[dict[str, Any]] = []
    if tm_el is not None:
        transmitters = _parse_transmitter_metadata(tm_el)
        feed['transmitter_metadata'] = transmitters
        if 'callsign' not in feed and transmitters:
            primary = next((t for t in transmitters if t.get('relationship') == 'primary'), transmitters[0])
            if cs := primary.get('callsign'):
                feed['callsign'] = cs

    output_el = el.find('output')
    if output_el is not None:
        output: dict[str, Any] = {}
        for ot in ('stream', 'udp', 'rtp', 'rtmp', 'srt', 'rtsp', 'webrtc',
                   'audio_device', 'framebuffer', 'dri', 'v4l2'):
            ot_el = output_el.find(ot)
            if ot_el is not None:
                output[ot] = _parse_output_sink(ot_el)
        pifm_el = output_el.find('PiFmAdv')
        if pifm_el is not None:
            pifm: dict[str, Any] = {}
            if (en := pifm_el.get('enabled')) is not None:
                pifm['enabled'] = _coerce_bool(en)
            transmitter_site = pifm_el.get('transmitter_site')
            if transmitter_site:
                pifm['transmitter_site'] = transmitter_site
            for child in pifm_el:
                if child.tag == 'ssh':
                    continue
                raw = (child.text or '').strip()
                if child.tag in ('tx_power', 'bandwidth_hz', 'deviation_hz'):
                    pifm[child.tag] = _coerce_int(raw) if raw else None
                else:
                    pifm[child.tag] = raw if raw else None
            if transmitter_site and transmitters:
                matched = next((t for t in transmitters if t.get('site_name') == transmitter_site), None)
                if matched and (freq := matched.get('frequency_mhz')) is not None:
                    pifm['frequency_mhz'] = freq
            pifm['alternative_frequencies'] = []
            ssh_el = pifm_el.find('ssh')
            if ssh_el is not None:
                ssh: dict[str, Any] = {}
                if (en := ssh_el.get('enabled')) is not None:
                    ssh['enabled'] = _coerce_bool(en)
                for child in ssh_el:
                    raw = (child.text or '').strip()
                    ssh[child.tag] = _coerce_int(raw) if child.tag == 'port' else (raw if raw else None)
                pifm['ssh'] = ssh
            output['PiFmAdv'] = pifm
        feed['output'] = output

    return feed


def _load_twc_env(config: dict[str, Any]) -> None:
    load_dotenv()
    raw_sources = config.get('sources')
    sources = dict(raw_sources) if isinstance(raw_sources, dict) else {}
    sources['twc'] = {
        'api_key': os.environ.get('TWC_API_KEY', ''),
        'units': os.environ.get('TWC_UNITS', 'm'),
        'language': os.environ.get('TWC_LANGUAGE', 'en-CA'),
    }
    config['sources'] = sources


def _load_feeds_sidecar(config: dict[str, Any], config_file: pathlib.Path) -> None:
    feeds_path = _resolve_sidecar_path(config_file.parent, config.get('feeds_file'), _DEFAULT_FEEDS_FILE)
    if not feeds_path.exists():
        config['feeds'] = list(config.get('feeds') or [])
        return

    try:
        root = ET.parse(feeds_path).getroot()
        feeds = [_parse_feed_el(el) for el in root.findall('feed') if el.get('id')]
    except Exception as exc:
        log.error('Failed to load feeds from %s: %s', feeds_path, exc)
        config['feeds'] = list(config.get('feeds') or [])
        return

    config['feeds'] = feeds


def load_config(config_path: str | None = None) -> dict[str, Any]:
    config_file = _resolve_config_file(config_path)
    raw_config = _load_yaml_file(config_file)
    config = dict(raw_config) if isinstance(raw_config, dict) else {}
    _load_twc_env(config)
    _load_feeds_sidecar(config, config_file)
    return config
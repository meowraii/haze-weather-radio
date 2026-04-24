import asyncio
import datetime
import coloredlogs
import logging.handlers
import os
import pathlib
import threading
import argparse
import json
from typing import Any

import yaml

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

from managed.events import (
    append_runtime_event,
    initial_synthesis_done,
    read_data_pool,
    shutdown_event,
    update_runtime_status,
)
from module.alert import alert_worker, purge_expired_alerts
from module.data import data_thread_worker, fetch_once
from module.playlist import playlist_thread_worker
from module.playout import playout_thread_worker
from module.same import generate_same, to_wav
from module.scheduler import clean_stale_data, scheduler_thread_worker
from module.static_phrases import init_static_phrases
from module.tts import generate_package, get_available_pyttsx3_voices, load_config, synthesize, synthesize_pcm
from module.webserve import start_web_server
from managed.packages import (
    alerts_package,
    climate_summary_package,
    current_conditions_package,
    date_time_package,
    eccc_discussion_package,
    forecast_package,
    geophysical_alert_package,
    station_id,
    user_bulletin_package,
)

_BULLETINS_PATH = pathlib.Path(__file__).parent / 'managed' / 'userbulletins.json'
_REGISTRY_DIR = pathlib.Path(__file__).parent / 'data' / 'alerts'
_DEFAULT_PACKAGE_TYPES = [
    'date_time',
    'station_id',
    'current_conditions',
    'forecast',
    'climate_summary',
    'eccc_discussion',
    'geophysical_alert',
    'user_bulletin',
    'alerts',
]

def _check_static_audio(config: dict[str, Any]) -> None:
    static_root = pathlib.Path(config.get('static', 'audio'))
    static_root.mkdir(parents=True, exist_ok=True)
    same_eom_path = static_root / 'same_eom.wav'
    if not same_eom_path.exists():
        logging.info('Generating SAME EOM cache: %s', same_eom_path)
        same_eom_path.write_bytes(to_wav(generate_same()))


def _naads_thread_worker(config: dict[str, Any], feeds: list[dict[str, Any]]) -> None:
    async def _run() -> None:
        shutdown = asyncio.Event()

        async def _watch() -> None:
            while not shutdown_event.is_set():
                await asyncio.sleep(0.5)
            shutdown.set()

        watchdog = asyncio.create_task(_watch())
        try:
            await alert_worker(config, feeds, shutdown)
        finally:
            watchdog.cancel()

    asyncio.run(_run())


def _setup_logging(config: dict[str, Any], override_level: str | None = None) -> None:
    log_cfg = config.get('logging', {})
    configured_level = override_level or os.environ.get('LOG_LEVEL') or log_cfg.get('level', 'INFO')
    level = getattr(logging, str(configured_level).upper(), logging.INFO)
    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'

    logging.basicConfig(level=level, format=fmt, handlers=[])

    if log_cfg.get('console', {}).get('enabled', True):
        coloredlogs.install(level=level, fmt=fmt)

    file_cfg = log_cfg.get('file', {})
    if file_cfg.get('enabled', False):
        log_path = pathlib.Path(file_cfg['path'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotate = file_cfg.get('rotate', {})
        handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=rotate.get('max_bytes', 10_485_760),
            backupCount=rotate.get('backup_count', 5),
        )
        handler.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(handler)


def _thread(target: Any, *args: Any, name: str) -> threading.Thread:
    return threading.Thread(target=target, args=args, name=name, daemon=True)


from module.feed_util import (
    climate_locations as _climate_locations,
    climate_name as _climate_name,
    current_conditions_name as _current_conditions_name,
    enabled_feeds as _enabled_feeds,
    feed_languages as _feed_languages,
    forecast_locations as _forecast_locations,
    forecast_name as _forecast_name,
    location_label as _location_label,
    observation_locations as _observation_locations,
)


def _load_json_list(path: pathlib.Path) -> list[Any]:
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _build_feed_package_lookup(config: dict[str, Any], feed: dict[str, Any], lang: str) -> dict[str, str]:
    feed_id = feed['id']
    tz = feed.get('timezone', 'UTC')
    observation_locs = _observation_locations(feed)
    forecast_locs = _forecast_locations(feed)
    climate_locs = _climate_locations(feed)
    primary_loc = observation_locs[0] if observation_locs else (forecast_locs[0] if forecast_locs else None)
    loc_name = _location_label(primary_loc) if primary_loc else None

    conditions_parts: list[str] = []
    for i, loc in enumerate(observation_locs):
        loc_id = loc.get('id')
        data = read_data_pool(f"{feed_id}:{loc_id}") if loc_id else None
        text = current_conditions_package(data, _current_conditions_name(loc), lang, secondary=(i > 0))
        if text:
            conditions_parts.append(text)

    forecast_parts: list[str] = []
    for loc in forecast_locs:
        loc_id = loc.get('id')
        forecast_data = read_data_pool(f"{feed_id}:forecast:{loc_id}") if loc_id else None
        part = forecast_package(forecast_data, _forecast_name(loc), lang)
        if part:
            forecast_parts.append(part)

    climate_parts: list[str] = []
    for loc in climate_locs:
        loc_id = loc.get('id')
        climate_data = read_data_pool(f"{feed_id}:climate:{loc_id}") if loc_id else None
        part = climate_summary_package(climate_data, _climate_name(loc), lang)
        if part:
            climate_parts.append(part)

    focn45_bulletin = read_data_pool('focn45')
    discussion_text = None
    if focn45_bulletin is not None:
        discussion_text = getattr(focn45_bulletin, 'text', None) or (focn45_bulletin if isinstance(focn45_bulletin, str) else None)

    registry = _load_json_list(_REGISTRY_DIR / f'{feed_id}.json')
    bulletins = _load_json_list(_BULLETINS_PATH)
    wwv_text = read_data_pool('wwv')

    return {
        'date_time': date_time_package(tz, lang),
        'station_id': station_id(config, feed_id, lang),
        'current_conditions': '  '.join(conditions_parts),
        'forecast': '  '.join(forecast_parts),
        'climate_summary': '  '.join(climate_parts),
        'eccc_discussion': eccc_discussion_package(discussion_text, loc_name, lang),
        'geophysical_alert': geophysical_alert_package(wwv_text),
        'user_bulletin': user_bulletin_package(bulletins, lang, tz),
        'alerts': alerts_package(registry, lang, tz, feed),
    }


def _selected_package_ids(config: dict[str, Any], selector: str | None) -> list[str]:
    if selector in (None, '*'):
        return list(_DEFAULT_PACKAGE_TYPES)
    normalized = selector.strip()
    if not normalized:
        return list(_DEFAULT_PACKAGE_TYPES)
    if normalized in _DEFAULT_PACKAGE_TYPES:
        return [normalized]
    suffix_match = normalized.split('_', 1)[-1] if '_' in normalized else normalized
    if suffix_match in _DEFAULT_PACKAGE_TYPES:
        return [suffix_match]
    return []


def _run_gen_pkg_text(config: dict[str, Any], selector: str | None) -> int:
    asyncio.run(fetch_once(config))
    package_ids = _selected_package_ids(config, selector)
    if not package_ids:
        print(f'No matching package type for: {selector}')
        return 1

    feeds = _enabled_feeds(config)
    for feed in feeds:
        for lang in _feed_languages(feed):
            pkg_lookup = _build_feed_package_lookup(config, feed, lang)
            for pkg_type in package_ids:
                text = pkg_lookup.get(pkg_type, '')
                print(f"--- {feed['id']}/{lang}/{pkg_type} ---")
                print(text)
                print()
    return 0


def _run_gen_tts(config: dict[str, Any], selector: str | None) -> int:
    feeds = _enabled_feeds(config)
    if not feeds:
        print('No enabled feeds found.')
        return 1

    if selector not in (None, '*') and selector.strip() and _selected_package_ids(config, selector) == []:
        feed = feeds[0]
        lang = _feed_languages(feed)[0]
        path = synthesize(config, selector, feed['id'], 'cli_manual', lang)
        if path is None:
            print('Failed to synthesize CLI text.')
            return 1
        print(path)
        return 0

    asyncio.run(fetch_once(config))
    package_ids = _selected_package_ids(config, selector)
    for feed in feeds:
        for lang in _feed_languages(feed):
            pkg_lookup = _build_feed_package_lookup(config, feed, lang)
            for pkg_type in package_ids:
                text = pkg_lookup.get(pkg_type, '')
                if not text:
                    continue
                path = synthesize(config, text, feed['id'], pkg_type, lang)
                if path is not None:
                    print(path)
    return 0

def main(config: dict[str, Any], log_level: str | None = None) -> None:
    _setup_logging(config, log_level)
    log = logging.getLogger('haze')
    update_runtime_status({
        'started_at': datetime.datetime.now(datetime.UTC).isoformat(),
        'shutdown_requested': False,
    })
    append_runtime_event('startup', 'Haze Weather Radio starting')
    log.info('Haze Weather Radio starting')
    _check_static_audio(config)

    feeds = _enabled_feeds(config)

    for feed in feeds:
        purge_count = clean_stale_data(feed)
        log.info('Cleaned %d stale audio file(s) for feed: %s', purge_count, feed['id'])

    expired_count = purge_expired_alerts()
    if expired_count:
        log.info('Purged %d long-expired alert(s) from registry on startup', expired_count)

    log.info('Initializing static phrase cache...')
    init_static_phrases(config, feeds)

    infra: list[threading.Thread] = [
        _thread(_naads_thread_worker, config, feeds, name='naads'),
    ]
    web_thread = start_web_server(config, feeds)
    if web_thread is not None:
        infra.append(web_thread)
    for t in infra:
        t.start()

    data: list[threading.Thread] = [
        _thread(data_thread_worker, config, name='data'),
        _thread(playlist_thread_worker, config, name='playlist'),
    ]
    for t in data:
        t.start()

    playout: list[threading.Thread] = [
        _thread(playout_thread_worker, config, feed, name=f'playout:{feed["id"]}')
        for feed in feeds
    ]
    for t in playout:
        t.start()

    log.info('Waiting for initial data fetch and audio synthesis...')
    initial_synthesis_done.wait()
    log.info('All systems nominal. Playout active for %d feed(s).', len(feeds))

    scheduler = _thread(scheduler_thread_worker, config, feeds, name='scheduler')
    scheduler.start()

    all_threads = infra + data + playout + [scheduler]
    try:
        for t in all_threads:
            t.join()
    except KeyboardInterrupt:
        log.info('Shutdown requested')
        update_runtime_status({'shutdown_requested': True})
        append_runtime_event('shutdown', 'Shutdown requested by operator')
        shutdown_event.set()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Haze Weather Radio')
    parser.add_argument('--config', '-c', type=str, default='config.yaml', help='Path to configuration file.')
    parser.add_argument('--log-level', '-l', type=str, help='Override log level (e.g. DEBUG, INFO, WARNING).')
    parser.add_argument('--gen-pkg-text', '-t', nargs='?', const='*', default=None, help='Generate text output for one package type, feed-prefixed package id, or all packages, then exit.')
    parser.add_argument('--gen-tts', '-s', nargs='?', const='*', default=None, help='Generate TTS audio for one package type, feed-prefixed package id, all packages, or raw input text, then exit.')
    parser.add_argument('--pytts-voices', action='store_true', help='List available pyttsx3 voices and exit.')
    args = parser.parse_args()

    if args.log_level:
        os.environ['LOG_LEVEL'] = args.log_level.upper()
    if args.config:
        os.environ['CONFIG_PATH'] = args.config

    config = load_config(args.config)

    if args.gen_pkg_text is not None:
        raise SystemExit(_run_gen_pkg_text(config, args.gen_pkg_text))
    if args.gen_tts is not None:
        raise SystemExit(_run_gen_tts(config, args.gen_tts))
    if args.pytts_voices:
        print(json.dumps(get_available_pyttsx3_voices(), indent=2))
        raise SystemExit(0)

    main(config, args.log_level)

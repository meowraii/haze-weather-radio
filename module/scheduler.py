from __future__ import annotations

import datetime as _dt
import fnmatch
import logging
import pathlib
import time
from typing import Any, cast

import numpy as np
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from module.alert_templates import load_alert_templates
from module.events import append_runtime_event, enqueue_scheduled_package, push_alert, shutdown_event, store_runtime_alert_entry
from module.alert import feed_same_codes, purge_expired_alerts
from module.buffer import CHANNELS, SAMPLE_RATE as BUS_SR
from module.same import SAMEHeader, generate_same, resample, resolve_flavor, to_pcm16
from module.tts import synthesize_pcm

log = logging.getLogger(__name__)

_CACHE_EXTS = frozenset({'.json', '.txt'})
_ALL_FILES: frozenset[str] = frozenset()
_DEFAULT_STATION_ID_SCHEDULE_CFG: dict[str, Any] = {
    'enabled': True,
    'minutes': [0, 15, 30, 45],
}

_DEFAULT_DATE_TIME_SCHEDULE_CFG: dict[str, Any] = {
    'enabled': True,
    'minutes': [5, 15, 25, 35, 45, 55],
}


def _feed_playout_cfg(feed: dict[str, Any]) -> dict[str, Any]:
    raw = feed.get('playout', {})
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        merged: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    return {}


def _station_id_schedule_cfg(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(_DEFAULT_STATION_ID_SCHEDULE_CFG)
    raw = config.get('playout', {}).get('station_id_schedule', {})
    if isinstance(raw, dict):
        merged.update(raw)

    raw_minutes = merged.get('minutes', _DEFAULT_STATION_ID_SCHEDULE_CFG['minutes'])
    if isinstance(raw_minutes, str):
        tokens = [part.strip() for part in raw_minutes.split(',') if part.strip()]
    elif isinstance(raw_minutes, (list, tuple, set)):
        tokens = list(raw_minutes)
    else:
        tokens = [raw_minutes]

    minutes: list[int] = []
    for token in tokens:
        try:
            minute = int(token)
        except (TypeError, ValueError):
            continue
        if 0 <= minute <= 59 and minute not in minutes:
            minutes.append(minute)

    return {
        'enabled': bool(merged.get('enabled', True)),
        'minutes': minutes or list(_DEFAULT_STATION_ID_SCHEDULE_CFG['minutes']),
    }


def _date_time_schedule_cfg(config: dict[str, Any]) -> dict[str, Any]:
    merged = dict(_DEFAULT_DATE_TIME_SCHEDULE_CFG)
    raw = config.get('playout', {}).get('date_time_schedule', {})
    if isinstance(raw, dict):
        merged.update(raw)

    raw_minutes = merged.get('minutes', _DEFAULT_DATE_TIME_SCHEDULE_CFG['minutes'])
    if isinstance(raw_minutes, str):
        tokens = [part.strip() for part in raw_minutes.split(',') if part.strip()]
    elif isinstance(raw_minutes, (list, tuple, set)):
        tokens = list(raw_minutes)
    else:
        tokens = [raw_minutes]

    minutes: list[int] = []
    for token in tokens:
        try:
            minute = int(token)
        except (TypeError, ValueError):
            continue
        if 0 <= minute <= 59 and minute not in minutes:
            minutes.append(minute)

    return {
        'enabled': bool(merged.get('enabled', True)),
        'minutes': minutes or list(_DEFAULT_DATE_TIME_SCHEDULE_CFG['minutes']),
    }

def clean_stale_data(feed: dict[str, Any]) -> int:
    feed_id = feed.get('id', '')
    audio_root = pathlib.Path('audio')
    data_root = pathlib.Path('data')

    targets: list[tuple[pathlib.Path, frozenset[str]]] = [
        (audio_root / '_uploads',          _ALL_FILES),
        (data_root / 'eccc',               _CACHE_EXTS),
        (data_root / 'nws',                _CACHE_EXTS),
        (data_root / 'weatherdotcom',       _CACHE_EXTS),
    ]

    purge_count = 0
    for directory, exts in targets:
        directory.mkdir(parents=True, exist_ok=True)
        for item in directory.iterdir():
            if not item.is_file():
                continue
            if exts and item.suffix not in exts:
                continue
            try:
                item.unlink()
                purge_count += 1
            except OSError as exc:
                log.warning('Could not remove stale file %s: %s', item, exc)

    return purge_count


def _load_test_templates() -> dict[str, Any]:
    return cast(dict[str, Any], load_alert_templates())


def _match_lang_key(msg: dict[str, Any], lang: str) -> str | None:
    for pattern, text in msg.items():
        if fnmatch.fnmatch(lang, pattern):
            return str(text)
    return None


def _pcm_to_voice_array(pcm: bytes, same_sr: int) -> np.ndarray:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
    if CHANNELS == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)
    return resample(samples, BUS_SR, same_sr)


def fire_test(config: dict[str, Any], feeds: list[dict[str, Any]], event_code: str = 'RWT') -> bool:
    templates = _load_test_templates()
    template = templates.get(event_code)
    if not template:
        log.error('No alertTemplates.xml template found for event code: %s', event_code)
        return False

    msg_map: dict[str, Any] = template.get('msg', {})
    same_expire: str = str(template.get('sameExpire', '0015'))
    same_event: str = str(template.get('sameEvent', event_code))
    same_cfg = config.get('same', {})
    same_block = template.get('same', {}) if isinstance(template.get('same'), dict) else {}
    callsign: str = str(same_block.get('sender_id') or same_cfg.get('sender', 'HAZE0000'))
    flavor = same_cfg.get('flavor') or None
    same_sr = resolve_flavor(flavor).sample_rate
    content_block = same_block.get('content', {}) if isinstance(same_block.get('content'), dict) else {}
    tone_value = str(content_block.get('attention_tone') or same_cfg.get('default_attention_tone', 'WXR')).upper()
    tone_type: str | None = None if tone_value == 'NONE' else tone_value
    template_locations = [
        str(location.get('id', '')).strip()
        for location in cast(list[Any], same_block.get('locations', []))
        if isinstance(location, dict) and str(location.get('id', '')).strip()
    ]

    fired_any = False
    for feed in feeds:
        if not feed.get('enabled', True):
            continue

        feed_id: str = feed.get('id', 'default')
        locations = template_locations or feed_same_codes(feed) or ['000000']
        feed_langs: list[str] = list(feed.get('languages', {}).keys()) or ['en-CA']

        voice_pcm_parts: list[bytes] = []
        overlay_text = ''
        for lang in feed_langs:
            text = _match_lang_key(msg_map, lang)
            if not text:
                continue
            if not overlay_text:
                overlay_text = str(text).strip()
            pcm = synthesize_pcm(config, text, lang=lang)
            if pcm:
                voice_pcm_parts.append(pcm)

        voice_array: np.ndarray | None = None
        if voice_pcm_parts:
            combined_pcm = b''.join(voice_pcm_parts)
            voice_array = _pcm_to_voice_array(combined_pcm, same_sr)

        header = SAMEHeader(
            originator='EAS',
            event=same_event,
            locations=tuple(locations[:31]),
            duration=same_expire,
            callsign=callsign,
        )

        full_signal = generate_same(
            header=header,
            tone_type=cast(Any, tone_type),
            audio_msg_array=voice_array,
            attn_duration_s=8.0,
            flavor=flavor,
        )

        alert_pcm = to_pcm16(resample(full_signal, same_sr, BUS_SR))
        issued_at = _dt.datetime.now(_dt.timezone.utc)
        expires_at = issued_at + _dt.timedelta(
            hours=int(same_expire[:2] or 0),
            minutes=int(same_expire[2:] or 0),
        )
        identifier = f'test_{event_code}_{int(time.time())}'
        store_runtime_alert_entry(feed_id, identifier, {
            'identifier': identifier,
            'feed_id': feed_id,
            'received_at': issued_at.isoformat(),
            'display_id': f'MSG{issued_at.strftime("%H%M%S")}',
            'metadata': {
                'event': same_event,
                'effective': issued_at.isoformat(),
                'onset': issued_at.isoformat(),
                'expires': expires_at.isoformat(),
            },
            'source': {
                'kind': 'test',
                'originator': 'EAS',
                'eventCode': same_event,
            },
            'text': {
                'description': overlay_text,
                'instruction': '',
            },
            'areas': [
                {'sameCode': location}
                for location in locations[:31]
            ],
        })
        push_alert(feed_id, 0, alert_pcm, identifier)
        log.info('[%s] Queued %s test: %s', feed_id, event_code, header.encoded)
        fired_any = True

    if fired_any:
        append_runtime_event('test', f'{event_code} test fired')

    return fired_any


def _week_to_day_range(week: int) -> str:
    start = (week - 1) * 7 + 1
    return f'{start}-{start + 6}'


def _weekly_trigger(cfg: dict[str, Any]) -> CronTrigger:
    weeks_raw = cfg.get('weeks', [1])
    weeks = weeks_raw if isinstance(weeks_raw, list) else [weeks_raw]
    day_ranges = ','.join(_week_to_day_range(int(w)) for w in weeks)
    return CronTrigger(
        day=day_ranges,
        day_of_week=str(cfg.get('weekday', 'wed')).lower(),
        hour=int(cfg.get('hour', 12)),
        minute=int(cfg.get('minute', 0)),
    )


def _monthly_trigger(cfg: dict[str, Any]) -> CronTrigger:
    return CronTrigger(
        day=_week_to_day_range(int(cfg.get('week', 1))),
        day_of_week=str(cfg.get('weekday', 'wed')).lower(),
        hour=int(cfg.get('hour', 12)),
        minute=int(cfg.get('minute', 0)),
    )


def _run_cleanup(feed: dict[str, Any]) -> None:
    feed_id = feed.get('id', '')
    count = clean_stale_data(feed)
    log.info('Nightly cleanup removed %d file(s) for feed: %s', count, feed_id)
    expired = purge_expired_alerts()
    if expired:
        log.info('Nightly cleanup purged %d long-expired alert(s) from registry', expired)
    append_runtime_event('cleanup', f'Nightly cleanup: {count} file(s) removed, {expired} expired alert(s) purged for {feed_id}')


def _queue_station_ids(feeds: list[dict[str, Any]]) -> None:
    queued_feeds: list[str] = []
    for feed in feeds:
        if not feed.get('enabled', True):
            continue
        if not _feed_playout_cfg(feed).get('routine', True):
            continue
        feed_id = str(feed.get('id', '')).strip()
        if not feed_id:
            continue
        enqueue_scheduled_package(feed_id, 'station_id')
        queued_feeds.append(feed_id)

    if queued_feeds:
        append_runtime_event('station-id', 'Scheduled station identification queued', extra={'feeds': queued_feeds})


def _queue_date_times(feeds: list[dict[str, Any]]) -> None:
    queued_feeds: list[str] = []
    for feed in feeds:
        if not feed.get('enabled', True):
            continue
        if not _feed_playout_cfg(feed).get('routine', True):
            continue
        feed_id = str(feed.get('id', '')).strip()
        if not feed_id:
            continue
        enqueue_scheduled_package(feed_id, 'date_time')
        queued_feeds.append(feed_id)

    if queued_feeds:
        append_runtime_event('date-time', 'Scheduled date and time package queued', extra={'feeds': queued_feeds})


def scheduler_thread_worker(config: dict[str, Any], feeds: list[dict[str, Any]]) -> None:
    same_cfg = config.get('same', {})
    rwt_cfg = same_cfg.get('weeklytest', {})
    rmt_cfg = same_cfg.get('monthlytest', {})

    scheduler = BackgroundScheduler()

    if bool(rwt_cfg.get('enabled', False)):
        event_code = str(rwt_cfg.get('event_code', 'RWT'))
        scheduler.add_job(
            fire_test,
            trigger=_weekly_trigger(rwt_cfg),
            args=[config, feeds, event_code],
            id='rwt',
            name=f'SAME {event_code} weekly test',
            misfire_grace_time=60,
        )
        log.info('Scheduled %s weekly test', event_code)

    if bool(rmt_cfg.get('enabled', False)):
        event_code = str(rmt_cfg.get('event_code', 'RMT'))
        scheduler.add_job(
            fire_test,
            trigger=_monthly_trigger(rmt_cfg),
            args=[config, feeds, event_code],
            id='rmt',
            name=f'SAME {event_code} monthly test',
            misfire_grace_time=60,
        )
        log.info('Scheduled %s monthly test', event_code)

    station_id_cfg = _station_id_schedule_cfg(config)
    if station_id_cfg['enabled'] and station_id_cfg['minutes']:
        minute_expr = ','.join(str(minute) for minute in station_id_cfg['minutes'])
        scheduler.add_job(
            _queue_station_ids,
            trigger=CronTrigger(minute=minute_expr),
            args=[feeds],
            id='station_id_schedule',
            name='Scheduled station identification',
            misfire_grace_time=30,
        )
        log.info('Scheduled station identification at minute(s): %s', minute_expr)

    date_time_cfg = _date_time_schedule_cfg(config)
    if date_time_cfg['enabled'] and date_time_cfg['minutes']:
        minute_expr = ','.join(str(minute) for minute in date_time_cfg['minutes'])
        scheduler.add_job(
            _queue_date_times,
            trigger=CronTrigger(minute=minute_expr),
            args=[feeds],
            id='date_time_schedule',
            name='Scheduled date and time',
            misfire_grace_time=30,
        )
        log.info('Scheduled date and time at minute(s): %s', minute_expr)

    for feed in feeds:
        feed_id = feed.get('id', 'default')
        scheduler.add_job(
            _run_cleanup,
            trigger=CronTrigger(hour=3, minute=0),
            args=[feed],
            id=f'cleanup:{feed_id}',
            name=f'Nightly cleanup [{feed_id}]',
            misfire_grace_time=3600,
        )
        log.info('Scheduled nightly cleanup for feed: %s', feed_id)

    scheduler.start()
    shutdown_event.wait()
    scheduler.shutdown(wait=False)

from __future__ import annotations

import fnmatch
import json
import logging
import os
import pathlib
import time
import wave
from datetime import datetime
from typing import Any

from managed.events import append_runtime_event, push_alert, shutdown_event
from module.alert import feed_same_codes
from module.same import SAMEHeader, generate_same, to_pcm16
from module.tts import synthesize

log = logging.getLogger(__name__)

_WEEKDAY_MAP: dict[str, int] = {
    'mon': 0, 'tue': 1, 'wed': 2, 'thu': 3, 'fri': 4, 'sat': 5, 'sun': 6,
}


def _load_test_templates() -> dict[str, Any]:
    path = pathlib.Path('managed') / 'sameTemplate.json'
    if not path.exists():
        path = pathlib.Path('managed') / 'sameTest.json'
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        merged: dict[str, Any] = {}
        for item in data:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    return data if isinstance(data, dict) else {}


def _match_lang_key(msg: dict[str, Any], lang: str) -> str | None:
    for pattern, text in msg.items():
        if fnmatch.fnmatch(lang, pattern):
            return str(text)
    return None


def _week_of_month(dt: datetime) -> int:
    return (dt.day - 1) // 7 + 1


def _concat_wavs(paths: list[pathlib.Path]) -> bytes:
    frames: list[bytes] = []
    params: tuple[int, int, int] | None = None
    for p in paths:
        with wave.open(str(p), 'rb') as wf:
            if params is None:
                params = (wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
            frames.append(wf.readframes(wf.getnframes()))
    combined = b''.join(frames)
    if not params:
        return b''
    buf_path = pathlib.Path('/tmp') / f'haze_test_concat_{int(time.time())}.wav'
    with wave.open(str(buf_path), 'wb') as wf:
        wf.setnchannels(params[0])
        wf.setsampwidth(params[1])
        wf.setframerate(params[2])
        wf.writeframes(combined)
    data = buf_path.read_bytes()
    buf_path.unlink(missing_ok=True)
    return data


def fire_test(config: dict[str, Any], feeds: list[dict[str, Any]], event_code: str = 'RWT') -> bool:
    templates = _load_test_templates()
    template = templates.get(event_code)
    if not template:
        log.error('No sameTest.json template found for event code: %s', event_code)
        return False

    msg_map: dict[str, Any] = template.get('msg', {})
    same_expire: str = str(template.get('sameExpire', '0015'))
    same_event: str = str(template.get('sameEvent', event_code))
    same_cfg = config.get('same', {})
    callsign: str = str(same_cfg.get('sender', 'HAZE0000'))
    same_sr: int = int(same_cfg.get('sample_rate_hz', 22050))
    tone_type: str = str(same_cfg.get('default_attention_tone', 'WXR'))

    fired_any = False
    for feed in feeds:
        if not feed.get('enabled', True):
            continue

        feed_id: str = feed.get('id', 'default')
        locations = feed_same_codes(feed) or ['000000']
        feed_langs: list[str] = list(feed.get('languages', {}).keys()) or ['en-CA']

        voice_wavs: list[pathlib.Path] = []
        for lang in feed_langs:
            text = _match_lang_key(msg_map, lang)
            if not text:
                continue
            pkg_id = f'sametest_{event_code}_{lang}'
            wav_path = synthesize(config, text, feed_id, pkg_id, lang=lang)
            if wav_path and wav_path.exists():
                voice_wavs.append(wav_path)

        voice_path: pathlib.Path | None = None
        if voice_wavs:
            if len(voice_wavs) == 1:
                voice_path = voice_wavs[0]
            else:
                combined_bytes = _concat_wavs(voice_wavs)
                combined_path = pathlib.Path('output') / feed_id / f'sametest_{event_code}_combined.wav'
                combined_path.parent.mkdir(parents=True, exist_ok=True)
                combined_path.write_bytes(combined_bytes)
                voice_path = combined_path

        header = SAMEHeader(
            originator='EAS',
            event=same_event,
            locations=locations[:31],
            duration=same_expire,
            callsign=callsign,
        )

        full_signal = generate_same(
            header=header,
            tone_type=tone_type,
            audio_msg_fp32=voice_path,
            sample_rate=same_sr,
            attn_duration_s=8.0,
        )

        out_dir = pathlib.Path('output') / feed_id / 'alerts'
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time())
        out_path = out_dir / f'{event_code.lower()}_{ts}.wav'
        tmp_path = out_dir / f'{event_code.lower()}_{ts}.tmp.wav'
        with wave.open(str(tmp_path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(same_sr)
            wf.writeframes(to_pcm16(full_signal))
        os.replace(str(tmp_path), str(out_path))

        push_alert(feed_id, 0, out_path)
        log.info('[%s] Queued %s test: %s', feed_id, event_code, header.encode())
        fired_any = True

    if fired_any:
        append_runtime_event('test', f'{event_code} test fired')

    return fired_any


def _should_fire_weekly(cfg: dict[str, Any], now: datetime) -> bool:
    weeks_raw = cfg.get('weeks', [])
    weeks: list[int] = [int(w) for w in weeks_raw] if isinstance(weeks_raw, list) else [int(weeks_raw)]
    weekday_str = str(cfg.get('weekday', 'wed')).lower()
    weekday = _WEEKDAY_MAP.get(weekday_str, 2)
    hour = int(cfg.get('hour', 12))
    minute = int(cfg.get('minute', 0))
    return (
        _week_of_month(now) in weeks
        and now.weekday() == weekday
        and now.hour == hour
        and now.minute == minute
    )


def _should_fire_monthly(cfg: dict[str, Any], now: datetime) -> bool:
    week = int(cfg.get('week', 1))
    weekday_str = str(cfg.get('weekday', 'wed')).lower()
    weekday = _WEEKDAY_MAP.get(weekday_str, 2)
    hour = int(cfg.get('hour', 12))
    minute = int(cfg.get('minute', 0))
    return (
        _week_of_month(now) == week
        and now.weekday() == weekday
        and now.hour == hour
        and now.minute == minute
    )


def scheduler_thread_worker(config: dict[str, Any], feeds: list[dict[str, Any]]) -> None:
    same_cfg = config.get('same', {})
    rwt_cfg = same_cfg.get('weeklytest', {})
    rmt_cfg = same_cfg.get('monthlytest', {})

    rwt_enabled = bool(rwt_cfg.get('enabled', False))
    rmt_enabled = bool(rmt_cfg.get('enabled', False))

    if not rwt_enabled and not rmt_enabled:
        return

    last_fired: dict[str, str] = {}

    while not shutdown_event.is_set():
        now = datetime.now()
        minute_key = now.strftime('%Y%m%d%H%M')

        if rwt_enabled and _should_fire_weekly(rwt_cfg, now):
            event_code = str(rwt_cfg.get('event_code', 'RWT'))
            fire_key = f'{event_code}:{minute_key}'
            if last_fired.get(event_code) != fire_key:
                last_fired[event_code] = fire_key
                log.info('Scheduler firing scheduled %s', event_code)
                fire_test(config, feeds, event_code)

        if rmt_enabled and _should_fire_monthly(rmt_cfg, now):
            event_code = str(rmt_cfg.get('event_code', 'RMT'))
            fire_key = f'{event_code}:{minute_key}'
            if last_fired.get(event_code) != fire_key:
                last_fired[event_code] = fire_key
                log.info('Scheduler firing scheduled %s', event_code)
                fire_test(config, feeds, event_code)

        shutdown_event.wait(timeout=20.0)

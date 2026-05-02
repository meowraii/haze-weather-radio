from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
import datetime
import logging
import pathlib
import queue
import subprocess
import threading
import zoneinfo
from typing import Any

from module.events import (
    NowPlayingMetadata,
    PlayableItem,
    append_runtime_event,
    mark_playout_item_started,
    pop_next_playout_item,
    register_alert_queue,
    register_scheduled_queue,
    shutdown_event,
    update_feed_runtime,
)
from module.packages import date_time_package, station_id
from module.output import (
    AudioDeviceSink, FileSink, FramebufferSink, DriSink, V4L2Sink,
    IcecastSink, PiFmAdvSink, RtmpSink, RtpSink, RtspSink, SrtSink, UdpSink,
)
from module.video import VideoIcecastSink, _VIDEO_FORMATS
from module.buffer import CHANNELS, SAMPLE_RATE, AudioPipeline, OnSegmentStart
from module.same import SAMEHeader, generate_same, resample, resolve_flavor, to_pcm16
from module.alert import feed_same_codes
from module.tts import smooth_pcm_edges, synthesize_pcm_stream
from module.static_phrases import splice_date_time, splice_station_id

log = logging.getLogger(__name__)

_CHIME_WAV_PATHS: dict[int, pathlib.Path] = {
    8:  pathlib.Path('audio') / '8-step_chime.wav',
    16: pathlib.Path('audio') / '16-step_chime.wav',
}


def _decode_chime_wav(path: pathlib.Path) -> bytes | None:
    try:
        result = subprocess.run(
            ['ffmpeg', '-loglevel', 'error', '-i', str(path),
             '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS), 'pipe:1'],
            capture_output=True, check=True,
        )
        return result.stdout or None
    except Exception as exc:
        log.warning('Failed to decode chime %s: %s', path.name, exc)
        return None


def _load_chime_pcm() -> dict[int, bytes]:
    pcm: dict[int, bytes] = {}
    for steps, path in _CHIME_WAV_PATHS.items():
        data = _decode_chime_wav(path)
        if data:
            pcm[steps] = data
        else:
            log.warning('Chime %d-step unavailable, will be skipped', steps)
    return pcm


_CHIME_PCM: dict[int, bytes] = _load_chime_pcm()

_SEGMENT_LOOKAHEAD = 3
_PCM_CACHE_LIMIT = 48

_PKG_LABELS: dict[str, str] = {
    'date_time': 'Date and Time',
    'station_id': 'Station Identification',
    'user_bulletin': 'User Bulletin',
    'current_conditions': 'Current Conditions',
    'forecast': 'Forecast',
    'eccc_discussion': 'Discussion',
    'geophysical_alert': 'Geophysical Alert',
    'climate_summary': 'Climate Summary',
    'chime_8': 'Half-Hour Chime',
    'chime_16': 'Top-of-Hour Chime',
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


def _package_gap_s(config: dict[str, Any], feed: dict[str, Any]) -> float:
    default_gap = 0.35
    playout_cfg = config.get('playout', {})
    if isinstance(playout_cfg, dict):
        pacing_cfg = playout_cfg.get('pacing', {})
        if isinstance(pacing_cfg, dict) and pacing_cfg.get('package_gap_s') is not None:
            default_gap = float(pacing_cfg.get('package_gap_s') or 0.0)
        elif playout_cfg.get('package_gap_s') is not None:
            default_gap = float(playout_cfg.get('package_gap_s') or 0.0)

    feed_playout_cfg = _feed_playout_cfg(feed)
    if isinstance(feed_playout_cfg.get('pacing'), dict) and feed_playout_cfg['pacing'].get('package_gap_s') is not None:
        return max(0.0, float(feed_playout_cfg['pacing'].get('package_gap_s') or 0.0))
    if feed_playout_cfg.get('package_gap_s') is not None:
        return max(0.0, float(feed_playout_cfg.get('package_gap_s') or 0.0))
    return max(0.0, default_gap)


def _build_scheduled_station_id_item(config: dict[str, Any], feed: dict[str, Any], feed_id: str) -> PlayableItem:
    lang = _preferred_metadata_language(config, feed)
    return PlayableItem(
        pkg_id='station_id',
        text=station_id(config, feed_id, lang),
        lang=lang,
        metadata=NowPlayingMetadata(title=_PKG_LABELS['station_id']),
        gap_after_s=_package_gap_s(config, feed),
    )


def _build_scheduled_date_time_item(config: dict[str, Any], feed: dict[str, Any]) -> PlayableItem:
    lang = _preferred_metadata_language(config, feed)
    timezone_name = str(feed.get('timezone', 'UTC'))
    return PlayableItem(
        pkg_id='date_time',
        text=date_time_package(timezone_name, lang),
        lang=lang,
        metadata=NowPlayingMetadata(title=_PKG_LABELS['date_time']),
        gap_after_s=_package_gap_s(config, feed),
    )

def _make_segment_callback(
    label: str,
    feed_id: str,
    metadata: NowPlayingMetadata | None,
    metadata_cb: Any,
) -> OnSegmentStart:
    async def _on_start() -> None:
        log.info('[%s] Now playing: %s', feed_id, label)
        update_feed_runtime(feed_id, {
            'now_playing': label,
            'last_played_at': datetime.datetime.now(datetime.UTC).isoformat(),
        })
        if metadata_cb is not None and metadata is not None:
            try:
                await metadata_cb(metadata)
            except Exception:
                pass
    return _on_start


def _preferred_metadata_language(config: dict[str, Any], feed: dict[str, Any]) -> str:
    languages_cfg = feed.get('languages')
    if isinstance(languages_cfg, dict) and languages_cfg:
        return str(next(iter(languages_cfg)))
    return str(feed.get('language') or config.get('language') or 'en-CA')


def _feed_stream_description(feed: dict[str, Any], lang: str) -> str:
    desc_block = feed.get('description')
    if not isinstance(desc_block, dict):
        return ''
    lang_short = lang[:2]
    choices = [
        desc_block.get(lang),
        desc_block.get(lang_short),
        desc_block.get('en-CA'),
        desc_block.get('en'),
    ]
    entry = next((item for item in choices if isinstance(item, dict)), None)
    if entry is None:
        entry = next((item for item in desc_block.values() if isinstance(item, dict)), None)
    if entry is None:
        return ''
    text = str(entry.get('text') or '').strip()
    suffix = str(entry.get('suffix') or '').strip()
    return ' '.join(part for part in (text, suffix) if part).strip()


def _generate_txp(config: dict[str, Any], feed: dict[str, Any]) -> bytes | None:
    same_cfg = config.get('same', {})
    if not same_cfg.get('send_txp_on_startup', False):
        return None
    flavor = same_cfg.get('flavor') or None
    same_sr = resolve_flavor(flavor).sample_rate

    locations = feed_same_codes(feed) or ['000000']
    data = SAMEHeader(
        originator="WXR",
        event="TXP",
        locations=tuple(locations),
        duration="0010",
        callsign=same_cfg.get('callsign', 'TESTCALL'),
    )

    message = generate_same(header=data, tone_type="EGG_TIMER", flavor=flavor)
    return to_pcm16(resample(message, same_sr, SAMPLE_RATE))

def _build_chime_pcm(
    config: dict[str, Any],
    feed: dict[str, Any],
    feed_id: str,
    chime_type: str,
) -> bytes:
    chimes_cfg = config.get('playout', {}).get('chimes', {})
    languages_cfg = feed.get('languages')
    lang = (
        str(next(iter(languages_cfg)))
        if isinstance(languages_cfg, dict) and languages_cfg
        else str(feed.get('language', 'en-CA'))
    )
    tz_name = feed.get('timezone', 'UTC')
    chime_steps = 16 if chime_type == 'top' else 8
    chime_key = 'top_of_hour' if chime_type == 'top' else 'half_hour'
    default_seq = ['chime']
    chime_seq: list[str] = chimes_cfg.get(chime_key, {}).get('sequence', default_seq)

    try:
        tz_obj = zoneinfo.ZoneInfo(tz_name)
    except Exception:
        tz_obj = zoneinfo.ZoneInfo('UTC')
    now = datetime.datetime.now(tz_obj)
    tz_abbr = now.tzname() or 'UTC'

    parts: list[bytes] = []
    for ci in chime_seq:
        pkg = f'chime_{chime_steps}' if ci == 'chime' else ci
        if pkg in ('chime_8', 'chime_16'):
            steps = 16 if pkg == 'chime_16' else 8
            pcm = _CHIME_PCM.get(steps)
            if pcm:
                parts.append(pcm)
            else:
                log.warning('[%s] Chime WAV %d-step unavailable', feed_id, steps)
        elif pkg == 'date_time':
            pcm = splice_date_time(config, lang, tz_abbr, now)
            if pcm:
                parts.append(pcm)
        elif pkg == 'station_id':
            pcm = splice_station_id(feed_id, lang)
            if pcm:
                parts.append(pcm)

    return b''.join(parts)


async def _produce_tts(
    pipeline: AudioPipeline,
    shutdown: asyncio.Event,
    feed_id: str,
    config: dict[str, Any],
    feed: dict[str, Any],
    on_air_name: str,
    metadata_cb: Any,
    txp_pcm: bytes | None,
    scheduled_queue: asyncio.Queue | None = None,
) -> None:
    pcm_cache: OrderedDict[tuple[str, str, str | None, str], bytes] = OrderedDict()

    def _cache_get(key: tuple[str, str, str | None, str]) -> bytes | None:
        cached = pcm_cache.get(key)
        if cached is None:
            return None
        pcm_cache.move_to_end(key)
        return cached

    def _cache_put(key: tuple[str, str, str | None, str], pcm: bytes) -> bytes:
        pcm_cache[key] = pcm
        pcm_cache.move_to_end(key)
        while len(pcm_cache) > _PCM_CACHE_LIMIT:
            pcm_cache.popitem(last=False)
        return pcm

    if txp_pcm:
        log.info('[%s] Sending TXP (Transmitter Primary On)', feed_id)
        update_feed_runtime(feed_id, {
            'txp_sent_at': datetime.datetime.now(datetime.UTC).isoformat(),
        })
        append_runtime_event('txp', 'Transmitter Primary On sent', feed_id)
        await pipeline.enqueue_segment(txp_pcm)

    segment_consumed = pipeline.segment_consumed_event

    def _next_scheduled_package() -> str | None:
        if scheduled_queue is None:
            return None
        pkg_id: str | None = None
        while True:
            try:
                pkg_id = scheduled_queue.get_nowait()
            except asyncio.QueueEmpty:
                return pkg_id

    while not shutdown.is_set():
        if pipeline.alert_active:
            await pipeline.wait_alert_done()
            continue

        scheduled_insert = False
        scheduled_pkg_id = _next_scheduled_package()
        if scheduled_pkg_id == 'station_id':
            item = _build_scheduled_station_id_item(config, feed, feed_id)
            scheduled_insert = True
        elif scheduled_pkg_id == 'date_time':
            item = _build_scheduled_date_time_item(config, feed)
            scheduled_insert = True
        else:
            item = pop_next_playout_item(feed_id)
        if item is None:
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass
            continue

        if shutdown.is_set():
            return

        if pipeline.alert_active:
            await pipeline.wait_alert_done()

        while pipeline.segments_queued >= _SEGMENT_LOOKAHEAD and not shutdown.is_set():
            if pipeline.alert_active:
                break
            segment_consumed.clear()
            try:
                await asyncio.wait_for(segment_consumed.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass

        if shutdown.is_set():
            return

        if not scheduled_insert:
            mark_playout_item_started(feed_id, item)

        pkg_id = item.pkg_id
        label = item.metadata.title if item.metadata and item.metadata.title else _PKG_LABELS.get(pkg_id, pkg_id.replace('_', ' ').title())
        pcm_data: bytes | None = None
        segment_source: bytes | Any | None = None
        smooth_static_source = False
        if pkg_id == 'date_time':
            try:
                tz = zoneinfo.ZoneInfo(feed.get('timezone', 'UTC'))

                def _splice_date_time_live(lang: str = item.lang, timezone_obj: zoneinfo.ZoneInfo = tz) -> bytes | None:
                    now_local = datetime.datetime.now(timezone_obj)
                    abbr = now_local.tzname() or 'UTC'
                    return splice_date_time(config, lang, abbr, now_local)

                segment_source = _splice_date_time_live
                smooth_static_source = True
            except Exception:
                log.debug('[%s] date_time splice failed, falling back to TTS', feed_id, exc_info=True)
        elif pkg_id == 'station_id':
            def _splice_station_id_live(lang: str = item.lang) -> bytes | None:
                return splice_station_id(feed_id, lang)

            segment_source = _splice_station_id_live
            smooth_static_source = True

        elif pkg_id in ('chime_8', 'chime_16'):
            steps = 16 if pkg_id == 'chime_16' else 8
            segment_source = _CHIME_PCM.get(steps)
            smooth_static_source = True

        if segment_source is None:
            cache_key = (pkg_id, item.lang, item.voice, item.text)
            pcm_data = _cache_get(cache_key)
            if pcm_data is None:
                def _render_pcm(text=item.text, lang=item.lang, voice=item.voice) -> bytes:
                    return b''.join(synthesize_pcm_stream(config, text, lang, voice))

                pcm_data = await asyncio.to_thread(_render_pcm)
                if pcm_data:
                    pcm_data = _cache_put(cache_key, pcm_data)
            segment_source = pcm_data
        elif smooth_static_source:
            if isinstance(segment_source, bytes):
                segment_source = smooth_pcm_edges(segment_source)
            elif callable(segment_source):
                raw_source = segment_source

                def _smoothed_segment_source(source=raw_source) -> bytes | None:
                    data = source()
                    if not data:
                        return data
                    return smooth_pcm_edges(data)

                segment_source = _smoothed_segment_source

        if not segment_source:
            log.warning('[%s] Empty synthesis for %s', feed_id, pkg_id)
            continue

        on_start = _make_segment_callback(label, feed_id, item.metadata, metadata_cb)
        await pipeline.enqueue_segment(segment_source, on_start, gap_after_s=item.gap_after_s)

async def _alert_feeder(
    pipeline: AudioPipeline,
    alert_queue: asyncio.PriorityQueue[tuple[int, bytes, str]],
    shutdown: asyncio.Event,
    feed_id: str,
    on_air_name: str,
    metadata_cb: Any,
    alert_start_cbs: list[Any] | None = None,
    alert_end_cbs: list[Any] | None = None,
) -> None:
    alert_metadata = NowPlayingMetadata(title='Alert')
    while not shutdown.is_set():
        try:
            priority, pcm_data, identifier = await asyncio.wait_for(
                alert_queue.get(), timeout=0.5,
            )
        except asyncio.TimeoutError:
            continue

        log.info('[%s] Playing alert: %s', feed_id, identifier or '(unnamed)')
        update_feed_runtime(feed_id, {
            'now_playing': 'Alert',
            'last_played_at': datetime.datetime.now(datetime.UTC).isoformat(),
        })
        if metadata_cb is not None:
            try:
                await metadata_cb(alert_metadata)
            except Exception:
                pass

        if alert_start_cbs:
            results = await asyncio.gather(
                *(cb(identifier) for cb in alert_start_cbs),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    log.debug('[%s] alert_start_cb failed: %s', feed_id, result)

        pipeline.enqueue_alert(pcm_data)
        await pipeline.wait_alert_done()

        if alert_end_cbs:
            results = await asyncio.gather(
                *(cb() for cb in alert_end_cbs),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    log.debug('[%s] alert_end_cb failed: %s', feed_id, result)

def _build_file_path(config: dict[str, Any], feed: dict[str, Any]) -> str:
    file_cfg = config.get('output', {}).get('file', {})
    directory = file_cfg.get('directory', './audio')
    pattern = file_cfg.get('filename_pattern', '{feed_id}-{timestamp}.bin')
    filename = pattern.format(
        feed_id=feed['id'],
        timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    path = pathlib.Path(directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)

def _find_video_cfg(config: dict[str, Any], feed_id: str) -> dict[str, Any] | None:
    for vc in (config.get('video') or []):
        if isinstance(vc, dict) and vc.get('feed') == feed_id and vc.get('enabled', True):
            return vc
    return None


async def feed_runner(
    config: dict[str, Any],
    feed: dict[str, Any],
    alert_queue: asyncio.PriorityQueue[tuple[int, bytes, str]],
    scheduled_queue: asyncio.Queue,
    shutdown: asyncio.Event,
) -> None:
    pipeline = AudioPipeline()
    feed_id = feed['id']
    output_cfg = feed.get('output', {})
    operator = config.get('operator', {})
    on_air_name: str = operator.get('on_air_name') or operator.get('name') or feed.get('name', feed_id)
    operator_name: str = str(operator.get('operator_name') or operator.get('name') or '').strip()
    stream_identity = f'{on_air_name} ({feed_id})' if on_air_name else feed_id
    metadata_lang = _preferred_metadata_language(config, feed)
    stream_description = _feed_stream_description(feed, metadata_lang)
    metadata_cbs: list[Any] = []
    alert_start_cbs: list[Any] = []
    alert_end_cbs: list[Any] = []

    update_feed_runtime(feed_id, {
        'display_name': feed.get('name', feed_id),
        'on_air_name': on_air_name,
        'started_at': datetime.datetime.now(datetime.UTC).isoformat(),
    })

    if output_cfg.get('stream', {}).get('enabled'):
        try:
            raw_stream_cfg = output_cfg['stream']
            stream_fmt = str(raw_stream_cfg.get('format', 'opus'))
            if stream_fmt in _VIDEO_FORMATS:
                video_cfg = _find_video_cfg(config, feed_id)
                if video_cfg:
                    vsink = VideoIcecastSink(feed, video_cfg, raw_stream_cfg)
                    pipeline.attach_sink(vsink, name=f'{feed_id}:video')
                    alert_start_cbs.append(vsink.on_alert_start)
                    alert_end_cbs.append(vsink.on_alert_end)
                else:
                    log.warning(
                        '[%s] Stream format %r is a video format but no video config found — stream skipped',
                        feed_id, stream_fmt,
                    )
            else:
                stream_cfg = {
                    **raw_stream_cfg,
                    'feed_id': feed_id,
                    'stream_name': stream_identity,
                    'stream_description': stream_description,
                    'stream_genre': 'Weather Radio',
                    'stream_album': stream_identity,
                    'stream_creator': operator_name,
                    'stream_artist': on_air_name,
                }
                sink = IcecastSink(stream_cfg)
                metadata_cbs.append(sink.set_metadata)
                pipeline.attach_sink(sink, name=f'{feed_id}:icecast')
        except Exception:
            log.exception('Failed to start stream sink for %s', feed_id)

    if output_cfg.get('audio_device', {}).get('enabled'):
        try:
            sink = AudioDeviceSink(output_cfg['audio_device'].get('name'))
            pipeline.attach_sink(sink, name=f'{feed_id}:audio_device')
        except Exception:
            log.exception('Failed to start audio device sink for %s', feed_id)

    if output_cfg.get('file', {}).get('enabled'):
        try:
            path = _build_file_path(config, feed)
            sink = FileSink(path)
            pipeline.attach_sink(sink, name=f'{feed_id}:file')
        except Exception:
            log.exception('Failed to start file sink for %s', feed_id)

    if output_cfg.get('PiFmAdv', {}).get('enabled'):
        try:
            sink = PiFmAdvSink(output_cfg['PiFmAdv'])
            pipeline.attach_sink(sink, name=f'{feed_id}:PiFmAdv')
        except Exception:
            log.exception('Failed to start PiFmAdv sink for %s', feed_id)

    _video_cfg = _find_video_cfg(config, feed_id)
    _feed_tz = str(feed.get('timezone', 'UTC'))

    for _sink_key, _factory, _label in (
        ('udp',       lambda cfg: UdpSink(cfg, feed_id, _video_cfg, _feed_tz),       'UDP'),
        ('rtp',       lambda cfg: RtpSink(cfg, feed_id, _video_cfg, _feed_tz),       'RTP'),
        ('rtmp',      lambda cfg: RtmpSink(cfg, feed_id, _video_cfg, _feed_tz),      'RTMP'),
        ('srt',       lambda cfg: SrtSink(cfg, feed_id, _video_cfg, _feed_tz),       'SRT'),
        ('rtsp',      lambda cfg: RtspSink(cfg, feed_id, _video_cfg, _feed_tz),      'RTSP'),
        ('framebuffer', lambda cfg: FramebufferSink(cfg, feed_id, _video_cfg, _feed_tz), 'framebuffer'),
        ('dri',       lambda cfg: DriSink(cfg, feed_id, _video_cfg, _feed_tz),       'DRI'),
        ('v4l2',      lambda cfg: V4L2Sink(cfg, feed_id, _video_cfg, _feed_tz),      'V4L2'),
    ):
        _sink_cfg = output_cfg.get(_sink_key, {})
        if not (_sink_cfg and _sink_cfg.get('enabled')):
            continue
        try:
            sink = _factory(_sink_cfg)
            pipeline.attach_sink(sink, name=f'{feed_id}:{_sink_key}')
            if hasattr(sink, 'on_alert_start'):
                alert_start_cbs.append(sink.on_alert_start)
            if hasattr(sink, 'on_alert_end'):
                alert_end_cbs.append(sink.on_alert_end)
        except Exception:
            log.exception('Failed to start %s sink for %s', _label, feed_id)

    async def _update_metadata(metadata: NowPlayingMetadata) -> None:
        for cb in metadata_cbs:
            try:
                await cb(metadata)
            except Exception:
                pass

    txp_pcm = _generate_txp(config, feed)
    pipeline.start()

    tts_task = asyncio.create_task(
        _produce_tts(
            pipeline, shutdown, feed_id, config, feed, on_air_name,
            _update_metadata if metadata_cbs else None, txp_pcm,
            scheduled_queue,
        ),
        name=f'{feed_id}:tts_producer',
    )

    alert_task = asyncio.create_task(
        _alert_feeder(
            pipeline, alert_queue, shutdown, feed_id, on_air_name,
            _update_metadata if metadata_cbs else None,
            alert_start_cbs or None,
            alert_end_cbs or None,
        ),
        name=f'{feed_id}:alert_feeder',
    )

    try:
        await asyncio.gather(tts_task, alert_task, return_exceptions=True)
    finally:
        await pipeline.stop()

async def _watch_shutdown(shutdown: asyncio.Event) -> None:
    while not shutdown_event.is_set():
        await asyncio.sleep(0.5)
    shutdown.set()

async def _playout_main(
    config: dict[str, Any],
    feed: dict[str, Any],
) -> None:
    feed_id = feed.get('id', '')
    shutdown = asyncio.Event()
    watchdog = asyncio.create_task(_watch_shutdown(shutdown))
    alert_queue: asyncio.PriorityQueue[tuple[int, bytes, str]] = asyncio.PriorityQueue()
    scheduled_queue: asyncio.Queue = asyncio.Queue()
    thread_alert_q = register_alert_queue(feed_id)
    thread_scheduled_q = register_scheduled_queue(feed_id)
    loop = asyncio.get_running_loop()

    _bridge_stop = asyncio.Event()

    def _bridge_thread() -> None:
        while not shutdown_event.is_set() and not _bridge_stop.is_set():
            try:
                item = thread_alert_q.get(timeout=0.1)
                loop.call_soon_threadsafe(alert_queue.put_nowait, item)
            except queue.Empty:
                pass
            try:
                pkg_id = thread_scheduled_q.get_nowait()
                loop.call_soon_threadsafe(scheduled_queue.put_nowait, pkg_id)
            except queue.Empty:
                pass

    bridge = threading.Thread(target=_bridge_thread, name=f'{feed_id}:bridge', daemon=True)
    bridge.start()

    try:
        await feed_runner(config, feed, alert_queue, scheduled_queue, shutdown)
    except BrokenPipeError:
        log.warning('Playout pipe closed for %s', feed_id)
    finally:
        _bridge_stop.set()
        watchdog.cancel()

def playout_thread_worker(
    config: dict[str, Any],
    feed: dict[str, Any],
) -> None:
    feed_id = feed.get('id', '')
    backoff = 1.0
    max_backoff = 60.0
    while not shutdown_event.is_set():
        try:
            asyncio.run(_playout_main(config, feed))
            break
        except Exception:
            log.exception('Playout for %s exited unexpectedly; restarting in %.1fs', feed_id, backoff)
            if shutdown_event.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2.0, max_backoff)

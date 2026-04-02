from __future__ import annotations

import asyncio
import datetime
import logging
import pathlib
import queue
from typing import Any
import json

from managed.events import (
    append_runtime_event,
    get_playout_sequence,
    register_alert_queue,
    shutdown_event,
    update_feed_runtime,
)
from module.output import AudioDeviceSink, FileSink, IcecastSink, PiFmAdvSink
from module.queue import SAMPLE_RATE, AudioPipeline, OnSegmentStart
from module.same import SAMEHeader, generate_same, resample, to_pcm16
from module.alert import feed_same_codes
from module.tts import synthesize_pcm_stream

log = logging.getLogger(__name__)

_PKG_LABELS: dict[str, str] = {
    'date_time': 'Date and Time',
    'station_id': 'Station Identification',
    'user_bulletin': 'User Bulletin',
    'current_conditions': 'Current Conditions',
    'forecast': 'Forecast',
    'eccc_discussion': 'Discussion',
    'geophysical_alert': 'Geophysical Alert',
    'climate_summary': 'Climate Summary',
}

def _make_segment_callback(
    label: str,
    feed_id: str,
    on_air_name: str,
    metadata_cb: Any,
) -> OnSegmentStart:
    async def _on_start() -> None:
        log.info('[%s] Now playing: %s', feed_id, label)
        update_feed_runtime(feed_id, {
            'now_playing': label,
            'last_played_at': datetime.datetime.now(datetime.UTC).isoformat(),
        })
        if metadata_cb is not None:
            title = f"{label} - {on_air_name} ({feed_id})" if on_air_name else label
            try:
                await metadata_cb(title)
            except Exception:
                pass
    return _on_start


def _generate_txp(config: dict[str, Any], feed: dict[str, Any]) -> bytes | None:
    same_cfg = config.get('same', {})
    if not same_cfg.get('send_txp_on_startup', False):
        return None

    locations = feed_same_codes(feed) or ['000000']
    data = SAMEHeader(
        originator="WXR",
        event="TXP",
        locations=locations,
        duration="0010",
        callsign=same_cfg.get('callsign', 'TESTCALL'),
    )

    same_sr = same_cfg.get('sample_rate_hz', 22050)
    message = generate_same(header=data, tone_type="EGG_TIMER", sample_rate=same_sr)
    return to_pcm16(resample(message, same_sr, SAMPLE_RATE))

async def _produce_tts(
    pipeline: AudioPipeline,
    shutdown: asyncio.Event,
    feed_id: str,
    config: dict[str, Any],
    on_air_name: str,
    metadata_cb: Any,
    txp_pcm: bytes | None,
) -> None:
    spool_dir = pathlib.Path('output') / feed_id / 'spool'
    await asyncio.to_thread(lambda: spool_dir.mkdir(parents=True, exist_ok=True))
    seg_counter = 0

    if txp_pcm:
        log.info('[%s] Sending TXP (Transmitter Primary On)', feed_id)
        update_feed_runtime(feed_id, {
            'txp_sent_at': datetime.datetime.now(datetime.UTC).isoformat(),
        })
        append_runtime_event('txp', 'Transmitter Primary On sent', feed_id)
        seg_counter += 1
        txp_path = spool_dir / f'seg_{seg_counter:06d}.bin'
        await asyncio.to_thread(txp_path.write_bytes, txp_pcm)
        await pipeline.enqueue_segment(txp_path)

    while not shutdown.is_set():
        if pipeline.alert_active:
            await asyncio.sleep(0.1)
            continue

        seq = get_playout_sequence(feed_id)
        if not seq:
            await asyncio.sleep(0.5)
            continue

        for item in seq:
            if shutdown.is_set():
                return

            while pipeline.alert_active and not shutdown.is_set():
                await asyncio.sleep(0.1)

            pkg_id = item.pkg_id
            label = _PKG_LABELS.get(pkg_id, pkg_id.replace('_', ' ').title())

            seg_counter += 1
            seg_path = spool_dir / f'seg_{seg_counter:06d}.bin'

            def _render(text=item.text, lang=item.lang, voice=item.voice, path=seg_path):
                stream = synthesize_pcm_stream(config, text, lang, voice)
                with open(path, 'wb') as f:
                    for chunk in stream:
                        f.write(chunk)

            await asyncio.to_thread(_render)

            if not seg_path.exists() or seg_path.stat().st_size == 0:
                log.warning('[%s] Empty synthesis for %s', feed_id, pkg_id)
                seg_path.unlink(missing_ok=True)
                continue

            on_start = _make_segment_callback(label, feed_id, on_air_name, metadata_cb)
            await pipeline.enqueue_segment(seg_path, on_start)

async def _alert_feeder(
    pipeline: AudioPipeline,
    alert_queue: asyncio.PriorityQueue[tuple[int, bytes, str]],
    shutdown: asyncio.Event,
    feed_id: str,
    on_air_name: str,
    metadata_cb: Any,
) -> None:
    while not shutdown.is_set():
        try:
            priority, pcm_data, identifier = alert_queue.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(0.1)
            continue

        log.info('[%s] Playing alert: %s', feed_id, identifier or '(unnamed)')
        update_feed_runtime(feed_id, {
            'now_playing': 'Alert',
            'last_played_at': datetime.datetime.now(datetime.UTC).isoformat(),
        })
        if metadata_cb is not None:
            title = f"Alert - {on_air_name} ({feed_id})" if on_air_name else "Alert"
            try:
                await metadata_cb(title)
            except Exception:
                pass

        pipeline.enqueue_alert(pcm_data)
        await pipeline.wait_alert_done()

def _build_file_path(config: dict[str, Any], feed: dict[str, Any]) -> str:
    file_cfg = config.get('output', {}).get('file', {})
    directory = file_cfg.get('directory', './output')
    pattern = file_cfg.get('filename_pattern', '{feed_id}-{timestamp}.bin')
    filename = pattern.format(
        feed_id=feed['id'],
        timestamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),
    )
    path = pathlib.Path(directory) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)

async def feed_runner(
    config: dict[str, Any],
    feed: dict[str, Any],
    alert_queue: asyncio.PriorityQueue[tuple[int, bytes, str]],
    shutdown: asyncio.Event,
) -> None:
    pipeline = AudioPipeline()
    feed_id = feed['id']
    output_cfg = feed.get('output', {})
    operator = config.get('operator', {})
    on_air_name: str = operator.get('on_air_name') or operator.get('name') or feed.get('name', feed_id)
    metadata_cbs: list[Any] = []

    update_feed_runtime(feed_id, {
        'display_name': feed.get('name', feed_id),
        'on_air_name': on_air_name,
        'started_at': datetime.datetime.now(datetime.UTC).isoformat(),
    })

    if output_cfg.get('stream', {}).get('enabled'):
        try:
            stream_cfg = {**output_cfg['stream'], 'feed_id': feed_id}
            sink = IcecastSink(stream_cfg)
            metadata_cbs.append(sink.set_metadata)
            pipeline.attach_sink(sink, name=f'{feed_id}:icecast')
        except Exception:
            log.exception('Failed to start Icecast sink for %s', feed_id)

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

    async def _update_metadata(title: str) -> None:
        for cb in metadata_cbs:
            try:
                await cb(title)
            except Exception:
                pass

    txp_pcm = _generate_txp(config, feed)
    pipeline.start()

    tts_task = asyncio.create_task(
        _produce_tts(
            pipeline, shutdown, feed_id, config, on_air_name,
            _update_metadata if metadata_cbs else None, txp_pcm,
        ),
        name=f'{feed_id}:tts_producer',
    )

    alert_task = asyncio.create_task(
        _alert_feeder(
            pipeline, alert_queue, shutdown, feed_id, on_air_name,
            _update_metadata if metadata_cbs else None,
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

async def _drain_thread_alerts(
    thread_q: queue.Queue[tuple[int, bytes, str]],
    async_q: asyncio.PriorityQueue[tuple[int, bytes, str]],
) -> bool:
    drained = False
    while True:
        try:
            item = thread_q.get_nowait()
            await async_q.put(item)
            drained = True
        except queue.Empty:
            break
    return drained

async def _playout_main(
    config: dict[str, Any],
    feed: dict[str, Any],
) -> None:
    feed_id = feed.get('id', '')
    shutdown = asyncio.Event()
    watchdog = asyncio.create_task(_watch_shutdown(shutdown))
    alert_queue: asyncio.PriorityQueue[tuple[int, bytes, str]] = asyncio.PriorityQueue()
    thread_alert_q = register_alert_queue(feed_id)

    async def _poll_alerts() -> None:
        while not shutdown.is_set():
            await _drain_thread_alerts(thread_alert_q, alert_queue)
            await asyncio.sleep(0.5)

    poller = asyncio.create_task(_poll_alerts(), name=f'{feed_id}:alert_poll')
    try:
        await feed_runner(config, feed, alert_queue, shutdown)
    except BrokenPipeError:
        log.warning('Playout pipe closed for %s', feed_id)
    finally:
        poller.cancel()
        watchdog.cancel()

def playout_thread_worker(
    config: dict[str, Any],
    feed: dict[str, Any],
) -> None:
    asyncio.run(_playout_main(config, feed))

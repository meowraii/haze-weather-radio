from __future__ import annotations

import asyncio
import datetime
import logging
import pathlib
import queue
import random
import wave
from typing import Any

from managed.events import (
    append_runtime_event,
    get_playout_sequence,
    register_alert_queue,
    shutdown_event,
    update_feed_runtime,
)
from module.output import AudioDeviceSink, FileSink, IcecastSink, PiFmAdvSink
from module.queue import CHANNELS, SAMPLE_RATE, AudioBus, PlaybackInterrupted, sink_runner
from module.same import SAMEHeader, generate_same, resample, to_pcm16
from module.alert import feed_same_codes

log = logging.getLogger(__name__)

_PKG_LABELS: dict[str, str] = {
    'date_time': 'Date and Time',
    'station_id': 'Station Identification',
    'user_bulletin': 'User Bulletin',
    'current_conditions': 'Current Conditions',
    'forecast': 'Forecast',
    'eccc_discussion': 'Discussion',
    'geophysical_alert': 'Geophysical Alert',
}

def _ensure_silence(feed_id: str) -> pathlib.Path:
    path = pathlib.Path('output') / feed_id / 'silence_1s.wav'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        n = int(22050 * 1.0)
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(bytes(n * 2))
    return path

async def _decode_wav(path: pathlib.Path) -> bytes:
    proc = await asyncio.create_subprocess_exec(
        'ffmpeg', '-loglevel', 'error',
        '-i', str(path),
        '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
        'pipe:1',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    return stdout

def _generate_txp(config: dict[str, Any], feed: dict[str, Any]) -> bytes | None:
    same_cfg = config.get('same', {})
    if not same_cfg.get('send_txp_on_startup', False):
        return None

    callsign = same_cfg.get('sender', 'HAZE0000')
    locations = feed_same_codes(feed)
    if not locations:
        locations = ['000000']

    data = SAMEHeader(
        originator="WXR",
        event="TXP",
        locations=locations,
        duration="0010",
        callsign="XLF323",
    )

    message = generate_same(
        header=data,
        tone_type="EGG_TIMER",
        sample_rate=config.get('same', {}).get('sample_rate_hz', 22050),
    )

    same_sr = config.get('same', {}).get('sample_rate_hz', 22050)
    return to_pcm16(resample(message, same_sr, SAMPLE_RATE))

async def pipe_writer(
    bus: AudioBus,
    alert_queue: asyncio.PriorityQueue[tuple[int, pathlib.Path]],
    shutdown: asyncio.Event,
    alert_interrupt: asyncio.Event,
    feed_id: str = '',
    on_air_name: str = '',
    metadata_cb: Any = None,
    shuffle: bool = False,
    shuffle_carry_over: int = 0,
    txp_pcm: bytes | None = None,
) -> None:
    silence_path = _ensure_silence(feed_id)
    silence_pcm: bytes | None = None
    _pinned = {'date_time', 'station_id'}
    _prev_tail: list[pathlib.Path] = []

    if txp_pcm:
        log.info('[%s] Sending TXP (Transmitter Primary On)', feed_id)
        update_feed_runtime(feed_id, {
            'txp_sent_at': datetime.datetime.now(datetime.UTC).isoformat(),
        })
        append_runtime_event('txp', 'Transmitter Primary On sent', feed_id)
        await bus.write(txp_pcm)

    async def _play(p: pathlib.Path, interrupt: asyncio.Event | None = None) -> None:
        nonlocal silence_pcm
        pkg_id = p.stem
        label = _PKG_LABELS.get(pkg_id, pkg_id.replace('_', ' ').title())
        log.info('[%s] Now playing: %s', feed_id, label)
        update_feed_runtime(feed_id, {
            'now_playing': label,
            'now_playing_file': str(p),
            'last_played_at': datetime.datetime.now(datetime.UTC).isoformat(),
        })
        if metadata_cb is not None:
            title = f"{label} - {on_air_name} ({feed_id})" if on_air_name else label
            try:
                await metadata_cb(title)
            except Exception:
                pass
        pcm = await _decode_wav(p)
        if pcm:
            await bus.write(pcm, interrupt=interrupt)
        if silence_pcm is None:
            silence_pcm = await _decode_wav(silence_path)
        if silence_pcm:
            await bus.write(silence_pcm, interrupt=interrupt)

    async def _drain_alerts() -> None:
        while not alert_queue.empty():
            try:
                _, alert_path = alert_queue.get_nowait()
                await _play(alert_path)
            except asyncio.QueueEmpty:
                break

    try:
        while not shutdown.is_set():
            await _drain_alerts()

            seq = get_playout_sequence(feed_id)
            if not seq:
                await asyncio.sleep(0.5)
                continue

            if shuffle:
                pinned = [(i, p) for i, p in enumerate(seq) if p.stem in _pinned]
                shuffleable = [p for p in seq if p.stem not in _pinned]

                carry = set(str(p) for p in _prev_tail) if shuffle_carry_over else set()
                front: list[pathlib.Path] = []
                rest: list[pathlib.Path] = []
                for p in shuffleable:
                    (front if str(p) in carry else rest).append(p)

                random.shuffle(rest)
                random.shuffle(front)
                shuffled = rest + front

                ordered: list[pathlib.Path] = list(shuffled)
                for idx, p in pinned:
                    pos = min(idx, len(ordered))
                    ordered.insert(pos, p)

                if shuffle_carry_over and ordered:
                    _prev_tail = ordered[-shuffle_carry_over:]

                seq = ordered

            interrupted = False
            for path in seq:
                if shutdown.is_set():
                    break

                await _drain_alerts()

                deadline = asyncio.get_running_loop().time() + 60.0
                while not path.exists():
                    if shutdown.is_set() or asyncio.get_running_loop().time() > deadline:
                        break
                    await asyncio.sleep(0.5)

                if not path.exists():
                    log.warning('[%s] Skipping missing: %s', feed_id, path.name)
                    continue

                try:
                    await _play(path, interrupt=alert_interrupt)
                except PlaybackInterrupted:
                    log.info('[%s] Playback interrupted by incoming alert', feed_id)
                    bus.flush()
                    alert_interrupt.clear()
                    await _drain_alerts()
                    interrupted = True
                    break

            if interrupted:
                continue
    finally:
        await bus.close()

def _build_file_path(config: dict[str, Any], feed: dict[str, Any]) -> str:
    file_cfg = config.get('output', {}).get('file', {})
    directory = file_cfg.get('directory', './output')
    pattern = file_cfg.get('filename_pattern', '{feed_id}-{timestamp}.raw')
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
    alert_queue: asyncio.PriorityQueue[tuple[int, pathlib.Path]],
    shutdown: asyncio.Event,
    alert_interrupt: asyncio.Event | None = None,
) -> None:
    bus = AudioBus()
    sink_tasks: list[asyncio.Task[None]] = []
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
            sink_tasks.append(asyncio.create_task(
                sink_runner(sink, bus.subscribe(), shutdown),
                name=f'{feed_id}:icecast',
            ))
        except Exception:
            log.exception('Failed to start Icecast sink for %s', feed_id)

    if output_cfg.get('audio_device', {}).get('enabled'):
        try:
            sink = AudioDeviceSink(output_cfg['audio_device'].get('name'))
            sink_tasks.append(asyncio.create_task(
                sink_runner(sink, bus.subscribe(), shutdown),
                name=f'{feed_id}:audio_device',
            ))
        except Exception:
            log.exception('Failed to start audio device sink for %s', feed_id)

    if output_cfg.get('file', {}).get('enabled'):
        try:
            path = _build_file_path(config, feed)
            sink = FileSink(path)
            sink_tasks.append(asyncio.create_task(
                sink_runner(sink, bus.subscribe(), shutdown),
                name=f'{feed_id}:file',
            ))
        except Exception:
            log.exception('Failed to start file sink for %s', feed_id)

    if output_cfg.get('PiFmAdv', {}).get('enabled'):
        try:
            sink = PiFmAdvSink(output_cfg['PiFmAdv'])
            sink_tasks.append(asyncio.create_task(
                sink_runner(sink, bus.subscribe(), shutdown),
                name=f'{feed_id}:PiFmAdv',
            ))
        except Exception:
            log.exception('Failed to start PiFmAdv sink for %s', feed_id)

    async def _update_metadata(title: str) -> None:
        for cb in metadata_cbs:
            try:
                await cb(title)
            except Exception:
                pass

    playout_cfg = config.get('playout', {})
    txp_pcm = _generate_txp(config, feed)

    writer_task = asyncio.create_task(
        pipe_writer(
            bus, alert_queue, shutdown, alert_interrupt or asyncio.Event(),
            feed_id=feed_id,
            on_air_name=on_air_name,
            metadata_cb=_update_metadata if metadata_cbs else None,
            shuffle=playout_cfg.get('shuffle', False),
            shuffle_carry_over=playout_cfg.get('shuffle_carry_over', 0),
            txp_pcm=txp_pcm,
        ),
        name=f'{feed_id}:writer',
    )

    await asyncio.gather(writer_task, *sink_tasks, return_exceptions=True)


async def _watch_shutdown(shutdown: asyncio.Event) -> None:
    while not shutdown_event.is_set():
        await asyncio.sleep(0.5)
    shutdown.set()


async def _drain_thread_alerts(
    thread_q: queue.Queue[tuple[int, pathlib.Path]],
    async_q: asyncio.PriorityQueue[tuple[int, pathlib.Path]],
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
    alert_queue: asyncio.PriorityQueue[tuple[int, pathlib.Path]] = asyncio.PriorityQueue()
    alert_interrupt = asyncio.Event()
    thread_alert_q = register_alert_queue(feed_id)

    async def _poll_alerts() -> None:
        while not shutdown.is_set():
            if await _drain_thread_alerts(thread_alert_q, alert_queue):
                alert_interrupt.set()
            await asyncio.sleep(1.0)

    poller = asyncio.create_task(_poll_alerts(), name=f'{feed_id}:alert_poll')
    try:
        await feed_runner(config, feed, alert_queue, shutdown, alert_interrupt)
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

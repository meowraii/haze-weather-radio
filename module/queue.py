from __future__ import annotations

import asyncio
import logging
import pathlib
import subprocess
import time
from typing import Any, Callable, Coroutine, Protocol

from .tts import load_config

log = logging.getLogger(__name__)

config = load_config()

CHUNK_SAMPLES = 2048
SAMPLE_RATE = config.get('playout', {}).get('sample_rate', 16000)
CHANNELS = config.get('playout', {}).get('channels', 1)
BYTES_PER_SAMPLE = 2

CHUNK_BYTES = CHUNK_SAMPLES * CHANNELS * BYTES_PER_SAMPLE
CHUNK_DURATION = CHUNK_BYTES / (SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE)
SILENCE_CHUNK = bytes(CHUNK_BYTES)

_SEGMENT_QUEUE_LIMIT = 12
_GAP_CHUNKS = max(1, round(1.0 / CHUNK_DURATION))
_GAP_SILENCE: bytes = b''
_AUDIO_DIR = pathlib.Path(__file__).parent.parent / 'audio'
_BUS_ALARM_PATH = _AUDIO_DIR / 'alarm.wav'
_BUS_SILENCE_PATH = _AUDIO_DIR / 'silence.wav'
_BUS_SILENCE_SECONDS = 1.0
_BUS_SILENCE_LOOPS = 2
_ASSET_PCM_CACHE: dict[pathlib.Path, bytes] = {}

OnSegmentStart = Callable[[], Coroutine[Any, Any, None]]
SegmentLoader = Callable[[], bytes | None]
SegmentSource = pathlib.Path | bytes | SegmentLoader
BusDiscrepancyNotifier = Callable[[str], None]


def _init_gap_silence() -> None:
    global _GAP_SILENCE
    _GAP_SILENCE = SILENCE_CHUNK * _GAP_CHUNKS


_init_gap_silence()


def _decode_asset_pcm(path: pathlib.Path) -> bytes | None:
    cached = _ASSET_PCM_CACHE.get(path)
    if cached is not None:
        return cached
    if not path.exists():
        return None
    try:
        proc = subprocess.run(
            [
                'ffmpeg', '-loglevel', 'error', '-i', str(path),
                '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', str(CHANNELS),
                'pipe:1',
            ],
            capture_output=True,
            check=True,
        )
    except Exception as exc:
        log.error('Failed to decode audio asset %s: %s', path.name, exc)
        return None
    if not proc.stdout:
        return None
    _ASSET_PCM_CACHE[path] = proc.stdout
    return proc.stdout


def _fallback_silence_pcm(seconds: float = _BUS_SILENCE_SECONDS) -> bytes:
    chunk_count = max(1, round(seconds / CHUNK_DURATION))
    return SILENCE_CHUNK * chunk_count


def _bus_failover_sequence_pcm() -> bytes | None:
    alarm_pcm = _decode_asset_pcm(_BUS_ALARM_PATH)
    if not alarm_pcm:
        log.error('Audio bus failover requested but %s is unavailable', _BUS_ALARM_PATH)
        return None
    silence_pcm = _decode_asset_pcm(_BUS_SILENCE_PATH) or _fallback_silence_pcm()
    return alarm_pcm + (silence_pcm * _BUS_SILENCE_LOOPS)


class OutputSink(Protocol):
    async def write(self, pcm: bytes) -> None: ...
    async def close(self) -> None: ...


class _BufferedSink:
    __slots__ = (
        '_sink', '_name', '_queue', '_task', '_closed',
        '_drop_oldest', '_dropped_chunks', '_clocked',
        '_prefill_chunks', '_fill_silence', '_discrepancy_notifier',
        '_underflow_active',
    )

    def __init__(
        self,
        sink: OutputSink,
        name: str,
        queue_limit: int,
        drop_oldest: bool,
        clocked: bool,
        prefill_chunks: int,
        fill_silence: bool,
        discrepancy_notifier: BusDiscrepancyNotifier | None,
    ) -> None:
        self._sink = sink
        self._name = name
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=queue_limit)
        self._task: asyncio.Task | None = None
        self._closed = False
        self._drop_oldest = drop_oldest
        self._dropped_chunks = 0
        self._clocked = clocked
        self._prefill_chunks = max(0, min(queue_limit, prefill_chunks))
        self._fill_silence = fill_silence
        self._discrepancy_notifier = discrepancy_notifier
        self._underflow_active = False

    def _report_discrepancy(self, reason: str) -> None:
        if self._discrepancy_notifier is not None:
            self._discrepancy_notifier(reason)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(
                self._run(), name=f'sink:{self._name}',
            )

    async def _run(self) -> None:
        if self._clocked:
            await self._run_clocked()
            return

        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            try:
                await self._sink.write(chunk)
            except Exception as exc:
                self._report_discrepancy(f'sink write failure: {exc}')
                log.error('Sink %s write failed: %s', self._name, exc)

    async def _run_clocked(self) -> None:
        chunk_ns = int(CHUNK_DURATION * 1_000_000_000)
        next_ns = time.monotonic_ns()
        prefilled = self._prefill_chunks == 0

        while True:
            if not prefilled:
                if self._closed and self._queue.empty():
                    return
                if self._queue.qsize() < self._prefill_chunks:
                    await asyncio.sleep(CHUNK_DURATION / 2)
                    continue
                prefilled = True
                next_ns = time.monotonic_ns()

            try:
                chunk = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                if not self._fill_silence:
                    await asyncio.sleep(CHUNK_DURATION / 2)
                    continue
                if not self._underflow_active:
                    self._underflow_active = True
                    self._report_discrepancy('mailbox underflow')
                chunk = SILENCE_CHUNK
            else:
                self._underflow_active = False

            if chunk is None:
                return

            try:
                await self._sink.write(chunk)
            except Exception as exc:
                self._report_discrepancy(f'sink write failure: {exc}')
                log.error('Sink %s write failed: %s', self._name, exc)

            next_ns += chunk_ns
            sleep_ns = next_ns - time.monotonic_ns()
            if sleep_ns > 0:
                await asyncio.sleep(sleep_ns / 1_000_000_000)
            elif sleep_ns < -chunk_ns * 8:
                next_ns = time.monotonic_ns()

    def _log_drop(self) -> None:
        self._dropped_chunks += 1
        self._report_discrepancy('mailbox overflow')
        if self._dropped_chunks in {1, 8, 32} or self._dropped_chunks % 128 == 0:
            log.warning(
                'Sink %s mailbox overflow - dropped %d chunk(s)',
                self._name,
                self._dropped_chunks,
            )

    async def reset(self) -> None:
        self._dropped_chunks = 0
        self._underflow_active = False
        while True:
            try:
                queued = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if queued is None:
                try:
                    self._queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass
                break
        resetter = getattr(self._sink, 'reset', None)
        if callable(resetter):
            await resetter()

    async def write(self, pcm: bytes) -> None:
        if self._closed or not pcm:
            return
        if self._task is None:
            self.start()
        try:
            self._queue.put_nowait(pcm)
            return
        except asyncio.QueueFull:
            pass
        if not self._drop_oldest:
            self._log_drop()
            return
        try:
            self._queue.get_nowait()
        except asyncio.QueueEmpty:
            self._log_drop()
            return
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            self._log_drop()
            return
        self._log_drop()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            while True:
                try:
                    self._queue.put_nowait(None)
                    break
                except asyncio.QueueFull:
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            await self._task
        await self._sink.close()


class AudioPipeline:
    __slots__ = (
        '_segment_queue', '_alert_queue', '_alert_active', '_alert_done',
        '_segment_consumed', '_shutdown', '_sinks', '_playout_task',
        '_bus_discrepancy_queue', '_bus_discrepancy_pending',
    )

    def __init__(self) -> None:
        self._segment_queue: asyncio.Queue[tuple[SegmentSource, OnSegmentStart | None]] = asyncio.Queue(
            maxsize=_SEGMENT_QUEUE_LIMIT,
        )
        self._alert_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._alert_active = asyncio.Event()
        self._alert_done = asyncio.Event()
        self._alert_done.set()
        self._segment_consumed = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._sinks: list[tuple[OutputSink, str]] = []
        self._playout_task: asyncio.Task | None = None
        self._bus_discrepancy_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._bus_discrepancy_pending = asyncio.Event()

    def _report_bus_discrepancy(self, sink_name: str, reason: str) -> None:
        if self._shutdown.is_set() or self._bus_discrepancy_pending.is_set():
            return
        self._bus_discrepancy_pending.set()
        self._bus_discrepancy_queue.put_nowait((sink_name, reason))
        log.error('Audio bus discrepancy detected on %s: %s', sink_name, reason)

    def attach_sink(self, sink: OutputSink, name: str = '') -> None:
        queue_limit = int(getattr(sink, 'bus_queue_limit', 0) or 0)
        discrepancy_notifier: BusDiscrepancyNotifier | None = None
        if bool(getattr(sink, 'bus_guard_enabled', False)):
            discrepancy_notifier = lambda reason, sink_name=(name or sink.__class__.__name__): self._report_bus_discrepancy(sink_name, reason)
        if queue_limit > 0:
            sink = _BufferedSink(
                sink,
                name,
                queue_limit,
                bool(getattr(sink, 'bus_drop_oldest', True)),
                bool(getattr(sink, 'bus_clocked', False)),
                int(getattr(sink, 'bus_prefill_chunks', 0) or 0),
                bool(getattr(sink, 'bus_fill_silence', False)),
                discrepancy_notifier,
            )
        self._sinks.append((sink, name))

    @property
    def alert_active(self) -> bool:
        return self._alert_active.is_set()

    @property
    def segments_queued(self) -> int:
        return self._segment_queue.qsize()

    @property
    def segment_consumed_event(self) -> asyncio.Event:
        return self._segment_consumed

    async def enqueue_segment(
        self, source: SegmentSource, on_start: OnSegmentStart | None = None,
    ) -> None:
        await self._segment_queue.put((source, on_start))

    def enqueue_alert(self, pcm: bytes) -> None:
        self._alert_done.clear()
        self._alert_queue.put_nowait(pcm)

    async def wait_alert_done(self) -> None:
        await self._alert_done.wait()

    def start(self) -> None:
        for sink, _ in self._sinks:
            starter = getattr(sink, 'start', None)
            if callable(starter):
                starter()
        self._playout_task = asyncio.get_running_loop().create_task(
            self._playout_loop(), name='playout',
        )

    async def stop(self) -> None:
        self._shutdown.set()
        if self._playout_task:
            self._playout_task.cancel()
            try:
                await self._playout_task
            except asyncio.CancelledError:
                pass
        for sink, name in self._sinks:
            try:
                await sink.close()
            except Exception as exc:
                log.error('Sink close failed (%s): %s', name, exc)

    async def _write_sinks(self, chunk: bytes) -> None:
        if not self._sinks:
            return
        if len(self._sinks) == 1:
            sink, name = self._sinks[0]
            try:
                await sink.write(chunk)
            except Exception as exc:
                log.error('Sink %s write failed: %s', name, exc)
            return
        results = await asyncio.gather(
            *(sink.write(chunk) for sink, _ in self._sinks),
            return_exceptions=True,
        )
        for (_, name), result in zip(self._sinks, results):
            if isinstance(result, Exception):
                log.error('Sink %s write failed: %s', name, result)

    async def _stream_pcm(self, data: bytes, interruptible: bool = True) -> bool:
        chunk_ns = int(CHUNK_DURATION * 1_000_000_000)
        next_ns = time.monotonic_ns()
        offset = 0
        while offset < len(data) and not self._shutdown.is_set():
            if interruptible and (not self._alert_queue.empty() or self._bus_discrepancy_pending.is_set()):
                return False
            end = offset + CHUNK_BYTES
            chunk = data[offset:end]
            if len(chunk) < CHUNK_BYTES:
                chunk += bytes(CHUNK_BYTES - len(chunk))
            next_ns += chunk_ns
            await self._write_sinks(chunk)
            offset = end
            sleep_ns = next_ns - time.monotonic_ns()
            if sleep_ns > 0:
                await asyncio.sleep(sleep_ns / 1_000_000_000)
            elif sleep_ns < -chunk_ns * 8:
                next_ns = time.monotonic_ns()
        return True

    async def _clear_segment_queue(self) -> None:
        while True:
            try:
                segment_source, _ = self._segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if isinstance(segment_source, pathlib.Path):
                try:
                    segment_source.unlink(missing_ok=True)
                except Exception:
                    pass

    async def _reset_audio_bus(self) -> None:
        for sink, _ in self._sinks:
            resetter = getattr(sink, 'reset', None)
            if callable(resetter):
                try:
                    await resetter()
                except Exception as exc:
                    log.error('Audio bus reset failed: %s', exc)

    async def _handle_bus_discrepancy(self) -> None:
        reasons: list[str] = []
        while True:
            try:
                sink_name, reason = self._bus_discrepancy_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            reasons.append(f'{sink_name}: {reason}')

        self._bus_discrepancy_pending.clear()
        if reasons:
            log.error('Audio bus failover sequence engaged: %s', '; '.join(reasons))
        else:
            log.error('Audio bus failover sequence engaged')

        self._alert_active.set()
        try:
            failover_pcm = _bus_failover_sequence_pcm()
            if failover_pcm:
                await self._stream_pcm(failover_pcm, interruptible=False)
            await self._clear_segment_queue()
            await self._reset_audio_bus()
        finally:
            self._alert_active.clear()
            self._alert_done.set()

    async def _playout_loop(self) -> None:
        chunk_ns = int(CHUNK_DURATION * 1_000_000_000)
        next_silence_ns = time.monotonic_ns()
        try:
            while not self._shutdown.is_set():
                if self._bus_discrepancy_pending.is_set():
                    await self._handle_bus_discrepancy()
                    next_silence_ns = time.monotonic_ns()
                    continue

                try:
                    alert_pcm = self._alert_queue.get_nowait()
                    self._alert_active.set()
                    try:
                        await self._stream_pcm(alert_pcm, interruptible=False)
                    finally:
                        self._alert_active.clear()
                        self._alert_done.set()
                    next_silence_ns = time.monotonic_ns()
                    continue
                except asyncio.QueueEmpty:
                    pass

                try:
                    segment_source, on_start = self._segment_queue.get_nowait()
                    self._segment_consumed.set()
                    self._segment_consumed.clear()
                except asyncio.QueueEmpty:
                    await self._write_sinks(SILENCE_CHUNK)
                    next_silence_ns += chunk_ns
                    sleep_ns = next_silence_ns - time.monotonic_ns()
                    if sleep_ns > 0:
                        await asyncio.sleep(sleep_ns / 1_000_000_000)
                    elif sleep_ns < -chunk_ns * 8:
                        next_silence_ns = time.monotonic_ns()
                    continue

                if on_start:
                    try:
                        await on_start()
                    except Exception:
                        pass

                if isinstance(segment_source, bytes):
                    data = segment_source
                elif callable(segment_source):
                    try:
                        data = segment_source()
                    except Exception as exc:
                        log.error('Segment render failed: %s', exc)
                        continue
                    if not data:
                        continue
                elif isinstance(segment_source, pathlib.Path):
                    try:
                        data = await asyncio.to_thread(segment_source.read_bytes)
                    except Exception as exc:
                        log.error('Segment read failed %s: %s', segment_source.name, exc)
                        continue
                    finally:
                        try:
                            segment_source.unlink(missing_ok=True)
                        except Exception:
                            pass
                else:
                    log.error('Unsupported segment source type: %s', type(segment_source).__name__)
                    continue

                completed = await self._stream_pcm(data)
                if completed:
                    await self._stream_pcm(_GAP_SILENCE)
                next_silence_ns = time.monotonic_ns()
        except asyncio.CancelledError:
            return

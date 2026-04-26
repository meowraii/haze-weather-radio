from __future__ import annotations

import asyncio
import logging
import pathlib
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

OnSegmentStart = Callable[[], Coroutine[Any, Any, None]]
SegmentLoader = Callable[[], bytes | None]
SegmentSource = pathlib.Path | bytes | SegmentLoader


def _gap_silence(duration_s: float) -> bytes:
    if duration_s <= 0.0:
        return b''
    chunks = max(1, round(duration_s / CHUNK_DURATION))
    return SILENCE_CHUNK * chunks


class OutputSink(Protocol):
    async def write(self, pcm: bytes) -> None: ...
    async def close(self) -> None: ...


class _BufferedSink:
    __slots__ = (
        '_sink', '_name', '_queue', '_task', '_closed',
        '_drop_oldest', '_dropped_chunks', '_clocked',
        '_prefill_chunks', '_fill_silence',
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
                chunk = SILENCE_CHUNK

            if chunk is None:
                return

            try:
                await self._sink.write(chunk)
            except Exception as exc:
                log.error('Sink %s write failed: %s', self._name, exc)

            next_ns += chunk_ns
            sleep_ns = next_ns - time.monotonic_ns()
            if sleep_ns > 0:
                await asyncio.sleep(sleep_ns / 1_000_000_000)
            elif sleep_ns < -chunk_ns * 8:
                next_ns = time.monotonic_ns()

    def _log_drop(self) -> None:
        self._dropped_chunks += 1
        if self._dropped_chunks in {1, 8, 32} or self._dropped_chunks % 128 == 0:
            log.warning(
                'Sink %s mailbox overflow - dropped %d chunk(s)',
                self._name,
                self._dropped_chunks,
            )

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
    )

    def __init__(self) -> None:
        self._segment_queue: asyncio.Queue[tuple[SegmentSource, OnSegmentStart | None, float]] = asyncio.Queue(
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

    def attach_sink(self, sink: OutputSink, name: str = '') -> None:
        queue_limit = int(getattr(sink, 'bus_queue_limit', 0) or 0)
        if queue_limit > 0:
            sink = _BufferedSink(
                sink,
                name,
                queue_limit,
                bool(getattr(sink, 'bus_drop_oldest', True)),
                bool(getattr(sink, 'bus_clocked', False)),
                int(getattr(sink, 'bus_prefill_chunks', 0) or 0),
                bool(getattr(sink, 'bus_fill_silence', False)),
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
        gap_after_s: float = 0.0,
    ) -> None:
        await self._segment_queue.put((source, on_start, gap_after_s))

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
            if interruptible and not self._alert_queue.empty():
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

    async def _playout_loop(self) -> None:
        chunk_ns = int(CHUNK_DURATION * 1_000_000_000)
        next_silence_ns = time.monotonic_ns()
        try:
            while not self._shutdown.is_set():
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
                    segment_source, on_start, gap_after_s = self._segment_queue.get_nowait()
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
                if completed and gap_after_s > 0.0:
                    await self._stream_pcm(_gap_silence(gap_after_s))
                next_silence_ns = time.monotonic_ns()
        except asyncio.CancelledError:
            return

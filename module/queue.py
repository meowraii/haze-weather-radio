# queue.py
# 

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Protocol
from . import load_config

config = load_config()

CHUNK_SAMPLES = 2048
SAMPLE_RATE = config.get('playout', {}).get('sample_rate', 16000)
CHANNELS = config.get('playout', {}).get('channels', 1)
BYTES_PER_SAMPLE = 2

class PlaybackInterrupted(Exception):
    pass


class OutputSink(Protocol):
    async def write(self, pcm: bytes) -> None: ...
    async def close(self) -> None: ...


@dataclass
class AudioBus:
    _sinks: list[asyncio.Queue[bytes | None]] = field(default_factory=list)

    def subscribe(self) -> asyncio.Queue[bytes | None]:
        q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=512)
        self._sinks.append(q)
        return q

    async def write(self, pcm: bytes, interrupt: asyncio.Event | None = None) -> None:
        for i in range(0, len(pcm), CHUNK_SAMPLES * BYTES_PER_SAMPLE):
            if interrupt is not None and interrupt.is_set():
                raise PlaybackInterrupted
            chunk = pcm[i:i + CHUNK_SAMPLES * BYTES_PER_SAMPLE]
            await asyncio.gather(*(q.put(chunk) for q in self._sinks))

    def flush(self) -> None:
        for q in self._sinks:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break

    async def close(self) -> None:
        for sink_q in self._sinks:
            await sink_q.put(None)


async def sink_runner(
    sink: OutputSink,
    q: asyncio.Queue[bytes | None],
    shutdown: asyncio.Event,
) -> None:
    try:
        while not shutdown.is_set():
            chunk = await q.get()
            if chunk is None:
                break
            try:
                await sink.write(chunk)
            except Exception:
                break
    finally:
        try:
            await sink.close()
        except Exception:
            pass
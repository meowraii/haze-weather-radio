from __future__ import annotations

import io
import logging
import pathlib
import threading
from typing import Any, Iterable

import av
import numpy as np

log = logging.getLogger(__name__)

BYTES_PER_SAMPLE = 2


def layout_name(channels: int) -> str:
    return 'stereo' if int(channels) == 2 else 'mono'


def pcm_frame_bytes(sample_rate: int, channels: int, samples: int) -> int:
    return int(samples) * int(channels) * BYTES_PER_SAMPLE


def _trim_pcm(pcm_s16le: bytes, channels: int) -> bytes:
    frame_width = max(1, int(channels)) * BYTES_PER_SAMPLE
    usable = len(pcm_s16le) - (len(pcm_s16le) % frame_width)
    return pcm_s16le[:usable]


def audio_frame_to_pcm_s16le(frame: av.AudioFrame) -> bytes:
    array = frame.to_ndarray()
    if array.dtype != np.int16:
        array = array.astype(np.int16, copy=False)
    return np.ascontiguousarray(array.reshape(-1)).tobytes()


def iter_pcm_audio_frames(
    pcm_s16le: bytes,
    *,
    sample_rate: int,
    channels: int,
    frame_samples: int = 2048,
) -> Iterable[av.AudioFrame]:
    pcm_s16le = _trim_pcm(pcm_s16le, channels)
    if not pcm_s16le:
        return

    chunk_bytes = pcm_frame_bytes(sample_rate, channels, frame_samples)
    layout = layout_name(channels)
    for offset in range(0, len(pcm_s16le), chunk_bytes):
        chunk = pcm_s16le[offset:offset + chunk_bytes]
        if not chunk:
            continue
        samples = np.frombuffer(chunk, dtype=np.int16)
        if samples.size == 0:
            continue
        packed = np.ascontiguousarray(samples.reshape(1, -1))
        frame = av.AudioFrame.from_ndarray(packed, format='s16', layout=layout)
        frame.sample_rate = int(sample_rate)
        yield frame


def normalize_pcm_s16le(
    pcm_s16le: bytes,
    *,
    channels: int,
    target_peak: float = 0.92,
    target_rms: float = 0.18,
) -> bytes | None:
    pcm_s16le = _trim_pcm(pcm_s16le, channels)
    if not pcm_s16le:
        return None

    samples = np.frombuffer(pcm_s16le, dtype=np.int16)
    if samples.size == 0:
        return None

    work = samples.astype(np.float32)
    work -= float(np.mean(work))
    peak = float(np.max(np.abs(work)))
    if peak <= 1.0:
        return pcm_s16le

    rms = float(np.sqrt(np.mean(np.square(work / 32768.0))))
    gain_for_peak = (32767.0 * target_peak) / peak
    gain_for_rms = (32768.0 * target_rms) / max(rms * 32768.0, 1.0)
    gain = max(0.05, min(gain_for_peak, gain_for_rms, 8.0))
    work *= gain

    limiter = 32767.0 * target_peak
    over = np.abs(work) > limiter
    if np.any(over):
        work[over] = np.sign(work[over]) * (
            limiter + np.tanh((np.abs(work[over]) - limiter) / 2048.0) * 2048.0
        )

    return np.clip(work, -32768, 32767).astype(np.int16).tobytes()


def decode_audio_file_pcm(
    file_path: pathlib.Path,
    *,
    sample_rate: int,
    channels: int,
    normalize: bool = False,
) -> bytes | None:
    try:
        with av.open(str(file_path), mode='r') as container:
            return _decode_container_pcm(
                container,
                sample_rate=sample_rate,
                channels=channels,
                normalize=normalize,
            )
    except Exception as exc:
        log.error('Failed to decode %s with PyAV: %s', file_path, exc)
        return None


def decode_audio_bytes_pcm(
    audio_bytes: bytes,
    *,
    sample_rate: int,
    channels: int,
    normalize: bool = False,
) -> bytes | None:
    if not audio_bytes:
        return None
    try:
        with av.open(io.BytesIO(audio_bytes), mode='r') as container:
            return _decode_container_pcm(
                container,
                sample_rate=sample_rate,
                channels=channels,
                normalize=normalize,
            )
    except Exception as exc:
        log.error('Failed to decode in-memory audio with PyAV: %s', exc)
        return None


def _decode_container_pcm(
    container: av.container.InputContainer,
    *,
    sample_rate: int,
    channels: int,
    normalize: bool,
) -> bytes | None:
    audio_stream = next((stream for stream in container.streams if stream.type == 'audio'), None)
    if audio_stream is None:
        return None

    resampler = av.AudioResampler(
        format='s16',
        layout=layout_name(channels),
        rate=int(sample_rate),
    )
    chunks: list[bytes] = []
    for frame in container.decode(audio_stream):
        for converted in resampler.resample(frame):
            chunks.append(audio_frame_to_pcm_s16le(converted))
    try:
        for converted in resampler.resample(None):
            chunks.append(audio_frame_to_pcm_s16le(converted))
    except Exception:
        pass

    pcm = b''.join(chunks)
    if not pcm:
        return None
    if normalize:
        return normalize_pcm_s16le(pcm, channels=channels) or pcm
    return pcm


def write_pcm_wav(
    path: pathlib.Path,
    pcm_s16le: bytes,
    *,
    sample_rate: int,
    channels: int,
) -> None:
    write_encoded_audio(
        path,
        pcm_s16le,
        sample_rate=sample_rate,
        channels=channels,
        codec='pcm_s16le',
        container_format='wav',
    )


def write_encoded_audio(
    path: pathlib.Path,
    pcm_s16le: bytes,
    *,
    sample_rate: int,
    channels: int,
    codec: str,
    container_format: str | None = None,
    bit_rate: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode='w', format=container_format) as container:
        stream = container.add_stream(codec, rate=int(sample_rate))
        stream.layout = layout_name(channels)
        if bit_rate is not None:
            stream.bit_rate = int(bit_rate)
        if metadata:
            container.metadata.update({str(k): str(v) for k, v in metadata.items() if v is not None})
            stream.metadata.update({str(k): str(v) for k, v in metadata.items() if v is not None})
        for frame in iter_pcm_audio_frames(
            pcm_s16le,
            sample_rate=sample_rate,
            channels=channels,
        ):
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def transcode_audio_file_to_wav(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    sample_rate: int,
    channels: int,
    normalize: bool = False,
) -> bool:
    pcm = decode_audio_file_pcm(src, sample_rate=sample_rate, channels=channels, normalize=normalize)
    if not pcm:
        return False
    try:
        write_pcm_wav(dst, pcm, sample_rate=sample_rate, channels=channels)
        return True
    except Exception as exc:
        log.error('Failed to write %s with PyAV: %s', dst, exc)
        return False


class PyAVStreamWriter:
    def __init__(
        self,
        *,
        url: str,
        container_format: str,
        codec: str,
        sample_rate: int,
        channels: int,
        bit_rate: int | None = None,
        metadata: dict[str, Any] | None = None,
        options: dict[str, str] | None = None,
        frame_samples: int = 2048,
    ) -> None:
        self._url = url
        self._container_format = container_format
        self._codec = codec
        self._sample_rate = int(sample_rate)
        self._channels = int(channels)
        self._bit_rate = bit_rate
        self._metadata = metadata or {}
        self._options = options or {}
        self._frame_samples = max(128, int(frame_samples))
        self._lock = threading.Lock()
        self._container: av.container.OutputContainer | None = None
        self._stream: av.AudioStream | None = None
        self._closed = False

    def open(self) -> None:
        with self._lock:
            if self._container is not None:
                return
            self._open_locked()

    def _open_locked(self) -> None:
        if '://' not in self._url:
            pathlib.Path(self._url).parent.mkdir(parents=True, exist_ok=True)
        container = av.open(
            self._url,
            mode='w',
            format=self._container_format,
            options=self._options,
        )
        stream = container.add_stream(self._codec, rate=self._sample_rate)
        stream.layout = layout_name(self._channels)
        if self._bit_rate is not None:
            stream.bit_rate = int(self._bit_rate)
        if self._metadata:
            clean = {str(k): str(v) for k, v in self._metadata.items() if v is not None and str(v).strip()}
            container.metadata.update(clean)
            stream.metadata.update(clean)
        self._container = container
        self._stream = stream

    def restart(self) -> None:
        with self._lock:
            self._close_locked()
            if not self._closed:
                self._open_locked()

    def write(self, pcm_s16le: bytes) -> None:
        if not pcm_s16le or self._closed:
            return
        with self._lock:
            if self._container is None or self._stream is None:
                self._open_locked()
            assert self._container is not None
            assert self._stream is not None
            for frame in iter_pcm_audio_frames(
                pcm_s16le,
                sample_rate=self._sample_rate,
                channels=self._channels,
                frame_samples=self._frame_samples,
            ):
                for packet in self._stream.encode(frame):
                    self._container.mux(packet)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._close_locked()

    def _close_locked(self) -> None:
        container = self._container
        stream = self._stream
        self._container = None
        self._stream = None
        if container is None:
            return
        try:
            if stream is not None:
                for packet in stream.encode(None):
                    container.mux(packet)
        finally:
            container.close()

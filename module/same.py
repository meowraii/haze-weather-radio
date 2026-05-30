"""
Specific Area Message Encoding (SAME) implementation.

SAME is an AFSK digital mode used by the Emergency Alert System (EAS) and
NOAA Weather Radio to deliver machine-readable alert headers.

Each bit lasts 1920 µs (bit rate ≈ 520.83 bps).
  mark  = 4 cycles of 2083⅓ Hz  (6250/3 Hz)
  space = 3 cycles of 1562.5 Hz  (3125/2 Hz)

Bytes are 8-bit ASCII (MSB=0), transmitted LSB-first.
Preamble is 16 × 0xAB.

Activation sequence:
  1. Preamble + Header ×3  (1 s silence between bursts)
  2. Attention tone (8–25 s)
  3. Voice / audio message
  4. Preamble + EOM ×3     (1 s silence between bursts)

Header format:  ZCZC-ORG-EEE-PSSCCC+TTTT-JJJHHMM-LLLLLLLL-
EOM:            NNNN
"""

from __future__ import annotations

import datetime
import io
import logging
import logging.handlers
import pathlib
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Final, Literal, Optional

import numpy as np
import soundfile as sf
from scipy import signal as sp_signal

_SAMPLE_RATE: Final[int] = 16_000

_MARK_FREQ: Final[float] = 6250 / 3
_SPACE_FREQ: Final[float] = 3125 / 2
_BIT_DURATION: Final[float] = 6 / 3125

_PREAMBLE_BYTE: Final[int] = 0xAB
_PREAMBLE_LEN: Final[int] = 16

_PILOT_PREFIX_S: Final[float] = 0.015
_PILOT_SUFFIX_S: Final[float] = 0.030
_PILOT_PREFIX_FREQ_HZ: Final[float] = 1575.0
_PILOT_SUFFIX_FREQ_HZ: Final[float] = 2088.0
_BURST_LEAD_S: Final[float] = 0.100
_SEGMENT_FADE_S: Final[float] = 0.003

_AFSK_HIGHPASS_HZ: Final[float] = 90.0
_AFSK_LOWPASS_HZ: Final[float] = 4000.0
_SEQUENCE_AMPLITUDE: Final[float] = 0.60

_INTER_BURST_S: Final[float] = 1.0
_PRE_ATTN_S: Final[float] = 1.0
_PRE_VOICE_S: Final[float] = 1.0
_EOM_LEAD_S: Final[float] = 1.0
_EOM_TAIL_S: Final[float] = 0.8

_ATTN_DEFAULT_S: Final[float] = 8.0
_NPAS_SATURATION_DRIVE: Final[float] = 1.85
_NPAS_SATURATION_BLEND: Final[float] = 0.32
SAME_SAMPLE_RATE: Final[int] = _SAMPLE_RATE

ToneType = Literal["WXR", "EAS", "NPAS", "EGG_TIMER", "QUEBEC"]

log = logging.getLogger(__name__)
_audit_log = logging.getLogger("same.audit")
_audit_log.propagate = False


@dataclass(frozen=True)
class SameConfig:
    sample_rate: int = _SAMPLE_RATE
    pilot_prefix_s: float = _PILOT_PREFIX_S
    pilot_prefix_freq_hz: float = _PILOT_PREFIX_FREQ_HZ
    pilot_suffix_s: float = _PILOT_SUFFIX_S
    pilot_suffix_freq_hz: float = _PILOT_SUFFIX_FREQ_HZ
    afsk_highpass_hz: float = _AFSK_HIGHPASS_HZ
    afsk_lowpass_hz: float = _AFSK_LOWPASS_HZ
    segment_fade_s: float = _SEGMENT_FADE_S


def _ensure_audit_handler() -> None:
    if _audit_log.handlers:
        return
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "same.log", maxBytes=5_242_880, backupCount=3
    )
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    _audit_log.setLevel(logging.INFO)
    _audit_log.addHandler(handler)


@dataclass(frozen=True)
class SAMEHeader:
    originator: str
    event: str
    locations: tuple[str, ...]
    duration: str
    callsign: str
    issue_time: Optional[str] = None

    @property
    def encoded(self) -> str:
        loc_str = "-".join(self.locations)
        issue = self.issue_time or get_issue_time()
        cs = self.callsign.replace("-", "/").ljust(8)[:8]
        return f"ZCZC-{self.originator}-{self.event}-{loc_str}+{self.duration}-{issue}-{cs}-"


def get_issue_time() -> str:
    now = time.gmtime()
    return f"{now.tm_yday:03d}{now.tm_hour:02d}{now.tm_min:02d}"


def normalise_time_code(time_code: str) -> str:
    if "T" in time_code:
        dt = datetime.datetime.fromisoformat(time_code)
        utc = dt.astimezone(datetime.timezone.utc)
        return f"{utc.timetuple().tm_yday:03d}{utc.hour:02d}{utc.minute:02d}"
    try:
        t = time.gmtime(float(time_code))
        return f"{t.tm_yday:03d}{t.tm_hour:02d}{t.tm_min:02d}"
    except (ValueError, OverflowError):
        pass
    if len(time_code) == 7 and time_code.isdigit():
        return time_code
    raise ValueError(f"Unsupported time_code format: {time_code!r}")


@lru_cache(maxsize=8)
def _bit_waves(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(sample_rate * _BIT_DURATION))
    t = np.linspace(0, _BIT_DURATION, n, endpoint=False)
    return (
        np.sin(2 * np.pi * _MARK_FREQ * t).astype(np.float32),
        np.sin(2 * np.pi * _SPACE_FREQ * t).astype(np.float32),
    )


def _silence(duration_s: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


def _pilot(duration_s: float, sample_rate: int, freq: float) -> np.ndarray:
    n = int(round(sample_rate * duration_s))
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    t = np.linspace(0, duration_s, n, endpoint=False)
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
    tone = _apply_lowpass(tone, 1500.0, sample_rate)
    return tone


def _apply_fade_in(signal: np.ndarray, fade_duration_s: float, sample_rate: int) -> np.ndarray:
    fade_samples = min(int(sample_rate * fade_duration_s), len(signal))
    if fade_samples == 0:
        return signal
    envelope = np.linspace(0, 1, fade_samples, dtype=np.float32)
    signal = signal.copy()
    signal[:fade_samples] *= envelope
    return signal


def _apply_fade_out(signal: np.ndarray, fade_duration_s: float, sample_rate: int) -> np.ndarray:
    fade_samples = min(int(sample_rate * fade_duration_s), len(signal))
    if fade_samples == 0:
        return signal
    envelope = np.linspace(1, 0, fade_samples, dtype=np.float32)
    signal = signal.copy()
    signal[-fade_samples:] *= envelope
    return signal


def _apply_soft_saturation(signal: np.ndarray, drive: float, blend: float) -> np.ndarray:
    if signal.size == 0 or drive <= 1.0 or blend <= 0.0:
        return signal

    peak = float(np.max(np.abs(signal)))
    if peak <= 0.0:
        return signal

    clean = signal / peak
    saturated = np.tanh(clean * drive)
    mixed = ((1.0 - blend) * clean) + (blend * saturated)

    mixed_peak = float(np.max(np.abs(mixed)))
    if mixed_peak > 0.0:
        mixed = mixed / mixed_peak

    clean64 = clean.astype(np.float64, copy=False)
    mixed64 = mixed.astype(np.float64, copy=False)
    clean_rms = float(np.sqrt(np.mean(clean64 * clean64)))
    mixed_rms = float(np.sqrt(np.mean(mixed64 * mixed64)))
    if clean_rms > 0.0 and mixed_rms > clean_rms:
        mixed = mixed * (clean_rms / mixed_rms)

    return (mixed * peak).astype(np.float32)


def _match_reference_level(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if signal.size == 0 or reference.size == 0:
        return signal

    reference_peak = float(np.max(np.abs(reference)))
    reference64 = reference.astype(np.float64, copy=False)
    signal64 = signal.astype(np.float64, copy=False)
    reference_rms = float(np.sqrt(np.mean(reference64 * reference64)))
    signal_rms = float(np.sqrt(np.mean(signal64 * signal64)))

    adjusted = signal.astype(np.float32, copy=True)
    if reference_rms > 0.0 and signal_rms > 0.0:
        adjusted *= reference_rms / signal_rms

    adjusted_peak = float(np.max(np.abs(adjusted)))
    if reference_peak > 0.0 and adjusted_peak > reference_peak:
        adjusted *= reference_peak / adjusted_peak

    return adjusted.astype(np.float32)


def _afsk_encode(data: bytes, sample_rate: int) -> np.ndarray:
    mark, space = _bit_waves(sample_rate)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    return np.stack([space, mark])[bits].ravel()


def _preamble(sample_rate: int, cfg: SameConfig) -> np.ndarray:
    pream = _afsk_encode(bytes([_PREAMBLE_BYTE] * _PREAMBLE_LEN), sample_rate)
    pream = _apply_fade_in(pream, cfg.segment_fade_s, sample_rate)
    return pream


def _tone_wxr(duration_s: float, sample_rate: int, cfg: SameConfig) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return np.sin(2 * np.pi * 1050.0 * t).astype(np.float32)


def _tone_quebec(duration_s: float, sample_rate: int, cfg: SameConfig) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return (
        (np.sin(2 * np.pi * 1050.0 * t) + np.sin(2 * np.pi * 650.0 * t)) / 2.0
    ).astype(np.float32)


def _tone_eas(duration_s: float, sample_rate: int, cfg: SameConfig) -> np.ndarray:
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    tone = (
        (np.sin(2 * np.pi * 853.0 * t) + np.sin(2 * np.pi * 960.0 * t)) / 2.0
    ).astype(np.float32)
    tone = _apply_fade_in(tone, 0.006, sample_rate)
    tone = _apply_fade_out(tone, 0.006, sample_rate)
    return tone


def _tone_npas(duration_s: float, sample_rate: int, cfg: SameConfig) -> np.ndarray:
    n = int(sample_rate * duration_s)
    half = sample_rate // 2
    freq_a: tuple[float, ...] = (932.33, 1046.50, 3135.96)
    freq_b: tuple[float, ...] = (440.00, 659.26, 3135.96)
    chunks: list[np.ndarray] = []
    for i in range(0, n, half):
        chunk_n = min(half, n - i)
        t = np.arange(chunk_n, dtype=np.float64) / sample_rate
        freqs = freq_a if (i // half) % 2 == 0 else freq_b
        chunk = sum(np.sin(2 * np.pi * f * t) for f in freqs).astype(np.float32)  # type: ignore[arg-type]
        chunk = _apply_fade_in(chunk, cfg.segment_fade_s, sample_rate)
        chunk = _apply_fade_out(chunk, cfg.segment_fade_s, sample_rate)
        chunks.append(chunk)
    tone = (np.concatenate(chunks) / 3.0).astype(np.float32)
    tone = _apply_highpass(tone, 900.0, sample_rate)
    tone = _apply_lowpass(tone, 2400.0, sample_rate)
    clean_tone = tone.copy()
    tone = _apply_soft_saturation(tone, _NPAS_SATURATION_DRIVE, _NPAS_SATURATION_BLEND)
    tone = _apply_highpass(tone, 900.0, sample_rate)
    tone = _apply_lowpass(tone, 2400.0, sample_rate)
    tone = _match_reference_level(tone, clean_tone)
    return tone


def _tone_egg_timer(duration_s: float, sample_rate: int, cfg: SameConfig) -> np.ndarray:
    n = int(sample_rate * duration_s)
    tone_len = int(round(sample_rate * 0.070))
    inter_len = int(round(sample_rate * 0.055))
    gap_len = int(round(sample_rate * 0.500))
    burst_period = tone_len + inter_len
    cycle_len = 4 * burst_period + gap_len
    fade_time = 0.006

    t = np.arange(n, dtype=np.float64) / sample_rate
    wave = (
        np.sin(2 * np.pi * 2055.0 * t)
#        + 0.25 * np.sin(2 * np.pi * 4110.0 * t)
#        + 0.10 * np.sin(2 * np.pi * 6165.0 * t)
#        + 0.05 * np.sin(2 * np.pi * 8220.0 * t)
    ).astype(np.float32) / 1.40

    pos = np.arange(n) % cycle_len
    active = (pos // burst_period < 4) & (pos % burst_period < tone_len)
    tone = np.where(active, wave, 0.0).astype(np.float32)

    for i in range(0, n, burst_period):
        for j in range(4):
            start = i + j * burst_period
            end = min(start + tone_len, n)
            if start < end:
                beep = tone[start:end].copy()
                beep = _apply_fade_in(beep, fade_time, sample_rate)
                beep = _apply_fade_out(beep, fade_time, sample_rate)
                tone[start:end] = beep

    return tone


ToneFn = Callable[[float, int, SameConfig], np.ndarray]

_TONE_DISPATCH: dict[str, ToneFn] = {
    "WXR": _tone_wxr,
    "EAS": _tone_eas,
    "NPAS": _tone_npas,
    "EGG_TIMER": _tone_egg_timer,
    "QUEBEC": _tone_quebec,
}


def build_header_burst(header: SAMEHeader, cfg: SameConfig = SameConfig()) -> np.ndarray:
    sample_rate = cfg.sample_rate
    preamble = _preamble(sample_rate, cfg)
    preamble = _apply_fade_in(preamble, cfg.segment_fade_s, sample_rate)
    afsk_data = _afsk_encode(header.encoded.encode("ascii"), sample_rate)
    afsk_data = _apply_fade_out(afsk_data, cfg.segment_fade_s, sample_rate)
    pilot_suffix = _pilot(cfg.pilot_suffix_s, sample_rate, cfg.pilot_suffix_freq_hz)
    pilot_suffix = _apply_fade_out(pilot_suffix, cfg.segment_fade_s, sample_rate)
    burst = np.concatenate([preamble, afsk_data, pilot_suffix])
    burst = _apply_highpass(burst, cfg.afsk_highpass_hz, sample_rate)
    burst = _apply_lowpass(burst, cfg.afsk_lowpass_hz, sample_rate)
    return burst


def build_eom_burst(cfg: SameConfig = SameConfig()) -> np.ndarray:
    sample_rate = cfg.sample_rate
    preamble = _preamble(sample_rate, cfg)
    preamble = _apply_fade_in(preamble, cfg.segment_fade_s, sample_rate)
    afsk_data = _afsk_encode(b"NNNN", sample_rate)
    afsk_data = _apply_fade_out(afsk_data, cfg.segment_fade_s, sample_rate)
    pilot_suffix = _pilot(cfg.pilot_suffix_s, sample_rate, cfg.pilot_suffix_freq_hz)
    pilot_suffix = _apply_fade_out(pilot_suffix, cfg.segment_fade_s, sample_rate)
    burst = np.concatenate([preamble, afsk_data, pilot_suffix])
    burst = _apply_highpass(burst, cfg.afsk_highpass_hz, sample_rate)
    burst = _apply_lowpass(burst, cfg.afsk_lowpass_hz, sample_rate)
    return burst


def _triple_burst(burst: np.ndarray, cfg: SameConfig) -> np.ndarray:
    sample_rate = cfg.sample_rate
    pilot = _pilot(cfg.pilot_prefix_s, sample_rate, cfg.pilot_prefix_freq_hz)
    pilot = _apply_fade_in(pilot, cfg.segment_fade_s, sample_rate)
    pilot = _apply_fade_out(pilot, cfg.segment_fade_s, sample_rate)
    gap = _silence(_INTER_BURST_S, sample_rate)
    return np.concatenate([pilot, burst, gap, burst, gap, burst])


def _load_audio(path: pathlib.Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = audio.astype(np.float32)
    if sr != target_sr:
        log.info("Resampling audio %d -> %d Hz", sr, target_sr)
        audio = resample(audio, sr, target_sr)
    return audio


def _apply_lowpass(signal: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    signal_len = int(np.asarray(signal).size)
    if signal_len == 0:
        return signal
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff >= 1.0:
        return signal
    coeffs = sp_signal.butter(5, normalized_cutoff, btype="low", output="ba")
    if not isinstance(coeffs, tuple) or len(coeffs) != 2:
        raise RuntimeError("Failed to design lowpass filter coefficients")
    b, a = coeffs
    a_len = int(np.asarray(a).size)
    b_len = int(np.asarray(b).size)
    if signal_len <= 3 * max(a_len, b_len):
        return signal
    return sp_signal.filtfilt(b, a, signal).astype(np.float32)


def _apply_highpass(signal: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    signal_len = int(np.asarray(signal).size)
    if signal_len == 0:
        return signal
    nyquist = sample_rate / 2
    normalized_cutoff = cutoff_hz / nyquist
    if normalized_cutoff <= 0:
        return signal
    coeffs = sp_signal.butter(5, normalized_cutoff, btype="high", output="ba")
    if not isinstance(coeffs, tuple) or len(coeffs) != 2:
        raise RuntimeError("Failed to design highpass filter coefficients")
    b, a = coeffs
    a_len = int(np.asarray(a).size)
    b_len = int(np.asarray(b).size)
    if signal_len <= 3 * max(a_len, b_len):
        return signal
    return sp_signal.filtfilt(b, a, signal).astype(np.float32)


def generate_same(
    header: Optional[SAMEHeader] = None,
    attn_duration_s: float = _ATTN_DEFAULT_S,
    tone_type: Optional[ToneType] = "WXR",
    audio_msg_path: Optional[pathlib.Path] = None,
    audio_msg_array: Optional[np.ndarray] = None,
    audio_label: Optional[str] = None,
) -> np.ndarray:
    _ensure_audit_handler()
    cfg = SameConfig()
    sample_rate = cfg.sample_rate

    pre: list[np.ndarray] = []
    voice: Optional[np.ndarray] = None

    if header is not None:
        log.info(
            "Generating SAME: %s tone=%s %.1fs @ %d Hz",
            header.encoded, tone_type, attn_duration_s, sample_rate,
        )
        pre.append(_silence(_BURST_LEAD_S, sample_rate))
        pre.append(_triple_burst(build_header_burst(header, cfg), cfg))

        if tone_type is not None:
            tone_fn = _TONE_DISPATCH.get(tone_type)
            if tone_fn is None:
                log.warning("Unknown tone_type %r — skipping", tone_type)
            else:
                pre.append(_silence(_PRE_ATTN_S, sample_rate))
                pre.append(tone_fn(attn_duration_s, sample_rate, cfg))

        if audio_msg_array is not None:
            pre.append(_silence(_PRE_VOICE_S, sample_rate))
            voice = audio_msg_array.astype(np.float32)
        elif audio_msg_path is not None:
            log.info("Loading audio: %s", audio_msg_path)
            pre.append(_silence(_PRE_VOICE_S, sample_rate))
            voice = _load_audio(audio_msg_path, sample_rate)

        encoded_label = header.encoded
    else:
        log.info("Generating EOM-only @ %d Hz", sample_rate)
        encoded_label = "EOM"

    eom_seq = np.concatenate([
        _silence(_EOM_LEAD_S, sample_rate),
        _triple_burst(build_eom_burst(cfg), cfg),
        _silence(_EOM_TAIL_S, sample_rate),
    ])

    segments: list[np.ndarray] = []
    if pre:
        segments.append(np.concatenate(pre) * _SEQUENCE_AMPLITUDE)
    if voice is not None:
        segments.append(voice)
    segments.append(eom_seq * _SEQUENCE_AMPLITUDE)

    out = np.concatenate(segments).astype(np.float32)
    total_s = len(out) / sample_rate

    log.info("SAME sequence done: %s (%.2fs)", encoded_label, total_s)
    _audit_log.info(
        "SAME_GENERATED | header=%s | tone=%s | attn=%.1fs | audio=%s | duration=%.2fs | sr=%d",
        encoded_label,
        tone_type if header is not None else "none",
        attn_duration_s,
        audio_label or (str(audio_msg_path) if audio_msg_path else ("embedded" if audio_msg_array is not None else "none")),
        total_s,
        sample_rate,
    )
    return out


def resample(signal: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return signal
    target_len = int(round(len(signal) * to_sr / from_sr))
    return np.interp(
        np.linspace(0, len(signal) - 1, target_len),
        np.arange(len(signal)),
        signal,
    ).astype(np.float32)


def to_pcm16(signal: np.ndarray) -> bytes:
    return (np.clip(signal, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def to_wav(signal: np.ndarray, sample_rate: int = _SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, signal, samplerate=sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a SAME test signal")
    parser.add_argument(
        "--att",
        choices=_TONE_DISPATCH.keys(),
        default="WXR",
        help="Attention tone type (default: WXR)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Generate one WAV per tone combination into ./debug_out/",
    )
    args = parser.parse_args()

    h = SAMEHeader(
        originator="WXR",
        event="NMN",
        locations=("065100",),
        duration="0015",
        callsign="EC/GC/CA",
    )
    print(f"Encoded: {h.encoded}")

    if args.debug:
        debug_dir = pathlib.Path("debug_out")
        debug_dir.mkdir(exist_ok=True)
        cfg = SameConfig()
        total = len(_TONE_DISPATCH)
        done = 0
        for tone_key in _TONE_DISPATCH:
            out_path = debug_dir / f"{tone_key}.wav"
            try:
                audio = generate_same(h, tone_type=tone_key)  # type: ignore[arg-type]
                out_path.write_bytes(to_wav(audio, cfg.sample_rate))
                done += 1
                print(f"[{done}/{total}] {out_path} ({len(audio) / cfg.sample_rate:.2f}s)")
            except Exception as exc:
                print(f"[{done}/{total}] FAILED {tone_key}: {exc}")
        print(f"Debug run complete — {done}/{total} files written to {debug_dir}/")
    else:
        cfg = SameConfig()
        full = generate_same(h, tone_type=args.att)
        out_path = pathlib.Path("same_test.wav")
        out_path.write_bytes(to_wav(full, cfg.sample_rate))
        print(f"Written: {out_path} ({len(full) / cfg.sample_rate:.2f}s)")
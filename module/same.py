"""
Specific Area Message Encoding (SAME) implementation.

SAME is an AFSK digital mode used by the Emergency Alert System (EAS) and
NOAA Weather Radio to deliver machine-readable alert headers.

Each bit lasts 1920 µs (bit rate = 520 5/6 bps).
  mark  = 4 cycles of 2083 1/3 Hz  (6250/3 Hz)
  space = 3 cycles of 1562.5 Hz    (3125/2 Hz)

Both frequencies were chosen so that exactly 4 mark cycles and exactly 3
space cycles fit into one bit period.  The bit duration is therefore:
    4 / (6250/3) = 3 / (3125/2) = 6/3125 s = 0.001920 s

Bytes are 8-bit ASCII, MSB forced to 0, transmitted LSB first.
The preamble is 16 bytes of 0xAB (10101011).

A complete SAME activation consists of:
  1. Preamble + Header ×3  (1-second silence between bursts)
  2. 1050 Hz attention tone (8–25 seconds)
  3. Voice / audio message
  4. Preamble + EOM ×3     (1-second silence between bursts)

Header format:
  ZCZC-ORG-EEE-PSSCCC+TTTT-JJJHHMM-LLLLLLLL-

EOM:
  NNNN
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
from typing import Optional

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]

log = logging.getLogger(__name__)

_same_log = logging.getLogger('same.audit')
_same_log.propagate = False


def _init_same_log() -> None:
    if _same_log.handlers:
        return
    log_dir = pathlib.Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_dir / 'same.log',
        maxBytes=5_242_880,
        backupCount=3,
    )
    handler.setFormatter(logging.Formatter(
        '%(asctime)s | %(message)s',
    ))
    _same_log.setLevel(logging.INFO)
    _same_log.addHandler(handler)

_SAME_SAMPLE_RATE: int = 16000

_MARK_FREQ: float = 2083 + 1 / 3
_SPACE_FREQ: float = 1562.5
_BIT_DURATION: float = 6 / 3125

_PREAMBLE_BYTE = 0xAB
_PREAMBLE_LENGTH = 16

_HEADER_SILENCE_S = 1.0
_ATTENTION_TONE_HZ = 1050.0
_DEFAULT_ATTN_DURATION_S = 8.0


@lru_cache(maxsize=8)
def _get_waves(sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(round(sample_rate * _BIT_DURATION))
    t = np.linspace(0, _BIT_DURATION, n, endpoint=False)
    mark = np.sin(2 * np.pi * _MARK_FREQ * t).astype(np.float32)
    space = np.sin(2 * np.pi * _SPACE_FREQ * t).astype(np.float32)
    return mark, space


@dataclass
class SAMEHeader:
    originator: str
    event: str
    locations: list[str]
    duration: str
    callsign: str
    issue_time: Optional[str] = None

    def encode(self) -> str:
        loc_str = "-".join(self.locations)
        issue = self.issue_time or get_time_code()
        cs = self.callsign.replace("-", "/").ljust(8)[:8]
        return f"ZCZC-{self.originator}-{self.event}-{loc_str}+{self.duration}-{issue}-{cs}-"


def get_time_code() -> str:
    now = time.gmtime()
    return f"{now.tm_yday:03d}{now.tm_hour:02d}{now.tm_min:02d}"


def convert_time_code(time_code: str) -> str:
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

    raise ValueError(f"Unsupported time format: {time_code}")


def _encode_bytes(data: bytes, sample_rate: int) -> np.ndarray:
    mark, space = _get_waves(sample_rate)
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    waves = np.stack([space, mark])
    return waves[bits].ravel()


def _preamble(sample_rate: int) -> np.ndarray:
    return _encode_bytes(bytes([_PREAMBLE_BYTE] * _PREAMBLE_LENGTH), sample_rate)


def _silence(duration_s: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(sample_rate * duration_s), dtype=np.float32)


def _attention_tone_wxr(duration_s: float, sample_rate: int) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return np.sin(2 * np.pi * _ATTENTION_TONE_HZ * t).astype(np.float32)

def _attention_tone_eas(duration_s: float, sample_rate: int) -> np.ndarray:
    # the EBS and EAS attention tone is a dual-tone signal of 853 hz and 960 hz sine waves. this signal is on for 8 seconds straight with no gaps.
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return ((np.sin(2 * np.pi * 853.0 * t) + np.sin(2 * np.pi * 960.0 * t)) / 2.0).astype(np.float32)

def _attention_tone_alert_ready(duration_s: float, sample_rate: int) -> np.ndarray:
    # The Canadian Alerting Attention Signal is an 8 second sequence of alternating, 500ms tones.
    # the first combonation of tones at frequencies of 932.33, 1046.50, and 3135.96 hz, followed by a second combination of tones at 440 Hz, 659.26 Hz and 3135.96 Hz
    n = int(sample_rate * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)

    combo_a = np.sin(2 * np.pi * 932.33 * t) + np.sin(2 * np.pi * 1046.50 * t) + np.sin(2 * np.pi * 3135.96 * t)
    combo_b = np.sin(2 * np.pi * 440.00 * t) + np.sin(2 * np.pi * 659.26 * t) + np.sin(2 * np.pi * 3135.96 * t)

    half = sample_rate // 2
    mask = ((np.arange(n) // half) % 2 == 0)
    return np.where(mask, combo_a, combo_b).astype(np.float32)

def _attention_tone_egg_timer(duration_s: float, sample_rate: int) -> np.ndarray:
    # this attention tone was used in Ontario's Weatheradio Canada network until 2021 before the AVIPADS system was decommissioned in favor of INOTIFY.
    # im just guessing but it sounds like in each cycle, there are four bursts of a 2450hz tone that lasts 65ms with a pause of 55ms between tones. repeat this to get 8 seconds. with a 1 second pause between each cucle of four bursts.
    n = int(sample_rate * duration_s)
    signal = np.zeros(n, dtype=np.float32)

    tone_len = int(round(sample_rate * 0.070))
    inter_len = int(round(sample_rate * 0.055))
    gap_len = int(round(sample_rate * 0.5))
    cycle_len = 4 * (tone_len + inter_len) + gap_len

    t = np.arange(n, dtype=np.float64) / sample_rate
    wave = (
        np.sin(2 * np.pi * 2055.0 * t) +
        0.25 * np.sin(2 * np.pi * 4110.0 * t) +
        0.10 * np.sin(2 * np.pi * 6165.0 * t) +
        0.05 * np.sin(2 * np.pi * 8220.0 * t)
    ).astype(np.float32) / 1.40

    pos = 0
    while pos < n:
        for j in range(4):
            s = pos + j * (tone_len + inter_len)
            e = min(s + tone_len, n)
            if s < n:
                signal[s:e] = wave[s:e]
        pos += cycle_len

    return signal

_PILOT_PREFIX_DURATION_S = 0.010
_PILOT_SUFFIX_DURATION_S = 0.030


def _pilot_tone(sample_rate: int, duration_s: float, freq: float = 2100.0) -> np.ndarray:
    n = int(round(sample_rate * duration_s))
    t = np.linspace(0, duration_s, n, endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def generate_header_burst(header: SAMEHeader, sample_rate: int = _SAME_SAMPLE_RATE) -> np.ndarray:
    pilot_prefix = _pilot_tone(sample_rate, _PILOT_PREFIX_DURATION_S, 2100.0)
    pre = _preamble(sample_rate)
    data = _encode_bytes(header.encode().encode("ascii"), sample_rate)
    pilot_suffix = _pilot_tone(sample_rate, _PILOT_SUFFIX_DURATION_S, 2100.0)
    return np.concatenate([_silence(0.100, sample_rate), pilot_prefix, pre, data, pilot_suffix])


def generate_eom_burst(sample_rate: int = _SAME_SAMPLE_RATE) -> np.ndarray:
    pilot_prefix = _pilot_tone(sample_rate, _PILOT_PREFIX_DURATION_S, 2100.0)
    pre = _preamble(sample_rate)
    data = _encode_bytes(b"NNNN", sample_rate)
    pilot_suffix = _pilot_tone(sample_rate, _PILOT_SUFFIX_DURATION_S, 2100.0)
    return np.concatenate([pilot_prefix, pre, data, pilot_suffix])

def generate_same(
    header: Optional[SAMEHeader] = None,
    attn_duration_s: float = _DEFAULT_ATTN_DURATION_S,
    sample_rate: int = _SAME_SAMPLE_RATE,
    tone_type: Optional[str] = "WXR",
    audio_msg_fp32: Optional[pathlib.Path] = None,
) -> np.ndarray:
    _init_same_log()
    gap = _silence(_HEADER_SILENCE_S, sample_rate)
    parts: list[np.ndarray] = []

    if header is not None:
        encoded = header.encode()
        log.info("Generating SAME alert: %s tone=%s %.1fs attn @ %d Hz", encoded, tone_type, attn_duration_s, sample_rate)

        burst = generate_header_burst(header, sample_rate)
        for i in range(3):
            parts.append(burst)
            if i < 2:
                parts.append(gap)

        if tone_type is not None:
            parts.append(_silence(1.0, sample_rate))
            if tone_type == "WXR":
                parts.append(_attention_tone_wxr(attn_duration_s, sample_rate))
            elif tone_type == "NPAS":
                parts.append(_attention_tone_alert_ready(attn_duration_s, sample_rate))
            elif tone_type == "EGG_TIMER":
                parts.append(_attention_tone_egg_timer(attn_duration_s, sample_rate))
            elif tone_type == "EAS":
                parts.append(_attention_tone_eas(attn_duration_s, sample_rate))
            else:
                log.warning("Unknown tone_type '%s', using silence", tone_type)
                parts.append(_silence(0.1, sample_rate))

        if audio_msg_fp32 is not None:
            log.info("Appending audio message from %s", audio_msg_fp32)
            parts.append(_silence(1.0, sample_rate))
            audio, sr = sf.read(audio_msg_fp32)  # type: ignore[no-untyped-call]
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            audio = audio.astype(np.float32)  # type: ignore
            if sr != sample_rate:
                log.info("Resampling audio message from %d Hz to %d Hz", sr, sample_rate)
                target_len = int(round(len(audio) * sample_rate / sr))
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, target_len),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)
            parts.append(audio)
    else:
        encoded = "EOM"
        log.info("Generating EOM-only sequence @ %d Hz", sample_rate)

    eom_burst = generate_eom_burst(sample_rate)
    parts.append(_silence(1.0, sample_rate))
    for i in range(3):
        parts.append(eom_burst)
        if i < 2:
            parts.append(gap)
    parts.append(_silence(1.0, sample_rate))

    total_s = sum(len(p) for p in parts) / sample_rate
    log.info("SAME sequence generated: %s (%.2fs at %d Hz)", encoded, total_s, sample_rate)
    _same_log.info(
        "SAME_GENERATED | header=%s | tone=%s | attn=%.1fs | audio=%s | duration=%.2fs | sr=%d",
        encoded, tone_type if header is not None else "none", attn_duration_s,
        str(audio_msg_fp32) if audio_msg_fp32 else "none", total_s, sample_rate,
    )

    out = np.concatenate(parts)
    return (out * 0.96).astype(np.float32)


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
    clipped = np.clip(signal, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def to_wav(signal: np.ndarray, sample_rate: int = _SAME_SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, signal, samplerate=sample_rate, format="WAV", subtype="PCM_16")  # type: ignore[no-untyped-call]
    return buf.getvalue()


if __name__ == "__main__":
    h = SAMEHeader(
        originator="WXR",
        event="CDW",
        locations=["065100"],
        duration="0030",
        callsign="XLF323",
    )

    sr = SAMPLE_RATE
    full = generate_same(h, sample_rate=sr, tone_type="WXR")

    import sys
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "same_test.wav"
    with open(wav_path, "wb") as f:
        f.write(to_wav(full, sample_rate=sr))

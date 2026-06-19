#!/usr/bin/env python3
"""Persistent Piper worker used by Haze's Go TTS provider."""

import argparse
import json
import sys
import traceback


def write_header(header):
    sys.stdout.buffer.write(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


def synthesize_request(voice, synthesis_config_cls, request):
    text = str(request.get("text") or "").strip()
    if not text:
        raise ValueError("empty synthesis text")

    config = synthesis_config_cls(
        speaker_id=request.get("speaker_id"),
        length_scale=request.get("length_scale"),
        noise_scale=request.get("noise_scale"),
        noise_w_scale=request.get("noise_w_scale"),
        normalize_audio=bool(request.get("normalize_audio", True)),
        volume=float(request.get("volume") or 1.0),
    )
    sentence_silence = float(request.get("sentence_silence") or 0.0)
    audio = bytearray()
    sample_rate = 0
    sample_width = 0
    channels = 0

    for index, chunk in enumerate(voice.synthesize(text, config)):
        if sample_rate == 0:
            sample_rate = int(chunk.sample_rate)
            sample_width = int(chunk.sample_width)
            channels = int(chunk.sample_channels)
        if index > 0 and sentence_silence > 0:
            frame_bytes = max(1, sample_width * channels)
            silence_frames = max(0, int(round(sample_rate * sentence_silence)))
            silence_bytes = silence_frames * frame_bytes
            audio.extend(b"\x00" * silence_bytes)
        audio.extend(chunk.audio_int16_bytes)

    if not audio:
        raise ValueError("piper produced no audio")
    return bytes(audio), sample_rate, sample_width, channels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    try:
        from piper import PiperVoice, SynthesisConfig

        voice = PiperVoice.load(args.model, args.config, use_cuda=args.cuda)
        write_header({"ready": True, "ok": True, "sample_rate": voice.config.sample_rate})
    except Exception as exc:
        write_header({"ready": True, "ok": False, "error": str(exc)})
        return 1

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            request_id = str(request.get("id") or "")
            pcm, sample_rate, sample_width, channels = synthesize_request(voice, SynthesisConfig, request)
            write_header(
                {
                    "id": request_id,
                    "ok": True,
                    "format": "pcm_s16le",
                    "sample_rate": sample_rate,
                    "sample_width": sample_width,
                    "channels": channels,
                    "bytes": len(pcm),
                }
            )
            sys.stdout.buffer.write(pcm)
            sys.stdout.buffer.flush()
        except Exception as exc:
            print(traceback.format_exc(), file=sys.stderr)
            write_header({"id": str(locals().get("request", {}).get("id", "")), "ok": False, "error": str(exc)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

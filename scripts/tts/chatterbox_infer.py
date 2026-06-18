#!/usr/bin/env python3
"""Small Chatterbox TTS CLI used by Haze's Go TTS provider."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-prompt", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=os.environ.get("HAZE_CHATTERBOX_DEVICE", "cpu"))
    parser.add_argument("--model", default=os.environ.get("HAZE_CHATTERBOX_MODEL", "turbo"))
    parser.add_argument("--language", default=os.environ.get("HAZE_CHATTERBOX_LANGUAGE", "en"))
    args = parser.parse_args()

    try:
        import torchaudio as ta
        if args.model.lower() in {"turbo", "chatterbox-turbo"}:
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            model = ChatterboxTurboTTS.from_pretrained(device=args.device)
            wav = model.generate(args.text, audio_prompt_path=args.audio_prompt)
        elif args.model.lower() in {"multilingual", "v3", "mtl"}:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            model = ChatterboxMultilingualTTS.from_pretrained(device=args.device, t3_model="v3")
            wav = model.generate(args.text, audio_prompt_path=args.audio_prompt, language_id=args.language)
        else:
            from chatterbox.tts import ChatterboxTTS

            model = ChatterboxTTS.from_pretrained(device=args.device)
            wav = model.generate(args.text, audio_prompt_path=args.audio_prompt)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        ta.save(args.output, wav, model.sr)
    except Exception as exc:  # noqa: BLE001 - CLI should report model import/runtime detail.
        print(f"chatterbox inference failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

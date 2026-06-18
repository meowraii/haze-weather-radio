#!/usr/bin/env python3
"""Small F5-TTS CLI used by Haze's Go TTS provider."""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", default="")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("HAZE_F5TTS_MODEL", ""))
    args = parser.parse_args()

    try:
        from f5_tts.api import F5TTS

        kwargs = {}
        if args.model:
            kwargs["model"] = args.model
        f5tts = F5TTS(**kwargs)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        f5tts.infer(
            ref_file=args.ref_audio,
            ref_text=args.ref_text,
            gen_text=args.text,
            file_wave=args.output,
            seed=None,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should report model import/runtime detail.
        print(f"f5-tts inference failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

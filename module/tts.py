import logging
import os
import pathlib
import queue
import re
import shutil
import subprocess
import wave
from typing import Any, Optional, cast
from urllib.parse import unquote, urlparse

from piper import PiperVoice, SynthesisConfig  # type: ignore[import-untyped]

from managed.events import shutdown_event, tts_queue
from module.dictionary import apply as apply_dictionary

log = logging.getLogger(__name__)

_FILE_URI_RE = re.compile(r'^file://', re.IGNORECASE)

_voice_cache: dict[str, PiperVoice] = {}


def _resolve_file_uri(uri: str) -> pathlib.Path | None:
    parsed = urlparse(uri.strip())
    if parsed.scheme.lower() != 'file':
        return None
    file_path = pathlib.Path(unquote(parsed.path))
    if not file_path.is_file():
        log.error("file:// URI target does not exist: %s", file_path)
        return None
    return file_path


def _transcode_to_wav(src: pathlib.Path, dst: pathlib.Path) -> bool:
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-i', str(src),
             '-ar', '22050', '-ac', '1', '-sample_fmt', 's16',
             str(dst)],
            check=True,
        )
        return True
    except Exception as e:
        log.error("Failed to transcode %s: %s", src, e)
        return False


def _load_voice(model_path: str, config_path: Optional[str] = None) -> PiperVoice:
    if model_path not in _voice_cache:
        _voice_cache[model_path] = PiperVoice.load(model_path, config_path=config_path)
    return _voice_cache[model_path]


def synthesize(
    config: dict[str, Any],
    text: str,
    feed_id: str,
    package_id: str,
    lang: str = 'en-CA',
    voice: Optional[str] = None,
) -> Optional[pathlib.Path]:
    out_dir = pathlib.Path("output") / feed_id / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{package_id}.wav"
    tmp_path = out_dir / f"{package_id}.tmp.wav"

    stripped = text.strip()
    if _FILE_URI_RE.match(stripped) and '\n' not in stripped:
        src = _resolve_file_uri(stripped)
        if src is None:
            return None
        if not _transcode_to_wav(src, tmp_path):
            return None
        os.replace(str(tmp_path), str(out_path))
        return out_path

    segments = _split_segments(text)
    has_file = any(kind == 'file' for kind, _ in segments)

    if has_file:
        return _synthesize_mixed(config, segments, feed_id, package_id, lang, voice, out_path, tmp_path)

    return _synthesize_text(config, text, feed_id, package_id, lang, voice, out_path, tmp_path)


def _split_segments(text: str) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    text_parts: list[str] = []

    for line in text.split('\n'):
        line_stripped = line.strip()
        if _FILE_URI_RE.match(line_stripped):
            if text_parts:
                segments.append(('text', '\n'.join(text_parts)))
                text_parts = []
            segments.append(('file', line_stripped))
        else:
            text_parts.append(line)

    if text_parts:
        combined = '\n'.join(text_parts).strip()
        if combined:
            segments.append(('text', combined))

    return segments


def _synthesize_mixed(
    config: dict[str, Any],
    segments: list[tuple[str, str]],
    feed_id: str,
    package_id: str,
    lang: str,
    voice: Optional[str],
    out_path: pathlib.Path,
    tmp_path: pathlib.Path,
) -> Optional[pathlib.Path]:
    part_paths: list[pathlib.Path] = []
    part_dir = out_path.parent / f".{package_id}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, (kind, content) in enumerate(segments):
            part_path = part_dir / f"seg_{i:03d}.wav"
            if kind == 'file':
                src = _resolve_file_uri(content)
                if src is None:
                    continue
                if not _transcode_to_wav(src, part_path):
                    continue
                part_paths.append(part_path)
            else:
                result = _synthesize_text(
                    config, content, feed_id,
                    f"{package_id}_seg{i}", lang, voice,
                    part_path, part_dir / f"seg_{i:03d}.tmp.wav",
                )
                if result:
                    part_paths.append(result)

        if not part_paths:
            return None

        if len(part_paths) == 1:
            shutil.copy2(str(part_paths[0]), str(tmp_path))
        else:
            concat_list = part_dir / "concat.txt"
            concat_list.write_text(
                '\n'.join(f"file '{p.resolve()}'" for p in part_paths),
                encoding='utf-8',
            )
            try:
                subprocess.run(
                    ['ffmpeg', '-y', '-loglevel', 'error',
                     '-f', 'concat', '-safe', '0',
                     '-i', str(concat_list),
                     '-ar', '22050', '-ac', '1', '-sample_fmt', 's16',
                     str(tmp_path)],
                    check=True,
                )
            except Exception as e:
                log.error("Failed to concat segments for %s/%s: %s", feed_id, package_id, e)
                return None

        os.replace(str(tmp_path), str(out_path))
        return out_path
    finally:
        shutil.rmtree(part_dir, ignore_errors=True)


def _synthesize_text(
    config: dict[str, Any],
    text: str,
    feed_id: str,
    package_id: str,
    lang: str = 'en-CA',
    voice: Optional[str] = None,
    out_path: pathlib.Path | None = None,
    tmp_path: pathlib.Path | None = None,
) -> Optional[pathlib.Path]:
    tts_cfg = config.get('tts', {})
    global_piper: dict[str, Any] = tts_cfg.get('piper', {})

    if not global_piper.get('enabled', True):
        log.warning("TTS is disabled")
        return None

    lang_piper: dict[str, Any] = (
        tts_cfg.get('lang', {})
        .get(lang, {})
        .get('backend', {})
        .get('piper', {})
    )

    if voice in ('male', 'female'):
        preferred = lang_piper.get(voice)
        voice_cfg_raw: Any = preferred if isinstance(preferred, dict) else (
            lang_piper.get('male') or lang_piper.get('female')
        )
    else:
        voice_cfg_raw = lang_piper.get('male') or lang_piper.get('female')
    if not isinstance(voice_cfg_raw, dict):
        log.error("No voice configuration found for language %s", lang)
        return None
    voice_cfg = cast(dict[str, Any], voice_cfg_raw)

    model_path: Optional[str] = voice_cfg.get('model')
    config_path: Optional[str] = voice_cfg.get('config')
    if not model_path:
        log.error("No model path for language %s", lang)
        return None

    try:
        voice = _load_voice(model_path, config_path)
    except Exception as e:
        log.error("Failed to load piper voice %s: %s", model_path, e)
        return None

    speaker_id_raw = voice_cfg.get('speaker', global_piper.get('speaker', 0))
    speaker_id: Optional[int] = int(speaker_id_raw) if speaker_id_raw is not None else None

    syn_config = SynthesisConfig(
        speaker_id=speaker_id,
        length_scale=global_piper.get('length_scale', 1.0),
        noise_scale=global_piper.get('noise_scale', 0.667),
        noise_w_scale=global_piper.get('noise_w', 0.8),
    )

    text = apply_dictionary(text, lang)

    if out_path is None or tmp_path is None:
        out_dir = pathlib.Path("output") / feed_id / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_path or (out_dir / f"{package_id}.wav")
        tmp_path = tmp_path or (out_dir / f"{package_id}.tmp.wav")

    try:
        with wave.open(str(tmp_path), 'wb') as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)
        os.replace(str(tmp_path), str(out_path))
    except Exception as e:
        log.error("Synthesis failed for %s/%s: %s", feed_id, package_id, e)
        return None

    return out_path


def tts_thread_worker(config: dict[str, Any]) -> None:
    while not shutdown_event.is_set():
        try:
            item = tts_queue.get(timeout=1)
        except queue.Empty:
            continue
        if item is None:
            break
        feed_id, pkg_id, text, lang, *rest = item
        voice: Optional[str] = rest[0] if rest else None
        log.info('[%s/%s] Synthesizing: %s', feed_id, lang, pkg_id)
        out = synthesize(config, text, feed_id, pkg_id, lang, voice)
        if out:
            log.debug('[%s/%s] Wrote %s', feed_id, lang, out)
        tts_queue.task_done()

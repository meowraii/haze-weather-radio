import json
import pathlib
import re
from typing import Optional

_DICT_PATH = pathlib.Path(__file__).parent.parent / 'managed' / 'dictionary.json'

_loaded: Optional[dict[str, dict[str, str]]] = None


def _load() -> dict[str, dict[str, str]]:
    global _loaded
    if _loaded is None:
        with open(_DICT_PATH, encoding='utf-8') as f:
            _loaded = json.load(f)
    return _loaded


def _get_entries(lang: str) -> list[tuple[str, str]]:
    data = _load()
    lang_prefix = lang[:2] + '-*'
    combined: dict[str, str] = {}
    for key, entries in data.items():
        if key == lang_prefix:
            combined.update(entries)
        elif key == 'en-*' and lang_prefix == 'en-*':
            combined.update(entries)
    if not combined:
        en = data.get('en-*', {})
        combined.update(en)
    return list(combined.items())


def apply(text: str, lang: str = 'en-CA') -> str:
    text = re.sub(r'(\d+)\s+MB\b', r'\1 millibars', text)
    entries = _get_entries(lang)
    entries.sort(key=lambda x: len(x[0]), reverse=True)
    for term, replacement in entries:
        pattern = r'(?<![A-Za-z0-9])' + re.escape(term) + r'(?![A-Za-z0-9])'
        text = re.sub(pattern, replacement, text)
    return text

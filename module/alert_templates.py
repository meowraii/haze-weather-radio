from __future__ import annotations

from collections.abc import Iterable
import pathlib
from typing import Any
import xml.etree.ElementTree as ET


DEFAULT_ALERT_TEMPLATES_PATH = pathlib.Path('managed') / 'alertTemplates.xml'


def _text(node: ET.Element | None) -> str:
    if node is None or node.text is None:
        return ''
    return node.text.strip()


def _optional_text(node: ET.Element | None) -> str | None:
    value = _text(node)
    return value or None


def _bool_attr(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {'1', 'true', 'yes', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'off'}:
        return False
    return default


def _int_attr(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip()) if value is not None else default
    except (TypeError, ValueError):
        return default


def _parse_locations(parent: ET.Element | None) -> list[dict[str, str]]:
    if parent is None:
        return []
    locations: list[dict[str, str]] = []
    for location_el in parent.findall('location'):
        location_id = location_el.get('id', '').strip()
        if not location_id:
            continue
        location: dict[str, str] = {'id': location_id}
        source = location_el.get('source', '').strip()
        if source:
            location['source'] = source
        locations.append(location)
    return locations


def _parse_langs(parent: ET.Element | None) -> dict[str, str]:
    if parent is None:
        return {}
    langs: dict[str, str] = {}
    for lang_el in parent.findall('lang'):
        code = lang_el.get('code', '').strip()
        if not code:
            continue
        text = _text(lang_el.find('text'))
        if text:
            langs[code] = text
    return langs


def _parse_weeks(parent: ET.Element | None) -> list[dict[str, Any]]:
    if parent is None:
        return []
    weeks: list[dict[str, Any]] = []
    for week_el in parent.findall('week'):
        week: dict[str, Any] = {
            'enabled': _bool_attr(week_el.get('enabled'), False),
            'value': _int_attr(week_el.text, 0),
        }
        event_override = week_el.get('event_override', '').strip()
        if event_override:
            week['event_override'] = event_override
        weeks.append(week)
    return weeks


def _parse_template(template_el: ET.Element) -> dict[str, Any]:
    name = _text(template_el.find('name'))
    description = _text(template_el.find('description'))

    automated_el = template_el.find('automated')
    timing_el = automated_el.find('timing') if automated_el is not None else None
    automated: dict[str, Any] = {
        'enabled': _bool_attr(automated_el.findtext('enabled') if automated_el is not None else None, False),
        'allow_postpone': _bool_attr(automated_el.findtext('allow_postpone') if automated_el is not None else None, False),
        'timing': {
            'timezone': _optional_text(timing_el.find('timezone') if timing_el is not None else None),
            'day_of_week': _text(timing_el.find('day_of_week') if timing_el is not None else None),
            'weeks': _parse_weeks(timing_el.find('weeks') if timing_el is not None else None),
            'hour': _int_attr(timing_el.findtext('hour') if timing_el is not None else None, 0),
            'minute': _int_attr(timing_el.findtext('minute') if timing_el is not None else None, 0),
            'second': _int_attr(timing_el.findtext('second') if timing_el is not None else None, 0),
        },
    }

    same_el = template_el.find('same')
    content_el = same_el.find('content') if same_el is not None else None
    duration_el = same_el.find('duration') if same_el is not None else None
    duration_hours = _int_attr(duration_el.get('hr') if duration_el is not None else None, 0)
    duration_minutes = _int_attr(duration_el.get('min') if duration_el is not None else None, 15)
    same_event = _text(same_el.find('event') if same_el is not None else None)
    same: dict[str, Any] = {
        'event': same_event,
        'locations': _parse_locations(same_el.find('locations') if same_el is not None else None),
        'duration': {
            'hr': duration_hours,
            'min': duration_minutes,
        },
        'sender_id': _optional_text(same_el.find('sender_id') if same_el is not None else None),
        'content': {
            'attention_tone': content_el.get('attention_tone', '').strip() if content_el is not None else '',
            'lang': _parse_langs(content_el),
        },
    }

    template_key = same_event or name or template_el.get('id', '').strip()
    template: dict[str, Any] = {
        'name': name,
        'description': description,
        'automated': automated,
        'same': same,
        'sameEvent': same_event,
        'sameExpire': f'{duration_hours:02d}{duration_minutes:02d}',
        'msg': same['content']['lang'],
    }
    if template_key:
        template['_key'] = template_key
    return template


def load_alert_templates(path: pathlib.Path | str | None = None) -> dict[str, dict[str, Any]]:
    template_path = pathlib.Path(path) if path is not None else DEFAULT_ALERT_TEMPLATES_PATH
    if not template_path.exists():
        return {}
    try:
        root = ET.parse(template_path).getroot()
    except ET.ParseError:
        return {}

    templates: dict[str, dict[str, Any]] = {}
    for template_el in root.findall('template'):
        template = _parse_template(template_el)
        key = str(template.pop('_key', '')).strip()
        if key:
            templates[key] = template
    return templates


def _set_text(parent: ET.Element, tag: str, value: Any, attrs: dict[str, str] | None = None) -> ET.Element:
    child = ET.SubElement(parent, tag, attrs or {})
    if value not in (None, ''):
        child.text = str(value)
    return child


def _write_langs(parent: ET.Element, langs: dict[str, str]) -> None:
    for code, text in langs.items():
        lang_el = ET.SubElement(parent, 'lang', {'code': str(code)})
        _set_text(lang_el, 'text', text)


def _write_weeks(parent: ET.Element, weeks: Iterable[dict[str, Any]]) -> None:
    for week in weeks:
        attrs = {'enabled': 'true' if bool(week.get('enabled', False)) else 'false'}
        event_override = str(week.get('event_override', '')).strip()
        if event_override:
            attrs['event_override'] = event_override
        week_el = ET.SubElement(parent, 'week', attrs)
        week_value = week.get('value', 0)
        if week_value not in (None, ''):
            week_el.text = str(int(week_value))


def _template_key(code: str, template: dict[str, Any]) -> str:
    same = template.get('same') if isinstance(template.get('same'), dict) else {}
    same_event = str(template.get('sameEvent') or same.get('event') or '').strip()
    return code.strip() or same_event or str(template.get('name', '')).strip()


def write_alert_templates(templates: dict[str, dict[str, Any]], path: pathlib.Path | str | None = None) -> None:
    template_path = pathlib.Path(path) if path is not None else DEFAULT_ALERT_TEMPLATES_PATH
    root = ET.Element('templates')

    for code, template in templates.items():
        if not isinstance(template, dict):
            continue
        template_key = _template_key(str(code), template)
        if not template_key:
            continue

        template_el = ET.SubElement(root, 'template')
        _set_text(template_el, 'name', template.get('name') or template_key)
        _set_text(template_el, 'description', template.get('description') or '')

        automated = template.get('automated') if isinstance(template.get('automated'), dict) else {}
        automated_el = ET.SubElement(template_el, 'automated')
        _set_text(automated_el, 'enabled', 'true' if bool(automated.get('enabled', False)) else 'false')
        _set_text(automated_el, 'allow_postpone', 'true' if bool(automated.get('allow_postpone', False)) else 'false')
        timing = automated.get('timing') if isinstance(automated.get('timing'), dict) else {}
        timing_el = ET.SubElement(automated_el, 'timing')
        _set_text(timing_el, 'timezone', timing.get('timezone') or '')
        _set_text(timing_el, 'day_of_week', timing.get('day_of_week') or '')
        weeks_el = ET.SubElement(timing_el, 'weeks')
        _write_weeks(weeks_el, timing.get('weeks', []) if isinstance(timing.get('weeks'), list) else [])
        _set_text(timing_el, 'hour', int(timing.get('hour', 0) or 0))
        _set_text(timing_el, 'minute', int(timing.get('minute', 0) or 0))
        _set_text(timing_el, 'second', int(timing.get('second', 0) or 0))

        same = template.get('same') if isinstance(template.get('same'), dict) else {}
        same_el = ET.SubElement(template_el, 'same')
        same_event = str(template.get('sameEvent') or same.get('event') or template_key).strip()
        _set_text(same_el, 'event', same_event)

        locations = same.get('locations') if isinstance(same.get('locations'), list) else []
        locations_el = ET.SubElement(same_el, 'locations')
        for location in locations:
            if not isinstance(location, dict):
                continue
            location_id = str(location.get('id', '')).strip()
            if not location_id:
                continue
            attrs = {'id': location_id}
            source = str(location.get('source', '')).strip()
            if source:
                attrs['source'] = source
            ET.SubElement(locations_el, 'location', attrs)

        duration = same.get('duration') if isinstance(same.get('duration'), dict) else {}
        same_expire = str(template.get('sameExpire', '')).strip()
        if not duration and len(same_expire) == 4 and same_expire.isdigit():
            duration = {'hr': int(same_expire[:2]), 'min': int(same_expire[2:])}
        ET.SubElement(
            same_el,
            'duration',
            {
                'hr': str(int(duration.get('hr', 0) or 0)),
                'min': str(int(duration.get('min', 15) or 15)),
            },
        )
        _set_text(same_el, 'sender_id', same.get('sender_id') or '')

        content = same.get('content') if isinstance(same.get('content'), dict) else {}
        content_el = ET.SubElement(same_el, 'content')
        attention_tone = str(content.get('attention_tone', '')).strip()
        if attention_tone:
            content_el.set('attention_tone', attention_tone)
        elif 'attention_tone' in content:
            content_el.set('attention_tone', '')

        langs = content.get('lang') if isinstance(content.get('lang'), dict) else {}
        if langs:
            _write_langs(content_el, langs)
        elif isinstance(template.get('msg'), dict):
            _write_langs(content_el, template['msg'])

    ET.indent(root, space='    ')
    template_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(template_path, encoding='utf-8', xml_declaration=True)


def merge_alert_templates(existing: dict[str, dict[str, Any]], updates: dict[str, Any]) -> dict[str, dict[str, Any]]:
    merged = {key: value.copy() for key, value in existing.items()}
    for code, template in updates.items():
        if not isinstance(template, dict):
            continue
        key = str(code).strip() or str(template.get('sameEvent', '')).strip() or str(template.get('name', '')).strip()
        if not key:
            continue
        current = merged.get(key, {})
        next_template = current.copy() if isinstance(current, dict) else {}
        for field in ('name', 'description', 'automated', 'same', 'sameEvent', 'sameExpire', 'msg'):
            if field in template:
                next_template[field] = template[field]
        if 'sameEvent' not in next_template:
            same = next_template.get('same') if isinstance(next_template.get('same'), dict) else {}
            same_event = str(same.get('event', '')).strip()
            if same_event:
                next_template['sameEvent'] = same_event
        if 'sameExpire' not in next_template:
            same = next_template.get('same') if isinstance(next_template.get('same'), dict) else {}
            duration = same.get('duration') if isinstance(same.get('duration'), dict) else {}
            next_template['sameExpire'] = f"{int(duration.get('hr', 0) or 0):02d}{int(duration.get('min', 15) or 15):02d}"
        if 'msg' not in next_template:
            same = next_template.get('same') if isinstance(next_template.get('same'), dict) else {}
            content = same.get('content') if isinstance(same.get('content'), dict) else {}
            langs = content.get('lang') if isinstance(content.get('lang'), dict) else {}
            next_template['msg'] = langs
        same = next_template.get('same') if isinstance(next_template.get('same'), dict) else {}
        duration_code = str(next_template.get('sameExpire', '0015')).strip()
        if len(duration_code) == 4 and duration_code.isdigit():
            same_duration = same.get('duration') if isinstance(same.get('duration'), dict) else {}
            same_duration['hr'] = int(duration_code[:2])
            same_duration['min'] = int(duration_code[2:])
            same['duration'] = same_duration
        if 'event' not in same and str(next_template.get('sameEvent', '')).strip():
            same['event'] = str(next_template['sameEvent']).strip()
        if 'content' not in same or not isinstance(same.get('content'), dict):
            same['content'] = {}
        content = same['content']
        if 'lang' not in content or not isinstance(content.get('lang'), dict):
            content['lang'] = dict(next_template.get('msg', {})) if isinstance(next_template.get('msg'), dict) else {}
        next_template['same'] = same
        merged[key] = next_template
    return merged
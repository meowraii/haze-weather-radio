from dataclasses import dataclass
import datetime
import os
import pathlib
import re
from typing import Any, ClassVar, Optional, cast
from zoneinfo import ZoneInfo


def _d(src: dict[str, Any], key: str) -> dict[str, Any]:
    v = src.get(key)
    return cast(dict[str, Any], v) if isinstance(v, dict) else {}


def _s(src: dict[str, Any], key: str) -> Any:
    return src.get(key)


def _scalar(field: Any) -> Any:
    if isinstance(field, dict):
        d: dict[str, Any] = cast(dict[str, Any], field)
        return d.get('value')
    return field


def get_feed_info(config: dict[str, Any], feed_id: str) -> Optional[dict[str, Any]]:
    return next((feed for feed in config['feeds'] if feed['id'] == feed_id), None)


_DT_PREFIXES: dict[str, dict[str, str]] = {
    'en-CA': {
        "morning": "Good morning.",
        "afternoon": "Good afternoon.",
        "evening": "Good evening.",
        "night": "Good night.",
    },
    'fr-CA': {
        "morning": "Bonjour.",
        "afternoon": "Bon après-midi.",
        "evening": "Bonsoir.",
        "night": "Bonne nuit.",
    },
    'es-US': {
        "morning": "Buenos días.",
        "afternoon": "Buenas tardes.",
        "evening": "Buenas noches.",
        "night": "Buenas noches.",
    },
}

_DT_ORDINALS: dict[str, list[str]] = {
    'en': ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'eleventh', 'twelfth', 'thirteenth', 'fourteenth', 'fifteenth', 'sixteenth', 'seventeenth', 'eighteenth', 'nineteenth', 'twentieth', 'twenty-first', 'twenty-second', 'twenty-third', 'twenty-fourth', 'twenty-fifth', 'twenty-sixth', 'twenty-seventh', 'twenty-eighth', 'twenty-ninth', 'thirtieth', 'thirty-first'],
}

_DT_DAYS: dict[str, list[str]] = {
    'en': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
    'fr': ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'],
    'es': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo'],
}

_DT_MONTHS: dict[str, list[str]] = {
    'en': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'],
    'fr': ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'],
    'es': ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'],
}

_DT_SENTENCES: dict[str, list[str]] = {
    'en': ["The current time is {hour}. {minute}. {ampm}. {tzabbr}."],
    'fr': ["Il est actuellement {hour} {minute} {ampm} {tzabbr}."],
    'es': ["La hora actual es {hour} {minute} {ampm} {tzabbr}."],
}

_CC_SOURCE: dict[str, dict[str, str]] = {
    'en': {
        "eccc": "Environment Canada",
        "nws": "the National Weather Service",
        "weatherdotcom": "Weather Dot Com",
    },
    'fr': {
        "eccc": "Environnement Canada",
        "nws": "le Service météorologique national",
        "weatherdotcom": "Weather Dot Com",
    },
    'es': {
        "eccc": "Environment Canada",
        "nws": "el Servicio Meteorológico Nacional",
        "weatherdotcom": "Weather Dot Com",
    },
}

_CC_PH: dict[str, dict[str, str]] = {
    'en': {
        'opener':            "The current weather conditions. Issued by {source} at {time}.",
        'unavailable':       "The report at {name} was not available.",
        'condition':         "The weather at {name} was {cond}.",
        'secondary_no_cond': "At {name}.",
        'no_cond':           "At {name}. The sky conditions were not available.",
        'temp':              "The temperature was {val} degrees Celsius.",
        'dewpoint':          "dewpoint {val} degrees.",
        'humidity':          "and the relative humidity was {val} percent.",
        'no_temp':           "The temperature was not available.",
        'wind':              "Winds were {dir}at {spd} kilometres per hour.",
        'wind_dir':          "out of the {dir} ",
        'no_wind':           "The wind information was not available.",
        'gust':              "Gusting to {val} kilometres per hour.",
        'no_humidity':       "The relative humidity was not available.",
        'visibility':        "The visibility was {val} kilometres.",
        'no_visibility':     "The visibility was not available.",
        'pressure':          "The pressure was {val} kilopascals{tend}.",
        'no_pressure':       "The barometric pressure was not available.",
        'tendency':          ", and {val}",
        'windchill':         "The wind chill was {val}.",
        'humidex':           "The humidex was {val}.",
        'station':           "this station",
    },
    'fr': {
        'opener':            "Les conditions météorologiques actuelles. Émis par {source} à {time}.",
        'unavailable':       "Les conditions actuelles à {name} ne sont pas disponibles pour le moment.",
        'condition':         "La météo à {name} était {cond}.",
        'secondary_no_cond': "À {name}.",
        'no_cond':           "À {name}. Les conditions du ciel n'étaient pas disponibles.",
        'temp':              "La température est de {val} degrés Celsius.",
        'dewpoint':          "Le point de rosée était de {val} degrés Celsius.",
        'no_temp':           "La température n'était pas disponible.",
        'wind':              "Les vents soufflaient {dir}à {spd} kilomètres par heure.",
        'wind_dir':          "du {dir} ",
        'no_wind':           "Les informations sur le vent n'étaient pas disponibles.",
        'gust':              "Avec des rafales atteignant {val} kilomètres par heure.",
        'humidity':          "L'humidité relative était de {val} pour cent.",
        'no_humidity':       "L'humidité relative n'était pas disponible.",
        'visibility':        "La visibilité était de {val} kilomètres.",
        'no_visibility':     "La visibilité n'était pas disponible.",
        'pressure':          "La pression barométrique était de {val} kilopascals{tend}.",
        'no_pressure':       "La pression barométrique n'était pas disponible.",
        'tendency':          ", et {val}",
        'windchill':         "Le refroidissement éolien était de {val}.",
        'humidex':           "L'humidex était de {val}.",
        'station':           "cette station",
    },
    'es': {
        'opener':            "Las condiciones meteorológicas actuales. Emitido por {source} a las {time}.",
        'unavailable':       "Las condiciones actuales en {name} no están disponibles en este momento.",
        'condition':         "El tiempo en {name} era {cond}.",
        'secondary_no_cond': "En {name}.",
        'no_cond':           "En {name}. Las condiciones del cielo no estaban disponibles.",
        'temp':              "La temperatura es de {val} grados Celsius.",
        'dewpoint':          "El punto de rocío era de {val} grados Celsius.",
        'no_temp':           "La temperatura no estaba disponible.",
        'wind':              "Los vientos soplaban {dir}a {spd} kilómetros por hora.",
        'wind_dir':          "del {dir} ",
        'no_wind':           "La información sobre el viento no estaba disponible.",
        'gust':              "Con rachas de hasta {val} kilómetros por hora.",
        'humidity':          "La humedad relativa era del {val} por ciento.",
        'no_humidity':       "La humedad relativa no estaba disponible.",
        'visibility':        "La visibilidad era de {val} kilómetros.",
        'no_visibility':     "La visibilidad no estaba disponible.",
        'pressure':          "La presión barométrica era de {val} kilopascales{tend}.",
        'no_pressure':       "La presión barométrica no estaba disponible.",
        'tendency':          ", y {val}",
        'windchill':         "La sensación térmica era de {val}.",
        'humidex':           "El humidex era de {val}.",
        'station':           "esta estación",
    },
}

_FX_PH: dict[str, dict[str, str]] = {
    'en': {
        'intro':       "Here is the forecast for {name}.",
        'unavailable': "Forecast information for {name} is unavailable at this time.",
        'station':     "this station",
    },
    'fr': {
        'intro':       "Voici les prévisions pour {name}.",
        'unavailable': "Les informations de prévision pour {name} ne sont pas disponibles pour le moment.",
        'station':     "cette station",
    },
    'es': {
        'intro':       "Aquí está el pronóstico para {name}.",
        'unavailable': "La información del pronóstico para {name} no está disponible en este momento.",
        'station':     "esta estación",
    },
}

_DISCUSSION_SECTION_HEADERS: dict[str, str] = {
    'SYNOPTIC OVERVIEW': 'Synoptic overview.',
    'DISCUSSION': 'Discussion.',
    'ALERTS IN EFFECT': 'Alerts in effect.',
}

_DISCUSSION_GEO_ABBR: dict[str, str] = {
    'AB': 'Alberta',
    'SK': 'Saskatchewan',
    'MB': 'Manitoba',
    'ON': 'Ontario',
    'QC': 'Quebec',
    'BC': 'British Columbia',
    'NT': 'Northwest Territories',
    'NU': 'Nunavut',
    'YT': 'Yukon',
    'NRN': 'northern',
    'SRN': 'southern',
    'NWT': 'Northwest Territories',
}

_DISCUSSION_KEEP_UPPER: set[str] = set(_DISCUSSION_GEO_ABBR.keys()) | {
    'AM', 'PM', 'CDT', 'CST', 'MDT', 'MST', 'PDT', 'PST', 'EDT', 'EST',
    'ADT', 'AST', 'NDT', 'NST', 'UTC', 'GMT',
    'NWS', 'NOAA', 'ECCC', 'MSC', 'WFO', 'SPC', 'NHC',
    'UV', 'AQI', 'RH',
}
_DISCUSSION_KEEP_UPPER -= {'ON', 'IN', 'IS', 'AS', 'AT', 'BE', 'BY', 'DO', 'GO', 'IF', 'IT', 'NO', 'OF', 'OR', 'SO', 'TO', 'UP', 'US', 'WE'}

_DISCUSSION_GEO_HEADER_RE = re.compile(
    r'^([A-Z]{2,}(?:[/,][A-Z]{2,})*(?:\s+[A-Z]{2,}(?:[/,][A-Z]{2,})*)*)\.\.\.'
)
_DISCUSSION_SECTION_HEADER_RE = re.compile(r'^([A-Z][A-Z ]{4,})\.\.\.')
_DISCUSSION_BYLINE_RE = re.compile(r'^END/')
_DISCUSSION_FOCN_RE = re.compile(r'^(?:FOCN|CWWG)\d')
_DISCUSSION_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')
_DISCUSSION_GEO_SPLIT_RE = re.compile(r'[/,]')


def _expand_geo_header(header: str) -> str:
    header = header.rstrip('.').rstrip('/')
    parts = _DISCUSSION_GEO_SPLIT_RE.split(header)
    expanded: list[str] = []
    for raw in parts:
        raw = raw.strip()
        tokens = raw.split()
        out_tokens: list[str] = []
        for tok in tokens:
            out_tokens.append(_DISCUSSION_GEO_ABBR.get(tok, tok.title()))
        expanded.append(' '.join(out_tokens))
    return ', '.join(expanded)


def _titlecase_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    if not sentence:
        return sentence
    words = sentence.split()
    result: list[str] = []
    for i, word in enumerate(words):
        clean = word.strip('.,;:!?()/\'-')
        if clean in _DISCUSSION_KEEP_UPPER:
            result.append(word)
        elif clean.isupper() and len(clean) >= 1:
            lowered = word.lower()
            if i == 0:
                result.append(lowered[0].upper() + lowered[1:])
            else:
                result.append(lowered)
        elif i == 0 and word:
            result.append(word[0].upper() + word[1:].lower() if word.isupper() else word[0].upper() + word[1:])
        else:
            result.append(word)
    return ' '.join(result)


def _flush_discussion(para_lines: list[str]) -> str:
    joined = ' '.join(para_lines)
    sentences = _DISCUSSION_SENTENCE_SPLIT_RE.split(joined)
    return ' '.join(_titlecase_sentence(s) for s in sentences)

@dataclass
class Package_Config:
    feed_id: str
    package_type: str
    lang: Optional[str] = None
    weather_data: Optional[dict[str, Any]] = None

    ttl: ClassVar[dict[str, float]] = {
        'date_time': 24.0,
        'current_conditions': 2700.0,
        'forecast': 7200.0,
        'eccc_discussion': 10800.0,
        'geophysical_alert': 10800.0,
    }

    per_package: ClassVar[dict[str, dict[str, Any]]] = { # config overrides for specific packages; omit a package to allow all languages
        'eccc_discussion': {
            'lang': ['en-CA'],
        },
        'forecast': {
            'use_voice': 'female', # male or female. if specified in the config for each language.
        },
        'geophysical_alert': {
            'lang': ['en-CA'],
            'use_voice': 'female', # male or female. if specified in the config for each language.
        },
    }


def date_time_package(tz: str, lang: Optional[str] = "en-CA") -> str:
    _lang = lang or 'en-CA'
    locale_prefixes = _DT_PREFIXES.get(_lang, _DT_PREFIXES['en-CA'])
    zone = ZoneInfo(tz)
    now = datetime.datetime.now(tz=zone)
    hour = now.hour
    if 5 <= hour < 12:
        prefix = locale_prefixes["morning"]
    elif 12 <= hour < 17:
        prefix = locale_prefixes["afternoon"]
    elif 17 <= hour < 22:
        prefix = locale_prefixes["evening"]
    else:
        prefix = locale_prefixes["night"]

    lang_short = _lang[:2]
    return " ".join([
        prefix,
        _DT_SENTENCES.get(lang_short, _DT_SENTENCES['en'])[0].format(
            hour=now.strftime('%-I').replace('0', ''),
            minute=now.strftime('%M'),
            ampm=now.strftime('%p'),
            tzabbr=now.strftime('%Z'),
        )
    ])

def station_id(config: dict[str, Any], feed_id: str, lang: Optional[str] = "en-CA") -> str:
    _lang = lang or 'en-CA'
    operator: dict[str, Any] = config['operator']
    feed = get_feed_info(config, feed_id)
    if feed is None:
        return "Feed not found."

    desc_block: dict[str, Any] = _d(feed, 'description')
    desc: dict[str, Any] = _d(desc_block, _lang)

    sentences: list[str] = []

    prefix = _s(desc, 'prefix')
    if prefix:
        sentences.append(str(prefix))
    else:
        on_air_name: str = operator.get('on_air_name') or feed['name']
        rf_cfg = _d(feed, 'rf')
        callsign_raw = rf_cfg.get('callsign')
        callsign: Optional[str] = str(callsign_raw) if callsign_raw else None
        freq_hz_raw = rf_cfg.get('frequency_hz')
        freq_mhz: Optional[str] = (
            f"{float(freq_hz_raw) / 1_000_000:.3f}".rstrip('0').rstrip('.')
            if freq_hz_raw is not None
            else None
        )
        callsign_spelled = " ".join(callsign) if callsign else None
        if callsign_spelled and freq_mhz:
            sentences.append(
                f"You are listening to {on_air_name}, "
                f"call sign {callsign_spelled}, "
                f"broadcasting on {freq_mhz} megahertz."
            )
        elif callsign_spelled:
            sentences.append(
                f"You are listening to {on_air_name}, "
                f"call sign {callsign_spelled}."
            )
        else:
            sentences.append(f"You are listening to {on_air_name}.")

    if desc:
        text = _s(desc, 'text')
        suffix = _s(desc, 'suffix')
        if text:
            sentences.append(str(text))
        if suffix:
            sentences.append(str(suffix))

    contact: list[str] = []
    email = operator.get('email')
    phone = operator.get('phone')
    if email and str(email) != 'None':
        contact.append(f"email us at {email}")
    if phone and str(phone) != 'None':
        contact.append(f"call us at {phone}")
    if contact:
        sentences.append(f"If you have comments or concerns, please direct them to {' or '.join(contact)}.")

    return " ".join(sentences)

def user_bulletin_package(
    bulletins: Optional[list[Any]] = None,
    lang: Optional[str] = "en-CA",
    tz: Optional[str] = "UTC",
) -> str:
    _lang = lang or 'en-CA'
    if not bulletins:
        return ""

    zone = ZoneInfo(tz or 'UTC')
    now = datetime.datetime.now(tz=zone)

    active: list[str] = []
    for item in bulletins:
        if not isinstance(item, dict):
            continue
        if not item.get('enabled', True):
            continue

        date_start_raw = item.get('dateStart')
        date_end_raw = item.get('dateEnd')

        if date_start_raw is not None:
            date_start = datetime.datetime.fromisoformat(date_start_raw)
            if date_start.tzinfo is None:
                date_start = date_start.replace(tzinfo=zone)
            if now < date_start:
                continue

        if date_end_raw is not None:
            date_end = datetime.datetime.fromisoformat(date_end_raw)
            if date_end.tzinfo is None:
                date_end = date_end.replace(tzinfo=zone)
            if now > date_end:
                continue

        text_block = item.get('text', {})
        lang_text = text_block.get(_lang) or text_block.get('en-CA', {})
        if not isinstance(lang_text, dict):
            continue

        parts: list[str] = []
        header = lang_text.get('header')
        body = lang_text.get('body')
        footer = lang_text.get('footer')
        if header:
            parts.append(str(header).strip())
        if body:
            parts.append(str(body).strip())
        if footer:
            parts.append(str(footer).strip())

        if parts:
            active.append(' '.join(parts))

    return '  '.join(active)


def current_conditions_package(
    weather_data: Optional[dict[str, Any]] = None,
    location_name: Optional[str] = None,
    lang: Optional[str] = "en-CA",
    secondary: bool = False,
) -> str:
    _lang = lang or 'en-CA'
    lang_short = _lang[:2]

    ph = _CC_PH.get(lang_short, _CC_PH['en'])

    if weather_data is None:
        if secondary:
            return ""
        return ph['unavailable'].format(name=location_name or ph['station'])

    props: dict[str, Any] = _d(weather_data, 'properties') or weather_data
    loc = _d(props, 'location')
    city_raw: Any = _s(loc, 'city')
    if isinstance(city_raw, dict):
        city_d: dict[str, Any] = cast(dict[str, Any], city_raw)
        city_name = str(city_d.get('en') or city_d.get('fr') or '')
    elif city_raw:
        city_name = str(city_raw)
    else:
        city_name = ''
    name: str = location_name or city_name or ph['station']

    condition_raw: Any = _s(props, 'condition')
    if isinstance(condition_raw, dict):
        cond_d: dict[str, Any] = cast(dict[str, Any], condition_raw)
        cond_text: Optional[str] = str(cond_d.get(lang_short) or cond_d.get('en') or '') or None
    elif condition_raw:
        cond_text = str(condition_raw)
    else:
        cond_text = None

    opener_text: Optional[str] = None
    _src_key = weather_data.get('source') if isinstance(weather_data, dict) else None
    _obs_at = weather_data.get('observed_at') if isinstance(weather_data, dict) else None
    if _src_key and _obs_at:
        _source_map = _CC_SOURCE.get(lang_short, _CC_SOURCE['en'])
        _source_name = _source_map.get(str(_src_key), str(_src_key).upper())
        try:
            _dt = datetime.datetime.fromisoformat(str(_obs_at).replace('Z', '+00:00'))
            _time_str = _dt.strftime("%-I:%M %p %Z")
        except ValueError:
            _time_str = str(_obs_at)
        opener_text = ph['opener'].format(source=_source_name, time=_time_str)

    sentences: list[str] = []
    if opener_text and not secondary:
        sentences.append(opener_text)

    if cond_text:
        sentences.append(ph['condition'].format(name=name, cond=cond_text.rstrip('.')))
    elif secondary:
        sentences.append(ph['secondary_no_cond'].format(name=name))
    else:
        sentences.append(ph['no_cond'].format(name=name))

    temp_val = _scalar(_d(props, 'temp') or _s(props, 'temp'))
    if temp_val is not None:
        sentences.append(ph['temp'].format(val=temp_val))
        sentences.append('.,....')
    elif not secondary:
        sentences.append(ph['no_temp'])

    dp_val = _scalar(_d(props, 'dewpoint') or _s(props, 'dewpoint'))
    if dp_val is not None:
        sentences.append(ph['dewpoint'].format(val=dp_val))

    hum_val = _scalar(_d(props, 'humidity') or _s(props, 'humidity'))
    if hum_val is not None:
        sentences.append(ph['humidity'].format(val=hum_val))
        if hum_val == 67:
            lol = pathlib.Path('audio', 'xiaomi-ringtone.mp3').resolve()
            sentences.append('\n' + lol.as_uri() + '\n')
    elif not secondary:
        sentences.append(ph['no_humidity'])

    wind = _d(props, 'wind')
    if wind:
        spd_val = _scalar(_d(wind, 'speed') or _s(wind, 'speed'))
        direction: Any = _s(wind, 'direction')
        gust_val = _scalar(_d(wind, 'gust') or _s(wind, 'gust'))
        if spd_val is not None:
            dir_text = ph['wind_dir'].format(dir=direction) if direction else ""
            sentences.append(ph['wind'].format(dir=dir_text, spd=spd_val))
            if gust_val:
                sentences.append(ph['gust'].format(val=gust_val))
        elif not secondary:
            sentences.append(ph['no_wind'])
    elif not secondary:
        sentences.append(ph['no_wind'])

    vis_val = _scalar(_d(props, 'visibility') or _s(props, 'visibility'))
    if vis_val is not None:
        sentences.append(ph['visibility'].format(val=vis_val))
    elif not secondary:
        sentences.append(ph['no_visibility'])

    pressure = _d(props, 'pressure')
    if pressure:
        pres_val: Any = _s(pressure, 'value')
        tendency_raw: Any = _s(pressure, 'tendency')
        if isinstance(tendency_raw, dict):
            tendency_d: dict[str, Any] = cast(dict[str, Any], tendency_raw)
            tendency: Optional[str] = str(tendency_d.get(lang_short) or tendency_d.get('en') or '') or None
        else:
            tendency = str(tendency_raw) if tendency_raw else None
        if pres_val is not None:
            tend_text = ph['tendency'].format(val=tendency) if tendency else ""
            sentences.append(ph['pressure'].format(val=pres_val, tend=tend_text))
        elif not secondary:
            sentences.append(ph['no_pressure'])
    elif not secondary:
        sentences.append(ph['no_pressure'])

    wc_val = _scalar(_d(props, 'windChill') or _s(props, 'windChill'))
    if wc_val is not None:
        sentences.append(ph['windchill'].format(val=wc_val))

    hx_val = _scalar(_d(props, 'humidex') or _s(props, 'humidex'))
    if hx_val is not None:
        sentences.append(ph['humidex'].format(val=hx_val))

    return " ".join(sentences)

def forecast_package(
    forecast_data: Optional[dict[str, Any]] = None,
    location_name: Optional[str] = None,
    lang: Optional[str] = "en-CA",
) -> str:
    _lang = lang or 'en-CA'
    lang_short = _lang[:2]
    fx = _FX_PH.get(lang_short, _FX_PH['en'])
    loc_name = location_name or fx['station']
    if forecast_data is None:
        return fx['unavailable'].format(name=loc_name)

    forecasts = forecast_data.get('forecast', [])
    if not forecasts:
        return fx['unavailable'].format(name=loc_name)

    sentences: list[str] = [fx['intro'].format(name=loc_name)]

    for period in forecasts[:6]:
        if not isinstance(period, dict):
            continue
        p: dict[str, Any] = cast(dict[str, Any], period)

        period_name_raw = p.get('period', {})
        if isinstance(period_name_raw, dict):
            pn: dict[str, Any] = cast(dict[str, Any], period_name_raw)
            period_name = str(pn.get(lang_short) or pn.get('en', ''))
        else:
            period_name = str(period_name_raw) if period_name_raw else ''

        text_raw = p.get('textSummary', {})
        if isinstance(text_raw, dict):
            ts: dict[str, Any] = cast(dict[str, Any], text_raw)
            text = str(ts.get(lang_short) or ts.get('en', ''))
        else:
            text = str(text_raw) if text_raw else ''

        if period_name and text:
            clean = text.rstrip('.')
            sentences.append(f"{period_name}. {clean}.")

    return " ".join(sentences)

def eccc_discussion_package(
    discussion_text: Optional[str] = None,
    location_name: Optional[str] = None,
    lang: Optional[str] = "en-CA",
) -> str:
    if not discussion_text:
        return ""

    lines = discussion_text.splitlines()

    paragraphs: list[str] = []
    current_para: list[str] = []

    in_body = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if in_body and current_para:
                paragraphs.append(_flush_discussion(current_para))
                current_para = []
            continue

        if _DISCUSSION_FOCN_RE.match(stripped):
            continue

        if _DISCUSSION_BYLINE_RE.match(stripped) or stripped.upper() == 'END':
            break

        m_geo = _DISCUSSION_GEO_HEADER_RE.match(stripped)
        m_sec = _DISCUSSION_SECTION_HEADER_RE.match(stripped)

        if m_sec and not m_geo:
            key = m_sec.group(1).strip()
            label = _DISCUSSION_SECTION_HEADERS.get(key)
            if label is None:
                label = key.capitalize() + '.'
            if in_body and current_para:
                paragraphs.append(_flush_discussion(current_para))
                current_para = []
            in_body = True
            paragraphs.append(label)
            rest = stripped[m_sec.end():].strip()
            if rest:
                current_para.append(rest)
            continue

        if m_geo:
            geo_label = _expand_geo_header(m_geo.group(1))
            if in_body and current_para:
                paragraphs.append(_flush_discussion(current_para))
                current_para = []
            in_body = True
            rest = stripped[m_geo.end():].strip()
            para_start = f"{geo_label}." if geo_label else ''
            if rest:
                current_para.append(para_start + ' ' + rest if para_start else rest)
            else:
                if para_start:
                    current_para.append(para_start)
            continue

        if not in_body:
            if stripped[0].isalpha() and len(stripped) > 20:
                in_body = True
            else:
                continue

        current_para.append(stripped)

    if current_para:
        paragraphs.append(_flush(current_para))

    if not paragraphs:
        return ""

    return '  '.join(paragraphs)

def geophysical_alert_package(
    wwv_text: Optional[str] = None,
) -> str:
    if wwv_text is None:
        return ""
    sentences = wwv_text.splitlines()
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if not s.startswith('#') if not s.startswith(':Product:') and not s.startswith(':Issued:') and not s.startswith('Prepared by')]
    return '  '.join(sentences)
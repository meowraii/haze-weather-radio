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
    'en': ["The current time is {time_str}."],
    'fr': ["Il est actuellement {time_str}."],
    'es': ["La hora actual es {time_str}."],
}

_TZ_NAMES: dict[str, str] = {
    'CDT': 'Central Daylight Time',
    'CST': 'Central Standard Time',
    'MDT': 'Mountain Daylight Time',
    'MST': 'Mountain Standard Time',
    'EDT': 'Eastern Daylight Time',
    'EST': 'Eastern Standard Time',
    'PDT': 'Pacific Daylight Time',
    'PST': 'Pacific Standard Time',
    'ADT': 'Atlantic Daylight Time',
    'AST': 'Atlantic Standard Time',
    'NDT': 'Newfoundland Daylight Time',
    'NST': 'Newfoundland Standard Time',
    'UTC': 'Coordinated Universal Time',
    'GMT': 'Greenwich Mean Time',
}

_TIME_RE = re.compile(
    r'\b(\d{1,2}):(\d{2})\s*(AM|PM)\s*([A-Z]{2,4})\b',
    re.IGNORECASE,
)

_ZULU_TIME_RE = re.compile(
    r'\b(\d{4})\s*(UTC|GMT)\b',
    re.IGNORECASE,
)


_ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
         'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
         'seventeen', 'eighteen', 'nineteen']
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty']


def _num_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return f'{_TENS[t]} {_ONES[o]}'.strip() if o else _TENS[t]


def _spoken_minute(mm: str) -> str:
    n = int(mm)
    if n == 0:
        return ''
    if n < 10:
        return f'oh {_num_to_words(n)}'
    return _num_to_words(n)


def _spoken_ampm(ampm: str) -> str:
    return 'a m' if ampm.upper() == 'AM' else 'p m'


def _spoken_zulu_hour(hh: int) -> str:
    if hh == 0:
        return 'zero zero'
    if hh < 10:
        return f'zero {_num_to_words(hh)}'
    return _num_to_words(hh)


def _spoken_tz(abbr: str) -> str:
    return _TZ_NAMES.get(abbr.upper(), abbr)


def _format_time_spoken(dt: datetime.datetime) -> str:
    hour = _num_to_words(int(dt.strftime('%-I')))
    minute = _spoken_minute(dt.strftime('%M'))
    ampm = _spoken_ampm(dt.strftime('%p'))
    tzname = _spoken_tz(dt.strftime('%Z'))
    if minute:
        return f"{hour} {minute} {ampm}, {tzname}"
    return f"{hour} {ampm}, {tzname}"


def _reformat_times_in_text(text: str) -> str:
    def _replace_ampm(m: re.Match[str]) -> str:
        hour = _num_to_words(int(m.group(1)))
        minute = _spoken_minute(m.group(2))
        ampm = _spoken_ampm(m.group(3))
        tz = _spoken_tz(m.group(4))
        if minute:
            return f"{hour} {minute} {ampm}, {tz}"
        return f"{hour} {ampm}, {tz}"
    def _replace_zulu(m: re.Match[str]) -> str:
        hhmm = m.group(1)
        h_str = _spoken_zulu_hour(int(hhmm[:2]))
        m_str = _spoken_minute(hhmm[2:]) or 'hundred'
        tz = _spoken_tz(m.group(2))
        return f"{h_str} {m_str}, {tz}"
    text = _TIME_RE.sub(_replace_ampm, text)
    return _ZULU_TIME_RE.sub(_replace_zulu, text)

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
        'windchill':         "With a wind chill of {val}.",
        'humidex':           "With a humidex of {val}.",
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
    'NW': 'northwestern',
    'NE': 'northeastern',
    'SW': 'southwestern',
    'SE': 'southeastern',
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
    joined = _reformat_times_in_text(joined)
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
            time_str=_format_time_spoken(now),
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


_ALERT_VERBS: dict[str, dict[str, str]] = {
    'en': {
        'Alert': 'has issued',
        'Update': 'has updated',
        'Cancel': 'has ended',
    },
    'fr': {
        'Alert': 'a émis',
        'Update': 'a mis à jour',
        'Cancel': 'a annulé',
    },
}

_AN_PREFIXES = ('a', 'e', 'i', 'o', 'u')

_FORECAST_LOC_DB: dict[str, tuple[str, str]] | None = None
_FORECAST_LOC_CSV = pathlib.Path(__file__).parent / 'FORECAST_LOCATIONS.csv'


def _load_forecast_loc_db() -> dict[str, tuple[str, str]]:
    global _FORECAST_LOC_DB
    if _FORECAST_LOC_DB is not None:
        return _FORECAST_LOC_DB
    db: dict[str, tuple[str, str]] = {}
    try:
        with open(_FORECAST_LOC_CSV, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3 and parts[0][:1].isdigit():
                    code = parts[0].strip()
                    if code not in db:
                        db[code] = (parts[1].strip(), parts[2].strip())
    except Exception:
        pass
    _FORECAST_LOC_DB = db
    return db


def _alert_param(params: list[dict[str, Any]], name: str) -> Any:
    for p in params:
        if p.get('valueName') == name:
            return p.get('value', '')
    return ''


def _format_date_spoken(dt: datetime.datetime, lang_short: str) -> str:
    day_name = _DT_DAYS.get(lang_short, _DT_DAYS['en'])[dt.weekday()]
    month_name = _DT_MONTHS.get(lang_short, _DT_MONTHS['en'])[dt.month - 1]
    ordinals = _DT_ORDINALS.get(lang_short, _DT_ORDINALS['en'])
    ordinal = ordinals[dt.day - 1] if dt.day <= len(ordinals) else str(dt.day)
    return f"{day_name}, {month_name} {ordinal}, {dt.year}"


def _resolve_alert_areas(
    params: list[dict[str, Any]],
    feed: dict[str, Any],
    lang_short: str,
) -> list[str]:
    raw = _alert_param(params, 'layer:EC-MSC-SMC:1.1:Newly_Active_Areas')
    if isinstance(raw, list):
        alert_clcs = [str(c).strip() for c in raw]
    elif isinstance(raw, str) and raw:
        alert_clcs = [c.strip() for c in raw.split(',')]
    else:
        return []

    feed_sames: list[str] = []
    for loc in feed.get('locations', []):
        same = loc.get('same')
        if same:
            feed_sames.append(str(same).strip())

    if not feed_sames:
        return []

    matched: list[str] = []
    for clc in alert_clcs:
        for same in feed_sames:
            if same.endswith('*'):
                if clc.startswith(same[:-1]):
                    matched.append(clc)
                    break
            elif clc == same:
                matched.append(clc)
                break

    if not matched:
        return []

    db = _load_forecast_loc_db()
    lang_idx = 1 if lang_short == 'fr' else 0
    seen: set[str] = set()
    names: list[str] = []
    for clc in matched:
        parent = clc[:4] + "00"
        if parent in db:
            if parent not in seen:
                seen.add(parent)
                name = db[parent][lang_idx].replace(' - ', ', ')
                names.append(name)
        elif clc in db:
            if clc not in seen:
                seen.add(clc)
                names.append(db[clc][lang_idx])
    return names


def alerts_package(
    registry: list[Any] | None = None,
    lang: str | None = "en-CA",
    tz: str | None = "UTC",
    feed: dict[str, Any] | None = None,
) -> str:
    if not registry:
        return ""

    _lang = lang or 'en-CA'
    lang_short = _lang[:2]
    zone = ZoneInfo(tz or 'UTC')
    now = datetime.datetime.now(tz=zone)
    verbs = _ALERT_VERBS.get(lang_short, _ALERT_VERBS['en'])

    parts: list[str] = []
    for entry in registry:
        if not isinstance(entry, dict):
            continue

        meta = entry.get('metadata', {})
        source = entry.get('source', {})
        params = entry.get('parameters', [])

        expires_raw = meta.get('expires')
        expires_dt: datetime.datetime | None = None
        if expires_raw:
            try:
                expires_dt = datetime.datetime.fromisoformat(expires_raw)
                if expires_dt.tzinfo is None:
                    expires_dt = expires_dt.replace(tzinfo=zone)
                if now > expires_dt:
                    continue
            except (ValueError, TypeError):
                pass

        sender_name = meta.get('senderName', '')
        msg_type = source.get('msgType', 'Alert')
        verb = verbs.get(msg_type, verbs['Alert'])

        alert_name = _alert_param(params, 'layer:EC-MSC-SMC:1.0:Alert_Name')
        if not alert_name:
            alert_name = meta.get('headline', meta.get('event', ''))
        alert_name_tc = str(alert_name).title()

        article = 'an' if alert_name_tc[:1].lower() in _AN_PREFIXES else 'a'

        coverage = _alert_param(params, 'layer:EC-MSC-SMC:1.0:Alert_Coverage')

        sentence = f"{sender_name} {verb} {article} {alert_name_tc}"
        if coverage:
            sentence += f", for {coverage}"
        sentence += "."

        onset_raw = meta.get('onset') or meta.get('effective')
        if onset_raw:
            try:
                onset_dt = datetime.datetime.fromisoformat(onset_raw).astimezone(zone)
                sentence += f" Starting {_format_date_spoken(onset_dt, lang_short)}."
            except (ValueError, TypeError):
                pass

        if expires_dt:
            exp_local = expires_dt.astimezone(zone)
            areas = _resolve_alert_areas(params, feed, lang_short) if feed else []
            exp_str = f"Until {_format_date_spoken(exp_local, lang_short)}"
            if areas:
                exp_str += f", for {', '.join(areas)}"
            exp_str += "."
            sentence += " " + exp_str

        text_block = entry.get('text', {})
        desc = (text_block.get('description') or '').strip()
        instr = (text_block.get('instruction') or '').strip()
        if desc:
            sentence += " " + desc
        if instr:
            sentence += " " + instr

        parts.append(sentence)

    if not parts:
        return ""

    return '  '.join(parts)


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
            _time_str = _format_time_spoken(_dt)
        except ValueError:
            _time_str = _reformat_times_in_text(str(_obs_at))
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

    wc_val = _scalar(_d(props, 'windChill') or _s(props, 'windChill'))
    if wc_val is not None:
        sentences.append(ph['windchill'].format(val=wc_val))

    hx_val = _scalar(_d(props, 'humidex') or _s(props, 'humidex'))
    if hx_val is not None:
        sentences.append(ph['humidex'].format(val=hx_val))

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
    sentences = [_reformat_times_in_text(s) for s in sentences]
    return '  '.join(sentences)
from dataclasses import dataclass
import datetime
import os
import pathlib
import re
from typing import Any, ClassVar, Optional, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


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
    return 'A.M.' if ampm.upper() == 'AM' else 'P.M.'


def _spoken_zulu_hour(hh: int) -> str:
    if hh == 0:
        return 'zero zero'
    if hh < 10:
        return f'zero {_num_to_words(hh)}'
    return _num_to_words(hh)


def _spoken_tz(abbr: str) -> str:
    return _TZ_NAMES.get(abbr.upper(), abbr)


def _spoken_hour(dt: datetime.datetime) -> str:
    return _num_to_words((dt.hour % 12) or 12)


def _format_time_spoken(dt: datetime.datetime) -> str:
    hour = _spoken_hour(dt)
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
        'repeating':         "Again, the weather at {name} was {cond}.",
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

_FC_PH: dict[str, dict[str, str]] = {
    'en': {
        'eccc_opener': "Your official Environment Canada forecast for the {name} region issued at {time}.",
        'eccc_unavailable': "The forecast for the {name} region is unavailable at this time.",
        'nws_opener': "Now for your official National Weather Service forecast for {name}, issued at {time}.",
        'twc_opener': "The forecast, courtesy of The Weather Channel, for {name} issued at {time}.",
        'generic_unavailable': "The forecast for {name} is unavailable at this time.",
        'generic_opener': "The forecast for {name} issued at {time}.",
    },
    'fr': {
        'eccc_opener': "Votre prévision officielle d'Environnement Canada pour la région de {name} émise à {time}.",
        'eccc_unavailable': "La prévision pour la région de {name} n'est pas disponible pour le moment.",
        'nws_opener': "Voici votre prévision officielle du Service météorologique national pour {name}, émise à {time}.",
        'twc_opener': "La prévision, courtoisie de The Weather Channel, pour {name} émise à {time}.",
        'generic_unavailable': "La prévision pour {name} n'est pas disponible pour le moment.",
        'generic_opener': "La prévision pour {name} émise à {time}.",
    },
    'es': {
        'eccc_opener': "Su pronóstico oficial de Environment Canada para la región de {name} emitido a las {time}.",
        'eccc_unavailable': "El pronóstico para la región de {name} no está disponible en este momento.",
        'nws_opener': "Ahora para su pronóstico oficial del Servicio Meteorológico Nacional para {name}, emitido a las {time}.",
        'twc_opener': "El pronóstico, cortesía de The Weather Channel, para {name} emitido a las {time}.",
        'generic_unavailable': "El pronóstico para {name} no está disponible en este momento.",
        'generic_opener': "El pronóstico para {name} emitido a las {time}.",
    },
}


_AE_PH: dict[str, dict[str, str]] = {
    'en': {
        'opener': "The {name} climate summary for {period}. As of {time}, {date}. Courtesy of {source}.",
        'unavailable': "Climate information for {name} is unavailable at this time.",
        'station': "this area",
        'latest_record': "This reflects the latest finalized daily climate record, from {date}.",
        'high': "Today's high temperature was {val} degrees Celsius.",
        'low': "And the low was {val} degrees Celsius.",
        'mean_temp': "The average temperature on this day was {val} degrees Celsius.",
        'temp_extremes': "The record high for this date is {high} degrees, set in {high_year}, and the record low is {low} degrees, set in {low_year}.",
        'cold_high': "The coldest high was {value}, set in {year}.",
        'warm_low': "The warmest low was {value}, set in {year}.",
        'avg_temp': "The average temperature on this day, was {val} degrees Celsius.",
        'temp_high_than_avg': "Today's high was {diff} degrees higher than the average.",
        'temp_low_than_avg': "Today's low was {diff} degrees lower than the average.",
        'temp_equal_avg': "Today's high and low were equal to the average. It seems like the weather boy in fact did like to know.",
        'normals': "The normals for this current period, the high is {high} and the normal low is {low}.",
        'precip_yea': "About {val} millimetres of precipitation has fallen today.",
        'precip_no': "No precipitation has fallen today.",
        'snow_yea': "About {val} centimetres of snow has fallen today.",
        'snow_no': "No snow has fallen today.",
        'precip_extreme': "The most precipitation to fall on this date was {val} millimetres, set in {year}.",
        'snow_extreme': "The most snowfall to fall on this date was {val} centimetres, set in {year}.",
        'snow_if_summer': "No snow has fallen today. Which would be a little bit concerning, because it is summer.",
        'snow_on_ground': "There were {val} centimetres of snow on the ground.",
        'sunset_tonight': "The sun will set at {time} tonight.",
        'sunrise_tomorrow': "The sun will rise at {time} tomorrow.",
        
    },
    'fr': {
        'opener': "Le résumé climatique de {name} pour {period}. En date du {date} à {time}. Courtoisie de {source}.",
        'unavailable': "Les informations climatiques pour {name} ne sont pas disponibles pour le moment.",
        'station': "cette zone",
        'latest_record': "Ceci reflète le plus récent relevé climatique quotidien finalisé, en date du {date}.",
        'high': "La température maximale d'aujourd'hui était de {val} degrés Celsius.",
        'low': "Et la minimale était de {val} degrés Celsius.",
        'temp_extremes': "Le record de chaleur pour cette date est de {high} degrés, établi en {high_year}, et le record de froid est de {low} degrés, établi en {low_year}.",
        'cold_high': "La température maximale la plus basse a été de {value}, établie en {year}.",
        'warm_low': "La température minimale la plus élevée a été de {value}, établie en {year}.",
        'mean_temp': "La température moyenne de cette journée était de {val} degrés Celsius.",
        'normals': "Les normales pour cette période actuelle, la maximale est de {high} et la minimale normale est de {low}.",
        'precip_yea': "Environ {val} millimètres de précipitations sont tombés aujourd'hui.",
        'precip_no': "Aucune précipitation n'est tombée aujourd'hui.",
        'snow_yea': "Environ {val} centimètres de neige sont tombés aujourd'hui.",
        'snow_no': "Aucune neige n'est tombée aujourd'hui.",
        'precip_extreme': "La plus grande quantité de précipitations à tomber à cette date était de {val} millimètres, établie en {year}.",
        'snow_extreme': "La plus grande quantité de neige à tomber à cette date était de {val} centimètres, établie en {year}.",
        'snow_if_summer': "Aucune neige n'est tombée aujourd'hui. Ce qui serait un peu inquiétant, car c'est l'été.",
        'snow_on_ground': "Il y avait {val} centimètres de neige au sol.",
        'sunset_tonight': "Le soleil se couchera à {time} ce soir.",
        'sunrise_tomorrow': "Le soleil se lèvera à {time} demain.",
    },
    'es': {
        'opener': "El resumen climático de {name} para {period}. A fecha de {date} a las {time}. Cortesía de {source}.",
        'unavailable': "La información climática para {name} no está disponible en este momento.",
        'station': "esta zona",
        'latest_record': "Esto refleja el registro climático diario finalizado más reciente, del {date}.",
        'high': "La temperatura máxima de hoy fue de {val} grados Celsius.",
        'low': "Y la mínima fue de {val} grados Celsius.",
        'mean_temp': "La temperatura media de este día fue de {val} grados Celsius.",
        'temp_extremes': "El récord de calor para esta fecha es de {high} grados, establecido en {high_year}, y el récord de frío es de {low} grados, establecido en {low_year}.",
        'cold_high': "La máxima más baja fue de {value}, establecida en {year}.",
        'warm_low': "La mínima más alta fue de {value}, establecida en {year}.",
        'normals': "Las normales para este período actual, la máxima es de {high} y la mínima normal es de {low}.",
        'precip_yea': "Alrededor de {val} milímetros de precipitación ha caído hoy.",
        'precip_no': "No ha caído precipitación hoy.",
        'snow_yea': "Alrededor de {val} centímetros de nieve ha caído hoy.",
        'snow_no': "No ha caído nieve hoy.",
        'precip_extreme': "La mayor cantidad de precipitación que ha caído en esta fecha fue de {val} milímetros, establecida en {year}.",
        'snow_extreme': "La mayor cantidad de nieve que cayó en esta fecha fue de {val} centímetros, establecida en {year}.",
        'snow_if_summer': "No ha caído nieve hoy. Lo cual sería un poco preocupante, porque es verano.",
        'snow_on_ground': "Había {val} centímetros de nieve en el suelo.",
        'sunset_tonight': "El sol se pondrá a las {time} esta noche.",
        'sunrise_tomorrow': "El sol saldrá a las {time} mañana.",
    },
}
_AQI_PH: dict[str, dict[str, str]] = {
    'en': {
        'opener_twc': "The global air quality report for {name}, powered by Copernicus Atmosphere Monitoring Service Information {copyyear} and provided by The Weather Channel. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus Information or Data it contains",
        'unavailable_report': "The air quality information for {name} was unavailable.",
        'now_eccc': "The air quality health index was observed at {name} and reported a value of {val} at {time}.",
        'low_eccc': "This is ideal air quality for outdoor activities for both at-risk and general populations.",
        'moderate_eccc': "This is acceptable air quality for outdoor activities for most people. However, at-risk individuals should consider reducing or rescheduling strenuous outdoor activities if symptoms occur.",
        'high_eccc': "This is unhealthy air quality for at-risk individuals, which include children, seniors, and those with pre-existing respiratory or heart conditions. At-risk individuals should reduce or reschedule strenuous outdoor activities. The general population is not likely to be affected.",
        'very_eccc': "I may not be a detective, but it is probably hazardous for most people to be outdoors right now. At-risk individuals should avoid strenuous outdoor activities. Everyone else should also reduce or reschedule strenuous outdoor activities.",
        #'super_frickin_high_eccc': "Attention! Breaking news! The air quality health index has reached a never-before-seen level of {val} at {name}! This is an unprecidented event, general population should avoid being outside at all costs, and at-risk individuals WILL EXPLODE if they even look out the smoggy window. If you have to go outside in case of an emergency, make sure to wear a Hazardous EnVironment suit from Half-Life 2, a facemask straight out of 2020, a canister of oxygen, and a personal air purifier from TEMU, and even then, general population individuals will experience a severe case of literally melting to the ground, and at-risk individuals, if not exploded already, might literally diffuse between plains of reality, ultimately causing a supernova that disperses atoms of themselves into various parallel universes, rendering search and rescue operations probably a little difficult. Stay safe out there, folks.",
        'forecast_opener_eccc': "The air quality health index forecast for {name} is {val} for {period_name} and is considered {risk}.",
        'forecast_eccc': "For {period_name}, the maximum air quality health index is forecast to be {val}, or {risk}.",
        'forecast_trailing_eccc': "{period2_val} on {period2_name}, and lastly, {period3_val} on {period3_name}.",
        'forecast_insmoke_eccc': "The air quality health index is expected to be {val} in smoke."
    },
    'fr': {
        'opener_twc': "Le rapport mondial sur la qualité de l'air pour {name}, propulsé par le Service de surveillance de l'atmosphère Copernicus Information {copyyear} et fourni par The Weather Channel. Ni la Commission européenne ni le Centre européen pour les prévisions météorologiques à moyen terme (CEPMMT) ne sont responsables de toute utilisation qui pourrait être faite des informations ou des données de Copernicus qu'il contient.",
        'unavailable_report': "Les informations sur la qualité de l'air pour {name} n'étaient pas disponibles.",
        'now_eccc': "L'indice de santé de la qualité de l'air a été observé à {name} et a rapporté une valeur de {val} à {time}.",
        'low_eccc': "C'est une qualité d'air idéale pour les activités de plein air, tant pour les populations à risque que pour le grand public.",
        'moderate_eccc': "C'est une qualité d'air acceptable pour les activités de plein air pour la plupart des gens. Cependant, les personnes à risque devraient envisager de réduire ou de reprogrammer les activités de plein air intenses si des symptômes surviennent.",
        'high_eccc': "C'est une qualité d'air malsaine pour les personnes à risque, qui comprennent les enfants, les personnes âgées et celles ayant des conditions respiratoires ou cardiaques préexistantes. Les personnes à risque devraient réduire ou reprogrammer les activités de plein air intenses. Le grand public n'est probablement pas affecté.",
        'very_eccc': "Je ne suis peut-être pas un détective, mais il est probablement dangereux pour la plupart des gens d'être à l'extérieur en ce moment. Les personnes à risque devraient éviter les activités de plein air intenses. Tout le monde devrait également réduire ou reprogrammer les activités de plein air intenses.",
        'forecast_opener_eccc': "La prévision de l'indice de santé de la qualité de l'air pour {name} est de {val} pour {period_name} et est considérée comme {risk}.",
        'forecast_eccc': "Pour {period_name}, l'indice de santé de la qualité de l'air maximum est prévu à {val}, ou {risk}.",
        'forecast_trailing_eccc': "{period2_val} pour {period2_name}, et enfin, {period3_val} pour {period3_name}.",
        'forecast_insmoke_eccc': "L'indice de santé de la qualité de l'air devrait être de {val} en présence de fumée.",
    },
    'es': {
        'opener_twc': "El informe global de calidad del aire para {name}, impulsado por el Servicio de Monitoreo de la Atmósfera Copernicus Information {copyyear} y proporcionado por The Weather Channel. Ni la Comisión Europea ni el ECMWF son responsables de cualquier uso que se pueda hacer de la Información o los Datos de Copernicus que contiene.",
        'unavailable_report': "La información sobre la calidad del aire para {name} no estaba disponible.",
        'now_eccc': "El índice de salud de la calidad del aire se observó en {name} y reportó un valor de {val} a las {time}.",
        'low_eccc': "Esta es una calidad del aire ideal para actividades al aire libre tanto para poblaciones en riesgo como para el público en general.",
        'moderate_eccc': "Esta es una calidad del aire aceptable para actividades al aire libre para la mayoría de las personas. Sin embargo, las personas en riesgo deben considerar reducir o reprogramar las actividades al aire libre extenuantes si ocurren síntomas.",
        'high_eccc': "Esta es una calidad del aire insalubre para personas en riesgo, que incluyen niños, ancianos y aquellos con condiciones respiratorias o cardíacas preexistentes. Las personas en riesgo deben reducir o reprogramar las actividades al aire libre extenuantes. El público en general no es probable que se vea afectado.",
        'very_eccc': "Puede que no sea un detective, pero probablemente sea peligroso para la mayoría de las personas estar al aire libre en este momento. Las personas en riesgo deben evitar las actividades al aire libre extenuantes. Todos los demás también deberían reducir o reprogramar las actividades al aire libre extenuantes.",
        'forecast_opener_eccc': "El pronóstico del índice de salud de la calidad del aire para {name} es {val} para {period_name} y se considera {risk}.",
        'forecast_eccc': "Para {period_name}, el índice máximo de salud de la calidad del aire se pronostica en {val}, o {risk}.",
        'forecast_trailing_eccc': "{period2_val} para {period2_name}, y finalmente, {period3_val} para {period3_name}.",
        'forecast_insmoke_eccc': "Se espera que el índice de salud de la calidad del aire sea {val} en presencia de humo.",
    },
}

_AQHI_RISK_LABELS: dict[str, dict[str, str]] = {
    'en': {'low': 'Low', 'moderate': 'Moderate', 'high': 'High', 'very_high': 'Very High'},
    'fr': {'low': 'Faible', 'moderate': 'Modéré', 'high': 'Élevé', 'very_high': 'Très élevé'},
    'es': {'low': 'Bajo', 'moderate': 'Moderado', 'high': 'Alto', 'very_high': 'Muy alto'},
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
        'air_quality': 3600.0,
        'climate_summary': 86400.0,
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
        'climate_summary': {
            'use_voice': 'female',
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
    callsign = feed.get('callsign') if feed else None
    frequency = feed['output']['PiFmAdv'].get('frequency') if feed else None
    if feed is None:
        return "Feed not found."

    desc_block: dict[str, Any] = _d(feed, 'description')
    desc: dict[str, Any] = _d(desc_block, _lang)

    sentences: list[str] = []

    operator_name = operator.get('operator_name')
    on_air_name = operator.get('on_air_name')
    email = operator.get('email')
    phone = operator.get('phone')

    email_clean = str(email).strip() if email and str(email) != 'None' else None
    email_clean = email_clean.replace('.', ' dot ').replace('@', ' at ') if email_clean else None

    phone_clean = str(phone).strip() if phone and str(phone) != 'None' else None
    phone_clean = re.sub(r'\D', ' ', phone_clean) if phone_clean else None

    if on_air_name:
        if 'en' in _lang:
            sentences.append(f"You are listening to {on_air_name}.")
        elif 'fr' in _lang:
            sentences.append(f"Vous écoutez {on_air_name}.")
        elif 'es' in _lang:
            sentences.append(f"Está escuchando {on_air_name}.")
    if callsign:
        if 'en' in _lang:
            sentences.append(f"Callsign {callsign}.")
        elif 'fr' in _lang:
            sentences.append(f"Indicatif d'appel {callsign}.")
        elif 'es' in _lang:
            sentences.append(f"Indicativo de llamada {callsign}.")
    if frequency and feed['output']['PiFmAdv'].get('enabled', False) is True:
        if 'en' in _lang:
            sentences.append(f"Broadcasting on a frequency of {frequency} megahertz.")
        elif 'fr' in _lang:
            sentences.append(f"Diffusion sur une fréquence de {frequency} mégahertz.")
        elif 'es' in _lang:
            sentences.append(f"Transmitiendo en una frecuencia de {frequency} megahercios.")
    if operator_name:
        if 'en' in _lang:
            sentences.append(f"This service is operated by {operator_name}.")
        elif 'fr' in _lang:
            sentences.append(f"Ce service est opéré par {operator_name}.")
        elif 'es' in _lang:
            sentences.append(f"Este servicio es operado por {operator_name}.")

    if desc:
        text = _s(desc, 'text')
        suffix = _s(desc, 'suffix')
        if text:
            sentences.append(str(text))
        if suffix:
            sentences.append(str(suffix))
    
    if email_clean and phone_clean:
        if 'en' in _lang:
            sentences.append(f"You can contact us by email at {email_clean}, or by phone at {phone_clean}.")
        elif 'fr' in _lang:
            sentences.append(f"Vous pouvez nous contacter par email à {email_clean}, ou par téléphone au {phone_clean}.")
        elif 'es' in _lang:
            sentences.append(f"Puede contactarnos por correo electrónico a {email_clean}, o por teléfono al {phone_clean}.")
    elif email_clean and not phone_clean:
        if 'en' in _lang:
            sentences.append(f"You can contact us by email at {email_clean}.")
        elif 'fr' in _lang:
            sentences.append(f"Vous pouvez nous contacter par email à {email_clean}.")
        elif 'es' in _lang:
            sentences.append(f"Puede contactarnos por correo electrónico a {email_clean}.")
    elif phone_clean and not email_clean:
        if 'en' in _lang:
            sentences.append(f"You can contact us by phone at {phone_clean}.")
        elif 'fr' in _lang:
            sentences.append(f"Vous pouvez nous contacter par téléphone au {phone_clean}.")
        elif 'es' in _lang:
            sentences.append(f"Puede contactarnos por teléfono al {phone_clean}.")

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
        'Cancel': 'has cancelled',
    },
    'fr': {
        'Alert': 'a émis',
        'Update': 'a mis à jour',
        'Cancel': 'a annulé',
    },
    'es': {
        'Alert': 'ha emitido',
        'Update': 'ha actualizado',
        'Cancel': 'ha cancelado',
    },
}

_AN_PREFIXES = ('a', 'e', 'i', 'o', 'u')

_FORECAST_LOC_DB: dict[str, tuple[str, str]] | None = None
_FORECAST_LOC_CSV = pathlib.Path(__file__).parent / 'FORECAST_LOCATIONS.csv'


def _load_forecast_loc_db() -> dict[str, tuple[str, str]]:
    global _FORECAST_LOC_DB
    if _FORECAST_LOC_DB is not None:
        return _FORECAST_LOC_DB
    import csv as _csv
    db: dict[str, tuple[str, str]] = {}
    try:
        with open(_FORECAST_LOC_CSV, newline='', encoding='utf-8-sig') as f:
            reader = _csv.reader(f)
            next(reader, None)  # row 0: BOM + title row
            header = next(reader, None)  # row 1: column headers
            if header is None:
                _FORECAST_LOC_DB = db
                return db
            h = [c.strip().upper() for c in header]
            code_idx, name_idx, nom_idx = h.index('CODE'), h.index('NAME'), h.index('NOM')
            for row in reader:
                if len(row) <= max(code_idx, name_idx, nom_idx):
                    continue
                code = row[code_idx].strip().strip('"')
                if code and code not in db:
                    db[code] = (row[name_idx].strip().strip('"'), row[nom_idx].strip().strip('"'))
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
    raw_param = _alert_param(params, 'layer:EC-MSC-SMC:1.1:Newly_Active_Areas')
    if isinstance(raw_param, list):
        alert_clcs = [str(c).strip() for c in raw_param if str(c).strip()]
    elif isinstance(raw_param, str) and raw_param:
        alert_clcs = [c.strip() for c in raw_param.split(',') if c.strip()]
    else:
        return []

    feed_patterns: list[str] = []
    for block in feed.get('locations', []):
        if not isinstance(block, dict):
            continue
        for loc in block.get('forecastLocations', []):
            if not isinstance(loc, dict):
                continue
            explicit = str(loc.get('same') or '').strip()
            if explicit:
                feed_patterns.extend(p.strip() for p in explicit.split(',') if p.strip())
            else:
                raw_fc = str(loc.get('forecast_region') or '').strip()
                if not raw_fc:
                    continue
                if '-' in raw_fc:
                    left, right = raw_fc.split('-', 1)
                    if left.replace('*', '').isdigit() and right.replace('*', '').isdigit():
                        raw_fc = right
                feed_patterns.extend(p.strip() for p in raw_fc.split(',') if p.strip())

    if not feed_patterns:
        return []

    seen: set[str] = set()
    matched: list[str] = []
    for clc in alert_clcs:
        for pattern in feed_patterns:
            if pattern.endswith('*'):
                if clc.startswith(pattern[:-1]) and clc not in seen:
                    seen.add(clc)
                    matched.append(clc)
                    break
            elif clc == pattern and clc not in seen:
                seen.add(clc)
                matched.append(clc)
                break

    if not matched:
        return []

    db = _load_forecast_loc_db()
    lang_idx = 1 if lang_short == 'fr' else 0
    seen_names: set[str] = set()
    names: list[str] = []
    for clc in matched:
        entry = db.get(clc)
        if entry:
            name = entry[lang_idx].replace(' - ', ', ')
            if name and name not in seen_names:
                seen_names.add(name)
                names.append(name)
    return names


_ECCC_BOILERPLATE_RE = re.compile(r'\s*###.*', re.DOTALL)
_ECCC_MULTISPACE_RE = re.compile(r'\n{2,}')

_AL_PH: dict[str, dict[str, str]] = {
    'en': {
        'opener_1':          "Attention: the following alert is currently in effect.",
        'opener':            "Attention: the following {count} alerts are currently in effect.",
        'eccc_issued':       "{sender} has issued a {subject} for {areas}.",
        'eccc_updated':      "{sender} has updated the {subject} for {areas}.",
        'eccc_cancelled':    "{sender} has cancelled the {subject} for {areas}.",
        'eccc_ended':        "The {subject} for {areas} has now ended.",
        'timing_span':       "In effect from {onset} through {expires}.",
        'timing_expires':    "In effect until {expires}.",
        'timing_onset':      "Beginning {onset}.",
        'confidence_impact': "Forecast confidence is {confidence}, with {impact} impacts expected.",
        'nws_header':        "{sender} has issued a {event} for {areas}.",
        'civil_issued':      "{sender} has issued a {event} for {areas}.",
        'civil_updated':     "{sender} has updated the {event} for {areas}.",
        'civil_cancelled':   "{sender} has cancelled the {event} for {areas}.",
        'civil_timing':      "In effect until {expires}.",
        'generic_issued':    "{sender} has issued a {event}.",
        'generic_expires':   "In effect until {datetime}.",
        'list_and':          "and",
    },
    'fr': {
        'opener_1':          "Attention : l'alerte suivante est actuellement en vigueur.",
        'opener':            "Attention : les {count} alertes suivantes sont actuellement en vigueur.",
        'eccc_issued':       "{sender} a émis un {subject} pour {areas}.",
        'eccc_updated':      "{sender} a mis à jour le {subject} pour {areas}.",
        'eccc_cancelled':    "{sender} a annulé le {subject} pour {areas}.",
        'eccc_ended':        "Le {subject} pour {areas} est maintenant terminé.",
        'timing_span':       "En vigueur du {onset} jusqu'au {expires}.",
        'timing_expires':    "En vigueur jusqu'au {expires}.",
        'timing_onset':      "À compter du {onset}.",
        'confidence_impact': "Niveau de confiance : {confidence}. Impacts prévus : {impact}.",
        'nws_header':        "{sender} a émis un {event} pour {areas}.",
        'civil_issued':      "{sender} a émis un {event} pour {areas}.",
        'civil_updated':     "{sender} a mis à jour le {event} pour {areas}.",
        'civil_cancelled':   "{sender} a annulé le {event} pour {areas}.",
        'civil_timing':      "En vigueur jusqu'au {expires}.",
        'generic_issued':    "{sender} a émis un {event}.",
        'generic_expires':   "En vigueur jusqu'au {datetime}.",
        'list_and':          "et",
    },
    'es': {
        'opener_1':          "Atención: la siguiente alerta está actualmente en vigor.",
        'opener':            "Atención: las siguientes {count} alertas están actualmente en vigor.",
        'eccc_issued':       "{sender} ha emitido un {subject} para {areas}.",
        'eccc_updated':      "{sender} ha actualizado el {subject} para {areas}.",
        'eccc_cancelled':    "{sender} ha cancelado el {subject} para {areas}.",
        'eccc_ended':        "El {subject} para {areas} ha finalizado.",
        'timing_span':       "En vigor desde {onset} hasta {expires}.",
        'timing_expires':    "En vigor hasta {expires}.",
        'timing_onset':      "A partir de {onset}.",
        'confidence_impact': "Confianza del pronóstico: {confidence}. Impactos esperados: {impact}.",
        'nws_header':        "{sender} ha emitido un {event} para {areas}.",
        'civil_issued':      "{sender} ha emitido un {event} para {areas}.",
        'civil_updated':     "{sender} ha actualizado el {event} para {areas}.",
        'civil_cancelled':   "{sender} ha cancelado el {event} para {areas}.",
        'civil_timing':      "En vigor hasta {expires}.",
        'generic_issued':    "{sender} ha emitido un {event}.",
        'generic_expires':   "En vigor hasta {datetime}.",
        'list_and':          "y",
    },
}


def _clean_alert_text(text: str) -> str:
    text = _ECCC_BOILERPLATE_RE.sub('', text)
    text = _ECCC_MULTISPACE_RE.sub(' ', text)
    text = _reformat_times_in_text(text)
    return text.strip()


def _detect_alert_source(params: list[dict[str, Any]], sender_name: str) -> str:
    for p in params:
        if p.get('valueName', '').startswith('layer:EC-MSC-SMC'):
            return 'eccc'
    sender_lower = sender_name.lower()
    if 'weather.gov' in sender_lower or 'national weather service' in sender_lower:
        return 'nws'
    for p in params:
        if p.get('valueName', '').upper() == 'EAS-ORG':
            return 'nws'
    return 'civil'


def _parse_eccc_subject(alert_name: str, colour: str, alert_type: str) -> str:
    name = alert_name.strip()
    if colour and alert_type:
        prefix = f"{colour} {alert_type} - "
        if name.lower().startswith(prefix.lower()):
            event = name[len(prefix):].strip().title()
            return f"{colour.title()} {event} {alert_type.title()}"
    return name.title()


def _join_areas(areas: list[str], lang_short: str) -> str:
    conj = {'fr': 'et', 'es': 'y'}.get(lang_short, 'and')
    if not areas:
        return ''
    if len(areas) == 1:
        return areas[0]
    if len(areas) == 2:
        return f"{areas[0]} {conj} {areas[1]}"
    return f"{', '.join(areas[:-1])}, {conj} {areas[-1]}"


def _format_datetime_spoken(dt: datetime.datetime, lang_short: str) -> str:
    date_part = _format_date_spoken(dt, lang_short)
    time_part = _format_time_spoken(dt)
    if lang_short == 'fr':
        return f"{date_part} à {time_part}"
    if lang_short == 'es':
        return f"{date_part} a las {time_part}"
    return f"{date_part}, at {time_part}"


def _civil_area_desc(entry: dict[str, Any]) -> str:
    for area in entry.get('areas', []):
        if isinstance(area, dict):
            desc = area.get('areaDesc', '').strip()
            if desc:
                return desc
    return ''

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
    ph = _AL_PH.get(lang_short, _AL_PH['en'])

    rendered: list[str] = []

    for entry in registry:
        if not isinstance(entry, dict):
            continue

        meta = entry.get('metadata', {})
        source_meta = entry.get('source', {})
        params = entry.get('parameters', [])

        expires_raw = meta.get('expires')
        expires_dt: datetime.datetime | None = None
        if expires_raw:
            try:
                expires_dt = datetime.datetime.fromisoformat(expires_raw).astimezone(zone)
                if now > expires_dt:
                    continue
            except (ValueError, TypeError):
                pass

        sender_name = meta.get('senderName', '')
        msg_type = source_meta.get('msgType', 'Alert')
        source_type = _detect_alert_source(params, sender_name)

        text_block = entry.get('text', {})
        description = _clean_alert_text(text_block.get('description') or '')
        instruction = _clean_alert_text(text_block.get('instruction') or '')

        sentences: list[str] = []

        if source_type == 'eccc':
            alert_name = str(_alert_param(params, 'layer:EC-MSC-SMC:1.0:Alert_Name') or '')
            colour     = str(_alert_param(params, 'layer:EC-MSC-SMC:1.1:Colour') or '')
            alert_type = str(_alert_param(params, 'layer:EC-MSC-SMC:1.0:Alert_Type') or '')
            coverage   = str(_alert_param(params, 'layer:EC-MSC-SMC:1.0:Alert_Coverage') or meta.get('event', ''))
            confidence = str(_alert_param(params, 'layer:EC-MSC-SMC:1.1:MSC_Confidence') or '')
            impact     = str(_alert_param(params, 'layer:EC-MSC-SMC:1.1:MSC_Impact') or '')
            loc_status = str(
                _alert_param(params, 'layer:EC-MSC-SMC:1.1:Alert_Location_Status') or
                _alert_param(params, 'layer:EC-MSC-SMC:1.0:Alert_Location_Status') or ''
            ).lower()

            subject = _parse_eccc_subject(alert_name, colour, alert_type) if alert_name else meta.get('event', '').title()

            areas = _resolve_alert_areas(params, feed, lang_short) if feed else []
            area_str = _join_areas(areas, lang_short) if areas else coverage

            if loc_status == 'ended':
                sentences.append(ph['eccc_ended'].format(subject=subject, areas=area_str))
            elif msg_type == 'Cancel':
                sentences.append(ph['eccc_cancelled'].format(sender=sender_name, subject=subject, areas=area_str))
            elif msg_type == 'Update':
                sentences.append(ph['eccc_updated'].format(sender=sender_name, subject=subject, areas=area_str))
            else:
                sentences.append(ph['eccc_issued'].format(sender=sender_name, subject=subject, areas=area_str))

            if loc_status != 'ended' and msg_type != 'Cancel':
                onset_dt_eccc: datetime.datetime | None = None
                onset_raw = meta.get('onset') or meta.get('effective')
                if onset_raw:
                    try:
                        onset_dt_eccc = datetime.datetime.fromisoformat(onset_raw).astimezone(zone)
                    except (ValueError, TypeError):
                        pass

                onset_future = onset_dt_eccc is not None and onset_dt_eccc > now
                if onset_future and expires_dt:
                    sentences.append(ph['timing_span'].format(
                        onset=_format_datetime_spoken(onset_dt_eccc, lang_short),
                        expires=_format_datetime_spoken(expires_dt, lang_short),
                    ))
                elif expires_dt:
                    sentences.append(ph['timing_expires'].format(expires=_format_datetime_spoken(expires_dt, lang_short)))
                elif onset_future:
                    sentences.append(ph['timing_onset'].format(onset=_format_datetime_spoken(onset_dt_eccc, lang_short)))

                if confidence and impact:
                    sentences.append(ph['confidence_impact'].format(confidence=confidence, impact=impact))

                if description:
                    sentences.append(description)
                if instruction:
                    sentences.append(instruction)

        elif source_type == 'nws':
            event_name = str(meta.get('event') or '').title()
            nws_area = _civil_area_desc(entry)
            nws_sender = sender_name or 'The National Weather Service'
            if nws_area:
                sentences.append(ph['nws_header'].format(sender=nws_sender, event=event_name, areas=nws_area))
            elif event_name:
                sentences.append(ph['generic_issued'].format(sender=nws_sender, event=event_name))
            if expires_dt:
                sentences.append(ph['timing_expires'].format(expires=_format_datetime_spoken(expires_dt, lang_short)))
            if description:
                sentences.append(description)
            if instruction:
                sentences.append(instruction)

        else:
            event_name = str(meta.get('event') or meta.get('headline') or '').title()
            civil_area = _civil_area_desc(entry)
            civil_sender = sender_name or 'Alert Ready'
            area_str_civil = civil_area or event_name

            if msg_type == 'Cancel':
                sentences.append(ph['civil_cancelled'].format(sender=civil_sender, event=event_name, areas=area_str_civil))
            elif msg_type == 'Update':
                sentences.append(ph['civil_updated'].format(sender=civil_sender, event=event_name, areas=area_str_civil))
            else:
                sentences.append(ph['civil_issued'].format(sender=civil_sender, event=event_name, areas=area_str_civil))

            if expires_dt:
                sentences.append(ph['civil_timing'].format(expires=_format_datetime_spoken(expires_dt, lang_short)))

            if description:
                sentences.append(description)
            if instruction:
                sentences.append(instruction)

        if sentences:
            rendered.append('  '.join(sentences))

    if not rendered:
        return ""

    count = len(rendered)
    opener = ph['opener_1'] if count == 1 else ph['opener'].format(count=_num_to_words(count))
    return opener + '  ' + '  '.join(rendered)


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

    source = weather_data.get('source') if isinstance(weather_data, dict) else None

    station_name = None
    if isinstance(weather_data, dict) and isinstance(weather_data.get('station'), dict):
        station_dict = cast(dict[str, Any], weather_data['station'])
        station_name = station_dict.get(lang_short) or station_dict.get('en')
    name = location_name or station_name or ph['station']

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
            local_tz = datetime.datetime.now().astimezone().tzinfo or ZoneInfo('UTC')
            _dt_local = _dt.astimezone(local_tz)
            _time_str = _format_time_spoken(_dt_local)
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
        sentences.append('.')
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
        sentences.append('.')
        sentences.append(ph['dewpoint'].format(val=dp_val))

    hum_val = _scalar(_d(props, 'humidity') or _s(props, 'humidity'))
    if hum_val is not None:
        sentences.append(ph['humidity'].format(val=hum_val))
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
    fx = _FC_PH.get(lang_short, _FC_PH['en'])
    if forecast_data is None:
        return fx['generic_unavailable'].format(name=location_name or fx['station'])

    source_key = str(forecast_data.get('source') or '').strip().lower()

    name_block = forecast_data.get('name') or {}
    forecast_name = None
    if isinstance(name_block, dict):
        forecast_name = name_block.get(lang_short) or name_block.get('en')
    elif name_block:
        forecast_name = str(name_block)

    loc_name = location_name or forecast_name or fx['station']

    forecasts = forecast_data.get('forecast', [])
    if not forecasts:
        unavailable_key = f'{source_key}_unavailable' if source_key else 'generic_unavailable'
        return fx.get(unavailable_key, fx['generic_unavailable']).format(name=loc_name)

    opener_key = f'{source_key}_opener' if source_key else 'generic_opener'
    opener = fx.get(opener_key, fx['generic_opener'])
    time_str = _format_time_spoken(datetime.datetime.now().astimezone())
    sentences: list[str] = [opener.format(name=loc_name, time=time_str)]

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


def _format_month_day(month: int | None, day: int | None, lang_short: str) -> str:
    if month is None or day is None or not (1 <= month <= 12) or not (1 <= day <= 31):
        return ''
    month_name = _DT_MONTHS.get(lang_short, _DT_MONTHS['en'])[month - 1]
    if lang_short == 'fr':
        return f'{day} {month_name}'
    if lang_short == 'es':
        return f'{day} de {month_name}'
    ordinals = _DT_ORDINALS.get(lang_short, _DT_ORDINALS['en'])
    ordinal = ordinals[day - 1] if day <= len(ordinals) else str(day)
    return f'{month_name} {ordinal}'


def _format_spoken_time_value(timestamp: Any, timezone_name: str | None = None) -> str:
    if not timestamp:
        return ''
    try:
        dt = datetime.datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
        if timezone_name:
            dt = dt.astimezone(ZoneInfo(timezone_name))
        else:
            local_tz = datetime.datetime.now().astimezone().tzinfo or ZoneInfo('UTC')
            dt = dt.astimezone(local_tz)
        return _format_time_spoken(dt)
    except (ValueError, ZoneInfoNotFoundError):
        return _reformat_times_in_text(str(timestamp))


def climate_summary_package(
    climate_data: Optional[dict[str, Any]] = None,
    location_name: Optional[str] = None,
    lang: Optional[str] = 'en-CA',
) -> str:
    _lang = lang or 'en-CA'
    lang_short = _lang[:2]
    ph = _AE_PH.get(lang_short, _AE_PH['en'])

    if climate_data is None:
        return ph['unavailable'].format(name=location_name or ph['station'])

    name_block = climate_data.get('name') or {}
    station_name = None
    if isinstance(name_block, dict):
        station_name = name_block.get(lang_short) or name_block.get('en')
    name = location_name or station_name or ph['station']

    normals = climate_data.get('normals') or {}
    records = climate_data.get('records') or {}
    astronomy = climate_data.get('astronomy') or {}
    observations = climate_data.get('observations') or {}
    date_block = climate_data.get('date') or {}
    month = date_block.get('month') if isinstance(date_block, dict) else None
    day = date_block.get('day') if isinstance(date_block, dict) else None
    spoken_date = _format_month_day(month, day, lang_short)
    source_key = str(climate_data.get('source') or 'eccc')
    source_name = _CC_SOURCE.get(lang_short, _CC_SOURCE['en']).get(source_key, source_key.upper())
    last_updated = climate_data.get('last_updated')

    spoken_time = ''
    spoken_updated_date = spoken_date
    if last_updated:
        try:
            updated_dt = datetime.datetime.fromisoformat(str(last_updated).replace('Z', '+00:00'))
            local_tz = datetime.datetime.now().astimezone().tzinfo or ZoneInfo('UTC')
            updated_local = updated_dt.astimezone(local_tz)
            spoken_time = _format_time_spoken(updated_local)
            spoken_updated_date = _format_date_spoken(updated_local, lang_short)
        except ValueError:
            spoken_time = _reformat_times_in_text(str(last_updated))

    has_normals = isinstance(normals, dict) and bool(normals)
    has_records = isinstance(records, dict) and bool(records)
    has_observations = isinstance(observations, dict) and bool(observations)
    if not has_normals and not has_records and not has_observations:
        return ph['unavailable'].format(name=name)

    timezone_name = astronomy.get('timezone') if isinstance(astronomy.get('timezone'), str) else None
    try:
        climate_zone = ZoneInfo(timezone_name) if timezone_name else (datetime.datetime.now().astimezone().tzinfo or ZoneInfo('UTC'))
    except ZoneInfoNotFoundError:
        climate_zone = datetime.datetime.now().astimezone().tzinfo or ZoneInfo('UTC')

    latest_record_sentence = ''
    observed_date_raw = observations.get('date') if isinstance(observations, dict) else None
    if observed_date_raw:
        try:
            observed_date = datetime.datetime.fromisoformat(str(observed_date_raw)).date()
            current_date = datetime.datetime.now(climate_zone).date()
            if observed_date < current_date:
                observed_dt = datetime.datetime.combine(observed_date, datetime.time.min, tzinfo=climate_zone)
                latest_record_template = ph.get('latest_record')
                if latest_record_template:
                    latest_record_sentence = latest_record_template.format(
                        date=_format_date_spoken(observed_dt, lang_short),
                    )
        except ValueError:
            latest_record_sentence = ''

    if spoken_date:
        period = spoken_date
    elif lang_short == 'fr':
        period = "aujourd'hui"
    elif lang_short == 'es':
        period = 'hoy'
    else:
        period = 'today'

    intro_template = ph['opener']
    intro = intro_template.format(
        name=name,
        period=period,
        time=spoken_time or period,
        date=spoken_updated_date or period,
        source=source_name,
    )

    sentences: list[str] = [intro]
    if latest_record_sentence:
        sentences.append(latest_record_sentence)

    if isinstance(observations, dict):
        observed_high = observations.get('high')
        observed_low = observations.get('low')
        observed_mean = observations.get('mean')
        if observed_high is not None:
            sentences.append(ph['high'].format(val=observed_high))
        if observed_low is not None:
            sentences.append(ph['low'].format(val=observed_low))
        mean_template = ph.get('mean_temp') or ph.get('avg_temp')
        if observed_mean is not None and mean_template:
            sentences.append(mean_template.format(val=observed_mean))

        precipitation = observations.get('precipitation')
        if precipitation is not None:
            if precipitation > 0:
                sentences.append(ph['precip_yea'].format(val=precipitation))
            else:
                sentences.append(ph['precip_no'])

        snowfall = observations.get('snowfall')
        if snowfall is not None:
            if snowfall > 0:
                sentences.append(ph['snow_yea'].format(val=snowfall))
            elif month in {6, 7, 8}:
                sentences.append(ph['snow_if_summer'])
            else:
                sentences.append(ph['snow_no'])

        snow_on_ground = observations.get('snow_on_ground')
        snow_on_ground_template = ph.get('snow_on_ground')
        if snow_on_ground is not None and snow_on_ground_template:
            sentences.append(snow_on_ground_template.format(val=snow_on_ground))

    if isinstance(normals, dict):
        temps = normals.get('temperature') or {}
        if isinstance(temps, dict):
            normal_low = temps.get('low')
            normal_high = temps.get('high')
            if normal_low is not None and normal_high is not None:
                sentences.append(ph['normals'].format(low=normal_low, high=normal_high))

    if isinstance(records, dict):
        high_max = records.get('high_max') or {}
        low_min = records.get('low_min') or {}
        if isinstance(high_max, dict) and isinstance(low_min, dict):
            high_value = high_max.get('value')
            high_year = high_max.get('year')
            low_value = low_min.get('value')
            low_year = low_min.get('year')
            temp_extremes_template = ph.get('temp_extremes')
            if (
                temp_extremes_template
                and high_value is not None
                and high_year is not None
                and low_value is not None
                and low_year is not None
            ):
                sentences.append(temp_extremes_template.format(
                    high=high_value,
                    high_year=high_year,
                    low=low_value,
                    low_year=low_year,
                ))

        for key, phrase_key in (
            ('low_max', 'cold_high'),
            ('high_min', 'warm_low'),
        ):
            record = records.get(key) or {}
            if not isinstance(record, dict):
                continue
            value = record.get('value')
            year = record.get('year')
            template = ph.get(phrase_key)
            if value is None or year is None or not template:
                continue
            sentences.append(template.format(value=value, year=year))

        precipitation = records.get('precipitation') or {}
        if isinstance(precipitation, dict):
            precip_value = precipitation.get('value')
            precip_year = precipitation.get('year')
            precip_template = ph.get('precip_extreme')
            if precip_value is not None and precip_year is not None and precip_template:
                sentences.append(precip_template.format(val=precip_value, year=precip_year))

        snowfall = records.get('snowfall') or {}
        if isinstance(snowfall, dict):
            snow_value = snowfall.get('value')
            snow_year = snowfall.get('year')
            snow_template = ph.get('snow_extreme')
            if snow_value is not None and snow_year is not None and snow_template:
                sentences.append(snow_template.format(val=snow_value, year=snow_year))

    if isinstance(astronomy, dict):
        sunset_value = _format_spoken_time_value(astronomy.get('sunset'), timezone_name)
        if sunset_value:
            sunset_template = ph.get('sunset_tonight')
            if sunset_template:
                sentences.append(sunset_template.format(time=sunset_value))

    return ' '.join(sentences)

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
        paragraphs.append(_flush_discussion(current_para))

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


def _aqhi_risk_key(val: float) -> str:
    if val <= 3:
        return 'low'
    if val <= 6:
        return 'moderate'
    if val <= 10:
        return 'high'
    return 'very_high'


def air_quality_package(
    aqhi_data: Optional[dict[str, Any]] = None,
    location_name: Optional[str] = None,
    lang: Optional[str] = "en-CA",
) -> str:
    _lang = lang or 'en-CA'
    lang_short = _lang[:2]
    ph = _AQI_PH.get(lang_short, _AQI_PH['en'])
    risk_labels = _AQHI_RISK_LABELS.get(lang_short, _AQHI_RISK_LABELS['en'])

    station_name: str | None = None
    if isinstance(aqhi_data, dict):
        location_block = aqhi_data.get('location')
        if isinstance(location_block, dict):
            station_name = location_block.get(lang_short) or location_block.get('en')
    name = location_name or station_name or 'this area'

    if aqhi_data is None:
        return ph['unavailable_report'].format(name=name)

    aqhi_val = aqhi_data.get('aqhi')
    observed_at = aqhi_data.get('observed_at')
    special_notes_block = aqhi_data.get('special_notes') or {}
    forecast_block = aqhi_data.get('forecast') or {}

    sentences: list[str] = []

    if aqhi_val is not None:
        time_str = _format_spoken_time_value(observed_at) if observed_at else ''
        sentences.append(ph['now_eccc'].format(name=name, val=aqhi_val, time=time_str))
        risk_key = _aqhi_risk_key(float(aqhi_val))
        narrative = ph.get(f'{risk_key}_eccc')
        if narrative:
            sentences.append(narrative)

    special_notes = ''
    if isinstance(special_notes_block, dict):
        special_notes = special_notes_block.get(lang_short) or special_notes_block.get('en') or ''
    if special_notes:
        sentences.append(special_notes.strip())

    periods: list[dict[str, Any]] = []
    if isinstance(forecast_block, dict):
        raw_periods = forecast_block.get('periods')
        if isinstance(raw_periods, list):
            periods = [p for p in raw_periods if isinstance(p, dict)]

    def _period_name(p: dict[str, Any]) -> str:
        blk = p.get('period') or {}
        return (blk.get(lang_short) or blk.get('en') or '') if isinstance(blk, dict) else str(blk)

    for i in range(min(2, len(periods))):
        period = periods[i]
        period_val = period.get('aqhi')
        if period_val is None:
            continue
        period_name = _period_name(period)
        risk_key = _aqhi_risk_key(float(period_val))
        risk_label = risk_labels[risk_key]
        if i == 0:
            opener = ph.get('forecast_opener_eccc', '')
            if opener:
                sentences.append(opener.format(name=name, val=period_val, period_name=period_name, risk=risk_label))
        else:
            fcst = ph.get('forecast_eccc', '')
            if fcst:
                sentences.append(fcst.format(period_name=period_name, val=period_val, risk=risk_label))
        period_insmoke = period.get('aqhi_insmoke')
        if period_insmoke is not None:
            insmoke = ph.get('forecast_insmoke_eccc', '')
            if insmoke:
                sentences.append(insmoke.format(val=period_insmoke))

    if len(periods) >= 4:
        p2, p3 = periods[2], periods[3]
        p2_val, p3_val = p2.get('aqhi'), p3.get('aqhi')
        if p2_val is not None and p3_val is not None:
            trailing = ph.get('forecast_trailing_eccc', '')
            if trailing:
                sentences.append(trailing.format(
                    period2_val=p2_val,
                    period2_name=_period_name(p2),
                    period3_val=p3_val,
                    period3_name=_period_name(p3),
                ))
    elif len(periods) >= 3:
        period = periods[2]
        period_val = period.get('aqhi')
        if period_val is not None:
            period_name = _period_name(period)
            risk_key = _aqhi_risk_key(float(period_val))
            risk_label = risk_labels[risk_key]
            fcst = ph.get('forecast_eccc', '')
            if fcst:
                sentences.append(fcst.format(period_name=period_name, val=period_val, risk=risk_label))
            period_insmoke = period.get('aqhi_insmoke')
            if period_insmoke is not None:
                insmoke = ph.get('forecast_insmoke_eccc', '')
                if insmoke:
                    sentences.append(insmoke.format(val=period_insmoke))

    if not sentences:
        return ph['unavailable_report'].format(name=name)

    return ' '.join(sentences)
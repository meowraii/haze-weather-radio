from __future__ import annotations

import asyncio
import csv
import datetime
import functools
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import aiohttp

from managed.events import data_ready, shutdown_event, update_data_pool

log = logging.getLogger(__name__)

_DEFAULT_POLL_INTERVAL = 2700
_TIMEOUT = aiohttp.ClientTimeout(connect=5, total=20)
_CITYPAGE_URL = (
    "https://api.weather.gc.ca/collections/citypageweather-realtime/items/{location_id}?f=json"
)
_BULLETINS_URL = (
    "https://api.weather.gc.ca/collections/bulletins-realtime/items"
    "?f=json&issuer_code=CWWG&type=FO&datetime={date}"
)
_NWS_OBSERVATIONS_URL = "https://api.weather.gov/stations/{station_id}/observations/latest"
_NWS_ZONE_FORECAST_URL = "https://api.weather.gov/zones/forecast/{zone_id}/forecast"
_NWS_HEADERS = {
    "User-Agent": "HazeWeatherRadio/1.0 (github.com/hazewx/haze-weather-radio)",
    "Accept": "application/geo+json",
}
_TWC_OBSERVATIONS_URL = (
    "https://api.weather.com/v3/wx/observations/current"
    "?icaoCode={station_id}&units={units}&language={language}&format=json&apiKey={api_key}"
)
_ECCC_AQHI_CURRENT_URL = (
    "https://api.weather.gc.ca/collections/aqhi-observations-realtime/items?offset=0&limit=1000&sortby=-latest&location_id={station_id}&f=json"
)
_ECCC_AQHI_URL = (
    "https://api.weather.gc.ca/collections/aqhi-forecasts-realtime/items?limit=1000&offset=0&f=json&sortby=-publication_datetime&aqhi_type=-Period&location_id={station_id}"
)
_TWC_LOCATION_POINT_URL = "https://api.weather.com/v3/location/point"
_ECCC_LTCE_PRECIP_URL = "https://api.weather.gc.ca/collections/ltce-precipitation/items/{vclimate_id}-{month}-{year}?lang=en&f=json"
_ECCC_LTCE_TEMP_URL = "https://api.weather.gc.ca/collections/ltce-temperature/items/{vclimate_id}-{month}-{year}?lang=en&f=json"
_ECCC_LTCE_SNOWFALL_URL = "https://api.weather.gc.ca/collections/ltce-snowfall/items/{vclimate_id}-{month}-{year}?lang=en&f=json"
_ECCC_CLIMATE_DAILY_URL = "https://api.weather.gc.ca/collections/climate-daily/items"

_ECCC_DATA_DIR = os.path.join("data", "eccc")
_NWS_DATA_DIR = os.path.join("data", "nws")
_TWC_DATA_DIR = os.path.join("data", "weatherdotcom")
_ECCC_FORECAST_REGIONS = os.path.join("managed", "FORECAST_LOCATIONS.csv")

_CARDINAL_DIRS = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)


@dataclass(frozen=True, slots=True)
class ObservationLocation:
    id: str
    source: str
    feed_id: str


@dataclass(frozen=True, slots=True)
class ForecastLocation:
    id: str
    forecast_region: str
    source: str
    feed_id: str
    timezone: str = "UTC"
    name_en: str = ""
    name_fr: str = ""


@dataclass(frozen=True, slots=True)
class ClimateLocation:
    id: str
    citypage_id: str
    source: str
    feed_id: str
    timezone: str = "UTC"


@dataclass(frozen=True, slots=True)
class AirQualityLocation:
    id: str
    source: str
    feed_id: str


@dataclass(frozen=True, slots=True)
class TWCLocale:
    locale1: str | None
    locale2: str | None
    locale3: str | None
    locale4: str | None


@dataclass(frozen=True, slots=True)
class TWCLocationPoint:
    latitude: float | None
    longitude: float | None
    city: str | None
    locale: TWCLocale
    neighborhood: str | None
    adminDistrict: str | None
    adminDistrictCode: str | None
    postalCode: str | None
    postalKey: str | None
    country: str | None
    countryCode: str | None
    ianaTimeZone: str | None
    displayName: str | None
    dstEnd: str | None
    dstStart: str | None
    dmaCd: str | None
    placeId: str | None
    featureId: str | None
    disputedArea: bool
    disputedCountries: list[str] | None
    disputedCountryCodes: list[str] | None
    disputedCustomers: list[str] | None
    disputedShowCountry: list[bool]
    canonicalCityId: str | None
    countyId: str | None
    locId: str | None
    locationCategory: str | None
    pollenId: str | None
    pwsId: str | None
    regionalSatellite: str | None
    tideId: str | None
    type: str | None
    zoneId: str | None
    airportName: str | None
    displayContext: str | None
    icaoCode: str | None
    iataCode: str | None


@dataclass(frozen=True, slots=True)
class ECCCConditions:
    temperature: float
    condition: str
    wind_speed: float
    wind_direction: str
    wind_gust: float | None
    humidity: int
    dewpoint: float
    pressure: float
    pressure_tendency: str | None
    visibility: float | None
    wind_chill: int | None
    humidex: int | None
    station: str
    observed_at: str


@dataclass(frozen=True, slots=True)
class FOBulletin:
    identifier: str
    issued_at: datetime.datetime
    url: str
    text: str


@functools.lru_cache(maxsize=1)
def _load_forecast_region_names() -> dict[str, tuple[str, str]]:
    names: dict[str, tuple[str, str]] = {}
    try:
        with open(_ECCC_FORECAST_REGIONS, newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader, None)  # row 0: BOM + title row
            header_row = next(reader, None)  # row 1: column headers
            if header_row is None:
                return names
            header = [cell.strip().upper() for cell in header_row]
            try:
                code_idx = header.index("CODE")
                name_idx = header.index("NAME")
                nom_idx = header.index("NOM")
            except ValueError as e:
                log.error("Failed to load forecast region names: missing column %s", e)
                return names
            for row in reader:
                if len(row) <= max(code_idx, name_idx, nom_idx):
                    continue
                code = row[code_idx].strip().strip('"')
                if code and code not in names:
                    names[code] = (
                        row[name_idx].strip().strip('"'),
                        row[nom_idx].strip().strip('"'),
                    )
    except (OSError, csv.Error) as e:
        log.error("Failed to load forecast region names: %s", e)
    return names


def _parse_forecast_region(forecast_region: str) -> tuple[str, str | None]:
    if "-" in forecast_region:
        base, sub = forecast_region.split("-", 1)
        return base.strip(), sub.strip()
    return forecast_region.strip(), None


def _derive_citypage_id_from_climate_id(climate_id: str) -> str:
    match = re.fullmatch(r"VS([A-Z]{2})([A-Z0-9]+)V", climate_id.strip().upper())
    if not match:
        return ""
    province, raw_suffix = match.groups()
    suffix = str(int(raw_suffix)) if raw_suffix.isdigit() else raw_suffix.lower()
    return f"{province.lower()}-{suffix}"


def _parse_locations_config(
    config: dict[str, Any],
) -> tuple[list[ObservationLocation], list[ForecastLocation], list[ClimateLocation], list[AirQualityLocation]]:
    region_names = _load_forecast_region_names()

    obs_locs: list[ObservationLocation] = []
    forecast_locs: list[ForecastLocation] = []
    climate_locs: list[ClimateLocation] = []
    aqhi_locs: list[AirQualityLocation] = []
    for feed in config.get("feeds", []):
        if not isinstance(feed, dict):
            continue
        feed_id = (feed.get("id") or "").strip()
        for block in feed.get("locations", []):
            if not isinstance(block, dict):
                continue

            for entry in block.get("observationLocations", []):
                if not isinstance(entry, dict):
                    continue
                loc_id = (entry.get("id") or "").strip()
                source = (entry.get("source") or "eccc").strip()
                if not loc_id:
                    continue
                obs_locs.append(ObservationLocation(id=loc_id, source=source, feed_id=feed_id))

            for entry in block.get("forecastLocations", []):
                if not isinstance(entry, dict):
                    continue
                loc_id = (entry.get("id") or "").strip()
                forecast_region = (entry.get("forecast_region") or "").strip()
                source = (entry.get("source") or "eccc").strip()
                if not loc_id or not forecast_region:
                    log.warning("Skipping forecast location with missing id or forecast_region: %s", entry)
                    continue
                base_code, _ = _parse_forecast_region(forecast_region)
                name_en, name_fr = region_names.get(base_code, ("", ""))
                forecast_locs.append(ForecastLocation(
                    id=loc_id,
                    forecast_region=forecast_region,
                    source=source,
                    feed_id=feed_id,
                    timezone=(feed.get("timezone") or "UTC"),
                    name_en=name_en,
                    name_fr=name_fr,
                ))

            for entry in block.get("airQualityLocations", []):
                if not isinstance(entry, dict):
                    continue
                loc_id = (entry.get("id") or "").strip()
                source = (entry.get("source") or "eccc").strip()
                if not loc_id:
                    continue
                aqhi_locs.append(AirQualityLocation(id=loc_id, source=source, feed_id=feed_id))

            for entry in block.get("climateLocations", []):
                if not isinstance(entry, dict):
                    continue
                loc_id = (entry.get("id") or "").strip()
                source = (entry.get("source") or "eccc").strip()
                citypage_id = (entry.get("citypage_id") or "").strip() or _derive_citypage_id_from_climate_id(loc_id)
                if not loc_id or not citypage_id:
                    log.warning("Skipping climate location with missing id or derived citypage_id: %s", entry)
                    continue
                climate_locs.append(ClimateLocation(
                    id=loc_id,
                    citypage_id=citypage_id,
                    source=source,
                    feed_id=feed_id,
                    timezone=(feed.get("timezone") or "UTC"),
                ))

    return obs_locs, forecast_locs, climate_locs, aqhi_locs


def iter_locations(
    config: dict[str, Any],
) -> tuple[list[ObservationLocation], list[ForecastLocation], list[ClimateLocation], list[AirQualityLocation]]:
    return _parse_locations_config(config)


def _degrees_to_cardinal(degrees: float) -> str:
    return _CARDINAL_DIRS[round(degrees / 22.5) % 16]


def _bilingual(obj: dict[str, Any], *keys: str) -> dict[str, Any] | None:
    d: Any = obj
    for k in keys:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    if isinstance(d, dict) and ("en" in d or "fr" in d):
        return {lang: d[lang] for lang in ("en", "fr") if lang in d}
    return None


def _ev(obj: dict[str, Any], *keys: str) -> Any:
    d: Any = obj
    for k in keys:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d if not isinstance(d, dict) else d.get("en")


async def _fetch_eccc_raw(
    session: aiohttp.ClientSession,
    location_id: str,
) -> dict[str, Any] | None:
    url = _CITYPAGE_URL.format(location_id=location_id)
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            raw = await resp.json()
    except aiohttp.ClientError as e:
        log.error("Failed to fetch ECCC data for %s: %s", location_id, e)
        return None
    return raw


async def _prefetch_eccc(
    session: aiohttp.ClientSession,
    location_id: str,
    cache: dict[str, dict[str, Any] | None],
) -> None:
    cache[location_id] = await _fetch_eccc_raw(session, location_id)


def _parse_eccc_conditions(raw: dict[str, Any], location_id: str) -> ECCCConditions | None:
    try:
        cc = raw["properties"]["currentConditions"]
        return ECCCConditions(
            temperature=cc["temperature"]["value"]["en"],
            condition=cc.get("condition", {}).get("en", ""),
            wind_speed=cc["wind"]["speed"]["value"]["en"],
            wind_direction=cc["wind"]["direction"]["value"]["en"],
            wind_gust=cc["wind"].get("gust", {}).get("value", {}).get("en"),
            humidity=cc["relativeHumidity"]["value"]["en"],
            dewpoint=cc["dewpoint"]["value"]["en"],
            pressure=cc["pressure"]["value"]["en"],
            pressure_tendency=cc.get("pressure", {}).get("tendency", {}).get("en"),
            visibility=cc.get("visibility", {}).get("value", {}).get("en"),
            wind_chill=cc.get("windChill", {}).get("value", {}).get("en"),
            humidex=cc.get("humidex", {}).get("value", {}).get("en"),
            station=cc["station"]["value"]["en"],
            observed_at=cc["timestamp"]["en"],
        )
    except (KeyError, TypeError) as e:
        log.error("Failed to parse ECCC conditions for %s: %s", location_id, e)
        return None


async def fetch_eccc_conditions(
    session: aiohttp.ClientSession,
    location_id: str,
) -> ECCCConditions | None:
    raw = await _fetch_eccc_raw(session, location_id)
    if raw is None:
        return None
    return _parse_eccc_conditions(raw, location_id)


def _normalize_eccc_hourly(h: dict[str, Any]) -> dict[str, Any]:
    wind_raw = h.get("wind", {}) or {}
    ts_raw = h.get("timestamp", {})
    timestamp: Any = ts_raw.get("en") if isinstance(ts_raw, dict) else ts_raw
    return {
        "timestamp": timestamp,
        "condition": _bilingual(h, "condition") or {},
        "temperature": _ev(h, "temperature", "value"),
        "windChill": _ev(h, "windChill", "value"),
        "wind": {
            "speed": _ev(wind_raw, "speed", "value"),
            "direction": _ev(wind_raw, "direction", "value"),
            "gust": _ev(wind_raw, "gust", "value"),
        },
    }


def _normalize_eccc_forecast(forecast_group: dict[str, Any]) -> list[dict[str, Any]]:
    periods: list[dict[str, Any]] = []
    for f in forecast_group.get("forecasts", []):
        period_name = _bilingual(f, "period", "textForecastName") or {}
        parts_en: list[str] = []
        parts_fr: list[str] = []
        for src_key in ("cloudPrecip",):
            chunk = f.get(src_key, {}) or {}
            if chunk.get("en"):
                parts_en.append(chunk["en"])
            if chunk.get("fr"):
                parts_fr.append(chunk["fr"])
        for nested_key in ("temperatures", "windChill"):
            summary = (f.get(nested_key, {}) or {}).get("textSummary", {}) or {}
            if summary.get("en"):
                parts_en.append(summary["en"])
            if summary.get("fr"):
                parts_fr.append(summary["fr"])
        periods.append({
            "period": period_name,
            "textSummary": {
                "en": " ".join(filter(None, parts_en)),
                "fr": " ".join(filter(None, parts_fr)),
            },
        })
    return periods


def _normalize_eccc_regional_normals(forecast_group: dict[str, Any]) -> dict[str, Any] | None:
    regional_normals = forecast_group.get("regionalNormals") or {}
    if not isinstance(regional_normals, dict) or not regional_normals:
        return None

    values: dict[str, Any] = {}
    for entry in regional_normals.get("temperature", []):
        if not isinstance(entry, dict):
            continue
        class_name = ((entry.get("class") or {}).get("en") if isinstance(entry.get("class"), dict) else None) or entry.get("class")
        if class_name not in {"high", "low"}:
            continue
        values[str(class_name)] = _ev(entry, "value")

    return {
        "textSummary": _bilingual(regional_normals, "textSummary") or {},
        "temperature": values,
    }

def _parse_eccc_aqhi_observation(raw: dict[str, Any], station_id: str) -> dict[str, Any] | None:
    try:
        features = raw.get("features") or []
        if not features:
            return None
        props = features[0]["properties"]
        return {
            "location": {
                "en": props.get("location_name_en") or "",
                "fr": props.get("location_name_fr") or "",
            },
            "observed_at": props.get("observation_datetime"),
            "aqhi": props.get("aqhi"),
            "special_notes": {
                "en": props.get("special_notes_en") or "",
                "fr": props.get("special_notes_fr") or "",
            },
        }
    except (KeyError, IndexError, TypeError) as e:
        log.error("Failed to parse ECCC AQHI observation for %s: %s", station_id, e)
        return None


def _parse_eccc_aqhi_forecast(raw: dict[str, Any], station_id: str) -> dict[str, Any] | None:
    try:
        features = raw.get("features") or []
        if not features:
            return None
        props = features[0]["properties"]
        forecast_period = props.get("forecast_period") or {}
        periods = [
            {
                "period": {
                    "en": p.get("forecast_period_en") or "",
                    "fr": p.get("forecast_period_fr") or "",
                },
                "aqhi": p.get("aqhi"),
                "aqhi_insmoke": p.get("aqhi_insmoke"),
            }
            for p in (
                forecast_period.get(key)
                for key in sorted(forecast_period)
            )
            if isinstance(p, dict)
        ]
        return {
            "published_at": props.get("publication_datetime"),
            "periods": periods,
        }
    except (KeyError, IndexError, TypeError) as e:
        log.error("Failed to parse ECCC AQHI forecast for %s: %s", station_id, e)
        return None


async def _fetch_eccc_aqhi_observation_raw(
    session: aiohttp.ClientSession,
    station_id: str,
) -> dict[str, Any] | None:
    url = _ECCC_AQHI_CURRENT_URL.format(station_id=station_id)
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            return await resp.json()
    except aiohttp.ClientError as e:
        log.error("Failed to fetch ECCC AQHI observation for %s: %s", station_id, e)
        return None


async def _fetch_eccc_aqhi_forecast_raw(
    session: aiohttp.ClientSession,
    station_id: str,
) -> dict[str, Any] | None:
    url = _ECCC_AQHI_URL.format(station_id=station_id)
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            return await resp.json()
    except aiohttp.ClientError as e:
        log.error("Failed to fetch ECCC AQHI forecast for %s: %s", station_id, e)
        return None


def _build_eccc_conditions_dict(raw_props: dict[str, Any]) -> dict[str, Any]:
    cc = raw_props.get("currentConditions", {})
    wind_raw = cc.get("wind", {}) or {}
    hfg = raw_props.get("hourlyForecastGroup", {}) or {}
    return {
        "source": "eccc",
        "observed_at": _ev(cc, "timestamp"),
        "station": _bilingual(cc, "station", "value") or {},
        "properties": {
            "temp": _ev(cc, "temperature", "value"),
            "condition": _bilingual(cc, "condition") or {},
            "wind": {
                "speed": _ev(wind_raw, "speed", "value"),
                "direction": _ev(wind_raw, "direction", "value"),
                "gust": _ev(wind_raw, "gust", "value"),
            },
            "humidity": _ev(cc, "relativeHumidity", "value"),
            "dewpoint": _ev(cc, "dewpoint", "value"),
            "visibility": _ev(cc, "visibility", "value"),
            "pressure": {
                "value": _ev(cc, "pressure", "value"),
                "tendency": _bilingual(cc, "pressure", "tendency"),
            },
            "windChill": _ev(cc, "windChill", "value"),
            "humidex": _ev(cc, "humidex", "value"),
            "heatIndex": None,
        },
        "hourly": [_normalize_eccc_hourly(h) for h in hfg.get("hourlyForecasts", [])],
    }


def _feed_local_month_day(timezone_name: str) -> tuple[int, int]:
    try:
        zone = ZoneInfo(timezone_name)
    except Exception:
        zone = ZoneInfo("UTC")
    now = datetime.datetime.now(zone)
    return now.month, now.day


def _province_code_from_citypage_id(citypage_id: str) -> str:
    if "-" not in citypage_id:
        return ""
    return citypage_id.split("-", 1)[0].strip().upper()


def _normalize_station_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _parse_climate_local_date(value: Any) -> datetime.date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value).date()
    except ValueError:
        return None


def _observation_field_count(props: dict[str, Any]) -> int:
    keys = (
        "MAX_TEMPERATURE",
        "MIN_TEMPERATURE",
        "MEAN_TEMPERATURE",
        "TOTAL_PRECIPITATION",
        "TOTAL_RAIN",
        "TOTAL_SNOW",
        "SNOW_ON_GROUND",
    )
    return sum(1 for key in keys if props.get(key) is not None)


def _pick_best_climate_daily_feature(
    features: list[dict[str, Any]],
    requested_name: str,
    province_code: str,
) -> dict[str, Any] | None:
    requested_name_norm = _normalize_station_name(requested_name)
    scored: list[tuple[datetime.date, int, int, int, str, dict[str, Any]]] = []
    for feature in features:
        if not isinstance(feature, dict):
            continue
        props = feature.get("properties") or {}
        if not isinstance(props, dict):
            continue
        local_date = _parse_climate_local_date(props.get("LOCAL_DATE"))
        if local_date is None:
            continue
        station_name = str(props.get("STATION_NAME") or "")
        station_name_norm = _normalize_station_name(station_name)
        scored.append((
            local_date,
            1 if province_code and props.get("PROVINCE_CODE") == province_code else 0,
            1 if requested_name_norm and station_name_norm.startswith(requested_name_norm) else 0,
            _observation_field_count(props),
            station_name_norm,
            props,
        ))

    if not scored:
        return None

    latest_date = max(item[0] for item in scored)
    newest = [item for item in scored if item[0] == latest_date]
    newest.sort(key=lambda item: (item[1], item[2], item[3], item[4]), reverse=True)
    return newest[0][5]


async def _fetch_eccc_climate_daily(
    session: aiohttp.ClientSession,
    location_name: str,
    province_code: str,
) -> dict[str, Any] | None:
    params = {
        "f": "json",
        "limit": 12,
        "sortby": "-LOCAL_DATE",
        "STATION_NAME": location_name,
    }
    try:
        async with session.get(_ECCC_CLIMATE_DAILY_URL, params=params, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            payload = await resp.json()
    except aiohttp.ClientError as e:
        log.error("Failed to fetch ECCC climate-daily data for %s: %s", location_name, e)
        return None

    features = payload.get("features") or []
    if not isinstance(features, list):
        return None

    best = _pick_best_climate_daily_feature(features, location_name, province_code)
    if best is not None:
        return best

    fallback_params = {
        "f": "json",
        "limit": 12,
        "sortby": "-LOCAL_DATE",
        "q": location_name,
    }
    try:
        async with session.get(_ECCC_CLIMATE_DAILY_URL, params=fallback_params, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            payload = await resp.json()
    except aiohttp.ClientError as e:
        log.error("Failed fallback ECCC climate-daily search for %s: %s", location_name, e)
        return None

    features = payload.get("features") or []
    if not isinstance(features, list):
        return None
    return _pick_best_climate_daily_feature(features, location_name, province_code)


def _build_eccc_observed_climate_dict(observed_props: dict[str, Any]) -> dict[str, Any] | None:
    local_date = observed_props.get("LOCAL_DATE")
    station_name = str(observed_props.get("STATION_NAME") or "").strip()
    if not station_name and not local_date:
        return None
    return {
        "station": {
            "en": station_name,
            "fr": station_name,
        },
        "date": local_date,
        "high": observed_props.get("MAX_TEMPERATURE"),
        "low": observed_props.get("MIN_TEMPERATURE"),
        "mean": observed_props.get("MEAN_TEMPERATURE"),
        "precipitation": observed_props.get("TOTAL_PRECIPITATION"),
        "rain": observed_props.get("TOTAL_RAIN"),
        "snowfall": observed_props.get("TOTAL_SNOW"),
        "snow_on_ground": observed_props.get("SNOW_ON_GROUND"),
    }


async def _fetch_eccc_ltce_temperature(
    session: aiohttp.ClientSession,
    vclimate_id: str,
    month: int,
    day: int,
) -> dict[str, Any] | None:
    url = _ECCC_LTCE_TEMP_URL.format(vclimate_id=vclimate_id, month=month, year=day)
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            raw = await resp.json()
    except aiohttp.ClientError as e:
        log.error("Failed to fetch ECCC LTCE temperature for %s-%d-%d: %s", vclimate_id, month, day, e)
        return None
    return raw


async def _fetch_eccc_ltce_precipitation(
    session: aiohttp.ClientSession,
    vclimate_id: str,
    month: int,
    day: int,
) -> dict[str, Any] | None:
    url = _ECCC_LTCE_PRECIP_URL.format(vclimate_id=vclimate_id, month=month, year=day)
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            raw = await resp.json()
    except aiohttp.ClientError as e:
        log.error("Failed to fetch ECCC LTCE precipitation for %s-%d-%d: %s", vclimate_id, month, day, e)
        return None
    return raw


async def _fetch_eccc_ltce_snowfall(
    session: aiohttp.ClientSession,
    vclimate_id: str,
    month: int,
    day: int,
) -> dict[str, Any] | None:
    url = _ECCC_LTCE_SNOWFALL_URL.format(vclimate_id=vclimate_id, month=month, year=day)
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            raw = await resp.json()
    except aiohttp.ClientError as e:
        log.error("Failed to fetch ECCC LTCE snowfall for %s-%d-%d: %s", vclimate_id, month, day, e)
        return None
    return raw


async def _prefetch_eccc_ltce_temperature(
    session: aiohttp.ClientSession,
    vclimate_id: str,
    month: int,
    day: int,
    cache: dict[str, dict[str, Any] | None],
) -> None:
    cache[f"{vclimate_id}:{month}:{day}"] = await _fetch_eccc_ltce_temperature(session, vclimate_id, month, day)


async def _prefetch_eccc_ltce_precipitation(
    session: aiohttp.ClientSession,
    vclimate_id: str,
    month: int,
    day: int,
    cache: dict[str, dict[str, Any] | None],
) -> None:
    cache[f"{vclimate_id}:{month}:{day}"] = await _fetch_eccc_ltce_precipitation(session, vclimate_id, month, day)


async def _prefetch_eccc_ltce_snowfall(
    session: aiohttp.ClientSession,
    vclimate_id: str,
    month: int,
    day: int,
    cache: dict[str, dict[str, Any] | None],
) -> None:
    cache[f"{vclimate_id}:{month}:{day}"] = await _fetch_eccc_ltce_snowfall(session, vclimate_id, month, day)


def _build_eccc_climate_dict(
    raw_props: dict[str, Any],
    ltce_temp_props: dict[str, Any] | None,
    ltce_precip_props: dict[str, Any] | None,
    ltce_snow_props: dict[str, Any] | None,
    observed_props: dict[str, Any] | None,
    timezone_name: str,
) -> dict[str, Any] | None:
    forecast_group = raw_props.get("forecastGroup") or {}
    if not isinstance(forecast_group, dict):
        forecast_group = {}

    normals = _normalize_eccc_regional_normals(forecast_group)
    rise_set = raw_props.get("riseSet") or {}

    if not isinstance(rise_set, dict):
        rise_set = {}

    if normals is None and not ltce_temp_props and not ltce_precip_props and not ltce_snow_props and not observed_props and not rise_set:
        return None

    name = {
        "en": (ltce_temp_props or ltce_precip_props or ltce_snow_props or {}).get("VIRTUAL_STATION_NAME_E") or _ev(raw_props, "name") or "",
        "fr": (ltce_temp_props or ltce_precip_props or ltce_snow_props or {}).get("VIRTUAL_STATION_NAME_F") or _ev(raw_props, "name") or "",
    }

    records = None
    if ltce_temp_props or ltce_precip_props or ltce_snow_props:
        records = {
            "high_max": {
                "value": (ltce_temp_props or {}).get("RECORD_HIGH_MAX_TEMP"),
                "year": (ltce_temp_props or {}).get("RECORD_HIGH_MAX_TEMP_YR"),
            },
            "low_max": {
                "value": (ltce_temp_props or {}).get("RECORD_LOW_MAX_TEMP"),
                "year": (ltce_temp_props or {}).get("RECORD_LOW_MAX_TEMP_YR"),
            },
            "high_min": {
                "value": (ltce_temp_props or {}).get("RECORD_HIGH_MIN_TEMP"),
                "year": (ltce_temp_props or {}).get("RECORD_HIGH_MIN_TEMP_YR"),
            },
            "low_min": {
                "value": (ltce_temp_props or {}).get("RECORD_LOW_MIN_TEMP"),
                "year": (ltce_temp_props or {}).get("RECORD_LOW_MIN_TEMP_YR"),
            },
            "precipitation": {
                "value": (ltce_precip_props or {}).get("RECORD_PRECIPITATION"),
                "year": (ltce_precip_props or {}).get("RECORD_PRECIPITATION_YR"),
            },
            "snowfall": {
                "value": (ltce_snow_props or {}).get("RECORD_SNOWFALL"),
                "year": (ltce_snow_props or {}).get("RECORD_SNOWFALL_YR"),
            },
        }

    observations = _build_eccc_observed_climate_dict(observed_props) if observed_props else None

    astronomy = {
        "sunrise": _ev(rise_set, "sunrise"),
        "sunset": _ev(rise_set, "sunset"),
        "timezone": timezone_name,
    }

    date_source = observed_props or ltce_temp_props or ltce_precip_props or ltce_snow_props or {}
    last_updated = (
        (ltce_temp_props or {}).get("LAST_UPDATED")
        or (ltce_precip_props or {}).get("LAST_UPDATED")
        or (ltce_snow_props or {}).get("LAST_UPDATED")
        or raw_props.get("lastUpdated")
    )

    return {
        "source": "eccc",
        "name": name,
        "date": {
            "month": date_source.get("LOCAL_MONTH"),
            "day": date_source.get("LOCAL_DAY"),
        },
        "observations": observations,
        "normals": normals,
        "records": records,
        "astronomy": astronomy,
        "last_updated": last_updated,
    }


def _twc_language(twc_cfg: dict[str, Any]) -> str:
    lang_cfg = str(twc_cfg.get("language") or "en-CA")
    lang_short = lang_cfg[:2].lower()
    if lang_short == "en":
        return "en-US"
    if lang_short == "fr":
        return "fr-FR"
    if lang_short == "es":
        return "es-US"
    return lang_cfg


def _twc_location_query_candidates(location_id: str) -> list[tuple[str, str]]:
    normalized = location_id.strip()
    upper = normalized.upper()
    candidates: list[tuple[str, str]] = []

    if re.fullmatch(r"-?\d+(?:\.\d+)?,-?\d+(?:\.\d+)?", normalized):
        candidates.append(("geocode", normalized))
    elif ":" in normalized:
        candidates.append(("locId", normalized))
    elif re.fullmatch(r"[A-Z]{4}", upper):
        candidates.append(("icaoCode", upper))
    elif re.fullmatch(r"[A-Z]{3}", upper):
        candidates.append(("iataCode", upper))
    elif re.fullmatch(r"[A-Z]{3}\d{3}", upper):
        if upper[2] == "C":
            candidates.extend([("countyId", upper), ("zoneId", upper)])
        elif upper[2] == "Z":
            candidates.extend([("zoneId", upper), ("countyId", upper)])
        else:
            candidates.append(("zoneId", upper))
    elif re.fullmatch(r"\d{6}", normalized):
        candidates.append(("zoneId", normalized))

    candidates.extend([
        ("icaoCode", upper),
        ("iataCode", upper),
        ("locId", normalized),
    ])

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def _parse_twc_location_point(payload: dict[str, Any]) -> TWCLocationPoint | None:
    location_payload = payload.get("location")
    location = location_payload if isinstance(location_payload, dict) else payload
    if not isinstance(location, dict) or not location:
        return None

    locale_payload = location.get("locale")
    locale = locale_payload if isinstance(locale_payload, dict) else {}

    return TWCLocationPoint(
        latitude=location.get("latitude"),
        longitude=location.get("longitude"),
        city=location.get("city"),
        locale=TWCLocale(
            locale1=locale.get("locale1") or location.get("locale1"),
            locale2=locale.get("locale2") or location.get("locale2"),
            locale3=locale.get("locale3") or location.get("locale3"),
            locale4=locale.get("locale4") or location.get("locale4"),
        ),
        neighborhood=location.get("neighborhood"),
        adminDistrict=location.get("adminDistrict"),
        adminDistrictCode=location.get("adminDistrictCode"),
        postalCode=location.get("postalCode"),
        postalKey=location.get("postalKey"),
        country=location.get("country"),
        countryCode=location.get("countryCode"),
        ianaTimeZone=location.get("ianaTimeZone"),
        displayName=location.get("displayName"),
        dstEnd=location.get("dstEnd"),
        dstStart=location.get("dstStart"),
        dmaCd=location.get("dmaCd"),
        placeId=location.get("placeId"),
        featureId=location.get("featureId"),
        disputedArea=bool(location.get("disputedArea", False)),
        disputedCountries=location.get("disputedCountries"),
        disputedCountryCodes=location.get("disputedCountryCodes"),
        disputedCustomers=location.get("disputedCustomers"),
        disputedShowCountry=location.get("disputedShowCountry") or [],
        canonicalCityId=location.get("canonicalCityId"),
        countyId=location.get("countyId"),
        locId=location.get("locId"),
        locationCategory=location.get("locationCategory"),
        pollenId=location.get("pollenId"),
        pwsId=location.get("pwsId"),
        regionalSatellite=location.get("regionalSatellite"),
        tideId=location.get("tideId"),
        type=location.get("type"),
        zoneId=location.get("zoneId"),
        airportName=location.get("airportName"),
        displayContext=location.get("displayContext"),
        icaoCode=location.get("icaoCode"),
        iataCode=location.get("iataCode"),
    )


def _twc_location_name(point: TWCLocationPoint | None, fallback: str) -> str:
    if point is None:
        return fallback
    return point.airportName or point.displayName or point.city or point.locale.locale2 or fallback


def _pause_forecast_region_name(value: str) -> str:
    return value.replace(' - ', '. ')


def _forecast_location_name_block(
    loc: ForecastLocation,
    point: TWCLocationPoint | None = None,
) -> dict[str, str]:
    if loc.name_en or loc.name_fr:
        fallback = loc.name_en or loc.name_fr or loc.id
        return {
            "en": _pause_forecast_region_name(loc.name_en or fallback),
            "fr": _pause_forecast_region_name(loc.name_fr or fallback),
        }
    resolved_name = _twc_location_name(point, loc.id)
    return {
        "en": resolved_name,
        "fr": resolved_name,
    }


async def _fetch_twc_location_point(
    session: aiohttp.ClientSession,
    location_id: str,
    twc_cfg: dict[str, Any],
) -> TWCLocationPoint | None:
    api_key = (twc_cfg.get("api_key") or "").strip()
    if not api_key:
        log.error("TWC API key not configured; cannot fetch location metadata for %s", location_id)
        return None

    params_base = {
        "language": _twc_language(twc_cfg),
        "format": "json",
        "apiKey": api_key,
    }
    last_error: aiohttp.ClientResponseError | None = None

    for query_key, query_value in _twc_location_query_candidates(location_id):
        try:
            async with session.get(
                _TWC_LOCATION_POINT_URL,
                params={**params_base, query_key: query_value},
                timeout=_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                payload = await resp.json()
        except aiohttp.ClientResponseError as e:
            last_error = e
            if e.status in {400, 404}:
                continue
            log.error("Failed to fetch TWC location metadata for %s: %s", location_id, e)
            return None
        except aiohttp.ClientError as e:
            log.error("Failed to fetch TWC location metadata for %s: %s", location_id, e)
            return None

        point = _parse_twc_location_point(payload)
        if point is not None:
            return point

    if last_error is not None:
        log.warning("No TWC location metadata found for %s", location_id)
    return None


async def _prefetch_twc_location_point(
    session: aiohttp.ClientSession,
    location_id: str,
    twc_cfg: dict[str, Any],
    cache: dict[str, TWCLocationPoint | None],
) -> None:
    cache[location_id] = await _fetch_twc_location_point(session, location_id, twc_cfg)


def _nws_val(props: dict[str, Any], key: str) -> float | None:
    m = props.get(key)
    if not isinstance(m, dict):
        return None
    v = m.get("value")
    return float(v) if v is not None else None


def _nws_round1(value: float | None) -> float | None:
    return round(value, 1) if value is not None else None


def _nws_unit(props: dict[str, Any], key: str) -> str:
    m = props.get(key)
    return m.get("unitCode", "") if isinstance(m, dict) else ""


def _nws_speed_kmh(props: dict[str, Any], key: str) -> float | None:
    v = _nws_val(props, key)
    if v is None:
        return None
    if "m_s-1" in _nws_unit(props, key):
        return _nws_round1(v * 3.6)
    return _nws_round1(v)


def _nws_pressure_kpa(props: dict[str, Any]) -> float | None:
    v = _nws_val(props, "seaLevelPressure")
    if v is None:
        v = _nws_val(props, "barometricPressure")
    return _nws_round1(v / 1000) if v is not None else None


def _nws_visibility_km(props: dict[str, Any]) -> float | None:
    v = _nws_val(props, "visibility")
    if v is None:
        return None
    unit = _nws_unit(props, "visibility")
    if "km" not in unit and "m" in unit:
        return _nws_round1(v / 1000)
    return _nws_round1(v)


async def _fetch_nws_raw(
    session: aiohttp.ClientSession,
    station_id: str,
) -> dict[str, Any] | None:
    url = _NWS_OBSERVATIONS_URL.format(station_id=station_id)
    try:
        async with session.get(url, timeout=_TIMEOUT, headers=_NWS_HEADERS) as resp:
            resp.raise_for_status()
            raw = await resp.json(content_type=None)
    except aiohttp.ClientError as e:
        log.error("Failed to fetch NWS observations for %s: %s", station_id, e)
        return None
    return raw


async def _fetch_nws_zone_forecast(
    session: aiohttp.ClientSession,
    zone_id: str,
) -> dict[str, Any] | None:
    url = _NWS_ZONE_FORECAST_URL.format(zone_id=zone_id)
    try:
        async with session.get(url, timeout=_TIMEOUT, headers=_NWS_HEADERS) as resp:
            resp.raise_for_status()
            raw = await resp.json(content_type=None)
    except aiohttp.ClientError as e:
        log.error("Failed to fetch NWS zone forecast for %s: %s", zone_id, e)
        return None
    return raw


def _build_nws_conditions_dict(props: dict[str, Any], location_name: str | None = None) -> dict[str, Any]:
    wind_dir_deg = _nws_val(props, "windDirection")
    return {
        "source": "nws",
        "observed_at": props.get("timestamp"),
        "station": {"en": location_name or props.get("name") or props.get("stationIdentifier") or ""},
        "properties": {
            "temp": _nws_round1(_nws_val(props, "temperature")),
            "condition": {"en": props.get("textDescription") or ""},
            "wind": {
                "speed": _nws_speed_kmh(props, "windSpeed"),
                "direction": _degrees_to_cardinal(wind_dir_deg) if wind_dir_deg is not None else None,
                "gust": _nws_speed_kmh(props, "windGust"),
            },
            "humidity": _nws_round1(_nws_val(props, "relativeHumidity")),
            "dewpoint": _nws_round1(_nws_val(props, "dewpoint")),
            "visibility": _nws_visibility_km(props),
            "pressure": {
                "value": _nws_pressure_kpa(props),
                "tendency": None,
            },
            "windChill": _nws_round1(_nws_val(props, "windChill")),
            "humidex": None,
            "heatIndex": _nws_round1(_nws_val(props, "heatIndex")),
        },
        "hourly": [],
    }


def _build_nws_forecast_dict(
    data: dict[str, Any],
    loc: ForecastLocation,
    point: TWCLocationPoint | None = None,
) -> dict[str, Any]:
    periods_raw = (data.get("properties") or {}).get("periods") or []
    periods = [
        {
            "period": {"en": p.get("name") or "", "fr": ""},
            "textSummary": {
                "en": p.get("detailedForecast") or p.get("shortForecast") or "",
                "fr": "",
            },
        }
        for p in periods_raw
        if isinstance(p, dict)
    ]
    return {
        "forecast": periods,
        "forecast_region": loc.forecast_region,
        "name": _forecast_location_name_block(loc, point),
    }


async def _fetch_twc_raw(
    session: aiohttp.ClientSession,
    station_id: str,
    twc_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    api_key = (twc_cfg.get("api_key") or "").strip()
    if not api_key:
        log.error("TWC API key not configured; skipping %s", station_id)
        return None
    units = twc_cfg.get("units") or "m"
    language = _twc_language(twc_cfg)
    url = _TWC_OBSERVATIONS_URL.format(
        station_id=station_id,
        units=units,
        language=language,
        api_key=api_key,
    )
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            raw = await resp.json()
    except aiohttp.ClientError as e:
        log.error("Failed to fetch TWC conditions for %s: %s", station_id, e)
        return None
    return raw


def _conditions_cache_path(source: str, location_id: str) -> str:
    base_dir = {
        "eccc": _ECCC_DATA_DIR,
        "nws": _NWS_DATA_DIR,
        "twc": _TWC_DATA_DIR,
    }.get(source, _ECCC_DATA_DIR)
    return os.path.join(base_dir, f"{location_id}.wx.json")


def _forecast_cache_path(source: str, location_id: str) -> str:
    base_dir = {
        "eccc": _ECCC_DATA_DIR,
        "nws": _NWS_DATA_DIR,
        "twc": _TWC_DATA_DIR,
    }.get(source, _ECCC_DATA_DIR)
    return os.path.join(base_dir, f"{location_id}.forecast.json")


def _climate_cache_path(source: str, location_id: str) -> str:
    base_dir = {
        "eccc": _ECCC_DATA_DIR,
        "nws": _NWS_DATA_DIR,
        "twc": _TWC_DATA_DIR,
    }.get(source, _ECCC_DATA_DIR)
    return os.path.join(base_dir, f"{location_id}.climate.json")


def _aqhi_cache_path(location_id: str) -> str:
    return os.path.join(_ECCC_DATA_DIR, f"{location_id}.aqhi.json")


def _build_twc_conditions_dict(
    raw: dict[str, Any],
    station_id: str,
    location_name: str | None = None,
) -> dict[str, Any]:
    pressure_hpa = raw.get("pressureMeanSeaLevel")
    pressure_kpa = round(pressure_hpa / 10, 2) if pressure_hpa is not None else None
    return {
        "source": "twc",
        "observed_at": raw.get("validTimeLocal"),
        "station": {"en": location_name or station_id},
        "properties": {
            "temp": raw.get("temperature"),
            "condition": {"en": raw.get("wxPhraseLong") or raw.get("wxPhraseMedium") or ""},
            "wind": {
                "speed": raw.get("windSpeed"),
                "direction": raw.get("windDirectionCardinal"),
                "gust": raw.get("windGust"),
            },
            "humidity": raw.get("relativeHumidity"),
            "dewpoint": raw.get("temperatureDewPoint"),
            "visibility": raw.get("visibility"),
            "pressure": {
                "value": pressure_kpa,
                "tendency": raw.get("pressureTendencyTrend"),
            },
            "windChill": raw.get("temperatureWindChill"),
            "humidex": None,
            "heatIndex": raw.get("temperatureHeatIndex"),
        },
        "hourly": [],
    }


async def fetch_focn45(session: aiohttp.ClientSession) -> FOBulletin | None:
    ct_date = datetime.datetime.now(ZoneInfo("America/Chicago")).strftime("%Y-%m-%d")
    collection_url = _BULLETINS_URL.format(date=ct_date)

    try:
        async with session.get(collection_url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except aiohttp.ClientError as e:
        log.error("Failed to fetch FOCN45 collection for %s: %s", ct_date, e)
        return None

    features: list[dict[str, Any]] = data.get("features", [])
    if not features:
        log.warning(
            "No FO bulletins matched for %s (numberMatched=%d); trying yesterday",
            ct_date,
            data.get("numberMatched", 0),
        )
        ct_date_yesterday = (
            datetime.datetime.now(ZoneInfo("America/Chicago")) - datetime.timedelta(days=1)
        ).strftime("%Y-%m-%d")
        collection_url = _BULLETINS_URL.format(date=ct_date_yesterday)
        try:
            async with session.get(collection_url, timeout=_TIMEOUT) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError as e:
            log.error("Failed to fetch FOCN45 for fallback date %s: %s", ct_date_yesterday, e)
            return None
        features = data.get("features", [])
        if not features:
            log.warning(
                "No FO bulletins matched for fallback date %s (numberMatched=%d)",
                ct_date_yesterday,
                data.get("numberMatched", 0),
            )
            return None

    try:
        latest = max(
            features,
            key=lambda f: datetime.datetime.fromisoformat(f["properties"]["datetime"]),
        )
    except (KeyError, ValueError) as e:
        log.error("Could not determine latest FOCN45 bulletin: %s", e)
        return None

    props = latest["properties"]
    identifier: str = props["identifier"]
    bulletin_url: str = props["url"]
    issued_at = datetime.datetime.fromisoformat(props["datetime"]).replace(
        tzinfo=datetime.timezone.utc
    )

    log.info("Fetching FOCN45 bulletin %s", identifier)
    try:
        async with session.get(bulletin_url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            text = await resp.text()
    except aiohttp.ClientError as e:
        log.error("Failed to fetch bulletin text for %s: %s", identifier, e)
        return None

    bulletin = FOBulletin(identifier=identifier, issued_at=issued_at, url=bulletin_url, text=text)
    file_path = os.path.join(_ECCC_DATA_DIR, "focn45.cwwg.txt")
    await asyncio.to_thread(_write_text, file_path, text)
    return bulletin


async def _fetch_and_publish_focn45(session: aiohttp.ClientSession) -> None:
    bulletin = await fetch_focn45(session)
    if bulletin:
        update_data_pool("focn45", bulletin, notify=False)
        log.debug("Data pool updated for focn45: %s", bulletin.identifier)
    else:
        log.warning("No FOCN45 bulletin returned")


async def fetch_geophysical_alerts(session: aiohttp.ClientSession) -> str | None:
    url = "https://services.swpc.noaa.gov/text/wwv.txt"
    file_path = os.path.join(_NWS_DATA_DIR, "wwv.txt")
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            text = await resp.text()
            await asyncio.to_thread(_write_text, file_path, text)
    except aiohttp.ClientError as e:
        log.error("Failed to fetch geophysical alerts: %s", e)
        return None
    return text


async def _fetch_and_publish_wwv(session: aiohttp.ClientSession) -> None:
    text = await fetch_geophysical_alerts(session)
    if text:
        update_data_pool("wwv", text, notify=False)
        log.debug("Data pool updated for wwv")
    else:
        log.warning("No WWV geophysical alert data returned")


async def _fetch_and_publish_conditions(
    session: aiohttp.ClientSession,
    loc: ObservationLocation,
    eccc_cache: dict[str, dict[str, Any] | None],
    twc_cfg: dict[str, Any],
    location_point_cache: dict[str, TWCLocationPoint | None],
) -> None:
    wx_dict: dict[str, Any]

    if loc.source == "eccc":
        raw = eccc_cache.get(loc.id)
        if raw is None:
            log.warning("No ECCC data available for observation location %s", loc.id)
            return
        wx_dict = _build_eccc_conditions_dict(raw.get("properties", {}))

    elif loc.source == "nws":
        raw = await _fetch_nws_raw(session, loc.id)
        if raw is None:
            log.warning("No NWS data returned for observation location %s", loc.id)
            return
        raw_props = raw.get("properties", {})
        location_name = _twc_location_name(
            location_point_cache.get(loc.id),
            str(raw_props.get("name") or raw_props.get("stationIdentifier") or loc.id),
        )
        wx_dict = _build_nws_conditions_dict(raw_props, location_name=location_name)

    elif loc.source == "twc":
        raw = await _fetch_twc_raw(session, loc.id, twc_cfg)
        if raw is None:
            log.warning("No TWC data returned for observation location %s", loc.id)
            return
        location_name = _twc_location_name(location_point_cache.get(loc.id), loc.id)
        wx_dict = _build_twc_conditions_dict(raw, loc.id, location_name=location_name)

    else:
        log.warning("Unsupported source '%s' for observation location %s", loc.source, loc.id)
        return

    await asyncio.to_thread(_write_json, _conditions_cache_path(loc.source, loc.id), wx_dict)

    update_data_pool(f"{loc.feed_id}:{loc.id}", wx_dict, notify=False)
    log.debug("Data pool updated: %s:%s (%s)", loc.feed_id, loc.id, loc.source)


async def _fetch_and_publish_forecast(
    session: aiohttp.ClientSession,
    loc: ForecastLocation,
    eccc_cache: dict[str, dict[str, Any] | None],
    location_point_cache: dict[str, TWCLocationPoint | None],
) -> None:
    normalized: dict[str, Any]

    if loc.source == "eccc":
        raw = eccc_cache.get(loc.id)
        if raw is None:
            log.warning("No ECCC data available for forecast location %s", loc.id)
            return
        forecast_group = (raw.get("properties") or {}).get("forecastGroup") or {}
        if not forecast_group:
            log.warning("No forecastGroup in ECCC data for %s", loc.id)
            return
        normalized = {
            "source": loc.source,
            "forecast": _normalize_eccc_forecast(forecast_group),
            "forecast_region": loc.forecast_region,
            "name": _forecast_location_name_block(loc),
        }

    elif loc.source == "nws":
        base_code, _ = _parse_forecast_region(loc.forecast_region)
        raw = await _fetch_nws_zone_forecast(session, base_code)
        if raw is None:
            log.warning("No NWS zone forecast for %s (zone %s)", loc.id, base_code)
            return
        normalized = _build_nws_forecast_dict(raw, loc, location_point_cache.get(loc.id))
        normalized["source"] = loc.source

    else:
        log.warning("Unsupported source '%s' for forecast location %s", loc.source, loc.id)
        return

    await asyncio.to_thread(_write_json, _forecast_cache_path(loc.source, loc.id), normalized)

    update_data_pool(f"{loc.feed_id}:forecast:{loc.id}", normalized, notify=False)
    log.debug("Forecast published: %s:forecast:%s (region %s, source %s)", loc.feed_id, loc.id, loc.forecast_region, loc.source)


async def _fetch_and_publish_climate(
    session: aiohttp.ClientSession,
    loc: ClimateLocation,
    eccc_cache: dict[str, dict[str, Any] | None],
    ltce_temp_cache: dict[str, dict[str, Any] | None],
    ltce_precip_cache: dict[str, dict[str, Any] | None],
    ltce_snow_cache: dict[str, dict[str, Any] | None],
) -> None:
    if loc.source != "eccc":
        return

    raw = eccc_cache.get(loc.citypage_id)
    if raw is None:
        return

    raw_props = (raw.get("properties") or {}) if isinstance(raw, dict) else {}
    location_name = str(_ev(raw_props, "name") or "").strip()
    province_code = _province_code_from_citypage_id(loc.citypage_id)
    observed_props = None
    if location_name:
        observed_props = await _fetch_eccc_climate_daily(session, location_name, province_code)

    month, day = _feed_local_month_day(loc.timezone)
    ltce_key = f"{loc.id}:{month}:{day}"

    ltce_temp_props = None
    ltce_temp_raw = ltce_temp_cache.get(ltce_key)
    if ltce_temp_raw is not None:
        ltce_temp_props = (ltce_temp_raw.get("properties") or {}) if isinstance(ltce_temp_raw, dict) else None

    ltce_precip_props = None
    ltce_precip_raw = ltce_precip_cache.get(ltce_key)
    if ltce_precip_raw is not None:
        ltce_precip_props = (ltce_precip_raw.get("properties") or {}) if isinstance(ltce_precip_raw, dict) else None

    ltce_snow_props = None
    ltce_snow_raw = ltce_snow_cache.get(ltce_key)
    if ltce_snow_raw is not None:
        ltce_snow_props = (ltce_snow_raw.get("properties") or {}) if isinstance(ltce_snow_raw, dict) else None

    climate = _build_eccc_climate_dict(
        raw_props,
        ltce_temp_props,
        ltce_precip_props,
        ltce_snow_props,
        observed_props,
        loc.timezone,
    )
    if climate is None:
        return

    await asyncio.to_thread(_write_json, _climate_cache_path(loc.source, loc.id), climate)
    update_data_pool(f"{loc.feed_id}:climate:{loc.id}", climate, notify=False)
    log.debug("Climate published: %s:climate:%s (citypage %s)", loc.feed_id, loc.id, loc.citypage_id)


async def _fetch_and_publish_aqhi(
    session: aiohttp.ClientSession,
    loc: AirQualityLocation,
) -> None:
    if loc.source != "eccc":
        log.warning("Unsupported source '%s' for air quality location %s", loc.source, loc.id)
        return

    obs_raw, fcst_raw = await asyncio.gather(
        _fetch_eccc_aqhi_observation_raw(session, loc.id),
        _fetch_eccc_aqhi_forecast_raw(session, loc.id),
    )

    observation = _parse_eccc_aqhi_observation(obs_raw, loc.id) if obs_raw is not None else None
    forecast = _parse_eccc_aqhi_forecast(fcst_raw, loc.id) if fcst_raw is not None else None

    if observation is None and forecast is None:
        log.warning("No AQHI data returned for %s", loc.id)
        return

    aqhi_dict: dict[str, Any] = {
        "source": "eccc",
        **(observation or {}),
        "forecast": forecast,
    }

    await asyncio.to_thread(_write_json, _aqhi_cache_path(loc.id), aqhi_dict)
    update_data_pool(f"{loc.feed_id}:aqhi:{loc.id}", aqhi_dict, notify=False)
    log.debug("AQHI published: %s:aqhi:%s", loc.feed_id, loc.id)


async def fetch_once(config: dict[str, Any]) -> None:
    obs_locs, forecast_locs, climate_locs, aqhi_locs = _parse_locations_config(config)
    twc_cfg = config.get("sources", {}).get("twc") or {}
    twc_cfg.setdefault("api_key", (config.get("twc_api_key") or ""))
    twc_cfg.setdefault("units", config.get("twc_units") or "m")
    twc_cfg.setdefault("language", config.get("language") or "en-CA")

    async with aiohttp.ClientSession() as session:
        eccc_ids = {loc.id for loc in obs_locs if loc.source == "eccc"} | {
            loc.id for loc in forecast_locs if loc.source == "eccc"
        } | {
            loc.citypage_id for loc in climate_locs if loc.source == "eccc"
        }
        eccc_cache: dict[str, dict[str, Any] | None] = {}
        if eccc_ids:
            await asyncio.gather(
                *[
                    asyncio.create_task(
                        _prefetch_eccc(session, loc_id, eccc_cache),
                        name=f"prefetch:eccc:{loc_id}",
                    )
                    for loc_id in eccc_ids
                ],
                return_exceptions=True,
            )

        location_lookup_ids = {loc.id for loc in obs_locs if loc.source in {"nws", "twc"}} | {
            loc.id for loc in forecast_locs if loc.source != "eccc"
        }
        location_point_cache: dict[str, TWCLocationPoint | None] = {}
        if location_lookup_ids:
            await asyncio.gather(
                *[
                    asyncio.create_task(
                        _prefetch_twc_location_point(session, loc_id, twc_cfg, location_point_cache),
                        name=f"prefetch:twc-location:{loc_id}",
                    )
                    for loc_id in location_lookup_ids
                ],
                return_exceptions=True,
            )

        ltce_requests = {
            (loc.id, *_feed_local_month_day(loc.timezone))
            for loc in climate_locs
            if loc.source == "eccc"
        }
        ltce_temp_cache: dict[str, dict[str, Any] | None] = {}
        ltce_precip_cache: dict[str, dict[str, Any] | None] = {}
        ltce_snow_cache: dict[str, dict[str, Any] | None] = {}
        if ltce_requests:
            await asyncio.gather(
                *[
                    asyncio.create_task(
                        _prefetch_eccc_ltce_temperature(session, climate_id, month, day, ltce_temp_cache),
                        name=f"prefetch:ltce:{climate_id}:{month}:{day}",
                    )
                    for climate_id, month, day in ltce_requests
                ],
                *[
                    asyncio.create_task(
                        _prefetch_eccc_ltce_precipitation(session, climate_id, month, day, ltce_precip_cache),
                        name=f"prefetch:ltce-precip:{climate_id}:{month}:{day}",
                    )
                    for climate_id, month, day in ltce_requests
                ],
                *[
                    asyncio.create_task(
                        _prefetch_eccc_ltce_snowfall(session, climate_id, month, day, ltce_snow_cache),
                        name=f"prefetch:ltce-snow:{climate_id}:{month}:{day}",
                    )
                    for climate_id, month, day in ltce_requests
                ],
                return_exceptions=True,
            )

        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(
                _fetch_and_publish_conditions(session, loc, eccc_cache, twc_cfg, location_point_cache),
                name=f"conditions:{loc.source}:{loc.id}",
            )
            for loc in obs_locs
        ]
        tasks += [
            asyncio.create_task(
                _fetch_and_publish_forecast(session, loc, eccc_cache, location_point_cache),
                name=f"forecast:{loc.source}:{loc.id}",
            )
            for loc in forecast_locs
        ]
        tasks += [
            asyncio.create_task(
                _fetch_and_publish_climate(session, loc, eccc_cache, ltce_temp_cache, ltce_precip_cache, ltce_snow_cache),
                name=f"climate:{loc.source}:{loc.id}",
            )
            for loc in climate_locs
        ]
        tasks += [
            asyncio.create_task(
                _fetch_and_publish_aqhi(session, loc),
                name=f"aqhi:{loc.source}:{loc.id}",
            )
            for loc in aqhi_locs
        ]
        tasks += [
            asyncio.create_task(_fetch_and_publish_focn45(session), name="focn45"),
            asyncio.create_task(_fetch_and_publish_wwv(session), name="wwv"),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

    data_ready.set()


async def data_worker(config: dict[str, Any]) -> None:
    poll_interval: int = (
        config.get("network", {})
        .get("requests", {})
        .get("poll_interval_seconds", _DEFAULT_POLL_INTERVAL)
    )

    loop = asyncio.get_event_loop()
    stop = loop.run_in_executor(None, shutdown_event.wait)

    while not shutdown_event.is_set():
        await fetch_once(config)
        try:
            await asyncio.wait_for(asyncio.shield(stop), timeout=poll_interval)
            break
        except asyncio.TimeoutError:
            pass


def data_thread_worker(config: dict[str, Any]) -> None:
    asyncio.run(data_worker(config))


def _write_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
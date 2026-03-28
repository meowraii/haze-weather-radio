from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator

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
_DATA_DIR = os.path.join("data", "eccc")


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

    file_path = os.path.join(_DATA_DIR, f"{location_id}.json")
    await asyncio.to_thread(_write_json, file_path, raw)
    return raw


async def fetch_focn45(session: aiohttp.ClientSession) -> FOBulletin | None:
    from zoneinfo import ZoneInfo
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
            "No FO bulletins matched for %s (numberMatched=%d)",
            ct_date,
            data.get("numberMatched", 0),
        )
        log.info("Trying yesterday's date as fallback")
        ct_date_yesterday = (datetime.datetime.now(ZoneInfo("America/Chicago")) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        collection_url = _BULLETINS_URL.format(date=ct_date_yesterday)
        try:
            async with session.get(collection_url, timeout=_TIMEOUT) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except aiohttp.ClientError as e:
            log.error("Failed to fetch FOCN45 collection for fallback date %s: %s", ct_date_yesterday, e)
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

    bulletin = FOBulletin(
        identifier=identifier,
        issued_at=issued_at,
        url=bulletin_url,
        text=text,
    )

    file_path = os.path.join(_DATA_DIR, "focn45.cwwg.txt")
    await asyncio.to_thread(_write_text, file_path, text)
    return bulletin

async def fetch_geophysical_alerts(session: aiohttp.ClientSession) -> str | None:
    url = "https://services.swpc.noaa.gov/text/wwv.txt"
    file_path = os.path.join(_DATA_DIR, "..", "nws", "wwv.txt")
    try:
        async with session.get(url, timeout=_TIMEOUT) as resp:
            resp.raise_for_status()
            text = await resp.text()
            await asyncio.to_thread(_write_text, file_path, text)
    except aiohttp.ClientError as e:
        log.error("Failed to fetch geophysical alerts: %s", e)
        return None
    return text


async def iter_locations(config: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
    for feed in config.get("feeds", []):
        if not feed.get("enabled", True):
            continue
        feed_id = feed["id"]
        data_source: str = feed.get("data_source", "eccc")
        for loc in feed.get("locations", []):
            if not isinstance(loc, dict):
                continue
            yield {"feed_id": feed_id, "data_source": data_source, "location": loc}


async def fetch_once(config: dict[str, Any]) -> None:
    tasks: list[asyncio.Task[None]] = []
    async with aiohttp.ClientSession() as session:
        async for entry in iter_locations(config):
            feed_id: str = entry["feed_id"]
            data_source: str = entry["data_source"]
            loc: dict[str, Any] = entry["location"]
            loc_id: str | None = loc.get("eccc_id") if data_source == "eccc" else None
            loc_name: str = loc.get("name", loc_id or "unknown")
            if not loc_id:
                log.warning("No location ID for %s in feed %s", loc_name, feed_id)
                continue
            log.info("Fetching conditions for %s (%s) via %s", loc_name, loc_id, data_source)
            tasks.append(
                asyncio.create_task(
                    _fetch_and_publish_conditions(session, feed_id, loc, loc_id, data_source),
                    name=f"conditions:{loc_id}",
                )
            )
        tasks.append(
            asyncio.create_task(
                _fetch_and_publish_focn45(session),
                name="focn45",
            )
        )
        tasks.append(
            asyncio.create_task(
                _fetch_and_publish_wwv(session),
                name="wwv",
            )
        )
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


def _normalize_hourly(h: dict[str, Any]) -> dict[str, Any]:
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


def _normalize_forecast(forecast_group: dict[str, Any]) -> dict[str, Any]:
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
    return {"forecast": periods}


def _build_conditions_dict(raw_props: dict[str, Any], source: str = "eccc") -> dict[str, Any]:
    cc = raw_props.get("currentConditions", {})
    wind_raw = cc.get("wind", {}) or {}
    hfg = raw_props.get("hourlyForecastGroup", {}) or {}
    return {
        "source": source,
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
        },
        "hourly": [_normalize_hourly(h) for h in hfg.get("hourlyForecasts", [])],
    }


async def _fetch_and_publish_conditions(
    session: aiohttp.ClientSession,
    feed_id: str,
    loc: dict[str, Any],
    loc_id: str,
    data_source: str = "eccc",
) -> None:
    if data_source == "eccc":
        raw = await _fetch_eccc_raw(session, loc_id)
    else:
        log.warning("Unsupported data_source '%s' for %s", data_source, loc_id)
        return

    if raw is None:
        log.warning("No data returned for %s", loc_id)
        return

    raw_props = raw.get("properties", {})
    wx_dict = _build_conditions_dict(raw_props, source=data_source)
    wx_path = os.path.join(_DATA_DIR, f"{loc_id}.wx.json")
    await asyncio.to_thread(_write_json, wx_path, wx_dict)

    generate_forecast = loc.get("generate_forecast", False)

    pool_key = f"{feed_id}:{loc_id}"
    update_data_pool(pool_key, wx_dict, notify=False)
    log.debug("Data pool updated for %s", pool_key)

    if generate_forecast:
        forecast_group = raw_props.get("forecastGroup", {})
        if forecast_group:
            normalized_forecast = _normalize_forecast(forecast_group)
            update_data_pool(f"{feed_id}:forecast:{loc_id}", normalized_forecast, notify=False)
            log.debug("Forecast data published for %s/%s", feed_id, loc_id)


async def _fetch_and_publish_focn45(session: aiohttp.ClientSession) -> None:
    bulletin = await fetch_focn45(session)
    if bulletin:
        update_data_pool("focn45", bulletin, notify=False)
        log.debug("Data pool updated for focn45: %s", bulletin.identifier)
    else:
        log.warning("No FOCN45 bulletin returned")


async def _fetch_and_publish_wwv(session: aiohttp.ClientSession) -> None:
    text = await fetch_geophysical_alerts(session)
    if text:
        update_data_pool("wwv", text, notify=False)
        log.debug("Data pool updated for wwv")
    else:
        log.warning("No WWV geophysical alert data returned")


def _write_json(path: str, data: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def data_thread_worker(config: dict[str, Any]) -> None:
    asyncio.run(data_worker(config))
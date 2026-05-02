from __future__ import annotations

from typing import Any


def _playout_routine_enabled(feed: dict[str, Any]) -> bool:
    playout = feed.get('playout')
    if isinstance(playout, dict) and 'routine' in playout:
        return bool(playout.get('routine', True))
    return True


def coverage_regions(feed: dict[str, Any]) -> list[dict[str, Any]]:
    regions = feed.get('coverage')
    return [region for region in regions if isinstance(region, dict)] if isinstance(regions, list) else []


def observation_locations(feed: dict[str, Any]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for block in feed.get('locations', []):
        if not isinstance(block, dict):
            continue
        for entry in block.get('observationLocations', []):
            if isinstance(entry, dict):
                locations.append(entry)
    return locations


def forecast_locations(feed: dict[str, Any]) -> list[dict[str, Any]]:
    if not _playout_routine_enabled(feed):
        return []

    coverage = coverage_regions(feed)
    if coverage:
        locations: list[dict[str, Any]] = []
        for region in coverage:
            forecast_id = str(region.get('derive_forecast') or '').strip() or str(region.get('id') or '').strip()
            if str(region.get('coverage_type') or 'region').strip().lower() != 'region':
                continue
            if not forecast_id:
                continue
            location: dict[str, Any] = {'id': forecast_id}
            if (source := region.get('source')):
                location['source'] = source
            if (region_id := str(region.get('id') or '').strip()):
                location['forecast_region'] = region_id
            if (name_override := region.get('name_override')):
                location['name_override'] = name_override
            locations.append(location)
        return locations

    locations: list[dict[str, Any]] = []
    for block in feed.get('locations', []):
        if not isinstance(block, dict):
            continue
        for entry in block.get('forecastLocations', []):
            if isinstance(entry, dict):
                locations.append(entry)
    return locations


def climate_locations(feed: dict[str, Any]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for block in feed.get('locations', []):
        if not isinstance(block, dict):
            continue
        for entry in block.get('climateLocations', []):
            if isinstance(entry, dict):
                locations.append(entry)
    return locations


def air_quality_locations(feed: dict[str, Any]) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for block in feed.get('locations', []):
        if not isinstance(block, dict):
            continue
        for entry in block.get('airQualityLocations', []):
            if isinstance(entry, dict):
                locations.append(entry)
    return locations


def location_label(loc: dict[str, Any], fallback_id: str | None = None) -> str | None:
    return loc.get('name_override') or loc.get('name') or fallback_id or loc.get('id')


def forecast_name(loc: dict[str, Any]) -> str | None:
    return loc.get('name_override') or loc.get('name')


def current_conditions_name(loc: dict[str, Any]) -> str | None:
    return loc.get('name_override') or loc.get('name')


def climate_name(loc: dict[str, Any]) -> str | None:
    return loc.get('name_override') or loc.get('name')


def air_quality_name(loc: dict[str, Any]) -> str | None:
    return loc.get('name_override') or loc.get('name')


def feed_languages(feed: dict[str, Any]) -> list[str]:
    languages_cfg = feed.get('languages')
    if isinstance(languages_cfg, dict) and languages_cfg:
        return list(languages_cfg.keys())
    return [feed.get('language', 'en-CA')]


def enabled_feeds(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [f for f in config.get('feeds', []) if f.get('enabled', True)]

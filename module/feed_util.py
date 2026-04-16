from __future__ import annotations

from typing import Any


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

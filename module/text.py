# text.py takes the json files from the data folder and generates SSML text packages for the TTS engine. It also loads the configuration from config.yaml.

import pathlib
import yaml
from typing import Any, Optional


def load_config() -> dict[str, Any]:
    config_path = pathlib.Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_package(
    config: dict[str, Any],
    feed_id: str,
    package_type: str,
    weather_data: Optional[dict[str, Any]] = None,
) -> str:
    from managed.packages import current_conditions_package, date_time_package, station_id

    feed = next((f for f in config.get('feeds', []) if f['id'] == feed_id), None)
    lang = feed.get('language', 'en-CA') if feed else 'en-CA'
    tz = feed.get('timezone', 'UTC') if feed else 'UTC'

    if package_type == 'date_time':
        return date_time_package(tz, lang)
    if package_type == 'station_id':
        return station_id(config, feed_id, lang)
    if package_type == 'current_conditions':
        location_sets = config.get('location_sets', [])
        loc_set = next((ls for ls in location_sets if ls['id'] == feed_id), None)
        loc_name = loc_set.get('primary', {}).get('name') if loc_set else None
        return current_conditions_package(weather_data, loc_name, lang)
    return ''

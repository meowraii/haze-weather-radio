#!/usr/bin/env python3
"""Build a high-confidence alert geocode crosswalk for Haze.

The canonical output is SQLite because this data is relational:

* one SGC/CAP-CP geocode can map to one CLC forecast/base zone;
* one NWS SAME county code can map to multiple NWS zones;
* one area can have many postal/ZIP codes and nearby stations;
* every inferred link needs confidence, distance, and method metadata.

CSV and XML exports are generated as interoperability/debug artifacts, but
SQLite should be the runtime source once Haze starts querying this directly.

Optional accuracy boosters:

* rapidfuzz: high-quality token matching
* sentence-transformers: neural reranking of top candidates, opt-in with
  --neural because it may download a model
* geopy: online Nominatim fallback for rows missing coordinates, opt-in with
  --online-geocode and cached locally
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - optional dependency
    fuzz = None

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - optional dependency
    cKDTree = None

EARTH_RADIUS_KM = 6371.0088

STOP_WORDS = {
    "a",
    "and",
    "area",
    "areas",
    "city",
    "co",
    "county",
    "de",
    "des",
    "district",
    "du",
    "including",
    "la",
    "le",
    "les",
    "municipal",
    "municipality",
    "near",
    "of",
    "region",
    "regional",
    "rural",
    "secteur",
    "the",
    "town",
    "village",
}

CANADA_PROVINCES = {
    "Alberta": "AB",
    "British Columbia": "BC",
    "Manitoba": "MB",
    "New Brunswick": "NB",
    "Newfoundland and Labrador": "NL",
    "Northwest Territories": "NT",
    "Nova Scotia": "NS",
    "Nunavut": "NU",
    "Ontario": "ON",
    "Prince Edward Island": "PE",
    "Quebec": "QC",
    "Saskatchewan": "SK",
    "Yukon": "YT",
}


@dataclass(slots=True)
class Place:
    source: str
    code: str
    name: str
    name_fr: str = ""
    region: str = ""
    country: str = ""
    kind: str = ""
    lat: float | None = None
    lon: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        bits = [self.name, self.region, self.country]
        return ", ".join(bit for bit in bits if bit)


@dataclass(slots=True)
class Link:
    link_type: str
    from_source: str
    from_code: str
    to_source: str
    to_code: str
    score: float
    confidence: str
    distance_km: float | None
    method: str
    components: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PostalLink:
    country: str
    postal_code: str
    city: str
    region: str
    lat: float
    lon: float
    area_source: str
    area_code: str
    distance_km: float
    score: float


@dataclass(slots=True)
class StationLink:
    area_source: str
    area_code: str
    station_id: str
    station_name: str
    distance_km: float


@dataclass(slots=True)
class NWSCountyZone:
    fips: str
    same_code: str
    state: str
    county_name: str
    zone_ugc: str
    zone_name: str
    cwa: str
    timezone_code: str
    lat: float | None
    lon: float | None


class NeuralReranker:
    def __init__(self, enabled: bool, model_name: str) -> None:
        self.enabled = enabled
        self.model_name = model_name
        self.model = None
        self.cache: dict[str, Any] = {}

    def available(self) -> bool:
        return self.enabled

    def _load(self) -> bool:
        if not self.enabled:
            return False
        if self.model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            return True
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"warning: neural matcher unavailable: {exc}", file=sys.stderr)
            self.enabled = False
            return False

    def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates or not self._load():
            return [0.0 for _ in candidates]
        texts = [query, *candidates]
        missing = [text for text in texts if text not in self.cache]
        if missing:
            vectors = self.model.encode(
                missing,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for text, vector in zip(missing, vectors):
                self.cache[text] = vector
        query_vec = self.cache[query]
        scores = []
        for text in candidates:
            scores.append(float(query_vec @ self.cache[text]))
        return scores


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    input_dir = args.input_dir
    if input_dir is None:
        util_base = Path(__file__).resolve().parent / "csv_base"
        bundle_base = repo_root / "bundle" / "managed" / "csv"
        input_dir = util_base if util_base.exists() else bundle_base
    output_dir = args.output_dir or (Path(__file__).resolve().parent / "csv_merged")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"input:  {input_dir}")
    print(f"output: {output_dir}")

    if args.download_nws_marine_zones:
        input_dir.mkdir(parents=True, exist_ok=True)
        download_nws_marine_zones(input_dir / args.nws_marine_csv_name, args.nws_marine_source_url, args.user_agent)

    clc = load_clc(input_dir / "CLC_Base_Zone.csv")
    sgc = load_sgc(input_dir / "CAP-CP_Geocodes.csv")
    forecasts = load_forecasts(input_dir / "FORECAST_LOCATIONS.csv")
    stations = load_stations(input_dir / "SWOB_STATIONS.csv")
    postal_codes = [] if args.no_postal else load_postal_codes(input_dir / "postal-codes-canada.csv")
    zip_codes = load_zip_codes(args.zip_csv) if args.zip_csv else []
    nws_places, nws_links = load_nws(input_dir / "NWS_ZONE_COUNTY_CORRELATION.csv")
    marine_places, marine_links = load_nws_marine_zones(input_dir / args.nws_marine_csv_name)

    all_places: list[Place] = []
    all_places.extend(clc)
    all_places.extend(sgc)
    all_places.extend(forecasts)
    all_places.extend(nws_places)
    all_places.extend(marine_places)
    all_places.extend(stations)

    if args.online_geocode:
        geocoder = OnlineGeocoder(output_dir / "geocode_cache.sqlite", args)
        geocoder.fill_missing(all_places)

    neural = NeuralReranker(args.neural, args.neural_model)
    links: list[Link] = []
    links.extend(match_sgc_to_clc(sgc, clc, neural, args))
    links.extend(match_forecasts_to_clc(forecasts, clc, neural, args))
    links.extend(nws_links)
    links.extend(marine_links)

    postal_links: list[PostalLink] = []
    if postal_codes:
        postal_links.extend(match_postal_to_areas(postal_codes, clc, args.max_postal_distance_km))
    if zip_codes:
        nws_same = [place for place in nws_places if place.source == "nws_same"]
        postal_links.extend(match_postal_to_areas(zip_codes, nws_same, args.max_zip_distance_km))

    canonical_areas = [
        *clc,
        *[place for place in nws_places if place.source in {"nws_same", "nws_zone"}],
        *[place for place in marine_places if place.source in {"nws_marine_same", "nws_marine_zone"}],
    ]
    station_links = match_nearest_stations(canonical_areas, stations, args.max_station_distance_km)
    low_confidence = [link for link in links if link.confidence in {"review", "low"}]

    sqlite_path = output_dir / args.sqlite_name
    write_sqlite(sqlite_path, all_places, links, postal_links, station_links)
    if not args.no_derived_csv:
        write_csvs(output_dir, all_places, links, postal_links, station_links, low_confidence)
    if args.xml:
        write_xml(output_dir / args.xml_name, all_places, links, postal_links, station_links)

    print(f"places:          {len(all_places):,}")
    print(f"links:           {len(links):,}")
    print(f"postal/zip links:{len(postal_links):,}")
    print(f"station links:   {len(station_links):,}")
    print(f"review links:    {len(low_confidence):,}")
    print(f"sqlite:          {sqlite_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge CLC, SGC/CAP-CP, NWS SAME/zone, postal, ZIP, and station references.",
    )
    parser.add_argument("--input-dir", type=Path, help="Directory containing base CSV files.")
    parser.add_argument("--output-dir", type=Path, help="Directory for SQLite/CSV/XML outputs.")
    parser.add_argument("--sqlite-name", default="alert_location_map.sqlite")
    parser.add_argument("--no-derived-csv", action="store_true", help="Only emit SQLite; skip derived CSV exports.")
    parser.add_argument("--xml", action="store_true", help="Also emit XML export.")
    parser.add_argument("--xml-name", default="alert_location_map.xml")
    parser.add_argument("--neural", action="store_true", help="Use sentence-transformers to rerank top fuzzy/spatial candidates.")
    parser.add_argument("--neural-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--online-geocode", action="store_true", help="Use geopy/Nominatim for rows missing lat/lon.")
    parser.add_argument("--geocoder-user-agent", default="haze-weather-radio-geocode-merge")
    parser.add_argument("--geocoder-min-delay", type=float, default=1.1)
    parser.add_argument("--download-nws-marine-zones", action="store_true", help="Download official NWS marine zone/SAME pairs into the input CSV directory.")
    parser.add_argument("--nws-marine-csv-name", default="NWS_MARINE_ZONES.csv")
    parser.add_argument("--nws-marine-source-url", default="https://www.weather.gov/source/gis/Shapefiles/WSOM/mareas20fe25.txt")
    parser.add_argument("--user-agent", default="haze-weather-radio-geocode-merge")
    parser.add_argument("--no-postal", action="store_true", help="Skip Canadian postal-code linking.")
    parser.add_argument("--zip-csv", type=Path, help="Optional US ZIP CSV with zip, city, state, latitude, longitude columns.")
    parser.add_argument("--max-postal-distance-km", type=float, default=85.0)
    parser.add_argument("--max-zip-distance-km", type=float, default=125.0)
    parser.add_argument("--max-station-distance-km", type=float, default=450.0)
    parser.add_argument("--review-threshold", type=float, default=0.72)
    parser.add_argument("--high-threshold", type=float, default=0.88)
    return parser.parse_args()


def load_clc(path: Path) -> list[Place]:
    rows = read_table(path, header_markers={"CLC"})
    out = []
    for row in rows:
        code = clean_code(row.get("clc"))
        if not code:
            continue
        out.append(
            Place(
                source="clc",
                code=code.zfill(6),
                name=clean_text(row.get("name") or row.get("english")),
                name_fr=clean_text(row.get("nom") or row.get("french")),
                region=clean_region(row.get("province_c")),
                country=clean_region(row.get("country_c") or "CA"),
                kind=clean_text(row.get("usage") or row.get("kind") or "CLCBaseZone"),
                lat=parse_float(row.get("lat_dd")),
                lon=parse_float(row.get("lon_dd")),
                attrs={
                    "feature_id": clean_text(row.get("feature_id") or row.get("uuid")),
                    "area_km2": clean_text(row.get("area_km2")),
                    "perim_km": clean_text(row.get("perim_km")),
                },
            )
        )
    return unique_places(out)


def load_sgc(path: Path) -> list[Place]:
    rows = read_table(path, header_markers={"CAPCPGCODE"})
    out = []
    for row in rows:
        code = clean_code(row.get("capcpgcode") or row.get("sgc"))
        if not code:
            continue
        out.append(
            Place(
                source="sgc",
                code=code,
                name=clean_text(row.get("name")),
                name_fr=clean_text(row.get("nom")),
                region=clean_region(row.get("province_c")),
                country=clean_region(row.get("country_c") or "CA"),
                kind="CAP-CP/NAADS SGC geocode",
                lat=parse_float(row.get("lat_dd")),
                lon=parse_float(row.get("lon_dd")),
                attrs={"waterbody": clean_text(row.get("watrbody_c"))},
            )
        )
    return unique_places(out)


def load_forecasts(path: Path) -> list[Place]:
    rows = read_table(path, header_markers={"CODE", "PROGRAMS"})
    seen = set()
    out = []
    for row in rows:
        code = clean_code(row.get("code"))
        if not code:
            continue
        programs = clean_text(row.get("programs"))
        key = (code, programs)
        if key in seen:
            continue
        seen.add(key)
        province = first_region(row.get("province_waterbody_2_province_plan_d_eau_2"))
        out.append(
            Place(
                source="forecast",
                code=code.zfill(6),
                name=clean_text(row.get("name")),
                name_fr=clean_text(row.get("nom")),
                region=province,
                country="CA",
                kind=programs or "forecast_location",
                attrs={
                    "programs": programs,
                    "programmes": clean_text(row.get("programmes")),
                    "province_raw": clean_text(row.get("province_waterbody_2_province_plan_d_eau_2")),
                },
            )
        )
    return unique_places(out)


def load_stations(path: Path) -> list[Place]:
    rows = read_table(path, header_markers={"IATA_ID"})
    out = []
    for row in rows:
        station_id = clean_text(row.get("iata_id"))
        if not station_id:
            continue
        province_name = clean_text(row.get("province_territory"))
        out.append(
            Place(
                source="station",
                code=station_id.upper(),
                name=clean_text(row.get("name")),
                region=CANADA_PROVINCES.get(province_name, province_name),
                country="CA",
                kind="SWOB station",
                lat=parse_float(row.get("latitude")),
                lon=parse_float(row.get("longitude")),
                attrs={
                    "wmo_id": clean_text(row.get("wmo_id")),
                    "msc_id": clean_text(row.get("msc_id")),
                    "elevation_m": clean_text(row.get("elevation_m")),
                    "provider": clean_text(row.get("data_provider")),
                    "network": clean_text(row.get("dataset_network")),
                    "mode": clean_text(row.get("auto_man")),
                    "icao": station_id.upper(),
                },
            )
        )
    return unique_places(out)


def load_postal_codes(path: Path) -> list[dict[str, Any]]:
    rows = read_table(path, header_markers={"postal-code", "postal_code"})
    out = []
    for row in rows:
        code = clean_text(row.get("postal_code") or row.get("postal_code_canada"))
        lat = parse_float(row.get("latitude"))
        lon = parse_float(row.get("longitude"))
        if not code or lat is None or lon is None:
            continue
        out.append(
            {
                "country": clean_region(row.get("country_code") or "CA"),
                "postal_code": normalize_postal(code),
                "city": clean_text(row.get("city")),
                "region": clean_region(row.get("province_code")),
                "lat": lat,
                "lon": lon,
            }
        )
    return out


def load_zip_codes(path: Path) -> list[dict[str, Any]]:
    rows = read_table(path, header_markers={"zip", "zipcode", "postal_code"})
    out = []
    for row in rows:
        code = clean_text(row.get("zip") or row.get("zipcode") or row.get("postal_code"))
        lat = parse_float(row.get("latitude") or row.get("lat"))
        lon = parse_float(row.get("longitude") or row.get("lon") or row.get("lng"))
        if not code or lat is None or lon is None:
            continue
        out.append(
            {
                "country": clean_region(row.get("country") or row.get("country_code") or "US"),
                "postal_code": code.zfill(5) if code.isdigit() else code,
                "city": clean_text(row.get("city") or row.get("place_name")),
                "region": clean_region(row.get("state") or row.get("state_code")),
                "lat": lat,
                "lon": lon,
            }
        )
    return out


def load_nws(path: Path) -> tuple[list[Place], list[Link]]:
    rows = read_table(path, header_markers={"STATE", "ZONE_CODE", "FIPS/SAME"})
    zone_places: dict[str, Place] = {}
    county_places: dict[str, Place] = {}
    direct_links: list[Link] = []
    seen_link = set()
    for row in rows:
        state = clean_region(row.get("state"))
        zone_code = clean_code(row.get("zone_code")).zfill(3)
        cwa = clean_text(row.get("cwa_id"))
        zone_name = clean_text(row.get("zone_name"))
        county_name = clean_text(row.get("county_name"))
        fips = clean_code(row.get("fips_same") or row.get("fips"))
        if not state or not zone_code:
            continue
        zone_ugc = f"{state}Z{zone_code}"
        lat = parse_float(row.get("center_latitude") or row.get("lat"))
        lon = parse_float(row.get("center_longitude") or row.get("lon"))
        zone_places.setdefault(
            zone_ugc,
            Place(
                source="nws_zone",
                code=zone_ugc,
                name=zone_name,
                region=state,
                country="US",
                kind="NWS forecast zone",
                lat=lat,
                lon=lon,
                attrs={"cwa": cwa, "zone_code": zone_code, "state_zone": f"{state}{zone_code}"},
            ),
        )
        if not fips:
            continue
        same = same_code_from_fips(fips)
        county = county_places.get(same)
        if county is None:
            county_places[same] = Place(
                source="nws_same",
                code=same,
                name=f"{county_name}, {state}" if county_name else same,
                region=state,
                country="US",
                kind="NWS SAME county/FIPS",
                lat=lat,
                lon=lon,
                attrs={"fips": fips, "same": same},
            )
        else:
            county.attrs.setdefault("zones", [])
            if zone_ugc not in county.attrs["zones"]:
                county.attrs["zones"].append(zone_ugc)
        link_key = (same, zone_ugc)
        if link_key not in seen_link:
            seen_link.add(link_key)
            direct_links.append(
                Link(
                    link_type="nws_same_to_zone",
                    from_source="nws_same",
                    from_code=same,
                    to_source="nws_zone",
                    to_code=zone_ugc,
                    score=1.0,
                    confidence="exact",
                    distance_km=0.0,
                    method="official_nws_zone_county_correlation",
                    components={
                        "fips": fips,
                        "state": state,
                        "county_name": county_name,
                        "zone_name": zone_name,
                        "cwa": cwa,
                        "timezone": clean_text(row.get("timezone")),
                    },
                )
            )
    return [*county_places.values(), *zone_places.values()], direct_links


def load_nws_marine_zones(path: Path) -> tuple[list[Place], list[Link]]:
    rows = read_table(path, header_markers={"ZONE_UGC", "SAME_CODE"})
    zone_places: dict[str, Place] = {}
    same_places: dict[str, Place] = {}
    direct_links: list[Link] = []
    seen_link = set()
    for row in rows:
        zone_ugc = clean_code(row.get("zone_ugc") or row.get("zone") or row.get("ugc"))
        same = same_code_from_fips(row.get("same_code") or row.get("same"))
        name = clean_text(row.get("name") or row.get("zone_name"))
        if not zone_ugc or not same or not name:
            continue
        region = clean_region(row.get("region") or row.get("basin"))
        operational = clean_text(row.get("operational") or "true").lower() not in {"0", "false", "no"}
        lat = parse_float(row.get("lat") or row.get("latitude"))
        lon = parse_float(row.get("lon") or row.get("longitude"))
        source_url = clean_text(row.get("source_url") or row.get("source"))
        zone_places.setdefault(
            zone_ugc,
            Place(
                source="nws_marine_zone",
                code=zone_ugc,
                name=name,
                region=region,
                country="US",
                kind="NWS marine forecast zone",
                lat=lat,
                lon=lon,
                attrs={
                    "same": same,
                    "source_url": source_url,
                    "operational_nwr": operational,
                },
            ),
        )
        same_places.setdefault(
            same,
            Place(
                source="nws_marine_same",
                code=same,
                name=name,
                region=region,
                country="US",
                kind="NWS marine SAME zone",
                lat=lat,
                lon=lon,
                attrs={
                    "zone_ugc": zone_ugc,
                    "source_url": source_url,
                    "operational_nwr": operational,
                },
            ),
        )
        link_key = (same, zone_ugc)
        if link_key in seen_link:
            continue
        seen_link.add(link_key)
        direct_links.append(
            Link(
                link_type="nws_marine_same_to_zone",
                from_source="nws_marine_same",
                from_code=same,
                to_source="nws_marine_zone",
                to_code=zone_ugc,
                score=1.0,
                confidence="exact",
                distance_km=0.0,
                method="official_nws_eas_nwr_mareas",
                components={
                    "name": name,
                    "region": region,
                    "source_url": source_url,
                    "operational_nwr": operational,
                },
            )
        )
    return [*same_places.values(), *zone_places.values()], direct_links


def download_nws_marine_zones(path: Path, source_url: str, user_agent: str) -> None:
    rows = parse_nws_marine_areas(fetch_text(source_url, user_agent), source_url)
    rows.sort(key=lambda item: (item["same_code"], item["zone_ugc"]))
    write_csv(
        path,
        ["region", "zone_ugc", "same_code", "name", "lon", "lat", "operational", "source_url"],
        rows,
    )
    print(f"downloaded NWS marine EAS/NWR zones: {len(rows):,} -> {path}")


def fetch_text(url: str, user_agent: str) -> str:
    request = Request(url, headers={"User-Agent": user_agent or "haze-weather-radio-geocode-merge"})
    with urlopen(request, timeout=30) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def parse_nws_marine_areas(text: str, source_url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in iter_marine_area_rows(text):
        alpha = clean_code(row[0])
        ssnum = clean_code(row[1])
        name = clean_text(row[2])
        if len(alpha) != 2 or len(ssnum) != 5 or not ssnum.isdigit() or not name:
            continue
        same = "0" + ssnum
        zone_ugc = f"{alpha}Z{ssnum[-3:]}"
        key = (same, zone_ugc)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "region": alpha,
                    "zone_ugc": zone_ugc,
                    "same_code": same,
                    "name": name,
                    "lon": clean_text(row[4]),
                    "lat": clean_text(row[3]),
                    "operational": "true",
                    "source_url": source_url,
                }
            )
    return rows


def iter_marine_area_rows(text: str) -> Iterable[list[str]]:
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if pending:
            pending += " " + line
        else:
            pending = line
        row = next(csv.reader([pending], delimiter="|"))
        if len(row) < 5:
            continue
        yield [cell.strip() for cell in row[:5]]
        pending = ""


def match_sgc_to_clc(sgc: list[Place], clc: list[Place], neural: NeuralReranker, args: argparse.Namespace) -> list[Link]:
    clc_by_code = {place.code: place for place in clc}
    clc_by_region = group_by_region(clc)
    links = []
    for item in sgc:
        if item.code in clc_by_code:
            target = clc_by_code[item.code]
            links.append(
                Link("sgc_to_clc", "sgc", item.code, "clc", target.code, 1.0, "exact", distance(item, target), "exact_code")
            )
            continue
        candidates = clc_by_region.get(item.region) or clc
        scored = score_candidates(item, candidates, neural, args, spatial_weight=0.62, name_weight=0.30, code_weight=0.08)
        if scored:
            links.append(scored[0])
    return links


def match_forecasts_to_clc(
    forecasts: list[Place],
    clc: list[Place],
    neural: NeuralReranker,
    args: argparse.Namespace,
) -> list[Link]:
    clc_by_code = {place.code: place for place in clc}
    clc_by_region = group_by_region(clc)
    links = []
    seen = set()
    for item in forecasts:
        key = item.code
        if key in seen:
            continue
        seen.add(key)
        if item.code in clc_by_code:
            target = clc_by_code[item.code]
            links.append(Link("forecast_to_clc", "forecast", item.code, "clc", target.code, 1.0, "exact", None, "exact_code"))
            continue
        candidates = clc_by_region.get(item.region) or clc
        scored = score_candidates(item, candidates, neural, args, spatial_weight=0.0, name_weight=0.88, code_weight=0.12)
        if scored:
            top = scored[0]
            links.append(
                Link(
                    "forecast_to_clc",
                    "forecast",
                    item.code,
                    "clc",
                    top.to_code,
                    top.score,
                    top.confidence,
                    top.distance_km,
                    top.method,
                    top.components,
                )
            )
    return links


def score_candidates(
    source: Place,
    candidates: list[Place],
    neural: NeuralReranker,
    args: argparse.Namespace,
    spatial_weight: float,
    name_weight: float,
    code_weight: float,
) -> list[Link]:
    rows = []
    for target in candidates:
        d_km = distance(source, target)
        spatial = spatial_score(d_km) if d_km is not None and spatial_weight else 0.0
        lexical = text_similarity(source.name, target.name)
        prefix = code_similarity(source.code, target.code)
        region = 1.0 if source.region and source.region == target.region else 0.0
        score = (spatial * spatial_weight) + (lexical * name_weight) + (prefix * code_weight) + (region * 0.02)
        rows.append(
            {
                "target": target,
                "distance_km": d_km,
                "spatial": spatial,
                "lexical": lexical,
                "prefix": prefix,
                "region": region,
                "score": min(1.0, score),
            }
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    rows = rows[:16]
    if neural.available() and rows:
        query = neural_label(source)
        neural_scores = neural.score(query, [neural_label(row["target"]) for row in rows])
        for row, neural_score in zip(rows, neural_scores):
            row["neural"] = max(0.0, min(1.0, (neural_score + 1.0) / 2.0))
            row["score"] = (row["score"] * 0.78) + (row["neural"] * 0.22)
        rows.sort(key=lambda item: item["score"], reverse=True)
    out = []
    for row in rows:
        target = row["target"]
        score = float(row["score"])
        out.append(
            Link(
                link_type=f"{source.source}_to_{target.source}",
                from_source=source.source,
                from_code=source.code,
                to_source=target.source,
                to_code=target.code,
                score=round(score, 6),
                confidence=confidence_label(score, args.high_threshold, args.review_threshold),
                distance_km=round(row["distance_km"], 3) if row["distance_km"] is not None else None,
                method="weighted_spatial_fuzzy_neural" if "neural" in row else "weighted_spatial_fuzzy",
                components={
                    "spatial": round(row["spatial"], 6),
                    "lexical": round(row["lexical"], 6),
                    "code": round(row["prefix"], 6),
                    "region": round(row["region"], 6),
                    "neural": round(row.get("neural", 0.0), 6),
                    "source_name": source.name,
                    "target_name": target.name,
                    "source_region": source.region,
                    "target_region": target.region,
                },
            )
        )
    return out


def match_postal_to_areas(postal_rows: list[dict[str, Any]], areas: list[Place], max_distance_km: float) -> list[PostalLink]:
    by_region = group_by_region([area for area in areas if area.lat is not None and area.lon is not None])
    tree_cache = {region: SpatialIndex(items) for region, items in by_region.items()}
    fallback = SpatialIndex([area for area in areas if area.lat is not None and area.lon is not None])
    out = []
    for row in postal_rows:
        index = tree_cache.get(row["region"]) or fallback
        target, d_km = index.nearest(row["lat"], row["lon"])
        if target is None or d_km > max_distance_km:
            continue
        score = spatial_score(d_km)
        out.append(
            PostalLink(
                country=row["country"],
                postal_code=row["postal_code"],
                city=row["city"],
                region=row["region"],
                lat=row["lat"],
                lon=row["lon"],
                area_source=target.source,
                area_code=target.code,
                distance_km=round(d_km, 3),
                score=round(score, 6),
            )
        )
    return out


def match_nearest_stations(areas: list[Place], stations: list[Place], max_distance_km: float) -> list[StationLink]:
    station_points = [station for station in stations if station.lat is not None and station.lon is not None]
    by_region = group_by_region(station_points)
    fallback = SpatialIndex(station_points)
    cache = {region: SpatialIndex(items) for region, items in by_region.items()}
    out = []
    for area in areas:
        if area.lat is None or area.lon is None:
            continue
        index = cache.get(area.region) or fallback
        station, d_km = index.nearest(area.lat, area.lon)
        if station is None or d_km > max_distance_km:
            continue
        out.append(
            StationLink(
                area_source=area.source,
                area_code=area.code,
                station_id=station.code,
                station_name=station.name,
                distance_km=round(d_km, 3),
            )
        )
    return out


class SpatialIndex:
    def __init__(self, places: list[Place]) -> None:
        self.places = places
        self.tree = None
        self.points = []
        if cKDTree is not None and places:
            self.points = [earth_vector(place.lat, place.lon) for place in places]
            self.tree = cKDTree(self.points)

    def nearest(self, lat: float, lon: float) -> tuple[Place | None, float]:
        if not self.places:
            return None, math.inf
        if self.tree is not None:
            _, index = self.tree.query(earth_vector(lat, lon), k=1)
            place = self.places[int(index)]
            return place, haversine_km(lat, lon, place.lat, place.lon)
        best = min(self.places, key=lambda item: haversine_km(lat, lon, item.lat, item.lon))
        return best, haversine_km(lat, lon, best.lat, best.lon)


class OnlineGeocoder:
    def __init__(self, cache_path: Path, args: argparse.Namespace) -> None:
        self.cache_path = cache_path
        self.args = args
        self.geocode = None
        self.conn = sqlite3.connect(cache_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS geocode_cache (query TEXT PRIMARY KEY, lat REAL, lon REAL, raw_json TEXT, updated_at TEXT)"
        )

    def fill_missing(self, places: list[Place]) -> None:
        try:
            from geopy.extra.rate_limiter import RateLimiter
            from geopy.geocoders import Nominatim
        except Exception as exc:  # pragma: no cover - optional dependency
            print(f"warning: geopy unavailable: {exc}", file=sys.stderr)
            return
        geolocator = Nominatim(user_agent=self.args.geocoder_user_agent)
        self.geocode = RateLimiter(
            geolocator.geocode,
            min_delay_seconds=self.args.geocoder_min_delay,
            swallow_exceptions=True,
            return_value_on_exception=None,
        )
        for place in places:
            if place.lat is not None and place.lon is not None:
                continue
            query = place.label()
            if not query:
                continue
            cached = self.cached(query)
            if cached:
                place.lat, place.lon = cached
                place.attrs["geocoded_online"] = True
                continue
            location = self.geocode(query, exactly_one=True, timeout=10)
            if location is None:
                continue
            place.lat = float(location.latitude)
            place.lon = float(location.longitude)
            place.attrs["geocoded_online"] = True
            self.store(query, place.lat, place.lon, getattr(location, "raw", {}))

    def cached(self, query: str) -> tuple[float, float] | None:
        row = self.conn.execute("SELECT lat, lon FROM geocode_cache WHERE query = ?", (query,)).fetchone()
        if row is None:
            return None
        return float(row[0]), float(row[1])

    def store(self, query: str, lat: float, lon: float, raw: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO geocode_cache(query, lat, lon, raw_json, updated_at) VALUES (?, ?, ?, ?, ?)",
            (query, lat, lon, json.dumps(raw, ensure_ascii=False), now_iso()),
        )
        self.conn.commit()


def write_sqlite(
    path: Path,
    places: list[Place],
    links: list[Link],
    postal_links: list[PostalLink],
    station_links: list[StationLink],
) -> None:
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(
        """
        CREATE TABLE metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE places (
            source TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            name_fr TEXT NOT NULL,
            region TEXT NOT NULL,
            country TEXT NOT NULL,
            kind TEXT NOT NULL,
            lat REAL,
            lon REAL,
            attrs_json TEXT NOT NULL,
            PRIMARY KEY (source, code)
        );
        CREATE TABLE links (
            link_type TEXT NOT NULL,
            from_source TEXT NOT NULL,
            from_code TEXT NOT NULL,
            to_source TEXT NOT NULL,
            to_code TEXT NOT NULL,
            score REAL NOT NULL,
            confidence TEXT NOT NULL,
            distance_km REAL,
            method TEXT NOT NULL,
            components_json TEXT NOT NULL
        );
        CREATE INDEX idx_links_from ON links(from_source, from_code);
        CREATE INDEX idx_links_to ON links(to_source, to_code);
        CREATE TABLE postal_links (
            country TEXT NOT NULL,
            postal_code TEXT NOT NULL,
            city TEXT NOT NULL,
            region TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            area_source TEXT NOT NULL,
            area_code TEXT NOT NULL,
            distance_km REAL NOT NULL,
            score REAL NOT NULL
        );
        CREATE INDEX idx_postal_code ON postal_links(country, postal_code);
        CREATE INDEX idx_postal_area ON postal_links(area_source, area_code);
        CREATE TABLE station_links (
            area_source TEXT NOT NULL,
            area_code TEXT NOT NULL,
            station_id TEXT NOT NULL,
            station_name TEXT NOT NULL,
            distance_km REAL NOT NULL,
            PRIMARY KEY (area_source, area_code)
        );
        """
    )
    conn.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", ("generated_at", now_iso()))
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        ("format_note", "SQLite is canonical; CSV/XML exports are derived artifacts."),
    )
    conn.executemany(
        """
        INSERT INTO places(source, code, name, name_fr, region, country, kind, lat, lon, attrs_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                place.source,
                place.code,
                place.name,
                place.name_fr,
                place.region,
                place.country,
                place.kind,
                place.lat,
                place.lon,
                json.dumps(place.attrs, ensure_ascii=False, sort_keys=True),
            )
            for place in places
        ],
    )
    conn.executemany(
        """
        INSERT INTO links(link_type, from_source, from_code, to_source, to_code, score, confidence, distance_km, method, components_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                link.link_type,
                link.from_source,
                link.from_code,
                link.to_source,
                link.to_code,
                link.score,
                link.confidence,
                link.distance_km,
                link.method,
                json.dumps(link.components, ensure_ascii=False, sort_keys=True),
            )
            for link in links
        ],
    )
    conn.executemany(
        """
        INSERT INTO postal_links(country, postal_code, city, region, lat, lon, area_source, area_code, distance_km, score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                item.country,
                item.postal_code,
                item.city,
                item.region,
                item.lat,
                item.lon,
                item.area_source,
                item.area_code,
                item.distance_km,
                item.score,
            )
            for item in postal_links
        ],
    )
    conn.executemany(
        """
        INSERT INTO station_links(area_source, area_code, station_id, station_name, distance_km)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (item.area_source, item.area_code, item.station_id, item.station_name, item.distance_km)
            for item in station_links
        ],
    )
    conn.commit()
    conn.execute("VACUUM")
    conn.close()


def write_csvs(
    output_dir: Path,
    places: list[Place],
    links: list[Link],
    postal_links: list[PostalLink],
    station_links: list[StationLink],
    low_confidence: list[Link],
) -> None:
    write_csv(
        output_dir / "locations.csv",
        ["source", "code", "name", "name_fr", "region", "country", "kind", "lat", "lon", "attrs_json"],
        [
            {
                "source": place.source,
                "code": place.code,
                "name": place.name,
                "name_fr": place.name_fr,
                "region": place.region,
                "country": place.country,
                "kind": place.kind,
                "lat": place.lat,
                "lon": place.lon,
                "attrs_json": json.dumps(place.attrs, ensure_ascii=False, sort_keys=True),
            }
            for place in places
        ],
    )
    link_rows = [link_to_row(link) for link in links]
    write_csv(
        output_dir / "links.csv",
        [
            "link_type",
            "from_source",
            "from_code",
            "to_source",
            "to_code",
            "score",
            "confidence",
            "distance_km",
            "method",
            "components_json",
        ],
        link_rows,
    )
    write_csv(output_dir / "low_confidence_review.csv", list(link_rows[0].keys()) if link_rows else [], [link_to_row(x) for x in low_confidence])
    write_csv(
        output_dir / "postal_links.csv",
        ["country", "postal_code", "city", "region", "lat", "lon", "area_source", "area_code", "distance_km", "score"],
        [asdict(item) for item in postal_links],
    )
    write_csv(
        output_dir / "station_links.csv",
        ["area_source", "area_code", "station_id", "station_name", "distance_km"],
        [asdict(item) for item in station_links],
    )
    nws_rows = [link_to_row(link) for link in links if link.link_type == "nws_same_to_zone"]
    write_csv(output_dir / "nws_county_zone_links.csv", list(nws_rows[0].keys()) if nws_rows else [], nws_rows)


def write_xml(
    path: Path,
    places: list[Place],
    links: list[Link],
    postal_links: list[PostalLink],
    station_links: list[StationLink],
) -> None:
    root = ET.Element("alertLocationMap", {"generatedAt": now_iso()})
    places_el = ET.SubElement(root, "places")
    for place in places:
        attrs = {
            "source": place.source,
            "code": place.code,
            "name": place.name,
            "region": place.region,
            "country": place.country,
            "kind": place.kind,
        }
        if place.name_fr:
            attrs["nameFr"] = place.name_fr
        if place.lat is not None:
            attrs["lat"] = f"{place.lat:.8f}"
        if place.lon is not None:
            attrs["lon"] = f"{place.lon:.8f}"
        item = ET.SubElement(places_el, "place", attrs)
        for key, value in sorted(place.attrs.items()):
            if value not in (None, ""):
                ET.SubElement(item, "attr", {"name": key, "value": str(value)})
    links_el = ET.SubElement(root, "links")
    for link in links:
        ET.SubElement(
            links_el,
            "link",
            {
                "type": link.link_type,
                "from": f"{link.from_source}:{link.from_code}",
                "to": f"{link.to_source}:{link.to_code}",
                "score": f"{link.score:.6f}",
                "confidence": link.confidence,
                "distanceKm": "" if link.distance_km is None else f"{link.distance_km:.3f}",
                "method": link.method,
            },
        )
    postal_el = ET.SubElement(root, "postalLinks")
    for item in postal_links:
        ET.SubElement(
            postal_el,
            "postal",
            {
                "country": item.country,
                "code": item.postal_code,
                "city": item.city,
                "region": item.region,
                "area": f"{item.area_source}:{item.area_code}",
                "distanceKm": f"{item.distance_km:.3f}",
            },
        )
    stations_el = ET.SubElement(root, "stationLinks")
    for item in station_links:
        ET.SubElement(
            stations_el,
            "station",
            {
                "area": f"{item.area_source}:{item.area_code}",
                "stationId": item.station_id,
                "stationName": item.station_name,
                "distanceKm": f"{item.distance_km:.3f}",
            },
        )
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def read_table(path: Path, header_markers: set[str]) -> list[dict[str, str]]:
    if not path.exists():
        print(f"warning: missing input {path}", file=sys.stderr)
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(8192)
        handle.seek(0)
        rows = list(csv.reader(handle, delimiter=detect_delimiter(sample)))
    header_index = 0
    markers = {normalize_header(marker) for marker in header_markers}
    for index, row in enumerate(rows):
        normalized = {normalize_header(cell) for cell in row}
        if normalized & markers:
            header_index = index
            break
    header = [normalize_header(value) for value in rows[header_index]]
    out = []
    for raw in rows[header_index + 1 :]:
        if not any(cell.strip() for cell in raw):
            continue
        item = {}
        for index, key in enumerate(header):
            if not key:
                continue
            item[key] = raw[index].strip() if index < len(raw) else ""
        out.append(item)
    return out


def detect_delimiter(sample: str) -> str:
    for line in sample.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.count("|") > stripped.count(","):
            return "|"
        return ","
    return ","


def normalize_header(value: str | None) -> str:
    value = clean_text(value).lower()
    value = value.replace("/", "_").replace("+", "_")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def link_to_row(link: Link) -> dict[str, Any]:
    return {
        "link_type": link.link_type,
        "from_source": link.from_source,
        "from_code": link.from_code,
        "to_source": link.to_source,
        "to_code": link.to_code,
        "score": link.score,
        "confidence": link.confidence,
        "distance_km": link.distance_km,
        "method": link.method,
        "components_json": json.dumps(link.components, ensure_ascii=False, sort_keys=True),
    }


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def clean_code(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", clean_text(value)).upper()


def clean_region(value: Any) -> str:
    value = clean_text(value).upper()
    if len(value) > 2 and value in {name.upper() for name in CANADA_PROVINCES}:
        for name, code in CANADA_PROVINCES.items():
            if value == name.upper():
                return code
    return value


def first_region(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    return clean_region(re.split(r"[,;/]", text)[0])


def parse_float(value: Any) -> float | None:
    text = clean_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_postal(value: str) -> str:
    value = clean_code(value)
    if len(value) == 6 and value[:1].isalpha():
        return value[:3] + " " + value[3:]
    return value


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    tokens = [token for token in value.split() if token not in STOP_WORDS]
    return " ".join(tokens)


def text_similarity(left: str, right: str) -> float:
    left_n = normalize_name(left)
    right_n = normalize_name(right)
    if not left_n or not right_n:
        return 0.0
    if left_n == right_n:
        return 1.0
    if fuzz is not None:
        return max(
            fuzz.token_set_ratio(left_n, right_n),
            fuzz.partial_ratio(left_n, right_n),
            fuzz.WRatio(left_n, right_n),
        ) / 100.0
    return sequence_ratio(left_n, right_n)


def sequence_ratio(left: str, right: str) -> float:
    from difflib import SequenceMatcher

    return SequenceMatcher(None, left, right).ratio()


def code_similarity(left: str, right: str) -> float:
    left = clean_code(left)
    right = clean_code(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    prefix = 0
    for a, b in zip(left, right):
        if a != b:
            break
        prefix += 1
    return prefix / max(len(left), len(right))


def confidence_label(score: float, high: float, review: float) -> str:
    if score >= 0.985:
        return "exact"
    if score >= high:
        return "high"
    if score >= review:
        return "review"
    return "low"


def spatial_score(distance_km: float | None) -> float:
    if distance_km is None:
        return 0.0
    if distance_km <= 2:
        return 1.0
    if distance_km <= 10:
        return 0.98
    if distance_km <= 25:
        return 0.93
    if distance_km <= 50:
        return 0.84
    if distance_km <= 100:
        return 0.68
    if distance_km <= 250:
        return 0.42
    return max(0.0, math.exp(-distance_km / 220.0))


def group_by_region(places: list[Place]) -> dict[str, list[Place]]:
    out: dict[str, list[Place]] = defaultdict(list)
    for place in places:
        if place.region:
            out[place.region].append(place)
    return out


def distance(left: Place, right: Place) -> float | None:
    if left.lat is None or left.lon is None or right.lat is None or right.lon is None:
        return None
    return haversine_km(left.lat, left.lon, right.lat, right.lon)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def earth_vector(lat: float, lon: float) -> tuple[float, float, float]:
    phi = math.radians(lat)
    lam = math.radians(lon)
    return (math.cos(phi) * math.cos(lam), math.cos(phi) * math.sin(lam), math.sin(phi))


def same_code_from_fips(fips: str) -> str:
    fips = clean_code(fips)
    if len(fips) == 5 and fips.isdigit():
        return "0" + fips
    if len(fips) == 6 and fips.isdigit():
        return fips
    return fips


def neural_label(place: Place) -> str:
    bits = [place.name, place.name_fr, place.region, place.country, place.kind]
    return " | ".join(bit for bit in bits if bit)


def unique_places(places: list[Place]) -> list[Place]:
    seen = set()
    out = []
    for place in places:
        key = (place.source, place.code)
        if key in seen:
            continue
        seen.add(key)
        out.append(place)
    return out


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())

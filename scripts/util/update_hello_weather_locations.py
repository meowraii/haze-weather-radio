#!/usr/bin/env python3
"""Build Haze's local Hello Weather location-code directory from Canada.ca."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


DEFAULT_SOURCE_URL = (
    "https://www.canada.ca/en/environment-climate-change/services/"
    "weather-general-tools-resources/telephone-services/"
    "recorded-observations-forecasts.html"
)

PROVINCES = {
    "British Columbia": "BC",
    "Alberta": "AB",
    "Saskatchewan": "SK",
    "Manitoba": "MB",
    "Ontario": "ON",
    "Quebec": "QC",
    "New Brunswick": "NB",
    "Nova Scotia": "NS",
    "Prince Edward Island": "PE",
    "Newfoundland and Labrador": "NL",
    "Yukon": "YT",
    "Northwest Territories": "NT",
    "Nunavut": "NU",
}

PROVINCE_ORDER = {code: index for index, code in enumerate(PROVINCES.values())}

REPRESENTATIVE_CODES = {
    "08074": ("Vancouver", "BC"),
    "07052": ("Calgary", "AB"),
    "06040": ("Saskatoon", "SK"),
    "05038": ("Winnipeg", "MB"),
    "04143": ("Toronto", "ON"),
    "03147": ("Montréal", "QC"),
    "01723": ("Saint John", "NB"),
    "01119": ("Halifax", "NS"),
    "01805": ("Charlottetown", "PE"),
    "02024": ("Mount Pearl", "NL"),
    "09116": ("Whitehorse", "YT"),
    "09524": ("Yellowknife", "NT"),
    "09821": ("Iqaluit", "NU"),
}


class HelloWeatherParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.heading_depth = 0
        self.heading_text: list[str] = []
        self.province = ""
        self.in_row = False
        self.cell_depth = 0
        self.cell_text: list[str] = []
        self.row: list[str] = []
        self.rows: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in {"h2", "summary"}:
            self.heading_depth += 1
            if self.heading_depth == 1:
                self.heading_text = []
        elif tag == "tr":
            self.in_row = True
            self.row = []
        elif tag == "td" and self.in_row:
            self.cell_depth += 1
            if self.cell_depth == 1:
                self.cell_text = []
        elif tag == "br" and self.cell_depth:
            self.cell_text.append(" ")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"h2", "summary"} and self.heading_depth:
            self.heading_depth -= 1
            if self.heading_depth == 0:
                heading = clean_text("".join(self.heading_text))
                self.province = PROVINCES.get(heading, "")
        elif tag == "td" and self.in_row and self.cell_depth:
            self.cell_depth -= 1
            if self.cell_depth == 0:
                self.row.append(clean_text("".join(self.cell_text)))
        elif tag == "tr" and self.in_row:
            self.in_row = False
            if self.province and len(self.row) >= 2:
                match = re.search(r"\bCode\s*:\s*([\d\s]+)", self.row[1], re.IGNORECASE)
                code = "" if match is None else "".join(re.findall(r"\d", match.group(1)))[:5]
                name = self.row[0]
                if name and len(code) == 5:
                    self.rows.append((code, name, self.province))

    def handle_data(self, data: str) -> None:
        if self.heading_depth:
            self.heading_text.append(data)
        if self.cell_depth:
            self.cell_text.append(data)


def clean_text(value: str) -> str:
    return " ".join(value.replace("\u00a0", " ").split())


def parse_locations(page: str, source_url: str = DEFAULT_SOURCE_URL) -> list[dict[str, str]]:
    parser = HelloWeatherParser()
    parser.feed(page)

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for code, name, province in parser.rows:
        pair = (name, province)
        if pair not in grouped[code]:
            grouped[code].append(pair)

    records: list[dict[str, str]] = []
    for code, names in grouped.items():
        provinces = {province for _, province in names}
        if len(provinces) != 1:
            raise ValueError(f"location code {code} appears in multiple provinces: {sorted(provinces)}")
        primary_name, province = names[0]
        aliases = [name for name, _ in names[1:] if name != primary_name]
        records.append(
            {
                "CODE": code,
                "NAME": primary_name,
                "PROVINCE": province,
                "ALIASES": "|".join(aliases),
                "SOURCE_URL": source_url,
            }
        )

    records.sort(key=lambda row: (PROVINCE_ORDER[row["PROVINCE"]], row["CODE"], row["NAME"]))
    validate_locations(records)
    return records


def validate_locations(records: list[dict[str, str]]) -> None:
    if len(records) < 750:
        raise ValueError(f"only {len(records)} location codes were parsed, expected at least 750")
    present_provinces = {row["PROVINCE"] for row in records}
    missing_provinces = set(PROVINCES.values()) - present_provinces
    if missing_provinces:
        raise ValueError(f"missing province sections: {sorted(missing_provinces)}")
    by_code = {row["CODE"]: row for row in records}
    for code, (name, province) in REPRESENTATIVE_CODES.items():
        row = by_code.get(code)
        if row is None:
            raise ValueError(f"missing representative location code {code}")
        names = {row["NAME"], *filter(None, row["ALIASES"].split("|"))}
        if name not in names or row["PROVINCE"] != province:
            raise ValueError(
                f"location code {code} mapped to {row['NAME']!r}/{row['PROVINCE']}, "
                f"expected {name!r}/{province}"
            )


def read_source(args: argparse.Namespace) -> str:
    if args.input_html:
        return args.input_html.read_text(encoding="utf-8")
    request = Request(args.source_url, headers={"User-Agent": args.user_agent})
    with urlopen(request, timeout=args.timeout) as response:
        return response.read().decode("utf-8")


def write_locations(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["CODE", "NAME", "PROVINCE", "ALIASES", "SOURCE_URL"])
        writer.writeheader()
        writer.writerows(records)


def update_sqlite(path: Path, records: list[dict[str, str]]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"location database does not exist: {path}")
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.execute("DELETE FROM places WHERE source = 'hello_weather'")
            conn.executemany(
                """
                INSERT INTO places(
                    source, code, name, name_fr, region, country, kind, lat, lon, attrs_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "hello_weather",
                        row["CODE"],
                        row["NAME"],
                        "",
                        row["PROVINCE"],
                        "CA",
                        "ECCC Hello Weather location code",
                        None,
                        None,
                        json.dumps(
                            {
                                "aliases": list(filter(None, row["ALIASES"].split("|"))),
                                "source_url": row["SOURCE_URL"],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                    for row in records
                ],
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("hello_weather_source", records[0]["SOURCE_URL"] if records else DEFAULT_SOURCE_URL),
            )
            conn.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("hello_weather_updated_at", datetime.now(timezone.utc).isoformat()),
            )
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=DEFAULT_SOURCE_URL)
    parser.add_argument("--input-html", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "csv_base" / "HELLO_WEATHER_LOCATIONS.csv",
    )
    parser.add_argument("--user-agent", default="HazeWeatherRadio/26.07 location-directory")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--sqlite", type=Path, help="Also replace hello_weather places in this Haze location database")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = parse_locations(read_source(args), args.source_url)
    write_locations(args.output, records)
    if args.sqlite:
        update_sqlite(args.sqlite, records)
    counts: dict[str, int] = defaultdict(int)
    for row in records:
        counts[row["PROVINCE"]] += 1
    print(f"wrote {len(records)} unique location codes to {args.output}")
    print("province counts: " + ", ".join(f"{code}={counts[code]}" for code in PROVINCE_ORDER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
import asyncio
import base64
import logging
import random
from datetime import datetime
from xml.etree import ElementTree as ET
from typing import Any, Awaitable, Callable
from dataclasses import dataclass

log = logging.getLogger(__name__)

NAAD_HOSTS = [
    "streaming1.naad-adna.pelmorex.com",
    "streaming2.naad-adna.pelmorex.com",
]

NAAD_PORT = 8080

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
_DELIMITER = b"</alert>"

BUFFER_SIZE = 4096
MAX_BUFFER_SIZE = 4 * 1024 * 1024

READ_TIMEOUT_S = 90
BASE_RECONNECT_DELAY = 3
MAX_RECONNECT_DELAY = 30

HEARTBEAT_IDENTIFIER = "Heartbeat"

def _t(tag: str) -> str:
    return f"{{{CAP_NS}}}{tag}"

def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None

@dataclass(slots=True, frozen=True)
class CAPResource:
    description: str
    mime_type: str
    uri: str
    data: bytes

@dataclass(slots=True, frozen=True)
class CAPParameter:
    name: str
    value: str

@dataclass(slots=True, frozen=True)
class CAPArea:
    description: str
    geocodes: tuple[str, ...]
    clc_codes: tuple[str, ...]
    polygons: tuple[str, ...]

@dataclass(slots=True, frozen=True)
class CAPInfo:
    language: str
    event: str
    categories: tuple[str, ...]
    response_types: tuple[str, ...]
    urgency: str
    severity: str
    certainty: str
    effective: datetime | None
    onset: datetime | None
    expires: datetime | None
    sender_name: str
    headline: str
    description: str
    instruction: str
    audience: str
    contact: str
    web: str
    areas: tuple[CAPArea, ...]
    resources: tuple[CAPResource, ...]
    event_codes: tuple[tuple[str, str], ...]
    parameters: tuple[CAPParameter, ...]

    def param_dict(self) -> dict[str, str]:
        result = {}
        for p in self.parameters:
            if isinstance(p, CAPParameter):
                result[p.name.lower()] = p.value
        return result

    @property
    def parameter_map(self) -> dict[str, str]:
        return self.param_dict()

    @property
    def area_geocodes(self) -> tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]:
        return tuple((area.description, area.geocodes, area.clc_codes) for area in self.areas)

    @property
    def area_descriptions(self) -> tuple[str, ...]:
        return tuple(area.description for area in self.areas if area.description)

    def param_dict(self) -> dict[str, str]:
        result = {}
        for p in self.parameters:
            if isinstance(p, CAPParameter):
                result[p.name.lower()] = p.value
        return result

@dataclass(slots=True, frozen=True)
class CAPAlert:
    identifier: str
    sender: str
    sent: datetime | None
    status: str
    msg_type: str
    scope: str
    source: str
    note: str
    references: str
    incidents: str
    codes: tuple[str, ...]
    infos: tuple[CAPInfo, ...]

    @property
    def parameters(self) -> tuple[CAPParameter, ...]:
        i = self.english or (self.infos[0] if self.infos else None)
        return i.parameters if i else ()

    def param_dict(self) -> dict[str, str]:
        result = {}
        for p in self.parameters:
            if isinstance(p, CAPParameter):
                result[p.name.lower()] = p.value
        return result

    def info_for_lang(self, lang: str) -> CAPInfo | None:
        for i in self.infos:
            if i.language.lower().startswith(lang.lower()):
                return i
        return self.infos[0] if self.infos else None

    @property
    def english(self) -> CAPInfo | None:
        return self.info_for_lang("en")

    @property
    def french(self) -> CAPInfo | None:
        return self.info_for_lang("fr")

    @property
    def headline(self) -> str:
        i = self.english or (self.infos[0] if self.infos else None)
        return i.headline if i else ""

    @property
    def event(self) -> str:
        i = self.english or (self.infos[0] if self.infos else None)
        return i.event if i else ""

    @property
    def severity(self) -> str:
        i = self.english or (self.infos[0] if self.infos else None)
        return i.severity if i else ""

    @property
    def urgency(self) -> str:
        i = self.english or (self.infos[0] if self.infos else None)
        return i.urgency if i else ""

    @property
    def certainty(self) -> str:
        i = self.english or (self.infos[0] if self.infos else None)
        return i.certainty if i else ""

    @property
    def same_event(self) -> str | None:
        for info in self.infos:
            for name, value in info.event_codes:
                if name.strip().upper() == 'SAME' and value.strip():
                    return value.strip().upper()
        return None

    @property
    def cap_cp_event(self) -> str | None:
        for info in self.infos:
            for name, value in info.event_codes:
                if name == 'profile:CAP-CP:Event:0.4' and value.strip():
                    return value.strip()
        return None

    @property
    def broadcast_immediately(self) -> bool:
        return any(
            info.param_dict().get('layer:sorem:1.0:broadcast_immediately', '').lower() == 'yes'
            for info in self.infos
        )

    @property
    def all_geocodes(self) -> tuple[str, ...]:
        values: list[str] = []
        seen: set[str] = set()
        for info in self.infos:
            for area in info.areas:
                for code in area.geocodes:
                    if code and code not in seen:
                        seen.add(code)
                        values.append(code)
        return tuple(values)

    @property
    def clc_codes(self) -> tuple[str, ...]:
        values: list[str] = []
        seen: set[str] = set()
        for info in self.infos:
            for area in info.areas:
                for code in area.clc_codes:
                    if code and code not in seen:
                        seen.add(code)
                        values.append(code)
        return tuple(values)


def _parse_resource(el: ET.Element) -> CAPResource:
    b64 = el.findtext(_t("derefUri"))
    data = base64.b64decode(b64) if b64 else b""

    return CAPResource(
        description=el.findtext(_t("resourceDesc"), ""),
        mime_type=el.findtext(_t("mimeType"), ""),
        uri=el.findtext(_t("uri"), ""),
        data=data,
    )

def _parse_area(el: ET.Element) -> CAPArea:
    geocodes = []
    clc_codes = []
    polygons = []
    for g in el.findall(_t("geocode")):
        name = g.findtext(_t("valueName"))
        val = g.findtext(_t("value"))
        if not val:
            continue
        geocodes.append(val)
        if name == "layer:EC-MSC-SMC:1.0:CLC":
            clc_codes.append(val)
    for p in el.findall(_t("polygon")):
        if p.text:
            polygons.append(p.text.strip())

    return CAPArea(
        description=el.findtext(_t("areaDesc"), ""),
        geocodes=tuple(geocodes),
        clc_codes=tuple(clc_codes),
        polygons=tuple(polygons),
    )

def _parse_info(el: ET.Element) -> CAPInfo:
    resources = tuple(_parse_resource(r) for r in el.findall(_t("resource")))
    areas = tuple(_parse_area(a) for a in el.findall(_t("area")))

    event_codes = tuple(
        (
            (ec.findtext(_t("valueName")) or "").strip(),
            (ec.findtext(_t("value")) or "").strip(),
        )
        for ec in el.findall(_t("eventCode"))
    )

    parameters = tuple(
        CAPParameter(
            name=(p.findtext(_t("valueName")) or "").strip(),
            value=(p.findtext(_t("value")) or "").strip(),
        )
        for p in el.findall(_t("parameter"))
        if isinstance(p, ET.Element)
    )

    return CAPInfo(
        language=el.findtext(_t("language"), "en-CA"),
        event=el.findtext(_t("event"), ""),
        categories=tuple(c.text.strip() for c in el.findall(_t("category")) if c.text),
        response_types=tuple(r.text.strip() for r in el.findall(_t("responseType")) if r.text),
        urgency=el.findtext(_t("urgency"), ""),
        severity=el.findtext(_t("severity"), ""),
        certainty=el.findtext(_t("certainty"), ""),
        effective=_parse_dt(el.findtext(_t("effective"))),
        onset=_parse_dt(el.findtext(_t("onset"))),
        expires=_parse_dt(el.findtext(_t("expires"))),
        sender_name=el.findtext(_t("senderName"), ""),
        headline=el.findtext(_t("headline"), ""),
        description=el.findtext(_t("description"), ""),
        instruction=el.findtext(_t("instruction"), ""),
        audience=el.findtext(_t("audience"), ""),
        contact=el.findtext(_t("contact"), ""),
        web=el.findtext(_t("web"), ""),
        areas=areas,
        resources=resources,
        event_codes=event_codes,
        parameters=parameters,
    )

def parse_cap(raw: bytes) -> CAPAlert | None:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        return None

    identifier = root.findtext(_t("identifier"), "")
    sender = root.findtext(_t("sender"), "")

    if HEARTBEAT_IDENTIFIER in identifier or HEARTBEAT_IDENTIFIER in sender:
        return None

    infos = tuple(_parse_info(i) for i in root.findall(_t("info")))

    return CAPAlert(
        identifier=identifier,
        sender=sender,
        sent=_parse_dt(root.findtext(_t("sent"))),

        status=root.findtext(_t("status"), ""),
        msg_type=root.findtext(_t("msgType"), ""),
        scope=root.findtext(_t("scope"), ""),

        source=root.findtext(_t("source"), ""),
        note=root.findtext(_t("note"), ""),
        references=root.findtext(_t("references"), ""),
        incidents=root.findtext(_t("incidents"), ""),

        codes=tuple(c.text or "" for c in root.findall(_t("code"))),
        infos=infos,
    )

async def _iter_messages(reader: asyncio.StreamReader):
    buf = b""
    while True:
        chunk = await asyncio.wait_for(reader.read(BUFFER_SIZE), timeout=READ_TIMEOUT_S)

        if not chunk:
            raise ConnectionResetError("Connection closed")

        buf += chunk

        if len(buf) > MAX_BUFFER_SIZE:
            log.warning("Buffer overflow, resetting")
            buf = b""
            continue

        while _DELIMITER in buf:
            msg, buf = buf.split(_DELIMITER, 1)
            yield msg.strip() + _DELIMITER


async def _run_connection(host: str, cb, shutdown):
    reader, writer = await asyncio.open_connection(host, NAAD_PORT)

    try:
        async for raw in _iter_messages(reader):
            if shutdown.is_set():
                break

            alert = parse_cap(raw)
            if alert:
                log.info("Alert: %s", alert.headline)
                await cb(alert)

    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def naad_listener(
    on_alert: Callable[[CAPAlert], Awaitable[None]],
    shutdown: asyncio.Event,
):
    attempt = 0

    while not shutdown.is_set():
        host = NAAD_HOSTS[attempt % len(NAAD_HOSTS)]

        try:
            await _run_connection(host, on_alert, shutdown)
            attempt = 0

        except Exception as e:
            delay = min(MAX_RECONNECT_DELAY, BASE_RECONNECT_DELAY * 2 ** attempt)
            delay += random.uniform(0, 2)

            log.warning("Reconnect in %.1fs (%s)", delay, e)

            try:
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                break

            attempt += 1
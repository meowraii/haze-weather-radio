import asyncio
import base64
import logging
from xml.etree import ElementTree as ET
from typing import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CAPResource:
    description: str
    mime_type: str
    uri: str
    data: bytes


@dataclass(frozen=True, slots=True)
class CAPInfo:
    language: str
    event: str
    urgency: str
    severity: str
    certainty: str
    expires: str
    headline: str
    description: str
    instruction: str
    same_event: str
    areas: tuple[str, ...]
    geocodes: tuple[str, ...]
    clc_codes: tuple[str, ...]
    resources: tuple[CAPResource, ...]
    area_geocodes: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = ()
    parameters: dict[str, str] = field(default_factory=lambda: {})


@dataclass(frozen=True, slots=True)
class CAPAlert:
    identifier: str
    sender: str
    sent: str
    status: str
    scope: str
    msg_type: str
    references: str
    infos: tuple[CAPInfo, ...]

    @property
    def event(self) -> str:
        return self.infos[0].event if self.infos else ''

    @property
    def severity(self) -> str:
        return self.infos[0].severity if self.infos else ''

    @property
    def urgency(self) -> str:
        return self.infos[0].urgency if self.infos else ''

    @property
    def certainty(self) -> str:
        return self.infos[0].certainty if self.infos else ''

    @property
    def headline(self) -> str:
        return self.infos[0].headline if self.infos else ''

    @property
    def geocodes(self) -> tuple[str, ...]:
        return self.infos[0].geocodes if self.infos else ()

    @property
    def all_geocodes(self) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []
        for info in self.infos:
            for gc in info.geocodes:
                if gc not in seen:
                    seen.add(gc)
                    result.append(gc)
        return tuple(result)

    @property
    def clc_codes(self) -> tuple[str, ...]:
        seen: set[str] = set()
        result: list[str] = []
        for info in self.infos:
            for code in info.clc_codes:
                if code not in seen:
                    seen.add(code)
                    result.append(code)
        return tuple(result)

    @property
    def same_event(self) -> str:
        for info in self.infos:
            if info.same_event:
                return info.same_event
        return ""

    @property
    def broadcast_immediately(self) -> bool:
        return any(
            info.parameters.get("layer:sorem:1.0:broadcast_immediately", "").lower() == "yes"
            for info in self.infos
        )

    def info_for_lang(self, lang_prefix: str) -> CAPInfo | None:
        for info in self.infos:
            if info.language.startswith(lang_prefix):
                return info
        return self.infos[0] if self.infos else None


log = logging.getLogger(__name__)

NAAD_HOSTS = [
    "streaming1.naad-adna.pelmorex.com",
    "streaming2.naad-adna.pelmorex.com",
]
NAAD_PORT = 8080
CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"
HEARTBEAT_IDENTIFIER = "Heartbeat"
BUFFER_SIZE = 4096
MAX_BUFFER_SIZE = 4 * 1024 * 1024
RECONNECT_DELAY_S = 5.0
READ_TIMEOUT_S = 90.0


def _t(tag: str) -> str:
    return f"{{{CAP_NS}}}{tag}"


def _parse_info(info: ET.Element) -> CAPInfo:
    areas: list[str] = []
    geocodes: list[str] = []
    clc_codes: list[str] = []
    resources: list[CAPResource] = []
    area_geocode_pairs: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []

    same_event: str = ""
    for ec in info.findall(_t("eventCode")):
        if ec.findtext(_t("valueName")) == "SAME":
            same_event = (ec.findtext(_t("value")) or "").strip()
            break

    for area in info.findall(_t("area")):
        desc = area.findtext(_t("areaDesc"), "")
        if desc:
            areas.append(desc)
        area_gcs: list[str] = []
        area_clcs: list[str] = []
        for gc in area.findall(_t("geocode")):
            vn = gc.findtext(_t("valueName"))
            val = gc.findtext(_t("value"))
            if not val:
                continue
            if vn == "profile:CAP-CP:Location:0.3":
                geocodes.append(val)
                area_gcs.append(val)
            elif vn == "layer:EC-MSC-SMC:1.0:CLC":
                clc_codes.append(val)
                area_clcs.append(val)
        if desc:
            area_geocode_pairs.append((desc, tuple(area_gcs), tuple(area_clcs)))

    for res in info.findall(_t("resource")):
        mime = res.findtext(_t("mimeType"), "")
        uri = res.findtext(_t("uri"), "")
        desc = res.findtext(_t("resourceDesc"), "")
        b64 = res.findtext(_t("derefUri"), "")
        data = base64.b64decode(b64) if b64 else b""
        resources.append(CAPResource(description=desc, mime_type=mime, uri=uri, data=data))

    parameters: dict[str, str] = {}
    for param in info.findall(_t("parameter")):
        vn = (param.findtext(_t("valueName")) or "").strip()
        val = (param.findtext(_t("value")) or "").strip()
        if vn:
            parameters[vn.lower()] = val
    
    log.debug("Parsed CAP info: %s — %s", info.findtext(_t("event"), ""), info.findtext(_t("headline"), ""))

    return CAPInfo(
        language=info.findtext(_t("language"), "en-CA"),
        event=info.findtext(_t("event"), ""),
        urgency=info.findtext(_t("urgency"), ""),
        severity=info.findtext(_t("severity"), ""),
        certainty=info.findtext(_t("certainty"), ""),
        expires=info.findtext(_t("expires"), ""),
        headline=info.findtext(_t("headline"), ""),
        description=info.findtext(_t("description"), ""),
        instruction=info.findtext(_t("instruction"), ""),
        same_event=same_event,
        areas=tuple(areas),
        geocodes=tuple(geocodes),
        clc_codes=tuple(clc_codes),
        resources=tuple(resources),
        area_geocodes=tuple(area_geocode_pairs),
        parameters=parameters,
    )


def parse_cap(raw: bytes) -> CAPAlert | None:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        log.error("CAP XML parse error: %s", e)
        return None

    identifier = root.findtext(_t("identifier"), "")
    sender = root.findtext(_t("sender"), "")
    if HEARTBEAT_IDENTIFIER in identifier or HEARTBEAT_IDENTIFIER in sender:
        log.debug("NAAD heartbeat received")
        return None

    info_elems = root.findall(_t("info"))
    if not info_elems:
        log.warning("CAP alert %s has no <info> elements, skipping", identifier)
        return None

    infos = tuple(_parse_info(el) for el in info_elems)

    return CAPAlert(
        identifier=identifier,
        sender=root.findtext(_t("sender"), ""),
        sent=root.findtext(_t("sent"), ""),
        status=root.findtext(_t("status"), ""),
        scope=root.findtext(_t("scope"), ""),
        msg_type=root.findtext(_t("msgType"), ""),
        references=root.findtext(_t("references"), ""),
        infos=infos,
    )


_DELIMITER = b"</alert>"


async def _iter_cap_messages(reader: asyncio.StreamReader):
    buf = b""
    while True:
        chunk = await asyncio.wait_for(reader.read(BUFFER_SIZE), timeout=READ_TIMEOUT_S)
        if not chunk:
            raise ConnectionResetError("NAAD connection closed by remote")
        buf += chunk
        if len(buf) > MAX_BUFFER_SIZE:
            log.error("NAAD buffer exceeded %d bytes without a complete message, resetting", MAX_BUFFER_SIZE)
            buf = b""
            continue
        while _DELIMITER in buf:
            msg, buf = buf.split(_DELIMITER, 1)
            msg = msg.strip() + _DELIMITER
            if msg:
                yield msg


async def _run_connection(
    host: str,
    on_alert: Callable[[CAPAlert], Awaitable[None]],
    shutdown: asyncio.Event,
) -> None:
    log.info("Connecting to NAAD at %s:%d", host, NAAD_PORT)
    reader, writer = await asyncio.open_connection(host, NAAD_PORT)
    log.info("Connected to %s", host)
    try:
        async for raw in _iter_cap_messages(reader):
            if shutdown.is_set():
                break
            alert = parse_cap(raw)
            if alert:
                log.info("CAP alert: %s — %s", alert.event, alert.headline)
                await on_alert(alert)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def naad_listener(
    on_alert: Callable[[CAPAlert], Awaitable[None]],
    shutdown: asyncio.Event,
) -> None:
    host_index = 0
    while not shutdown.is_set():
        host = NAAD_HOSTS[host_index % len(NAAD_HOSTS)]
        try:
            await _run_connection(host, on_alert, shutdown)
        except (OSError, ConnectionResetError, TimeoutError, asyncio.TimeoutError) as e:
            log.warning("NAAD connection lost (%s): %s — reconnecting in %ds", host, e, RECONNECT_DELAY_S)
            host_index += 1
        except asyncio.CancelledError:
            break

        if not shutdown.is_set():
            await asyncio.sleep(RECONNECT_DELAY_S)
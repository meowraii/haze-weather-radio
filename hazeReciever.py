from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

pifm_bin = "/home/rai/PiFmAdv/src/pi_fm_adv"


@dataclass(frozen=True)
class ReceiverConfig:
    udp_host: str
    udp_port: int
    output_sample_rate: int
    channels: int
    ffmpeg_bin: str
    pifmadv_bin: str
    pifm_frequency: float | None
    pifm_bandwidth_khz: float
    pifm_deviation_hz: int
    pifm_preemphasis: str
    pifm_extra_args: tuple[str, ...]
    ffmpeg_log_level: str
    reconnect_initial_delay_s: float
    reconnect_max_delay_s: float
    reconnect_backoff: float
    stream_stall_timeout_s: float
    write_chunk_size: int
    metadata_probe_timeout_s: float
    feeds_file: str


_FREQ_RE = re.compile(r"(?<!\d)(\d{2,3}(?:\.\d{1,3})?)(?:\s*(?:mhz|m))?", re.IGNORECASE)


def _resolve_ffprobe_bin(ffmpeg_bin: str) -> str:
    base = os.path.basename(ffmpeg_bin).lower()
    if base.startswith("ffmpeg"):
        return os.path.join(os.path.dirname(ffmpeg_bin), "ffprobe") if os.path.dirname(ffmpeg_bin) else "ffprobe"
    return "ffprobe"


def _extract_frequency_from_tags(tags: dict[str, object]) -> float | None:
    preferred_keys = (
        "haze_tx_frequency_mhz",
        "tx_frequency_mhz",
        "frequency_mhz",
        "frequency",
        "tx_data",
        "service_name",
        "comment",
    )

    for key in preferred_keys:
        value = tags.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            parsed = float(text)
            if 60.0 <= parsed <= 110.0:
                return parsed
        except ValueError:
            pass
        match = _FREQ_RE.search(text)
        if not match:
            continue
        try:
            parsed = float(match.group(1))
        except ValueError:
            continue
        if 60.0 <= parsed <= 110.0:
            return parsed
    return None


def _extract_station_name_from_service_name(service_name: str) -> str:
    text = service_name.strip()
    if not text:
        return ""
    if "(" in text and ")" in text:
        start = text.rfind("(")
        end = text.rfind(")")
        if start >= 0 and end > start:
            inside = text[start + 1 : end].strip()
            if inside:
                return inside
    return text


def _build_local_station_frequency_map(feeds_file: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    path = pathlib.Path(feeds_file)
    if not path.exists():
        return mapping
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return mapping

    for feed_el in root.findall("feed"):
        tx_meta = feed_el.find("transmitter_metadata")
        if tx_meta is None:
            continue
        for tx in tx_meta.findall("transmitter"):
            callsign = (tx.findtext("callsign") or feed_el.get("callsign") or "").strip()
            site_name = (tx.findtext("site_name") or "").strip()
            raw_freq = (tx.findtext("frequency_mhz") or "").strip()
            if not raw_freq:
                continue
            try:
                freq = float(raw_freq)
            except ValueError:
                continue
            key = " ".join(part for part in (callsign, site_name) if part).strip().lower()
            if key:
                mapping[key] = freq
    return mapping


class ReceiverSupervisor:
    def __init__(self, config: ReceiverConfig) -> None:
        self.config = config
        self.stop_event = asyncio.Event()
        self.last_audio_ts = 0.0
        self._local_station_freq_map = _build_local_station_frequency_map(config.feeds_file)

    async def run_forever(self) -> None:
        self._install_signal_handlers()
        delay = self.config.reconnect_initial_delay_s

        while not self.stop_event.is_set():
            reason = await self._run_pipeline_once()
            if self.stop_event.is_set():
                break
            logging.warning("Pipeline restart requested: %s", reason)
            await asyncio.sleep(delay)
            delay = min(self.config.reconnect_max_delay_s, delay * self.config.reconnect_backoff)

        logging.info("Receiver stopped")

    async def _run_pipeline_once(self) -> str:
        ffmpeg_proc: asyncio.subprocess.Process | None = None
        pifm_proc: asyncio.subprocess.Process | None = None
        selected_frequency: float | None = self.config.pifm_frequency
        initial_audio_chunk: bytes = b""

        try:
            if selected_frequency is None:
                selected_frequency = await self._detect_frequency_from_stream()
                if selected_frequency is None:
                    return "mpegts metadata frequency unavailable"

            try:
                ffmpeg_proc = await self._start_ffmpeg()
            except Exception as exc:
                return f"ffmpeg startup failed: {exc}"

            assert ffmpeg_proc.stdout is not None
            assert ffmpeg_proc.stderr is not None

            self.last_audio_ts = time.monotonic()
            initial_audio_chunk, warmup_reason = await self._read_initial_audio_chunk(ffmpeg_proc.stdout, ffmpeg_proc)
            if warmup_reason:
                return warmup_reason

            try:
                pifm_proc = await self._start_pifmadv(selected_frequency)
            except Exception as exc:
                return f"piFmAdv startup failed: {exc}"

            assert pifm_proc.stdin is not None
            assert pifm_proc.stderr is not None

            pump_task = asyncio.create_task(
                self._pump_audio(ffmpeg_proc.stdout, pifm_proc.stdin, initial_audio_chunk),
                name="pump_audio",
            )
            ffmpeg_err_task = asyncio.create_task(self._log_stream(ffmpeg_proc.stderr, "ffmpeg"), name="ffmpeg_stderr")
            pifm_err_task = asyncio.create_task(self._log_stream(pifm_proc.stderr, "piFmAdv"), name="pifm_stderr")
            health_task = asyncio.create_task(self._monitor_health(ffmpeg_proc, pifm_proc), name="monitor_health")

            done, pending = await asyncio.wait({pump_task, health_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()

            reason = "pipeline stopped"
            for task in done:
                try:
                    result = await task
                    if isinstance(result, str) and result:
                        reason = result
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    reason = f"task failure: {exc}"

            for task in (ffmpeg_err_task, pifm_err_task):
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

            return reason
        finally:
            await self._terminate_process(ffmpeg_proc, "ffmpeg")
            await self._terminate_process(pifm_proc, "piFmAdv")

    def _build_udp_input_url(self, host_override: str | None = None) -> str:
        host = (host_override if host_override is not None else self.config.udp_host).strip() or "0.0.0.0"
        return (
            f"udp://{host}:{self.config.udp_port}"
            "?fifo_size=1000000"
            "&overrun_nonfatal=1"
            "&buffer_size=1048576"
        )

    async def _start_ffmpeg(self) -> asyncio.subprocess.Process:
        input_url = self._build_udp_input_url()
        cmd = [
            self.config.ffmpeg_bin,
            "-hide_banner",
            "-nostats",
            "-loglevel",
            self.config.ffmpeg_log_level,
            "-fflags",
            "+nobuffer",
            "-flags",
            "low_delay",
            "-i",
            input_url,
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            str(self.config.channels),
            "-ar",
            str(self.config.output_sample_rate),
            "-f",
            "wav",
            "pipe:1",
        ]
        logging.info("Starting ffmpeg UDP receiver (%s)", input_url)
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _detect_frequency_from_stream(self) -> float | None:
        ffprobe_bin = _resolve_ffprobe_bin(self.config.ffmpeg_bin)

        async def _probe(cmd: list[str], label: str) -> dict[str, object] | None:
            proc: asyncio.subprocess.Process | None = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.config.metadata_probe_timeout_s)
            except asyncio.TimeoutError:
                logging.info("ffprobe timed out while probing %s", label)
                if proc is not None and proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await proc.wait()
                    except Exception:
                        pass
                return None
            except Exception as exc:
                logging.info("ffprobe failed for %s: %s", label, exc)
                if proc is not None and proc.returncode is None:
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await proc.wait()
                    except Exception:
                        pass
                return None

            if proc.returncode != 0:
                detail = stderr.decode(errors="replace").strip()
                if detail:
                    logging.info("ffprobe probe failed for %s: %s", label, detail)
                return None

            try:
                payload = json.loads(stdout.decode(errors="replace") or "{}")
            except Exception as exc:
                logging.info("ffprobe JSON parse failed for %s: %s", label, exc)
                return None
            if isinstance(payload, dict):
                return payload
            return None

        host = self.config.udp_host.strip()
        candidate_hosts = ["0.0.0.0", "127.0.0.1", "localhost"]
        if host and host not in {"0.0.0.0", "::", "*"}:
            candidate_hosts.insert(1, host)

        probe_targets: list[tuple[list[str], str]] = []
        seen_urls: set[str] = set()
        for candidate in candidate_hosts:
            url = self._build_udp_input_url(host_override=candidate)
            if url in seen_urls:
                continue
            seen_urls.add(url)
            probe_targets.append(
                (
                    [
                        ffprobe_bin,
                        "-v",
                        "error",
                        "-print_format",
                        "json",
                        "-show_format",
                        "-show_programs",
                        url,
                    ],
                    url,
                )
            )

        for cmd, label in probe_targets:
            payload = await _probe(cmd, label)
            if payload is None:
                continue

            format_block = payload.get("format")
            format_tags = format_block.get("tags", {}) if isinstance(format_block, dict) else {}
            if isinstance(format_tags, dict):
                frequency = _extract_frequency_from_tags(format_tags)
                if frequency is not None:
                    logging.info("Detected transmitter frequency %.3f MHz from %s format metadata", frequency, label)
                    return frequency

            programs = payload.get("programs")
            if isinstance(programs, list):
                for program in programs:
                    if not isinstance(program, dict):
                        continue
                    tags = program.get("tags", {})
                    if not isinstance(tags, dict):
                        continue
                    frequency = _extract_frequency_from_tags(tags)
                    if frequency is not None:
                        logging.info("Detected transmitter frequency %.3f MHz from %s program metadata", frequency, label)
                        return frequency

                    if self._local_station_freq_map:
                        service_name = str(tags.get("service_name") or "").strip()
                        station_name = _extract_station_name_from_service_name(service_name).lower()
                        if station_name and station_name in self._local_station_freq_map:
                            frequency = self._local_station_freq_map[station_name]
                            logging.info(
                                "Resolved transmitter frequency %.3f MHz from local feeds map using service_name '%s'",
                                frequency,
                                service_name,
                            )
                            return frequency

        logging.warning("Unable to determine frequency from MPEG-TS metadata")
        return None

    async def _start_pifmadv(self, frequency_mhz: float) -> asyncio.subprocess.Process:
        cmd = [
            self.config.pifmadv_bin,
            "--audio",
            "-",
            "--freq",
            str(frequency_mhz),
            "--dev",
            str(self.config.pifm_deviation_hz),
            "--rds",
            "0",
            *self.config.pifm_extra_args,
        ]
        preemph = str(self.config.pifm_preemphasis or "").strip().lower()
        if preemph == "50":
            cmd.extend(["--preemph", "50us"])
        elif preemph == "75":
            cmd.extend(["--preemph", "75us"])
        logging.info("Starting piFmAdv transmitter (%s)", self.config.pifmadv_bin)
        return await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _read_initial_audio_chunk(
        self,
        ffmpeg_stdout: asyncio.StreamReader,
        ffmpeg_proc: asyncio.subprocess.Process,
    ) -> tuple[bytes, str | None]:
        try:
            chunk = await asyncio.wait_for(
                ffmpeg_stdout.read(self.config.write_chunk_size),
                timeout=self.config.stream_stall_timeout_s,
            )
        except asyncio.TimeoutError:
            return (
                b"",
                (
                    f"no UDP audio received for {self.config.stream_stall_timeout_s:.1f}s "
                    f"on {self.config.udp_host}:{self.config.udp_port}"
                ),
            )

        if not chunk:
            if ffmpeg_proc.returncode is not None:
                return b"", f"ffmpeg exited with code {ffmpeg_proc.returncode} before audio"
            return b"", "ffmpeg output ended before first audio packet"

        self.last_audio_ts = time.monotonic()
        return chunk, None

    async def _pump_audio(
        self,
        ffmpeg_stdout: asyncio.StreamReader,
        pifm_stdin: asyncio.StreamWriter,
        initial_chunk: bytes = b"",
    ) -> str:
        if initial_chunk:
            try:
                pifm_stdin.write(initial_chunk)
                await pifm_stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return "piFmAdv stdin closed"
            except Exception as exc:
                return f"piFmAdv write failed: {exc}"

        while not self.stop_event.is_set():
            chunk = await ffmpeg_stdout.read(self.config.write_chunk_size)
            if not chunk:
                return "ffmpeg output ended"
            self.last_audio_ts = time.monotonic()
            try:
                pifm_stdin.write(chunk)
                await pifm_stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                return "piFmAdv stdin closed"
            except Exception as exc:
                return f"piFmAdv write failed: {exc}"
        return "shutdown requested"

    async def _monitor_health(
        self,
        ffmpeg_proc: asyncio.subprocess.Process,
        pifm_proc: asyncio.subprocess.Process,
    ) -> str:
        while not self.stop_event.is_set():
            if ffmpeg_proc.returncode is not None:
                return f"ffmpeg exited with code {ffmpeg_proc.returncode}"
            if pifm_proc.returncode is not None:
                return f"piFmAdv exited with code {pifm_proc.returncode}"

            idle_for = time.monotonic() - self.last_audio_ts
            if idle_for >= self.config.stream_stall_timeout_s:
                return f"udp stream stalled for {idle_for:.1f}s"

            await asyncio.sleep(1.0)

        return "shutdown requested"

    async def _log_stream(self, stream: asyncio.StreamReader, name: str) -> None:
        while not self.stop_event.is_set():
            line = await stream.readline()
            if not line:
                return
            text = line.decode(errors="replace").rstrip()
            if text:
                logging.info("[%s] %s", name, text)

    async def _terminate_process(self, proc: asyncio.subprocess.Process | None, name: str) -> None:
        if proc is None:
            return

        if proc.returncode is None:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()

        logging.info("%s stopped with code %s", name, proc.returncode)

    def _install_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()

        def _trigger_stop() -> None:
            if not self.stop_event.is_set():
                logging.info("Shutdown signal received")
                self.stop_event.set()

        for sig_name in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                loop.add_signal_handler(sig, _trigger_stop)
            except NotImplementedError:
                signal.signal(sig, lambda _s, _f: _trigger_stop())


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _windows_is_admin() -> bool:
    if os.name != "nt":
        return False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _windows_resolve_program_path(path_or_name: str) -> str | None:
    if not path_or_name:
        return None
    if os.path.isabs(path_or_name) and os.path.exists(path_or_name):
        return os.path.abspath(path_or_name)
    resolved = shutil.which(path_or_name)
    if resolved:
        return os.path.abspath(resolved)
    return None


def _windows_firewall_rule_exists(rule_name: str) -> bool:
    query = (
        "$r = Get-NetFirewallRule -DisplayName "
        f"{_ps_quote(rule_name)} -ErrorAction SilentlyContinue; "
        "if ($r) { exit 0 } else { exit 1 }"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", query],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return False
    return result.returncode == 0


def _windows_add_firewall_rules_admin(rule_defs: list[tuple[str, str, int]]) -> bool:
    commands: list[str] = []
    for rule_name, program_path, port in rule_defs:
        commands.append(
            "if (-not (Get-NetFirewallRule -DisplayName "
            f"{_ps_quote(rule_name)} -ErrorAction SilentlyContinue)) "
            "{ New-NetFirewallRule -DisplayName "
            f"{_ps_quote(rule_name)} -Direction Inbound -Action Allow -Protocol UDP "
            f"-LocalPort {port} -Program {_ps_quote(program_path)} -Profile Private,Public | Out-Null }}"
        )
    if not commands:
        return True
    script = "; ".join(commands)
    result = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logging.warning("Firewall rule creation failed: %s", (result.stderr or result.stdout or "").strip())
        return False
    return True


def _windows_request_firewall_rules_uac(rule_defs: list[tuple[str, str, int]]) -> bool:
    if not rule_defs:
        return True
    lines: list[str] = []
    for rule_name, program_path, port in rule_defs:
        lines.append(
            "if (-not (Get-NetFirewallRule -DisplayName "
            f"{_ps_quote(rule_name)} -ErrorAction SilentlyContinue)) "
            "{ New-NetFirewallRule -DisplayName "
            f"{_ps_quote(rule_name)} -Direction Inbound -Action Allow -Protocol UDP "
            f"-LocalPort {port} -Program {_ps_quote(program_path)} -Profile Private,Public | Out-Null }}"
        )
    script = "; ".join(lines)
    try:
        import ctypes

        rc = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            "powershell.exe",
            f"-NoProfile -ExecutionPolicy Bypass -Command \"{script}\"",
            None,
            1,
        )
    except Exception as exc:
        logging.warning("Could not request elevated firewall setup: %s", exc)
        return False

    if rc <= 32:
        logging.warning("Firewall setup elevation was canceled or failed (code %s)", rc)
        return False
    return True


def _ensure_windows_firewall_access(config: ReceiverConfig) -> None:
    if os.name != "nt":
        return

    python_path = _windows_resolve_program_path(sys.executable)
    ffmpeg_path = _windows_resolve_program_path(config.ffmpeg_bin)

    rule_defs: list[tuple[str, str, int]] = []
    if python_path:
        rule_defs.append((f"Haze Receiver UDP {config.udp_port} Python", python_path, config.udp_port))
    if ffmpeg_path:
        rule_defs.append((f"Haze Receiver UDP {config.udp_port} FFmpeg", ffmpeg_path, config.udp_port))

    missing = [rule for rule in rule_defs if not _windows_firewall_rule_exists(rule[0])]
    if not missing:
        return

    if _windows_is_admin():
        if _windows_add_firewall_rules_admin(missing):
            logging.info("Windows firewall rules added for UDP port %s", config.udp_port)
        return

    logging.info("Requesting UAC elevation to add Windows firewall allow rules for UDP port %s", config.udp_port)
    _windows_request_firewall_rules_uac(missing)


def _parse_args() -> ReceiverConfig:
    parser = argparse.ArgumentParser(
        prog="hazeReciever.py",
        description="Standalone UDP MPEG-TS to piFmAdv receiver with automatic restart and stream recovery.",
    )

    parser.add_argument("--udp-host", "--rtp-host", dest="udp_host", default="0.0.0.0")
    parser.add_argument("--udp-port", "--rtp-port", dest="udp_port", type=int, default=8898)
    parser.add_argument("--output-sample-rate", type=int, default=16000)
    parser.add_argument("--channels", type=int, default=1)

    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffmpeg-log-level", default="warning")

    parser.add_argument("--freq", type=float, default=None)
    parser.add_argument("--bandwidth-khz", type=float, default=12.5)
    parser.add_argument("--deviation-hz", type=int, default=5000)
    parser.add_argument("--preemphasis", choices=["none", "50", "75"], default="none")
    parser.add_argument("--pi-extra-arg", action="append", default=[])

    parser.add_argument("--stall-timeout", type=float, default=12.0)
    parser.add_argument("--reconnect-initial-delay", type=float, default=1.0)
    parser.add_argument("--reconnect-max-delay", type=float, default=8.0)
    parser.add_argument("--reconnect-backoff", type=float, default=1.5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--metadata-probe-timeout", type=float, default=2.5)
    parser.add_argument("--feeds-file", default="managed/configs/feeds.xml")

    args = parser.parse_args()

    return ReceiverConfig(
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        output_sample_rate=args.output_sample_rate,
        channels=args.channels,
        ffmpeg_bin=args.ffmpeg_bin,
        pifmadv_bin=pifm_bin,
        pifm_frequency=args.freq,
        pifm_bandwidth_khz=args.bandwidth_khz,
        pifm_deviation_hz=args.deviation_hz,
        pifm_preemphasis=args.preemphasis,
        pifm_extra_args=tuple(args.pi_extra_arg),
        ffmpeg_log_level=args.ffmpeg_log_level,
        reconnect_initial_delay_s=max(0.1, args.reconnect_initial_delay),
        reconnect_max_delay_s=max(0.1, args.reconnect_max_delay),
        reconnect_backoff=max(1.0, args.reconnect_backoff),
        stream_stall_timeout_s=max(2.0, args.stall_timeout),
        write_chunk_size=max(512, args.chunk_size),
        metadata_probe_timeout_s=max(0.5, args.metadata_probe_timeout),
        feeds_file=os.path.expanduser(args.feeds_file),
    )


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


async def _main() -> None:
    _configure_logging()
    config = _parse_args()
    _ensure_windows_firewall_access(config)
    supervisor = ReceiverSupervisor(config)
    await supervisor.run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass

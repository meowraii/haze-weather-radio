import asyncio
import datetime
import logging
import logging.handlers
import pathlib
import threading
from typing import Any

from managed.events import (
    append_runtime_event,
    initial_synthesis_done,
    shutdown_event,
    tts_queue,
    update_runtime_status,
)
from module.alert import alert_worker
from module.data import data_thread_worker
from module.playlist import playlist_thread_worker
from module.playout import playout_thread_worker
from module.same import generate_same, to_wav
from module.scheduler import scheduler_thread_worker
from module.text import load_config
from module.tts import tts_thread_worker
from module.webserve import start_web_server

config = load_config()


def _check_static_audio(config: dict[str, Any]) -> None:
    static_root = pathlib.Path(config.get('static', 'audio'))
    static_root.mkdir(parents=True, exist_ok=True)
    same_eom_path = static_root / 'same_eom.wav'
    if not same_eom_path.exists():
        logging.info('Generating SAME EOM cache: %s', same_eom_path)
        same_eom_path.write_bytes(to_wav(generate_same()))


def _naads_thread_worker(config: dict[str, Any], feeds: list[dict[str, Any]]) -> None:
    async def _run() -> None:
        shutdown = asyncio.Event()

        async def _watch() -> None:
            while not shutdown_event.is_set():
                await asyncio.sleep(0.5)
            shutdown.set()

        watchdog = asyncio.create_task(_watch())
        try:
            await alert_worker(config, feeds, shutdown)
        finally:
            watchdog.cancel()

    asyncio.run(_run())


def _setup_logging(config: dict[str, Any]) -> None:
    log_cfg = config.get('logging', {})
    level = getattr(logging, log_cfg.get('level', 'INFO').upper(), logging.INFO)
    fmt = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    handlers: list[logging.Handler] = []

    if log_cfg.get('console', {}).get('enabled', True):
        handlers.append(logging.StreamHandler())

    file_cfg = log_cfg.get('file', {})
    if file_cfg.get('enabled', False):
        log_path = pathlib.Path(file_cfg['path'])
        log_path.parent.mkdir(parents=True, exist_ok=True)
        rotate = file_cfg.get('rotate', {})
        handlers.append(logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=rotate.get('max_bytes', 10_485_760),
            backupCount=rotate.get('backup_count', 5),
        ))

    logging.basicConfig(level=level, format=fmt, handlers=handlers)


def _thread(target: Any, *args: Any, name: str) -> threading.Thread:
    return threading.Thread(target=target, args=args, name=name, daemon=True)


def main() -> None:
    _setup_logging(config)
    log = logging.getLogger('haze')
    update_runtime_status({
        'started_at': datetime.datetime.now(datetime.UTC).isoformat(),
        'shutdown_requested': False,
    })
    append_runtime_event('startup', 'Haze Weather Radio starting')
    log.info('Haze Weather Radio starting')
    _check_static_audio(config)

    feeds = [f for f in config.get('feeds', []) if f.get('enabled', True)]

    # --- Phase 1: Infrastructure (web panel + alert listener) ---
    infra: list[threading.Thread] = [
        _thread(_naads_thread_worker, config, feeds, name='naads'),
    ]
    web_thread = start_web_server(config, feeds)
    if web_thread is not None:
        infra.append(web_thread)
    for t in infra:
        t.start()

    data: list[threading.Thread] = [
        _thread(data_thread_worker, config, name='data'),
        _thread(playlist_thread_worker, config, name='playlist'),
    ]
    for t in data:
        t.start()

    log.info('Waiting for initial data fetch and audio synthesis...')
    initial_synthesis_done.wait()

    tts = _thread(tts_thread_worker, config, name='tts')
    tts.start()

    log.info('Waiting for TTS queue to drain...')
    tts_queue.join()

    playout: list[threading.Thread] = [
        _thread(playout_thread_worker, config, feed, name=f'playout:{feed["id"]}')
        for feed in feeds
    ]
    for t in playout:
        t.start()

    log.info('All systems nominal. Playout active for %d feed(s).', len(feeds))

    scheduler = _thread(scheduler_thread_worker, config, feeds, name='scheduler')
    scheduler.start()

    all_threads = infra + data + [tts] + playout + [scheduler]
    try:
        for t in all_threads:
            t.join()
    except KeyboardInterrupt:
        log.info('Shutdown requested')
        update_runtime_status({'shutdown_requested': True})
        append_runtime_event('shutdown', 'Shutdown requested by operator')
        shutdown_event.set()


if __name__ == '__main__':
    main()

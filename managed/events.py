import queue
import threading
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class NowPlayingMetadata:
    title: str
    track: str | None = None


@dataclass(frozen=True, slots=True)
class PlayableItem:
    pkg_id: str
    text: str
    lang: str
    voice: str | None = None
    metadata: NowPlayingMetadata | None = None

shutdown_event: threading.Event = threading.Event()
initial_synthesis_done: threading.Event = threading.Event()

_data_pool: dict[str, Any] = {}
_data_pool_lock: threading.Lock = threading.Lock()
data_ready: threading.Event = threading.Event()

tts_queue: queue.Queue[tuple[Any, ...] | None] = queue.Queue()

_sequences_lock: threading.Lock = threading.Lock()
_playout_sequences: dict[str, tuple[PlayableItem, ...]] = {}
_sequence_versions: dict[str, int] = {}

_alert_queues_lock: threading.Lock = threading.Lock()
_alert_queues: dict[str, queue.Queue[tuple[int, bytes, str]]] = {}

_runtime_lock: threading.Lock = threading.Lock()
_runtime_state: dict[str, Any] = {
    'system': {},
    'feeds': {},
    'events': [],
}

def register_alert_queue(feed_id: str) -> queue.Queue[tuple[int, bytes, str]]:
    with _alert_queues_lock:
        q: queue.Queue[tuple[int, bytes, str]] = queue.Queue(maxsize=128)
        _alert_queues[feed_id] = q
        return q

def push_alert(feed_id: str, priority: int, pcm: bytes, identifier: str = '') -> None:
    with _alert_queues_lock:
        q = _alert_queues.get(feed_id)
    if q is not None:
        q.put((priority, pcm, identifier))


def push_alert_all(priority: int, pcm: bytes, identifier: str = '') -> None:
    with _alert_queues_lock:
        queues = list(_alert_queues.values())
    for q in queues:
        q.put((priority, pcm, identifier))


def update_playout_sequence(feed_id: str, sequence: list[PlayableItem]) -> None:
    with _sequences_lock:
        _playout_sequences[feed_id] = tuple(sequence)
        _sequence_versions[feed_id] = _sequence_versions.get(feed_id, 0) + 1


def get_playout_sequence(feed_id: str) -> list[PlayableItem]:
    with _sequences_lock:
        return list(_playout_sequences.get(feed_id, ()))


def get_sequence_version(feed_id: str) -> int:
    with _sequences_lock:
        return _sequence_versions.get(feed_id, 0)


def update_data_pool(key: str, value: Any, notify: bool = True) -> None:
    with _data_pool_lock:
        _data_pool[key] = value
    if notify:
        data_ready.set()


def read_data_pool(key: str) -> Any:
    with _data_pool_lock:
        return _data_pool.get(key)


def snapshot_data_pool() -> dict[str, Any]:
    with _data_pool_lock:
        return dict(_data_pool)


def snapshot_playout_sequences() -> dict[str, list[PlayableItem]]:
    with _sequences_lock:
        return {feed_id: list(sequence) for feed_id, sequence in _playout_sequences.items()}


def snapshot_alert_queues() -> dict[str, int]:
    with _alert_queues_lock:
        return {feed_id: q.qsize() for feed_id, q in _alert_queues.items()}


def update_runtime_status(values: dict[str, Any]) -> None:
    with _runtime_lock:
        _runtime_state['system'].update(values)


def update_feed_runtime(feed_id: str, values: dict[str, Any]) -> None:
    with _runtime_lock:
        feed_state = _runtime_state['feeds'].setdefault(feed_id, {})
        feed_state.update(values)


def append_runtime_event(
    kind: str,
    message: str,
    feed_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    event: dict[str, Any] = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'kind': kind,
        'message': message,
    }
    if feed_id:
        event['feed_id'] = feed_id
    if extra:
        event['extra'] = extra

    with _runtime_lock:
        events = _runtime_state['events']
        events.append(event)
        if len(events) > 100:
            del events[:-100]


def snapshot_runtime() -> dict[str, Any]:
    with _runtime_lock:
        return deepcopy(_runtime_state)


_chime_lock: threading.Lock = threading.Lock()
_pending_chime_version: int = 0
_pending_chime_type: str | None = None
_chime_consumed_versions: dict[str, int] = {}


_chime_feed_queues: dict[str, queue.Queue] = {}
_chime_queues_lock: threading.Lock = threading.Lock()


def register_chime_queue(feed_id: str) -> queue.Queue:
    with _chime_queues_lock:
        q: queue.Queue = queue.Queue(maxsize=4)
        _chime_feed_queues[feed_id] = q
        return q


def set_pending_chime(chime_type: str) -> None:
    global _pending_chime_version, _pending_chime_type
    with _chime_lock:
        _pending_chime_type = chime_type
        _pending_chime_version += 1
    with _chime_queues_lock:
        for q in _chime_feed_queues.values():
            try:
                q.put_nowait(chime_type)
            except queue.Full:
                pass


def consume_pending_chime(feed_id: str) -> str | None:
    with _chime_lock:
        consumed = _chime_consumed_versions.get(feed_id, 0)
        if consumed >= _pending_chime_version:
            return None
        _chime_consumed_versions[feed_id] = _pending_chime_version
        return _pending_chime_type

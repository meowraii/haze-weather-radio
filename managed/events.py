import pathlib
import queue
import threading
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

shutdown_event: threading.Event = threading.Event()
initial_synthesis_done: threading.Event = threading.Event()

_data_pool: dict[str, Any] = {}
_data_pool_lock: threading.Lock = threading.Lock()
data_ready: threading.Event = threading.Event()

tts_queue: queue.Queue[tuple[Any, ...] | None] = queue.Queue()

_sequences_lock: threading.Lock = threading.Lock()
playout_sequences: dict[str, list[pathlib.Path]] = {}

_alert_queues_lock: threading.Lock = threading.Lock()
_alert_queues: dict[str, queue.Queue[tuple[int, pathlib.Path]]] = {}

_runtime_lock: threading.Lock = threading.Lock()
_runtime_state: dict[str, Any] = {
    'system': {},
    'feeds': {},
    'events': [],
}


def register_alert_queue(feed_id: str) -> queue.Queue[tuple[int, pathlib.Path]]:
    with _alert_queues_lock:
        q: queue.Queue[tuple[int, pathlib.Path]] = queue.Queue()
        _alert_queues[feed_id] = q
        return q


def push_alert(feed_id: str, priority: int, path: pathlib.Path) -> None:
    with _alert_queues_lock:
        q = _alert_queues.get(feed_id)
    if q is not None:
        q.put((priority, path))


def update_playout_sequence(feed_id: str, sequence: list[pathlib.Path]) -> None:
    with _sequences_lock:
        playout_sequences[feed_id] = list(sequence)


def get_playout_sequence(feed_id: str) -> list[pathlib.Path]:
    with _sequences_lock:
        return list(playout_sequences.get(feed_id, []))


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


def snapshot_playout_sequences() -> dict[str, list[pathlib.Path]]:
    with _sequences_lock:
        return {feed_id: list(sequence) for feed_id, sequence in playout_sequences.items()}


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
    event = {
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

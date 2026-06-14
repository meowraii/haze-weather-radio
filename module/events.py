import queue
import threading
from collections import deque
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
    item_id: str = ''
    group_id: str | None = None
    gap_after_s: float = 0.0
    estimated_duration: float | None = None

shutdown_event: threading.Event = threading.Event()
initial_synthesis_done: threading.Event = threading.Event()

_data_pool: dict[str, Any] = {}
_data_pool_lock: threading.Lock = threading.Lock()
_data_pool_version = 0
data_ready: threading.Event = threading.Event()

tts_queue: queue.Queue[tuple[Any, ...] | None] = queue.Queue()

_sequences_lock: threading.Lock = threading.Lock()
_playout_sequences: dict[str, list[PlayableItem]] = {}
_sequence_versions: dict[str, int] = {}
_last_played_item_ids: dict[str, str] = {}

_alert_queues_lock: threading.Lock = threading.Lock()
_alert_queues: dict[str, queue.Queue[tuple[int, bytes, str]]] = {}
_alert_queue_version = 0

_alert_audio_streams_lock: threading.Lock = threading.Lock()
_alert_audio_streams: dict[str, list[queue.Queue[tuple[bytes, str]]]] = {}

_feed_audio_streams_lock: threading.Lock = threading.Lock()
_feed_audio_streams: dict[str, list[queue.Queue[tuple[bytes, str]]]] = {}

_runtime_alert_entries_lock: threading.Lock = threading.Lock()
_runtime_alert_entries: dict[str, dict[str, dict[str, Any]]] = {}

_runtime_lock: threading.Lock = threading.Lock()
_runtime_state: dict[str, Any] = {
    'system': {},
    'feeds': {},
    'events': [],
}
_runtime_version = 0

def register_alert_queue(feed_id: str) -> queue.Queue[tuple[int, bytes, str]]:
    with _alert_queues_lock:
        q: queue.Queue[tuple[int, bytes, str]] = queue.Queue(maxsize=128)
        _alert_queues[feed_id] = q
        return q


def register_alert_audio_stream(feed_id: str) -> queue.Queue[tuple[bytes, str]]:
    q: queue.Queue[tuple[bytes, str]] = queue.Queue(maxsize=8)
    with _alert_audio_streams_lock:
        streams = _alert_audio_streams.setdefault(feed_id, [])
        streams.append(q)
    return q


def unregister_alert_audio_stream(feed_id: str, stream: queue.Queue[tuple[bytes, str]]) -> None:
    with _alert_audio_streams_lock:
        streams = _alert_audio_streams.get(feed_id)
        if not streams:
            return
        remaining = [item for item in streams if item is not stream]
        if remaining:
            _alert_audio_streams[feed_id] = remaining
        else:
            _alert_audio_streams.pop(feed_id, None)


def register_feed_audio_stream(feed_id: str) -> queue.Queue[tuple[bytes, str]]:
    q: queue.Queue[tuple[bytes, str]] = queue.Queue(maxsize=96)
    with _feed_audio_streams_lock:
        streams = _feed_audio_streams.setdefault(feed_id, [])
        streams.append(q)
    return q


def unregister_feed_audio_stream(feed_id: str, stream: queue.Queue[tuple[bytes, str]]) -> None:
    with _feed_audio_streams_lock:
        streams = _feed_audio_streams.get(feed_id)
        if not streams:
            return
        remaining = [item for item in streams if item is not stream]
        if remaining:
            _feed_audio_streams[feed_id] = remaining
        else:
            _feed_audio_streams.pop(feed_id, None)


def _publish_alert_audio(feed_id: str, pcm: bytes, identifier: str = '') -> None:
    with _alert_audio_streams_lock:
        streams = list(_alert_audio_streams.get(feed_id, ()))

    for stream in streams:
        try:
            stream.put_nowait((pcm, identifier))
            continue
        except queue.Full:
            pass

        try:
            stream.get_nowait()
        except queue.Empty:
            pass

        try:
            stream.put_nowait((pcm, identifier))
        except queue.Full:
            pass


def publish_feed_audio(feed_id: str, pcm: bytes, label: str = '') -> None:
    with _feed_audio_streams_lock:
        streams = list(_feed_audio_streams.get(feed_id, ()))

    if not streams or not pcm:
        return

    for stream in streams:
        try:
            stream.put_nowait((pcm, label))
            continue
        except queue.Full:
            pass

        try:
            stream.get_nowait()
        except queue.Empty:
            pass

        try:
            stream.put_nowait((pcm, label))
        except queue.Full:
            pass

def push_alert(feed_id: str, priority: int, pcm: bytes, identifier: str = '') -> bool:
    global _alert_queue_version
    with _alert_queues_lock:
        q = _alert_queues.get(feed_id)
    if q is not None:
        q.put((priority, pcm, identifier))
        with _alert_queues_lock:
            _alert_queue_version += 1
    _publish_alert_audio(feed_id, pcm, identifier)
    return q is not None


def push_alert_all(priority: int, pcm: bytes, identifier: str = '') -> None:
    global _alert_queue_version
    with _alert_queues_lock:
        queues = list(_alert_queues.values())
    with _alert_audio_streams_lock:
        stream_feed_ids = tuple(_alert_audio_streams.keys())
    for q in queues:
        q.put((priority, pcm, identifier))
    if queues:
        with _alert_queues_lock:
            _alert_queue_version += 1
    for feed_id in stream_feed_ids:
        _publish_alert_audio(feed_id, pcm, identifier)


def store_runtime_alert_entry(feed_id: str, identifier: str, entry: dict[str, Any]) -> None:
    if not feed_id or not identifier:
        return
    with _runtime_alert_entries_lock:
        feed_entries = _runtime_alert_entries.setdefault(feed_id, {})
        feed_entries[identifier] = deepcopy(entry)


def remove_runtime_alert_entries(feed_id: str, identifiers: list[str] | set[str] | tuple[str, ...]) -> None:
    if not feed_id:
        return
    removable = {str(identifier or '').strip() for identifier in identifiers if str(identifier or '').strip()}
    if not removable:
        return

    with _runtime_alert_entries_lock:
        feed_entries = _runtime_alert_entries.get(feed_id)
        if not feed_entries:
            return
        for identifier in removable:
            feed_entries.pop(identifier, None)
        if not feed_entries:
            _runtime_alert_entries.pop(feed_id, None)


def get_runtime_alert_entries(feed_id: str) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    with _runtime_alert_entries_lock:
        feed_entries = _runtime_alert_entries.get(feed_id, {})
        expired: list[str] = []
        active: list[dict[str, Any]] = []

        for identifier, entry in feed_entries.items():
            expires_raw = ((entry.get('metadata') or {}).get('expires') if isinstance(entry, dict) else None)
            if expires_raw:
                try:
                    expires_at = datetime.fromisoformat(str(expires_raw))
                    if expires_at.tzinfo is None:
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                    if expires_at < now:
                        expired.append(identifier)
                        continue
                except ValueError:
                    pass
            active.append(deepcopy(entry))

        for identifier in expired:
            feed_entries.pop(identifier, None)
        if not feed_entries:
            _runtime_alert_entries.pop(feed_id, None)

    return active


def update_playout_sequence(feed_id: str, sequence: list[PlayableItem]) -> None:
    with _sequences_lock:
        _playout_sequences[feed_id] = list(sequence)
        _sequence_versions[feed_id] = _sequence_versions.get(feed_id, 0) + 1


def append_playout_items(feed_id: str, sequence: list[PlayableItem]) -> None:
    if not sequence:
        return
    with _sequences_lock:
        pending = _playout_sequences.setdefault(feed_id, [])
        pending.extend(sequence)
        _sequence_versions[feed_id] = _sequence_versions.get(feed_id, 0) + 1


def pop_next_playout_item(feed_id: str) -> PlayableItem | None:
    with _sequences_lock:
        pending = _playout_sequences.get(feed_id)
        if not pending:
            return None
        item = pending.pop(0)
        _sequence_versions[feed_id] = _sequence_versions.get(feed_id, 0) + 1
        return item


def mark_playout_item_started(feed_id: str, item: PlayableItem) -> None:
    item_id = item.item_id.strip() if item.item_id else ''
    if not item_id:
        return
    with _sequences_lock:
        _last_played_item_ids[feed_id] = item_id


def get_last_played_item_id(feed_id: str) -> str | None:
    with _sequences_lock:
        return _last_played_item_ids.get(feed_id)


def get_playout_sequence(feed_id: str) -> list[PlayableItem]:
    with _sequences_lock:
        return list(_playout_sequences.get(feed_id, ()))


def get_sequence_version(feed_id: str) -> int:
    with _sequences_lock:
        return _sequence_versions.get(feed_id, 0)


def update_data_pool(key: str, value: Any, notify: bool = True) -> None:
    global _data_pool_version
    with _data_pool_lock:
        _data_pool[key] = value
        _data_pool_version += 1
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
    global _runtime_version
    with _runtime_lock:
        _runtime_state['system'].update(values)
        _runtime_version += 1


def update_feed_runtime(feed_id: str, values: dict[str, Any]) -> None:
    global _runtime_version
    with _runtime_lock:
        feed_state = _runtime_state['feeds'].setdefault(feed_id, {})
        feed_state.update(values)
        _runtime_version += 1


def push_public_stream_item(feed_id: str, title: str, queue_depth: int | None = None) -> None:
    global _runtime_version
    now = datetime.now(timezone.utc).isoformat()
    with _runtime_lock:
        feed_state = _runtime_state['feeds'].setdefault(feed_id, {})
        recent = feed_state.get('public_stream_recent_items')
        if not isinstance(recent, list):
            recent_list: deque[str] = deque(maxlen=8)
        else:
            recent_list = deque((str(item) for item in recent if str(item).strip()), maxlen=8)
        clean_title = str(title or '').strip() or 'Unknown'
        recent_list.appendleft(clean_title)
        feed_state['public_stream_now_playing'] = clean_title
        feed_state['public_stream_started_at'] = now
        feed_state['public_stream_recent_items'] = list(recent_list)
        if queue_depth is not None:
            feed_state['public_stream_queue_depth'] = max(0, int(queue_depth))
        _runtime_version += 1


def append_runtime_event(
    kind: str,
    message: str,
    feed_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    global _runtime_version
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
        _runtime_version += 1


def snapshot_runtime() -> dict[str, Any]:
    with _runtime_lock:
        return deepcopy(_runtime_state)


def snapshot_change_versions() -> dict[str, Any]:
    with _data_pool_lock:
        data_pool_version = _data_pool_version
    with _runtime_lock:
        runtime_version = _runtime_version
    with _sequences_lock:
        sequence_versions = dict(_sequence_versions)
    with _alert_queues_lock:
        alert_queue_version = _alert_queue_version
    return {
        'data_pool': data_pool_version,
        'runtime': runtime_version,
        'sequences': sequence_versions,
        'alert_queues': alert_queue_version,
    }


_scheduled_package_queues: dict[str, queue.Queue[str]] = {}
_scheduled_package_queues_lock: threading.Lock = threading.Lock()


def register_scheduled_queue(feed_id: str) -> queue.Queue[str]:
    with _scheduled_package_queues_lock:
        q: queue.Queue[str] = queue.Queue(maxsize=8)
        _scheduled_package_queues[feed_id] = q
        return q


def enqueue_scheduled_package(feed_id: str, pkg_id: str) -> None:
    with _scheduled_package_queues_lock:
        q = _scheduled_package_queues.get(feed_id)
    if q is None:
        return
    try:
        q.put_nowait(pkg_id)
    except queue.Full:
        pass

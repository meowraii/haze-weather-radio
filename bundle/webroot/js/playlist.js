import { panelClient } from './lib/ws-client.js';

const feedSelect = document.getElementById('playlistFeedSelect');
const statusBanner = document.getElementById('playlistStatusBanner');
const modeMetric = document.getElementById('playlistModeMetric');
const queueMetric = document.getElementById('playlistQueueMetric');
const currentTitle = document.getElementById('playlistCurrentTitle');
const currentMeta = document.getElementById('playlistCurrentMeta');
const nextTitle = document.getElementById('playlistNextTitle');
const nextMeta = document.getElementById('playlistNextMeta');
const queueList = document.getElementById('playlistQueueList');
const insertKind = document.getElementById('playlistInsertKind');
const insertProduct = document.getElementById('playlistInsertProduct');
const insertTitle = document.getElementById('playlistInsertTitle');
const insertText = document.getElementById('playlistInsertText');
const insertAudio = document.getElementById('playlistInsertAudio');
const insertSameEvent = document.getElementById('playlistInsertSameEvent');
const insertSameLocations = document.getElementById('playlistInsertSameLocations');
const insertPosition = document.getElementById('playlistInsertPosition');
const insertButton = document.getElementById('playlistInsertButton');

let bound = false;
let state = { feeds: [] };
let selectedFeed = '';

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

function fmtTime(value) {
    if (!value) return 'not set';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtDuration(ms) {
    const total = Math.max(0, Math.round(Number(ms || 0) / 1000));
    const min = Math.floor(total / 60);
    const sec = total % 60;
    return min ? `${min}m ${sec}s` : `${sec}s`;
}

function itemMeta(item) {
    if (!item) return '';
    const rows = [
        ['Queued', fmtTime(item.queued_at)],
        ['Target', fmtTime(item.target_start_at)],
        ['Starts', fmtTime(item.predicted_start_at)],
        ['Finishes', fmtTime(item.predicted_finish_at)],
        ['Duration', fmtDuration(item.duration_ms)],
    ];
    return rows.map(([key, value]) => `<dt>${escapeHtml(key)}</dt><dd>${escapeHtml(value)}</dd>`).join('');
}

function currentFeed() {
    return state.feeds.find((feed) => feed.feed_id === selectedFeed) || state.feeds[0] || null;
}

function renderFeedOptions() {
    const previous = selectedFeed;
    feedSelect.innerHTML = state.feeds.map((feed) => `
        <option value="${escapeHtml(feed.feed_id)}">${escapeHtml(feed.feed_name || feed.feed_id)}</option>
    `).join('');
    selectedFeed = state.feeds.some((feed) => feed.feed_id === previous) ? previous : (state.feeds[0]?.feed_id || '');
    feedSelect.value = selectedFeed;
}

function renderState() {
    renderFeedOptions();
    const feed = currentFeed();
    if (!feed) {
        statusBanner.textContent = 'No playlist feeds are available.';
        statusBanner.dataset.state = 'warn';
        return;
    }
    const queue = Array.isArray(feed.queue) ? feed.queue : [];
    const current = feed.current;
    const next = feed.next || queue[0];
    modeMetric.textContent = feed.mode || 'unknown';
    queueMetric.textContent = String(queue.length);
    currentTitle.textContent = current?.title || 'Idle';
    currentMeta.innerHTML = current ? itemMeta(current) : '<dt>Status</dt><dd>No product is currently playing.</dd>';
    nextTitle.textContent = next?.title || 'No queued product';
    nextMeta.innerHTML = next ? itemMeta(next) : '<dt>Status</dt><dd>The scheduler has not queued the next product yet.</dd>';
    queueList.innerHTML = queue.length ? queue.map((item) => `
        <article class="playlist-queue-item">
            <div>
                <strong>${escapeHtml(item.title || item.package_id || item.kind)}</strong>
                <span>${escapeHtml(item.package_id || item.kind || 'item')} · ${escapeHtml(item.source || 'scheduler')}</span>
            </div>
            <dl>${itemMeta(item)}</dl>
        </article>
    `).join('') : '<article class="playlist-empty">No queued products.</article>';
    statusBanner.textContent = feed.last_error ? `Playlist warning: ${feed.last_error}` : 'Playlist state loaded.';
    statusBanner.dataset.state = feed.last_error ? 'warn' : 'ok';
}

async function loadState() {
    const payload = await panelClient.command('playlist.state', {}, 8000);
    state = payload || { feeds: [] };
    renderState();
}

async function loadProducts() {
    const payload = await panelClient.command('wx.packages', {}, 8000);
    const packages = Array.isArray(payload.packages) ? payload.packages : [];
    insertProduct.innerHTML = packages.map((id) => `<option value="${escapeHtml(id)}">${escapeHtml(id)}</option>`).join('');
}

async function sendControl(action) {
    const feed = currentFeed();
    if (!feed) return;
    statusBanner.textContent = 'Sending playlist control…';
    statusBanner.dataset.state = 'pending';
    const result = await panelClient.command('playlist.control', { feed_id: feed.feed_id, action }, 8000);
    if (result?.playlist?.feeds) {
        state = result.playlist;
        renderState();
        if (!result.settled) {
            statusBanner.textContent = 'Control sent. Waiting for playlist service state…';
            statusBanner.dataset.state = 'pending';
        }
        return;
    }
    await loadState();
}

function insertPayload() {
    const feed = currentFeed();
    const kind = insertKind.value;
    const payload = {
        feed_id: feed?.feed_id,
        kind,
        position: insertPosition.value,
        title: insertTitle.value.trim(),
    };
    if (kind === 'product') payload.package_id = insertProduct.value;
    if (kind === 'tts') payload.text = insertText.value.trim();
    if (kind === 'audio') payload.audio_path = insertAudio.value.trim();
    if (kind === 'same') {
        payload.originator = 'WXR';
        payload.event = insertSameEvent.value.trim().toUpperCase() || 'RWT';
        payload.locations = insertSameLocations.value.split(',').map((part) => part.trim()).filter(Boolean);
        payload.duration_hours = 0;
        payload.duration_minutes = 15;
        payload.tone_type = 'WXR';
    }
    return payload;
}

async function insertItem() {
    statusBanner.textContent = 'Inserting playlist item…';
    statusBanner.dataset.state = 'pending';
    const result = await panelClient.command('playlist.insert', insertPayload(), 15000);
    if (result?.playlist?.feeds) {
        state = result.playlist;
        renderState();
        if (!result.settled) {
            statusBanner.textContent = 'Insert sent. Waiting for rendered playlist item…';
            statusBanner.dataset.state = 'pending';
        }
        return;
    }
    await loadState();
}

function updateInsertFields() {
    const kind = insertKind.value;
    document.querySelectorAll('[data-insert-field]').forEach((field) => {
        const value = field.dataset.insertField;
        field.classList.toggle('sp-hidden', !(
            value === 'title' ||
            value === kind ||
            (kind === 'tts' && value === 'text') ||
            (kind === 'audio' && value === 'audio')
        ));
    });
}

export function initPlaylistView() {
    if (bound) {
        loadState().catch((error) => {
            statusBanner.textContent = error.message || 'Unable to load playlist state.';
            statusBanner.dataset.state = 'err';
        });
        return;
    }
    bound = true;
    feedSelect.addEventListener('change', () => {
        selectedFeed = feedSelect.value;
        renderState();
    });
    document.querySelectorAll('[data-playlist-action]').forEach((button) => {
        button.addEventListener('click', () => {
            sendControl(button.dataset.playlistAction).catch((error) => {
                statusBanner.textContent = error.message || 'Playlist control failed.';
                statusBanner.dataset.state = 'err';
            });
        });
    });
    insertKind.addEventListener('change', updateInsertFields);
    insertButton.addEventListener('click', () => {
        insertItem().catch((error) => {
            statusBanner.textContent = error.message || 'Insert failed.';
            statusBanner.dataset.state = 'err';
        });
    });
    window.addEventListener('haze:admin-state', (event) => {
        if (event.detail?.playlist?.feeds) {
            state = event.detail.playlist;
            renderState();
        }
    });
    updateInsertFields();
    Promise.all([loadProducts(), loadState()]).catch((error) => {
        statusBanner.textContent = error.message || 'Unable to load playlist page.';
        statusBanner.dataset.state = 'err';
    });
}

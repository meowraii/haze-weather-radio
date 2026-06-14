import { createControlClient, panelClient } from './lib/ws-client.js';

const heroTitle = document.getElementById('heroTitle');
const heroSubtitle = document.getElementById('heroSubtitle');
const summaryCards = document.getElementById('summaryCards');
const feedsGrid = document.getElementById('feedsGrid');
const eventsList = document.getElementById('eventsList');
const logsView = document.getElementById('logsView');
const datapoolView = document.getElementById('datapoolView');
const configView = document.getElementById('configView');
const logSourceSelect = document.getElementById('logSourceSelect');
const rwtButton = document.getElementById('rwtButton');

let handlersBound = false;
let socketEventsBound = false;
let lastDashboardState = null;
let authCloseCheckPending = false;
let initialStateTimer = null;
const stateWaiters = new Set();

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

function setCodeBlock(element, value) {
    element.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function formatUptime(seconds) {
    const s = Math.round(seconds || 0);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function renderSummary(summary) {
    const enabled = summary.enabled_feed_count ?? 0;
    const total = summary.feed_count ?? 0;
    heroTitle.textContent = `${enabled} of ${total} feed${total !== 1 ? 's' : ''} online`;
    heroSubtitle.textContent = `${summary.data_pool_key_count ?? 0} data pool keys · uptime ${formatUptime(summary.uptime_seconds)}`;

    const totalQueue = (summary.feeds || []).reduce((n, f) => n + (f.alert_queue_depth || 0), 0);

    const cards = [
        ['Feeds online', `${enabled}/${total}`],
        ['Alert queue', totalQueue],
        ['Weather keys', summary.data_pool_key_count ?? 0],
        ['Uptime', formatUptime(summary.uptime_seconds)],
    ];

    summaryCards.innerHTML = cards.map(([label, value]) => `
        <article class="metric-card">
            <p>${escapeHtml(label)}</p>
            <strong>${escapeHtml(value)}</strong>
        </article>
    `).join('');
}

function populateWxFeedSelect(feeds) {
    const sel = document.getElementById('wxFeedSelect');
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = feeds.map((f) =>
        `<option value="${escapeHtml(f.id)}">${escapeHtml(f.name || f.id)}</option>`
    ).join('');
    if (prev && feeds.some((f) => f.id === prev)) sel.value = prev;
}

function renderFeeds(feeds) {
    populateWxFeedSelect(feeds);

    if (!feeds.length) {
        feedsGrid.innerHTML = '<article class="feed-card empty">No feeds configured.</article>';
        return;
    }

    feedsGrid.innerHTML = feeds.map((feed) => {
        const runtime = feed.runtime || {};
        const playlist = (feed.playlist_items || []).slice(0, 6).map((item) => `<li>${escapeHtml(item)}</li>`).join('');
        const outputs = (feed.outputs || []).map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join('');
        const latestAlert = runtime.last_alert_event
            ? `${runtime.last_alert_event} · ${runtime.last_alert_severity || 'n/a'}`
            : 'No queued alert activity yet';

        return `
            <article class="feed-card">
                <div class="feed-topline">
                    <div>
                        <p class="feed-id">${escapeHtml(feed.id)}</p>
                        <h3>${escapeHtml(feed.name)}</h3>
                    </div>
                    <span class="queue-chip" data-active="${feed.alert_queue_depth > 0}">Queue ${escapeHtml(feed.alert_queue_depth)}</span>
                </div>
                <div class="feed-meta">
                    <span class="tag">${escapeHtml(feed.timezone)}</span>
                    <span class="tag">${escapeHtml(feed.location_count)} locations</span>
                    <span class="tag">${escapeHtml((feed.languages || []).join(', '))}</span>
                </div>
                <div class="tag-row">${outputs || '<span class="tag muted">No outputs</span>'}</div>
                <dl class="feed-stats">
                    <div>
                        <dt>Now Playing</dt>
                        <dd>${escapeHtml(runtime.now_playing || 'Idle')}</dd>
                    </div>
                    <div>
                        <dt>Latest Alert</dt>
                        <dd>${escapeHtml(latestAlert)}</dd>
                    </div>
                </dl>
                <div class="playlist-block">
                    <p>Playlist Snapshot</p>
                    <ul>${playlist || '<li>No playlist entries generated yet.</li>'}</ul>
                </div>
            </article>
        `;
    }).join('');
}

function renderEvents(events) {
    if (!events.length) {
        eventsList.innerHTML = '<article class="event-item empty">No runtime events captured yet.</article>';
        return;
    }

    eventsList.innerHTML = events.slice().reverse().map((event) => `
        <article class="event-item">
            <div class="event-head">
                <span class="event-kind">${escapeHtml(event.kind)}</span>
                <time>${escapeHtml(event.timestamp)}</time>
            </div>
            <p>${escapeHtml(event.message)}</p>
            ${event.feed_id ? `<span class="event-feed">${escapeHtml(event.feed_id)}</span>` : ''}
        </article>
    `).join('');
}

function formatDateTime(value) {
    if (!value) return 'unknown time';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString([], {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    });
}

function updateLastConnected(lastConnected) {
    const el = document.getElementById('lastConnected');
    if (!el) return;
    const ip = lastConnected?.ip || 'unknown IP';
    el.textContent = `Last connected from ${ip} at ${formatDateTime(lastConnected?.at)}`;
}

async function confirmAuthBeforeRedirect() {
    if (authCloseCheckPending) return;
    authCloseCheckPending = true;
    const client = createControlClient();
    try {
        const state = await client.request('auth_check', {}, 5000);
        if (state.authenticated) {
            setCodeBlock(logsView, 'Live panel connection refreshed.');
            panelClient.close();
            panelClient.connect();
            return;
        }
    } catch {
        // Fall through to the sign-in page only when the control socket cannot
        // confirm that this browser still has a valid session.
    } finally {
        client.close();
        authCloseCheckPending = false;
    }
    panelClient.close();
    setCodeBlock(logsView, 'Panel session could not be verified. Refresh or sign in again from the login page.');
}

function renderDashboardState(payload) {
    if (initialStateTimer) {
        window.clearTimeout(initialStateTimer);
        initialStateTimer = null;
    }
    lastDashboardState = payload;
    const summary = payload.summary || {};
    renderSummary(summary);
    renderFeeds(summary.feeds || []);
    renderEvents(payload.events || []);
    setCodeBlock(logsView, (payload.logs?.lines || []).join('\n'));
    setCodeBlock(datapoolView, payload.datapool || {});
    setCodeBlock(configView, payload.config || {});
    updateLastConnected(payload.last_connected);
    window.dispatchEvent(new CustomEvent('haze:admin-state', { detail: payload }));
    stateWaiters.forEach((resolve) => resolve(payload));
    stateWaiters.clear();
}

function connectDashboardSocket(force = false) {
    if (!lastDashboardState) {
        heroTitle.textContent = 'Waiting for panel data';
        heroSubtitle.textContent = 'Opening live websocket...';
        summaryCards.innerHTML = '';
        feedsGrid.innerHTML = '<article class="feed-card empty">Waiting for feed data.</article>';
        setCodeBlock(logsView, 'Connecting to live panel stream...');
    }
    panelClient.params = {
        source: logSourceSelect.value || 'app',
        lines: '120',
    };
    if (force) {
        panelClient.close();
    }
    panelClient.connect();
    if (!initialStateTimer) {
        initialStateTimer = window.setTimeout(() => {
            initialStateTimer = null;
            if (lastDashboardState) return;
            panelClient.command('state', {}, 5000)
                .then((payload) => renderDashboardState(payload))
                .catch(() => setCodeBlock(logsView, 'Connected, waiting for the first panel snapshot...'));
        }, 1200);
    }
}

export async function refreshDashboard(options = {}) {
    connectDashboardSocket(Boolean(options.force));
}

export function getDashboardState() {
    return lastDashboardState;
}

export function waitForDashboardState(timeoutMs = 5000) {
    if (lastDashboardState) {
        return Promise.resolve(lastDashboardState);
    }
    return new Promise((resolve, reject) => {
        const timer = window.setTimeout(() => {
            stateWaiters.delete(done);
            reject(new Error('Timed out waiting for live panel state.'));
        }, timeoutMs);
        function done(payload) {
            window.clearTimeout(timer);
            resolve(payload);
        }
        stateWaiters.add(done);
        refreshDashboard();
    });
}

export function initDashboard() {
    if (handlersBound) {
        refreshDashboard();
        return;
    }
    handlersBound = true;

    rwtButton.addEventListener('click', async () => {
        rwtButton.disabled = true;
        const prev = rwtButton.innerHTML;
        rwtButton.textContent = 'Sending…';
        try {
            await panelClient.command('same.test', { event_code: 'RWT' });
            rwtButton.textContent = 'Sent ✓';
            setTimeout(() => { rwtButton.innerHTML = prev; rwtButton.disabled = false; }, 3000);
        } catch {
            rwtButton.textContent = 'Failed';
            setTimeout(() => { rwtButton.innerHTML = prev; rwtButton.disabled = false; }, 3000);
        }
    });

    logSourceSelect.addEventListener('change', () => {
        setCodeBlock(logsView, 'Switching log source...');
        refreshDashboard({ force: true });
    });

    if (!socketEventsBound) {
        socketEventsBound = true;
        panelClient.addEventListener('admin_state', (event) => renderDashboardState(event.detail || {}));
        panelClient.addEventListener('auth_state', (event) => {
            if (event.detail?.authenticated === false) confirmAuthBeforeRedirect();
        });
        panelClient.addEventListener('close', (event) => {
            if (event.detail?.code === 1008) {
                setCodeBlock(logsView, 'Panel session check required. Verifying...');
                confirmAuthBeforeRedirect();
            }
        });
        panelClient.addEventListener('error', () => {
            setCodeBlock(logsView, 'Live panel connection interrupted. Reconnecting...');
        });
    }

    (function initScrollspy() {
        const links = document.querySelectorAll('.sidebar-link[data-section]');
        if (!links.length) return;
        const wrap = document.querySelector('.main-wrap');
        const sectionIds = [...links].map((l) => l.dataset.section).filter((id) => document.getElementById(id));
        function updateActive() {
            const scrollTop = wrap.scrollTop;
            let active = sectionIds[0];
            for (const id of sectionIds) {
                const el = document.getElementById(id);
                if (el && el.offsetTop - 100 <= scrollTop) active = id;
            }
            links.forEach((l) => l.classList.toggle('active', l.dataset.section === active));
        }
        wrap.addEventListener('scroll', updateActive, { passive: true });
        updateActive();
    })();

    refreshDashboard();
}

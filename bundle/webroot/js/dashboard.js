import { createControlClient, panelClient } from './lib/ws-client.js';

const heroTitle = document.getElementById('heroTitle');
const heroSubtitle = document.getElementById('heroSubtitle');
const summaryCards = document.getElementById('summaryCards');
const adminBuildInfo = document.getElementById('adminBuildInfo');
const feedsGrid = document.getElementById('feedsGrid');
const homeStatusBanner = document.getElementById('homeStatusBanner');
const rwtButton = document.getElementById('rwtButton');
const tlsNotice = document.getElementById('tlsNotice');
const tlsNoticeText = document.getElementById('tlsNoticeText');

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

function setHomeStatus(text, state = '') {
    if (!homeStatusBanner) return;
    homeStatusBanner.textContent = text;
    homeStatusBanner.dataset.state = state;
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
    const totalQueue = (summary.feeds || []).reduce((n, f) => n + (f.alert_queue_depth || 0), 0);
    if (heroTitle) {
        heroTitle.textContent = `${enabled} of ${total} feed${total !== 1 ? 's' : ''} online`;
    }
    if (heroSubtitle) {
        heroSubtitle.textContent = `${summary.data_pool_key_count ?? 0} data pool keys · uptime ${formatUptime(summary.uptime_seconds)}`;
    }

    const cards = [
        ['Site', summary.name || 'Haze Weather Radio'],
        ['Operator', summary.operator || 'unconfigured'],
        ['IP Address', summary.ip_address || 'unknown'],
        ['Hostname', summary.hostname || 'unknown'],
        ['Feeds', `${enabled}/${total}`],
        ['Uptime', formatUptime(summary.uptime_seconds)],
    ];

    summaryCards.innerHTML = cards.map(([label, value]) => `
        <div class="public-status-item">
            <span>${escapeHtml(label)}:</span>
            <strong>${escapeHtml(value)}</strong>
        </div>
    `).join('');
    if (adminBuildInfo) {
        adminBuildInfo.innerHTML = [
            ['Version', summary.version || 'dev'],
            ['Commit', summary.git_commit || 'unknown'],
            ['System', [summary.os || 'unknown', summary.architecture || 'unknown'].join(' / ')],
            ['Weather keys', summary.data_pool_key_count ?? 0],
            ['Alert queue', totalQueue],
        ].map(([label, value]) => `
            <span><b>${escapeHtml(label)}:</b><em>${escapeHtml(value)}</em></span>
        `).join('');
    }
    renderTLSNotice(summary.tls || {});
}

function renderTLSNotice(tls) {
    if (!tlsNotice || !tlsNoticeText) return;
    const show = Boolean(tls.actual_domain || tls.enabled || tls.needs_setup);
    tlsNotice.hidden = !show;
    if (!show) return;
    const domains = (tls.configured_domains || []).join(', ');
    const suffix = domains ? ` Domains: ${domains}.` : '';
    tlsNotice.dataset.state = tls.https ? 'ok' : (tls.needs_setup ? 'warn' : 'info');
    tlsNoticeText.textContent = `${tls.message || 'TLS status unavailable.'}${suffix}`;
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

function arrayValue(value) {
    return Array.isArray(value) ? value : [];
}

function compactList(values, fallback, limit = 3) {
    const items = arrayValue(values).map((item) => String(item || '').trim()).filter(Boolean);
    if (!items.length) return fallback;
    const shown = items.slice(0, limit);
    const rest = items.length - shown.length;
    return rest > 0 ? `${shown.join(', ')} +${rest}` : shown.join(', ');
}

function coverageNames(feed) {
    const regions = arrayValue(feed.coverage_regions)
        .map((region) => region?.name || region?.id || '')
        .filter(Boolean);
    if (regions.length) return compactList(regions, 'No coverage regions', 3);
    const count = Number(feed.location_count || 0);
    return count ? `${count} locations` : 'No configured locations';
}

function renderFeeds(feeds) {
    populateWxFeedSelect(feeds);

    if (!feeds.length) {
        feedsGrid.innerHTML = '<article class="admin-feed-empty">No feeds configured.</article>';
        setHomeStatus('No feeds configured.', 'warn');
        return;
    }

    const enabledCount = feeds.filter((feed) => feed.enabled !== false).length;
    const queued = feeds.reduce((total, feed) => total + Number(feed.alert_queue_depth || 0), 0);
    setHomeStatus(`${enabledCount}/${feeds.length} enabled · ${queued} queued alert${queued === 1 ? '' : 's'}`);

    feedsGrid.innerHTML = feeds.map((feed) => {
        const runtime = feed.runtime || {};
        const feedId = String(feed.id || '');
        const enabled = feed.enabled !== false;
        const playlist = arrayValue(feed.playlist_items).slice(0, 3).map((item) => `<li>${escapeHtml(item)}</li>`).join('');
        const outputs = arrayValue(feed.outputs).slice(0, 4).map((item) => `<span class="tag">${escapeHtml(item)}</span>`).join('');
        const outputOverflow = arrayValue(feed.outputs).length > 4 ? `<span class="tag muted">+${arrayValue(feed.outputs).length - 4}</span>` : '';
        const transmitter = feed.transmitter || {};
        const siteNames = compactList(transmitter.site_names, transmitter.site_name || feed.name || feedId, 2);
        const languages = compactList(feed.languages, 'en-CA', 3);
        const coverage = coverageNames(feed);
        const latestAlert = runtime.last_alert_event
            ? `${runtime.last_alert_event} · ${runtime.last_alert_severity || 'n/a'}`
            : 'No queued alert activity yet';
        const listenHref = `/listen?feed=${encodeURIComponent(feedId)}`;

        return `
            <article class="admin-feed-card" data-enabled="${enabled}">
                <div class="admin-feed-main">
                    <div class="admin-feed-title">
                        <div class="admin-feed-idline">
                            <span>${escapeHtml(feedId)}</span>
                            <span class="admin-feed-pill" data-state="${enabled ? 'on' : 'off'}">${enabled ? 'enabled' : 'disabled'}</span>
                        </div>
                        <h3 title="${escapeHtml(siteNames)}">${escapeHtml(siteNames)}</h3>
                    </div>
                    <div class="admin-feed-meta">
                        <span class="tag">${escapeHtml(feed.timezone || 'Local')}</span>
                        <span class="tag">${escapeHtml(languages)}</span>
                        <span class="tag">${escapeHtml(coverage)}</span>
                        ${outputs || '<span class="tag muted">No outputs</span>'}${outputOverflow}
                    </div>
                </div>

                <div class="admin-feed-detail">
                    <dl class="admin-feed-live">
                        <div>
                            <dt>Now playing</dt>
                            <dd title="${escapeHtml(runtime.now_playing || 'Idle')}">${escapeHtml(runtime.now_playing || 'Idle')}</dd>
                        </div>
                        <div>
                            <dt>Latest alert</dt>
                            <dd title="${escapeHtml(latestAlert)}">${escapeHtml(latestAlert)}</dd>
                        </div>
                    </dl>
                    <div class="admin-feed-playlist">
                        <p>Next playlist items</p>
                        <ol>${playlist || '<li>No playlist entries generated yet.</li>'}</ol>
                    </div>
                </div>

                <div class="admin-feed-side">
                    <div class="admin-feed-queue" data-active="${Number(feed.alert_queue_depth || 0) > 0}">
                        <strong>${escapeHtml(feed.alert_queue_depth || 0)}</strong>
                        <span>queued</span>
                    </div>
                    <div class="admin-feed-actions">
                        <a class="btn-action btn-link" href="#/same">Alert</a>
                        <a class="btn-action btn-link" href="#/wx">WX</a>
                        <a class="btn-action btn-link" href="${escapeHtml(listenHref)}">Listen</a>
                    </div>
                </div>
            </article>
        `;
    }).join('');
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
            setHomeStatus('Live panel connection verified.', 'ok');
            return true;
        }
    } catch {
        // Fall through to the sign-in page only when the control socket cannot
        // confirm that this browser still has a valid session.
    } finally {
        client.close();
        authCloseCheckPending = false;
    }
    panelClient.close();
    setHomeStatus('Panel session expired. Sign in again from the login page.', 'err');
    return false;
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
    updateLastConnected(payload.last_connected);
    window.dispatchEvent(new CustomEvent('haze:admin-state', { detail: payload }));
    stateWaiters.forEach((resolve) => resolve(payload));
    stateWaiters.clear();
}

function connectDashboardSocket(force = false) {
    if (!lastDashboardState) {
        if (heroTitle) heroTitle.textContent = 'Waiting for panel data';
        if (heroSubtitle) heroSubtitle.textContent = 'Opening live websocket...';
        summaryCards.innerHTML = '';
        if (adminBuildInfo) adminBuildInfo.innerHTML = '';
        feedsGrid.innerHTML = '<article class="admin-feed-empty">Waiting for feed data.</article>';
        setHomeStatus('Connecting to live panel stream...', 'pending');
    }
    panelClient.params = {
        source: 'app',
        lines: '0',
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
                .catch(() => setHomeStatus('Connected, waiting for the first panel snapshot...', 'pending'));
        }, 1200);
    }
}

function requestFreshStateAfterRecovery() {
    panelClient.command('state', {}, 8000)
        .then((payload) => renderDashboardState(payload))
        .catch(() => {
            if (!lastDashboardState) {
                setHomeStatus('Connected, waiting for the first panel snapshot...', 'pending');
            }
        });
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

    if (!socketEventsBound) {
        socketEventsBound = true;
        panelClient.addEventListener('open', (event) => {
            if (event.detail?.recovered) {
                setHomeStatus('Live panel connection recovered. Refreshing state...', 'ok');
                requestFreshStateAfterRecovery();
            }
            panelClient.request('auth_check', {}, 5000)
                .then((state) => {
                    if (state.authenticated === false) confirmAuthBeforeRedirect();
                })
                .catch(() => {});
        });
        panelClient.addEventListener('reconnecting', (event) => {
            const seconds = Math.max(1, Math.round((event.detail?.delay || 1000) / 1000));
            setHomeStatus(`Live panel connection interrupted. Reconnecting in ${seconds}s...`, 'warn');
        });
        panelClient.addEventListener('recovered', () => {
            setHomeStatus('Live panel websocket recovered.', 'ok');
        });
        panelClient.addEventListener('admin_state', (event) => renderDashboardState(event.detail || {}));
        panelClient.addEventListener('auth_state', (event) => {
            if (event.detail?.authenticated === false) confirmAuthBeforeRedirect();
        });
        panelClient.addEventListener('close', (event) => {
            if (event.detail?.code === 1008) {
                setHomeStatus('Panel session check required. Verifying...', 'warn');
                confirmAuthBeforeRedirect();
            } else if (event.detail?.reconnecting) {
                setHomeStatus('Live panel connection closed. Reconnecting...', 'warn');
            }
        });
        panelClient.addEventListener('error', () => {
            setHomeStatus('Live panel connection interrupted. Reconnecting...', 'err');
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

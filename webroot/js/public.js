import { session } from './lib/session.js';
import { PanelClient } from './lib/ws-client.js';

const API_BASE = '/api/public/v1';

const heroTitle = document.getElementById('publicHeroTitle');
const heroSubtitle = document.getElementById('publicHeroSubtitle');
const summaryCards = document.getElementById('publicSummaryCards');
const feedsGrid = document.getElementById('publicFeedsGrid');
const adminLink = document.getElementById('adminLink');
const adminHint = document.getElementById('adminHint');
const feedNotice = document.getElementById('publicFeedNotice');
const publicSummarySection = document.getElementById('publicSummary');
const publicFeedsSection = document.getElementById('publicFeedsSection');

let summaryState = null;
const playerState = new Map();
const currentPage = window.location.pathname === '/feeds' ? 'feeds' : 'home';
const publicClient = new PanelClient({
    base: API_BASE,
    stream: true,
    params: currentPage === 'feeds' ? { feeds: '1' } : {},
});
let lastSummarySignature = '';
let lastFeedsSignature = '';
let lastNoticeText = '';

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

function selectorValue(value) {
    return window.CSS?.escape ? CSS.escape(String(value ?? '')) : String(value ?? '').replace(/["\\]/g, '\\$&');
}

function authHeaders(extra = {}) {
    return session.authHeaders(extra);
}

async function apiPost(path, body) {
    const response = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        credentials: 'same-origin',
        body: JSON.stringify(body),
    });
    if (!response.ok) {
        const bodyJson = await response.json().catch(() => ({ detail: `Request failed: ${response.status}` }));
        throw new Error(bodyJson.detail || `Request failed: ${response.status}`);
    }
    return response.json();
}

function formatUptime(seconds) {
    const s = Math.round(seconds || 0);
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m`;
    return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function applyNavState() {
    document.querySelectorAll('[data-public-link]').forEach((link) => {
        link.classList.toggle('active', link.dataset.publicLink === currentPage);
    });
}

function applyPageLayout() {
    if (publicSummarySection) {
        publicSummarySection.hidden = currentPage !== 'home';
    }
    if (publicFeedsSection) {
        publicFeedsSection.hidden = currentPage !== 'feeds';
    }
}

function setTextIfChanged(element, text) {
    if (element && element.textContent !== text) {
        element.textContent = text;
    }
}

function setNotice(text) {
    if (lastNoticeText === text) return;
    lastNoticeText = text;
    setTextIfChanged(feedNotice, text);
}

function summarySignature(summary) {
    return JSON.stringify({
        name: summary.name || '',
        hostname: summary.hostname || '',
        operator: summary.operator || '',
        feed_count: summary.feed_count || 0,
        enabled_feed_count: summary.enabled_feed_count || 0,
        feeds_access: summary.feeds_access || '',
        admin_url: summary.admin_url || '',
        webrtc_enabled: Boolean(summary.webrtc_enabled),
        uptime_bucket: currentPage === 'home' ? Math.floor((summary.uptime_seconds || 0) / 10) : 0,
    });
}

function feedsSignature(feeds) {
    return JSON.stringify((feeds || []).map((feed) => {
        const tx = feed.transmitter || {};
        return {
            id: feed.id,
            enabled: Boolean(feed.enabled),
            site_names: tx.site_names || [tx.site_name || feed.name || ''],
            webrtc_enabled: Boolean(summaryState?.webrtc_enabled),
        };
    }));
}

function renderSummary(summary) {
    summaryState = summary;
    const signature = summarySignature(summary);
    if (signature === lastSummarySignature) {
        return;
    }
    lastSummarySignature = signature;

    setTextIfChanged(heroTitle, summary.name || 'Haze Weather Radio');
    const bits = [
        summary.hostname || 'unknown host',
        summary.operator || 'operator unconfigured',
        `uptime ${formatUptime(summary.uptime_seconds)}`,
    ];
    setTextIfChanged(heroSubtitle, bits.join(' · '));

    if (summary.admin_url) {
        adminLink.href = summary.admin_url;
        setTextIfChanged(adminHint, summary.feeds_access === 'auth_required'
            ? 'Protected feeds need an authenticated admin session in this browser.'
            : 'Open the admin panel for protected feeds and station controls.');
    }

    const cards = [
        ['Hostname', summary.hostname || 'unknown'],
        ['Operator', summary.operator || 'unconfigured'],
        ['Feeds', `${summary.enabled_feed_count}/${summary.feed_count}`],
        ['Uptime', formatUptime(summary.uptime_seconds)],
    ];

    summaryCards.innerHTML = cards.map(([label, value]) => `
        <article class="metric-card">
            <p>${escapeHtml(label)}</p>
            <strong>${escapeHtml(value)}</strong>
        </article>
    `).join('');
}

function cardMarkup(feed) {
    const tx = feed.transmitter || {};
    const siteNames = (tx.site_names || [tx.site_name]).filter(Boolean).join(', ') || feed.name || 'Unnamed site';
    const canStream = Boolean(summaryState?.webrtc_enabled) && feed.enabled;
    const feedId = String(feed.id || '');

    return `
        <article class="feed-card public-feed-card" data-feed-card="${escapeHtml(feedId)}">
            <div class="public-feed-main">
                <div class="public-feed-overview">
                    <div>
                        <p class="feed-id">${escapeHtml(feedId)}</p>
                        <h3>${escapeHtml(siteNames)}</h3>
                    </div>
                </div>
            </div>
            <div class="public-feed-controls">
                <div class="public-player">
                    <button class="btn-action public-play-btn" type="button" data-feed-play="${escapeHtml(feedId)}" ${canStream ? '' : 'disabled'}>
                        <i data-lucide="play" width="13" height="13"></i>
                        Play feed
                    </button>
                    <button class="btn-action public-stop-btn" type="button" data-feed-stop="${escapeHtml(feedId)}" disabled>
                        <i data-lucide="square" width="13" height="13"></i>
                        Stop
                    </button>
                    <span class="try-status" data-feed-status="${escapeHtml(feedId)}">${canStream ? 'Ready' : 'Streaming unavailable'}</span>
                </div>
                <div class="public-player-volume">
                    <i data-lucide="volume-2" width="14" height="14"></i>
                    <input type="range" min="0" max="100" value="100" step="1" data-feed-volume="${escapeHtml(feedId)}" aria-label="Feed volume">
                    <span data-feed-volume-label="${escapeHtml(feedId)}">100%</span>
                </div>
                <audio data-feed-audio="${escapeHtml(feedId)}" hidden autoplay></audio>
            </div>
        </article>
    `;
}

function renderFeeds(feeds) {
    if (summaryState?.feeds_access === 'disabled') {
        setNotice('Public feeds are disabled on this system.');
        if (lastFeedsSignature !== 'disabled') {
            lastFeedsSignature = 'disabled';
            feedsGrid.innerHTML = '';
        }
        return;
    }

    if (!feeds.length) {
        setNotice('No feeds are currently configured.');
        if (lastFeedsSignature !== 'empty') {
            lastFeedsSignature = 'empty';
            feedsGrid.innerHTML = '<article class="feed-card empty">No feeds configured.</article>';
        }
        return;
    }

    setNotice(summaryState?.feeds_access === 'auth_required'
        ? 'Feed streaming requires an authenticated admin session.'
        : 'Select a feed to start direct queue-based playback.');
    const signature = feedsSignature(feeds);
    if (signature === lastFeedsSignature) {
        return;
    }
    lastFeedsSignature = signature;
    feedsGrid.innerHTML = feeds.map(cardMarkup).join('');
    window.lucide?.createIcons();
    bindPlayerButtons();
}

function renderPublicState(payload) {
    const summary = payload.summary || {};
    renderSummary(summary);
    if (currentPage !== 'feeds') {
        return;
    }
    if (summary.feeds_access === 'disabled') {
        renderFeeds([]);
        return;
    }
    if (playerState.size === 0) {
        renderFeeds(summary.feeds || []);
    } else {
        setNotice('Live feed playback active.');
    }
}

function connectPublicSocket() {
    publicClient.connect();
}

function setFeedStatus(feedId, text) {
    const el = document.querySelector(`[data-feed-status="${selectorValue(feedId)}"]`);
    if (el) el.textContent = text;
}

function applyVolume(feedId, value) {
    const normalized = Math.max(0, Math.min(1, value / 100));
    const audio = document.querySelector(`[data-feed-audio="${selectorValue(feedId)}"]`);
    if (audio) {
        audio.volume = normalized;
    }
    const label = document.querySelector(`[data-feed-volume-label="${selectorValue(feedId)}"]`);
    if (label) {
        label.textContent = `${Math.round(value)}%`;
    }
    const state = playerState.get(feedId);
    if (state) {
        state.volume = normalized;
    }
}

async function stopFeed(feedId) {
    const state = playerState.get(feedId);
    if (!state) return;
    try {
        state.pc.getSenders().forEach((sender) => {
            if (sender.track) sender.track.stop();
        });
        state.pc.close();
    } catch {
        // ignore close errors
    }
    const audio = document.querySelector(`[data-feed-audio="${selectorValue(feedId)}"]`);
    if (audio) {
        audio.pause();
        audio.srcObject = null;
    }
    const stopBtn = document.querySelector(`[data-feed-stop="${selectorValue(feedId)}"]`);
    const playBtn = document.querySelector(`[data-feed-play="${selectorValue(feedId)}"]`);
    if (stopBtn) stopBtn.disabled = true;
    if (playBtn) playBtn.disabled = false;
    playerState.delete(feedId);
    setFeedStatus(feedId, 'Stopped');
}

async function startFeed(feedId) {
    if (playerState.has(feedId)) return;
    const audio = document.querySelector(`[data-feed-audio="${selectorValue(feedId)}"]`);
    const stopBtn = document.querySelector(`[data-feed-stop="${selectorValue(feedId)}"]`);
    const playBtn = document.querySelector(`[data-feed-play="${selectorValue(feedId)}"]`);
    if (!(audio && stopBtn && playBtn)) return;

    playBtn.disabled = true;
    setFeedStatus(feedId, 'Connecting…');

    try {
        const pc = new RTCPeerConnection();
        const volumeSlider = document.querySelector(`[data-feed-volume="${selectorValue(feedId)}"]`);
        const requestedVolume = Number(volumeSlider?.value || 100);
        playerState.set(feedId, { pc, volume: Math.max(0, Math.min(1, requestedVolume / 100)) });
        pc.addTransceiver('audio', { direction: 'recvonly' });
        pc.ontrack = (event) => {
            audio.srcObject = event.streams[0];
            const state = playerState.get(feedId);
            audio.volume = typeof state?.volume === 'number' ? state.volume : 1;
            audio.hidden = false;
            audio.play().catch(() => {});
        };
        pc.onconnectionstatechange = () => {
            if (pc.connectionState === 'connected') {
                stopBtn.disabled = false;
                setFeedStatus(feedId, 'Live');
            } else if (pc.connectionState === 'disconnected') {
                setFeedStatus(feedId, 'Reconnecting…');
            } else if (['failed', 'closed'].includes(pc.connectionState)) {
                stopFeed(feedId).catch(() => {});
            }
        };

        const offer = await pc.createOffer({ offerToReceiveAudio: true });
        await pc.setLocalDescription(offer);
        const answer = await apiPost(`/feeds/${encodeURIComponent(feedId)}/webrtc/offer`, {
            sdp: offer.sdp,
            type: offer.type,
        });
        await pc.setRemoteDescription(answer);
    } catch (error) {
        await stopFeed(feedId);
        setFeedStatus(feedId, error instanceof Error ? error.message : 'Unable to connect');
    }
}

function bindPlayerButtons() {
    document.querySelectorAll('[data-feed-play]').forEach((button) => {
        button.addEventListener('click', () => {
            startFeed(button.dataset.feedPlay).catch(() => {});
        });
    });
    document.querySelectorAll('[data-feed-stop]').forEach((button) => {
        button.addEventListener('click', () => {
            stopFeed(button.dataset.feedStop).catch(() => {});
        });
    });
    document.querySelectorAll('[data-feed-volume]').forEach((slider) => {
        const feedId = slider.dataset.feedVolume;
        applyVolume(feedId, Number(slider.value || 100));
        slider.addEventListener('input', () => {
            applyVolume(feedId, Number(slider.value || 100));
        });
    });
}

applyNavState();
applyPageLayout();
window.lucide?.createIcons();

publicClient.addEventListener('public_state', (event) => renderPublicState(event.detail || {}));
publicClient.addEventListener('decode_error', () => {
    heroSubtitle.textContent = 'Unable to decode public status update.';
});
publicClient.addEventListener('error', () => {
    if (currentPage === 'feeds') {
        feedNotice.textContent = 'Live feed directory unavailable. Reconnecting...';
    } else {
        heroSubtitle.textContent = 'Public status unavailable. Reconnecting...';
    }
});

connectPublicSocket();

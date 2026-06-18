import { PanelClient } from './lib/ws-client.js';

const API_BASE = '/api/public/v1';

const heroTitle = document.getElementById('publicHeroTitle');
const heroSubtitle = document.getElementById('publicHeroSubtitle');
const summaryCards = document.getElementById('publicSummaryCards');
const feedsGrid = document.getElementById('publicFeedsGrid');
const adminLink = document.getElementById('adminLink');
const adminHint = document.getElementById('adminHint');
const feedNotice = document.getElementById('publicFeedNotice');
const alertsNotice = document.getElementById('publicAlertsNotice');
const alertsList = document.getElementById('publicAlertsList');
const alertsLink = document.getElementById('publicAlertsLink');
const publicSummarySection = document.getElementById('publicSummary');
const publicFeedsSection = document.getElementById('publicFeedsSection');
const publicAlertsSection = document.getElementById('publicAlertsSection');
const publicTlsNotice = document.getElementById('publicTlsNotice');
const publicTlsNoticeText = document.getElementById('publicTlsNoticeText');

let summaryState = null;
const currentPath = window.location.pathname.replace(/\/+$/, '') || '/';
const currentPage = currentPath === '/feeds'
    ? 'feeds'
    : (currentPath === '/alerts' || currentPath === '/alerts/archive' ? 'alerts' : 'home');
const publicClient = new PanelClient({
    base: API_BASE,
    stream: true,
    params: currentPage === 'feeds' ? { feeds: '1' } : (currentPage === 'alerts' ? { alerts: '1' } : {}),
});
let lastSummarySignature = '';
let lastFeedsSignature = '';
let lastAlertsSignature = '';
let lastNoticeText = '';
let activeAlertTab = 'accepted';
const feedPlayers = new Map();
window.hazeFeedPlayers = feedPlayers;

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
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
    if (publicAlertsSection) {
        publicAlertsSection.hidden = currentPage !== 'alerts';
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
        media_available: Boolean(summary.media_available),
        alerts_archive: summary.alerts_archive || '',
        tls: {
            enabled: Boolean(summary.tls?.enabled),
            https: Boolean(summary.tls?.https),
            host: summary.tls?.host || '',
            needs_setup: Boolean(summary.tls?.needs_setup),
            domains: summary.tls?.configured_domains || [],
        },
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
            webrtc_enabled: Boolean(feed.webrtc_enabled),
            media_available: Boolean(summaryState?.media_available),
        };
    }));
}

function findFeedElement(name, feedId) {
    return Array.from(document.querySelectorAll(`[data-${name}]`))
        .find((element) => element.dataset[toDatasetKey(name)] === feedId) || null;
}

function toDatasetKey(name) {
    return name.replace(/-([a-z])/g, (_, char) => char.toUpperCase());
}

function renderSummary(summary) {
    summaryState = summary;
    const signature = summarySignature(summary);
    if (signature === lastSummarySignature) {
        return;
    }
    lastSummarySignature = signature;

    setTextIfChanged(heroTitle, summary.name || 'Haze Weather Radio');
    if (alertsLink) {
        alertsLink.hidden = summary.alerts_archive !== 'public' && !summary.capabilities?.public_alerts;
    }
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
    renderTLSNotice(summary.tls || {});
}

function renderTLSNotice(tls) {
    if (!publicTlsNotice || !publicTlsNoticeText) return;
    const show = Boolean(tls.actual_domain || tls.enabled || tls.needs_setup);
    publicTlsNotice.hidden = !show;
    if (!show) return;
    publicTlsNotice.dataset.state = tls.https ? 'ok' : (tls.needs_setup ? 'warn' : 'info');
    publicTlsNoticeText.textContent = tls.message || 'TLS status unavailable.';
}

function cardMarkup(feed) {
    const tx = feed.transmitter || {};
    const siteNames = (tx.site_names || [tx.site_name]).filter(Boolean).join(', ') || feed.name || 'Unnamed site';
    const feedId = String(feed.id || '');
    const nowPlaying = feed.runtime?.now_playing || 'Idle';
    const canPlay = Boolean(feed.enabled && summaryState?.webrtc_enabled);
    const status = canPlay
        ? (summaryState?.media_available ? 'Ready' : 'Waiting for playout audio')
        : (feed.enabled ? 'Streaming unavailable' : 'Feed disabled');

    return `
        <article class="feed-card public-feed-card" data-feed-card="${escapeHtml(feedId)}">
            <div class="public-feed-main">
                <div class="public-feed-overview">
                    <div>
                        <p class="feed-id">${escapeHtml(feedId)}</p>
                        <h3>${escapeHtml(siteNames)}</h3>
                        <p class="public-feed-now" data-feed-now="${escapeHtml(feedId)}">${escapeHtml(nowPlaying)}</p>
                    </div>
                </div>
            </div>
            <div class="public-feed-controls">
                <button class="btn-action public-player-btn" type="button" data-feed-play="${escapeHtml(feedId)}" data-feed-playable="${canPlay ? '1' : '0'}" ${canPlay ? '' : 'disabled'}>
                    <i data-lucide="play" width="14" height="14"></i>
                    <span>Play</span>
                </button>
                <button class="btn-action public-player-btn" type="button" data-feed-stop="${escapeHtml(feedId)}" disabled>
                    <i data-lucide="square" width="14" height="14"></i>
                    <span>Stop</span>
                </button>
                <label class="public-volume" aria-label="Feed volume">
                    <i data-lucide="volume-2" width="14" height="14"></i>
                    <input type="range" min="0" max="1" step="0.01" value="1" data-feed-volume="${escapeHtml(feedId)}">
                </label>
                <span class="public-player-status" data-feed-status="${escapeHtml(feedId)}">${escapeHtml(status)}</span>
                <audio data-feed-audio="${escapeHtml(feedId)}" autoplay playsinline></audio>
            </div>
        </article>
    `;
}

function updateFeedRuntime(feeds) {
    feeds.forEach((feed) => {
        const feedId = String(feed.id || '');
        const now = findFeedElement('feed-now', feedId);
        if (now) {
            setTextIfChanged(now, feed.runtime?.now_playing || 'Idle');
        }
    });
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

    if (summaryState?.feeds_access === 'auth_required') {
        setNotice('Feed details require an authenticated admin session.');
    } else {
        setNotice(summaryState?.webrtc_enabled
            ? (summaryState?.media_available ? 'Live feed streaming is available.' : 'Feeds are configured; waiting for playout audio.')
            : 'No WebRTC output sink is enabled for public feeds.');
    }
    const signature = feedsSignature(feeds);
    if (signature === lastFeedsSignature) {
        updateFeedRuntime(feeds);
        return;
    }
    stopRemovedPlayers(new Set(feeds.map((feed) => String(feed.id || ''))));
    lastFeedsSignature = signature;
    feedsGrid.innerHTML = feeds.map(cardMarkup).join('');
    attachFeedControls();
    reattachActivePlayers();
    window.lucide?.createIcons();
}

function attachFeedControls() {
    feedsGrid.querySelectorAll('[data-feed-play]').forEach((button) => {
        button.addEventListener('click', () => startFeed(button.dataset.feedPlay));
    });
    feedsGrid.querySelectorAll('[data-feed-stop]').forEach((button) => {
        button.addEventListener('click', () => stopFeed(button.dataset.feedStop));
    });
    feedsGrid.querySelectorAll('[data-feed-volume]').forEach((input) => {
        input.addEventListener('input', () => {
            const player = feedPlayers.get(input.dataset.feedVolume);
            if (player?.audio) {
                player.audio.volume = Number(input.value);
            }
        });
    });
}

function setPlayerStatus(feedId, message) {
    const status = findFeedElement('feed-status', feedId);
    if (status) {
        setTextIfChanged(status, message);
    }
}

function setPlayerButtons(feedId, playing) {
    const playButton = findFeedElement('feed-play', feedId);
    const stopButton = findFeedElement('feed-stop', feedId);
    if (playButton) playButton.disabled = playing || !summaryState?.webrtc_enabled || playButton.dataset.feedPlayable !== '1';
    if (stopButton) stopButton.disabled = !playing;
}

async function startFeed(feedId) {
    feedId = String(feedId || '');
    if (!feedId || !summaryState?.webrtc_enabled) {
        setPlayerStatus(feedId, 'Streaming unavailable');
        return;
    }
    const existingPlayer = feedPlayers.get(feedId);
    if (existingPlayer?.trackAttached) {
        await resumeFeedAudio(feedId, existingPlayer);
        return;
    }
    stopFeed(feedId, { silent: true });
    const audio = findFeedElement('feed-audio', feedId);
    const volume = findFeedElement('feed-volume', feedId);
    if (!audio) {
        return;
    }
    setPlayerStatus(feedId, 'Connecting...');
    setPlayerButtons(feedId, true);

    const pc = new RTCPeerConnection();
    const fallbackStream = new MediaStream();
    const player = {
        pc,
        audio,
        fallbackStream,
        remoteStream: fallbackStream,
        trackAttached: false,
        connected: false,
        mediaRecent: null,
    };
    feedPlayers.set(feedId, player);

    audio.volume = Number(volume?.value ?? 1);
    audio.autoplay = true;
    audio.controls = false;
    audio.muted = false;
    audio.playsInline = true;
    audio.srcObject = fallbackStream;
    audio.dataset.hazePlayerState = 'connecting';
    audio.dataset.hazeTrackAttached = '0';
    audio.onplaying = () => {
        if (isActivePlayer(feedId, player)) {
            audio.dataset.hazePlayerState = 'playing';
            setPlayerStatus(feedId, 'Playing');
        }
    };
    audio.onwaiting = () => {
        if (isActivePlayer(feedId, player)) {
            audio.dataset.hazePlayerState = 'waiting';
            setPlayerStatus(feedId, 'Buffering...');
        }
    };
    audio.onstalled = () => {
        if (isActivePlayer(feedId, player)) {
            audio.dataset.hazePlayerState = 'stalled';
            setPlayerStatus(feedId, 'Audio stalled');
        }
    };
    audio.onerror = () => {
        if (isActivePlayer(feedId, player)) {
            audio.dataset.hazePlayerState = 'error';
            setPlayerStatus(feedId, 'Audio playback error');
        }
    };
    audio.play()
        .then(() => {
            if (isActivePlayer(feedId, player) && !player.trackAttached) {
                audio.controls = false;
                audio.dataset.hazePlayerState = 'primed';
                setPlayerStatus(feedId, 'Connecting audio...');
            }
        })
        .catch(() => {
            if (isActivePlayer(feedId, player)) {
                audio.dataset.hazePlayerState = 'needs-play';
            }
        });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    pc.addEventListener('track', (event) => {
        const currentAudio = bindPlayerAudio(feedId, fallbackStream);
        if (!currentAudio) {
            setPlayerStatus(feedId, 'Audio element unavailable');
            return;
        }
        player.trackAttached = true;
        currentAudio.dataset.hazeTrackAttached = '1';
        currentAudio.dataset.hazeTrackState = event.track.readyState || '';
        event.track.onmute = () => {
            if (isActivePlayer(feedId, player)) {
                currentAudio.dataset.hazeTrackMuted = '1';
                setPlayerStatus(feedId, 'Audio track muted');
            }
        };
        event.track.onunmute = () => {
            if (isActivePlayer(feedId, player)) {
                currentAudio.dataset.hazeTrackMuted = '0';
                if (currentAudio.dataset.hazePlayerState === 'play-blocked' || currentAudio.dataset.hazePlayerState === 'needs-play') {
                    setPlayerStatus(feedId, 'Press Play to start audio');
                } else {
                    setPlayerStatus(feedId, currentAudio.paused ? 'Audio ready' : 'Playing');
                }
            }
        };
        event.track.onended = () => {
            if (isActivePlayer(feedId, player)) {
                currentAudio.dataset.hazeTrackState = 'ended';
                setPlayerStatus(feedId, 'Audio track ended');
            }
        };
        const stream = event.streams?.[0] || fallbackStream;
        if (!stream.getTracks().some((track) => track.id === event.track.id)) {
            stream.addTrack(event.track);
        }
        player.remoteStream = stream;
        currentAudio.srcObject = stream;
        currentAudio.dataset.hazeStreamAttached = currentAudio.srcObject ? '1' : '0';
        currentAudio.dataset.hazePlayerState = 'audio-attached';
        setPlayerStatus(feedId, 'Starting audio...');
        currentAudio.play()
            .then(() => {
                if (isActivePlayer(feedId, player) && !currentAudio.paused) {
                    currentAudio.controls = false;
                    currentAudio.dataset.hazePlayerState = 'playing';
                    setPlayerStatus(feedId, 'Playing');
                } else if (isActivePlayer(feedId, player)) {
                    currentAudio.dataset.hazePlayerState = 'audio-ready';
                    setPlayerStatus(feedId, 'Audio ready');
                }
            })
            .catch(() => {
                if (isActivePlayer(feedId, player)) {
                    currentAudio.controls = true;
                    currentAudio.dataset.hazePlayerState = 'play-blocked';
                    setPlayerStatus(feedId, 'Press Play to start audio');
                    setPlayerButtons(feedId, false);
                    const stopButton = findFeedElement('feed-stop', feedId);
                    if (stopButton) stopButton.disabled = false;
                }
            });
    });
    pc.addEventListener('connectionstatechange', () => {
        if (!isActivePlayer(feedId, player)) return;
        const currentAudio = player.audio;
        if (currentAudio) {
            currentAudio.dataset.hazeConnectionState = pc.connectionState;
            currentAudio.dataset.hazeIceState = pc.iceConnectionState;
        }
        if (pc.connectionState === 'connected') {
            player.connected = true;
            if (currentAudio?.dataset.hazePlayerState === 'play-blocked' || currentAudio?.dataset.hazePlayerState === 'needs-play') {
                setPlayerStatus(feedId, 'Press Play to start audio');
                setPlayerButtons(feedId, false);
                const stopButton = findFeedElement('feed-stop', feedId);
                if (stopButton) stopButton.disabled = false;
            } else {
                setPlayerStatus(feedId, player.trackAttached ? 'Audio connected' : 'Connected, waiting for audio...');
                setPlayerButtons(feedId, true);
            }
        } else if (pc.connectionState === 'disconnected') {
            setPlayerStatus(feedId, 'Reconnecting...');
            setPlayerButtons(feedId, true);
        } else if (pc.connectionState === 'failed' || pc.connectionState === 'closed') {
            stopFeed(feedId);
        } else {
            setPlayerStatus(feedId, pc.connectionState);
        }
    });

    try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitForIceGathering(pc);
        const local = pc.localDescription;
        const answer = await publicClient.request('webrtc_offer', {
            feed_id: feedId,
            sdp: local.sdp,
            sdp_type: local.type,
        }, 15000);
        player.mediaRecent = answer.media_recent !== false;
        await pc.setRemoteDescription({
            type: answer.sdp_type || 'answer',
            sdp: answer.sdp,
        });
        window.setTimeout(() => {
            const active = feedPlayers.get(feedId);
            if (active === player && !active.trackAttached) {
                setPlayerStatus(feedId, 'Connected, no audio track yet');
            }
        }, 5000);
        setPlayerStatus(feedId, player.mediaRecent ? 'Waiting for audio...' : 'Connected, waiting for playout audio...');
    } catch (error) {
        stopFeed(feedId, { silent: true });
        setPlayerStatus(feedId, error.message || 'Could not start stream');
    }
}

async function resumeFeedAudio(feedId, player) {
    const audio = bindPlayerAudio(feedId, player?.remoteStream || player?.fallbackStream || null);
    if (!audio) {
        setPlayerStatus(feedId, 'Audio element unavailable');
        return;
    }
    setPlayerButtons(feedId, true);
    audio.muted = false;
    try {
        await audio.play();
        audio.controls = false;
        audio.dataset.hazePlayerState = audio.paused ? 'audio-ready' : 'playing';
        setPlayerStatus(feedId, audio.paused ? 'Audio ready' : 'Playing');
    } catch {
        audio.controls = true;
        audio.dataset.hazePlayerState = 'play-blocked';
        setPlayerStatus(feedId, 'Press Play to start audio');
        setPlayerButtons(feedId, false);
        const stopButton = findFeedElement('feed-stop', feedId);
        if (stopButton) stopButton.disabled = false;
    }
}

function stopFeed(feedId, { silent = false } = {}) {
    feedId = String(feedId || '');
    const player = feedPlayers.get(feedId);
    if (player) {
        window.hazeLastStop = {
            feed_id: feedId,
            silent,
            connection_state: player.pc?.connectionState || '',
            ice_state: player.pc?.iceConnectionState || '',
            signaling_state: player.pc?.signalingState || '',
            track_attached: Boolean(player.trackAttached),
            at: new Date().toISOString(),
        };
        player.pc?.close();
        const audio = findFeedElement('feed-audio', feedId) || player.audio;
        if (audio) {
            audio.pause();
            audio.srcObject = null;
            audio.dataset.hazePlayerState = 'stopped';
            audio.dataset.hazeTrackAttached = '0';
        }
        feedPlayers.delete(feedId);
    }
    setPlayerButtons(feedId, false);
    if (!silent) {
        setPlayerStatus(feedId, summaryState?.webrtc_enabled ? 'Stopped' : 'Streaming unavailable');
    }
}

function isActivePlayer(feedId, player) {
    return Boolean(player && feedPlayers.get(String(feedId || '')) === player);
}

function stopRemovedPlayers(activeFeedIds) {
    Array.from(feedPlayers.keys()).forEach((feedId) => {
        if (!activeFeedIds.has(feedId)) {
            stopFeed(feedId, { silent: true });
        }
    });
}

function stopAllPlayers(options = {}) {
    Array.from(feedPlayers.keys()).forEach((feedId) => stopFeed(feedId, options));
}

function restartActivePlayersAfterRecovery() {
    const activeFeedIds = Array.from(feedPlayers.keys());
    if (!activeFeedIds.length) return;
    for (const feedId of activeFeedIds) {
        setPlayerStatus(feedId, 'Stream recovered. Reconnecting audio...');
        stopFeed(feedId, { silent: true });
        window.setTimeout(() => startFeed(feedId), 250);
    }
}

function bindPlayerAudio(feedId, fallbackStream = null) {
    const player = feedPlayers.get(feedId);
    const audio = findFeedElement('feed-audio', feedId) || player?.audio;
    if (!player || !audio) return null;
    const volume = findFeedElement('feed-volume', feedId);
    audio.volume = Number(volume?.value ?? audio.volume ?? 1);
    if (!audio.srcObject && fallbackStream) {
        audio.srcObject = fallbackStream;
    }
    audio.dataset.hazeStreamAttached = audio.srcObject ? '1' : '0';
    player.audio = audio;
    return audio;
}

function reattachActivePlayers() {
    for (const [feedId, player] of feedPlayers.entries()) {
        const audio = findFeedElement('feed-audio', feedId);
        if (!audio || audio === player.audio) {
            continue;
        }
        audio.volume = player.audio?.volume ?? 1;
        audio.srcObject = player.remoteStream || player.audio?.srcObject || null;
        audio.dataset.hazePlayerState = player.audio?.dataset?.hazePlayerState || 'reattached';
        audio.dataset.hazeTrackAttached = player.trackAttached ? '1' : '0';
        player.audio = audio;
        setPlayerButtons(feedId, true);
        setPlayerStatus(feedId, player.trackAttached ? 'Audio connected' : 'Connecting...');
        audio.play().catch(() => {});
    }
}

function waitForIceGathering(pc) {
    if (pc.iceGatheringState === 'complete') {
        return Promise.resolve();
    }
    return new Promise((resolve) => {
        const timeout = window.setTimeout(done, 3000);
        function done() {
            window.clearTimeout(timeout);
            pc.removeEventListener('icegatheringstatechange', onStateChange);
            resolve();
        }
        function onStateChange() {
            if (pc.iceGatheringState === 'complete') {
                done();
            }
        }
        pc.addEventListener('icegatheringstatechange', onStateChange);
    });
}

function normalizePublicAlerts(summary) {
    const source = summary.alerts || summary.alerts_archive_data || {};
    const byFeed = source.by_feed || {};
    const accepted = [];
    Object.entries(byFeed).forEach(([feedId, records]) => {
        (records || []).forEach((record) => accepted.push({ ...record, feed_id: record.feed_id || feedId }));
    });
    return {
        accepted,
        rejected: source.rejected || [],
        expired: source.expired || [],
    };
}

function alertInfo(record) {
    return record.alert?.infos?.[0] || {};
}

function alertTitle(record) {
    const info = alertInfo(record);
    return info.headline || info.event || record.id || 'Weather Alert';
}

function alertMeta(record) {
    const info = alertInfo(record);
    return [
        record.feed_id,
        record.status,
        info.severity,
        info.urgency,
        record.updated_at,
    ].filter(Boolean).join(' · ');
}

function alertBody(record) {
    const info = alertInfo(record);
    const areas = (info.areas || []).map((area) => area.description).filter(Boolean).join('; ');
    return [areas, info.description, info.instruction, record.reason].filter(Boolean).join('\n\n');
}

function renderPublicAlerts(summary) {
    if (summary.alerts_archive !== 'public' && !summary.capabilities?.public_alerts) {
        if (alertsNotice) alertsNotice.textContent = 'Public alert archive is disabled.';
        if (alertsList) alertsList.innerHTML = '';
        return;
    }
    const buckets = normalizePublicAlerts(summary);
    const records = buckets[activeAlertTab] || [];
    const signature = JSON.stringify({
        tab: activeAlertTab,
        records: records.map((record) => [record.id, record.feed_id, record.status, record.updated_at]),
    });
    if (signature === lastAlertsSignature) {
        return;
    }
    lastAlertsSignature = signature;
    if (alertsNotice) {
        alertsNotice.textContent = records.length
            ? `${records.length} ${activeAlertTab} alert${records.length === 1 ? '' : 's'}`
            : `No ${activeAlertTab} alerts.`;
    }
    if (!alertsList) {
        return;
    }
    alertsList.innerHTML = records.length ? records.map((record) => `
        <details class="alert-archive-card public-alert-card">
            <summary>
                <span>
                    <strong>${escapeHtml(alertTitle(record))}</strong>
                    <small>${escapeHtml(alertMeta(record))}</small>
                </span>
            </summary>
            <pre>${escapeHtml(alertBody(record) || 'No public alert details available.')}</pre>
        </details>
    `).join('') : '<article class="feed-card empty">No alerts in this bucket.</article>';
}

function renderPublicState(payload) {
    const summary = payload.summary || {};
    renderSummary(summary);
    if (currentPage === 'alerts') {
        renderPublicAlerts(summary);
        return;
    }
    if (currentPage !== 'feeds') {
        return;
    }
    if (summary.feeds_access === 'disabled') {
        renderFeeds([]);
        return;
    }
    renderFeeds(summary.feeds || []);
}

function connectPublicSocket() {
    publicClient.connect();
}

applyNavState();
applyPageLayout();
window.lucide?.createIcons();
document.querySelectorAll('[data-public-alert-tab]').forEach((button) => {
    button.addEventListener('click', () => {
        activeAlertTab = button.dataset.publicAlertTab || 'accepted';
        document.querySelectorAll('[data-public-alert-tab]').forEach((item) => {
            item.classList.toggle('active', item === button);
        });
        lastAlertsSignature = '';
        if (summaryState) {
            renderPublicAlerts(summaryState);
        }
    });
});

publicClient.addEventListener('public_state', (event) => renderPublicState(event.detail || {}));
publicClient.addEventListener('open', (event) => {
    if (event.detail?.recovered) {
        if (currentPage === 'feeds') setNotice('Live feed directory recovered.');
        if (currentPage === 'alerts' && alertsNotice) alertsNotice.textContent = 'Alert archive recovered.';
    }
});
publicClient.addEventListener('reconnecting', (event) => {
    const seconds = Math.max(1, Math.round((event.detail?.delay || 1000) / 1000));
    if (currentPage === 'feeds') {
        setNotice(`Live feed directory unavailable. Reconnecting in ${seconds}s...`);
        for (const feedId of feedPlayers.keys()) {
            setPlayerStatus(feedId, 'Connection interrupted. Reconnecting...');
        }
    } else if (currentPage === 'alerts') {
        alertsNotice.textContent = `Alert archive unavailable. Reconnecting in ${seconds}s...`;
    } else {
        heroSubtitle.textContent = `Public status unavailable. Reconnecting in ${seconds}s...`;
    }
});
publicClient.addEventListener('recovered', () => {
    if (currentPage === 'feeds') restartActivePlayersAfterRecovery();
});
publicClient.addEventListener('close', (event) => {
    if (!event.detail?.reconnecting) return;
    if (currentPage === 'feeds') {
        setNotice('Live feed directory connection closed. Reconnecting...');
    } else if (currentPage === 'alerts') {
        alertsNotice.textContent = 'Alert archive connection closed. Reconnecting...';
    } else {
        heroSubtitle.textContent = 'Public status connection closed. Reconnecting...';
    }
});
publicClient.addEventListener('decode_error', () => {
    heroSubtitle.textContent = 'Unable to decode public status update.';
});
publicClient.addEventListener('error', () => {
    if (currentPage === 'feeds') {
        feedNotice.textContent = 'Live feed directory unavailable. Reconnecting...';
    } else if (currentPage === 'alerts') {
        alertsNotice.textContent = 'Alert archive unavailable. Reconnecting...';
    } else {
        heroSubtitle.textContent = 'Public status unavailable. Reconnecting...';
    }
});

connectPublicSocket();

import { PanelClient } from './lib/ws-client.js';

const API_BASE = '/api/public/v1';

const summaryCards = document.getElementById('publicSummaryCards');
const buildInfo = document.getElementById('publicBuildInfo');
const feedsGrid = document.getElementById('publicFeedsGrid');
const adminLink = document.getElementById('adminLink');
const adminHint = document.getElementById('adminHint');
const feedNotice = document.getElementById('publicFeedNotice');
const alertsNotice = document.getElementById('publicAlertsNotice');
const alertsList = document.getElementById('publicAlertsList');
const alertsLink = document.getElementById('publicAlertsLink');
const publicSummarySection = document.getElementById('publicSummary');
const publicSiteTitle = document.getElementById('publicSiteTitle');
const publicFeedsSection = document.getElementById('publicFeedsSection');
const publicListenSection = document.getElementById('publicListenSection');
const listenNotice = document.getElementById('publicListenNotice');
const listenPanel = document.getElementById('publicListenPanel');
const publicAlertsSection = document.getElementById('publicAlertsSection');
const publicTlsNotice = document.getElementById('publicTlsNotice');
const publicTlsNoticeText = document.getElementById('publicTlsNoticeText');

let summaryState = null;
const currentPath = window.location.pathname.replace(/\/+$/, '') || '/';
const currentPage = currentPath === '/feeds' || currentPath === '/listen'
    ? 'feeds'
    : (currentPath === '/alerts' || currentPath === '/alerts/archive' ? 'alerts' : 'home');
const isListenPage = currentPath === '/listen';
const publicClient = new PanelClient({
    base: API_BASE,
    stream: true,
    params: currentPage === 'feeds' ? { feeds: '1' } : (currentPage === 'alerts' ? { alerts: '1' } : {}),
});
let lastSummarySignature = '';
let lastFeedsSignature = '';
let lastAlertsSignature = '';
let lastListenSignature = '';
let lastNoticeText = '';
let activeAlertTab = 'accepted';
const feedPlayers = new Map();
const feedPreferences = new Map();
const feedCodecFallbacks = new Map();
const feedReconnectBackoffs = new Map();
window.hazeFeedPlayers = feedPlayers;

function recordWebRTCEvent(feedId, event, details = {}) {
    const events = window.hazeWebRTCEvents || [];
    events.push({
        feed_id: String(feedId || ''),
        event,
        ...details,
        at: new Date().toISOString(),
    });
    window.hazeWebRTCEvents = events.slice(-80);
}

window.hazeDumpWebRTC = function hazeDumpWebRTC(feedId = '') {
    const requestedFeed = String(feedId || '').trim();
    const players = {};
    for (const [id, player] of feedPlayers.entries()) {
        if (requestedFeed && id !== requestedFeed) continue;
        const audio = player.audio || null;
        const codecFallback = feedCodecFallbacks.get(id) || null;
        const now = Date.now();
        players[id] = {
            mode: player.mode || '',
            connection_state: player.pc?.connectionState || '',
            ice_state: player.pc?.iceConnectionState || '',
            signaling_state: player.pc?.signalingState || '',
            track_attached: Boolean(player.trackAttached),
            has_live_track: hasLiveWebRTCAudioTrack(player),
            last_packet_age_ms: player.lastPacketAt ? Date.now() - player.lastPacketAt : null,
            last_audio_progress_age_ms: player.lastAudioProgressAt ? Date.now() - player.lastAudioProgressAt : null,
            packet_source_age_ms: webRTCPacketSourceAgeMS(player),
            packets_recent: hasRecentWebRTCPackets(player),
            playback_watchdog_active: Boolean(player.playbackWatchdogTimer),
            stagnant_stats_polls: Number(player.stagnantStatsPolls || 0),
            missing_stats_polls: Number(player.missingStatsPolls || 0),
            reconnect_pending: Boolean(player.reconnectPending || player.reconnectTimer),
            disconnect_reconnect_pending: Boolean(player.disconnectReconnectTimer),
            media_recent: player.mediaRecent,
            requested_codec: player.requestedCodec || 'auto',
            negotiated_codec: player.negotiatedCodec || '',
            negotiated_payload_type: player.negotiatedPayloadType,
            codec_fallback_applied: Boolean(player.codecFallbackApplied || codecFallback),
            codec_fallback: codecFallback,
            track_mute_pending: Boolean(player.trackMuteTimer),
            track_muted_reported: Boolean(player.trackMutedReported),
            track_mute_signals: Number(player.trackMuteSignalCount || 0),
            track_unmute_signals: Number(player.trackUnmuteSignalCount || 0),
            last_track_mute_signal_age_ms: player.lastTrackMuteSignalAt ? now - player.lastTrackMuteSignalAt : null,
            last_track_unmute_signal_age_ms: player.lastTrackUnmuteSignalAt ? now - player.lastTrackUnmuteSignalAt : null,
            track_states: webRTCAudioTrackStates(player, audio),
            output_mixer_active: Boolean(player.audioOutputMixer),
            output_mixer_state: player.audioOutputMixer?.context?.state || '',
            audio: audio ? {
                paused: audio.paused,
                ended: audio.ended,
                ready_state: audio.readyState,
                network_state: audio.networkState,
                muted: audio.muted,
                volume: audio.volume,
                player_state: audio.dataset.hazePlayerState || '',
                track_attached: audio.dataset.hazeTrackAttached || '',
                stream_attached: audio.dataset.hazeStreamAttached || '',
                stream_track_count: audio.srcObject?.getAudioTracks?.().length || 0,
            } : null,
        };
    }
    const stats = window.hazeLastWebRTCStats || {};
    return {
        at: new Date().toISOString(),
        requested_feed: requestedFeed,
        players,
        stats: requestedFeed ? { [requestedFeed]: stats[requestedFeed] || null } : stats,
        events: (window.hazeWebRTCEvents || [])
            .filter((event) => !requestedFeed || event.feed_id === requestedFeed)
            .slice(-30),
    };
};

const WEBRTC_CODECS = [
    ['auto', 'Auto'],
    ['g722', 'G.722'],
    ['pcmu', 'PCMU'],
    ['opus', 'Opus'],
];
const HTTP_CODECS = [
    ['pcm16', 'PCM16 / WAV'],
    ['opus', 'Opus / Ogg'],
    ['webm_opus', 'Opus / WebM'],
    ['aac', 'AAC / ADTS'],
    ['m4a', 'AAC / fragmented MP4'],
    ['mp3', 'MP3'],
    ['vorbis', 'Vorbis / Ogg'],
    ['flac', 'FLAC'],
    ['ogg_flac', 'FLAC / Ogg'],
    ['ulaw', 'G.711 u-law'],
    ['alaw', 'G.711 A-law'],
    ['raw_pcm16', 'Raw PCM16'],
];
const HTTP_CODEC_VALUES = new Set(HTTP_CODECS.map(([value]) => value));
const WEBRTC_TRANSIENT_STATUS_DELAY_MS = 2000;
const WEBRTC_STATS_INTERVAL_MS = 2000;
const WEBRTC_STAGNANT_STATS_POLLS = 3;
const WEBRTC_RECOVER_STATS_POLLS = 15;
const WEBRTC_RECONNECT_BASE_DELAY_MS = 1000;
const WEBRTC_RECONNECT_MAX_DELAY_MS = 10000;
const WEBRTC_DISCONNECT_GRACE_MS = 15000;
const WEBRTC_MEDIA_EVENT_GRACE_MS = 12000;
const WEBRTC_RECENT_PACKET_GRACE_MS = 30000;
const WEBRTC_HARD_PACKET_STALE_MS = 120000;
const WEBRTC_PLAYBACK_WATCHDOG_MS = 2500;
const WEBRTC_TRACK_MUTE_GRACE_MS = 2500;
const WEBRTC_OUTPUT_BED_GAIN = 0.0025;
const WEBRTC_OUTPUT_BED_FREQUENCY = 180;
const WEBRTC_OUTPUT_MIXER_RESUME_GRACE_MS = 1500;

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

function formatDateTime(value) {
    if (!value) return 'not set';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
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
        publicFeedsSection.hidden = currentPage !== 'feeds' || isListenPage;
    }
    if (publicListenSection) {
        publicListenSection.hidden = !isListenPage;
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
        ip_address: summary.ip_address || '',
        operator: summary.operator || '',
        version: summary.version || '',
        git_commit: summary.git_commit || '',
        os: summary.os || '',
        architecture: summary.architecture || '',
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
            http_stream_enabled: Boolean(feed.http_stream_enabled),
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

function publicWebRTCAudioElement(feedId) {
    feedId = String(feedId || '');
    let audio = Array.from(document.querySelectorAll('[data-feed-audio-sink]'))
        .find((element) => element.dataset.feedAudioSink === feedId) || null;
    if (!audio) {
        audio = document.createElement('audio');
        audio.dataset.feedAudioSink = feedId;
        audio.autoplay = true;
        audio.playsInline = true;
        audio.muted = false;
        audio.style.position = 'fixed';
        audio.style.width = '1px';
        audio.style.height = '1px';
        audio.style.opacity = '0';
        audio.style.pointerEvents = 'none';
        document.body.appendChild(audio);
    }
    if (audio.dataset.hazeSinkBound !== '1') {
        audio.dataset.hazeSinkBound = '1';
        audio.addEventListener('pause', () => {
            const feedId = audio.dataset.feedAudioSink;
            const player = feedPlayers.get(feedId);
            recordWebRTCEvent(audio.dataset.feedAudioSink, 'audio_pause', {
                ready_state: audio.readyState,
                network_state: audio.networkState,
                ended: audio.ended,
                packets_recent: hasRecentWebRTCPackets(player),
            });
            if (player?.mode === 'webrtc' && hasRecentWebRTCPackets(player)) {
                ensureWebRTCAudioPlaying(feedId, player, audio);
            }
        });
        audio.addEventListener('playing', () => {
            recordWebRTCEvent(audio.dataset.feedAudioSink, 'audio_playing', {
                ready_state: audio.readyState,
                network_state: audio.networkState,
            });
        });
    }
    return audio;
}

function renderSummary(summary) {
    summaryState = summary;
    const signature = summarySignature(summary);
    if (signature === lastSummarySignature) {
        return;
    }
    lastSummarySignature = signature;

    if (alertsLink) {
        alertsLink.hidden = summary.alerts_archive !== 'public' && !summary.capabilities?.public_alerts;
    }

    if (summary.admin_url) {
        adminLink.href = summary.admin_url;
        setTextIfChanged(adminHint, summary.feeds_access === 'auth_required'
            ? 'Protected feeds need an authenticated admin session in this browser.'
            : 'Open the admin panel for protected feeds and station controls.');
    }

    const siteName = summary.name || 'Haze Weather Radio';
    setTextIfChanged(publicSiteTitle, siteName);
    document.title = siteName;

    const cards = [
        ['Operator', summary.operator || 'unconfigured'],
        ['IP Address', summary.ip_address || 'unknown'],
        ['Hostname', summary.hostname || 'unknown'],
        ['Feeds', `${summary.enabled_feed_count}/${summary.feed_count}`],
        ['Uptime', formatUptime(summary.uptime_seconds)],
    ];

    if (summaryCards) {
        summaryCards.innerHTML = cards.map(([label, value]) => `
            <div class="public-status-item">
                <span>${escapeHtml(label)}:</span>
                <strong>${escapeHtml(value)}</strong>
            </div>
        `).join('');
    }
    if (buildInfo) {
        buildInfo.innerHTML = [
            ['Version', summary.version || 'dev'],
            ['Commit', summary.git_commit || 'unknown'],
            ['System', [summary.os || 'unknown', summary.architecture || 'unknown'].join(' / ')],
        ].map(([label, value]) => `
            <span><b>${escapeHtml(label)}:</b><em>${escapeHtml(value)}</em></span>
        `).join('');
    }
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
    const canWebRTC = feedCanWebRTC(feed);
    const canHTTP = feedCanHTTP(feed);
    const canPlay = canWebRTC || canHTTP;
    const prefs = normalizedFeedPreferences(feedId, feed);
    const feedState = !feed.enabled
        ? 'disabled'
        : (canPlay ? 'ready' : (summaryState?.media_available ? 'unavailable' : 'waiting'));
    const feedStateLabel = {
        disabled: 'Disabled',
        ready: 'Live',
        unavailable: 'Unavailable',
        waiting: 'Waiting',
    }[feedState];
    const status = canPlay
        ? (canWebRTC ? 'Ready' : (summaryState?.media_available ? 'Ready' : 'Waiting for playout audio'))
        : (feed.enabled ? 'Streaming unavailable' : 'Feed disabled');
    const modeOptions = [
        ['webrtc', 'WebRTC', canWebRTC],
        ['http', 'HTTP', canHTTP],
    ].map(([value, label, enabled]) => `
        <option value="${value}" ${prefs.mode === value ? 'selected' : ''} ${enabled ? '' : 'disabled'}>${label}</option>
    `).join('');
    const codecOptions = codecOptionsForMode(prefs.mode).map(([value, label]) => `
        <option value="${value}" ${prefs.codec === value ? 'selected' : ''}>${label}</option>
    `).join('');

    return `
        <article class="feed-card public-feed-card" data-feed-card="${escapeHtml(feedId)}" data-feed-state="${escapeHtml(feedState)}">
            <div class="public-feed-main">
                <div class="public-feed-overview">
                    <div>
                        <div class="public-feed-kicker">
                            <p class="feed-id">${escapeHtml(feedId)}</p>
                            <span class="public-feed-status-pill">${escapeHtml(feedStateLabel)}</span>
                        </div>
                        <h3>${escapeHtml(siteNames)}</h3>
                        <p class="public-feed-now" data-feed-now="${escapeHtml(feedId)}">${escapeHtml(nowPlaying)}</p>
                    </div>
                </div>
            </div>
            <div class="public-feed-controls">
                <label class="public-listen-field">
                    <span>Listen</span>
                    <select data-feed-mode="${escapeHtml(feedId)}" ${canPlay ? '' : 'disabled'}>
                        ${modeOptions}
                    </select>
                </label>
                <label class="public-listen-field">
                    <span>Codec</span>
                    <select data-feed-codec="${escapeHtml(feedId)}" ${canPlay ? '' : 'disabled'}>
                        ${codecOptions}
                    </select>
                </label>
                <button class="btn-action public-player-btn" type="button" data-feed-play="${escapeHtml(feedId)}" data-feed-playable="${canPlay ? '1' : '0'}" ${canPlay ? '' : 'disabled'}>
                    <i data-lucide="play" width="14" height="14"></i>
                    <span>Play</span>
                </button>
                <button class="btn-action public-player-btn" type="button" data-feed-stop="${escapeHtml(feedId)}" disabled>
                    <i data-lucide="square" width="14" height="14"></i>
                    <span>Stop</span>
                </button>
                <button class="btn-action public-player-btn" type="button" data-feed-share="${escapeHtml(feedId)}" title="Copy HTTP stream link" ${canHTTP ? '' : 'disabled'}>
                    <i data-lucide="copy" width="14" height="14"></i>
                    <span>Share</span>
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

function createFeedCard(feed) {
    const template = document.createElement('template');
    template.innerHTML = cardMarkup(feed).trim();
    return template.content.firstElementChild;
}

function updateExistingFeedCard(card, feed) {
    const tx = feed.transmitter || {};
    const feedId = String(feed.id || '');
    const siteNames = (tx.site_names || [tx.site_name]).filter(Boolean).join(', ') || feed.name || 'Unnamed site';
    const canWebRTC = feedCanWebRTC(feed);
    const canHTTP = feedCanHTTP(feed);
    const canPlay = canWebRTC || canHTTP;
    const feedState = !feed.enabled
        ? 'disabled'
        : (canPlay ? 'ready' : (summaryState?.media_available ? 'unavailable' : 'waiting'));
    const feedStateLabel = {
        disabled: 'Disabled',
        ready: 'Live',
        unavailable: 'Unavailable',
        waiting: 'Waiting',
    }[feedState];
    card.dataset.feedState = feedState;
    setTextIfChanged(card.querySelector('h3'), siteNames);
    setTextIfChanged(card.querySelector('.public-feed-status-pill'), feedStateLabel);
    const now = findFeedElement('feed-now', feedId);
    if (now) setTextIfChanged(now, feed.runtime?.now_playing || 'Idle');
    const mode = findFeedElement('feed-mode', feedId);
    if (mode) {
        mode.disabled = !canPlay;
        for (const option of mode.options) {
            option.disabled = option.value === 'webrtc' ? !canWebRTC : !canHTTP;
        }
    }
    const codec = findFeedElement('feed-codec', feedId);
    if (codec) codec.disabled = !canPlay;
    const play = findFeedElement('feed-play', feedId);
    if (play) {
        play.dataset.feedPlayable = canPlay ? '1' : '0';
        play.disabled = feedPlayers.has(feedId) || !canPlay;
    }
    const stop = findFeedElement('feed-stop', feedId);
    if (stop) stop.disabled = !feedPlayers.has(feedId);
    const share = findFeedElement('feed-share', feedId);
    if (share) share.disabled = !canHTTP;
    if (!feedPlayers.has(feedId)) {
        setPlayerStatus(feedId, canPlay
            ? (canWebRTC ? 'Ready' : (summaryState?.media_available ? 'Ready' : 'Waiting for playout audio'))
            : (feed.enabled ? 'Streaming unavailable' : 'Feed disabled'));
    }
}

function feedCanWebRTC(feed) {
    return Boolean(feed.enabled && feed.webrtc_enabled && summaryState?.webrtc_enabled);
}

function feedCanHTTP(feed) {
    return Boolean(feed.enabled && feed.http_stream_enabled && summaryState?.media_available);
}

function normalizedFeedPreferences(feedId, feed = null) {
    const current = feedPreferences.get(feedId) || { mode: 'webrtc', codec: 'auto' };
    const canWebRTC = feed ? feedCanWebRTC(feed) : true;
    const canHTTP = feed ? feedCanHTTP(feed) : true;
    let mode = current.mode === 'http' ? 'http' : 'webrtc';
    if (mode === 'webrtc' && !canWebRTC && canHTTP) mode = 'http';
    if (mode === 'http' && !canHTTP && canWebRTC) mode = 'webrtc';
    const allowedCodecs = codecOptionsForMode(mode).map(([value]) => value);
    let codec = allowedCodecs.includes(current.codec) ? current.codec : allowedCodecs[0];
    const prefs = { mode, codec };
    feedPreferences.set(feedId, prefs);
    return prefs;
}

function codecOptionsForMode(mode) {
    return mode === 'http' ? HTTP_CODECS : WEBRTC_CODECS;
}

function selectedFeedMode(feedId) {
    const select = findFeedElement('feed-mode', feedId);
    const value = select?.value === 'http' ? 'http' : 'webrtc';
    const prefs = normalizedFeedPreferences(feedId);
    prefs.mode = value;
    feedPreferences.set(feedId, prefs);
    return value;
}

function selectedFeedCodec(feedId) {
    const select = findFeedElement('feed-codec', feedId);
    const prefs = normalizedFeedPreferences(feedId);
    const allowedCodecs = codecOptionsForMode(prefs.mode).map(([value]) => value);
    const selected = select?.value || prefs.codec;
    const value = allowedCodecs.includes(selected) ? selected : allowedCodecs[0];
    prefs.codec = value;
    if (select && select.value !== value) {
        select.value = value;
    }
    feedPreferences.set(feedId, prefs);
    return value;
}

function setFeedMode(feedId, mode) {
    const prefs = normalizedFeedPreferences(feedId);
    prefs.mode = mode === 'http' ? 'http' : 'webrtc';
    prefs.codec = codecOptionsForMode(prefs.mode)[0][0];
    feedPreferences.set(feedId, prefs);
    updateFeedCodecSelect(feedId);
}

function updateFeedCodecSelect(feedId) {
    const codecSelect = findFeedElement('feed-codec', feedId);
    if (!codecSelect) return;
    const prefs = normalizedFeedPreferences(feedId);
    codecSelect.innerHTML = codecOptionsForMode(prefs.mode).map(([value, label]) => `
        <option value="${value}" ${prefs.codec === value ? 'selected' : ''}>${label}</option>
    `).join('');
}

function httpStreamURL(feedId, absolute = false, codec = null) {
    const url = new URL(`${API_BASE}/feed/audio`, window.location.origin);
    url.searchParams.set('feed', feedId);
    url.searchParams.set('codec', normalizeHTTPCodec(codec || selectedFeedCodec(feedId)));
    return absolute ? url.toString() : `${url.pathname}${url.search}`;
}

function listenPageURL(feedId, absolute = false, codec = null) {
    const url = new URL('/listen', window.location.origin);
    url.searchParams.set('feed', feedId);
    url.searchParams.set('codec', normalizeHTTPCodec(codec || selectedFeedCodec(feedId)));
    return absolute ? url.toString() : `${url.pathname}${url.search}`;
}

function normalizeHTTPCodec(codec) {
    const value = String(codec || '').trim().toLowerCase().replaceAll('-', '_');
    return HTTP_CODEC_VALUES.has(value) ? value : 'pcm16';
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
            ? 'Live feed streaming is available.'
            : 'No WebRTC output sink is enabled for public feeds.');
    }
    const signature = feedsSignature(feeds);
    if (signature === lastFeedsSignature) {
        updateFeedRuntime(feeds);
        return;
    }
    stopRemovedPlayers(new Set(feeds.map((feed) => String(feed.id || ''))));
    lastFeedsSignature = signature;
    const fragment = document.createDocumentFragment();
    for (const feed of feeds) {
        const feedId = String(feed.id || '');
        const existing = findFeedElement('feed-card', feedId);
        if (existing && feedPlayers.has(feedId)) {
            updateExistingFeedCard(existing, feed);
            fragment.appendChild(existing);
        } else {
            fragment.appendChild(createFeedCard(feed));
        }
    }
    feedsGrid.replaceChildren(fragment);
    attachFeedControls();
    reattachActivePlayers();
    window.lucide?.createIcons();
}

function attachFeedControls() {
    feedsGrid.querySelectorAll('[data-feed-mode]').forEach((select) => {
        if (select.dataset.hazeBound === '1') return;
        select.dataset.hazeBound = '1';
        select.addEventListener('change', () => {
            const feedId = select.dataset.feedMode;
            feedCodecFallbacks.delete(feedId);
            feedReconnectBackoffs.delete(feedId);
            setFeedMode(feedId, select.value);
            stopFeed(feedId, { silent: true });
            setPlayerStatus(feedId, select.value === 'http' ? 'HTTP selected' : 'WebRTC selected');
        });
    });
    feedsGrid.querySelectorAll('[data-feed-codec]').forEach((select) => {
        if (select.dataset.hazeBound === '1') return;
        select.dataset.hazeBound = '1';
        select.addEventListener('change', () => {
            const feedId = select.dataset.feedCodec;
            const prefs = normalizedFeedPreferences(feedId);
            prefs.codec = select.value;
            feedPreferences.set(feedId, prefs);
            feedCodecFallbacks.delete(feedId);
            feedReconnectBackoffs.delete(feedId);
            stopFeed(feedId, { silent: true });
            setPlayerStatus(feedId, 'Codec changed');
        });
    });
    feedsGrid.querySelectorAll('[data-feed-play]').forEach((button) => {
        if (button.dataset.hazeBound === '1') return;
        button.dataset.hazeBound = '1';
        button.addEventListener('click', () => startFeed(button.dataset.feedPlay));
    });
    feedsGrid.querySelectorAll('[data-feed-stop]').forEach((button) => {
        if (button.dataset.hazeBound === '1') return;
        button.dataset.hazeBound = '1';
        button.addEventListener('click', () => stopFeed(button.dataset.feedStop));
    });
    feedsGrid.querySelectorAll('[data-feed-share]').forEach((button) => {
        if (button.dataset.hazeBound === '1') return;
        button.dataset.hazeBound = '1';
        button.addEventListener('click', () => copyHTTPLink(button.dataset.feedShare));
    });
    feedsGrid.querySelectorAll('[data-feed-volume]').forEach((input) => {
        if (input.dataset.hazeBound === '1') return;
        input.dataset.hazeBound = '1';
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
    if (playButton) playButton.disabled = playing || playButton.dataset.feedPlayable !== '1';
    if (stopButton) stopButton.disabled = !playing;
}

function clearPlayerTimer(player, key) {
    const timer = player?.[key];
    if (timer) {
        window.clearTimeout(timer);
        player[key] = null;
    }
}

function clearPlayerInterval(player, key) {
    const timer = player?.[key];
    if (timer) {
        window.clearInterval(timer);
        player[key] = null;
    }
}

function setHealthyWebRTCStatus(feedId, player, audio = player?.audio) {
    cancelWebRTCReconnect(player);
    clearPlayerTimer(player, 'connectionStateTimer');
    clearPlayerTimer(player, 'disconnectReconnectTimer');
    clearPlayerTimer(player, 'mediaEventTimer');
    if (!isActivePlayer(feedId, player)) return;
    if (audio?.dataset.hazePlayerState === 'play-blocked' || audio?.dataset.hazePlayerState === 'needs-play') {
        setPlayerStatus(feedId, 'Press Play to start audio');
        return;
    }
    ensureWebRTCAudioPlaying(feedId, player, audio);
    setPlayerStatus(feedId, audio?.paused ? 'Audio ready' : 'Playing');
}

function ensureWebRTCAudioPlaying(feedId, player, audio = player?.audio) {
    if (!isActivePlayer(feedId, player) || player?.mode !== 'webrtc' || !audio) return;
    keepWebRTCAudioLive(feedId, player, audio);
    if (!audio.paused || audio.dataset.hazePlayerState === 'play-blocked' || audio.dataset.hazePlayerState === 'needs-play') return;
    recordWebRTCEvent(feedId, 'audio_resume_attempt', {
        ready_state: audio.readyState,
        network_state: audio.networkState,
        has_live_track: hasLiveWebRTCAudioTrack(player),
        packets_recent: hasRecentWebRTCPackets(player),
    });
    const playPromise = audio.play();
    if (playPromise && typeof playPromise.catch === 'function') {
        playPromise
            .then(() => {
                if (isActivePlayer(feedId, player)) {
                    audio.dataset.hazePlayerState = audio.paused ? 'audio-ready' : 'playing';
                    recordWebRTCEvent(feedId, 'audio_resume_ok', {
                        paused: audio.paused,
                        ready_state: audio.readyState,
                    });
                }
            })
            .catch((error) => {
                if (isActivePlayer(feedId, player)) {
                    audio.dataset.hazePlayerState = 'needs-play';
                    setPlayerStatus(feedId, 'Press Play to start audio');
                    recordWebRTCEvent(feedId, 'audio_resume_blocked', {
                        error: error?.name || 'play_failed',
                    });
                }
            });
    }
}

function keepWebRTCAudioLive(feedId, player, audio = player?.audio) {
    if (!isActivePlayer(feedId, player) || player?.mode !== 'webrtc' || !audio) return;
    let repaired = false;
    if (audio.muted) {
        audio.muted = false;
        repaired = true;
    }
    const tracks = [
        ...(audio.srcObject?.getAudioTracks?.() || []),
        ...(player?.remoteStream?.getAudioTracks?.() || []),
    ];
    tracks.forEach((track) => {
        if (track && track.enabled === false) {
            track.enabled = true;
            repaired = true;
        }
    });
    if (repaired) {
        recordWebRTCEvent(feedId, 'audio_liveness_repaired', {
            track_count: tracks.length,
            ready_state: audio.readyState,
            network_state: audio.networkState,
            packets_recent: hasRecentWebRTCPackets(player),
        });
    }
    if (player.audioOutputMixer?.context?.state === 'suspended') {
        player.audioOutputMixer.context.resume().catch(() => {});
    }
}

function closeWebRTCAudioOutput(player) {
    const mixer = player?.audioOutputMixer;
    if (!mixer) return;
    player.audioOutputMixer = null;
    if (mixer.resumeFallbackTimer) {
        window.clearTimeout(mixer.resumeFallbackTimer);
    }
    if (mixer.retryTimer) {
        window.clearTimeout(mixer.retryTimer);
    }
    if (mixer.outputTrackMuteTimer) {
        window.clearTimeout(mixer.outputTrackMuteTimer);
    }
    if (mixer.context) {
        mixer.context.onstatechange = null;
    }
    for (const track of mixer.outputStream?.getAudioTracks?.() || []) {
        track.onmute = null;
        track.onunmute = null;
        track.onended = null;
    }
    try {
        mixer.oscillator?.stop();
    } catch {
        // The oscillator may already be stopped during reconnect cleanup.
    }
    try {
        mixer.source?.disconnect();
        mixer.gain?.disconnect();
        mixer.oscillator?.disconnect();
    } catch {
        // Best-effort Web Audio cleanup.
    }
    if (mixer.context?.state !== 'closed') {
        mixer.context?.close?.().catch(() => {});
    }
}

function bindWebRTCAudioOutput(feedId, player, sourceStream, audio) {
    if (!sourceStream || !audio) return sourceStream;
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor || !sourceStream.getAudioTracks?.().length) {
        closeWebRTCAudioOutput(player);
        audio.srcObject = sourceStream;
        return sourceStream;
    }
    if (player.audioOutputMixer?.sourceStream === sourceStream) {
        audio.srcObject = player.audioOutputMixer.outputStream;
        player.audioOutputMixer.context?.resume?.().catch(() => {});
        return player.audioOutputMixer.outputStream;
    }
    closeWebRTCAudioOutput(player);
    try {
        const context = new AudioContextCtor({ latencyHint: 'interactive' });
        const destination = context.createMediaStreamDestination();
        const source = context.createMediaStreamSource(sourceStream);
        const oscillator = context.createOscillator();
        const gain = context.createGain();
        oscillator.type = 'sine';
        oscillator.frequency.value = WEBRTC_OUTPUT_BED_FREQUENCY;
        gain.gain.value = WEBRTC_OUTPUT_BED_GAIN;
        source.connect(destination);
        oscillator.connect(gain);
        gain.connect(destination);
        oscillator.start();
        const outputStream = destination.stream;
        player.audioOutputMixer = { context, source, oscillator, gain, destination, sourceStream, outputStream };
        context.onstatechange = () => {
            if (player.audioOutputMixer?.context !== context) return;
            recordWebRTCEvent(feedId, 'audio_output_mixer_state', {
                context_state: context.state,
                packets_recent: hasRecentWebRTCPackets(player),
            });
            if (context.state === 'suspended') {
                context.resume?.().catch(() => {});
            }
        };
        for (const track of outputStream.getAudioTracks?.() || []) {
            track.onmute = () => {
                if (player.audioOutputMixer?.outputStream !== outputStream) return;
                if (player.audioOutputMixer.outputTrackMuteTimer) {
                    window.clearTimeout(player.audioOutputMixer.outputTrackMuteTimer);
                }
                player.audioOutputMixer.outputTrackMuteTimer = window.setTimeout(() => {
                    if (player.audioOutputMixer?.outputStream !== outputStream
                        || hasRecentWebRTCPackets(player)
                        || !hasHardStaleWebRTCPackets(player)) {
                        return;
                    }
                    player.audioOutputMixer.outputTrackMutedReported = true;
                    recordWebRTCEvent(feedId, 'audio_output_track_mute', {
                        track_id: track.id || '',
                        ready_state: track.readyState || '',
                        context_state: context.state,
                        packets_recent: hasRecentWebRTCPackets(player),
                        packet_source_age_ms: webRTCPacketSourceAgeMS(player),
                    });
                }, WEBRTC_TRACK_MUTE_GRACE_MS);
                context.resume?.().catch(() => {});
            };
            track.onunmute = () => {
                if (player.audioOutputMixer?.outputStream !== outputStream) return;
                if (player.audioOutputMixer.outputTrackMuteTimer) {
                    window.clearTimeout(player.audioOutputMixer.outputTrackMuteTimer);
                    player.audioOutputMixer.outputTrackMuteTimer = null;
                }
                if (player.audioOutputMixer.outputTrackMutedReported) {
                    player.audioOutputMixer.outputTrackMutedReported = false;
                    recordWebRTCEvent(feedId, 'audio_output_track_unmute', {
                        track_id: track.id || '',
                        ready_state: track.readyState || '',
                        context_state: context.state,
                        packets_recent: hasRecentWebRTCPackets(player),
                    });
                }
            };
            track.onended = () => {
                if (player.audioOutputMixer?.outputStream !== outputStream) return;
                recordWebRTCEvent(feedId, 'audio_output_track_ended', {
                    track_id: track.id || '',
                    context_state: context.state,
                });
            };
        }
        audio.srcObject = outputStream;
        context.resume?.().catch(() => {});
        player.audioOutputMixer.resumeFallbackTimer = window.setTimeout(() => {
            if (player.audioOutputMixer?.outputStream !== outputStream || context.state === 'running') return;
            recordWebRTCEvent(feedId, 'audio_output_mixer_resume_retry', {
                context_state: context.state,
            });
            context.resume?.().catch(() => {});
        }, WEBRTC_OUTPUT_MIXER_RESUME_GRACE_MS);
        recordWebRTCEvent(feedId, 'audio_output_mixer_bound', {
            source_track_count: sourceStream.getAudioTracks?.().length || 0,
            output_track_count: outputStream.getAudioTracks?.().length || 0,
        });
        return outputStream;
    } catch (error) {
        closeWebRTCAudioOutput(player);
        audio.srcObject = sourceStream;
        recordWebRTCEvent(feedId, 'audio_output_mixer_failed', {
            error: error?.name || 'mixer_failed',
        });
        return sourceStream;
    }
}

function hasRecentWebRTCPackets(player, now = Date.now()) {
    const lastProgressAt = Number(player?.lastAudioProgressAt || player?.lastPacketAt || 0);
    return lastProgressAt > 0 && now - lastProgressAt <= WEBRTC_RECENT_PACKET_GRACE_MS;
}

function webRTCPacketSourceAgeMS(player, now = Date.now()) {
    const lastProgressAt = Number(player?.lastAudioProgressAt || player?.lastPacketAt || 0);
    const startedAt = Number(player?.startedAt || 0);
    const sourceAt = lastProgressAt || startedAt;
    return sourceAt > 0 ? now - sourceAt : Number.POSITIVE_INFINITY;
}

function hasHardStaleWebRTCPackets(player, now = Date.now()) {
    return webRTCPacketSourceAgeMS(player, now) >= WEBRTC_HARD_PACKET_STALE_MS;
}

function startWebRTCPlaybackWatchdog(feedId, player) {
    clearPlayerInterval(player, 'playbackWatchdogTimer');
    if (!player) return;
    player.playbackWatchdogTimer = window.setInterval(() => {
        if (!isActivePlayer(feedId, player) || player.mode !== 'webrtc' || player.stopping) {
            clearPlayerInterval(player, 'playbackWatchdogTimer');
            return;
        }
        if (hasRecentWebRTCPackets(player)) {
            keepWebRTCAudioLive(feedId, player, player.audio);
            ensureWebRTCAudioPlaying(feedId, player, player.audio);
        }
    }, WEBRTC_PLAYBACK_WATCHDOG_MS);
}

function webRTCStatsDelta(previous, snapshot, field, reset = false) {
    if (!previous || reset) return 0;
    return Number(snapshot?.[field] || 0) - Number(previous?.[field] || 0);
}

function hasLiveWebRTCAudioTrack(player) {
    const tracks = [
        ...(player?.audio?.srcObject?.getAudioTracks?.() || []),
        ...(player?.remoteStream?.getAudioTracks?.() || []),
    ];
    return tracks.some((track) => track.readyState === 'live');
}

function hasUsableWebRTCAudio(player) {
    return Boolean(player?.trackAttached || hasLiveWebRTCAudioTrack(player));
}

function shouldReconnectWebRTCForMissingPackets(player) {
    return !hasUsableWebRTCAudio(player) || hasHardStaleWebRTCPackets(player);
}

function markWebRTCPacketsRecent(feedId, player) {
    if (!player) return;
    const now = Date.now();
    player.lastPacketAt = now;
    player.lastAudioProgressAt = now;
    player.reconnectAttempts = 0;
    feedReconnectBackoffs.delete(String(feedId || ''));
    cancelWebRTCReconnect(player);
    clearPlayerTimer(player, 'connectionStateTimer');
    clearPlayerTimer(player, 'disconnectReconnectTimer');
    clearPlayerTimer(player, 'mediaEventTimer');
    clearWebRTCTrackMuteReport(feedId, player);
}

function scheduleWebRTCMediaEventStatus(feedId, player, state, message) {
    clearPlayerTimer(player, 'mediaEventTimer');
    player.mediaEventTimer = window.setTimeout(() => {
        player.mediaEventTimer = null;
        if (!isActivePlayer(feedId, player)
            || hasRecentWebRTCPackets(player)
            || (hasUsableWebRTCAudio(player) && !hasHardStaleWebRTCPackets(player))) {
            return;
        }
        const audio = player.audio;
        if (audio) {
            audio.dataset.hazePlayerState = state;
        }
        setPlayerStatus(feedId, message);
    }, WEBRTC_MEDIA_EVENT_GRACE_MS);
}

function webRTCTrackEventDetails(player) {
    const audio = player?.audio || null;
    return {
        connection_state: player?.pc?.connectionState || '',
        ice_state: player?.pc?.iceConnectionState || '',
        last_packet_age_ms: player?.lastPacketAt ? Date.now() - player.lastPacketAt : null,
        last_audio_progress_age_ms: player?.lastAudioProgressAt ? Date.now() - player.lastAudioProgressAt : null,
        packet_source_age_ms: webRTCPacketSourceAgeMS(player),
        packets_recent: hasRecentWebRTCPackets(player),
        hard_stale_packets: hasHardStaleWebRTCPackets(player),
        audio_paused: audio ? audio.paused : null,
        audio_ready_state: audio ? audio.readyState : null,
        output_mixer_state: player?.audioOutputMixer?.context?.state || '',
        track_states: webRTCAudioTrackStates(player, audio),
    };
}

function webRTCAudioTrackStates(player, audio = player?.audio || null) {
    const seen = new Set();
    const tracks = [];
    const addTracks = (source, stream) => {
        for (const track of stream?.getAudioTracks?.() || []) {
            if (!track || seen.has(`${source}:${track.id}`)) continue;
            seen.add(`${source}:${track.id}`);
            tracks.push({ source, track });
        }
    };
    addTracks('audio_element', audio?.srcObject);
    addTracks('mixer_output', player?.audioOutputMixer?.outputStream);
    addTracks('mixer_source', player?.audioOutputMixer?.sourceStream);
    addTracks('remote_stream', player?.remoteStream);
    return tracks
        .map(({ source, track }) => ({
            source,
            id: track.id || '',
            ready_state: track.readyState || '',
            enabled: Boolean(track.enabled),
            muted: Boolean(track.muted),
        }));
}

function scheduleWebRTCTrackMuteReport(feedId, player) {
    clearPlayerTimer(player, 'trackMuteTimer');
    player.trackMuteTimer = window.setTimeout(() => {
        player.trackMuteTimer = null;
        if (!isActivePlayer(feedId, player) || hasRecentWebRTCPackets(player) || !hasHardStaleWebRTCPackets(player)) {
            return;
        }
        player.trackMutedReported = true;
        recordWebRTCEvent(feedId, 'track_muted', webRTCTrackEventDetails(player));
        maybeFallbackWebRTCCodecToPCMU(feedId, player, 'sustained_track_mute');
    }, WEBRTC_TRACK_MUTE_GRACE_MS);
}

function clearWebRTCTrackMuteReport(feedId, player) {
    const hadPendingMute = Boolean(player?.trackMuteTimer);
    clearPlayerTimer(player, 'trackMuteTimer');
    if (!isActivePlayer(feedId, player)) return;
    if (player.trackMutedReported) {
        player.trackMutedReported = false;
        recordWebRTCEvent(feedId, 'track_unmuted', webRTCTrackEventDetails(player));
    }
}

function maybeFallbackWebRTCCodecToPCMU(feedId, player, reason) {
    if (!isActivePlayer(feedId, player) || player.mode !== 'webrtc' || player.stopping) return false;
    const requested = String(player.requestedCodec || 'auto').trim().toLowerCase();
    const negotiated = String(player.negotiatedCodec || '').trim().toLowerCase();
    if (player.codecFallbackApplied || requested !== 'auto' || negotiated !== 'g722') return false;
    player.codecFallbackApplied = true;
    const prefs = normalizedFeedPreferences(feedId);
    prefs.mode = 'webrtc';
    prefs.codec = 'pcmu';
    feedPreferences.set(feedId, prefs);
    feedCodecFallbacks.set(feedId, {
        reason,
        previous_codec: negotiated,
        requested_codec: requested,
        fallback_codec: 'pcmu',
        at: new Date().toISOString(),
    });
    const codecSelect = findFeedElement('feed-codec', feedId);
    if (codecSelect) codecSelect.value = 'pcmu';
    recordWebRTCEvent(feedId, 'codec_fallback_pcmu', {
        reason,
        previous_codec: negotiated,
        requested_codec: requested,
    });
    scheduleWebRTCReconnect(feedId, player, 'Reconnecting with PCMU audio...', { force: true });
    return true;
}

function startWebRTCStatsMonitor(feedId, player) {
    stopWebRTCStatsMonitor(player);
    if (!player?.pc?.getStats) return;
    player.statsPollTimer = window.setInterval(async () => {
        if (!isActivePlayer(feedId, player) || player.pc.connectionState === 'closed') {
            stopWebRTCStatsMonitor(player);
            return;
        }
        try {
            const statsResult = await readActiveInboundAudioStats(player.audioReceiver, player.pc, player.audioTrackId);
            const snapshot = statsResult?.snapshot || null;
            if (!snapshot) {
                if (player.pc.connectionState === 'connected') {
                    player.missingStatsPolls = (player.missingStatsPolls || 0) + 1;
                    player.stagnantStatsPolls = 0;
                    if (player.missingStatsPolls === WEBRTC_STAGNANT_STATS_POLLS) {
                        console.warn('Haze WebRTC inbound audio stats are missing.', {
                            feed_id: feedId,
                            connection_state: player.pc.connectionState,
                            ice_state: player.pc.iceConnectionState,
                            track_attached: Boolean(player.trackAttached),
                            at: new Date().toISOString(),
                        });
                        if (isActivePlayer(feedId, player) && !hasUsableWebRTCAudio(player)) {
                            setPlayerStatus(feedId, 'Waiting for audio frames...');
                        }
                    }
                    if (shouldReconnectWebRTCForMissingPackets(player) && player.missingStatsPolls >= WEBRTC_RECOVER_STATS_POLLS) {
                        if (maybeFallbackWebRTCCodecToPCMU(feedId, player, 'missing_inbound_stats')) {
                            return;
                        }
                        scheduleWebRTCReconnect(feedId, player, 'Reconnecting missing audio stream...');
                    }
                }
                return;
            }
            player.missingStatsPolls = 0;
            const previous = player.lastStats || null;
            const packetsReset = Boolean(previous && snapshot.packetsReceived < previous.packetsReceived);
            const packetsDelta = previous && !packetsReset ? snapshot.packetsReceived - previous.packetsReceived : 0;
            const bytesDelta = webRTCStatsDelta(previous, snapshot, 'bytesReceived', packetsReset);
            const concealedDelta = webRTCStatsDelta(previous, snapshot, 'concealedSamples', packetsReset);
            const silentConcealedDelta = webRTCStatsDelta(previous, snapshot, 'silentConcealedSamples', packetsReset);
            const jitterBufferEmittedDelta = webRTCStatsDelta(previous, snapshot, 'jitterBufferEmittedCount', packetsReset);
            const audioProgress = webRTCStatsShowAudioProgress(snapshot, {
                packetsDelta,
                bytesDelta,
                concealedDelta,
                silentConcealedDelta,
                jitterBufferEmittedDelta,
                packetsReset,
                previous,
            });
            player.lastStats = snapshot;
            window.hazeLastWebRTCStats = {
                ...(window.hazeLastWebRTCStats || {}),
                [feedId]: {
                    ...snapshot,
                    packets_delta: packetsDelta,
                    bytes_delta: bytesDelta,
                    concealed_samples_delta: concealedDelta,
                    silent_concealed_samples_delta: silentConcealedDelta,
                    jitter_buffer_emitted_delta: jitterBufferEmittedDelta,
                    audio_progress: audioProgress,
                    packets_reset: packetsReset,
                    stats_source: statsResult.source,
                    connection_state: player.pc.connectionState,
                    ice_state: player.pc.iceConnectionState,
                    packets_recent: hasRecentWebRTCPackets(player),
                    at: new Date().toISOString(),
                },
            };
            if (previous && !audioProgress && player.pc.connectionState === 'connected') {
                player.stagnantStatsPolls = (player.stagnantStatsPolls || 0) + 1;
                if (player.stagnantStatsPolls === WEBRTC_STAGNANT_STATS_POLLS) {
                    console.warn('Haze WebRTC inbound audio packets stalled.', window.hazeLastWebRTCStats[feedId]);
                    if (isActivePlayer(feedId, player) && !hasUsableWebRTCAudio(player)) {
                        setPlayerStatus(feedId, 'Waiting for audio frames...');
                    }
                }
                if (shouldReconnectWebRTCForMissingPackets(player) && player.stagnantStatsPolls >= WEBRTC_RECOVER_STATS_POLLS) {
                    console.warn('Haze WebRTC inbound audio packets stayed stalled; reconnecting.', window.hazeLastWebRTCStats[feedId]);
                    if (maybeFallbackWebRTCCodecToPCMU(feedId, player, 'stalled_inbound_packets')) {
                        return;
                    }
                    scheduleWebRTCReconnect(feedId, player, 'Reconnecting stalled audio...');
                }
            } else {
                if (audioProgress && player.audio) {
                    markWebRTCPacketsRecent(feedId, player);
                    ensureWebRTCAudioPlaying(feedId, player);
                }
                if (player.stagnantStatsPolls > 0 && isActivePlayer(feedId, player)) {
                    setHealthyWebRTCStatus(feedId, player);
                }
                player.stagnantStatsPolls = 0;
            }
        } catch (error) {
            console.warn('Unable to read Haze WebRTC stats.', error);
        }
    }, WEBRTC_STATS_INTERVAL_MS);
}

function webRTCStatsShowAudioProgress(snapshot, deltas) {
    if (!snapshot) return false;
    return Boolean(deltas?.packetsReset
        || Number(deltas?.packetsDelta || 0) > 0
        || Number(deltas?.bytesDelta || 0) > 0
        || Number(deltas?.concealedDelta || 0) > 0
        || Number(deltas?.silentConcealedDelta || 0) > 0
        || Number(deltas?.jitterBufferEmittedDelta || 0) > 0
        || (!deltas?.previous && (Number(snapshot.packetsReceived || 0) > 0 || Number(snapshot.bytesReceived || 0) > 0)));
}

function stopWebRTCStatsMonitor(player) {
    if (player?.statsPollTimer) {
        window.clearInterval(player.statsPollTimer);
        player.statsPollTimer = null;
    }
}

function detachWebRTCPlayerForReconnect(feedId, player) {
    if (!isActivePlayer(feedId, player) || player.mode === 'http') return;
    clearPlayerTimer(player, 'connectionStateTimer');
    clearPlayerTimer(player, 'disconnectReconnectTimer');
    clearPlayerTimer(player, 'mediaEventTimer');
    clearPlayerTimer(player, 'trackMuteTimer');
    stopWebRTCStatsMonitor(player);
    clearPlayerInterval(player, 'playbackWatchdogTimer');
    closeWebRTCAudioOutput(player);
    player.trackAttached = false;
    player.connected = false;
    player.remoteStream = player.fallbackStream || new MediaStream();
    player.audioReceiver = null;
    player.audioTrackId = '';
    player.lastStats = null;
    player.lastPacketAt = 0;
    player.lastAudioProgressAt = 0;
    player.stagnantStatsPolls = 0;
    player.missingStatsPolls = 0;
    player.mediaRecent = null;
    player.negotiatedCodec = '';
    player.negotiatedPayloadType = null;
    player.trackMutedReported = false;
    try {
        player.pc?.close();
    } catch {
        // Closing an already-failed peer is best-effort cleanup.
    }
    const audio = player.audio || publicWebRTCAudioElement(feedId);
    if (audio) {
        bindWebRTCAudioOutput(feedId, player, player.remoteStream, audio);
        audio.dataset.hazeTrackAttached = '0';
        audio.dataset.hazePlayerState = 'reconnecting';
    }
}

function scheduleWebRTCDisconnectReconnect(feedId, player, pc) {
    if (!isActivePlayer(feedId, player) || player.mode === 'http' || player.stopping) return;
    if (player.disconnectReconnectTimer || player.reconnectPending || player.reconnectTimer) return;
    player.disconnectReconnectTimer = window.setTimeout(() => {
        player.disconnectReconnectTimer = null;
        if (isActivePlayer(feedId, player) && player.pc === pc && pc.connectionState === 'disconnected') {
            if (!hasHardStaleWebRTCPackets(player)) {
                recordWebRTCEvent(feedId, 'disconnect_reconnect_deferred', {
                    connection_state: pc.connectionState,
                    ice_state: pc.iceConnectionState,
                    packets_recent: hasRecentWebRTCPackets(player),
                    last_audio_progress_age_ms: player.lastAudioProgressAt ? Date.now() - player.lastAudioProgressAt : null,
                    packet_source_age_ms: webRTCPacketSourceAgeMS(player),
                });
                scheduleWebRTCDisconnectReconnect(feedId, player, pc);
                return;
            }
            scheduleWebRTCReconnect(feedId, player, 'Reconnecting disconnected audio...');
        }
    }, WEBRTC_DISCONNECT_GRACE_MS);
}

function scheduleWebRTCReconnect(feedId, player, reason = 'Reconnecting audio...', { force = false } = {}) {
    if (!isActivePlayer(feedId, player) || player.mode === 'http' || player.stopping) return;
    if (force) {
        clearPlayerTimer(player, 'reconnectTimer');
        player.reconnectPending = false;
    } else if (player.reconnectPending || player.reconnectTimer) {
        return;
    }
    recordWebRTCEvent(feedId, 'reconnect_scheduled', {
        reason,
        connection_state: player.pc?.connectionState || '',
        ice_state: player.pc?.iceConnectionState || '',
        track_attached: Boolean(player.trackAttached),
        has_live_track: hasLiveWebRTCAudioTrack(player),
        last_packet_age_ms: player.lastPacketAt ? Date.now() - player.lastPacketAt : null,
        last_audio_progress_age_ms: player.lastAudioProgressAt ? Date.now() - player.lastAudioProgressAt : null,
        packet_source_age_ms: webRTCPacketSourceAgeMS(player),
    });
    const backoffKey = String(feedId || '');
    const attempts = Math.max(0, Number(feedReconnectBackoffs.get(backoffKey) ?? player.reconnectAttempts ?? 0));
    const delay = Math.min(WEBRTC_RECONNECT_BASE_DELAY_MS * (2 ** attempts), WEBRTC_RECONNECT_MAX_DELAY_MS);
    player.reconnectAttempts = attempts + 1;
    feedReconnectBackoffs.set(backoffKey, player.reconnectAttempts);
    player.reconnectPending = true;
    player.reconnectTimer = window.setTimeout(() => {
        player.reconnectTimer = null;
        player.reconnectPending = false;
        if (!isActivePlayer(feedId, player) || player.stopping) return;
        detachWebRTCPlayerForReconnect(feedId, player);
        feedPlayers.delete(String(feedId || ''));
        startFeedWebRTC(feedId);
    }, delay);
    setPlayerStatus(feedId, reason);
}

function cancelWebRTCReconnect(player) {
    if (!player) return;
    clearPlayerTimer(player, 'reconnectTimer');
    player.reconnectPending = false;
}

async function readInboundAudioStats(source, trackId = '') {
    const report = await source.getStats();
    let selected = null;
    const expectedTrackId = String(trackId || '').trim();
    report.forEach((stats) => {
        const kind = stats.kind || stats.mediaType;
        if (stats.type !== 'inbound-rtp' || stats.isRemote || (kind && kind !== 'audio')) return;
        const statsTrackId = String(stats.trackIdentifier || '').trim();
        if (expectedTrackId && statsTrackId && statsTrackId !== expectedTrackId) return;
        const statsTimestamp = Number(stats.timestamp || 0);
        const selectedTimestamp = Number(selected?.timestamp || 0);
        if (!selected
            || statsTimestamp > selectedTimestamp
            || (statsTimestamp === selectedTimestamp && (stats.packetsReceived || 0) > (selected.packetsReceived || 0))) {
            selected = stats;
        }
    });
    if (!selected) return null;
    return {
        packetsReceived: Number(selected.packetsReceived || 0),
        bytesReceived: Number(selected.bytesReceived || 0),
        packetsLost: Number(selected.packetsLost || 0),
        jitter: Number(selected.jitter || 0),
        concealedSamples: Number(selected.concealedSamples || 0),
        silentConcealedSamples: Number(selected.silentConcealedSamples || 0),
        jitterBufferDelay: Number(selected.jitterBufferDelay || 0),
        jitterBufferEmittedCount: Number(selected.jitterBufferEmittedCount || 0),
        trackIdentifier: selected.trackIdentifier || '',
        timestamp: selected.timestamp || performance.now(),
    };
}

async function readActiveInboundAudioStats(receiver, peer, trackId = '') {
    if (receiver?.getStats) {
        const snapshot = await readInboundAudioStats(receiver, trackId);
        if (snapshot) return { snapshot, source: 'receiver' };
    }
    if (peer?.getStats) {
        const snapshot = await readInboundAudioStats(peer, trackId);
        if (snapshot) return { snapshot, source: 'peer' };
    }
    return null;
}

async function startFeed(feedId) {
    if (selectedFeedMode(feedId) === 'http') {
        return startFeedHTTP(feedId);
    }
    return startFeedWebRTC(feedId);
}

async function startFeedHTTP(feedId) {
    feedId = String(feedId || '');
    if (!feedId || !summaryState?.media_available) {
        setPlayerStatus(feedId, 'HTTP streaming unavailable');
        return;
    }
    stopFeed(feedId, { silent: true });
    const audio = findFeedElement('feed-audio', feedId);
    const volume = findFeedElement('feed-volume', feedId);
    if (!audio) {
        return;
    }
    const streamURL = httpStreamURL(feedId);
    const player = {
        mode: 'http',
        audio,
        httpURL: streamURL,
        trackAttached: true,
        connected: true,
    };
    feedPlayers.set(feedId, player);
    audio.volume = Number(volume?.value ?? 1);
    audio.autoplay = true;
    audio.controls = false;
    audio.muted = false;
    audio.playsInline = true;
    audio.srcObject = null;
    audio.src = streamURL;
    audio.dataset.hazePlayerState = 'connecting-http';
    audio.onplaying = () => {
        if (isActivePlayer(feedId, player)) {
            audio.dataset.hazePlayerState = 'playing';
            setPlayerStatus(feedId, 'Playing over HTTP');
        }
    };
    audio.onwaiting = () => {
        if (isActivePlayer(feedId, player)) {
            audio.dataset.hazePlayerState = 'waiting';
            setPlayerStatus(feedId, 'Buffering HTTP audio...');
        }
    };
    audio.onerror = () => {
        if (isActivePlayer(feedId, player)) {
            audio.dataset.hazePlayerState = 'error';
            setPlayerStatus(feedId, 'HTTP audio error');
            setPlayerButtons(feedId, false);
        }
    };
    setPlayerStatus(feedId, 'Connecting over HTTP...');
    setPlayerButtons(feedId, true);
    try {
        await audio.play();
        setPlayerStatus(feedId, audio.paused ? 'HTTP audio ready' : 'Playing over HTTP');
    } catch {
        audio.controls = true;
        audio.dataset.hazePlayerState = 'play-blocked';
        setPlayerStatus(feedId, 'Press Play to start audio');
        setPlayerButtons(feedId, false);
        const stopButton = findFeedElement('feed-stop', feedId);
        if (stopButton) stopButton.disabled = false;
    }
}

async function startFeedWebRTC(feedId) {
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
    const audio = publicWebRTCAudioElement(feedId);
    const volume = findFeedElement('feed-volume', feedId);
    if (!audio) {
        return;
    }
    setPlayerStatus(feedId, 'Connecting...');
    setPlayerButtons(feedId, true);

    const pc = new RTCPeerConnection();
    const fallbackStream = new MediaStream();
    const player = {
        mode: 'webrtc',
        pc,
        audio,
        fallbackStream,
        remoteStream: fallbackStream,
        audioReceiver: null,
        audioTrackId: '',
        trackAttached: false,
        connected: false,
        mediaRecent: null,
        requestedCodec: 'auto',
        connectionStateTimer: null,
        disconnectReconnectTimer: null,
        mediaEventTimer: null,
        trackMuteTimer: null,
        trackMuteSignalCount: 0,
        trackUnmuteSignalCount: 0,
        lastTrackMuteSignalAt: 0,
        lastTrackUnmuteSignalAt: 0,
        statsPollTimer: null,
        playbackWatchdogTimer: null,
        lastStats: null,
        startedAt: Date.now(),
        lastPacketAt: 0,
        lastAudioProgressAt: 0,
        stagnantStatsPolls: 0,
        missingStatsPolls: 0,
        reconnectTimer: null,
        reconnectAttempts: Math.max(0, Number(feedReconnectBackoffs.get(feedId) || 0)),
        reconnectPending: false,
        codecFallbackApplied: feedCodecFallbacks.has(feedId),
        trackMutedReported: false,
        stopping: false,
    };
    feedPlayers.set(feedId, player);
    startWebRTCPlaybackWatchdog(feedId, player);

    audio.volume = Number(volume?.value ?? 1);
    audio.autoplay = true;
    audio.controls = false;
    audio.muted = false;
    audio.playsInline = true;
    audio.dataset.hazePlayerState = 'connecting';
    audio.dataset.hazeTrackAttached = '0';
    audio.onplaying = () => {
        if (isActivePlayer(feedId, player)) {
            setHealthyWebRTCStatus(feedId, player, audio);
        }
    };
    audio.onwaiting = () => {
        if (isActivePlayer(feedId, player)) {
            if (hasRecentWebRTCPackets(player)) {
                ensureWebRTCAudioPlaying(feedId, player, audio);
                return;
            }
            scheduleWebRTCMediaEventStatus(feedId, player, 'waiting', 'Buffering...');
        }
    };
    audio.onstalled = () => {
        if (isActivePlayer(feedId, player)) {
            if (hasRecentWebRTCPackets(player)) {
                ensureWebRTCAudioPlaying(feedId, player, audio);
                return;
            }
            scheduleWebRTCMediaEventStatus(feedId, player, 'stalled', 'Audio stalled');
        }
    };
    audio.onerror = () => {
        if (isActivePlayer(feedId, player)) {
            audio.dataset.hazePlayerState = 'error';
            setPlayerStatus(feedId, 'Audio playback error');
        }
    };
    pc.addTransceiver('audio', { direction: 'recvonly' });
    pc.addEventListener('track', (event) => {
        const currentAudio = bindPlayerAudio(feedId, fallbackStream);
        if (!currentAudio) {
            setPlayerStatus(feedId, 'Audio element unavailable');
            return;
        }
        player.audioReceiver = event.receiver || null;
        player.audioTrackId = event.track.id || '';
        player.lastStats = null;
        player.stagnantStatsPolls = 0;
        player.missingStatsPolls = 0;
        player.trackAttached = true;
        currentAudio.dataset.hazeTrackAttached = '1';
        currentAudio.dataset.hazeTrackState = event.track.readyState || '';
        event.track.onmute = () => {
            if (isActivePlayer(feedId, player)) {
                if (hasRecentWebRTCPackets(player) || !hasHardStaleWebRTCPackets(player)) {
                    ensureWebRTCAudioPlaying(feedId, player, currentAudio);
                    return;
                }
                player.trackMuteSignalCount = Number(player.trackMuteSignalCount || 0) + 1;
                player.lastTrackMuteSignalAt = Date.now();
                scheduleWebRTCTrackMuteReport(feedId, player);
            }
        };
        event.track.onunmute = () => {
            if (isActivePlayer(feedId, player)) {
                if (!player.trackMuteTimer && !player.trackMutedReported) {
                    return;
                }
                player.trackUnmuteSignalCount = Number(player.trackUnmuteSignalCount || 0) + 1;
                player.lastTrackUnmuteSignalAt = Date.now();
                clearWebRTCTrackMuteReport(feedId, player);
            }
        };
        event.track.onended = () => {
            if (isActivePlayer(feedId, player)) {
                clearPlayerTimer(player, 'trackMuteTimer');
                player.trackMutedReported = false;
                recordWebRTCEvent(feedId, 'track_ended', {
                    connection_state: pc.connectionState,
                    ice_state: pc.iceConnectionState,
                });
                currentAudio.dataset.hazeTrackState = 'ended';
                scheduleWebRTCReconnect(feedId, player, 'Reconnecting ended audio track...');
            }
        };
        const stream = event.streams?.[0] || fallbackStream;
        if (!stream.getTracks().some((track) => track.id === event.track.id)) {
            stream.addTrack(event.track);
        }
        player.remoteStream = stream;
        bindWebRTCAudioOutput(feedId, player, stream, currentAudio);
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
        recordWebRTCEvent(feedId, 'connection_state', {
            connection_state: pc.connectionState,
            ice_state: pc.iceConnectionState,
            signaling_state: pc.signalingState || '',
            track_attached: Boolean(player.trackAttached),
            has_live_track: hasLiveWebRTCAudioTrack(player),
        });
        if (currentAudio) {
            currentAudio.dataset.hazeConnectionState = pc.connectionState;
            currentAudio.dataset.hazeIceState = pc.iceConnectionState;
        }
        if (pc.connectionState === 'connected') {
            player.connected = true;
            cancelWebRTCReconnect(player);
            ensureWebRTCAudioPlaying(feedId, player, currentAudio);
            clearPlayerTimer(player, 'connectionStateTimer');
            clearPlayerTimer(player, 'disconnectReconnectTimer');
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
            clearPlayerTimer(player, 'connectionStateTimer');
            player.connectionStateTimer = window.setTimeout(() => {
                if (isActivePlayer(feedId, player) && pc.connectionState === 'disconnected' && hasHardStaleWebRTCPackets(player)) {
                    setPlayerStatus(feedId, 'Reconnecting...');
                }
            }, WEBRTC_TRANSIENT_STATUS_DELAY_MS);
            scheduleWebRTCDisconnectReconnect(feedId, player, pc);
            setPlayerButtons(feedId, true);
        } else if (pc.connectionState === 'failed' || pc.connectionState === 'closed') {
            scheduleWebRTCReconnect(feedId, player, 'Reconnecting audio...');
        } else {
            setPlayerStatus(feedId, pc.connectionState);
        }
    });

    try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        await waitForIceGathering(pc);
        const local = pc.localDescription;
        const codec = selectedFeedCodec(feedId);
        player.requestedCodec = codec || 'auto';
        const payload = {
            feed_id: feedId,
            sdp: local.sdp,
            sdp_type: local.type,
        };
        if (codec && codec !== 'auto') {
            payload.codec = codec;
        }
        const answer = await publicClient.request('webrtc_offer', {
            ...payload,
        }, 15000);
        player.mediaRecent = answer.media_recent !== false;
        player.negotiatedCodec = answer.codec || '';
        player.negotiatedPayloadType = answer.payload_type ?? null;
        recordWebRTCEvent(feedId, 'answer_received', {
            codec: player.negotiatedCodec,
            payload_type: player.negotiatedPayloadType,
            media_recent: player.mediaRecent,
        });
        await pc.setRemoteDescription({
            type: answer.sdp_type || 'answer',
            sdp: answer.sdp,
        });
        startWebRTCStatsMonitor(feedId, player);
        window.setTimeout(() => {
            const active = feedPlayers.get(feedId);
            if (active === player && !active.trackAttached) {
                setPlayerStatus(feedId, 'Connected, no audio track yet');
            }
        }, 5000);
        setPlayerStatus(feedId, player.mediaRecent ? 'Waiting for audio...' : 'Connected');
    } catch (error) {
        stopFeed(feedId, { silent: true });
        setPlayerStatus(feedId, error.message || 'Could not start stream');
    }
}

async function copyHTTPLink(feedId) {
    feedId = String(feedId || '');
    if (!feedId) return;
    const link = listenPageURL(feedId, true);
    window.hazeLastShareLink = link;
    try {
        await copyTextToClipboard(link);
        setPlayerStatus(feedId, 'Listen link copied');
    } catch {
        setPlayerStatus(feedId, `Copy blocked: ${link}`);
    }
}

async function copyTextToClipboard(text) {
    if (navigator.clipboard?.writeText) {
        try {
            await navigator.clipboard.writeText(text);
            return;
        } catch {
            // Fall through to the selection-based path for HTTP/LAN browsers.
        }
    }
    const input = document.createElement('textarea');
    input.value = text;
    input.setAttribute('readonly', '');
    input.style.position = 'fixed';
    input.style.top = '0';
    input.style.left = '0';
    input.style.width = '1px';
    input.style.height = '1px';
    input.style.opacity = '0';
    document.body.appendChild(input);
    input.focus();
    input.select();
    input.setSelectionRange(0, input.value.length);
    const copied = document.execCommand?.('copy') === true;
    input.remove();
    if (!copied) {
        throw new Error('copy command rejected');
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
    if (!silent) {
        feedReconnectBackoffs.delete(feedId);
    }
    const player = feedPlayers.get(feedId);
    if (player) {
        player.stopping = true;
        window.hazeLastStop = {
            feed_id: feedId,
            silent,
            mode: player.mode || 'webrtc',
            connection_state: player.pc?.connectionState || '',
            ice_state: player.pc?.iceConnectionState || '',
            signaling_state: player.pc?.signalingState || '',
            track_attached: Boolean(player.trackAttached),
            at: new Date().toISOString(),
        };
        player.pc?.close();
        clearPlayerTimer(player, 'connectionStateTimer');
        clearPlayerTimer(player, 'disconnectReconnectTimer');
        clearPlayerTimer(player, 'mediaEventTimer');
        clearPlayerTimer(player, 'trackMuteTimer');
        clearPlayerTimer(player, 'reconnectTimer');
        clearPlayerInterval(player, 'playbackWatchdogTimer');
        player.reconnectPending = false;
        player.trackMutedReported = false;
        stopWebRTCStatsMonitor(player);
        closeWebRTCAudioOutput(player);
        const audio = player.mode === 'webrtc'
            ? (player.audio || publicWebRTCAudioElement(feedId))
            : (findFeedElement('feed-audio', feedId) || player.audio);
        if (audio) {
            audio.pause();
            if (player.mode === 'http') {
                audio.removeAttribute('src');
                audio.load();
            } else {
                audio.srcObject = null;
            }
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

function bindPlayerAudio(feedId, fallbackStream = null) {
    const player = feedPlayers.get(feedId);
    const audio = player?.mode === 'webrtc'
        ? publicWebRTCAudioElement(feedId)
        : (findFeedElement('feed-audio', feedId) || player?.audio);
    if (!player || !audio) return null;
    const volume = findFeedElement('feed-volume', feedId);
    audio.volume = Number(volume?.value ?? audio.volume ?? 1);
    const fallbackTrackCount = fallbackStream?.getAudioTracks?.().length || 0;
    const currentTrackCount = audio.srcObject?.getAudioTracks?.().length || 0;
    if (fallbackStream && fallbackTrackCount > 0 && audio.srcObject !== fallbackStream) {
        recordWebRTCEvent(feedId, 'audio_stream_bound', {
            replaced_existing_stream: Boolean(audio.srcObject),
            track_count: fallbackStream.getTracks?.().length || 0,
            current_track_count: currentTrackCount,
        });
        bindWebRTCAudioOutput(feedId, player, fallbackStream, audio);
    }
    audio.dataset.hazeStreamAttached = audio.srcObject ? '1' : '0';
    player.audio = audio;
    return audio;
}

function reattachActivePlayers() {
    for (const [feedId, player] of feedPlayers.entries()) {
        if (player.mode === 'webrtc') {
            bindPlayerAudio(feedId, player.remoteStream || player.audio?.srcObject || null);
            setPlayerButtons(feedId, true);
            setPlayerStatus(feedId, player.trackAttached ? 'Audio connected' : 'Connecting...');
            continue;
        }
        const audio = findFeedElement('feed-audio', feedId);
        if (!audio || audio === player.audio) {
            continue;
        }
        audio.volume = player.audio?.volume ?? 1;
        if (player.mode === 'http') {
            audio.srcObject = null;
            audio.src = player.httpURL || httpStreamURL(feedId);
            audio.dataset.hazePlayerState = player.audio?.dataset?.hazePlayerState || 'reattached-http';
            player.audio = audio;
            setPlayerButtons(feedId, true);
            setPlayerStatus(feedId, 'HTTP audio connected');
            audio.play().catch(() => {});
            continue;
        }
        bindWebRTCAudioOutput(feedId, player, player.remoteStream || player.audio?.srcObject || null, audio);
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

function publicAlertMetaItems(record) {
    const items = [
        ['Event', record.event || 'unknown'],
        ['Feed', record.feed_id || 'none'],
        ['Status', record.status || record.bucket || 'unknown'],
        ['Message', record.message_type || 'unknown'],
        ['Severity', record.severity || 'unknown'],
        ['Urgency', record.urgency || 'unknown'],
        ['Certainty', record.certainty || 'unknown'],
        ['Sent', formatDateTime(record.sent)],
        ['Expires', formatDateTime(record.expires)],
    ];
    if (record.reason) items.splice(3, 0, ['Reason', record.reason]);
    if (record.audio_url) items.push(['Audio', record.audio_mime_type || 'available']);
    return items;
}

function groupedPublicAccepted(records) {
    const groups = new Map();
    for (const record of records) {
        const feedID = record.feed_id || 'unassigned';
        if (!groups.has(feedID)) groups.set(feedID, []);
        groups.get(feedID).push(record);
    }
    return [...groups.entries()];
}

function publicAlertAreas(record) {
    if (Array.isArray(record.areas) && record.areas.length) return record.areas.join('; ');
    return record.area_text || record.sender || 'No area text available';
}

function publicAlertCard(record) {
    const id = record.id || '';
    const feedID = record.feed_id || '';
    const headline = record.headline || record.event || 'Weather Alert';
    const identifier = record.cap_xml_url
        ? `<a class="alert-card-id" href="${escapeHtml(record.cap_xml_url)}" target="_blank" rel="noopener noreferrer" title="Open CAP XML">${escapeHtml(id)}</a>`
        : `<span class="alert-card-id">${escapeHtml(id)}</span>`;
    const meta = publicAlertMetaItems(record).map(([key, value]) => `
        <span><b>${escapeHtml(key)}</b>${escapeHtml(value)}</span>
    `).join('');
    const capAudioLink = record.audio_url ? `
        <a class="btn-action btn-link" href="${escapeHtml(record.audio_url)}" target="_blank" rel="noopener noreferrer">
            <i data-lucide="circle-play" width="13" height="13"></i>
            CAP Audio
        </a>` : '';
    const capXMLLink = record.cap_xml_url ? `
        <a class="btn-action btn-link" href="${escapeHtml(record.cap_xml_url)}" target="_blank" rel="noopener noreferrer">
            <i data-lucide="file-code-2" width="13" height="13"></i>
            CAP XML
        </a>` : '';
    const actions = [capAudioLink, capXMLLink].filter(Boolean).join('');
    return `
        <article class="alert-card public-alert-card" data-alert-id="${escapeHtml(id)}" data-feed-id="${escapeHtml(feedID)}">
            <div class="alert-card-main">
                <div class="alert-card-head">
                    <div>
                        <h3><span>${escapeHtml(headline)}</span>${identifier}</h3>
                        <p>${escapeHtml(publicAlertAreas(record))}</p>
                    </div>
                    <span class="alert-card-time">${escapeHtml(formatDateTime(record.updated_at || record.sent))}</span>
                </div>
                <div class="alert-card-meta">${meta}</div>
                <details class="alert-details">
                    <summary>Details</summary>
                    <div class="alert-details-grid">
                        <section>
                            <h4>Description</h4>
                            <p>${escapeHtml(record.description || record.message || 'No description provided.')}</p>
                        </section>
                        <section>
                            <h4>Instruction</h4>
                            <p>${escapeHtml(record.instruction || 'No instruction provided.')}</p>
                        </section>
                    </div>
                </details>
            </div>
            <div class="alert-card-actions public-alert-card-actions">
                ${actions || '<span class="public-alert-action-note">Public details only</span>'}
            </div>
        </article>
    `;
}

function renderPublicAccepted(records) {
    const groups = groupedPublicAccepted(records);
    if (!groups.length) return '<article class="alert-empty">No accepted alerts are active for any feed.</article>';
    return groups.map(([feedID, items]) => `
        <section class="alert-feed-group">
            <div class="alert-feed-group-hd">
                <strong>${escapeHtml(feedID)}</strong>
                <span>${items.length} active</span>
            </div>
            ${items.map(publicAlertCard).join('')}
        </section>
    `).join('');
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
    alertsList.innerHTML = activeAlertTab === 'accepted'
        ? renderPublicAccepted(records)
        : (records.length
            ? records.map(publicAlertCard).join('')
            : `<article class="alert-empty">No ${escapeHtml(activeAlertTab)} alerts are archived.</article>`);
    window.lucide?.createIcons();
}

function requestedListenFeedID(feeds) {
    const params = new URLSearchParams(window.location.search);
    const requested = params.get('feed') || params.get('feed_id') || '';
    if (requested && feeds.some((feed) => String(feed.id || '') === requested)) {
        return requested;
    }
    return String(feeds.find(feedCanHTTP)?.id || feeds[0]?.id || '');
}

function requestedListenCodec() {
    const params = new URLSearchParams(window.location.search);
    return normalizeHTTPCodec(params.get('codec') || params.get('format') || 'pcm16');
}

function renderListen(feeds) {
    if (!listenNotice || !listenPanel) return;
    if (summaryState?.feeds_access === 'disabled') {
        listenNotice.textContent = 'Public feeds are disabled on this system.';
        listenPanel.innerHTML = '';
        return;
    }
    const feedID = requestedListenFeedID(feeds || []);
    const feed = (feeds || []).find((item) => String(item.id || '') === feedID);
    if (!feed) {
        listenNotice.textContent = 'No public feed was found for this link.';
        listenPanel.innerHTML = '';
        return;
    }
    if (!feedCanHTTP(feed)) {
        listenNotice.textContent = 'This feed does not have public HTTP audio enabled.';
        listenPanel.innerHTML = '';
        lastListenSignature = 'unavailable';
        return;
    }
    const codec = requestedListenCodec();
    const listenSignature = JSON.stringify({ feedID, codec, media: Boolean(summaryState?.media_available) });
    if (listenSignature === lastListenSignature) {
        return;
    }
    lastListenSignature = listenSignature;
    const tx = feed.transmitter || {};
    const siteNames = (tx.site_names || [tx.site_name]).filter(Boolean).join(', ') || feed.name || 'Unnamed site';
    const nowPlaying = feed.runtime?.now_playing || 'Idle';
    const streamURL = httpStreamURL(feedID, false, codec);
    listenNotice.textContent = 'Public HTTP listener ready.';
    listenPanel.innerHTML = `
        <article class="feed-card public-listen-card">
            <div class="public-listen-title">
                <p class="feed-id">${escapeHtml(feedID)}</p>
                <h3>${escapeHtml(siteNames)}</h3>
                <p class="public-feed-now">${escapeHtml(nowPlaying)}</p>
            </div>
            <div class="public-listen-toolbar">
                <label class="public-listen-field">
                    <span>Format</span>
                    <select data-listen-codec>
                        ${HTTP_CODECS.map(([value, label]) => `
                            <option value="${value}" ${codec === value ? 'selected' : ''}>${label}</option>
                        `).join('')}
                    </select>
                </label>
                <a class="btn-action public-player-btn" href="${escapeHtml(streamURL)}" data-listen-raw>
                    <i data-lucide="link" width="14" height="14"></i>
                    <span>Raw</span>
                </a>
            </div>
            <audio class="public-listen-audio" src="${escapeHtml(streamURL)}" controls autoplay playsinline preload="none"></audio>
        </article>
    `;
    const select = listenPanel.querySelector('[data-listen-codec]');
    select?.addEventListener('change', () => {
        const nextCodec = normalizeHTTPCodec(select.value);
        const nextURL = listenPageURL(feedID, false, nextCodec);
        window.location.assign(nextURL);
    });
    window.lucide?.createIcons();
}

function renderPublicState(payload) {
    const summary = payload.summary || {};
    renderSummary(summary);
    if (currentPage === 'alerts') {
        renderPublicAlerts(summary);
        return;
    }
    if (isListenPage) {
        renderListen(summary.feeds || []);
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
    } else if (currentPage === 'alerts') {
        alertsNotice.textContent = `Alert archive unavailable. Reconnecting in ${seconds}s...`;
    }
});
publicClient.addEventListener('recovered', () => {
    if (currentPage !== 'feeds') return;
    reattachActivePlayers();
    for (const [feedId, player] of feedPlayers.entries()) {
        if (player.mode === 'webrtc' && player.trackAttached) {
            setHealthyWebRTCStatus(feedId, player);
        }
    }
});
publicClient.addEventListener('close', (event) => {
    if (!event.detail?.reconnecting) return;
    if (currentPage === 'feeds') {
        setNotice('Live feed directory connection closed. Reconnecting...');
    } else if (currentPage === 'alerts') {
        alertsNotice.textContent = 'Alert archive connection closed. Reconnecting...';
    }
});
publicClient.addEventListener('decode_error', () => {
    if (currentPage === 'feeds' && feedNotice) {
        feedNotice.textContent = 'Unable to decode public status update.';
    }
});
publicClient.addEventListener('error', () => {
    if (currentPage === 'feeds') {
        feedNotice.textContent = 'Live feed directory unavailable. Reconnecting...';
    } else if (currentPage === 'alerts') {
        alertsNotice.textContent = 'Alert archive unavailable. Reconnecting...';
    }
});

connectPublicSocket();

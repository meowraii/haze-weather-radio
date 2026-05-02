import { token, API_BASE } from './lib/api.js';
import { initTheme } from './lib/theme.js';
import { pcmToWav } from './lib/audio.js';


const authOverlay = document.getElementById('authOverlay');
const authStatus = document.getElementById('authStatus');
const authPill = document.getElementById('authPill');
const healthPill = document.getElementById('healthPill');
const apiDot = document.getElementById('apiDot');
const authDot = document.getElementById('authDot');
const heroTitle = document.getElementById('heroTitle');
const heroSubtitle = document.getElementById('heroSubtitle');
const summaryCards = document.getElementById('summaryCards');
const feedsGrid = document.getElementById('feedsGrid');
const sidebarFeeds = document.getElementById('sidebarFeeds');
const eventsList = document.getElementById('eventsList');
const logsView = document.getElementById('logsView');
const datapoolView = document.getElementById('datapoolView');
const configView = document.getElementById('configView');
const logSourceSelect = document.getElementById('logSourceSelect');
const refreshButton = document.getElementById('refreshButton');
const logoutButton = document.getElementById('logoutButton');
const themeToggle = document.getElementById('themeToggle');
const loginForm = document.getElementById('loginForm');
const passwordInput = document.getElementById('passwordInput');

let healthState = { auth_required: true };

initTheme(themeToggle);
window.lucide?.createIcons();

function setCodeBlock(element, value) {
	element.textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function setAuthState(authenticated) {
	authPill.textContent = authenticated ? 'Authenticated' : 'Required';
	authDot.dataset.state = authenticated ? 'ok' : 'warn';
	authOverlay.classList.toggle('hidden', authenticated || healthState.auth_required === false);
}

function setHealthState(ok, text) {
	healthPill.textContent = text;
	apiDot.dataset.state = ok ? 'ok' : 'err';
}

async function api(path, options = {}) {
	const headers = new Headers(options.headers || {});
	const t = token.get();
	if (t) {
		headers.set('Authorization', `Bearer ${t}`);
	}
	if (!headers.has('Content-Type') && options.body) {
		headers.set('Content-Type', 'application/json');
	}

	const response = await fetch(`${API_BASE}${path}`, { ...options, headers });
	if (response.status === 401) {
		token.clear();
		setAuthState(false);
		throw new Error('Authentication required');
	}
	if (!response.ok) {
		throw new Error(`Request failed with status ${response.status}`);
	}
	return response.json();
}

async function loadHealth() {
	try {
		healthState = await fetch(`${API_BASE}/health`).then((response) => response.json());
		setHealthState(true, `API healthy • ${Math.round(healthState.uptime_seconds || 0)}s uptime`);
		if (healthState.auth_required === false) {
			setAuthState(true);
			authStatus.textContent = 'Authentication disabled. Panel is in read-only open mode.';
		} else {
			setAuthState(Boolean(token.get()));
			authStatus.textContent = token.get() ? 'Stored session found. Loading dashboard.' : 'Authentication is required for this panel.';
		}
	} catch (error) {
		setHealthState(false, 'API unavailable');
		authStatus.textContent = 'Unable to reach the panel API.';
		setAuthState(false);
		throw error;
	}
}

function formatUptime(seconds) {
	const s = Math.round(seconds || 0);
	if (s < 60) return `${s}s`;
	if (s < 3600) return `${Math.floor(s / 60)}m`;
	return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function renderSummary(summary) {
	const enabled = summary.enabled_feed_count;
	const total = summary.feed_count;
	heroTitle.textContent = `${enabled} of ${total} feed${total !== 1 ? 's' : ''} online`;
	heroSubtitle.textContent = `${summary.data_pool_key_count} data pool keys · uptime ${formatUptime(summary.uptime_seconds)}`;

	const totalQueue = (summary.feeds || []).reduce((n, f) => n + (f.alert_queue_depth || 0), 0);

	const cards = [
		['Feeds online', `${enabled}/${total}`],
		['Alert queue', totalQueue],
		['Weather keys', summary.data_pool_key_count],
		['Uptime', formatUptime(summary.uptime_seconds)],
	];

	summaryCards.innerHTML = cards.map(([label, value]) => `
		<article class="metric-card">
			<p>${label}</p>
			<strong>${value}</strong>
		</article>
	`).join('');
}

function renderSidebarFeeds(feeds) {
	if (!feeds.length) {
		sidebarFeeds.innerHTML = '<p class="sidebar-empty">No feeds configured.</p>';
		return;
	}

	sidebarFeeds.innerHTML = feeds.map((feed) => {
		const runtime = feed.runtime || {};
		const queueDepth = feed.alert_queue_depth || 0;
		const playing = runtime.now_playing || 'Idle';
		return `
			<div class="sidebar-feed">
				<div class="sidebar-feed-head">
					<span class="status-dot" data-state="ok"></span>
					<span class="sidebar-feed-name">${feed.name || feed.id}</span>
					<span class="sidebar-feed-q" data-active="${queueDepth > 0}">${queueDepth}</span>
				</div>
				<div class="sidebar-feed-playing">${playing}</div>
			</div>
		`;
	}).join('');
}

function populateWxFeedSelect(feeds) {
	const sel = document.getElementById('wxFeedSelect');
	if (!sel) return;
	const prev = sel.value;
	sel.innerHTML = feeds.map((f) =>
		`<option value="${f.id}">${f.name || f.id}</option>`
	).join('');
	if (prev && feeds.some((f) => f.id === prev)) sel.value = prev;
}

function renderFeeds(feeds) {
	populateWxFeedSelect(feeds);
	renderSidebarFeeds(feeds);

	if (!feeds.length) {
		feedsGrid.innerHTML = '<article class="feed-card empty">No feeds configured.</article>';
		return;
	}

	feedsGrid.innerHTML = feeds.map((feed) => {
		const runtime = feed.runtime || {};
		const playlist = (feed.playlist_items || []).slice(0, 6).map((item) => `<li>${item}</li>`).join('');
		const outputs = (feed.outputs || []).map((item) => `<span class="tag">${item}</span>`).join('');
		const latestAlert = runtime.last_alert_event
			? `${runtime.last_alert_event} • ${runtime.last_alert_severity || 'n/a'}`
			: 'No queued alert activity yet';

		return `
			<article class="feed-card">
				<div class="feed-topline">
					<div>
						<p class="feed-id">${feed.id}</p>
						<h3>${feed.name}</h3>
					</div>
					<span class="queue-chip" data-active="${feed.alert_queue_depth > 0}">Queue ${feed.alert_queue_depth}</span>
				</div>
				<div class="feed-meta">
					<span class="tag">${feed.timezone}</span>
					<span class="tag">${feed.location_count} locations</span>
					<span class="tag">${feed.languages.join(', ')}</span>
				</div>
				<div class="tag-row">${outputs || '<span class="tag muted">No outputs</span>'}</div>
				<dl class="feed-stats">
					<div>
						<dt>Now Playing</dt>
						<dd>${runtime.now_playing || 'Idle'}</dd>
					</div>
					<div>
						<dt>Latest Alert</dt>
						<dd>${latestAlert}</dd>
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
				<span class="event-kind">${event.kind}</span>
				<time>${event.timestamp}</time>
			</div>
			<p>${event.message}</p>
			${event.feed_id ? `<span class="event-feed">${event.feed_id}</span>` : ''}
		</article>
	`).join('');
}

async function refreshDashboard() {
	const [summary, datapool, config, logs, events] = await Promise.all([
		api('/summary'),
		api('/datapool'),
		api('/config'),
		api(`/logs?source=${encodeURIComponent(logSourceSelect.value)}&lines=120`),
		api('/events'),
	]);

	renderSummary(summary);
	renderFeeds(summary.feeds || []);
	renderEvents(events || []);
	setCodeBlock(logsView, (logs.lines || []).join('\n'));
	setCodeBlock(datapoolView, datapool);
	setCodeBlock(configView, config);
}

(async function initWxOnDemand() {
	const pkgChips = document.getElementById('pkgChips');
	const wxFeedSelect = document.getElementById('wxFeedSelect');
	const wxLangSelect = document.getElementById('wxLangSelect');
	const wxVoiceSelect = document.getElementById('wxVoiceSelect');
	const wxGenerateBtn = document.getElementById('wxGenerateBtn');
	const wxStopBtn = document.getElementById('wxStopBtn');
	const wxStatus = document.getElementById('wxStatus');
	const wxAudio = document.getElementById('wxAudio');
	const wxCodecSelect = document.getElementById('wxCodecSelect');

	if (!pkgChips || !wxGenerateBtn) return;

	let activeObjectUrl = null;
	let activeAbort = null;

	const selectedPkgs = new Set(['date_time', 'current_conditions', 'forecast']);

	try {
		const { packages } = await fetch(`${API_BASE}/wx/packages`).then((r) => r.json());
		pkgChips.innerHTML = packages.map((pkg) => {
			const active = selectedPkgs.has(pkg) ? ' active' : '';
			return `<span class="pkg-chip${active}" data-pkg="${pkg}">${pkg.replace(/_/g, ' ')}</span>`;
		}).join('');
	} catch {
		pkgChips.innerHTML = '<span style="font-size:12px;color:var(--muted)">Could not load packages.</span>';
	}

	pkgChips.addEventListener('click', (e) => {
		const chip = e.target.closest('.pkg-chip');
		if (!chip) return;
		const pkg = chip.dataset.pkg;
		if (pkg === 'all') {
			const allActive = !chip.classList.contains('active');
			if (allActive) selectedPkgs.add('all');
			else selectedPkgs.delete('all');
		} else {
			if (selectedPkgs.has(pkg)) selectedPkgs.delete(pkg);
			else selectedPkgs.add(pkg);
		}
		pkgChips.querySelectorAll('.pkg-chip').forEach((c) => {
			c.classList.toggle('active', selectedPkgs.has(c.dataset.pkg));
		});
	});

	wxGenerateBtn.addEventListener('click', async () => {
		const feedId = wxFeedSelect.value;
		const lang = wxLangSelect.value;
		const voice = wxVoiceSelect.value || null;
		const codec = wxCodecSelect.value;
		const pkgs = [...selectedPkgs];

		if (!feedId) {
			wxStatus.textContent = 'Select a feed first.';
			wxStatus.className = 'wx-status err';
			return;
		}

		if (activeAbort) activeAbort.abort();
		if (activeObjectUrl) { URL.revokeObjectURL(activeObjectUrl); activeObjectUrl = null; }

		activeAbort = new AbortController();
		wxGenerateBtn.disabled = true;
		wxStopBtn.disabled = false;
		wxStatus.textContent = 'Generating…';
		wxStatus.className = 'wx-status';
		wxAudio.style.display = 'none';

		try {
			const headers = new Headers({ 'Content-Type': 'application/json' });
			const t = token.get();
			if (t) headers.set('Authorization', `Bearer ${t}`);

			const resp = await fetch(`${API_BASE}/wx/generate`, {
				method: 'POST',
				headers,
				body: JSON.stringify({ feed_id: feedId, packages: pkgs, lang, voice, codec }),
				signal: activeAbort.signal,
			});

			if (!resp.ok) throw new Error(`Server error ${resp.status}`);

			if (codec !== 'raw') {
				wxStatus.textContent = 'Transcoding…';
				const blob = await resp.blob();
				activeObjectUrl = URL.createObjectURL(blob);
				wxAudio.src = activeObjectUrl;
				wxAudio.style.display = 'block';
				wxAudio.play();
				wxStatus.textContent = `Ready · ${(blob.size / 1024).toFixed(1)} KB`;
				wxStatus.className = 'wx-status ok';
				return;
			}

			const sampleRate = parseInt(resp.headers.get('X-Audio-Sample-Rate') || '16000', 10);
			const channels = parseInt(resp.headers.get('X-Audio-Channels') || '1', 10);

			const chunks = [];
			const reader = resp.body.getReader();
			let received = 0;

			while (true) {
				const { done, value } = await reader.read();
				if (done) break;
				chunks.push(value);
				received += value.byteLength;
				wxStatus.textContent = `Buffering… ${(received / 1024).toFixed(0)} KB`;
			}

			const pcm = new Uint8Array(received);
			let offset = 0;
			for (const chunk of chunks) { pcm.set(chunk, offset); offset += chunk.byteLength; }

			const wav = pcmToWav(pcm, sampleRate, channels);
			activeObjectUrl = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }));
			wxAudio.src = activeObjectUrl;
			wxAudio.style.display = 'block';
			wxAudio.play();

			wxStatus.textContent = `Ready · ${(received / 1024).toFixed(1)} KB`;
			wxStatus.className = 'wx-status ok';
		} catch (err) {
			if (err.name !== 'AbortError') {
				console.error(err);
				wxStatus.textContent = err.message || 'Generation failed.';
				wxStatus.className = 'wx-status err';
			} else {
				wxStatus.textContent = 'Stopped.';
				wxStatus.className = 'wx-status';
			}
		} finally {
			wxGenerateBtn.disabled = false;
			wxStopBtn.disabled = true;
			activeAbort = null;
		}
	});

	wxStopBtn.addEventListener('click', () => {
		if (activeAbort) activeAbort.abort();
		wxAudio.pause();
	});
})();


async function login(password) {
	const response = await fetch(`${API_BASE}/auth/login`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ password }),
	});

	if (!response.ok) {
		throw new Error('Login failed');
	}

	const payload = await response.json();
	const t = payload.token || '';
	if (t) token.set(t);
	setAuthState(true);
}

async function boot() {
	try {
		await loadHealth();
		if (healthState.auth_required === false || token.get()) {
			await refreshDashboard();
		}
	} catch (error) {
		heroTitle.textContent = 'Panel unavailable';
		heroSubtitle.textContent = 'The web API did not respond cleanly.';
	}
}

loginForm.addEventListener('submit', async (event) => {
	event.preventDefault();
	authStatus.textContent = 'Authenticating...';
	try {
		await login(passwordInput.value);
		authStatus.textContent = 'Authenticated. Loading dashboard.';
		passwordInput.value = '';
		await refreshDashboard();
	} catch (error) {
		authStatus.textContent = 'Incorrect password or unavailable API.';
		setAuthState(false);
	}
});

refreshButton.addEventListener('click', async () => {
	try {
		await refreshDashboard();
	} catch (error) {
		setHealthState(false, 'Refresh failed');
	}
});

logoutButton.addEventListener('click', () => {
	token.clear();
	setAuthState(false);
	authStatus.textContent = 'Logged out.';
});

const rwtButton = document.getElementById('rwtButton');

rwtButton.addEventListener('click', async () => {
	rwtButton.disabled = true;
	const prev = rwtButton.innerHTML;
	rwtButton.textContent = 'Sending…';
	try {
		await api('/same/test?event_code=RWT', { method: 'POST' });
		rwtButton.textContent = 'Sent ✓';
		setTimeout(() => { rwtButton.innerHTML = prev; rwtButton.disabled = false; }, 3000);
	} catch (error) {
		rwtButton.textContent = 'Failed';
		setTimeout(() => { rwtButton.innerHTML = prev; rwtButton.disabled = false; }, 3000);
	}
});


logSourceSelect.addEventListener('change', async () => {
	try {
		const logs = await api(`/logs?source=${encodeURIComponent(logSourceSelect.value)}&lines=120`);
		setCodeBlock(logsView, (logs.lines || []).join('\n'));
	} catch (error) {
		setCodeBlock(logsView, 'Unable to load log source.');
	}
});

boot().then(() => {
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
});

setInterval(async () => {
	try {
		await loadHealth();
		if (healthState.auth_required === false || token.get()) {
			await refreshDashboard();
		}
	} catch (error) {
		setHealthState(false, 'Background refresh failed');
	}
}, 15000);

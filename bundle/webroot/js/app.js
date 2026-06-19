import { session, token } from './lib/api.js';
import { initTheme } from './lib/theme.js';
import { createControlClient } from './lib/ws-client.js';
import { initDashboard } from './dashboard.js';
import { initSameView } from './same.js';
import { initWxView } from './wx.js';
import { initDaemonView } from './daemon.js';
import { initPlaylistView } from './playlist.js';
import { initBreakInView } from './breakin.js';
import { initAlertsArchiveView } from './alerts.js';
import { initAutomationsView } from './automations.js';
import { initDictionaryView } from './dictionary.js';

const authPill = document.getElementById('authPill');
const healthPill = document.getElementById('healthPill');
const apiDot = document.getElementById('apiDot');
const authDot = document.getElementById('authDot');
const themeToggle = document.getElementById('themeToggle');
const logoutButton = document.getElementById('logoutButton');
const refreshButton = document.getElementById('refreshButton');

let healthState = { auth_required: true };
let dashboardInitialized = false;
const viewInit = { same: false, automations: false, wx: false, daemon: false, playlist: false, breakin: false, alerts: false, dictionary: false };

session.importUrlToken();
initTheme(themeToggle);
window.lucide?.createIcons();

function setAuthState(authenticated) {
    authPill.textContent = authenticated ? 'Authenticated' : 'Required';
    authDot.dataset.state = authenticated ? 'ok' : 'warn';
}

function setHealthState(ok, text) {
    healthPill.textContent = text;
    apiDot.dataset.state = ok ? 'ok' : 'err';
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

function setLastConnected(lastConnected) {
    const el = document.getElementById('lastConnected');
    if (!el) return;
    const ip = lastConnected?.ip || 'unknown IP';
    el.textContent = `Last connected from ${ip} at ${formatDateTime(lastConnected?.at)}`;
}

function loginUrl() {
    return `/login?next=${encodeURIComponent(`${window.location.pathname}${window.location.hash || ''}`)}`;
}

function delay(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function readAuthState() {
    const client = createControlClient();
    try {
        healthState = await client.request('auth_check');
    } finally {
        client.close();
    }
    setHealthState(true, `API healthy · ${Math.round(healthState.uptime_seconds || 0)}s uptime`);
    setLastConnected(healthState.last_connected);
    setAuthState(Boolean(healthState.authenticated));
    return healthState;
}

async function loadHealth() {
    try {
        for (let attempt = 0; attempt < 3; attempt += 1) {
            const state = await readAuthState();
            if (!state.auth_required || state.authenticated) return true;
            await delay(250);
        }
        if (healthState.auth_required && !healthState.authenticated) {
            if (!token.get()) {
                token.clear();
                window.location.href = loginUrl();
                return false;
            }
            setHealthState(false, 'Session verification pending');
            setAuthState(false);
            return true;
        }
        return true;
    } catch {
        setHealthState(false, 'API unavailable');
        setAuthState(false);
        if (token.get()) {
            return true;
        }
        return false;
    }
}

function navigate(view) {
    document.querySelectorAll('.view').forEach((v) => {
        v.classList.toggle('active', v.dataset.view === view);
    });

    document.querySelectorAll('.sidebar-link').forEach((l) => {
        l.classList.toggle('active', l.dataset.view === view);
    });

    if (view === 'same' && !viewInit.same) {
        viewInit.same = true;
        initSameView();
    }
    if (view === 'automations' && !viewInit.automations) {
        viewInit.automations = true;
        initAutomationsView();
    }
    if (view === 'wx' && !viewInit.wx) {
        viewInit.wx = true;
        initWxView();
    }
    if (view === 'daemon' && !viewInit.daemon) {
        viewInit.daemon = true;
        initDaemonView();
    }
    if (view === 'playlist' && !viewInit.playlist) {
        viewInit.playlist = true;
        initPlaylistView();
    }
    if (view === 'breakin' && !viewInit.breakin) {
        viewInit.breakin = true;
        initBreakInView();
    }
    if (view === 'alerts' && !viewInit.alerts) {
        viewInit.alerts = true;
        initAlertsArchiveView();
    }
    if (view === 'dictionary' && !viewInit.dictionary) {
        viewInit.dictionary = true;
        initDictionaryView();
    }

    window.lucide?.createIcons();
}

function handleHash() {
    const hash = window.location.hash.replace('#', '') || '/';
    navigate(hash.replace(/^\//, '') || 'home');
}

function bootstrap() {
    if (!dashboardInitialized) {
        dashboardInitialized = true;
        initDashboard();
    }
    handleHash();
}

async function logout() {
    const client = createControlClient();
    try {
        return await client.request('logout', { token: token.get() }, 8000);
    } finally {
        client.close();
    }
}

logoutButton.addEventListener('click', async () => {
    let publicUrl = healthState.public_url || 'http://127.0.0.1:8086/';
    try {
        const payload = await logout();
        publicUrl = payload.public_url || publicUrl;
    } catch {
        // Local token cleanup and redirect still happen if the server is already gone.
    }
    token.clear();
    setAuthState(false);
    window.location.href = publicUrl;
});

window.addEventListener('hashchange', handleHash);

refreshButton.addEventListener('click', () => {
    import('./dashboard.js').then((m) => m.refreshDashboard({ force: true })).catch(() => {});
});

window.addEventListener('haze:admin-state', (event) => {
    const payload = event.detail || {};
    const summary = payload.summary || {};
    if (typeof summary.uptime_seconds === 'number') {
        setHealthState(true, `Live socket · ${Math.round(summary.uptime_seconds)}s uptime`);
    } else {
        setHealthState(true, 'Live socket connected');
    }
    setLastConnected(payload.last_connected);
});

loadHealth().then((ready) => {
    if (ready) bootstrap();
}).catch(() => {
    const heroTitle = document.getElementById('heroTitle');
    const heroSubtitle = document.getElementById('heroSubtitle');
    if (heroTitle) heroTitle.textContent = 'Panel unavailable';
    if (heroSubtitle) heroSubtitle.textContent = 'The web API did not respond cleanly.';
});

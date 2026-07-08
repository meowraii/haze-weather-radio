import { API_BASE } from './api.js';
import { session } from './session.js';

const DEFAULT_RECONNECT_MS = 800;
const MAX_RECONNECT_MS = 15000;
const RECONNECT_JITTER = 0.35;
const POLL_INTERVAL_MS = 5000;

export class AdminTransportClient extends EventTarget {
    constructor({ base = API_BASE, stream = true, params = {}, includeSessionToken = true } = {}) {
        super();
        this.base = base;
        this.stream = stream;
        this.params = params;
        this.includeSessionToken = includeSessionToken;
        this.source = null;
        this.connected = false;
        this.reconnectTimer = null;
        this.pollTimer = null;
        this.reconnectEnabled = false;
        this.reconnectAttempt = 0;
        this.hadOpen = false;
        this.manualClose = false;
        this.lastStateSignature = '';
        window.addEventListener('online', () => {
            if (this.reconnectEnabled && !this.connected) this.reconnectNow();
        });
    }

    url(extra = {}) {
        return this.eventsUrl(extra);
    }

    eventsUrl(extra = {}) {
        return this.endpointURL('/panel/events', extra).toString();
    }

    stateUrl(extra = {}) {
        return this.endpointURL('/panel/state', extra).toString();
    }

    endpointURL(path, extra = {}) {
        const url = new URL(`${this.base}${path}`, window.location.origin);
        const params = { ...this.params, ...extra };
        if (!this.stream) params.mode = params.mode || 'control';
        if (this.includeSessionToken && session.token) params.token = session.token;
        Object.entries(params).forEach(([key, value]) => {
            if (value !== undefined && value !== null && value !== '') {
                url.searchParams.set(key, String(value));
            }
        });
        return url;
    }

    connect() {
        if (!this.stream) return null;
        this.reconnectEnabled = true;
        this.manualClose = false;
        if (this.source) return this.source;
        this.dispatch('connecting', { attempt: this.reconnectAttempt });
        if (!('EventSource' in window)) {
            this.startPolling();
            return null;
        }
        const source = new EventSource(this.eventsUrl(), { withCredentials: true });
        this.source = source;

        source.addEventListener('open', () => {
            const recovered = this.hadOpen || this.reconnectAttempt > 0;
            this.connected = true;
            this.hadOpen = true;
            this.reconnectAttempt = 0;
            if (this.reconnectTimer) window.clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
            this.dispatch('open', { recovered });
            if (recovered) this.dispatch('recovered');
        });
        for (const type of ['admin_state', 'public_state', 'auth_state', 'heartbeat', 'hello']) {
            source.addEventListener(type, (event) => this.handleStreamEvent(type, event.data));
        }
        source.addEventListener('error', () => {
            if (this.source !== source) return;
            this.dispatch('error', { attempt: this.reconnectAttempt });
            this.source = null;
            this.connected = false;
            source.close();
            if (this.reconnectEnabled && !this.manualClose) this.scheduleReconnect();
        });
        return source;
    }

    close() {
        this.reconnectEnabled = false;
        this.manualClose = true;
        if (this.reconnectTimer) window.clearTimeout(this.reconnectTimer);
        if (this.pollTimer) window.clearTimeout(this.pollTimer);
        this.reconnectTimer = null;
        this.pollTimer = null;
        this.connected = false;
        if (this.source) {
            this.source.close();
            this.source = null;
        }
    }

    reconnectNow() {
        this.reconnectEnabled = true;
        this.manualClose = false;
        if (this.reconnectTimer) window.clearTimeout(this.reconnectTimer);
        if (this.pollTimer) window.clearTimeout(this.pollTimer);
        this.reconnectTimer = null;
        this.pollTimer = null;
        this.connected = false;
        if (this.source) {
            this.source.close();
            this.source = null;
        }
        this.connect();
    }

    scheduleReconnect(event = {}) {
        if (this.reconnectTimer || this.pollTimer) return;
        this.reconnectAttempt += 1;
        const exponential = DEFAULT_RECONNECT_MS * (2 ** Math.min(this.reconnectAttempt - 1, 5));
        const baseDelay = Math.min(MAX_RECONNECT_MS, exponential);
        const jitter = baseDelay * RECONNECT_JITTER * Math.random();
        const delay = Math.round(baseDelay + jitter);
        this.dispatch('reconnecting', {
            attempt: this.reconnectAttempt,
            delay,
            code: event.code,
            reason: event.reason,
        });
        this.reconnectTimer = window.setTimeout(() => {
            this.reconnectTimer = null;
            if (this.reconnectAttempt >= 3) {
                this.startPolling();
                return;
            }
            this.connect();
        }, delay);
    }

    startPolling() {
        if (this.pollTimer || !this.reconnectEnabled) return;
        const poll = async () => {
            try {
                const state = await this.fetchState();
                const recovered = this.hadOpen || this.reconnectAttempt > 0;
                if (!this.connected) {
                    this.connected = true;
                    this.hadOpen = true;
                    this.reconnectAttempt = 0;
                    this.dispatch('open', { recovered });
                    if (recovered) this.dispatch('recovered');
                }
                this.dispatchState(state);
                this.pollTimer = window.setTimeout(poll, POLL_INTERVAL_MS);
            } catch {
                this.connected = false;
                this.dispatch('error', { attempt: this.reconnectAttempt });
                this.pollTimer = null;
                if (this.reconnectEnabled && !this.manualClose) this.scheduleReconnect();
            }
        };
        this.pollTimer = window.setTimeout(poll, 0);
    }

    async fetchState(timeoutMs = 8000) {
        return this.fetchJSON(this.stateUrl(), {
            method: 'GET',
            headers: session.authHeaders(),
        }, timeoutMs);
    }

    dispatchState(state) {
        const signature = JSON.stringify(state || {});
        if (signature && signature === this.lastStateSignature) return;
        this.lastStateSignature = signature;
        this.dispatch(this.stateEventType(), state || {});
        this.dispatch('message', { type: this.stateEventType(), data: state || {} });
    }

    stateEventType() {
        return String(this.base || '').includes('/api/public/') ? 'public_state' : 'admin_state';
    }

    handleStreamEvent(type, raw) {
        let data = {};
        try {
            data = raw ? JSON.parse(raw) : {};
        } catch {
            this.dispatch('decode_error', raw);
            return;
        }
        if (type === 'admin_state' || type === 'public_state') {
            this.lastStateSignature = JSON.stringify(data || {});
        }
        this.dispatch(type, data);
        this.dispatch('message', { type, data });
    }

    send() {
        throw new Error('Panel streaming is one way. Use request() or command() for client actions.');
    }

    async request(type, payload = {}, timeoutMs = 8000) {
        if (type === 'auth_check') return this.fetchJSON(`${this.base}/auth/check`, {
            method: 'GET',
            headers: session.authHeaders(),
        }, timeoutMs);
        if (type === 'login') return this.fetchJSON(`${this.base}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password: payload.password || '' }),
        }, timeoutMs);
        if (type === 'logout') return this.fetchJSON(`${this.base}/auth/logout`, {
            method: 'POST',
            headers: session.authHeaders({ 'Content-Type': 'application/json', 'X-Haze-Admin-Intent': 'command' }),
            body: JSON.stringify({}),
        }, timeoutMs);
        if (type === 'webrtc_offer') {
            return this.fetchJSON(`${this.base}/feed/webrtc/offer`, {
                method: 'POST',
                headers: session.authHeaders({ 'Content-Type': 'application/json' }),
                body: JSON.stringify(payload || {}),
            }, timeoutMs);
        }
        if (type === 'command') {
            return this.postCommand(payload.command, payload.payload || {}, timeoutMs);
        }
        throw new Error(`Unsupported panel request: ${type}`);
    }

    async command(name, payload = {}, timeoutMs = 15000) {
        return this.postCommand(name, payload, timeoutMs);
    }

    async postCommand(command, payload = {}, timeoutMs = 15000) {
        const response = await this.fetchJSON(`${this.base}/panel/command`, {
            method: 'POST',
            headers: session.authHeaders({ 'Content-Type': 'application/json', 'X-Haze-Admin-Intent': 'command' }),
            body: JSON.stringify({ command, payload }),
        }, timeoutMs, { rawResponse: true });
        if (response.payload?.type === 'command_error') {
            throw new Error(response.payload.detail || 'Panel command failed.');
        }
        if (!response.ok) {
            throw new Error(response.payload?.detail || `Panel command failed with HTTP ${response.status}.`);
        }
        return response.payload?.result !== undefined ? response.payload.result : response.payload;
    }

    async fetchJSON(url, options = {}, timeoutMs = 8000, { rawResponse = false } = {}) {
        const controller = new AbortController();
        const timer = window.setTimeout(() => controller.abort(), timeoutMs);
        try {
            const response = await fetch(url, {
                credentials: 'same-origin',
                cache: 'no-store',
                ...options,
                signal: controller.signal,
            });
            let payload = {};
            try {
                payload = await response.json();
            } catch {
                payload = {};
            }
            if (rawResponse) return { ok: response.ok, status: response.status, payload };
            if (!response.ok) {
                throw new Error(payload.detail || `Panel request failed with HTTP ${response.status}.`);
            }
            if (payload.type === 'auth_error' || payload.type === 'error' || payload.type === 'webrtc_error') {
                throw new Error(payload.detail || 'Panel request failed.');
            }
            return payload;
        } catch (error) {
            if (error.name === 'AbortError') throw new Error('Panel command timed out.');
            throw error;
        } finally {
            window.clearTimeout(timer);
        }
    }

    dispatch(type, detail) {
        this.dispatchEvent(new CustomEvent(type, { detail }));
    }
}

export const PanelClient = AdminTransportClient;
export const panelClient = new AdminTransportClient({ stream: true, params: { source: 'app', lines: '120' } });

export function createControlClient() {
    return new AdminTransportClient({ stream: false });
}

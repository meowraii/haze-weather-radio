import { API_BASE } from './api.js';
import { session } from './session.js';

const DEFAULT_RECONNECT_MS = 1200;

export class PanelClient extends EventTarget {
    constructor({ base = API_BASE, stream = true, params = {} } = {}) {
        super();
        this.base = base;
        this.stream = stream;
        this.params = params;
        this.socket = null;
        this.connected = false;
        this.reconnectTimer = null;
        this.requestId = 0;
        this.pending = new Map();
        this.reconnectEnabled = false;
    }

    url(extra = {}) {
        const url = new URL(`${this.base}/panel/ws`, window.location.origin);
        url.protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const params = { ...this.params, ...extra };
        if (!this.stream) params.mode = params.mode || 'control';
        if (session.token) params.token = session.token;
        Object.entries(params).forEach(([key, value]) => {
            if (value !== undefined && value !== null && value !== '') {
                url.searchParams.set(key, String(value));
            }
        });
        return url.toString();
    }

    connect() {
        this.reconnectEnabled = true;
        if (this.socket && this.socket.readyState <= WebSocket.OPEN) return this.socket;
        const socket = new WebSocket(this.url());
        this.socket = socket;

        socket.addEventListener('open', () => {
            this.connected = true;
            this.dispatch('open');
        });
        socket.addEventListener('message', (event) => this.handleMessage(event.data));
        socket.addEventListener('error', () => this.dispatch('error'));
        socket.addEventListener('close', (event) => {
            this.connected = false;
            if (this.socket === socket) this.socket = null;
            for (const { reject, timer } of this.pending.values()) {
                window.clearTimeout(timer);
                reject(new Error(`Panel websocket closed: ${event.code}`));
            }
            this.pending.clear();
            this.dispatch('close', event);
            if (this.reconnectEnabled) this.scheduleReconnect();
        });
        return socket;
    }

    close() {
        this.reconnectEnabled = false;
        if (this.reconnectTimer) window.clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
        if (this.socket) this.socket.close();
    }

    scheduleReconnect() {
        if (this.reconnectTimer) return;
        this.reconnectTimer = window.setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, DEFAULT_RECONNECT_MS);
    }

    handleMessage(raw) {
        let payload;
        try {
            payload = JSON.parse(raw);
        } catch {
            this.dispatch('decode_error', raw);
            return;
        }
        const replyTo = payload.reply_to || payload.request_id;
        if (replyTo && this.pending.has(replyTo)) {
            const pending = this.pending.get(replyTo);
            this.pending.delete(replyTo);
            window.clearTimeout(pending.timer);
            if (payload.type === 'command_error' || payload.type === 'auth_error' || payload.type === 'error') {
                pending.reject(new Error(payload.detail || 'Panel command failed.'));
            } else {
                pending.resolve(payload.result !== undefined ? payload.result : payload);
            }
            return;
        }
        const compatibleReply = this.findCompatiblePending(payload);
        if (compatibleReply) {
            const [requestId, pending] = compatibleReply;
            this.pending.delete(requestId);
            window.clearTimeout(pending.timer);
            if (payload.type === 'auth_error' || payload.type === 'error') {
                pending.reject(new Error(payload.detail || 'Panel command failed.'));
            } else {
                pending.resolve(payload);
            }
            return;
        }
        this.dispatch(payload.type || 'message', payload);
        this.dispatch('message', payload);
    }

    findCompatiblePending(payload) {
        if (!payload?.type) return null;
        for (const entry of this.pending.entries()) {
            const [, pending] = entry;
            if (pending.accepts?.has(payload.type)) return entry;
        }
        return null;
    }

    send(payload) {
        const socket = this.connect();
        if (socket.readyState === WebSocket.OPEN) {
            socket.send(JSON.stringify(payload));
            return;
        }
        socket.addEventListener('open', () => socket.send(JSON.stringify(payload)), { once: true });
    }

    request(type, payload = {}, timeoutMs = 8000) {
        const requestId = `r${Date.now().toString(36)}${(++this.requestId).toString(36)}`;
        const message = { type, request_id: requestId, ...payload };
        return new Promise((resolve, reject) => {
            const timer = window.setTimeout(() => {
                this.pending.delete(requestId);
                reject(new Error('Panel command timed out.'));
            }, timeoutMs);
            this.pending.set(requestId, { resolve, reject, timer, accepts: acceptedReplyTypes(type) });
            this.send(message);
        });
    }

    command(name, payload = {}, timeoutMs = 15000) {
        return this.request('command', { command: name, payload }, timeoutMs);
    }

    dispatch(type, detail) {
        this.dispatchEvent(new CustomEvent(type, { detail }));
    }
}

export const panelClient = new PanelClient({ stream: true, params: { source: 'app', lines: '120' } });

export function createControlClient() {
    return new PanelClient({ stream: false });
}

function acceptedReplyTypes(requestType) {
    if (requestType === 'auth_check') return new Set(['auth_state']);
    if (requestType === 'login') return new Set(['auth_ok', 'auth_error']);
    if (requestType === 'logout') return new Set(['logout_ok']);
    if (requestType === 'ping') return new Set(['pong']);
    return new Set();
}

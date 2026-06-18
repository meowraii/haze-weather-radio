import { API_BASE } from './api.js';
import { session } from './session.js';

const DEFAULT_RECONNECT_MS = 800;
const MAX_RECONNECT_MS = 15000;
const RECONNECT_JITTER = 0.35;
const MAX_OUTBOX_MESSAGES = 100;

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
        this.reconnectAttempt = 0;
        this.hadOpen = false;
        this.manualClose = false;
        this.outbox = [];
        this.manualCloseSockets = new WeakSet();
        this.reconnectCloseSockets = new WeakSet();
        window.addEventListener('online', () => {
            if (this.reconnectEnabled && !this.connected) this.reconnectNow();
        });
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
        this.manualClose = false;
        if (this.socket && this.socket.readyState <= WebSocket.OPEN) return this.socket;
        this.dispatch('connecting', { attempt: this.reconnectAttempt });
        const socket = new WebSocket(this.url());
        this.socket = socket;

        socket.addEventListener('open', () => {
            const recovered = this.hadOpen || this.reconnectAttempt > 0;
            this.connected = true;
            this.hadOpen = true;
            this.reconnectAttempt = 0;
            if (this.reconnectTimer) window.clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
            this.dispatch('open', { recovered });
            if (recovered) this.dispatch('recovered');
            this.flushOutbox();
        });
        socket.addEventListener('message', (event) => this.handleMessage(event.data));
        socket.addEventListener('error', () => this.dispatch('error', { attempt: this.reconnectAttempt }));
        socket.addEventListener('close', (event) => {
            const isCurrentSocket = this.socket === socket;
            const closedForManual = this.manualCloseSockets.has(socket);
            const closedForReconnect = this.reconnectCloseSockets.has(socket);
            if (isCurrentSocket) this.connected = false;
            if (isCurrentSocket) this.socket = null;
            if (isCurrentSocket) {
                for (const { reject, timer } of this.pending.values()) {
                    window.clearTimeout(timer);
                    reject(new Error(this.reconnectEnabled && !this.manualClose
                        ? 'Panel websocket interrupted. Reconnecting...'
                        : `Panel websocket closed: ${event.code}`));
                }
                this.pending.clear();
            }
            const reconnecting = isCurrentSocket && (closedForReconnect || (this.reconnectEnabled && !this.manualClose && !closedForManual));
            this.dispatch('close', {
                code: event.code,
                reason: event.reason,
                wasClean: event.wasClean,
                reconnecting,
            });
            if (reconnecting) this.scheduleReconnect(event);
        });
        return socket;
    }

    close() {
        this.reconnectEnabled = false;
        this.manualClose = true;
        if (this.reconnectTimer) window.clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
        this.outbox = [];
        if (this.socket) {
            this.manualCloseSockets.add(this.socket);
            this.socket.close();
        }
    }

    reconnectNow() {
        this.reconnectEnabled = true;
        this.manualClose = false;
        if (this.reconnectTimer) window.clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
        if (this.socket && this.socket.readyState <= WebSocket.OPEN) {
            this.reconnectCloseSockets.add(this.socket);
            this.socket.close(1000, 'client reconnect');
            return;
        }
        this.connect();
    }

    scheduleReconnect(event = {}) {
        if (this.reconnectTimer) return;
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
            this.connect();
        }, delay);
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
            if (payload.type === 'auth_error' || payload.type === 'error' || payload.type === 'webrtc_error') {
                pending.reject(new Error(payload.detail || 'Panel command failed.'));
            } else {
                pending.resolve(payload);
            }
            return;
        }
        this.dispatch(payload.type || 'message', payload.data !== undefined ? payload.data : payload);
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
        this.queueMessage(payload);
    }

    queueMessage(payload) {
        if (this.outbox.length >= MAX_OUTBOX_MESSAGES) {
            this.outbox.shift();
        }
        this.outbox.push(payload);
        this.dispatch('queued', { size: this.outbox.length });
    }

    flushOutbox() {
        if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
        while (this.outbox.length && this.socket.readyState === WebSocket.OPEN) {
            const payload = this.outbox.shift();
            if (payload.request_id && !this.pending.has(payload.request_id)) {
                continue;
            }
            this.socket.send(JSON.stringify(payload));
        }
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
    if (requestType === 'webrtc_offer') return new Set(['webrtc_answer', 'webrtc_error', 'auth_error']);
    return new Set();
}

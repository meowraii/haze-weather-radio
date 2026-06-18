import { session, token } from './session.js';

export const API_BASE = '/api/v1';
export { session, token };

async function command(name, payload = {}, timeoutMs = 15000) {
    const { createControlClient } = await import('./ws-client.js');
    const client = createControlClient();
    try {
        return await client.command(name, payload, timeoutMs);
    } finally {
        client.close();
    }
}

export function apiCommand(name, payload = {}, timeoutMs = 15000) {
    return command(name, payload, timeoutMs);
}

async function authCheck() {
    const { createControlClient } = await import('./ws-client.js');
    const client = createControlClient();
    try {
        return await client.request('auth_check', {}, 5000);
    } finally {
        client.close();
    }
}

async function onUnauth() {
    try {
        const state = await authCheck();
        if (state.authenticated) return true;
    } catch {
        // If the control socket cannot confirm the session, fall back to login.
    }
    return false;
}

async function unwrapError(response) {
    const body = await response.json().catch(() => ({ detail: `Request failed: ${response.status}` }));
    return new Error(body.detail || `Request failed: ${response.status}`);
}

function commandFor(method, path, body) {
    const cleanPath = path.split('?')[0];
    const query = new URLSearchParams(path.includes('?') ? path.slice(path.indexOf('?') + 1) : '');

    if (method === 'GET' && cleanPath === '/health') return ['health', {}];
    if (method === 'GET' && cleanPath === '/automations') return ['automations.get', {}];
    if (method === 'PUT' && cleanPath === '/automations') return ['automations.put', body || {}];
    if (method === 'GET' && cleanPath === '/same/templates') return ['same.templates.get', {}];
    if (method === 'PUT' && cleanPath === '/same/templates') return ['same.templates.put', body || {}];
    if (method === 'POST' && cleanPath === '/same/test') {
        return ['same.test', { ...(body || {}), event_code: query.get('event_code') || body?.event_code || 'RWT' }];
    }
    if (method === 'POST' && cleanPath === '/same/intro') return ['same.intro', body || {}];
    if (method === 'POST' && cleanPath === '/same/generate') return ['same.generate', body || {}];
    if (method === 'POST' && cleanPath === '/same/air') return ['same.air', body || {}];
    if (method === 'POST' && cleanPath === '/alert/broadcast') return ['alert.broadcast', body || {}];
    if (method === 'POST' && cleanPath === '/same/upload-audio') return ['same.upload_audio', { filename: body?.get?.('file')?.name || '' }];
    if (method === 'GET' && cleanPath === '/same/event-codes') return ['same.event_codes', {}];
    if (method === 'GET' && cleanPath === '/same/location-names') return ['same.location_names', {}];
    return null;
}

async function requestJson(method, path, body = undefined) {
    const mapped = commandFor(method, path, body);
    if (mapped) {
        const [name, payload] = mapped;
        return command(name, payload);
    }
    throw new Error(`No websocket command is mapped for ${method} ${path}.`);
}

export function apiGet(path) {
    return requestJson('GET', path);
}

export function apiPost(path, body) {
    return requestJson('POST', path, body || {});
}

export function apiPut(path, body) {
    return requestJson('PUT', path, body || {});
}

export async function apiPostForm(path, formData) {
    const mapped = commandFor('POST', path, formData);
    if (!mapped) {
        throw new Error(`No websocket command is mapped for POST ${path}.`);
    }
    const [name, payload] = mapped;
    return command(name, payload);
}

export async function apiRaw(path, options = {}) {
    throw new Error(`Raw REST requests are disabled for ${path}. Use a websocket command instead.`);
}

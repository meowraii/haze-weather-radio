export const API_BASE = '/api/v1';
export const TOKEN_KEY = 'haze.panel.token';

export const token = {
    get: () => localStorage.getItem(TOKEN_KEY) || '',
    set: (t) => localStorage.setItem(TOKEN_KEY, t),
    clear: () => localStorage.removeItem(TOKEN_KEY),
};

function authHeaders(extra = {}) {
    const h = new Headers(extra);
    const t = token.get();
    if (t) h.set('Authorization', `Bearer ${t}`);
    return h;
}

function onUnauth() {
    token.clear();
    window.location.href = '/';
}

async function unwrapError(r) {
    const body = await r.json().catch(() => ({ detail: `Request failed: ${r.status}` }));
    return new Error(body.detail || `Request failed: ${r.status}`);
}

export async function apiGet(path) {
    const r = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
    if (r.status === 401) { onUnauth(); throw new Error('Not authenticated'); }
    if (!r.ok) throw await unwrapError(r);
    return r.json();
}

export async function apiPost(path, body) {
    const r = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(body),
    });
    if (r.status === 401) { onUnauth(); throw new Error('Not authenticated'); }
    if (!r.ok) throw await unwrapError(r);
    return r.json();
}

export async function apiPostForm(path, formData) {
    const r = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: authHeaders(),
        body: formData,
    });
    if (r.status === 401) { onUnauth(); throw new Error('Not authenticated'); }
    if (!r.ok) throw await unwrapError(r);
    return r.json();
}

export async function apiPut(path, body) {
    const r = await fetch(`${API_BASE}${path}`, {
        method: 'PUT',
        headers: authHeaders({ 'Content-Type': 'application/json' }),
        body: JSON.stringify(body),
    });
    if (r.status === 401) { onUnauth(); throw new Error('Not authenticated'); }
    if (!r.ok) throw await unwrapError(r);
    return r.json();
}

export async function apiRaw(path, options = {}) {
    const extra = options.body && typeof options.body === 'string'
        ? { 'Content-Type': 'application/json' }
        : {};
    const r = await fetch(`${API_BASE}${path}`, { ...options, headers: authHeaders(extra) });
    if (r.status === 401) { onUnauth(); throw new Error('Not authenticated'); }
    return r;
}

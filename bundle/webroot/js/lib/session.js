export const TOKEN_KEY = 'haze.panel.token';
export const SESSION_COOKIE = 'haze_admin_session';

let memoryToken = '';

function storageGet(key) {
    try { return localStorage.getItem(key) || ''; } catch { return ''; }
}

function storageSet(key, value) {
    try {
        if (value) localStorage.setItem(key, value);
        else localStorage.removeItem(key);
    } catch {
        // Some browser contexts block storage. The memory fallback still works.
    }
}

function cookieToken() {
    try {
        const prefix = `${SESSION_COOKIE}=`;
        const found = document.cookie
            .split(';')
            .map((part) => part.trim())
            .find((part) => part.startsWith(prefix));
        return found ? decodeURIComponent(found.slice(prefix.length)) : '';
    } catch {
        return '';
    }
}

function setCookieToken(value) {
    try {
        const secure = window.location.protocol === 'https:' ? '; Secure' : '';
        if (value) {
            document.cookie = `${SESSION_COOKIE}=${encodeURIComponent(value)}; Path=/; SameSite=Lax; Max-Age=43200${secure}`;
        } else {
            document.cookie = `${SESSION_COOKIE}=; Path=/; SameSite=Lax; Max-Age=0${secure}`;
        }
    } catch {
        // Ignore; URL token and memory/local storage remain available.
    }
}

export const session = {
    tokenFromUrl() {
        try {
            return new URLSearchParams(window.location.search).get('token') || '';
        } catch {
            return '';
        }
    },

    get token() {
        return storageGet(TOKEN_KEY) || memoryToken || cookieToken() || this.tokenFromUrl();
    },

    setToken(value) {
        memoryToken = value || '';
        storageSet(TOKEN_KEY, memoryToken);
        setCookieToken(memoryToken);
    },

    clear() {
        memoryToken = '';
        storageSet(TOKEN_KEY, '');
        setCookieToken('');
    },

    importUrlToken() {
        const value = this.tokenFromUrl();
        if (!value) return false;
        this.setToken(value);
        const params = new URLSearchParams(window.location.search);
        params.delete('token');
        const cleanSearch = params.toString();
        const cleanUrl = `${window.location.pathname}${cleanSearch ? `?${cleanSearch}` : ''}${window.location.hash || ''}`;
        window.history.replaceState(null, '', cleanUrl);
        return true;
    },

    authHeaders(extra = {}) {
        const headers = new Headers(extra);
        if (this.token) headers.set('Authorization', `Bearer ${this.token}`);
        return headers;
    },
};

export const token = {
    get: () => session.token,
    set: (value) => session.setToken(value),
    clear: () => session.clear(),
};

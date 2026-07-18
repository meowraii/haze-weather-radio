export const SESSION_COOKIE = 'haze_admin_session';

export const session = {
    get token() {
        return '';
    },

    setToken() {},

	clear() {},

    importUrlToken() {
        return false;
    },

    authHeaders(extra = {}) {
        return new Headers(extra);
    },
};

export const token = {
    get: () => session.token,
    set: (value) => session.setToken(value),
    clear: () => session.clear(),
};

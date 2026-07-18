import { initTheme } from './lib/theme.js';

initTheme();

const form = document.getElementById('loginForm');
const usernameInput = document.getElementById('usernameInput');
const passwordInput = document.getElementById('passwordInput');
const totpInput = document.getElementById('totpInput');
const persistentInput = document.getElementById('persistentInput');
const mfaFields = document.getElementById('mfaFields');
const mfaEnrollment = document.getElementById('mfaEnrollment');
const mfaEnrollmentSecret = document.getElementById('mfaEnrollmentSecret');
const statusEl = document.getElementById('authStatus');
const button = document.getElementById('loginButton');
const loginTitle = document.getElementById('loginTitle');
const loginSubtitle = document.getElementById('loginSubtitle');
let loginInFlight = false;

const params = new URLSearchParams(window.location.search);
const nextPath = params.get('next')?.startsWith('/') ? params.get('next') : '/admin';

document.documentElement.dataset.loginJs = 'ready';
form.dataset.loginBound = 'true';
button.dataset.loginBound = 'true';

function setStatus(text, state = '') {
    statusEl.textContent = text;
    statusEl.dataset.state = state;
}

function destination(path) {
    const destination = new URL(path, window.location.origin);
    return `${destination.pathname}${destination.search}${destination.hash}`;
}

function showMFA({ enrollmentSecret = '' } = {}) {
    mfaFields.hidden = false;
    totpInput.required = true;
    if (enrollmentSecret) {
        mfaEnrollment.hidden = false;
        mfaEnrollmentSecret.textContent = enrollmentSecret;
    }
    totpInput.focus();
}

async function loadHealth() {
    try {
        const response = await fetch('/api/v1/auth/check', {
            credentials: 'same-origin',
            cache: 'no-store',
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const health = await response.json();
        const siteName = health.site_name || 'Haze Weather Radio';
        const onAirName = health.on_air_name || siteName;
        const version = health.version || 'dev';
        const gitCommit = health.git_commit || 'unknown';
        loginTitle.textContent = siteName;
        document.title = `${siteName} Sign In`;
        loginSubtitle.textContent = `Haze Weather Radio for ${onAirName} with git ${gitCommit} of version ${version}`;
		if (health.authenticated) {
			setStatus('Already signed in. Opening panel...', 'ok');
			window.location.href = destination(health.password_change_required ? '/admin#/accounts' : nextPath);
		}
    } catch {
        loginTitle.textContent = 'Haze Weather Radio';
        loginSubtitle.textContent = 'Haze Weather Radio';
    }
}

async function signIn(event) {
    event?.preventDefault();
    event?.stopPropagation();
    if (loginInFlight) return;
    if (!usernameInput.value.trim()) {
        usernameInput.focus();
        setStatus('Enter your account username.', 'err');
        return;
    }
    if (!passwordInput.value) {
        passwordInput.focus();
        setStatus('Enter the operator password.', 'err');
        return;
    }
    loginInFlight = true;
    button.disabled = true;
    setStatus('Opening secure sign in...', 'pending');
    try {
        setStatus('Authenticating...', 'pending');
        const payload = await loginWithHTTP({
            username: usernameInput.value.trim(),
            password: passwordInput.value,
            totp: totpInput.value.trim(),
            persistent: persistentInput.checked,
        });
        if (payload.type !== 'auth_ok') {
            throw new Error(payload.detail || 'Incorrect password or unavailable sign in service.');
        }
        passwordInput.value = '';
        setStatus('Signed in. Opening panel...', 'ok');
        window.location.href = destination(payload.password_change_required ? '/admin#/accounts' : nextPath);
    } catch (error) {
        if (error.payload?.code === 'mfa_required' || error.payload?.code === 'invalid_mfa') {
            showMFA();
        }
        if (error.payload?.mfa_enrollment_required) {
            showMFA({ enrollmentSecret: error.payload.mfa_enrollment_secret || '' });
        }
        setStatus(error.message || 'Unable to sign in.', 'err');
        if (mfaFields.hidden) passwordInput.select();
    } finally {
        loginInFlight = false;
        button.disabled = false;
    }
}

async function loginWithHTTP(credentials) {
    const controller = new AbortController();
    const timer = window.setTimeout(() => controller.abort(), 8000);
    try {
        const response = await fetch('/api/v1/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            credentials: 'same-origin',
            cache: 'no-store',
            body: JSON.stringify(credentials),
            signal: controller.signal,
        });
        let payload = {};
        try {
            payload = await response.json();
        } catch {
            payload = {};
        }
        if (!response.ok) {
            const error = new Error(payload.detail || `Sign in failed with HTTP ${response.status}.`);
            error.payload = payload;
            throw error;
        }
        return payload;
    } catch (error) {
        if (error.name === 'AbortError') {
            throw new Error('Sign in timed out.');
        }
        throw error;
    } finally {
        window.clearTimeout(timer);
    }
}

form.addEventListener('submit', signIn);
button.addEventListener('click', signIn);
document.addEventListener('submit', (event) => {
    if (event.target === form) signIn(event);
}, true);
document.addEventListener('click', (event) => {
    if (event.target?.closest?.('#loginButton')) signIn(event);
}, true);
passwordInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') signIn(event);
});
totpInput.addEventListener('input', () => {
    totpInput.value = totpInput.value.replace(/\D/g, '').slice(0, 6);
});
window.hazeLoginSignIn = signIn;

loadHealth();

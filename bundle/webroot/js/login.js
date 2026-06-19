import { token } from './lib/api.js';
import { initTheme } from './lib/theme.js';
import { createControlClient } from './lib/ws-client.js';

initTheme();

const form = document.getElementById('loginForm');
const passwordInput = document.getElementById('passwordInput');
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

function destinationWithToken(path, sessionToken = token.get()) {
    const destination = new URL(path, window.location.origin);
    if (sessionToken) destination.searchParams.set('token', sessionToken);
    return `${destination.pathname}${destination.search}${destination.hash}`;
}

async function loadHealth() {
    const client = createControlClient();
    try {
        const health = await client.request('auth_check');
        const siteName = health.site_name || 'Haze Weather Radio';
        const onAirName = health.on_air_name || siteName;
        const version = health.version || 'dev';
        const gitCommit = health.git_commit || 'unknown';
        loginTitle.textContent = siteName;
        document.title = `${siteName} Sign In`;
        loginSubtitle.textContent = `Haze Weather Radio for ${onAirName} with git ${gitCommit} of version ${version}`;
        if (health.authenticated) {
            setStatus('Already signed in. Opening panel...', 'ok');
            window.location.href = destinationWithToken(nextPath);
        }
    } catch {
        loginTitle.textContent = 'Haze Weather Radio';
        loginSubtitle.textContent = 'Haze Weather Radio';
    } finally {
        client.close();
    }
}

async function signIn(event) {
    event?.preventDefault();
    event?.stopPropagation();
    if (loginInFlight) return;
    if (!passwordInput.value) {
        passwordInput.focus();
        setStatus('Enter the operator password.', 'err');
        return;
    }
    loginInFlight = true;
    button.disabled = true;
    setStatus('Opening authentication websocket...', 'pending');
    const client = createControlClient();
    try {
        setStatus('Authenticating...', 'pending');
        const payload = await client.request('login', { password: passwordInput.value }, 8000);
        if (payload.type !== 'auth_ok') {
            throw new Error(payload.detail || 'Incorrect password or unavailable websocket.');
        }
        passwordInput.value = '';
        setStatus('Signed in. Opening panel...', 'ok');
        window.location.href = destinationWithToken(nextPath, payload.token);
    } catch (error) {
        setStatus(error.message || 'Unable to sign in.', 'err');
        passwordInput.select();
    } finally {
        client.close();
        loginInFlight = false;
        button.disabled = false;
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
window.hazeLoginSignIn = signIn;

loadHealth();

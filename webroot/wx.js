const TOKEN_KEY = 'haze.panel.token';
const THEME_KEY = 'haze.theme';

const PACKAGE_DESCRIPTIONS = {
    date_time: 'Current date and time announcement for the selected feed timezone.',
    station_id: 'Station identification and configured callsign.',
    current_conditions: 'Observed surface weather conditions from observation locations.',
    forecast: 'Forecast text for configured ECCC, NWS, or TWC-backed forecast locations.',
    climate_summary: 'Daily climate normal and anomaly summary for configured climate locations.',
    eccc_discussion: 'Raw ECCC FOCN45 forecast discussion bulletin.',
    geophysical_alert: 'NOAA WWV geophysical alert and solar activity bulletin.',
    user_bulletin: 'User-defined bulletin text loaded from userbulletins.json.',
};

const FORMAT_INFO = {
    raw: { mime: 'audio/raw', delivery: 'streamed', type: 'audio' },
    wav: { mime: 'audio/wav', delivery: 'buffered', type: 'audio' },
    mp3: { mime: 'audio/mpeg', delivery: 'buffered', type: 'audio' },
    ogg: { mime: 'audio/ogg', delivery: 'buffered', type: 'audio' },
    flac: { mime: 'audio/flac', delivery: 'buffered', type: 'audio' },
    aac: { mime: 'audio/aac', delivery: 'buffered', type: 'audio' },
    opus: { mime: 'audio/ogg; codecs=opus', delivery: 'buffered', type: 'audio' },
    ulaw: { mime: 'audio/basic', delivery: 'buffered', type: 'audio' },
    alaw: { mime: 'audio/x-alaw', delivery: 'buffered', type: 'audio' },
    g722: { mime: 'audio/G722', delivery: 'buffered', type: 'audio' },
    webm: { mime: 'audio/webm', delivery: 'buffered', type: 'audio' },
    json: { mime: 'application/json', delivery: 'buffered', type: 'text' },
    xml: { mime: 'application/xml', delivery: 'buffered', type: 'text' },
    ssml: { mime: 'application/ssml+xml', delivery: 'buffered', type: 'text' },
    html: { mime: 'text/html', delivery: 'buffered', type: 'text' },
    markdown: { mime: 'text/markdown', delivery: 'buffered', type: 'text' },
    latex: { mime: 'application/x-latex', delivery: 'buffered', type: 'text' },
};

const TEXT_FORMATS = new Set(['json', 'xml', 'ssml', 'html', 'markdown', 'latex']);
const DEFAULT_PACKAGES = ['date_time', 'current_conditions', 'forecast'];
const SUN_ICON = '<i data-lucide="sun" width="13" height="13"></i>';
const MOON_ICON = '<i data-lucide="moon" width="13" height="13"></i>';

const state = {
    apiBase: detectApiBase(),
    wxBase: '/api/wx-on-demand/v1',
    allPackages: [],
    selectedPackages: new Set(),
    abortController: null,
    objectUrl: null,
    busy: false,
};

const apiDot = document.getElementById('apiDot');
const wxDot = document.getElementById('wxDot');
const healthPill = document.getElementById('healthPill');
const wxBasePill = document.getElementById('wxBasePill');
const statusBanner = document.getElementById('statusBanner');
const themeToggle = document.getElementById('themeToggle');
const themeLabel = document.getElementById('themeLabel');
const tryLocations = document.getElementById('tryLocations');
const trySource = document.getElementById('trySource');
const tryLang = document.getElementById('tryLang');
const tryVoice = document.getElementById('tryVoice');
const tryFormat = document.getElementById('tryFormat');
const tryPkgs = document.getElementById('tryPkgs');
const tryBtn = document.getElementById('tryBtn');
const tryStop = document.getElementById('tryStop');
const tryStatus = document.getElementById('tryStatus');
const tryAudio = document.getElementById('tryAudio');
const tryText = document.getElementById('tryText');
const pkgTable = document.getElementById('pkgTable');
const formatTable = document.getElementById('formatTable');
const generatePath = document.getElementById('generatePath');
const packagesPath = document.getElementById('packagesPath');
const wxBaseMetric = document.getElementById('wxBaseMetric');
const sourceMetric = document.getElementById('sourceMetric');
const formatMetric = document.getElementById('formatMetric');
const packageMetric = document.getElementById('packageMetric');
const responseTypeMetric = document.getElementById('responseTypeMetric');
const requestBodyPreview = document.getElementById('requestBodyPreview');
const curlPreview = document.getElementById('curlPreview');
const responseMeta = document.getElementById('responseMeta');

function detectApiBase() {
    const stored = sessionStorage.getItem('haze.api_base');
    if (stored) {
        return stored;
    }
    const match = window.location.pathname.match(/^(\/api\/[^/]+)\//);
    const base = match ? match[1] : '/api/v1';
    sessionStorage.setItem('haze.api_base', base);
    return base;
}

function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
    const isDark = theme === 'dark';
    themeToggle.innerHTML = isDark ? SUN_ICON : MOON_ICON;
    themeLabel.textContent = isDark ? 'Light mode' : 'Dark mode';
    if (window.lucide) {
        lucide.createIcons({ nodes: [themeToggle] });
    }
}

function initializeTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved) {
        applyTheme(saved);
    } else {
        themeLabel.textContent = 'Dark mode';
    }
    themeToggle.addEventListener('click', () => {
        const nextTheme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
        applyTheme(nextTheme);
    });
}

function splitLocations(value) {
    return String(value || '')
        .split(/[\n,]+/)
        .map((item) => item.trim())
        .filter(Boolean);
}

function getSelectedPackages() {
    return state.allPackages.filter((pkg) => state.selectedPackages.has(pkg));
}

function getPayload() {
    const locations = splitLocations(tryLocations.value);
    const packages = getSelectedPackages();
    const payload = {
        locations: locations.length === 1 ? locations[0] : locations,
        packages: packages.length === state.allPackages.length ? 'all' : packages,
        lang: tryLang.value,
        format: tryFormat.value,
    };
    const voice = tryVoice.value.trim();
    if (voice) {
        payload.voice = voice;
    }
    if (trySource.value) {
        payload.source = trySource.value;
    }
    return payload;
}

function updateMetrics() {
    const selectedPackages = getSelectedPackages();
    sourceMetric.textContent = trySource.value || 'auto';
    formatMetric.textContent = tryFormat.value;
    packageMetric.textContent = selectedPackages.length === state.allPackages.length ? 'all' : String(selectedPackages.length);
    responseTypeMetric.textContent = TEXT_FORMATS.has(tryFormat.value) ? 'text' : 'audio';
    wxBaseMetric.textContent = state.wxBase;
    wxBasePill.textContent = state.wxBase;
    generatePath.textContent = `${state.wxBase}/generate`;
    packagesPath.textContent = `${state.wxBase}/packages`;
}

function updateRequestPreviews() {
    const payload = getPayload();
    requestBodyPreview.textContent = JSON.stringify(payload, null, 2);

    const curlLines = [
        `curl -s -X POST ${state.wxBase}/generate`,
        `  -H 'Content-Type: application/json'`,
        `  -d '${JSON.stringify(payload)}'`,
    ];
    if (!TEXT_FORMATS.has(payload.format)) {
        curlLines.push(`  --output weather.${payload.format === 'raw' ? 'pcm' : payload.format}`);
    }
    curlPreview.textContent = curlLines.join(' \\\n');

    tryBtn.textContent = TEXT_FORMATS.has(payload.format) ? 'Generate' : 'Generate & Play';
    tryBtn.disabled = state.busy || getSelectedPackages().length === 0;
    updateMetrics();
}

function populatePackageTable(packages) {
    pkgTable.innerHTML = packages.map((pkg) => (
        `<tr><td><code>${pkg}</code></td><td>${PACKAGE_DESCRIPTIONS[pkg] || ''}</td></tr>`
    )).join('');
}

function populateFormatTable() {
    formatTable.innerHTML = Object.entries(FORMAT_INFO).map(([format, info]) => (
        `<tr><td><code>${format}</code></td><td><code>${info.mime}</code></td><td>${info.delivery}</td><td>${info.type}</td></tr>`
    )).join('');
}

function ensureDefaultPackages() {
    state.selectedPackages.clear();
    for (const pkg of DEFAULT_PACKAGES) {
        if (state.allPackages.includes(pkg)) {
            state.selectedPackages.add(pkg);
        }
    }
    if (!state.selectedPackages.size) {
        for (const pkg of state.allPackages.slice(0, 3)) {
            state.selectedPackages.add(pkg);
        }
    }
}

function renderPackageChips() {
    tryPkgs.innerHTML = '';
    for (const pkg of state.allPackages) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = `pkg-chip${state.selectedPackages.has(pkg) ? ' active' : ''}`;
        chip.textContent = pkg.replace(/_/g, ' ');
        chip.addEventListener('click', () => {
            if (state.selectedPackages.has(pkg)) {
                state.selectedPackages.delete(pkg);
            } else {
                state.selectedPackages.add(pkg);
            }
            renderPackageChips();
            updateRequestPreviews();
        });
        tryPkgs.appendChild(chip);
    }
}

function setHealth(ok, bannerText, detailText) {
    apiDot.dataset.state = ok ? 'ok' : 'err';
    wxDot.dataset.state = ok ? 'ok' : 'err';
    healthPill.textContent = detailText;
    statusBanner.textContent = bannerText;
    statusBanner.className = `status-banner ${ok ? 'ok' : 'err'}`;
}

function setBusy(busy, message = '') {
    state.busy = busy;
    if (message) {
        tryStatus.textContent = message;
    }
    tryStop.disabled = !busy;
    updateRequestPreviews();
}

function resetOutput() {
    if (state.objectUrl) {
        URL.revokeObjectURL(state.objectUrl);
        state.objectUrl = null;
    }
    tryAudio.pause();
    tryAudio.removeAttribute('src');
    tryAudio.hidden = true;
    tryText.hidden = true;
    tryText.textContent = '';
}

function renderResponseHeaders(response) {
    const fields = [
        ['Content-Type', response.headers.get('Content-Type') || FORMAT_INFO[tryFormat.value]?.mime || 'unknown'],
        ['X-Format', response.headers.get('X-Format') || tryFormat.value],
        ['X-Source', response.headers.get('X-Source') || 'auto'],
        ['X-Packages', response.headers.get('X-Packages') || getSelectedPackages().join(',') || 'none'],
        ['Content-Length', response.headers.get('Content-Length') || 'streamed'],
        ['X-Audio-Sample-Rate', response.headers.get('X-Audio-Sample-Rate') || 'n/a'],
        ['X-Audio-Channels', response.headers.get('X-Audio-Channels') || 'n/a'],
    ];
    responseMeta.innerHTML = fields.map(([label, value]) => (
        `<article class="wx-panel-response-card"><p>${label}</p><strong>${value}</strong></article>`
    )).join('');
}

function renderResponseError(message) {
    responseMeta.innerHTML = `<article class="wx-panel-response-card is-empty"><p>${message}</p></article>`;
}

function pcmToWav(pcm, sampleRate, channels) {
    const dataLength = pcm.byteLength;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);
    const writeString = (offset, value) => {
        for (let index = 0; index < value.length; index += 1) {
            view.setUint8(offset + index, value.charCodeAt(index));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * channels * 2, true);
    view.setUint16(32, channels * 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, dataLength, true);
    new Uint8Array(buffer).set(pcm, 44);
    return buffer;
}

async function readError(response) {
    const contentType = response.headers.get('Content-Type') || '';
    if (contentType.includes('application/json')) {
        try {
            const payload = await response.json();
            return payload.detail || JSON.stringify(payload);
        } catch {
            return `Request failed with status ${response.status}`;
        }
    }
    const body = await response.text().catch(() => '');
    return body || `Request failed with status ${response.status}`;
}

function buildHeaders() {
    const headers = new Headers({ 'Content-Type': 'application/json' });
    const token = localStorage.getItem(TOKEN_KEY) || '';
    if (token) {
        headers.set('Authorization', `Bearer ${token}`);
    }
    return headers;
}

async function loadHealth() {
    const response = await fetch(`${state.apiBase}/health`, { headers: buildHeaders() });
    if (!response.ok) {
        throw new Error(await readError(response));
    }
    const health = await response.json();
    state.wxBase = health.wx_base || state.wxBase;
    const authState = health.auth_required ? 'panel auth enabled' : 'panel auth disabled';
    setHealth(true, `API healthy. WX base ${state.wxBase}. ${authState}.`, `Healthy • ${authState}`);
}

async function loadPackages() {
    const response = await fetch(`${state.wxBase}/packages`, { headers: buildHeaders() });
    if (!response.ok) {
        throw new Error(await readError(response));
    }
    const payload = await response.json();
    const packages = Array.isArray(payload.packages) ? payload.packages : [];
    state.allPackages = packages.length ? packages : Object.keys(PACKAGE_DESCRIPTIONS);
}

async function generate() {
    resetOutput();
    renderResponseError('Waiting for response.');
    if (state.abortController) {
        state.abortController.abort();
    }
    state.abortController = new AbortController();
    const payload = getPayload();
    setBusy(true, 'Generating…');
    tryStatus.className = 'try-status';

    try {
        const response = await fetch(`${state.wxBase}/generate`, {
            method: 'POST',
            headers: buildHeaders(),
            body: JSON.stringify(payload),
            signal: state.abortController.signal,
        });

        if (!response.ok) {
            throw new Error(await readError(response));
        }

        renderResponseHeaders(response);

        if (TEXT_FORMATS.has(payload.format)) {
            const text = await response.text();
            tryText.textContent = text;
            tryText.hidden = false;
        } else if (payload.format !== 'raw') {
            tryStatus.textContent = 'Downloading audio…';
            const blob = await response.blob();
            state.objectUrl = URL.createObjectURL(blob);
            tryAudio.src = state.objectUrl;
            tryAudio.hidden = false;
            await tryAudio.play();
        } else {
            const sampleRate = Number.parseInt(response.headers.get('X-Audio-Sample-Rate') || '16000', 10);
            const channels = Number.parseInt(response.headers.get('X-Audio-Channels') || '1', 10);
            const reader = response.body?.getReader();
            if (!reader) {
                throw new Error('Response stream was unavailable.');
            }
            const chunks = [];
            let totalBytes = 0;
            while (true) {
                const { done, value } = await reader.read();
                if (done) {
                    break;
                }
                if (value) {
                    chunks.push(value);
                    totalBytes += value.byteLength;
                    tryStatus.textContent = `Buffering raw PCM… ${Math.round(totalBytes / 1024)} KB`;
                }
            }
            const pcm = new Uint8Array(totalBytes);
            let offset = 0;
            for (const chunk of chunks) {
                pcm.set(chunk, offset);
                offset += chunk.byteLength;
            }
            state.objectUrl = URL.createObjectURL(new Blob([pcmToWav(pcm, sampleRate, channels)], { type: 'audio/wav' }));
            tryAudio.src = state.objectUrl;
            tryAudio.hidden = false;
            await tryAudio.play();
        }

        tryStatus.textContent = 'Ready';
        tryStatus.className = 'try-status ok';
    } catch (error) {
        if (error.name === 'AbortError') {
            tryStatus.textContent = 'Stopped.';
            tryStatus.className = 'try-status';
            renderResponseError('Request aborted.');
        } else {
            tryStatus.textContent = error.message || 'Request failed.';
            tryStatus.className = 'try-status err';
            renderResponseError(error.message || 'Request failed.');
        }
    } finally {
        state.abortController = null;
        setBusy(false);
    }
}

function bindEvents() {
    tryLocations.addEventListener('input', updateRequestPreviews);
    trySource.addEventListener('change', updateRequestPreviews);
    tryLang.addEventListener('change', updateRequestPreviews);
    tryVoice.addEventListener('input', updateRequestPreviews);
    tryFormat.addEventListener('change', updateRequestPreviews);
    tryBtn.addEventListener('click', generate);
    tryStop.addEventListener('click', () => {
        if (state.abortController) {
            state.abortController.abort();
        }
        tryAudio.pause();
    });
}

async function boot() {
    if (window.lucide) {
        lucide.createIcons();
    }
    initializeTheme();
    populateFormatTable();
    bindEvents();

    try {
        await loadHealth();
    } catch (error) {
        setHealth(false, `Panel API unavailable: ${error.message || 'request failed'}`, 'Unavailable');
    }

    try {
        await loadPackages();
    } catch {
        state.allPackages = Object.keys(PACKAGE_DESCRIPTIONS);
    }

    ensureDefaultPackages();
    populatePackageTable(state.allPackages);
    renderPackageChips();
    updateRequestPreviews();
    renderResponseError('No request has been sent yet.');
    tryStatus.textContent = 'Ready';
}

boot();
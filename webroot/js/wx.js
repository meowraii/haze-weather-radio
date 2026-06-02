import { token } from './lib/api.js';
import { initTheme } from './lib/theme.js';
import { pcmToWav } from './lib/audio.js';


const PACKAGE_DESCRIPTIONS = {
    date_time: 'Current date and time announcement for the selected feed timezone.',
    station_id: 'Station identification and configured callsign.',
    current_conditions: 'Observed surface weather conditions from observation locations.',
    forecast: 'Forecast text for configured ECCC, NWS, or TWC-backed forecast locations.',
    air_quality: 'Observed or forecast air quality guidance for configured AQHI locations.',
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
const DEFAULT_WX_SUFFIX = '/wx-on-demand';

const apiBase = detectApiBase();


const state = {
    apiBase,
    wxBase: `${apiBase}${DEFAULT_WX_SUFFIX}`,
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
const pkgDefaultBtn = document.getElementById('pkgDefaultBtn');
const pkgAllBtn = document.getElementById('pkgAllBtn');
const pkgClearBtn = document.getElementById('pkgClearBtn');
const pkgSummary = document.getElementById('pkgSummary');
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


function splitLocations(value) {
    return String(value || '')
        .split(/[\n,]+/)
        .map((item) => item.trim())
        .filter(Boolean);
}

function escapeHtml(value) {
    const element = document.createElement('span');
    element.textContent = String(value ?? '');
    return element.innerHTML;
}

function handleUnauthorized() {
    token.clear();
    window.location.href = '/';
}

function buildHeaders(extraHeaders = undefined) {
    const headers = new Headers(extraHeaders || {});
    const currentToken = token.get();
    if (currentToken) {
        headers.set('Authorization', `Bearer ${currentToken}`);
    }
    return headers;
}

async function fetchWithAuth(url, options = {}) {
    const { headers, ...rest } = options;
    const response = await fetch(url, { ...rest, headers: buildHeaders(headers) });
    if (response.status === 401) {
        handleUnauthorized();
        throw new Error('Not authenticated');
    }
    return response;
}

function getSelectedPackages() {
    return state.allPackages.filter((pkg) => state.selectedPackages.has(pkg));
}

function hasAllPackagesSelected() {
    return state.allPackages.length > 0 && getSelectedPackages().length === state.allPackages.length;
}

function setSelectedPackages(packageIds) {
    state.selectedPackages.clear();
    for (const packageId of packageIds) {
        if (state.allPackages.includes(packageId)) {
            state.selectedPackages.add(packageId);
        }
    }
}

function getPayload() {
    const locations = splitLocations(tryLocations.value);
    const packages = getSelectedPackages();
    const payload = {
        locations: locations.length === 1 ? locations[0] : locations,
        packages: hasAllPackagesSelected() ? 'all' : packages,
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
    packageMetric.textContent = hasAllPackagesSelected() ? 'all' : String(selectedPackages.length);
    responseTypeMetric.textContent = TEXT_FORMATS.has(tryFormat.value) ? 'text' : 'audio';
    wxBaseMetric.textContent = state.wxBase;
    wxBasePill.textContent = state.wxBase;
    generatePath.textContent = `${state.wxBase}/generate`;
    packagesPath.textContent = `${state.wxBase}/packages`;
}

function updatePackageSummary() {
    const selectedCount = getSelectedPackages().length;
    const totalCount = state.allPackages.length;
    if (!totalCount) {
        pkgSummary.textContent = 'Packages are unavailable.';
        return;
    }
    if (!selectedCount) {
        pkgSummary.textContent = 'Select at least one package to enable generate.';
        return;
    }
    if (selectedCount === totalCount) {
        pkgSummary.textContent = `All ${totalCount} packages selected.`;
        return;
    }
    pkgSummary.textContent = `${selectedCount} of ${totalCount} packages selected.`;
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
    pkgDefaultBtn.disabled = state.busy || state.allPackages.length === 0;
    pkgAllBtn.disabled = state.busy || state.allPackages.length === 0;
    pkgClearBtn.disabled = state.busy || getSelectedPackages().length === 0;
    updatePackageSummary();
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
    const defaults = DEFAULT_PACKAGES.filter((pkg) => state.allPackages.includes(pkg));
    setSelectedPackages(defaults.length ? defaults : state.allPackages.slice(0, 3));
}

function renderPackageChips() {
    tryPkgs.innerHTML = '';
    for (const pkg of state.allPackages) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = `pkg-chip${state.selectedPackages.has(pkg) ? ' active' : ''}`;
        chip.textContent = pkg.replace(/_/g, ' ');
        chip.title = PACKAGE_DESCRIPTIONS[pkg] || pkg;
        chip.disabled = state.busy;
        chip.setAttribute('aria-pressed', String(state.selectedPackages.has(pkg)));
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
    updatePackageSummary();
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

async function revealAudio(blob) {
    if (!blob.size) {
        throw new Error('The request returned no audio.');
    }
    state.objectUrl = URL.createObjectURL(blob);
    tryAudio.src = state.objectUrl;
    tryAudio.hidden = false;
    try {
        await tryAudio.play();
        return { message: 'Playing audio…', className: 'try-status ok' };
    } catch {
        return { message: 'Audio ready. Press play.', className: 'try-status warn' };
    }
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
        `<article class="wx-panel-response-card"><p>${escapeHtml(label)}</p><strong>${escapeHtml(value)}</strong></article>`
    )).join('');
}

function renderResponseError(message) {
    responseMeta.innerHTML = `<article class="wx-panel-response-card is-empty"><p>${escapeHtml(message)}</p></article>`;
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

async function loadHealth() {
    const response = await fetchWithAuth(`${state.apiBase}/health`);
    if (!response.ok) {
        throw new Error(await readError(response));
    }
    const health = await response.json();
    state.wxBase = health.wx_base || state.wxBase;
    const authState = health.auth_required ? 'panel auth enabled' : 'panel auth disabled';
    setHealth(true, `API healthy. WX base ${state.wxBase}. ${authState}.`, `Healthy • ${authState}`);
}

async function loadPackages() {
    const response = await fetchWithAuth(`${state.wxBase}/packages`);
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
        const response = await fetchWithAuth(`${state.wxBase}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: state.abortController.signal,
        });

        if (!response.ok) {
            throw new Error(await readError(response));
        }

        renderResponseHeaders(response);

        let completion = { message: 'Response ready.', className: 'try-status ok' };

        if (TEXT_FORMATS.has(payload.format)) {
            const text = await response.text();
            tryText.textContent = text;
            tryText.hidden = false;
            completion = { message: 'Text response ready.', className: 'try-status ok' };
        } else if (payload.format !== 'raw') {
            tryStatus.textContent = 'Downloading audio…';
            completion = await revealAudio(await response.blob());
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
            if (!totalBytes) {
                throw new Error('The request returned no audio.');
            }
            const pcm = new Uint8Array(totalBytes);
            let offset = 0;
            for (const chunk of chunks) {
                pcm.set(chunk, offset);
                offset += chunk.byteLength;
            }
            completion = await revealAudio(new Blob([pcmToWav(pcm, sampleRate, channels)], { type: 'audio/wav' }));
        }

        tryStatus.textContent = completion.message;
        tryStatus.className = completion.className;
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
    pkgDefaultBtn.addEventListener('click', () => {
        ensureDefaultPackages();
        renderPackageChips();
        updateRequestPreviews();
    });
    pkgAllBtn.addEventListener('click', () => {
        setSelectedPackages(state.allPackages);
        renderPackageChips();
        updateRequestPreviews();
    });
    pkgClearBtn.addEventListener('click', () => {
        state.selectedPackages.clear();
        renderPackageChips();
        updateRequestPreviews();
    });
    tryBtn.addEventListener('click', generate);
    tryStop.addEventListener('click', () => {
        if (state.abortController) {
            state.abortController.abort();
        }
        tryAudio.pause();
        tryStatus.textContent = 'Stopped.';
        tryStatus.className = 'try-status';
    });
}

async function boot() {
    if (window.lucide) {
        lucide.createIcons();
    }
    initTheme(themeToggle);
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
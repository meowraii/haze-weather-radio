import { apiCommand, apiGet } from './lib/api.js';
import { pcmToWav } from './lib/audio.js';

const PACKAGE_DESCRIPTIONS = {
    date_time: 'Date and time',
    current_conditions: 'Current conditions',
    forecast: 'Forecast',
    air_quality: 'Air quality',
    climate_summary: 'Climate summary',
    thunderstorm_outlook: 'Thunderstorm outlook',
    hydrometric: 'Hydrometric conditions',
    coastal_flood: 'Coastal flood guidance',
    hurricane_tracks: 'Hurricane tracks',
    precipitation_analysis: 'Precipitation analysis',
    geophysical_alert: 'Geophysical alert',
    alerts: 'Active alerts',
};

const FORMAT_INFO = {
    wav: { mime: 'audio/wav', type: 'audio', browser: true },
    mp3: { mime: 'audio/mpeg', type: 'audio', browser: true },
    opus: { mime: 'audio/ogg; codecs=opus', type: 'audio', browser: true },
    ogg: { mime: 'audio/ogg; codecs=vorbis', type: 'audio', browser: true },
    webm: { mime: 'audio/webm; codecs=opus', type: 'audio', browser: true },
    flac: { mime: 'audio/flac', type: 'audio', browser: true },
    aac: { mime: 'audio/aac', type: 'audio', browser: true },
    raw: { mime: 'audio/L16', type: 'audio', browser: false },
    ulaw: { mime: 'audio/basic', type: 'audio', browser: false },
    alaw: { mime: 'audio/x-alaw-basic', type: 'audio', browser: false },
    g722: { mime: 'audio/G722', type: 'audio', browser: false },
    json: { mime: 'application/json', type: 'text', browser: true },
    xml: { mime: 'application/xml', type: 'text', browser: true },
    ssml: { mime: 'application/ssml+xml', type: 'text', browser: true },
    html: { mime: 'text/html', type: 'text', browser: true },
    markdown: { mime: 'text/markdown', type: 'text', browser: true },
    latex: { mime: 'application/x-latex', type: 'text', browser: true },
};

const TEXT_FORMATS = new Set(['json', 'xml', 'ssml', 'html', 'markdown', 'latex']);
const DEFAULT_PACKAGES = ['date_time', 'current_conditions', 'forecast'];
const DEFAULT_WX_SUFFIX = '/wx-on-demand';

const state = {
    wxBase: `${detectApiBase()}${DEFAULT_WX_SUFFIX}`,
    readers: [],
    allPackages: [],
    selectedPackages: new Set(),
    objectUrl: null,
    busy: false,
    generation: 0,
};

const el = {
    apiDot: document.getElementById('apiDot'),
    wxDot: document.getElementById('wxDot'),
    healthPill: document.getElementById('healthPill'),
    wxBasePill: document.getElementById('wxBasePill'),
    statusBanner: document.getElementById('wxStatusBanner'),
    builderSummary: document.getElementById('wxBuilderSummary'),
    requestSummary: document.getElementById('wxRequestSummary'),
    responseSummary: document.getElementById('wxResponseSummary'),
    locations: document.getElementById('tryLocations'),
    source: document.getElementById('trySource'),
    lang: document.getElementById('tryLang'),
    voice: document.getElementById('tryVoice'),
    format: document.getElementById('tryFormat'),
    packages: document.getElementById('tryPkgs'),
    pkgDefaultBtn: document.getElementById('pkgDefaultBtn'),
    pkgAllBtn: document.getElementById('pkgAllBtn'),
    pkgClearBtn: document.getElementById('pkgClearBtn'),
    pkgSummary: document.getElementById('pkgSummary'),
    button: document.getElementById('tryBtn'),
    stop: document.getElementById('tryStop'),
    status: document.getElementById('tryStatus'),
    audio: document.getElementById('tryAudio'),
    text: document.getElementById('tryText'),
    apiRequestUrl: document.getElementById('apiRequestUrl'),
    copyRequestUrl: document.getElementById('copyRequestUrlBtn'),
    requestPreview: document.getElementById('requestBodyPreview'),
    curlPreview: document.getElementById('curlPreview'),
    responseMeta: document.getElementById('responseMeta'),
    readerCatalog: document.getElementById('readerCatalog'),
    formatCatalog: document.getElementById('formatCatalog'),
    generatePath: document.getElementById('generatePath'),
    packagesPath: document.getElementById('packagesPath'),
    readersPath: document.getElementById('readersPath'),
};

function detectApiBase() {
    const stored = sessionStorage.getItem('haze.api_base');
    if (stored) return stored;
    const match = window.location.pathname.match(/^(\/api\/[^/]+)\//);
    const base = match ? match[1] : '/api/v1';
    sessionStorage.setItem('haze.api_base', base);
    return base;
}

function escapeHtml(value) {
    const span = document.createElement('span');
    span.textContent = String(value ?? '');
    return span.innerHTML;
}

function splitLocations(value) {
    return String(value || '')
        .split(/[\n,]+/)
        .map((item) => item.trim())
        .filter(Boolean);
}

function selectedPackages() {
    return state.allPackages.filter((pkg) => state.selectedPackages.has(pkg));
}

function allPackagesSelected() {
    return state.allPackages.length > 0 && selectedPackages().length === state.allPackages.length;
}

function setSelectedPackages(packages) {
    state.selectedPackages.clear();
    for (const pkg of packages) {
        if (state.allPackages.includes(pkg)) state.selectedPackages.add(pkg);
    }
}

function payload() {
    const locations = splitLocations(el.locations.value);
    const body = {
        locations: locations.length === 1 ? locations[0] : locations,
        packages: allPackagesSelected() ? 'all' : selectedPackages(),
        lang: el.lang.value,
        format: el.format.value,
    };
    if (el.source.value) body.source = el.source.value;
    if (el.voice.value) body.reader_id = el.voice.value;
    return body;
}

function apiRequestUrl() {
    return new URL(`${state.wxBase}/generate`, window.location.origin).href;
}

function setHealth(ok, message, pill) {
    if (el.apiDot) el.apiDot.dataset.state = ok ? 'ok' : 'err';
    if (el.wxDot) el.wxDot.dataset.state = ok ? 'ok' : 'err';
    el.healthPill.textContent = pill;
    el.statusBanner.textContent = message;
    el.statusBanner.className = `status-banner ${ok ? 'ok' : 'err'}`;
}

function setBusy(busy, message = '') {
    state.busy = busy;
    if (message) el.status.textContent = message;
    el.stop.disabled = !busy;
    update();
}

function resetOutput() {
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = null;
    el.audio.pause();
    el.audio.removeAttribute('src');
    el.audio.hidden = true;
    el.text.hidden = true;
    el.text.textContent = '';
}

function responseCard(label, value) {
    return `<article class="wx-response-card"><p>${escapeHtml(label)}</p><strong>${escapeHtml(value)}</strong></article>`;
}

function renderResponseMeta(result = null, message = '') {
    if (!result) {
        el.responseMeta.innerHTML = `<article class="wx-response-card is-empty"><p>${escapeHtml(message || 'No request has been sent yet.')}</p></article>`;
        el.responseSummary.textContent = message || 'No request yet';
        return;
    }
    const rows = [
        ['Format', result.format || el.format.value],
        ['Reader', result.reader_id || 'automatic'],
        ['Language', result.language || el.lang.value],
        ['Packages', result.packages || selectedPackages().join(', ') || 'none'],
    ];
    if (result.sample_rate) rows.push(['Sample rate', `${result.sample_rate} Hz`]);
    if (result.channels) rows.push(['Channels', result.channels]);
    el.responseMeta.innerHTML = rows.map(([label, value]) => responseCard(label, value)).join('');
    el.responseSummary.textContent = result.text ? 'Text ready' : 'Audio ready';
}

async function revealAudio(bytes, result) {
    if (!bytes.length) throw new Error('The request returned no audio.');
    let blobBytes = bytes;
    let mime = FORMAT_INFO[result.format]?.mime || result.content_type || 'application/octet-stream';
    if (result.format === 'raw') {
        blobBytes = pcmToWav(bytes, result.sample_rate || 48000, result.channels || 1);
        mime = 'audio/wav';
    }
    state.objectUrl = URL.createObjectURL(new Blob([blobBytes], { type: mime }));
    el.audio.src = state.objectUrl;
    el.audio.hidden = false;
    try {
        await el.audio.play();
        return 'Playing audio.';
    } catch {
        return 'Audio ready. Press play.';
    }
}

function updatePackageSummary() {
    const count = selectedPackages().length;
    if (!state.allPackages.length) {
        el.pkgSummary.textContent = 'Package catalog is unavailable.';
    } else if (!count) {
        el.pkgSummary.textContent = 'Select at least one package.';
    } else if (count === state.allPackages.length) {
        el.pkgSummary.textContent = `All ${count} packages selected.`;
    } else {
        el.pkgSummary.textContent = `${count} of ${state.allPackages.length} packages selected.`;
    }
}

function updateCatalogs() {
    el.readerCatalog.innerHTML = state.readers.length
        ? state.readers.map((reader) => `<span>${escapeHtml(reader.id)} ${escapeHtml(reader.gender || '')} ${escapeHtml(reader.language || '')}</span>`).join('')
        : '<span>No readers found</span>';
    el.formatCatalog.innerHTML = Object.entries(FORMAT_INFO)
        .map(([id, info]) => `<span>${escapeHtml(id)} ${info.type === 'text' ? 'text' : info.mime.split(';')[0]}</span>`)
        .join('');
}

function updateEndpoints() {
    el.wxBasePill.textContent = state.wxBase;
    el.generatePath.textContent = `${state.wxBase}/generate`;
    el.packagesPath.textContent = `${state.wxBase}/packages`;
    el.readersPath.textContent = `${state.wxBase}/readers`;
}

function update() {
    const body = payload();
    const packageCount = selectedPackages().length;
    const responseType = TEXT_FORMATS.has(body.format) ? 'text' : 'audio';
    const locationLabel = splitLocations(el.locations.value)[0] || 'default location';
    el.builderSummary.textContent = `${locationLabel} - ${packageCount || 0} package${packageCount === 1 ? '' : 's'}`;
    el.requestSummary.textContent = `${responseType}, ${body.format}`;
    el.apiRequestUrl.textContent = apiRequestUrl();
    el.requestPreview.textContent = JSON.stringify(body, null, 2);
    const curl = [
        `curl -s -X POST ${apiRequestUrl()}`,
        `  -H 'Content-Type: application/json'`,
        `  -d '${JSON.stringify(body)}'`,
    ];
    if (!TEXT_FORMATS.has(body.format)) {
        curl.push(`  --output weather.${body.format === 'raw' ? 's16le' : body.format}`);
    }
    el.curlPreview.textContent = curl.join(' \\\n');
    el.button.textContent = responseType === 'text' ? 'Generate Text' : 'Generate & Play';
    el.button.disabled = state.busy || packageCount === 0;
    el.pkgDefaultBtn.disabled = state.busy || state.allPackages.length === 0;
    el.pkgAllBtn.disabled = state.busy || state.allPackages.length === 0;
    el.pkgClearBtn.disabled = state.busy || packageCount === 0;
    updatePackageSummary();
    updateEndpoints();
}

async function copyText(text) {
    if (navigator.clipboard?.writeText) {
        try {
            await navigator.clipboard.writeText(text);
            return;
        } catch {
            // Browser privacy settings can reject clipboard writes even on localhost.
        }
    }
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    textarea.style.top = '0';
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    const copied = document.execCommand('copy');
    textarea.remove();
    if (!copied) {
        throw new Error('copy command was rejected');
    }
}

function renderReaders() {
    const current = el.voice.value;
    el.voice.innerHTML = '<option value="">Automatic reader</option>' + state.readers.map((reader) => {
        const parts = [reader.id, reader.gender, reader.language, reader.voice_id].filter(Boolean);
        return `<option value="${escapeHtml(reader.id)}">${escapeHtml(parts.join(' - '))}</option>`;
    }).join('');
    if (current && state.readers.some((reader) => reader.id === current)) {
        el.voice.value = current;
    }
    updateCatalogs();
}

function packageColumnCount(count) {
    for (const cols of [4, 3, 2]) {
        if (count >= cols && count % cols === 0) return cols;
    }
    return Math.min(4, Math.max(1, count || 1));
}

function renderPackageChips() {
    el.packages.innerHTML = '';
    el.packages.style.setProperty('--pkg-cols', packageColumnCount(state.allPackages.length));
    for (const pkg of state.allPackages) {
        const label = document.createElement('label');
        label.className = `pkg-chip${state.selectedPackages.has(pkg) ? ' active' : ''}`;
        label.title = PACKAGE_DESCRIPTIONS[pkg] || pkg;

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = state.selectedPackages.has(pkg);
        checkbox.disabled = state.busy;
        checkbox.addEventListener('change', () => {
            if (state.selectedPackages.has(pkg)) {
                state.selectedPackages.delete(pkg);
            } else {
                state.selectedPackages.add(pkg);
            }
            renderPackageChips();
            update();
        });

        const text = document.createElement('span');
        text.textContent = pkg.replace(/_/g, ' ');
        label.append(checkbox, text);
        el.packages.appendChild(label);
    }
}

function selectDefaultPackages() {
    const defaults = DEFAULT_PACKAGES.filter((pkg) => state.allPackages.includes(pkg));
    setSelectedPackages(defaults.length ? defaults : state.allPackages.slice(0, 3));
}

async function loadHealth() {
    const health = await apiGet('/health');
    state.wxBase = health.wx_base || state.wxBase;
    const ready = health.capabilities?.wx_generate !== false;
    const auth = health.auth_required ? 'auth enabled' : 'auth disabled';
    setHealth(ready, ready ? `WX generator ready. ${auth}.` : `WX generator needs the event bridge. ${auth}.`, ready ? 'Ready' : 'Bridge offline');
}

async function loadPackages() {
    const result = await apiCommand('wx.packages');
    state.allPackages = Array.isArray(result.packages) && result.packages.length
        ? result.packages
        : Object.keys(PACKAGE_DESCRIPTIONS);
    selectDefaultPackages();
    renderPackageChips();
}

async function loadReaders() {
    const result = await apiCommand('wx.readers');
    state.readers = Array.isArray(result.readers) ? result.readers : [];
    renderReaders();
}

async function generate() {
    resetOutput();
    renderResponseMeta(null, 'Waiting for response.');
    const body = payload();
    const generation = ++state.generation;
    setBusy(true, 'Generating...');
    el.status.className = 'try-status';
    try {
        const result = await apiCommand('wx.generate', body, 120000);
        if (generation !== state.generation) return;
        renderResponseMeta(result);
        if (typeof result.text === 'string') {
            el.text.textContent = result.text;
            el.text.hidden = false;
            el.status.textContent = 'Text response ready.';
            el.status.className = 'try-status ok';
        } else if (typeof result.audio_base64 === 'string') {
            const bytes = Uint8Array.from(atob(result.audio_base64), (char) => char.charCodeAt(0));
            const message = await revealAudio(bytes, result);
            el.status.textContent = message;
            el.status.className = message.includes('Playing') ? 'try-status ok' : 'try-status warn';
        } else {
            throw new Error('Weather package generation returned no usable payload.');
        }
    } catch (error) {
        if (generation !== state.generation) return;
        el.status.textContent = error.message || 'Request failed.';
        el.status.className = 'try-status err';
        renderResponseMeta(null, error.message || 'Request failed.');
    } finally {
        if (generation === state.generation) setBusy(false);
    }
}

function bindEvents() {
    for (const item of [el.locations, el.source, el.lang, el.voice, el.format]) {
        item.addEventListener(item.tagName === 'TEXTAREA' ? 'input' : 'change', update);
    }
    el.pkgDefaultBtn.addEventListener('click', () => {
        selectDefaultPackages();
        renderPackageChips();
        update();
    });
    el.pkgAllBtn.addEventListener('click', () => {
        setSelectedPackages(state.allPackages);
        renderPackageChips();
        update();
    });
    el.pkgClearBtn.addEventListener('click', () => {
        state.selectedPackages.clear();
        renderPackageChips();
        update();
    });
    el.button.addEventListener('click', generate);
    el.copyRequestUrl.addEventListener('click', async () => {
        const text = apiRequestUrl();
        try {
            await copyText(text);
            el.copyRequestUrl.textContent = 'Copied';
            window.setTimeout(() => {
                el.copyRequestUrl.textContent = 'Copy';
            }, 1200);
        } catch {
            el.status.textContent = 'Unable to copy URL.';
            el.status.className = 'try-status err';
        }
    });
    el.stop.addEventListener('click', () => {
        state.generation += 1;
        el.audio.pause();
        el.status.textContent = 'Stopped.';
        el.status.className = 'try-status';
        setBusy(false);
    });
}

async function boot() {
    if (window.lucide) lucide.createIcons();
    bindEvents();
    updateCatalogs();
    renderResponseMeta();
    try {
        await loadHealth();
    } catch (error) {
        setHealth(false, `Panel API unavailable: ${error.message || 'request failed'}`, 'Unavailable');
    }
    await Promise.allSettled([loadPackages(), loadReaders()]);
    if (!state.allPackages.length) {
        state.allPackages = Object.keys(PACKAGE_DESCRIPTIONS);
        selectDefaultPackages();
        renderPackageChips();
    }
    update();
    el.status.textContent = 'Ready';
}

export function initWxView() {
    boot();
}

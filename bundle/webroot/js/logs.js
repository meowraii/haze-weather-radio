import { apiCommand } from './lib/api.js';

const maxRenderedCharacters = 2 * 1024 * 1024;
const catalogRefreshMilliseconds = 15000;

let initialized = false;
let files = [];
let selectedFile = '';
let nextOffset = null;
let paused = false;
let tailEpoch = 0;
let requestInFlight = false;
let pollTimer = null;
let lastCatalogRefresh = 0;

const byID = (id) => document.getElementById(id);

function setStatus(message, state = '') {
    const element = byID('logsStatus');
    element.textContent = message;
    element.dataset.state = state;
}

function formatBytes(value) {
    const bytes = Number(value || 0);
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KiB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GiB`;
}

function formatModified(value) {
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return 'unknown time';
    return parsed.toLocaleString();
}

function logViewActive() {
    return !document.hidden && document.querySelector('.view[data-view="logs"]')?.classList.contains('active');
}

function schedulePoll(delay = 1000) {
    if (pollTimer) window.clearTimeout(pollTimer);
    pollTimer = window.setTimeout(pollSelectedFile, delay);
}

function renderFileCatalog() {
    const select = byID('logsFileSelect');
    const previous = selectedFile || select.value;
    select.replaceChildren(...files.map((file) => {
        const option = document.createElement('option');
        option.value = file.name;
        option.textContent = `${file.name} (${formatBytes(file.size)})`;
        return option;
    }));
    if (files.some((file) => file.name === previous)) {
        select.value = previous;
        selectedFile = previous;
    } else {
        selectedFile = files[0]?.name || '';
        select.value = selectedFile;
    }
    select.disabled = files.length === 0;
    byID('logsFileCount').textContent = String(files.length);
    byID('logsSelectedMetric').textContent = selectedFile ? selectedFile.split('/').pop() : 'None';
}

async function refreshFiles({ beginIfChanged = true } = {}) {
    const previous = selectedFile;
    try {
        const payload = await apiCommand('logs.list', {}, 8000);
        files = Array.isArray(payload.files) ? payload.files : [];
        lastCatalogRefresh = Date.now();
        renderFileCatalog();
        if (!selectedFile) {
            byID('logsTailOutput').textContent = '';
            byID('logsTailMeta').textContent = 'No log files found.';
            setStatus('No readable log files are present in the logs folder.', 'pending');
            schedulePoll(3000);
            return;
        }
        if (beginIfChanged && previous !== selectedFile) beginTail();
    } catch (error) {
        setStatus(error.message || 'Unable to list log files.', 'err');
        schedulePoll(3000);
    }
}

function appendLogContent(content) {
    if (!content) return;
    const output = byID('logsTailOutput');
    const atBottom = output.scrollHeight - output.scrollTop - output.clientHeight < 48;
    let combined = output.textContent + content;
    if (combined.length > maxRenderedCharacters) {
        combined = `[Earlier output removed from this browser view]\n${combined.slice(-maxRenderedCharacters)}`;
    }
    output.textContent = combined;
    if (byID('logsAutoScroll').checked && atBottom) output.scrollTop = output.scrollHeight;
}

function beginTail() {
    tailEpoch += 1;
    nextOffset = null;
    byID('logsTailOutput').textContent = '';
    byID('logsTailMeta').textContent = selectedFile ? `Opening ${selectedFile}...` : 'Select a log file.';
    setStatus(selectedFile ? `Opening ${selectedFile}...` : 'Select a log file.', 'pending');
    schedulePoll(0);
}

async function pollSelectedFile() {
    if (!selectedFile || paused || !logViewActive()) {
        schedulePoll(750);
        return;
    }
    if (requestInFlight) {
        schedulePoll(250);
        return;
    }
    if (Date.now() - lastCatalogRefresh > catalogRefreshMilliseconds) {
        await refreshFiles({ beginIfChanged: false });
        if (!selectedFile) return;
    }
    const epoch = tailEpoch;
    const payload = { file: selectedFile };
    if (nextOffset !== null) payload.offset = nextOffset;
    requestInFlight = true;
    try {
        const result = await apiCommand('logs.tail', payload, 8000);
        if (epoch !== tailEpoch || result.file !== selectedFile) return;
        if (result.reset) {
            byID('logsTailOutput').textContent = '[Log was rotated or truncated. Tail restarted.]\n';
        }
        appendLogContent(String(result.content || ''));
        nextOffset = Number(result.next_offset || 0);
        byID('logsTailMeta').textContent = `${result.file} · ${formatBytes(result.size)} · modified ${formatModified(result.modified_at)}`;
        setStatus(paused ? 'Tail paused.' : `Following ${result.file}.`, paused ? 'pending' : 'ok');
        schedulePoll(result.caught_up ? 1000 : 50);
    } catch (error) {
        if (epoch !== tailEpoch) return;
        setStatus(error.message || 'Unable to read the selected log.', 'err');
        if (/no longer available/i.test(error.message || '')) {
            await refreshFiles();
        } else {
            schedulePoll(2500);
        }
    } finally {
        requestInFlight = false;
    }
}

function togglePause() {
    paused = !paused;
    const button = byID('logsPauseButton');
    button.textContent = paused ? 'Resume' : 'Pause';
    setStatus(paused ? 'Tail paused.' : `Following ${selectedFile}.`, paused ? 'pending' : 'ok');
    if (!paused) schedulePoll(0);
}

function bindEvents() {
    byID('logsFileSelect').addEventListener('change', (event) => {
        selectedFile = event.currentTarget.value;
        byID('logsSelectedMetric').textContent = selectedFile ? selectedFile.split('/').pop() : 'None';
        beginTail();
    });
    byID('logsRefreshButton').addEventListener('click', () => refreshFiles());
    byID('logsPauseButton').addEventListener('click', togglePause);
    byID('logsClearButton').addEventListener('click', () => {
        byID('logsTailOutput').textContent = '';
    });
    byID('logsBottomButton').addEventListener('click', () => {
        const output = byID('logsTailOutput');
        output.scrollTop = output.scrollHeight;
    });
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden) schedulePoll(0);
    });
    window.addEventListener('hashchange', () => schedulePoll(0));
}

export async function initLogsView() {
    if (initialized) return;
    initialized = true;
    bindEvents();
    await refreshFiles();
}

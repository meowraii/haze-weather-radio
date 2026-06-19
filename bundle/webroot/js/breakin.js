import { panelClient } from './lib/ws-client.js';

const statusBanner = document.getElementById('breakinStatusBanner');
const feedRows = document.getElementById('breakinFeedRows');
const selectedMetric = document.getElementById('breakinSelectedMetric');
const sourceMetric = document.getElementById('breakinSourceMetric');
const selectAllButton = document.getElementById('breakinSelectAll');
const selectNoneButton = document.getElementById('breakinSelectNone');
const micPanel = document.getElementById('breakinMicPanel');
const urlPanel = document.getElementById('breakinUrlPanel');
const titleInput = document.getElementById('breakinTitle');
const prerollSelect = document.getElementById('breakinPrerollSelect');
const startButton = document.getElementById('breakinStartMic');
const stopButton = document.getElementById('breakinStopMic');
const cancelButton = document.getElementById('breakinCancelMic');
const meterFill = document.getElementById('breakinMeterFill');
const recordState = document.getElementById('breakinRecordState');
const elapsedEl = document.getElementById('breakinElapsed');
const sentBytesEl = document.getElementById('breakinSentBytes');
const uploadFile = document.getElementById('breakinUploadFile');
const uploadButton = document.getElementById('breakinUploadPreroll');
const toneHz = document.getElementById('breakinToneHz');
const toneMs = document.getElementById('breakinToneMs');
const toneButton = document.getElementById('breakinGenerateTone');
const urlTitle = document.getElementById('breakinUrlTitle');
const streamUrl = document.getElementById('breakinStreamUrl');
const queueUrlButton = document.getElementById('breakinQueueUrl');

let bound = false;
let state = { feeds: [] };
let selectedFeeds = new Set();
let session = null;
let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let processorNode = null;
let flushTimer = null;
let elapsedTimer = null;
let pendingPCM = [];
let pendingBytes = 0;
let sendChain = Promise.resolve();

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

function setStatus(text, stateName = 'ok') {
    statusBanner.textContent = text;
    statusBanner.dataset.state = stateName;
}

function bytesLabel(bytes) {
    if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function elapsedLabel(startedAt) {
    if (!startedAt) return '00:00';
    const total = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
    const min = Math.floor(total / 60);
    const sec = total % 60;
    return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

function feedName(feed) {
    return feed.feed_name || feed.name || feed.feed_id || feed.id || 'feed';
}

function feedID(feed) {
    return feed.feed_id || feed.id || '';
}

function selectedFeedIDs() {
    return [...selectedFeeds].filter(Boolean);
}

function renderFeeds() {
    const feeds = Array.isArray(state.feeds) ? state.feeds : [];
    if (!feeds.length) {
        feedRows.innerHTML = '<tr><td colspan="5" class="breakin-empty">No feeds are available.</td></tr>';
        selectedMetric.textContent = '0';
        return;
    }
    for (const feed of feeds) {
        const id = feedID(feed);
        if (id && !selectedFeeds.size) selectedFeeds.add(id);
    }
    feedRows.innerHTML = feeds.map((feed) => {
        const id = feedID(feed);
        const queue = Array.isArray(feed.queue) ? feed.queue.length : 0;
        return `
            <tr>
                <td><input type="checkbox" data-breakin-feed="${escapeHtml(id)}" ${selectedFeeds.has(id) ? 'checked' : ''}></td>
                <td>${escapeHtml(feedName(feed))}<br><code>${escapeHtml(id)}</code></td>
                <td>${escapeHtml(feed.mode || 'unknown')}</td>
                <td>${escapeHtml(queue)}</td>
                <td>${escapeHtml(feed.current?.title || 'Idle')}</td>
            </tr>
        `;
    }).join('');
    feedRows.querySelectorAll('[data-breakin-feed]').forEach((checkbox) => {
        checkbox.addEventListener('change', () => {
            if (checkbox.checked) selectedFeeds.add(checkbox.dataset.breakinFeed);
            else selectedFeeds.delete(checkbox.dataset.breakinFeed);
            updateSelectedMetric();
        });
    });
    updateSelectedMetric();
}

function updateSelectedMetric() {
    selectedMetric.textContent = String(selectedFeedIDs().length);
}

async function loadFeeds() {
    const payload = await panelClient.command('playlist.state', {}, 8000);
    state = payload || { feeds: [] };
    renderFeeds();
    setStatus('Break-in controls ready.', 'ok');
}

async function loadPrerolls(selectedPath = '') {
    const payload = await panelClient.command('operator_breakin.prerolls', {}, 8000);
    const files = Array.isArray(payload.files) ? payload.files : [];
    const previous = selectedPath || prerollSelect.value;
    prerollSelect.innerHTML = '<option value="">None</option>' + files.map((file) => (
        `<option value="${escapeHtml(file.path)}">${escapeHtml(file.name || file.path)}</option>`
    )).join('');
    if (previous && files.some((file) => file.path === previous)) {
        prerollSelect.value = previous;
    }
}

function setSource(source) {
    const mic = source === 'mic';
    micPanel.classList.toggle('active', mic);
    urlPanel.classList.toggle('active', !mic);
    sourceMetric.textContent = mic ? 'Mic' : 'URL';
}

function requireFeeds() {
    const feedIDs = selectedFeedIDs();
    if (!feedIDs.length) {
        throw new Error('Select at least one target feed.');
    }
    return feedIDs;
}

function floatToPCM16(input) {
    const out = new Int16Array(input.length);
    for (let i = 0; i < input.length; i += 1) {
        const sample = Math.max(-1, Math.min(1, input[i]));
        out[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    }
    return new Uint8Array(out.buffer);
}

function appendPCM(bytes, level) {
    pendingPCM.push(bytes);
    pendingBytes += bytes.byteLength;
    meterFill.style.width = `${Math.min(100, Math.round(level * 100))}%`;
}

function drainPCM() {
    if (!pendingPCM.length) return null;
    const out = new Uint8Array(pendingBytes);
    let offset = 0;
    for (const chunk of pendingPCM) {
        out.set(chunk, offset);
        offset += chunk.byteLength;
    }
    pendingPCM = [];
    pendingBytes = 0;
    return out;
}

function bytesToBase64(bytes) {
    let binary = '';
    const step = 0x8000;
    for (let i = 0; i < bytes.length; i += step) {
        binary += String.fromCharCode(...bytes.subarray(i, i + step));
    }
    return btoa(binary);
}

function enqueueChunkSend(bytes) {
    if (!session || !bytes?.length) return;
    const sessionID = session.id;
    sendChain = sendChain.then(async () => {
        const result = await panelClient.command('operator_breakin.chunk', {
            session_id: sessionID,
            data: bytesToBase64(bytes),
        }, 12000);
        if (session?.id === sessionID) {
            session.sentBytes = Number(result.bytes || session.sentBytes || 0);
            sentBytesEl.textContent = bytesLabel(session.sentBytes);
        }
    });
}

function flushPCM() {
    const chunk = drainPCM();
    if (chunk) enqueueChunkSend(chunk);
}

function setRecordingUI(active) {
    startButton.disabled = active;
    stopButton.disabled = !active;
    cancelButton.disabled = !active;
    recordState.textContent = active ? 'Recording' : 'Idle';
}

async function startMic() {
    const feedIDs = requireFeeds();
    pendingPCM = [];
    pendingBytes = 0;
    sendChain = Promise.resolve();
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            channelCount: 1,
        },
    });
    audioContext = new AudioContext();
    mediaStream = stream;
    const started = await panelClient.command('operator_breakin.start', {
        feed_ids: feedIDs,
        title: titleInput.value.trim() || 'Operator Break-in',
        preroll_path: prerollSelect.value,
        sample_rate: Math.round(audioContext.sampleRate),
        channels: 1,
    }, 12000);
    session = {
        id: started.session_id,
        startedAt: Date.now(),
        sentBytes: 0,
    };
    sourceNode = audioContext.createMediaStreamSource(stream);
    processorNode = audioContext.createScriptProcessor(4096, 1, 1);
    processorNode.onaudioprocess = (event) => {
        if (!session) return;
        const input = event.inputBuffer.getChannelData(0);
        let peak = 0;
        for (let i = 0; i < input.length; i += 1) {
            peak = Math.max(peak, Math.abs(input[i]));
        }
        appendPCM(floatToPCM16(input), peak);
    };
    sourceNode.connect(processorNode);
    processorNode.connect(audioContext.destination);
    flushTimer = window.setInterval(flushPCM, 450);
    elapsedTimer = window.setInterval(() => {
        elapsedEl.textContent = elapsedLabel(session?.startedAt);
    }, 250);
    setRecordingUI(true);
    setStatus('Microphone break-in recording.', 'pending');
}

function stopCaptureNodes() {
    if (flushTimer) window.clearInterval(flushTimer);
    if (elapsedTimer) window.clearInterval(elapsedTimer);
    flushTimer = null;
    elapsedTimer = null;
    if (processorNode) processorNode.disconnect();
    if (sourceNode) sourceNode.disconnect();
    processorNode = null;
    sourceNode = null;
    if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
    if (audioContext) audioContext.close().catch(() => {});
    audioContext = null;
    meterFill.style.width = '0%';
}

async function stopMicAndQueue() {
    if (!session) return;
    const activeSession = session;
    stopCaptureNodes();
    flushPCM();
    await sendChain;
    setStatus('Finalizing priority queue item...', 'pending');
    const result = await panelClient.command('operator_breakin.finish', { session_id: activeSession.id }, 15000);
    session = null;
    setRecordingUI(false);
    await loadFeeds();
    setStatus(`Operator break-in queued for ${result?.item?.feed_ids?.length || selectedFeedIDs().length} feed(s).`, 'ok');
}

async function cancelMic() {
    if (!session) return;
    const activeSession = session;
    stopCaptureNodes();
    session = null;
    setRecordingUI(false);
    try {
        await panelClient.command('operator_breakin.cancel', { session_id: activeSession.id }, 8000);
    } finally {
        setStatus('Break-in recording cancelled.', 'warn');
    }
}

async function uploadPreroll() {
    const file = uploadFile.files?.[0];
    if (!file) throw new Error('Choose a WAV file first.');
    const raw = new Uint8Array(await file.arrayBuffer());
    const result = await panelClient.command('operator_breakin.upload_preroll', {
        name: file.name,
        data: bytesToBase64(raw),
    }, 20000);
    await loadPrerolls(result.path);
    setStatus('Preroll uploaded.', 'ok');
}

async function generateTone() {
    const result = await panelClient.command('operator_breakin.generate_tone', {
        frequency_hz: Number(toneHz.value || 1050),
        duration_ms: Number(toneMs.value || 850),
    }, 10000);
    await loadPrerolls(result.path);
    setStatus('Tone preroll generated.', 'ok');
}

async function queueURL() {
    const feedIDs = requireFeeds();
    const result = await panelClient.command('operator_breakin.url', {
        feed_ids: feedIDs,
        title: urlTitle.value.trim() || 'Operator Break-in Stream',
        audio_url: streamUrl.value.trim(),
    }, 15000);
    await loadFeeds();
    setStatus(`Stream queued for ${result.feed_ids?.length || feedIDs.length} feed(s).`, 'ok');
}

function bind() {
    selectAllButton.addEventListener('click', () => {
        for (const feed of state.feeds || []) {
            const id = feedID(feed);
            if (id) selectedFeeds.add(id);
        }
        renderFeeds();
    });
    selectNoneButton.addEventListener('click', () => {
        selectedFeeds.clear();
        renderFeeds();
    });
    document.querySelectorAll('input[name="breakinSource"]').forEach((input) => {
        input.addEventListener('change', () => setSource(input.value));
    });
    startButton.addEventListener('click', () => {
        startMic().catch((error) => {
            stopCaptureNodes();
            session = null;
            setRecordingUI(false);
            setStatus(error.message || 'Unable to start microphone break-in.', 'err');
        });
    });
    stopButton.addEventListener('click', () => {
        stopMicAndQueue().catch((error) => {
            setStatus(error.message || 'Unable to queue break-in audio.', 'err');
            setRecordingUI(false);
            session = null;
        });
    });
    cancelButton.addEventListener('click', () => {
        cancelMic().catch((error) => {
            setStatus(error.message || 'Unable to cancel break-in.', 'err');
        });
    });
    uploadButton.addEventListener('click', () => {
        uploadPreroll().catch((error) => setStatus(error.message || 'Unable to upload preroll.', 'err'));
    });
    toneButton.addEventListener('click', () => {
        generateTone().catch((error) => setStatus(error.message || 'Unable to generate tone.', 'err'));
    });
    queueUrlButton.addEventListener('click', () => {
        queueURL().catch((error) => setStatus(error.message || 'Unable to queue stream URL.', 'err'));
    });
    window.addEventListener('haze:admin-state', (event) => {
        if (event.detail?.playlist?.feeds) {
            state = event.detail.playlist;
            renderFeeds();
        }
    });
}

export function initBreakInView() {
    if (!bound) {
        bound = true;
        bind();
    }
    Promise.all([loadFeeds(), loadPrerolls()]).catch((error) => {
        setStatus(error.message || 'Unable to load break-in controls.', 'err');
    });
}

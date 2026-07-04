import { panelClient } from './lib/ws-client.js';

const statusBanner = document.getElementById('bulletinStatusBanner');
const tableBody = document.getElementById('bulletinTableBody');
const countMetric = document.getElementById('bulletinCountMetric');
const enabledMetric = document.getElementById('bulletinEnabledMetric');
const pathLabel = document.getElementById('bulletinPathLabel');
const addButton = document.getElementById('bulletinAddButton');
const saveButton = document.getElementById('bulletinSaveButton');
const importButton = document.getElementById('bulletinImportButton');
const exportAllButton = document.getElementById('bulletinExportAllButton');
const importFile = document.getElementById('bulletinImportFile');
const fields = {
    id: document.getElementById('bulletinID'),
    title: document.getElementById('bulletinTitle'),
    enabled: document.getElementById('bulletinEnabled'),
    start: document.getElementById('bulletinStart'),
    expire: document.getElementById('bulletinExpire'),
    scheduleMode: document.getElementById('bulletinScheduleMode'),
    hours: document.getElementById('bulletinHours'),
    days: document.getElementById('bulletinDays'),
    endCycle: document.getElementById('bulletinEndCycle'),
    contentType: document.getElementById('bulletinContentType'),
    textEn: document.getElementById('bulletinTextEn'),
    textFr: document.getElementById('bulletinTextFr'),
    audioURL: document.getElementById('bulletinAudioURL'),
    audioFile: document.getElementById('bulletinAudioFile'),
    audioUpload: document.getElementById('bulletinAudioUpload'),
    feeds: document.getElementById('bulletinFeeds'),
};

let bound = false;
let bulletins = [];
let selectedID = '';

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

function setStatus(text, state = 'ok') {
    statusBanner.textContent = text;
    statusBanner.dataset.state = state;
}

function csv(values) {
    return (values || []).map((value) => String(value || '').trim()).filter(Boolean).join(', ');
}

function splitList(value) {
    return String(value || '').split(/[,\n;]/).map((part) => part.trim()).filter(Boolean);
}

function datetimeLocal(value) {
    if (!value) return '';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    const pad = (number) => String(number).padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
}

function formatTime(value) {
    if (!value) return 'any';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function selected() {
    return bulletins.find((item) => item.id === selectedID) || bulletins[0] || null;
}

function readEditor() {
    return {
        id: fields.id.value.trim(),
        title: fields.title.value.trim(),
        enabled: fields.enabled.checked,
        start: fields.start.value,
        expire: fields.expire.value,
        schedule_mode: fields.scheduleMode.value,
        hours: splitList(fields.hours.value),
        days: splitList(fields.days.value),
        end_of_cycle: fields.endCycle.checked,
        content_type: fields.contentType.value,
        text_en_ca: fields.textEn.value.trim(),
        text_fr_ca: fields.textFr.value.trim(),
        audio_url: fields.audioURL.value.trim(),
        audio_file: fields.audioFile.value.trim(),
        feeds: splitList(fields.feeds.value),
    };
}

function updateSelectedFromEditor() {
    const current = selected();
    if (!current) return;
    const next = readEditor();
    const oldID = current.id;
    Object.assign(current, next);
    if (next.id && next.id !== oldID) {
        selectedID = next.id;
    }
}

function writeEditor(item) {
    const empty = !item;
    Object.values(fields).forEach((field) => {
        if (field !== fields.audioUpload) field.disabled = empty;
    });
    if (!item) {
        Object.entries(fields).forEach(([key, field]) => {
            if (!field || field === fields.audioUpload) return;
            if (field.type === 'checkbox') field.checked = key === 'endCycle';
            else field.value = key === 'scheduleMode' ? 'always' : (key === 'contentType' ? 'tts' : '');
        });
        updateContentVisibility();
        return;
    }
    fields.id.value = item.id || '';
    fields.title.value = item.title || '';
    fields.enabled.checked = item.enabled !== false;
    fields.start.value = datetimeLocal(item.start);
    fields.expire.value = datetimeLocal(item.expire);
    fields.scheduleMode.value = item.schedule_mode || 'always';
    fields.hours.value = csv(item.hours);
    fields.days.value = csv(item.days);
    fields.endCycle.checked = item.end_of_cycle !== false;
    fields.contentType.value = item.content_type || 'tts';
    fields.textEn.value = item.text_en_ca || item.text?.['en-CA'] || '';
    fields.textFr.value = item.text_fr_ca || item.text?.['fr-CA'] || '';
    fields.audioURL.value = item.audio_url || '';
    fields.audioFile.value = item.audio_file || '';
    fields.feeds.value = csv(item.feeds);
    updateContentVisibility();
}

function renderTable() {
    countMetric.textContent = String(bulletins.length);
    enabledMetric.textContent = String(bulletins.filter((item) => item.enabled !== false).length);
    if (!bulletins.length) {
        tableBody.innerHTML = '<tr><td colspan="7" class="panel-empty-cell">No bulletins configured.</td></tr>';
        writeEditor(null);
        return;
    }
    if (!selectedID || !bulletins.some((item) => item.id === selectedID)) {
        selectedID = bulletins[0].id;
    }
    tableBody.innerHTML = bulletins.map((item) => {
        const windowText = `${formatTime(item.start)} to ${formatTime(item.expire)}`;
        const schedule = item.schedule_mode === 'hours'
            ? `Hours ${csv(item.hours) || 'any'}`
            : (item.schedule_mode === 'days' ? `Days ${csv(item.days) || 'any'}` : 'All times');
        const content = item.content_type === 'audio'
            ? `Audio ${item.audio_file || item.audio_url || ''}`
            : 'TTS';
        const feeds = (item.feeds || []).length ? csv(item.feeds) : 'All feeds';
        const active = item.id === selectedID;
        const enabledState = item.enabled !== false ? 'enabled' : 'disabled';
        const contentState = item.content_type === 'audio' ? 'audio' : 'tts';
        return `
            <tr class="${active ? 'active' : ''}" data-bulletin-id="${escapeHtml(item.id)}" aria-selected="${active ? 'true' : 'false'}" tabindex="0">
                <td><input type="checkbox" ${item.enabled !== false ? 'checked' : ''} data-bulletin-toggle aria-label="Enable ${escapeHtml(item.title || item.id)}"> <span class="table-pill" data-state="${enabledState}">${enabledState}</span></td>
                <td class="bulletin-title-cell"><strong>${escapeHtml(item.title || item.id)}</strong><span>${escapeHtml(item.id)}</span></td>
                <td title="${escapeHtml(windowText)}">${escapeHtml(windowText)}</td>
                <td title="${escapeHtml(schedule)}">${escapeHtml(schedule)}</td>
                <td title="${escapeHtml(content)}"><span class="table-pill" data-state="${contentState}">${escapeHtml(item.content_type === 'audio' ? 'Audio' : 'TTS')}</span></td>
                <td title="${escapeHtml(feeds)}">${escapeHtml(feeds)}</td>
                <td>
                    <div class="bulletin-row-actions table-row-actions">
                        <button class="btn-action" type="button" data-bulletin-export>Export</button>
                        <button class="btn-danger" type="button" data-bulletin-delete>Delete</button>
                    </div>
                </td>
            </tr>
        `;
    }).join('');
    writeEditor(selected());
}

function updateContentVisibility() {
    const type = fields.contentType.value;
    document.querySelectorAll('[data-bulletin-content]').forEach((node) => {
        node.hidden = node.dataset.bulletinContent !== type;
    });
}

async function loadBulletins() {
    setStatus('Loading bulletins...', 'pending');
    const payload = await panelClient.command('bulletins.get', {}, 10000);
    bulletins = Array.isArray(payload.bulletins) ? payload.bulletins : [];
    if (pathLabel) pathLabel.textContent = payload.path || 'managed/configs/userBulletins.xml';
    renderTable();
    setStatus('Bulletins loaded.', 'ok');
}

async function saveBulletins() {
    updateSelectedFromEditor();
    setStatus('Saving bulletins...', 'pending');
    const payload = await panelClient.command('bulletins.save', { bulletins }, 15000);
    bulletins = Array.isArray(payload.bulletins) ? payload.bulletins : [];
    renderTable();
    setStatus('Bulletins saved. Restart or let the product renderer reload before the next package cycle if needed.', 'ok');
}

function addBulletin() {
    updateSelectedFromEditor();
    const id = `bulletin-${Date.now()}`;
    bulletins.unshift({
        id,
        title: 'New Bulletin',
        enabled: true,
        schedule_mode: 'always',
        end_of_cycle: true,
        content_type: 'tts',
        text_en_ca: '',
        text_fr_ca: '',
        feeds: [],
    });
    selectedID = id;
    renderTable();
}

function downloadText(filename, text) {
    const blob = new Blob([text], { type: 'application/xml' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename || 'userBulletins.xml';
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
}

async function exportBulletin(id = '') {
    await saveBulletins();
    const payload = await panelClient.command('bulletins.export', { id }, 10000);
    downloadText(payload.filename || 'userBulletins.xml', payload.xml || '');
    setStatus(id ? 'Bulletin XML exported.' : 'All bulletin XML exported.', 'ok');
}

async function importXML(file) {
    if (!file) return;
    const xml = await file.text();
    setStatus('Importing bulletin XML...', 'pending');
    const payload = await panelClient.command('bulletins.import', { xml }, 15000);
    bulletins = Array.isArray(payload.bulletins) ? payload.bulletins : [];
    renderTable();
    setStatus('Bulletin XML imported.', 'ok');
}

async function uploadAudio(file) {
    if (!file) return;
    const raw = await file.arrayBuffer();
    let binary = '';
    const bytes = new Uint8Array(raw);
    for (let index = 0; index < bytes.length; index += 1) {
        binary += String.fromCharCode(bytes[index]);
    }
    const payload = await panelClient.command('bulletins.upload_audio', {
        filename: file.name,
        audio_base64: btoa(binary),
    }, 20000);
    fields.audioFile.value = payload.path || '';
    updateSelectedFromEditor();
    renderTable();
    setStatus('Bulletin audio uploaded.', 'ok');
}

export function initBulletinsView() {
    if (bound) {
        loadBulletins().catch((error) => setStatus(error.message || 'Unable to load bulletins.', 'err'));
        return;
    }
    bound = true;
    addButton.addEventListener('click', addBulletin);
    saveButton.addEventListener('click', () => saveBulletins().catch((error) => setStatus(error.message || 'Unable to save bulletins.', 'err')));
    exportAllButton.addEventListener('click', () => exportBulletin('').catch((error) => setStatus(error.message || 'Unable to export bulletins.', 'err')));
    importButton.addEventListener('click', () => importFile.click());
    importFile.addEventListener('change', () => {
        importXML(importFile.files?.[0]).catch((error) => setStatus(error.message || 'Unable to import bulletins.', 'err'));
        importFile.value = '';
    });
    fields.audioUpload.addEventListener('change', () => {
        uploadAudio(fields.audioUpload.files?.[0]).catch((error) => setStatus(error.message || 'Unable to upload audio.', 'err'));
        fields.audioUpload.value = '';
    });
    fields.contentType.addEventListener('change', updateContentVisibility);
    Object.values(fields).forEach((field) => {
        if (field && field !== fields.audioUpload) {
            field.addEventListener('input', updateSelectedFromEditor);
            field.addEventListener('change', updateSelectedFromEditor);
        }
    });
    tableBody.addEventListener('click', (event) => {
        const row = event.target.closest('[data-bulletin-id]');
        if (!row) return;
        updateSelectedFromEditor();
        selectedID = row.dataset.bulletinId;
        if (event.target.closest('[data-bulletin-delete]')) {
            const removedID = selectedID;
            bulletins = bulletins.filter((item) => item.id !== selectedID);
            selectedID = bulletins[0]?.id || '';
            renderTable();
            setStatus(`Removed ${removedID}. Save to persist.`, 'pending');
            return;
        }
        if (event.target.closest('[data-bulletin-export]')) {
            exportBulletin(selectedID).catch((error) => setStatus(error.message || 'Unable to export bulletin.', 'err'));
            return;
        }
        const toggle = event.target.closest('[data-bulletin-toggle]');
        if (toggle) {
            const item = selected();
            if (item) item.enabled = toggle.checked;
            setStatus('Unsaved bulletin changes.', 'pending');
        }
        renderTable();
    });
    tableBody.addEventListener('keydown', (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        if (event.target.closest('button,input')) return;
        const row = event.target.closest('[data-bulletin-id]');
        if (!row) return;
        event.preventDefault();
        updateSelectedFromEditor();
        selectedID = row.dataset.bulletinId;
        renderTable();
    });
    loadBulletins().catch((error) => setStatus(error.message || 'Unable to load bulletins.', 'err'));
}

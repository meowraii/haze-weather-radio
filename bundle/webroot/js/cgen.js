import { panelClient } from './lib/ws-client.js';

const statusBanner = document.getElementById('cgenStatusBanner');
const pathLabel = document.getElementById('cgenPathLabel');
const countMetric = document.getElementById('cgenCountMetric');
const modeMetric = document.getElementById('cgenModeMetric');
const instanceSelect = document.getElementById('cgenInstanceSelect');
const addButton = document.getElementById('cgenAddButton');
const saveButton = document.getElementById('cgenSaveButton');
const globalEnabled = document.getElementById('cgenGlobalEnabled');
const preview = document.getElementById('cgenPreview');
const metaProgramInput = document.getElementById('cgenProgramInputMeta');
const metaPriority = document.getElementById('cgenPriorityMeta');
const metaOutput = document.getElementById('cgenOutputMeta');

const fields = {
    id: document.getElementById('cgenID'),
    name: document.getElementById('cgenName'),
    enabled: document.getElementById('cgenEnabled'),
    mode: document.getElementById('cgenMode'),
    programInput: document.getElementById('cgenProgramInput'),
    programInputFormat: document.getElementById('cgenProgramInputFormat'),
    priorityFeed: document.getElementById('cgenPriorityFeed'),
    priorityInput: document.getElementById('cgenPriorityInput'),
    programOutput: document.getElementById('cgenProgramOutput'),
    alertOutput: document.getElementById('cgenAlertOutput'),
    outputFormat: document.getElementById('cgenOutputFormat'),
    vcodec: document.getElementById('cgenVCodec'),
    acodec: document.getElementById('cgenACodec'),
    duckDB: document.getElementById('cgenDuckDB'),
    width: document.getElementById('cgenWidth'),
    height: document.getElementById('cgenHeight'),
    fps: document.getElementById('cgenFPS'),
    interlaced: document.getElementById('cgenInterlaced'),
    fieldOrder: document.getElementById('cgenFieldOrder'),
    standard: document.getElementById('cgenStandard'),
    backgroundColor: document.getElementById('cgenBackgroundColor'),
    font: document.getElementById('cgenFont'),
    fontSize: document.getElementById('cgenFontSize'),
    textX: document.getElementById('cgenTextX'),
    textY: document.getElementById('cgenTextY'),
    textColor: document.getElementById('cgenTextColor'),
    bannerX: document.getElementById('cgenBannerX'),
    bannerY: document.getElementById('cgenBannerY'),
    bannerWidth: document.getElementById('cgenBannerWidth'),
    bannerHeight: document.getElementById('cgenBannerHeight'),
    bannerColor: document.getElementById('cgenBannerColor'),
    bannerGradientColor: document.getElementById('cgenBannerGradientColor'),
    bannerMode: document.getElementById('cgenBannerMode'),
    bannerBackgroundEnabled: document.getElementById('cgenBannerBackgroundEnabled'),
    scrollSpeed: document.getElementById('cgenScrollSpeed'),
    text: document.getElementById('cgenText'),
    textEnabled: document.getElementById('cgenTextEnabled'),
    textFontSize: document.getElementById('cgenTextFontSize'),
    clockEnabled: document.getElementById('cgenClockEnabled'),
    clockFormat: document.getElementById('cgenClockFormat'),
    clockX: document.getElementById('cgenClockX'),
    clockY: document.getElementById('cgenClockY'),
    clockFontSize: document.getElementById('cgenClockFontSize'),
    clockColor: document.getElementById('cgenClockColor'),
    smpteBars: document.getElementById('cgenSmpteBars'),
};

let bound = false;
let cgenEnabled = true;
let feeds = [];
let selectedID = '';

function setStatus(text, state = 'ok') {
    statusBanner.textContent = text;
    statusBanner.dataset.state = state;
}

function selected() {
    return feeds.find((feed) => feed.id === selectedID) || feeds[0] || null;
}

function value(key, fallback = '') {
    const field = fields[key];
    if (!field) return fallback;
    if (field.type === 'checkbox') return field.checked;
    return String(field.value || fallback).trim();
}

function setValue(key, raw) {
    const field = fields[key];
    if (!field) return;
    if (field.type === 'checkbox') {
        field.checked = Boolean(raw);
    } else {
        field.value = raw ?? '';
    }
}

function sanitizeID(value) {
    const cleaned = String(value || '').trim().replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
    return cleaned || `cgen-${Date.now().toString(36)}`;
}

function readEditor() {
    const id = sanitizeID(value('id'));
    return {
        id,
        name: value('name', id),
        enabled: value('enabled'),
        mode: value('mode', 'release'),
        smpte_bars: value('smpteBars'),
        program_input_url: value('programInput'),
        program_input_format: value('programInputFormat', 'mpegts'),
        priority_feed_id: value('priorityFeed', id),
        priority_input_url: value('priorityInput'),
        priority_input_format: 'priority-audio',
        program_output_url: value('programOutput'),
        program_output_format: value('outputFormat', 'mpegts'),
        alert_output_url: value('alertOutput') || value('programOutput'),
        alert_output_format: value('outputFormat', 'mpegts'),
        vcodec: value('vcodec', 'mpeg2video'),
        acodec: value('acodec', 'ac3'),
        video_bitrate_kbps: selected()?.video_bitrate_kbps || '12000',
        audio_bitrate_kbps: selected()?.audio_bitrate_kbps || '192',
        width: value('width', '1920'),
        height: value('height', '1080'),
        fps: value('fps', '30000/1001'),
        interlaced: value('interlaced'),
        field_order: value('fieldOrder', 'tff'),
        standard: value('standard', 'atsc'),
        audio_idle: 'source',
        audio_alert_mode: 'replace',
        duck_db: value('duckDB', '-18'),
        banner_mode: value('bannerMode', 'auto'),
        ticker_height: value('bannerHeight', '128'),
        font: value('font', 'Arial'),
        font_size: value('fontSize', '58'),
        scroll_speed: value('scrollSpeed', '8'),
        background_color: value('backgroundColor', '#000000'),
        banner_background_color: value('bannerColor', '#b45309'),
        banner_background_gradient_color: value('bannerGradientColor', '#7f1d1d'),
        banner_background_enabled: value('bannerBackgroundEnabled'),
        banner_x: value('bannerX', '0'),
        banner_y: value('bannerY', '0'),
        banner_width: value('bannerWidth', value('width', '1920')),
        banner_height: value('bannerHeight', '128'),
        text_enabled: value('textEnabled'),
        text: fields.text.value,
        text_x: value('textX', '48'),
        text_y: value('textY', '128'),
        text_font_size: value('textFontSize', '58'),
        text_color: value('textColor', '#ffffff'),
        clock_enabled: value('clockEnabled'),
        clock_format: value('clockFormat', 'Jan 02 15:04:05'),
        clock_x: value('clockX', '48'),
        clock_y: value('clockY', '48'),
        clock_font_size: value('clockFontSize', '30'),
        clock_color: value('clockColor', '#ffffff'),
    };
}

function writeEditor(feed) {
    if (!feed) return;
    setValue('id', feed.id);
    setValue('name', feed.name || feed.id);
    setValue('enabled', Boolean(feed.enabled));
    setValue('mode', feed.mode || 'release');
    setValue('smpteBars', Boolean(feed.smpte_bars));
    setValue('programInput', feed.program_input_url || '');
    setValue('programInputFormat', feed.program_input_format || 'mpegts');
    setValue('priorityFeed', feed.priority_feed_id || feed.id);
    setValue('priorityInput', feed.priority_input_url || '');
    setValue('programOutput', feed.program_output_url || '');
    setValue('alertOutput', feed.alert_output_url || feed.program_output_url || '');
    setValue('outputFormat', feed.program_output_format || feed.alert_output_format || 'mpegts');
    setValue('vcodec', feed.vcodec || 'mpeg2video');
    setValue('acodec', feed.acodec || 'ac3');
    setValue('duckDB', feed.duck_db || '-18');
    setValue('width', feed.width || '1920');
    setValue('height', feed.height || '1080');
    setValue('fps', feed.fps || '30000/1001');
    setValue('interlaced', Boolean(feed.interlaced));
    setValue('fieldOrder', feed.field_order || 'tff');
    setValue('standard', feed.standard || 'atsc');
    setValue('backgroundColor', feed.background_color || '#000000');
    setValue('font', feed.font || 'Arial');
    setValue('fontSize', feed.font_size || '58');
    setValue('scrollSpeed', feed.scroll_speed || '8');
    setValue('textX', feed.text_x || '48');
    setValue('textY', feed.text_y || '128');
    setValue('textColor', feed.text_color || '#ffffff');
    setValue('bannerX', feed.banner_x || '0');
    setValue('bannerY', feed.banner_y || '0');
    setValue('bannerWidth', feed.banner_width || feed.width || '1920');
    setValue('bannerHeight', feed.banner_height || feed.ticker_height || '128');
    setValue('bannerColor', feed.banner_background_color || '#b45309');
    setValue('bannerGradientColor', feed.banner_background_gradient_color || '#7f1d1d');
    setValue('bannerMode', feed.banner_mode || 'auto');
    setValue('bannerBackgroundEnabled', feed.banner_background_enabled !== false);
    setValue('text', feed.text || '');
    setValue('textEnabled', Boolean(feed.text_enabled));
    setValue('textFontSize', feed.text_font_size || '58');
    setValue('clockEnabled', Boolean(feed.clock_enabled));
    setValue('clockFormat', feed.clock_format || 'Jan 02 15:04:05');
    setValue('clockX', feed.clock_x || '48');
    setValue('clockY', feed.clock_y || '48');
    setValue('clockFontSize', feed.clock_font_size || '30');
    setValue('clockColor', feed.clock_color || '#ffffff');
    renderPreview();
    renderMeta();
}

function upsertEditor() {
    const edited = readEditor();
    const index = feeds.findIndex((feed) => feed.id === selectedID);
    if (index >= 0) {
        feeds[index] = edited;
    } else {
        feeds.push(edited);
    }
    selectedID = edited.id;
}

function renderInstances() {
    instanceSelect.innerHTML = feeds.map((feed) => `<option value="${escapeHtml(feed.id)}">${escapeHtml(feed.name || feed.id)}</option>`).join('');
    if (selectedID && feeds.some((feed) => feed.id === selectedID)) {
        instanceSelect.value = selectedID;
    } else if (feeds[0]) {
        selectedID = feeds[0].id;
        instanceSelect.value = selectedID;
    }
    countMetric.textContent = String(feeds.length);
    modeMetric.textContent = selected()?.mode || 'release';
}

function renderMeta() {
    const feed = readEditor();
    metaProgramInput.textContent = feed.program_input_url || '-';
    metaPriority.textContent = feed.priority_feed_id || '-';
    metaOutput.textContent = feed.alert_output_url || feed.program_output_url || '-';
}

function drawSmpte(ctx, width, height) {
    const top = ['#c0c0c0', '#c0c000', '#00c0c0', '#00c000', '#c000c0', '#c00000', '#0000c0'];
    const topHeight = Math.floor(height * 0.67);
    top.forEach((color, index) => {
        ctx.fillStyle = color;
        ctx.fillRect(Math.floor(index * width / top.length), 0, Math.ceil(width / top.length), topHeight);
    });
    const mid = ['#0000c0', '#131313', '#c000c0', '#131313', '#00c0c0', '#131313', '#c0c0c0'];
    const midHeight = Math.floor(height * 0.12);
    mid.forEach((color, index) => {
        ctx.fillStyle = color;
        ctx.fillRect(Math.floor(index * width / mid.length), topHeight, Math.ceil(width / mid.length), midHeight);
    });
    const bottomY = topHeight + midHeight;
    const bottom = ['#00214c', '#ffffff', '#32006a', '#131313', '#090909', '#1d1d1d', '#090909', '#131313'];
    bottom.forEach((color, index) => {
        ctx.fillStyle = color;
        ctx.fillRect(Math.floor(index * width / bottom.length), bottomY, Math.ceil(width / bottom.length), height - bottomY);
    });
}

function renderPreview() {
    if (!preview) return;
    const ctx = preview.getContext('2d');
    const width = preview.width;
    const height = preview.height;
    const feed = readEditor();
    if (feed.smpte_bars) {
        drawSmpte(ctx, width, height);
    } else {
        ctx.fillStyle = feed.background_color || '#000000';
        ctx.fillRect(0, 0, width, height);
    }
    const sx = width / Number(feed.width || 1280);
    const sy = height / Number(feed.height || 720);
    if (feed.banner_background_enabled) {
        const bx = Number(feed.banner_x || 0) * sx;
        const by = Number(feed.banner_y || 0) * sy;
        const bw = Number(feed.banner_width || feed.width || 1280) * sx;
        const bh = Number(feed.banner_height || 128) * sy;
        const gradient = ctx.createLinearGradient(0, by, 0, by + bh);
        gradient.addColorStop(0, feed.banner_background_gradient_color || '#7f1d1d');
        gradient.addColorStop(0.5, feed.banner_background_color || '#b45309');
        gradient.addColorStop(1, feed.banner_background_gradient_color || '#7f1d1d');
        ctx.fillStyle = gradient;
        ctx.fillRect(bx, by, bw, bh);
    }
    ctx.fillStyle = feed.text_color || '#ffffff';
    ctx.font = `${Math.max(8, Number(feed.text_font_size || feed.font_size || 58) * sy)}px sans-serif`;
    if (feed.text_enabled && feed.text) {
        ctx.fillText(feed.text.slice(0, 80), Number(feed.text_x || 48) * sx, Number(feed.text_y || 96) * sy);
    }
    if (feed.clock_enabled) {
        ctx.fillStyle = feed.clock_color || '#ffffff';
        ctx.font = `${Math.max(8, Number(feed.clock_font_size || 30) * sy)}px monospace`;
        ctx.fillText(new Date().toLocaleString(), Number(feed.clock_x || 48) * sx, Number(feed.clock_y || 48) * sy);
    }
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

async function loadCgen() {
    const payload = await panelClient.command('cgen.get', {}, 10000);
    cgenEnabled = payload.enabled !== false;
    feeds = Array.isArray(payload.feeds) ? payload.feeds : [];
    if (pathLabel) pathLabel.textContent = payload.path || 'managed/configs/cgen.xml';
    globalEnabled.checked = cgenEnabled;
    if (!feeds.length) {
        feeds.push(defaultFeed());
    }
    renderInstances();
    writeEditor(selected());
    setStatus('CGEN config loaded.', 'ok');
}

async function saveCgen() {
    upsertEditor();
    const payload = await panelClient.command('cgen.save', { enabled: globalEnabled.checked, feeds }, 12000);
    cgenEnabled = payload.enabled !== false;
    feeds = Array.isArray(payload.feeds) ? payload.feeds : feeds;
    globalEnabled.checked = cgenEnabled;
    renderInstances();
    writeEditor(selected());
    setStatus('CGEN XML saved.', 'ok');
}

function defaultFeed() {
    return {
        id: 'CAP-IT-ALL',
        name: 'CAP-IT-ALL CGEN',
        enabled: true,
        mode: 'release',
        program_input_url: 'udp://239.0.0.1:9000?overrun_nonfatal=1&reuse=1',
        program_input_format: 'mpegts',
        priority_feed_id: '*',
        program_output_url: 'udp://239.0.0.2:9001?pkt_size=1316',
        alert_output_url: 'udp://239.0.0.2:9001?pkt_size=1316',
        program_output_format: 'mpegts',
        vcodec: 'mpeg2video',
        acodec: 'ac3',
        video_bitrate_kbps: '12000',
        audio_bitrate_kbps: '192',
        width: '1920',
        height: '1080',
        fps: '30000/1001',
        interlaced: true,
        field_order: 'tff',
        standard: 'atsc',
        background_color: '#000000',
        banner_background_color: '#b45309',
        banner_background_gradient_color: '#7f1d1d',
        banner_mode: 'auto',
        banner_background_enabled: true,
        scroll_speed: '8',
        banner_x: '0',
        banner_y: '0',
        banner_width: '1920',
        banner_height: '128',
        font: 'Arial',
        font_size: '58',
        text_x: '48',
        text_y: '128',
        text_font_size: '58',
        text_color: '#ffffff',
        clock_x: '48',
        clock_y: '48',
        clock_font_size: '30',
        clock_color: '#ffffff',
    };
}

async function runAction(action, extra = {}) {
    upsertEditor();
    const payload = await panelClient.command('cgen.action', { feed_id: selectedID, action, ...extra }, 10000);
    feeds = Array.isArray(payload.feeds) ? payload.feeds : feeds;
    renderInstances();
    writeEditor(feeds.find((feed) => feed.id === selectedID) || feeds[0]);
    setStatus(`CGEN action applied: ${action.replaceAll('_', ' ')}.`, 'ok');
}

function addInstance() {
    upsertEditor();
    const next = defaultFeed();
    next.id = sanitizeID(`cgen-${feeds.length + 1}`);
    next.name = `CGEN ${feeds.length + 1}`;
    feeds.push(next);
    selectedID = next.id;
    renderInstances();
    writeEditor(next);
    setStatus('New CGEN instance ready to edit.', 'pending');
}

export function initCgenView() {
    if (bound) return;
    bound = true;
    addButton.addEventListener('click', addInstance);
    saveButton.addEventListener('click', () => saveCgen().catch((error) => setStatus(error.message || 'Unable to save CGEN config.', 'err')));
    instanceSelect.addEventListener('change', () => {
        upsertEditor();
        selectedID = instanceSelect.value;
        writeEditor(selected());
    });
    Object.values(fields).forEach((field) => {
        field?.addEventListener('input', () => {
            renderPreview();
            renderMeta();
        });
        field?.addEventListener('change', () => {
            renderPreview();
            renderMeta();
        });
    });
    document.getElementById('cgenReleaseButton')?.addEventListener('click', () => runAction('release').catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenOverlayButton')?.addEventListener('click', () => runAction('overlay').catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenBarsButton')?.addEventListener('click', () => runAction('smpte_bars', { enabled: !fields.smpteBars.checked }).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenClockButton')?.addEventListener('click', () => runAction('clock', { enabled: !fields.clockEnabled.checked }).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenInsertTextButton')?.addEventListener('click', () => runAction('insert_text', { text: fields.text.value }).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenClearTextButton')?.addEventListener('click', () => runAction('clear_text').catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenInsertBannerButton')?.addEventListener('click', () => runAction('insert_banner_background', { color: fields.bannerColor.value, gradient_color: fields.bannerGradientColor.value }).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenClearBannerButton')?.addEventListener('click', () => runAction('clear_banner_background').catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    window.setInterval(renderPreview, 1000);
    loadCgen().catch((error) => setStatus(error.message || 'Unable to load CGEN config.', 'err'));
}

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
const metaRuntime = document.getElementById('cgenRuntimeMeta');
const metaDrift = document.getElementById('cgenDriftMeta');
const metaVisual = document.getElementById('cgenVisualMeta');

const fields = {
    id: document.getElementById('cgenID'),
    name: document.getElementById('cgenName'),
    enabled: document.getElementById('cgenEnabled'),
    mode: document.getElementById('cgenMode'),
    programInput: document.getElementById('cgenProgramInput'),
    programInputFormat: document.getElementById('cgenProgramInputFormat'),
    priorityFeed: document.getElementById('cgenPriorityFeed'),
    audioSource: document.getElementById('cgenAudioSource'),
    muteStandbyRoutine: document.getElementById('cgenMuteStandbyRoutine'),
    programOutput: document.getElementById('cgenProgramOutput'),
    outputFormat: document.getElementById('cgenOutputFormat'),
    vcodec: document.getElementById('cgenVCodec'),
    acodec: document.getElementById('cgenACodec'),
    hdBitrate: document.getElementById('cgenHdBitrate'),
    p720Enabled: document.getElementById('cgenP720Enabled'),
    p720Bitrate: document.getElementById('cgenP720Bitrate'),
    sdEnabled: document.getElementById('cgenSdEnabled'),
    sdBitrate: document.getElementById('cgenSdBitrate'),
    surroundEnabled: document.getElementById('cgenSurroundEnabled'),
    surroundBitrate: document.getElementById('cgenSurroundBitrate'),
    stereoEnabled: document.getElementById('cgenStereoEnabled'),
    stereoBitrate: document.getElementById('cgenStereoBitrate'),
    syncHardReset: document.getElementById('cgenSyncHardReset'),
    syncMaxAudioFrames: document.getElementById('cgenSyncMaxAudioFrames'),
    syncSourceBuffer: document.getElementById('cgenSyncSourceBuffer'),
    syncReconnectInitial: document.getElementById('cgenSyncReconnectInitial'),
    syncReconnectMax: document.getElementById('cgenSyncReconnectMax'),
    syncStatusInterval: document.getElementById('cgenSyncStatusInterval'),
    width: document.getElementById('cgenWidth'),
    height: document.getElementById('cgenHeight'),
    fps: document.getElementById('cgenFPS'),
    interlaced: document.getElementById('cgenInterlaced'),
    fieldOrder: document.getElementById('cgenFieldOrder'),
    standard: document.getElementById('cgenStandard'),
    font: document.getElementById('cgenFont'),
    fontWeight: document.getElementById('cgenFontWeight'),
    fontSize: document.getElementById('cgenFontSize'),
    bannerHeight: document.getElementById('cgenBannerHeight'),
    bannerMode: document.getElementById('cgenBannerMode'),
    scrollRepeatMode: document.getElementById('cgenScrollRepeatMode'),
    afterEomRepeats: document.getElementById('cgenAfterEomRepeats'),
    fixedRepeats: document.getElementById('cgenFixedRepeats'),
    standbyMode: document.getElementById('cgenStandbyMode'),
    standbyText: document.getElementById('cgenStandbyText'),
    standbyFontSize: document.getElementById('cgenStandbyFontSize'),
    standbyYPercent: document.getElementById('cgenStandbyYPercent'),
    scrollSpeed: document.getElementById('cgenScrollSpeed'),
    text: document.getElementById('cgenText'),
    textEnabled: document.getElementById('cgenTextEnabled'),
    textFontSize: document.getElementById('cgenTextFontSize'),
    clockEnabled: document.getElementById('cgenClockEnabled'),
    clockFormat: document.getElementById('cgenClockFormat'),
    clockX: document.getElementById('cgenClockX'),
    clockY: document.getElementById('cgenClockY'),
    clockFontSize: document.getElementById('cgenClockFontSize'),
    smpteBars: document.getElementById('cgenSmpteBars'),
};

let bound = false;
let cgenEnabled = true;
let feeds = [];
let selectedID = '';
let editorDirty = false;
let renderScheduled = false;
let previewDirty = false;
let metaDirty = false;

function setStatus(text, state = 'ok') {
    if (statusBanner.textContent !== text) {
        statusBanner.textContent = text;
    }
    statusBanner.dataset.state = state;
}

function setText(element, text) {
    if (element && element.textContent !== text) {
        element.textContent = text;
    }
}

function scheduleRender({ preview: needsPreview = true, meta: needsMeta = true } = {}) {
    previewDirty = previewDirty || needsPreview;
    metaDirty = metaDirty || needsMeta;
    if (renderScheduled) return;
    renderScheduled = true;
    window.requestAnimationFrame(() => {
        renderScheduled = false;
        if (previewDirty) {
            previewDirty = false;
            renderPreview();
        }
        if (metaDirty) {
            metaDirty = false;
            renderMeta();
        }
    });
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
        audio_source: value('audioSource', 'priority'),
        priority_input_format: 'priority-audio',
        program_output_url: value('programOutput'),
        program_output_format: value('outputFormat', 'mpegts'),
        vcodec: value('vcodec', 'mpeg2video'),
        acodec: value('acodec', 'ac3'),
        video_bitrate_kbps: value('hdBitrate', '12000'),
        audio_bitrate_kbps: value('stereoBitrate', '192'),
        hd_enabled: 'auto',
        hd_bitrate_kbps: value('hdBitrate', '12000'),
        p720_enabled: value('p720Enabled'),
        p720_bitrate_kbps: value('p720Bitrate', '8000'),
        sd_enabled: value('sdEnabled'),
        sd_bitrate_kbps: value('sdBitrate', '5000'),
        surround_enabled: value('surroundEnabled'),
        surround_bitrate_kbps: value('surroundBitrate', '384'),
        stereo_enabled: value('stereoEnabled'),
        stereo_bitrate_kbps: value('stereoBitrate', '192'),
        width: value('width', '1920'),
        height: value('height', '1080'),
        fps: value('fps', '30000/1001'),
        interlaced: value('interlaced'),
        field_order: value('fieldOrder', 'tff'),
        standard: value('standard', 'atsc'),
        audio_idle: 'source',
        audio_alert_mode: 'replace',
        mute_standby_routine: value('muteStandbyRoutine', true),
        banner_mode: value('bannerMode', 'auto'),
        ticker_height: value('bannerHeight', '128'),
        font: value('font', 'Arial'),
        font_weight: value('fontWeight', 'regular'),
        font_size: value('fontSize', '58'),
        scroll_speed: value('scrollSpeed', '8'),
        scroll_repeat_mode: value('scrollRepeatMode', 'until_audio_end'),
        after_eom_repeats: value('afterEomRepeats', '0'),
        fixed_repeats: value('fixedRepeats', '1'),
        banner_background_enabled: true,
        banner_height: value('bannerHeight', '128'),
        standby_mode: value('standbyMode', 'banner'),
        standby_text: value('standbyText', 'EAS Details Channel'),
        standby_font_size: value('standbyFontSize', value('fontSize', '58')),
        standby_y_percent: value('standbyYPercent', '10'),
        text_enabled: value('textEnabled'),
        text: fields.text.value,
        text_font_size: value('textFontSize', '58'),
        text_color: '#ffffff',
        clock_enabled: value('clockEnabled'),
        clock_format: value('clockFormat', 'Jan 02 15:04:05'),
        clock_x: value('clockX', '48'),
        clock_y: value('clockY', '48'),
        clock_font_size: value('clockFontSize', '30'),
        clock_color: '#ffffff',
        sync_hard_reset_ms: value('syncHardReset', '250'),
        sync_max_audio_frames_per_video: value('syncMaxAudioFrames', '8'),
        sync_source_buffer_ms: value('syncSourceBuffer', '240'),
        sync_reconnect_initial_ms: value('syncReconnectInitial', '500'),
        sync_reconnect_max_ms: value('syncReconnectMax', '10000'),
        sync_status_interval_ms: value('syncStatusInterval', '750'),
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
    setValue('audioSource', feed.audio_source || 'priority');
    setValue('muteStandbyRoutine', feed.mute_standby_routine !== false);
    setValue('programOutput', feed.program_output_url || '');
    setValue('outputFormat', feed.program_output_format || 'mpegts');
    setValue('vcodec', feed.vcodec || 'mpeg2video');
    setValue('acodec', feed.acodec || 'ac3');
    setValue('hdBitrate', feed.hd_bitrate_kbps || feed.video_bitrate_kbps || '12000');
    setValue('p720Enabled', Boolean(feed.p720_enabled));
    setValue('p720Bitrate', feed.p720_bitrate_kbps || '8000');
    setValue('sdEnabled', Boolean(feed.sd_enabled));
    setValue('sdBitrate', feed.sd_bitrate_kbps || '5000');
    setValue('surroundEnabled', feed.surround_enabled !== false);
    setValue('surroundBitrate', feed.surround_bitrate_kbps || '384');
    setValue('stereoEnabled', feed.stereo_enabled !== false);
    setValue('stereoBitrate', feed.stereo_bitrate_kbps || feed.audio_bitrate_kbps || '192');
    setValue('syncHardReset', feed.sync_hard_reset_ms || '250');
    setValue('syncMaxAudioFrames', feed.sync_max_audio_frames_per_video || '8');
    setValue('syncSourceBuffer', feed.sync_source_buffer_ms || '240');
    setValue('syncReconnectInitial', feed.sync_reconnect_initial_ms || '500');
    setValue('syncReconnectMax', feed.sync_reconnect_max_ms || '10000');
    setValue('syncStatusInterval', feed.sync_status_interval_ms || '750');
    setValue('width', feed.width || '1920');
    setValue('height', feed.height || '1080');
    setValue('fps', feed.fps || '30000/1001');
    setValue('interlaced', Boolean(feed.interlaced));
    setValue('fieldOrder', feed.field_order || 'tff');
    setValue('standard', feed.standard || 'atsc');
    setValue('font', feed.font || 'Arial');
    setValue('fontWeight', feed.font_weight || 'regular');
    setValue('fontSize', feed.font_size || '58');
    setValue('scrollSpeed', feed.scroll_speed || '8');
    setValue('scrollRepeatMode', feed.scroll_repeat_mode || 'until_audio_end');
    setValue('afterEomRepeats', feed.after_eom_repeats || '0');
    setValue('fixedRepeats', feed.fixed_repeats || '1');
    setValue('bannerHeight', feed.banner_height || feed.ticker_height || '128');
    setValue('bannerMode', feed.banner_mode || 'auto');
    setValue('standbyMode', feed.standby_mode || 'banner');
    setValue('standbyText', feed.standby_text || 'EAS Details Channel');
    setValue('standbyFontSize', feed.standby_font_size || feed.font_size || '58');
    setValue('standbyYPercent', feed.standby_y_percent || '10');
    setValue('text', feed.text || '');
    setValue('textEnabled', Boolean(feed.text_enabled));
    setValue('textFontSize', feed.text_font_size || '58');
    setValue('clockEnabled', Boolean(feed.clock_enabled));
    setValue('clockFormat', feed.clock_format || 'Jan 02 15:04:05');
    setValue('clockX', feed.clock_x || '48');
    setValue('clockY', feed.clock_y || '48');
    setValue('clockFontSize', feed.clock_font_size || '30');
    scheduleRender();
    editorDirty = false;
}

function upsertEditor() {
    const edited = readEditor();
    const index = feeds.findIndex((feed) => feed.id === selectedID);
    if (index >= 0) {
        edited.runtime = feeds[index].runtime || {};
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
    setText(countMetric, String(feeds.length));
    setText(modeMetric, selected()?.mode || 'release');
}

function renderMeta() {
    const feed = readEditor();
    const runtime = selected()?.runtime || {};
    const inputHealth = runtime.input_health && typeof runtime.input_health === 'object' ? runtime.input_health : {};
    const diagnostics = runtime.pipeline_diagnostics && typeof runtime.pipeline_diagnostics === 'object' ? runtime.pipeline_diagnostics : {};
    setText(metaProgramInput, feed.program_input_url || '-');
    const standbyMute = feed.mute_standby_routine === false ? 'standby routine live' : 'standby routine muted';
    setText(metaPriority, `${feed.priority_feed_id || feed.id || '-'} / ${feed.audio_source || 'priority'} / ${standbyMute}`);
    setText(metaOutput, feed.program_output_url || '-');
    if (metaRuntime) {
        const videoLive = runtime.input_video_connected === true || inputHealth.video_connected === true;
        const audioLive = runtime.input_audio_connected === true || inputHealth.audio_connected === true;
        const videoTimedOut = inputHealth.video_timed_out === true;
        const audioTimedOut = inputHealth.audio_timed_out === true;
        const connected = `${formatStreamHealth('video', videoLive, videoTimedOut, inputHealth.last_video_frame_age_ms || inputHealth.last_program_frame_age_ms)}, ${formatStreamHealth('audio', audioLive, audioTimedOut, inputHealth.last_audio_frame_age_ms)}`;
        const output = runtime.output_active === true ? 'output active' : 'output idle';
        const backend = runtime.media_backend || 'cgen';
        const gstState = runtime.gst_state ? `, ${runtime.gst_state}` : '';
        setText(metaRuntime, `${backend}: ${connected}, ${output}${gstState}`);
    }
    if (metaDrift) {
        setText(metaDrift, formatPipelineDiagnostics(diagnostics));
    }
    if (metaVisual) {
        const visual = runtime.visual_lifecycle || runtime.visual_mode || feed.mode || 'release';
        const video = runtime.video_selector || 'video?';
        const audio = runtime.audio_selector || 'audio?';
        setText(metaVisual, `${visual} / ${video} video / ${audio} audio`);
    }
}

function formatStreamHealth(label, live, timedOut, ageValue) {
    const state = live ? 'live' : timedOut ? 'timeout' : 'waiting';
    return `${label} ${state}${formatRuntimeAge(ageValue)}`;
}

function formatRuntimeAge(value) {
    const age = Number(value);
    if (!Number.isFinite(age)) return '';
    if (age < 1000) return ` ${Math.round(age)} ms`;
    return ` ${(age / 1000).toFixed(1)} s`;
}

function formatPipelineDiagnostics(diagnostics) {
    const parts = [];
    const warnings = Number(diagnostics.warning_count);
    const qos = Number(diagnostics.qos_count);
    const latency = Number(diagnostics.latency_recalculation_count);
    if (Number.isFinite(latency)) parts.push(`${latency} latency recalcs`);
    if (Number.isFinite(qos)) parts.push(`${qos} QoS`);
    if (Number.isFinite(warnings)) parts.push(`${warnings} warnings`);
    if (diagnostics.last_latency_error) parts.push(`latency: ${diagnostics.last_latency_error}`);
    if (diagnostics.last_qos && typeof diagnostics.last_qos === 'object') {
        const q = diagnostics.last_qos;
        const jitter = Number(q.jitter_ns);
        const proportion = Number(q.proportion);
        const qosParts = [];
        if (Number.isFinite(jitter)) qosParts.push(`${(jitter / 1000000).toFixed(1)} ms jitter`);
        if (Number.isFinite(proportion)) qosParts.push(`${proportion.toFixed(3)}x`);
        if (q.dropped) qosParts.push(`dropped ${q.dropped}`);
        if (qosParts.length) parts.push(`last QoS ${qosParts.join(' ')}`);
    }
    if (diagnostics.last_warning) parts.push(`warn: ${diagnostics.last_warning}`);
    return parts.join(', ') || '-';
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

function clampNumber(value, min, max, fallback) {
    const n = Number(value);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(max, Math.max(min, n));
}

function drawStandby(ctx, width, height, feed) {
    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, '#111827');
    gradient.addColorStop(0.5, '#000000');
    gradient.addColorStop(1, '#111827');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    const outputHeight = clampNumber(feed.height, 1, 4320, 1080);
    const sy = height / outputHeight;
    const fontSize = Math.max(8, clampNumber(feed.standby_font_size || feed.font_size, 8, 220, 58) * sy);
    const yPercent = clampNumber(feed.standby_y_percent, 0, 100, 10);
    const text = feed.standby_text || 'EAS Details Channel';
    const font = feed.font || 'Arial';
    ctx.font = `${canvasFontWeight(feed.font_weight, 'regular')} ${fontSize}px ${font}, Arial, sans-serif`;
    ctx.textBaseline = 'top';
    const x = Math.max(0, (width - ctx.measureText(text).width) / 2);
    const y = height * (yPercent / 100);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.85)';
    ctx.fillText(text, x + 2, y + 3);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, x, y);
}

function drawProgramPlaceholder(ctx, width, height, feed, runtime) {
    const live = runtime.input_connected === true && runtime.no_signal !== true;
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, live ? '#12324b' : '#111827');
    gradient.addColorStop(0.5, live ? '#1b5f7a' : '#050505');
    gradient.addColorStop(1, live ? '#14301f' : '#111827');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(255,255,255,.12)';
    ctx.lineWidth = 1;
    for (let y = 0; y < height; y += 18) {
        ctx.beginPath();
        ctx.moveTo(0, y + 0.5);
        ctx.lineTo(width, y + 0.5);
        ctx.stroke();
    }
    ctx.fillStyle = 'rgba(0,0,0,.42)';
    ctx.fillRect(14, 14, Math.min(width - 28, 420), 74);
    ctx.fillStyle = '#ffffff';
    ctx.font = `700 ${Math.max(13, width / 38)}px Arial, sans-serif`;
    ctx.textBaseline = 'top';
    ctx.fillText(live ? 'PROGRAM INPUT LIVE' : 'NO PROGRAM INPUT', 28, 26);
    ctx.font = `500 ${Math.max(10, width / 58)}px Arial, sans-serif`;
    const label = `${feed.width || 1920}x${feed.height || 1080} ${feed.interlaced ? 'interlaced' : 'progressive'}  ${feed.vcodec || 'mpeg2video'} / ${feed.acodec || 'ac3'}`;
    ctx.fillText(label, 28, 58);
}

function drawCompositorTicker(ctx, width, height, feed, runtime, text) {
    const outputWidth = clampNumber(feed.width, 1, 7680, 1920);
    const outputHeight = clampNumber(feed.height, 1, 4320, 1080);
    const sx = width / outputWidth;
    const sy = height / outputHeight;
    const y = clampNumber(runtime.ticker_y, 0, outputHeight, Math.round(outputHeight * 0.08)) * sy;
    const h = clampNumber(runtime.ticker_height ?? feed.banner_height, 24, outputHeight, 128) * sy;
    const gradientStops = Array.isArray(runtime.ticker_gradient) && runtime.ticker_gradient.length >= 3
        ? runtime.ticker_gradient
        : ['#111827', runtime.ticker_color || '#019310', '#111827'];
    const gradient = ctx.createLinearGradient(0, y, 0, y + h);
    gradient.addColorStop(0, gradientStops[0]);
    gradient.addColorStop(0.5, gradientStops[1]);
    gradient.addColorStop(1, gradientStops[2]);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, y, width, h);
    const fontSize = Math.max(10, clampNumber(runtime.font_size ?? feed.font_size, 8, 220, 58) * sy);
    const font = runtime.font || feed.font || 'Arial';
    ctx.font = `${canvasFontWeight(runtime.font_weight || feed.font_weight, 'regular')} ${fontSize}px ${font}, Arial, sans-serif`;
    ctx.textBaseline = 'middle';
    const textY = y + h / 2;
    ctx.fillStyle = 'rgba(0,0,0,.9)';
    ctx.fillText(text, width * 0.03 + 3, textY + 4);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, width * 0.03, textY);
}

function renderPreview() {
    if (!preview) return;
    const ctx = preview.getContext('2d');
    const width = preview.width;
    const height = preview.height;
    const feed = readEditor();
    const runtime = selected()?.runtime || {};
    const liveNoSignal = runtime.no_signal === true || runtime.input_connected === false || runtime.visual_lifecycle === 'standby';
    const visualMode = String(runtime.visual_mode || '').toLowerCase();
    const activeText = String(runtime.overlay_text || '').trim();
    const isSmpte = feed.smpte_bars || visualMode === 'smpte' || (!runtime.input_connected && feed.standby_mode === 'smpte');
    if (isSmpte) {
        drawSmpte(ctx, width, height);
    } else if (liveNoSignal && activeText) {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, width, height);
    } else if (liveNoSignal && !activeText && feed.standby_mode === 'black') {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, width, height);
    } else if (liveNoSignal && !activeText) {
        drawStandby(ctx, width, height, feed);
    } else {
        drawProgramPlaceholder(ctx, width, height, feed, runtime);
    }
    if (activeText) {
        drawCompositorTicker(ctx, width, height, feed, runtime, activeText);
    }
    ctx.fillStyle = '#ffffff';
    const sx = width / Number(feed.width || 1280);
    const sy = height / Number(feed.height || 720);
    ctx.font = `${Math.max(8, Number(feed.text_font_size || feed.font_size || 58) * sy)}px sans-serif`;
    if (feed.text_enabled && feed.text && !activeText) {
        ctx.fillText(feed.text.slice(0, 80), 48 * sx, 96 * sy);
    }
    if (feed.clock_enabled) {
        ctx.fillStyle = '#ffffff';
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

async function refreshRuntime() {
    const payload = await panelClient.command('cgen.get', {}, 10000);
    const latest = Array.isArray(payload.feeds) ? payload.feeds : [];
    for (const next of latest) {
        const index = feeds.findIndex((feed) => feed.id === next.id);
        if (index >= 0) {
            feeds[index].runtime = next.runtime || {};
            if (!editorDirty && next.id === selectedID) {
                feeds[index] = next;
            }
        }
    }
    if (!editorDirty) {
        renderInstances();
    }
    scheduleRender();
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
        program_input_url: 'udp://239.0.0.1:9000?fifo_size=2000000&overrun_nonfatal=1&reuse=1&buffer_size=1048576',
        program_input_format: 'mpegts',
        priority_feed_id: '*',
        audio_source: 'priority',
        mute_standby_routine: true,
        program_output_url: 'udp://239.0.0.2:9001?pkt_size=1316&buffer_size=1048576&reuse=1',
        program_output_format: 'mpegts',
        vcodec: 'mpeg2video',
        acodec: 'ac3',
        video_bitrate_kbps: '12000',
        audio_bitrate_kbps: '192',
        hd_enabled: 'auto',
        hd_bitrate_kbps: '12000',
        p720_enabled: false,
        p720_bitrate_kbps: '8000',
        sd_enabled: false,
        sd_bitrate_kbps: '5000',
        surround_enabled: true,
        surround_bitrate_kbps: '384',
        stereo_enabled: true,
        stereo_bitrate_kbps: '192',
        sync_hard_reset_ms: '250',
        sync_max_audio_frames_per_video: '8',
        sync_source_buffer_ms: '240',
        sync_reconnect_initial_ms: '500',
        sync_reconnect_max_ms: '10000',
        sync_status_interval_ms: '750',
        width: '1920',
        height: '1080',
        fps: '30000/1001',
        interlaced: true,
        field_order: 'tff',
        standard: 'atsc',
        banner_mode: 'auto',
        scroll_speed: '8',
        scroll_repeat_mode: 'until_audio_end',
        after_eom_repeats: '0',
        fixed_repeats: '1',
        banner_height: '128',
        standby_mode: 'banner',
        standby_text: 'EAS Details Channel',
        standby_font_size: '58',
        standby_y_percent: '10',
        font: 'Arial',
        font_weight: 'regular',
        font_size: '58',
        text_font_size: '58',
        clock_x: '48',
        clock_y: '48',
        clock_font_size: '30',
    };
}

function canvasFontWeight(value, fallback = 'regular') {
    switch (String(value || fallback).trim().toLowerCase()) {
        case 'thin':
        case '100':
            return '100';
        case 'extra-light':
        case 'extralight':
        case '200':
            return '200';
        case 'light':
        case '300':
            return '300';
        case 'medium':
        case '500':
            return '500';
        case 'semibold':
        case 'semi-bold':
        case '600':
            return '600';
        case 'bold':
        case '700':
            return '700';
        case 'extra-bold':
        case 'extrabold':
        case '800':
            return '800';
        case 'black':
        case '900':
            return '900';
        default:
            return '400';
    }
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
            editorDirty = true;
            scheduleRender();
        });
        field?.addEventListener('change', () => {
            editorDirty = true;
            scheduleRender();
        });
    });
    document.getElementById('cgenReleaseButton')?.addEventListener('click', () => runAction('release').catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenOverlayButton')?.addEventListener('click', () => runAction('overlay').catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenBarsButton')?.addEventListener('click', () => runAction('smpte_bars', { enabled: !fields.smpteBars.checked }).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenClockButton')?.addEventListener('click', () => runAction('clock', { enabled: !fields.clockEnabled.checked }).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenInsertTextButton')?.addEventListener('click', () => runAction('insert_text', { text: fields.text.value }).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenClearTextButton')?.addEventListener('click', () => runAction('clear_text').catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    window.setInterval(() => {
        refreshRuntime().catch(() => scheduleRender({ preview: true, meta: false }));
    }, 1500);
    loadCgen().catch((error) => setStatus(error.message || 'Unable to load CGEN config.', 'err'));
}

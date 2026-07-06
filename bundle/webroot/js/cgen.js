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
const previewStream = document.getElementById('cgenPreviewStream');
const metaProgramInput = document.getElementById('cgenProgramInputMeta');
const metaPriority = document.getElementById('cgenPriorityMeta');
const metaOutput = document.getElementById('cgenOutputMeta');
const metaRuntime = document.getElementById('cgenRuntimeMeta');
const metaDrift = document.getElementById('cgenDriftMeta');
const metaVisual = document.getElementById('cgenVisualMeta');
const sunnyField = document.getElementById('cgenSunnyField');
const sunnyButton = document.getElementById('cgenSunnyButton');
const refreshFontsButton = document.getElementById('cgenRefreshFontsButton');
const fontPreview = document.getElementById('cgenFontPreview');
const fontPicker = document.getElementById('cgenFontPicker');
const fontPickerLabel = document.getElementById('cgenFontPickerLabel');
const fontMenu = document.getElementById('cgenFontMenu');
const cgenTabs = Array.from(document.querySelectorAll('[data-cgen-tab]'));
const cgenTabPanels = Array.from(document.querySelectorAll('[data-cgen-panel]'));
const cgenInputOptionRows = Array.from(document.querySelectorAll('[data-cgen-input-option]'));

const fields = {
    id: document.getElementById('cgenID'),
    name: document.getElementById('cgenName'),
    enabled: document.getElementById('cgenEnabled'),
    mode: document.getElementById('cgenMode'),
    programInputType: document.getElementById('cgenProgramInputType'),
    programInput: document.getElementById('cgenProgramInput'),
    programInputFormat: document.getElementById('cgenProgramInputFormat'),
    hardwareDecoderEnabled: document.getElementById('cgenHardwareDecoderEnabled'),
    hardwareDecoder: document.getElementById('cgenHardwareDecoder'),
    browserUrl: document.getElementById('cgenBrowserUrl'),
    browserAutoSize: document.getElementById('cgenBrowserAutoSize'),
    browserWidth: document.getElementById('cgenBrowserWidth'),
    browserHeight: document.getElementById('cgenBrowserHeight'),
    browserFPS: document.getElementById('cgenBrowserFPS'),
    priorityFeed: document.getElementById('cgenPriorityFeed'),
    audioSource: document.getElementById('cgenAudioSource'),
    audioIdle: document.getElementById('cgenAudioIdle'),
    muteStandbyRoutine: document.getElementById('cgenMuteStandbyRoutine'),
    programOutput: document.getElementById('cgenProgramOutput'),
    outputFormat: document.getElementById('cgenOutputFormat'),
    vcodec: document.getElementById('cgenVCodec'),
    acodec: document.getElementById('cgenACodec'),
    hdBitrate: document.getElementById('cgenHdBitrate'),
    videoGop: document.getElementById('cgenVideoGop'),
    videoBFrames: document.getElementById('cgenVideoBFrames'),
    videoPreset: document.getElementById('cgenVideoPreset'),
    videoTune: document.getElementById('cgenVideoTune'),
    videoProfile: document.getElementById('cgenVideoProfile'),
    videoLevel: document.getElementById('cgenVideoLevel'),
    audioEncoderBitrate: document.getElementById('cgenAudioEncoderBitrate'),
    audioProfile: document.getElementById('cgenAudioProfile'),
    audioLevel: document.getElementById('cgenAudioLevel'),
    p720Enabled: document.getElementById('cgenP720Enabled'),
    p720Bitrate: document.getElementById('cgenP720Bitrate'),
    sdEnabled: document.getElementById('cgenSdEnabled'),
    sdBitrate: document.getElementById('cgenSdBitrate'),
    surroundEnabled: document.getElementById('cgenSurroundEnabled'),
    surroundBitrate: document.getElementById('cgenSurroundBitrate'),
    stereoEnabled: document.getElementById('cgenStereoEnabled'),
    stereoBitrate: document.getElementById('cgenStereoBitrate'),
    serviceName: document.getElementById('cgenServiceName'),
    providerName: document.getElementById('cgenProviderName'),
    serviceID: document.getElementById('cgenServiceID'),
    transportStreamID: document.getElementById('cgenTransportStreamID'),
    hdProgram: document.getElementById('cgenHdProgram'),
    hdVideoPID: document.getElementById('cgenHdVideoPID'),
    hdPmtPID: document.getElementById('cgenHdPmtPID'),
    p720Program: document.getElementById('cgenP720Program'),
    p720VideoPID: document.getElementById('cgenP720VideoPID'),
    p720PmtPID: document.getElementById('cgenP720PmtPID'),
    sdProgram: document.getElementById('cgenSdProgram'),
    sdVideoPID: document.getElementById('cgenSdVideoPID'),
    sdPmtPID: document.getElementById('cgenSdPmtPID'),
    stereoProgram: document.getElementById('cgenStereoProgram'),
    stereoAudioPID: document.getElementById('cgenStereoAudioPID'),
    stereoPmtPID: document.getElementById('cgenStereoPmtPID'),
    surroundProgram: document.getElementById('cgenSurroundProgram'),
    surroundAudioPID: document.getElementById('cgenSurroundAudioPID'),
    surroundPmtPID: document.getElementById('cgenSurroundPmtPID'),
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
    sunnyCat: document.getElementById('cgenSunnyCat'),
};

let bound = false;
let cgenEnabled = true;
let feeds = [];
let selectedID = '';
let editorDirty = false;
let renderScheduled = false;
let previewDirty = false;
let metaDirty = false;
let previewStreamFeedID = '';
let previewRetryTimer = 0;
let cgenCatalog = {
    formats: [],
    video_codecs: [],
    audio_codecs: [],
    video_decoders: [],
    browser_sources: [],
    fonts: [],
};
const managedFontFaces = new Map();

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

function setCgenTab(tabID) {
    const activeID = cgenTabs.some((tab) => tab.dataset.cgenTab === tabID)
        ? tabID
        : (cgenTabs[0]?.dataset.cgenTab || '');
    for (const tab of cgenTabs) {
        const active = tab.dataset.cgenTab === activeID;
        tab.classList.toggle('active', active);
        tab.setAttribute('aria-selected', active ? 'true' : 'false');
        tab.tabIndex = active ? 0 : -1;
    }
    for (const panel of cgenTabPanels) {
        panel.hidden = panel.dataset.cgenPanel !== activeID;
    }
}

function bindCgenTabs() {
    if (!cgenTabs.length || !cgenTabPanels.length) return;
    for (const tab of cgenTabs) {
        tab.addEventListener('click', () => setCgenTab(tab.dataset.cgenTab || ''));
        tab.addEventListener('keydown', (event) => {
            if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
            event.preventDefault();
            const current = cgenTabs.indexOf(tab);
            let next = current;
            if (event.key === 'Home') {
                next = 0;
            } else if (event.key === 'End') {
                next = cgenTabs.length - 1;
            } else {
                const delta = event.key === 'ArrowRight' ? 1 : -1;
                next = (current + delta + cgenTabs.length) % cgenTabs.length;
            }
            const nextTab = cgenTabs[next];
            setCgenTab(nextTab.dataset.cgenTab || '');
            nextTab.focus();
        });
    }
    setCgenTab(cgenTabs.find((tab) => tab.getAttribute('aria-selected') === 'true')?.dataset.cgenTab);
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

function inputTypeValue() {
    return String(fields.programInputType?.value || 'stream').trim() === 'browser' ? 'browser' : 'stream';
}

function browserSourceAvailable() {
    return Array.isArray(cgenCatalog.browser_sources)
        && cgenCatalog.browser_sources.some((entry) => String(entry?.id || entry?.element || '').trim() === 'cefsrc');
}

function cleanBrowserFPS(value) {
    const n = Math.trunc(Number(value));
    if (!Number.isFinite(n) || n <= 0) return '0';
    return String(Math.min(120, Math.max(5, n)));
}

function cleanDimension(value, fallback, max) {
    const n = Math.trunc(Number(value));
    if (!Number.isFinite(n) || n < 1) return fallback;
    return String(Math.min(max, n));
}

function updateProgramInputVisibility() {
    const browserOption = fields.programInputType?.querySelector('option[value="browser"]');
    if (browserOption) browserOption.disabled = !browserSourceAvailable();
    if (!browserSourceAvailable() && fields.programInputType?.value === 'browser') {
        fields.programInputType.value = 'stream';
    }
    const type = inputTypeValue();
    for (const row of cgenInputOptionRows) {
        row.hidden = row.dataset.cgenInputOption !== type;
    }
    const hardwareDecoderEnabled = type === 'stream' && fields.hardwareDecoderEnabled?.checked === true;
    if (fields.hardwareDecoder) fields.hardwareDecoder.disabled = !hardwareDecoderEnabled;
    const manualSize = type === 'browser' && fields.browserAutoSize?.checked === false;
    if (fields.browserWidth) fields.browserWidth.disabled = !manualSize;
    if (fields.browserHeight) fields.browserHeight.disabled = !manualSize;
}

function optionLabelForValue(value) {
    const text = String(value || '').trim();
    return text ? `${text} (current/custom)` : '';
}

function ensureSelectOption(select, value, label = '') {
    if (!select || select.tagName !== 'SELECT') return;
    const text = String(value ?? '').trim();
    if (!text) return;
    const exists = Array.from(select.options).some((option) => option.value === text);
    if (exists) return;
    const option = document.createElement('option');
    option.value = text;
    option.textContent = label || optionLabelForValue(text);
    option.dataset.custom = 'true';
    if (select.id === 'cgenFont') {
        option.style.fontFamily = fontCssStack(text);
    }
    select.appendChild(option);
}

function setValue(key, raw) {
    const field = fields[key];
    if (!field) return;
    if (field.type === 'checkbox') {
        field.checked = Boolean(raw);
    } else {
        ensureSelectOption(field, raw);
        field.value = raw ?? '';
    }
    if (key === 'font') {
        updateFontPreview();
    }
}

function sanitizeID(value) {
    const cleaned = String(value || '').trim().replace(/[^a-zA-Z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
    return cleaned || `cgen-${Date.now().toString(36)}`;
}

function readEditor() {
    const id = sanitizeID(value('id'));
    const programInputType = inputTypeValue();
    const browserWidth = cleanDimension(value('browserWidth', value('width', '1920')), '1920', 7680);
    const browserHeight = cleanDimension(value('browserHeight', value('height', '1080')), '1080', 4320);
    const browserFPS = cleanBrowserFPS(value('browserFPS', '60'));
    return {
        id,
        name: value('name', id),
        enabled: value('enabled'),
        mode: value('mode', 'release'),
        smpte_bars: value('smpteBars'),
        sunny_cat: value('sunnyCat'),
        program_input_type: programInputType,
        program_input_url: programInputType === 'browser' ? value('browserUrl') : value('programInput'),
        program_input_format: programInputType === 'browser' ? 'cef' : value('programInputFormat', 'mpegts'),
        hardware_decoder_enabled: value('hardwareDecoderEnabled'),
        hardware_decoder: value('hardwareDecoder'),
        browser_auto_size: value('browserAutoSize', true),
        browser_width: browserWidth,
        browser_height: browserHeight,
        browser_fps: browserFPS,
        priority_feed_id: value('priorityFeed', id),
        audio_source: value('audioSource', 'priority'),
        priority_input_format: 'priority-audio',
        program_output_url: value('programOutput'),
        program_output_format: value('outputFormat', 'mpegts'),
        vcodec: value('vcodec', 'avenc_mpeg2video'),
        acodec: value('acodec', 'avenc_ac3'),
        video_bitrate_kbps: value('hdBitrate', '12000'),
        audio_bitrate_kbps: value('stereoBitrate', '192'),
        video_encoder_codec: value('vcodec', 'avenc_mpeg2video'),
        video_encoder_bitrate_kbps: value('hdBitrate', '12000'),
        video_gop: value('videoGop', '15'),
        video_bframes: value('videoBFrames', '0'),
        video_preset: value('videoPreset'),
        video_tune: value('videoTune'),
        video_profile: value('videoProfile'),
        video_level: value('videoLevel'),
        audio_encoder_codec: value('acodec', 'avenc_ac3'),
        audio_encoder_bitrate_kbps: value('audioEncoderBitrate', value('stereoBitrate', '192')),
        audio_profile: value('audioProfile'),
        audio_level: value('audioLevel'),
        service_name: value('serviceName'),
        provider_name: value('providerName'),
        service_id: value('serviceID'),
        transport_stream_id: value('transportStreamID'),
        hd_enabled: 'auto',
        hd_bitrate_kbps: value('hdBitrate', '12000'),
        hd_program: value('hdProgram', '1'),
        hd_video_pid: value('hdVideoPID', '256'),
        hd_pmt_pid: value('hdPmtPID', '4096'),
        p720_enabled: value('p720Enabled'),
        p720_bitrate_kbps: value('p720Bitrate', '8000'),
        p720_program: value('p720Program', '2'),
        p720_video_pid: value('p720VideoPID', '288'),
        p720_pmt_pid: value('p720PmtPID', '4097'),
        sd_enabled: value('sdEnabled'),
        sd_bitrate_kbps: value('sdBitrate', '5000'),
        sd_program: value('sdProgram', '3'),
        sd_video_pid: value('sdVideoPID', '320'),
        sd_pmt_pid: value('sdPmtPID', '4098'),
        surround_enabled: value('surroundEnabled'),
        surround_bitrate_kbps: value('surroundBitrate', '384'),
        surround_program: value('surroundProgram', '1'),
        surround_audio_pid: value('surroundAudioPID', '258'),
        surround_pmt_pid: value('surroundPmtPID', '4096'),
        stereo_enabled: value('stereoEnabled'),
        stereo_bitrate_kbps: value('stereoBitrate', '192'),
        stereo_program: value('stereoProgram', '1'),
        stereo_audio_pid: value('stereoAudioPID', '257'),
        stereo_pmt_pid: value('stereoPmtPID', '4096'),
        width: value('width', '1920'),
        height: value('height', '1080'),
        fps: value('fps', '30000/1001'),
        interlaced: value('interlaced'),
        field_order: value('fieldOrder', 'tff'),
        standard: value('standard', 'atsc'),
        audio_idle: value('audioIdle', 'source'),
        audio_alert_mode: 'replace',
        mute_standby_routine: value('muteStandbyRoutine', true),
        banner_mode: value('bannerMode', 'auto'),
        ticker_height: value('bannerHeight', '128'),
        font: value('font', 'Arial'),
        font_weight: value('fontWeight', 'regular'),
        font_size: value('fontSize', '58'),
        scroll_speed: value('scrollSpeed', '8'),
        scroll_repeat_mode: value('scrollRepeatMode', 'until_audio_end'),
        after_eom_repeats: value('afterEomRepeats', '1'),
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
    setValue('sunnyCat', Boolean(feed.sunny_cat));
    const programInputType = feed.program_input_type === 'browser' ? 'browser' : 'stream';
    setValue('programInputType', programInputType);
    setValue('programInput', programInputType === 'stream' ? feed.program_input_url || '' : '');
    setValue('browserUrl', programInputType === 'browser' ? feed.program_input_url || '' : feed.browser_url || '');
    setValue('programInputFormat', feed.program_input_format || 'mpegts');
    setValue('hardwareDecoderEnabled', Boolean(feed.hardware_decoder_enabled));
    setValue('hardwareDecoder', feed.hardware_decoder || '');
    setValue('browserAutoSize', feed.browser_auto_size !== false);
    setValue('browserWidth', feed.browser_width || feed.width || '1920');
    setValue('browserHeight', feed.browser_height || feed.height || '1080');
    setValue('browserFPS', feed.browser_fps || '60');
    setValue('priorityFeed', feed.priority_feed_id || feed.id);
    setValue('audioSource', feed.audio_source || 'priority');
    setValue('audioIdle', feed.audio_idle || 'source');
    setValue('muteStandbyRoutine', feed.mute_standby_routine !== false);
    setValue('programOutput', feed.program_output_url || '');
    setValue('outputFormat', feed.program_output_format || 'mpegts');
    setValue('vcodec', feed.vcodec || 'avenc_mpeg2video');
    setValue('acodec', feed.acodec || 'avenc_ac3');
    setValue('hdBitrate', feed.hd_bitrate_kbps || feed.video_bitrate_kbps || '12000');
    setValue('videoGop', feed.video_gop || '15');
    setValue('videoBFrames', feed.video_bframes || '0');
    setValue('videoPreset', feed.video_preset || '');
    setValue('videoTune', feed.video_tune || '');
    setValue('videoProfile', feed.video_profile || '');
    setValue('videoLevel', feed.video_level || '');
    setValue('audioEncoderBitrate', feed.audio_encoder_bitrate_kbps || feed.stereo_bitrate_kbps || feed.audio_bitrate_kbps || '192');
    setValue('audioProfile', feed.audio_profile || '');
    setValue('audioLevel', feed.audio_level || '');
    setValue('serviceName', feed.service_name || '');
    setValue('providerName', feed.provider_name || '');
    setValue('serviceID', feed.service_id || '1');
    setValue('transportStreamID', feed.transport_stream_id || '1');
    setValue('hdProgram', feed.hd_program || '1');
    setValue('hdVideoPID', feed.hd_video_pid || '256');
    setValue('hdPmtPID', feed.hd_pmt_pid || '4096');
    setValue('p720Enabled', Boolean(feed.p720_enabled));
    setValue('p720Bitrate', feed.p720_bitrate_kbps || '8000');
    setValue('p720Program', feed.p720_program || '2');
    setValue('p720VideoPID', feed.p720_video_pid || '288');
    setValue('p720PmtPID', feed.p720_pmt_pid || '4097');
    setValue('sdEnabled', Boolean(feed.sd_enabled));
    setValue('sdBitrate', feed.sd_bitrate_kbps || '5000');
    setValue('sdProgram', feed.sd_program || '3');
    setValue('sdVideoPID', feed.sd_video_pid || '320');
    setValue('sdPmtPID', feed.sd_pmt_pid || '4098');
    setValue('surroundEnabled', feed.surround_enabled !== false);
    setValue('surroundBitrate', feed.surround_bitrate_kbps || '384');
    setValue('surroundProgram', feed.surround_program || '1');
    setValue('surroundAudioPID', feed.surround_audio_pid || '258');
    setValue('surroundPmtPID', feed.surround_pmt_pid || '4096');
    setValue('stereoEnabled', feed.stereo_enabled !== false);
    setValue('stereoBitrate', feed.stereo_bitrate_kbps || feed.audio_bitrate_kbps || '192');
    setValue('stereoProgram', feed.stereo_program || '1');
    setValue('stereoAudioPID', feed.stereo_audio_pid || '257');
    setValue('stereoPmtPID', feed.stereo_pmt_pid || '4096');
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
    setValue('afterEomRepeats', feed.after_eom_repeats || '1');
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
    updateProgramInputVisibility();
    updateEncoderControlVisibility();
    scheduleRender();
    editorDirty = false;
}

function catalogID(entry) {
    if (!entry || typeof entry !== 'object') return '';
    return String(entry.id || entry.value || '').trim();
}

function catalogLabel(entry) {
    if (!entry || typeof entry !== 'object') return '';
    return String(entry.label || entry.id || '').trim();
}

function catalogOptionTitle(entry) {
    if (!entry || typeof entry !== 'object') return '';
    const parts = [];
    if (entry.source) parts.push(`Source: ${entry.source}`);
    if (entry.element) parts.push(`GStreamer: ${entry.element}`);
    if (entry.kind) parts.push(`Kind: ${entry.kind}`);
    return parts.join('\n');
}

function populateCatalogSelect(select, entries, fallbackEntries = [], options = {}) {
    if (!select) return;
    const previous = String(select.value || '').trim();
    const list = Array.isArray(entries) && entries.length ? entries : fallbackEntries;
    select.replaceChildren();
    const seen = new Set();
    for (const entry of list) {
        const id = catalogID(entry);
        if (!id || seen.has(id)) continue;
        seen.add(id);
        const option = document.createElement('option');
        option.value = id;
        option.textContent = options.font
            ? `${catalogLabel(entry) || id} - ${entry.preview || 'The quick brown fox 0123456789'}`
            : (catalogLabel(entry) || id);
        const title = catalogOptionTitle(entry);
        if (title) option.title = title;
        if (entry.kind) option.dataset.kind = entry.kind;
        if (entry.element) option.dataset.element = entry.element;
        if (options.font) {
            option.style.fontFamily = fontCssStack(id);
        }
        select.appendChild(option);
    }
    ensureSelectOption(select, previous);
    if (previous) {
        select.value = previous;
    }
}

function populateCgenCatalogSelectors() {
    const formats = cgenCatalog.formats || [];
    populateCatalogSelect(fields.programInputFormat, formats, [{ id: 'mpegts', label: 'MPEG-TS' }]);
    populateCatalogSelect(fields.hardwareDecoder, streamVideoDecoderEntries(), [
        { id: 'nvh264dec', label: 'H.264 / AVC - NVIDIA NVDEC (nvh264dec)', element: 'nvh264dec' },
        { id: 'avdec_h264', label: 'H.264 / AVC - libav (avdec_h264)', element: 'avdec_h264' },
        { id: 'avdec_mpeg2video', label: 'MPEG-2 Video - libav (avdec_mpeg2video)', element: 'avdec_mpeg2video' },
    ]);
    populateCatalogSelect(fields.outputFormat, formats, [{ id: 'mpegts', label: 'MPEG-TS' }]);
    populateCatalogSelect(fields.vcodec, cgenCatalog.video_codecs || [], [
        { id: 'avenc_mpeg2video', label: 'MPEG-2 Video - libav (avenc_mpeg2video)', element: 'avenc_mpeg2video' },
        { id: 'x264enc', label: 'H.264 / AVC - x264 software (x264enc)', element: 'x264enc' },
    ]);
    populateCatalogSelect(fields.acodec, cgenCatalog.audio_codecs || [], [
        { id: 'avenc_ac3', label: 'AC-3 - libav (avenc_ac3)', element: 'avenc_ac3' },
        { id: 'avenc_aac', label: 'AAC - libav (avenc_aac)', element: 'avenc_aac' },
    ]);
    populateCatalogSelect(fields.font, cgenCatalog.fonts || [], [
        { id: 'Arial', label: 'Arial', preview: 'The quick brown fox 0123456789' },
        { id: 'Segoe UI', label: 'Segoe UI', preview: 'The quick brown fox 0123456789' },
    ], { font: true });
    renderFontPicker();
    updateFontPreview();
    updateProgramInputVisibility();
    updateEncoderControlVisibility();
}

function streamVideoDecoderEntries() {
    const entries = Array.isArray(cgenCatalog.video_decoders) ? cgenCatalog.video_decoders : [];
    const format = String(fields.programInputFormat?.value || 'mpegts').trim().toLowerCase();
    if (!format || format === 'mpegts' || format === 'ts') return entries;
    return entries.filter((entry) => {
        const haystack = `${entry?.id || ''} ${entry?.label || ''} ${entry?.kind || ''}`.toLowerCase();
        return haystack.includes(format);
    });
}

function updateFontPreview() {
    if (!fontPreview || !fields.font) return;
    const family = String(fields.font.value || 'Arial').trim() || 'Arial';
    fontPreview.style.fontFamily = fontCssStack(family);
    fontPreview.textContent = 'The quick brown fox 0123456789';
    fontPreview.title = family;
    updateFontPickerSelection();
}

function updateEncoderControlVisibility() {
    const videoCodec = String(fields.vcodec?.value || '').trim().toLowerCase();
    const audioCodec = String(fields.acodec?.value || '').trim().toLowerCase();
    const supportsPreset = /(?:x264|x265|h264|h265|hevc)/.test(videoCodec);
    const supportsBFrames = /(?:x264|x265|h264|h265|hevc)/.test(videoCodec);
    const supportsProfile = /(?:x264|x265|h264|h265|hevc|aac)/.test(videoCodec);
    const videoControls = [
        [fields.videoPreset, supportsPreset],
        [fields.videoTune, supportsPreset],
        [fields.videoBFrames, supportsBFrames],
        [fields.videoProfile, supportsProfile],
        [fields.videoLevel, supportsProfile],
    ];
    for (const [field, enabled] of videoControls) {
        if (field) field.disabled = !enabled;
    }
    const audioAdvanced = /(?:aac|opus|ac3|eac3|mp2|mp3)/.test(audioCodec);
    if (fields.audioProfile) fields.audioProfile.disabled = !audioAdvanced;
    if (fields.audioLevel) fields.audioLevel.disabled = !audioAdvanced;
}

function fontCssStack(family) {
    const cleaned = String(family || 'Arial').trim().replace(/\\/g, '\\\\').replace(/"/g, '\\"') || 'Arial';
    const base = cleaned.replace(/\s+(regular|bold|italic|black|medium|semibold|semi-bold|light|thin|cond|condensed|narrow|roman|book|heavy|variable|[1-9][0-9]{1,2})+$/i, '').trim();
    const fallback = base && base !== cleaned ? `, "${base.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"` : '';
    return `"${cleaned}"${fallback}, Arial, sans-serif`;
}

function registerManagedFontFaces() {
    if (!('FontFace' in window) || !document.fonts) return;
    const fonts = Array.isArray(cgenCatalog.fonts) ? cgenCatalog.fonts : [];
    for (const entry of fonts) {
        const family = catalogID(entry);
        const source = String(entry?.url || '').trim();
        if (!family || !source) continue;
        const key = `${family}\n${source}`;
        if (managedFontFaces.has(key)) continue;
        const face = new FontFace(family, `url(${JSON.stringify(source)})`);
        managedFontFaces.set(key, face);
        face.load().then((loaded) => {
            document.fonts.add(loaded);
            updateFontPreview();
            updateFontPickerSelection();
        }).catch(() => {});
    }
}

function fontPickerEntries() {
    const source = Array.isArray(cgenCatalog.fonts) && cgenCatalog.fonts.length
        ? cgenCatalog.fonts
        : [
            { id: 'Arial', label: 'Arial', preview: 'The quick brown fox 0123456789' },
            { id: 'Segoe UI', label: 'Segoe UI', preview: 'The quick brown fox 0123456789' },
        ];
    const seen = new Set();
    const entries = [];
    for (const entry of source) {
        const id = catalogID(entry);
        if (!id || seen.has(id.toLowerCase())) continue;
        seen.add(id.toLowerCase());
        entries.push({
            id,
            label: catalogLabel(entry) || id,
            preview: entry?.preview || 'The quick brown fox 0123456789',
        });
    }
    const current = String(fields.font?.value || '').trim();
    if (current && !seen.has(current.toLowerCase())) {
        entries.unshift({ id: current, label: current, preview: 'The quick brown fox 0123456789' });
    }
    return entries;
}

function renderFontPicker() {
    if (!fontMenu) return;
    fontMenu.replaceChildren();
    for (const entry of fontPickerEntries()) {
        const option = document.createElement('button');
        option.type = 'button';
        option.className = 'cgen-font-option';
        option.dataset.value = entry.id;
        option.setAttribute('role', 'option');
        option.style.fontFamily = fontCssStack(entry.id);
        const name = document.createElement('span');
        name.className = 'cgen-font-option-name';
        name.textContent = entry.label;
        const preview = document.createElement('span');
        preview.className = 'cgen-font-option-preview';
        preview.textContent = `- ${entry.preview}`;
        option.append(name, preview);
        option.addEventListener('click', () => selectFontFamily(entry.id));
        fontMenu.appendChild(option);
    }
    updateFontPickerSelection();
}

function updateFontPickerSelection() {
    const family = String(fields.font?.value || 'Arial').trim() || 'Arial';
    const entry = fontPickerEntries().find((item) => item.id === family);
    if (fontPickerLabel) {
        fontPickerLabel.textContent = `${entry?.label || family} - ${entry?.preview || 'The quick brown fox 0123456789'}`;
        fontPickerLabel.style.fontFamily = fontCssStack(family);
    }
    if (fontPicker) {
        fontPicker.title = family;
    }
    if (fontMenu) {
        fontMenu.querySelectorAll('.cgen-font-option').forEach((option) => {
            option.setAttribute('aria-selected', option.dataset.value === family ? 'true' : 'false');
        });
    }
}

function selectFontFamily(family) {
    if (!fields.font) return;
    ensureSelectOption(fields.font, family);
    fields.font.value = family;
    editorDirty = true;
    setFontPickerOpen(false);
    updateFontPreview();
    scheduleRender();
}

function setFontPickerOpen(open) {
    if (!fontPicker || !fontMenu) return;
    fontMenu.hidden = !open;
    fontPicker.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) {
        updateFontPickerSelection();
        const selectedOption = fontMenu.querySelector('.cgen-font-option[aria-selected="true"]') || fontMenu.querySelector('.cgen-font-option');
        selectedOption?.scrollIntoView({ block: 'nearest' });
    }
}

function toggleFontPicker() {
    setFontPickerOpen(fontMenu?.hidden !== false);
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
    const standbyAudio = feed.audio_idle === 'routine' ? 'feed routine standby' : 'program input standby';
    setText(metaPriority, `${feed.priority_feed_id || feed.id || '-'} / ${feed.audio_source || 'priority'} / ${standbyAudio} / ${standbyMute}`);
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
        const sunny = sunnyAvailableForFeed(feed) ? ` / Sunny ${feed.sunny_cat ? 'unleashed' : 'banished'}` : '';
        setText(metaVisual, `${visual} / ${video} video / ${audio} audio${sunny}`);
    }
    updateSunnyVisibility(feed, runtime);
    updatePreviewStream(feed);
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
    const label = `${feed.width || 1920}x${feed.height || 1080} ${feed.interlaced ? 'interlaced' : 'progressive'}  ${feed.vcodec || 'avenc_mpeg2video'} / ${feed.acodec || 'avenc_ac3'}`;
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

function drawSunnyPreview(ctx, width, height) {
    const w = Math.min(width * 0.22, 120);
    const h = w * 1.25;
    const x = width - w - 18;
    const y = height - h - 18;
    const gradient = ctx.createLinearGradient(x, y, x + w, y + h);
    gradient.addColorStop(0, '#ffd98a');
    gradient.addColorStop(0.55, '#f28c28');
    gradient.addColorStop(1, '#5b2f13');
    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, w, h);
    ctx.fillStyle = '#1b1009';
    ctx.fillRect(x + w * 0.18, y + h * 0.18, w * 0.16, h * 0.08);
    ctx.fillRect(x + w * 0.66, y + h * 0.18, w * 0.16, h * 0.08);
    ctx.fillStyle = '#ffffff';
    ctx.font = `700 ${Math.max(10, w / 9)}px Arial, sans-serif`;
    ctx.textBaseline = 'bottom';
    ctx.fillText('SUNNY', x + w * 0.08, y + h - 8);
}

function renderPreview() {
    if (!preview || (previewStream && !previewStream.hidden)) return;
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
    if (feed.sunny_cat && sunnyAvailableForFeed(feed)) {
        drawSunnyPreview(ctx, width, height);
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

function updatePreviewStream(feed = selected()) {
    if (!previewStream || !feed?.id) return;
    if (previewStreamFeedID === feed.id && previewStream.src) return;
    previewStreamFeedID = feed.id;
    previewStream.hidden = true;
    if (preview) preview.hidden = false;
    const url = `/api/v1/cgen/preview?feed=${encodeURIComponent(feed.id)}&t=${Date.now()}`;
    previewStream.src = url;
}

function sunnyAvailableForFeed(feed) {
    const runtime = feed?.runtime || {};
    if (runtime.sunny_cat_available === true) return true;
    const compositor = runtime.compositor || runtime.text_overlay || {};
    if (compositor.sunny_cat_available === true) return true;
    const renderer = runtime.graphics_renderer || {};
    if (renderer.sunny_cat_available === true) return true;
    if (Array.isArray(renderer.renditions)) {
        return renderer.renditions.some((rendition) => rendition && rendition.sunny_cat_available === true);
    }
    return false;
}

function updateSunnyVisibility(feed = selected(), runtime = feed?.runtime || {}) {
    const available = sunnyAvailableForFeed({ ...feed, runtime });
    if (sunnyField) sunnyField.hidden = !available;
    if (sunnyButton) {
        sunnyButton.hidden = !available;
        sunnyButton.textContent = feed?.sunny_cat ? 'Banish Loathed Creature' : 'Unleash Sunny, the Cat';
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
    updatePreviewStream(selected());
    setStatus('CGEN config loaded.', 'ok');
}

async function loadCgenCatalog({ announce = false } = {}) {
    const payload = await panelClient.command('cgen.catalog', {}, 15000);
    cgenCatalog = {
        formats: Array.isArray(payload.formats) ? payload.formats : [],
        video_codecs: Array.isArray(payload.video_codecs) ? payload.video_codecs : [],
        audio_codecs: Array.isArray(payload.audio_codecs) ? payload.audio_codecs : [],
        video_decoders: Array.isArray(payload.video_decoders) ? payload.video_decoders : [],
        browser_sources: Array.isArray(payload.browser_sources) ? payload.browser_sources : [],
        fonts: Array.isArray(payload.fonts) ? payload.fonts : [],
    };
    registerManagedFontFaces();
    populateCgenCatalogSelectors();
    if (announce) {
        const fontCount = cgenCatalog.fonts.length;
        const videoCount = cgenCatalog.video_codecs.length;
        const audioCount = cgenCatalog.audio_codecs.length;
        setStatus(`Catalog refreshed: ${videoCount} video codecs, ${audioCount} audio codecs, ${fontCount} fonts.`, 'ok');
    }
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
    updatePreviewStream(selected());
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
        id: 'CFSP-CAP',
        name: 'CFSP/CAP CGEN',
        enabled: true,
        mode: 'release',
        program_input_type: 'stream',
        program_input_url: 'udp://239.0.0.1:9000?fifo_size=2000000&overrun_nonfatal=1&reuse=1&buffer_size=1048576',
        program_input_format: 'mpegts',
        hardware_decoder_enabled: false,
        hardware_decoder: '',
        browser_auto_size: true,
        browser_width: '1920',
        browser_height: '1080',
        browser_fps: '60',
        priority_feed_id: '*',
        audio_source: 'priority',
        audio_idle: 'source',
        mute_standby_routine: true,
        program_output_url: 'udp://239.0.0.2:9001?pkt_size=1316&buffer_size=1048576&reuse=1',
        program_output_format: 'mpegts',
        vcodec: 'avenc_mpeg2video',
        acodec: 'avenc_ac3',
        video_bitrate_kbps: '12000',
        audio_bitrate_kbps: '192',
        video_encoder_codec: 'avenc_mpeg2video',
        video_encoder_bitrate_kbps: '12000',
        video_gop: '15',
        video_bframes: '0',
        video_preset: '',
        video_tune: '',
        video_profile: '',
        video_level: '',
        audio_encoder_codec: 'avenc_ac3',
        audio_encoder_bitrate_kbps: '192',
        audio_profile: '',
        audio_level: '',
        service_name: 'Haze CGEN',
        provider_name: 'Haze',
        service_id: '1',
        transport_stream_id: '1',
        hd_enabled: 'auto',
        hd_bitrate_kbps: '12000',
        hd_program: '1',
        hd_video_pid: '256',
        hd_pmt_pid: '4096',
        p720_enabled: false,
        p720_bitrate_kbps: '8000',
        p720_program: '2',
        p720_video_pid: '288',
        p720_pmt_pid: '4097',
        sd_enabled: false,
        sd_bitrate_kbps: '5000',
        sd_program: '3',
        sd_video_pid: '320',
        sd_pmt_pid: '4098',
        surround_enabled: true,
        surround_bitrate_kbps: '384',
        surround_program: '1',
        surround_audio_pid: '258',
        surround_pmt_pid: '4096',
        stereo_enabled: true,
        stereo_bitrate_kbps: '192',
        stereo_program: '1',
        stereo_audio_pid: '257',
        stereo_pmt_pid: '4096',
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
        after_eom_repeats: '1',
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
        sunny_cat: false,
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
    const current = readEditor();
    const payload = await panelClient.command('cgen.action', { ...current, feed_id: selectedID, action, ...extra }, 10000);
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
    bindCgenTabs();
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
            if (field === fields.font) updateFontPreview();
            if (field === fields.vcodec || field === fields.acodec) updateEncoderControlVisibility();
            if (field === fields.programInputType || field === fields.browserAutoSize || field === fields.hardwareDecoderEnabled) updateProgramInputVisibility();
            if (field === fields.programInputFormat) {
                populateCatalogSelect(fields.hardwareDecoder, streamVideoDecoderEntries(), [
                    { id: 'nvh264dec', label: 'H.264 / AVC - NVIDIA NVDEC (nvh264dec)', element: 'nvh264dec' },
                    { id: 'avdec_h264', label: 'H.264 / AVC - libav (avdec_h264)', element: 'avdec_h264' },
                    { id: 'avdec_mpeg2video', label: 'MPEG-2 Video - libav (avdec_mpeg2video)', element: 'avdec_mpeg2video' },
                ]);
                updateProgramInputVisibility();
            }
        });
        field?.addEventListener('change', () => {
            editorDirty = true;
            scheduleRender();
            if (field === fields.font) updateFontPreview();
            if (field === fields.vcodec || field === fields.acodec) updateEncoderControlVisibility();
            if (field === fields.programInputType || field === fields.browserAutoSize || field === fields.hardwareDecoderEnabled) updateProgramInputVisibility();
            if (field === fields.programInputFormat) {
                populateCatalogSelect(fields.hardwareDecoder, streamVideoDecoderEntries(), [
                    { id: 'nvh264dec', label: 'H.264 / AVC - NVIDIA NVDEC (nvh264dec)', element: 'nvh264dec' },
                    { id: 'avdec_h264', label: 'H.264 / AVC - libav (avdec_h264)', element: 'avdec_h264' },
                    { id: 'avdec_mpeg2video', label: 'MPEG-2 Video - libav (avdec_mpeg2video)', element: 'avdec_mpeg2video' },
                ]);
                updateProgramInputVisibility();
            }
        });
    });
    refreshFontsButton?.addEventListener('click', () => {
        refreshFontsButton.disabled = true;
        loadCgenCatalog({ announce: true })
            .catch((error) => setStatus(error.message || 'Unable to refresh CGEN catalog.', 'err'))
            .finally(() => {
                refreshFontsButton.disabled = false;
            });
    });
    fontPicker?.addEventListener('click', toggleFontPicker);
    fontPicker?.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ' || event.key === 'ArrowDown') {
            event.preventDefault();
            setFontPickerOpen(true);
        }
    });
    fontMenu?.addEventListener('keydown', (event) => {
        if (event.key === 'Escape') {
            event.preventDefault();
            setFontPickerOpen(false);
            fontPicker?.focus();
        }
    });
    document.addEventListener('click', (event) => {
        if (!fontMenu || fontMenu.hidden) return;
        const target = event.target;
        if (target instanceof Node && (fontMenu.contains(target) || fontPicker?.contains(target))) {
            return;
        }
        setFontPickerOpen(false);
    });
    document.getElementById('cgenReleaseButton')?.addEventListener('click', () => runAction('release').catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenOverlayButton')?.addEventListener('click', () => runAction('overlay').catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenBarsButton')?.addEventListener('click', () => runAction('smpte_bars', { enabled: !fields.smpteBars.checked }).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenClockButton')?.addEventListener('click', () => runAction('clock', { enabled: !fields.clockEnabled.checked }).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenInsertTextButton')?.addEventListener('click', () => runAction('insert_text', { text: fields.text.value }).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    document.getElementById('cgenClearTextButton')?.addEventListener('click', () => runAction('clear_text').catch((error) => setStatus(error.message || 'CGEN action failed.', 'err')));
    sunnyButton?.addEventListener('click', () => {
        const action = fields.sunnyCat.checked ? 'banish_sunny' : 'unleash_sunny';
        runAction(action).catch((error) => setStatus(error.message || 'CGEN action failed.', 'err'));
    });
    previewStream?.addEventListener('load', () => {
        previewStream.hidden = false;
        if (preview) preview.hidden = true;
    });
    previewStream?.addEventListener('error', () => {
        previewStream.hidden = true;
        if (preview) preview.hidden = false;
        window.clearTimeout(previewRetryTimer);
        previewRetryTimer = window.setTimeout(() => {
            previewStreamFeedID = '';
            updatePreviewStream(selected());
        }, 2500);
    });
    window.setInterval(() => {
        refreshRuntime().catch(() => scheduleRender({ preview: true, meta: false }));
    }, 1500);
    loadCgenCatalog()
        .catch(() => {
            populateCgenCatalogSelectors();
        })
        .finally(() => {
            loadCgen().catch((error) => setStatus(error.message || 'Unable to load CGEN config.', 'err'));
        });
}

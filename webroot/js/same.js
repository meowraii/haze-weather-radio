import { apiGet, apiPost, apiPut, apiPostForm, token } from './lib/api.js';
import { initTheme } from './lib/theme.js';

const TEST_CODES = new Set(['DMO', 'RWT', 'RMT']);

const ORIGINATORS = [
    { value: 'EAS', label: 'EAS — EAS Participant' },
    { value: 'WXR', label: 'WXR — Weather' },
    { value: 'CIV', label: 'CIV — Civil Authority' },
    { value: 'PEP', label: 'PEP — Primary Entry Point' },
];

const TONES = [
    { code: 'EAS',       desc: 'Dual 853/960 Hz' },
    { code: 'WXR',       desc: '1050 Hz single'  },
    { code: 'NPAS',      desc: 'Alert Ready'      },
    { code: 'EGG_TIMER', desc: 'Legacy Ontario'   },
    { code: 'NONE',      desc: 'No attention tone'},
];

const AUDIO_MODES = [
    { mode: 'tts',  label: 'TTS Text'    },
    { mode: 'tone', label: 'Tone Only'   },
    { mode: 'none', label: 'No Audio'    },
    { mode: 'file', label: 'Upload File' },
];

let allFeedsData        = [];
let sameMapping         = {};
let locationNames       = {};
let configuredCallsign  = 'XXXXXXXX';
let selectedTone        = 'EAS';
let selectedFeedIds     = new Set();
let selectedLocCodes    = new Set();
let customLocEntries    = [];
let audioMode           = 'tts';
let uploadedFilePath    = '';
let selectedFile        = null;
let generatedPreviewPath = '';
let generatedPreviewUrl   = '';
let generatedPreviewReady = false;
let generatedDraftSig     = '';
let airConfirmPending   = false;
let airConfirmTimer     = null;
const recentBroadcasts  = [];
let templateData        = {};

function cloneData(value) {
    if (typeof structuredClone === 'function') return structuredClone(value);
    return JSON.parse(JSON.stringify(value));
}

function makeEmptyTemplate(code) {
    return {
        name: code,
        description: '',
        automated: {
            enabled: false,
            allow_postpone: false,
            timing: { timezone: '', day_of_week: '', weeks: [], hour: 0, minute: 0, second: 0 },
        },
        same: {
            enabled: true,
            event: code,
            locations: [],
            duration: { hr: 0, min: 15 },
            sender_id: '',
            content: {
                attention_tone: '',
                lang: { 'en-CA': '' },
                file: { 'en-CA': '' },
            },
        },
        sameEvent: code,
        sameExpire: '0015',
        msg: { 'en-CA': '' },
        files: { 'en-CA': '' },
    };
}

function invalidateGeneratedPreview() {
    generatedPreviewPath = '';
    generatedPreviewUrl = '';
    generatedPreviewReady = false;
}

function buildAlertSignature() {
    return JSON.stringify({
        orig: origSelect?.value || 'EAS',
        event: eventSelect?.value || 'DMO',
        locs: getAllSelectedCodes(),
        hours: durationHours?.value || '0',
        minutes: durationMins?.value || '0',
        tone: selectedTone,
        audioMode,
        voice: voiceInput?.value || '',
        file: audioMode === 'file' ? (uploadedFilePath || selectedFile?.name || '') : '',
        feeds: [...selectedFeedIds].sort(),
    });
}

function buildLayout() {
    document.getElementById('sameMain').innerHTML = `
        <section class="section-block">
            <div class="section-hd">
                <i data-lucide="file-code-2" width="14" height="14" class="sp-hd-icon"></i>
                <span>Auto-fill from CAP XML</span>
                <span class="section-hd-sub">Pre-fill from a CAP-CP or NWS alert XML.</span>
                <div class="section-hd-actions">
                    <label class="btn-action sp-file-label" for="capXmlInput">
                        <i data-lucide="upload" width="13" height="13"></i>
                        Choose File
                    </label>
                    <input type="file" id="capXmlInput" accept=".xml,text/xml" class="sp-hidden">
                    <span id="capStatus" class="sp-status-text"></span>
                </div>
            </div>
        </section>

        <div class="same-top-grid">
            <section class="section-block">
                <div class="section-hd">
                    <i data-lucide="settings-2" width="14" height="14" class="sp-hd-icon"></i>
                    <span>Alert Configuration</span>
                </div>
                <div class="section-body">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="origSelect">Originator</label>
                            <select id="origSelect">
                                ${ORIGINATORS.map((o) => `<option value="${o.value}">${o.label}</option>`).join('')}
                            </select>
                        </div>
                        <div class="form-group">
                            <label for="eventSelect">Event</label>
                            <select id="eventSelect"></select>
                            <p id="eventNameHint" class="sp-hint-accent sp-hidden"></p>
                        </div>
                    </div>
                    <div class="form-row sp-row-mt">
                        <div class="form-group">
                            <label for="durationHours">Hours</label>
                            <input id="durationHours" type="number" min="0" max="99" value="1">
                        </div>
                        <div class="form-group">
                            <label for="durationMinutes">Minutes</label>
                            <input id="durationMinutes" type="number" min="0" max="59" value="0">
                        </div>
                    </div>
                    <div class="sp-tone-row">
                        <p class="sp-sublabel sp-tone-label">Attention Tone</p>
                        <div class="sp-tone-carousel" id="toneCarousel">
                            ${TONES.map((t, i) => `
                            <button type="button" class="sp-tone-btn${i === 0 ? ' active' : ''}" data-tone="${t.code}">
                                <span class="sp-tone-code">${t.code}</span>
                                <span class="sp-tone-desc">${t.desc}</span>
                            </button>`).join('')}
                        </div>
                    </div>
                </div>
            </section>

            <section class="section-block">
                <div class="section-hd">
                    <i data-lucide="radio" width="14" height="14" class="sp-hd-icon"></i>
                    <span>Feeds &amp; Locations</span>
                </div>
                <div class="sp-feeds-locs-grid">
                    <div class="sp-feeds-col">
                        <div class="sp-col-hd">
                            <span class="sp-sublabel">Target Feeds</span>
                            <div class="sp-col-actions">
                                <button id="feedSelectAll"   type="button" class="btn-action">All</button>
                                <button id="feedSelectNone"  type="button" class="btn-action">None</button>
                                <button id="feedSelectFirst" type="button" class="btn-action">First</button>
                            </div>
                        </div>
                        <div id="feedPicker" class="sp-check-picker">
                            <p class="sp-picker-empty">Loading feeds…</p>
                        </div>
                    </div>
                    <div class="sp-locs-col">
                        <div class="sp-col-hd">
                            <span class="sp-sublabel">Locations</span>
                            <div class="sp-col-actions">
                                <button id="locSelectAll"    type="button" class="btn-action">All</button>
                                <button id="locSelectNone"   type="button" class="btn-action">None</button>
                                <button id="locAddCustomBtn" type="button" class="btn-action">+ Custom</button>
                            </div>
                        </div>
                        <div id="locPicker" class="sp-check-picker">
                            <p class="sp-picker-empty">Select a feed to load locations.</p>
                        </div>
                        <div id="customLocRow" class="sp-custom-loc-row sp-hidden">
                            <input id="customLocInput" class="input-field sp-custom-loc-input" type="text"
                                placeholder="6-digit CLC" maxlength="6" pattern="\\d{6}">
                            <input id="customLocNameInput" class="input-field sp-custom-loc-name" type="text"
                                placeholder="Label">
                            <button id="customLocAdd" type="button" class="btn-action">Add</button>
                        </div>
                    </div>
                </div>
            </section>
        </div>

        <section class="section-block">
            <div class="section-hd">
                <i data-lucide="mic" width="14" height="14" class="sp-hd-icon"></i>
                <span>Audio Message</span>
            </div>
            <div class="section-body">
                <div class="sp-audio-tabs" id="audioModeTabs">
                    ${AUDIO_MODES.map((m, i) => `
                    <button type="button" class="sp-audio-tab${i === 0 ? ' active' : ''}" data-mode="${m.mode}">${m.label}</button>`).join('')}
                </div>

                <div id="audioPanelTts" class="sp-audio-panel">
                    <div class="form-group">
                        <label for="voiceInput">Text to Synthesize</label>
                        <textarea id="voiceInput" rows="3" placeholder="Spoken message appended after the attention tone…"></textarea>
                    </div>
                </div>

                <div id="audioPanelFile" class="sp-audio-panel sp-hidden">
                    <div class="form-group">
                        <label>Audio File</label>
                        <div class="upload-area" id="uploadArea">
                            <input type="file" id="audioFile" accept="audio/*,video/*" class="sp-upload-input">
                            <div class="upload-area-text">
                                <i data-lucide="file-audio" width="20" height="20" class="sp-upload-icon"></i>
                                <p id="uploadPromptEl">Click or drag an audio / video file here.<br>
                                    <span class="sp-upload-hint">ffmpeg will encode to PCM WAV.</span></p>
                                <p id="uploadFilenameEl" class="sp-upload-filename sp-hidden"></p>
                            </div>
                        </div>
                        <div id="uploadActions" class="form-actions sp-mt-sm sp-hidden">
                            <button id="uploadBtn"      type="button" class="btn-action">Upload &amp; Encode</button>
                            <button id="clearUploadBtn" type="button" class="btn-ghost">Clear</button>
                        </div>
                        <p id="uploadStatusEl" class="sp-hint sp-mt-xs"></p>
                    </div>
                </div>

                <div id="audioPanelTone" class="sp-audio-panel sp-hidden">
                    <p class="sp-hint">Only the selected attention tone will play. No spoken message.</p>
                </div>

                <div id="audioPanelNone" class="sp-audio-panel sp-hidden">
                    <p class="sp-hint">The SAME header will be transmitted with no audio segment.</p>
                </div>
            </div>
        </section>

        <section class="section-block">
            <div class="section-hd">
                <i data-lucide="clipboard-list" width="14" height="14" class="sp-hd-icon"></i>
                <span>SAME Test Templates</span>
                <span class="section-hd-sub">Stored in <code class="sp-inline-code">alertTemplates.xml</code>.</span>
                <div class="section-hd-actions">
                    <button id="templateAddBtn"  type="button" class="btn-action">+ New</button>
                    <button id="templateSaveBtn" type="button" class="btn-action">Save All</button>
                </div>
            </div>
            <div class="section-body">
                <p id="templateStatus" class="status-banner sp-mb-sm"></p>
                <div id="templateList" class="sp-template-list"></div>
            </div>
        </section>

        <section class="section-block">
            <div class="section-hd">
                <i data-lucide="history" width="14" height="14" class="sp-hd-icon"></i>
                <span>Recent Broadcasts</span>
            </div>
            <div class="section-body">
                <div id="recentList">
                    <p class="sp-hint">No broadcasts in current session.</p>
                </div>
            </div>
        </section>
    `;

    document.getElementById('sameRail').innerHTML = `
        <div class="sp-rail">
            <div class="sp-rail-section">
                <p class="sp-rail-label">SAME Header</p>
                <code id="headerPreview" class="sp-header-preview">Complete the form.</code>
            </div>
            <div class="sp-rail-section">
                <p class="sp-rail-label">Attention Tone</p>
                <span id="railToneBadge" class="sp-tone-badge">EAS</span>
            </div>
            <div class="sp-rail-section">
                <div class="sp-rail-loc-hd">
                    <p class="sp-rail-label">Locations</p>
                    <span id="railLocCount" class="sp-loc-count">0</span>
                </div>
                <div id="railLocList" class="sp-rail-loc-list">
                    <p class="sp-hint">None selected.</p>
                </div>
            </div>
            <div id="railTtsSection" class="sp-rail-section sp-hidden">
                <p class="sp-rail-label">Message Preview</p>
                <p id="railTtsText" class="sp-rail-tts-text"></p>
            </div>
            <div class="sp-rail-section sp-rail-air">
                <p id="statusBanner" class="status-banner sp-status-banner"></p>
                <div class="sp-rail-actions">
                    <button id="previewBtn" type="button" class="btn-action">Preview</button>
                    <button id="airBtn" type="button" class="btn-danger sp-air-btn">Air Now</button>
                </div>
                <audio id="previewAudio" class="sp-preview-player sp-hidden" controls></audio>
            </div>
        </div>
    `;
}

buildLayout();

const apiDot             = document.getElementById('apiDot');
const healthPill         = document.getElementById('healthPill');
const origSelect         = document.getElementById('origSelect');
const eventSelect        = document.getElementById('eventSelect');
const eventNameHint      = document.getElementById('eventNameHint');
const durationHours      = document.getElementById('durationHours');
const durationMins       = document.getElementById('durationMinutes');
const toneCarousel       = document.getElementById('toneCarousel');
const feedPicker         = document.getElementById('feedPicker');
const locPicker          = document.getElementById('locPicker');
const customLocRow       = document.getElementById('customLocRow');
const customLocInput     = document.getElementById('customLocInput');
const customLocNameInput = document.getElementById('customLocNameInput');
const customLocAdd       = document.getElementById('customLocAdd');
const audioModeTabs      = document.getElementById('audioModeTabs');
const voiceInput         = document.getElementById('voiceInput');
const audioFile          = document.getElementById('audioFile');
const uploadArea         = document.getElementById('uploadArea');
const uploadPromptEl     = document.getElementById('uploadPromptEl');
const uploadFilenameEl   = document.getElementById('uploadFilenameEl');
const uploadActions      = document.getElementById('uploadActions');
const uploadBtn          = document.getElementById('uploadBtn');
const clearUploadBtn     = document.getElementById('clearUploadBtn');
const uploadStatusEl     = document.getElementById('uploadStatusEl');
const headerPreview      = document.getElementById('headerPreview');
const railToneBadge      = document.getElementById('railToneBadge');
const railLocCount       = document.getElementById('railLocCount');
const railLocList        = document.getElementById('railLocList');
const railTtsSection     = document.getElementById('railTtsSection');
const railTtsText        = document.getElementById('railTtsText');
const statusBanner       = document.getElementById('statusBanner');
const previewBtn         = document.getElementById('previewBtn');
const airBtn             = document.getElementById('airBtn');
const previewAudio       = document.getElementById('previewAudio');
const capXmlInput        = document.getElementById('capXmlInput');
const capStatus          = document.getElementById('capStatus');
const templateList       = document.getElementById('templateList');
const templateStatus     = document.getElementById('templateStatus');
const templateAddBtn     = document.getElementById('templateAddBtn');
const templateSaveBtn    = document.getElementById('templateSaveBtn');
const recentList         = document.getElementById('recentList');

initTheme(document.getElementById('themeToggle'));

function showStatus(msg, type) {
    statusBanner.textContent = msg;
    statusBanner.className = `status-banner sp-status-banner ${type}`;
}

function clearStatus() {
    statusBanner.textContent = '';
    statusBanner.className = 'status-banner sp-status-banner';
}

function getEventCatalog() {
    const eas = sameMapping.eas || {};
    const naadsToEas = sameMapping.naadsToEas || {};
    const reverseNaads = new Map();
    Object.entries(naadsToEas).forEach(([naads, easCode]) => {
        if (!reverseNaads.has(easCode)) reverseNaads.set(easCode, naads);
    });
    return Object.entries(eas)
        .map(([code, description]) => ({
            code,
            description,
            naads: reverseNaads.get(code) || '',
        }))
        .sort((a, b) => a.code.localeCompare(b.code));
}

function eventLabelFor(code) {
    const entry = (sameMapping.eas || {})[code] || '';
    const naads = getEventCatalog().find((item) => item.code === code)?.naads || '';
    return `${code} - ${entry || code}${naads ? ` - ${naads}` : ''}`;
}

function buildPreview() {
    const orig  = (origSelect.value  || 'EAS').toUpperCase().slice(0, 3);
    const event = (eventSelect.value || 'DMO').toUpperCase().slice(0, 3);
    const locs  = getAllSelectedCodes();
    const locStr = (locs.length ? locs.slice(0, 31) : ['000000']).join('-');
    const h   = Math.max(0, Math.min(parseInt(durationHours.value, 10) || 0, 99));
    const m   = Math.max(0, Math.min(parseInt(durationMins.value,  10) || 0, 59));
    const dur = (h === 0 && m === 0) ? '0100' : `${String(h).padStart(2, '0')}${String(m).padStart(2, '0')}`;
    const now = new Date();
    const doy = String(Math.floor((now - new Date(now.getFullYear(), 0, 0)) / 86400000)).padStart(3, '0');
    const utcH = String(now.getUTCHours()).padStart(2, '0');
    const utcM = String(now.getUTCMinutes()).padStart(2, '0');
    const cs   = configuredCallsign.replace(/-/g, '/').padEnd(8).slice(0, 8);
    return `ZCZC-${orig}-${event}-${locStr}+${dur}-${doy}${utcH}${utcM}-${cs}-`;
}

function updatePreview() {
    const signature = buildAlertSignature();
    if (generatedDraftSig && generatedDraftSig !== signature) {
        invalidateGeneratedPreview();
    }
    generatedDraftSig = signature;
    headerPreview.textContent = buildPreview();
    updateEventHint();
    updateRailLocations();
    if (audioMode === 'tts') generateTtsText();
    updateRailTts();
}

function updateEventHint() {
    const entry = (sameMapping.eas || {})[eventSelect.value];
    const naads = getEventCatalog().find((item) => item.code === eventSelect.value)?.naads || '';
    const hintText = [entry || '', naads ? `NAADS: ${naads}` : ''].filter(Boolean).join(' • ');
    const hasText = Boolean(hintText);
    eventNameHint.classList.toggle('sp-hidden', !hasText);
    eventNameHint.textContent = hintText;
}

toneCarousel.addEventListener('click', (e) => {
    const btn = e.target.closest('.sp-tone-btn');
    if (!btn) return;
    toneCarousel.querySelectorAll('.sp-tone-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    selectedTone = btn.dataset.tone;
    railToneBadge.textContent = selectedTone;
    updatePreview();
});

function populateEventSelect() {
    const options = getEventCatalog();
    eventSelect.innerHTML = options
        .map(({ code, description, naads }) => `<option value="${code}">${code} - ${description || code}${naads ? ` - ${naads}` : ''}</option>`)
        .join('');
    eventSelect.value = 'DMO';
    updateEventHint();
}

[origSelect, eventSelect, durationHours, durationMins].forEach((el) => {
    el.addEventListener('change', updatePreview);
    el.addEventListener('input',  updatePreview);
});

function renderFeedPicker() {
    if (!allFeedsData.length) {
        feedPicker.innerHTML = '<p class="sp-picker-empty">No feeds configured.</p>';
        return;
    }
    feedPicker.innerHTML = allFeedsData.map((f) => {
        const checked = selectedFeedIds.has(f.id) ? 'checked' : '';
        return `<label class="sp-ex-item">
            <input type="checkbox" class="sp-feed-check" value="${f.id}" ${checked}>
            <span class="sp-ex-name" title="${f.name || f.id}">${f.name || f.id}</span>
        </label>`;
    }).join('');
    feedPicker.querySelectorAll('.sp-feed-check').forEach((cb) => {
        cb.addEventListener('change', () => {
            if (cb.checked) selectedFeedIds.add(cb.value);
            else selectedFeedIds.delete(cb.value);
            renderLocPicker();
            updatePreview();
        });
    });
}

document.getElementById('feedSelectAll').addEventListener('click', () => {
    selectedFeedIds = new Set(allFeedsData.map((f) => f.id));
    renderFeedPicker(); renderLocPicker(); updatePreview();
});
document.getElementById('feedSelectNone').addEventListener('click', () => {
    selectedFeedIds.clear();
    renderFeedPicker(); renderLocPicker(); updatePreview();
});
document.getElementById('feedSelectFirst').addEventListener('click', () => {
    selectedFeedIds.clear();
    if (allFeedsData.length) selectedFeedIds.add(allFeedsData[0].id);
    renderFeedPicker(); renderLocPicker(); updatePreview();
});

function getCodesForSelectedFeeds() {
    const codes = new Map();
    for (const feed of allFeedsData) {
        if (!selectedFeedIds.has(feed.id)) continue;
        for (const code of (feed.clc_codes || [])) {
            if (!codes.has(code)) codes.set(code, locationNames[code] || '');
        }
    }
    return codes;
}

function getAllSelectedCodes() {
    return [...selectedLocCodes];
}

function renderLocPicker() {
    const feedCodes = getCodesForSelectedFeeds();
    if (!feedCodes.size && !customLocEntries.length) {
        locPicker.innerHTML = '<p class="sp-picker-empty">Select feeds above to see available location codes.</p>';
        updateRailLocations();
        return;
    }

    let html = '';
    for (const [code, name] of feedCodes) {
        const checked = selectedLocCodes.has(code) ? 'checked' : '';
        html += `<label class="sp-ex-item">
            <input type="checkbox" class="sp-loc-check" value="${code}" ${checked}>
            <span class="sp-ex-name" title="${name || code}">${name || code}</span>
            ${name ? `<span class="sp-ex-code">${code}</span>` : ''}
        </label>`;
    }
    for (const entry of customLocEntries) {
        const checked = selectedLocCodes.has(entry.code) ? 'checked' : '';
        html += `<label class="sp-ex-item">
            <input type="checkbox" class="sp-loc-check" value="${entry.code}" ${checked}>
            <span class="sp-ex-name" title="${entry.name || entry.code}">${entry.name || entry.code}</span>
            <span class="sp-ex-code">${entry.code} <em class="sp-custom-tag">custom</em></span>
            <button type="button" class="sp-remove-custom" data-code="${entry.code}" title="Remove">&#x2715;</button>
        </label>`;
    }

    locPicker.innerHTML = html;
    locPicker.querySelectorAll('.sp-loc-check').forEach((cb) => {
        cb.addEventListener('change', () => {
            if (cb.checked) selectedLocCodes.add(cb.value);
            else selectedLocCodes.delete(cb.value);
            updateRailLocations();
            updatePreview();
        });
    });
    locPicker.querySelectorAll('.sp-remove-custom').forEach((btn) => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            customLocEntries = customLocEntries.filter((en) => en.code !== btn.dataset.code);
            selectedLocCodes.delete(btn.dataset.code);
            renderLocPicker();
            updatePreview();
        });
    });
    updateRailLocations();
}

document.getElementById('locSelectAll').addEventListener('click', () => {
    for (const code of getCodesForSelectedFeeds().keys()) selectedLocCodes.add(code);
    for (const { code } of customLocEntries) selectedLocCodes.add(code);
    renderLocPicker(); updatePreview();
});
document.getElementById('locSelectNone').addEventListener('click', () => {
    selectedLocCodes.clear();
    renderLocPicker(); updatePreview();
});
document.getElementById('locAddCustomBtn').addEventListener('click', () => {
    const wasHidden = customLocRow.classList.contains('sp-hidden');
    customLocRow.classList.toggle('sp-hidden');
    if (wasHidden) customLocInput.focus();
});

customLocAdd.addEventListener('click', () => {
    const code = customLocInput.value.trim().replace(/\D/g, '').padStart(6, '0').slice(0, 6);
    if (code.length !== 6 || customLocEntries.some((e) => e.code === code)) return;
    const name = customLocNameInput.value.trim() || locationNames[code] || '';
    customLocEntries.push({ code, name });
    selectedLocCodes.add(code);
    customLocInput.value = '';
    customLocNameInput.value = '';
    renderLocPicker();
    updatePreview();
});
customLocInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') customLocAdd.click(); });

function updateRailLocations() {
    const codes = getAllSelectedCodes();
    railLocCount.textContent = codes.length;
    if (!codes.length) {
        railLocList.innerHTML = '<p class="sp-hint">None selected.</p>';
        return;
    }
    const allNames = new Map([...getCodesForSelectedFeeds()]);
    for (const { code, name } of customLocEntries) allNames.set(code, name);
    railLocList.innerHTML = codes.slice(0, 20).map((code) => {
        const name = allNames.get(code) || locationNames[code] || '';
        return `<div class="sp-rl-item">
            <span class="sp-rl-name">${name || code}</span>
            ${name ? `<span class="sp-rl-code">${code}</span>` : ''}
        </div>`;
    }).join('') + (codes.length > 20
        ? `<p class="sp-hint sp-mt-xs">+${codes.length - 20} more</p>`
        : '');
}

function updateRailTts() {
    const text = voiceInput.value.trim();
    const show = audioMode === 'tts' && Boolean(text);
    railTtsSection.classList.toggle('sp-hidden', !show);
    if (show) railTtsText.textContent = text.length > 160 ? text.slice(0, 160) + '\u2026' : text;
}
voiceInput.addEventListener('input', updateRailTts);

function generateTtsText() {
    const ORIGINATOR_LABELS = {
        WXR: 'Environment and Climate Change Canada',
        CIV: 'Civil Authorities',
        PEP: 'A Primary Entry Point System',
    };
    const issuer    = ORIGINATOR_LABELS[origSelect.value] ?? 'An EAS Participant';
    const code      = eventSelect.value || 'DMO';
    const eventName = (sameMapping.eas || {})[code] || code;
    const codes       = getAllSelectedCodes();
    const allNamesMap = new Map([...getCodesForSelectedFeeds()]);
    for (const { code: c, name } of customLocEntries) allNamesMap.set(c, name);
    const names = [...new Set(codes.map((c) => allNamesMap.get(c) || locationNames[c] || c))];

    let areaClause = '';
    if (names.length === 1)      areaClause = ` for ${names[0]}`;
    else if (names.length === 2) areaClause = ` for ${names[0]} and ${names[1]}`;
    else if (names.length <= 4)  areaClause = ` for ${names.slice(0, -1).join(', ')}, and ${names.at(-1)}`;
    else                         areaClause = ` for ${names.slice(0, 3).join(', ')}, and ${names.length - 3} other ${names.length - 3 === 1 ? 'area' : 'areas'}`;

    const h = parseInt(durationHours.value, 10) || 0;
    const m = parseInt(durationMins.value,  10) || 0;
    const dParts = [];
    if (h) dParts.push(`${h} hour${h !== 1 ? 's' : ''}`);
    if (m) dParts.push(`${m} minute${m !== 1 ? 's' : ''}`);
    voiceInput.value = `${issuer} has issued a ${eventName}${areaClause}. In effect for ${dParts.join(' and ') || '1 hour'}.`;
}

const audioPanels = {
    tts:  document.getElementById('audioPanelTts'),
    file: document.getElementById('audioPanelFile'),
    tone: document.getElementById('audioPanelTone'),
    none: document.getElementById('audioPanelNone'),
};

audioModeTabs.addEventListener('click', (e) => {
    const btn = e.target.closest('.sp-audio-tab');
    if (!btn) return;
    audioMode = btn.dataset.mode;
    audioModeTabs.querySelectorAll('.sp-audio-tab').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    for (const [mode, panel] of Object.entries(audioPanels)) {
        panel.classList.toggle('sp-hidden', mode !== audioMode);
    }
    updateRailTts();
});

capXmlInput.addEventListener('change', () => {
    const file = capXmlInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            parseCapXml(e.target.result);
            capStatus.textContent = `Loaded: ${file.name}`;
            capStatus.className = 'sp-status-text ok';
        } catch (err) {
            capStatus.textContent = `Parse error: ${err.message}`;
            capStatus.className = 'sp-status-text err';
        }
    };
    reader.readAsText(file);
});

function getCurrentAudioFilePath() {
    if (generatedPreviewReady && generatedPreviewPath) return generatedPreviewPath;
    if (audioMode !== 'file') return '';
    return uploadedFilePath || '';
}

function buildAirPayload() {
    const feedIds = [...selectedFeedIds];
    const allEnabled = allFeedsData.filter((f) => f.enabled !== false).map((f) => f.id);
    const isAll = feedIds.length === allEnabled.length && allEnabled.every((id) => feedIds.includes(id));
    return {
        originator:       origSelect.value,
        event:            eventSelect.value,
        locations:        getAllSelectedCodes(),
        duration_hours:   parseInt(durationHours.value, 10) || 0,
        duration_minutes: parseInt(durationMins.value, 10) || 0,
        tone_type:        selectedTone,
        voice_message:    audioMode === 'tts'  ? voiceInput.value.trim() : '',
        audio_file_path:  generatedPreviewReady && generatedPreviewPath
            ? generatedPreviewPath
            : (audioMode === 'file' ? getCurrentAudioFilePath() : ''),
        air_on_all_feeds: isAll,
        feed_ids:         isAll ? [] : feedIds,
        feed_id:          feedIds[0] || '',
    };
}

async function generatePreviewAudio() {
    if (!getAllSelectedCodes().length) {
        showStatus('Select at least one location code.', 'err');
        return null;
    }
    const button = previewBtn;
    const previousLabel = button.textContent;
    button.disabled = true;
    button.textContent = 'Previewing…';
    try {
        const result = await apiPost('/same/generate', buildAirPayload());
        generatedPreviewPath = result.path || '';
        generatedPreviewUrl = result.download_url || '';
        const sessionToken = token.get();
        if (generatedPreviewUrl && sessionToken) {
            const separator = generatedPreviewUrl.includes('?') ? '&' : '?';
            generatedPreviewUrl = `${generatedPreviewUrl}${separator}token=${encodeURIComponent(sessionToken)}`;
        }
        generatedPreviewReady = true;
        generatedDraftSig = buildAlertSignature();
        if (generatedPreviewUrl) {
            previewAudio.src = generatedPreviewUrl;
            previewAudio.classList.remove('sp-hidden');
        }
        if (generatedPreviewUrl) {
            try {
                await previewAudio.play();
            } catch {
                // autoplay can be blocked; the user can still hit play
            }
            showStatus(`Preview ready: ${result.header}`, 'ok');
        }
        headerPreview.textContent = result.header;
        return result;
    } catch (err) {
        showStatus(`Generation failed: ${err.message}`, 'err');
        return null;
    } finally {
        button.disabled = false;
        button.textContent = previousLabel;
    }
}

function parseCapXml(xmlText) {
    const doc = new DOMParser().parseFromString(xmlText, 'application/xml');
    const ns  = 'urn:oasis:names:tc:emergency:cap:1.2';
    const get = (p, tag) => {
        const el = p.querySelector(tag)
            || p.getElementsByTagNameNS(ns, tag)[0]
            || p.getElementsByTagName(tag)[0];
        return el ? el.textContent.trim() : '';
    };

    const event    = get(doc, 'event');
    const expires  = get(doc, 'expires');
    const sent     = get(doc, 'sent');
    const geocodes = [...doc.getElementsByTagName('geocode'), ...doc.getElementsByTagNameNS(ns, 'geocode')];
    const sameCodes = geocodes
        .filter((gc) => { const vn = get(gc, 'valueName'); return vn === 'SAME' || vn.includes('SAME'); })
        .flatMap((gc) => get(gc, 'value').split(/\s+/).map((s) => s.replace(/^0/, '').padStart(6, '0')))
        .filter((s) => /^\d{6}$/.test(s));

    if (sameCodes.length) {
        const feedCodeKeys = new Set(getCodesForSelectedFeeds().keys());
        for (const code of [...new Set(sameCodes)]) {
            if (!feedCodeKeys.has(code) && !customLocEntries.some((e) => e.code === code)) {
                customLocEntries.push({ code, name: locationNames[code] || '' });
            }
            selectedLocCodes.add(code);
        }
    }

    if (event) {
        const upper = event.toUpperCase();
        const match = getEventCatalog().find((entry) => entry.description.toUpperCase() === upper || entry.code === upper);
        if (match) eventSelect.value = match.code;
    }

    if (expires && sent) {
        const diffMs = new Date(expires) - new Date(sent);
        if (Number.isFinite(diffMs) && diffMs > 0) {
            const totalMins = Math.round(diffMs / 60000);
            durationHours.value = Math.floor(totalMins / 60);
            durationMins.value  = totalMins % 60;
        }
    }
    renderLocPicker();
    updatePreview();
}

function handleFileSelect(file) {
    selectedFile     = file;
    uploadedFilePath = '';
    invalidateGeneratedPreview();
    uploadArea.classList.add('has-file');
    uploadPromptEl.classList.add('sp-hidden');
    uploadFilenameEl.textContent = `Selected: ${file.name}`;
    uploadFilenameEl.classList.remove('sp-hidden');
    uploadActions.classList.remove('sp-hidden');
    uploadStatusEl.textContent = '';
    uploadBtn.textContent = 'Upload & Encode';
}

async function uploadFile() {
    if (!selectedFile) return;
    uploadBtn.disabled    = true;
    uploadBtn.textContent = 'Encoding\u2026';
    uploadStatusEl.textContent = '';
    try {
        const fd = new FormData();
        fd.append('file', selectedFile);
        const result      = await apiPostForm('/same/upload-audio', fd);
        uploadedFilePath  = result.path;
        invalidateGeneratedPreview();
        uploadStatusEl.textContent = `Ready \u2014 encoded to ${result.sample_rate} Hz PCM WAV`;
        uploadBtn.textContent = 'Re-upload';
    } catch (err) {
        uploadStatusEl.textContent = `Upload failed: ${err.message}`;
        uploadBtn.textContent = 'Retry Upload';
        uploadedFilePath = '';
    } finally {
        uploadBtn.disabled = false;
    }
}

function clearUpload() {
    selectedFile     = null;
    uploadedFilePath = '';
    invalidateGeneratedPreview();
    audioFile.value  = '';
    uploadArea.classList.remove('has-file');
    uploadPromptEl.classList.remove('sp-hidden');
    uploadFilenameEl.classList.add('sp-hidden');
    uploadFilenameEl.textContent = '';
    uploadActions.classList.add('sp-hidden');
    uploadStatusEl.textContent = '';
    uploadBtn.textContent = 'Upload & Encode';
}

uploadArea.tabIndex = 0;
uploadArea.setAttribute('role', 'button');
uploadArea.setAttribute('aria-label', 'Choose an audio or video file to upload');
uploadArea.addEventListener('click', (event) => {
    if (event.target.closest('button, input, textarea, select, a')) return;
    audioFile.click();
});
uploadArea.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    audioFile.click();
});

audioFile.addEventListener('change', () => {
    if (audioFile.files[0]) handleFileSelect(audioFile.files[0]);
});
uploadBtn.addEventListener('click', uploadFile);
clearUploadBtn.addEventListener('click', clearUpload);
uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
});

function resetAirButton() {
    clearInterval(airConfirmTimer);
    airConfirmPending = false;
    airConfirmTimer   = null;
    airBtn.className  = 'btn-danger sp-air-btn';
    airBtn.textContent = 'Air Now';
    airBtn.disabled   = false;
}

function startAirConfirm() {
    airConfirmPending = true;
    let countdown     = 5;
    airBtn.className  = 'btn-confirm sp-air-btn';
    airBtn.textContent = `CONFIRM AIR \u2014 ${countdown}s`;
    airConfirmTimer   = setInterval(() => {
        countdown--;
        if (countdown <= 0) { clearInterval(airConfirmTimer); resetAirButton(); }
        else airBtn.textContent = `CONFIRM AIR \u2014 ${countdown}s`;
    }, 1000);
}

async function doAir() {
    const codes = getAllSelectedCodes();
    if (!codes.length) { showStatus('Select at least one location code.', 'err'); return; }
    if (audioMode === 'file' && !getCurrentAudioFilePath()) {
        showStatus('Upload and encode the audio file before airing.', 'err');
        return;
    }
    clearStatus();
    airBtn.disabled    = true;
    airBtn.textContent = 'Transmitting\u2026';
    try {
        const result   = await apiPost('/same/air', buildAirPayload());
        const feedLabel = (result.feeds_aired || [result.feed_id]).join(', ');
        showStatus(`Broadcast queued on ${feedLabel}: ${result.header}`, 'ok');
        headerPreview.textContent = result.header;
        addRecentBroadcast(result);
    } catch (err) {
        showStatus(`Failed to air: ${err.message}`, 'err');
    } finally {
        resetAirButton();
    }
}

airBtn.addEventListener('click', () => {
    if (TEST_CODES.has(eventSelect.value)) { doAir(); return; }
    if (airConfirmPending) { resetAirButton(); doAir(); }
    else { startAirConfirm(); }
});

previewBtn.addEventListener('click', () => generatePreviewAudio());

function addRecentBroadcast(result) {
    recentBroadcasts.unshift({ ...result, ts: new Date().toLocaleTimeString() });
    recentList.innerHTML = recentBroadcasts.slice(0, 16).map((b) => {
        const feedLabel = (b.feeds_aired || [b.feed_id]).join(', ');
        return `<article class="event-item sp-broadcast-item">
            <div class="event-head">
                <span class="event-kind">SAME</span>
                <span class="sp-broadcast-feed">${feedLabel}</span>
                <time>${b.ts}</time>
            </div>
            <p class="sp-broadcast-hdr">${b.header}</p>
        </article>`;
    }).join('');
}

function templateShowStatus(msg, type) {
    templateStatus.textContent = msg;
    templateStatus.className   = `status-banner ${type}`;
    if (type === 'ok') {
        setTimeout(() => {
            templateStatus.className   = 'status-banner';
            templateStatus.textContent = '';
        }, 3000);
    }
}

function renderTemplates() {
    if (!Object.keys(templateData).length) {
        templateList.innerHTML = '<p class="sp-tpl-empty">No templates defined. Click &quot;+ New&quot; to add one.</p>';
        return;
    }
    templateList.innerHTML = Object.entries(templateData).map(([code, tpl]) => {
        const same = tpl.same || {};
        const content = same.content || {};
        const msgs = tpl.msg || content.lang || {};
        const files = tpl.files || content.file || {};
        const langRows = Object.entries(msgs).map(([lang, text]) => `
            <div class="template-lang-row" data-code="${code}" data-lang="${lang}">
                <div class="form-row sp-tpl-lang-row">
                    <div class="form-group sp-tpl-lang-key-group">
                        <label>Language pattern</label>
                        <input type="text" class="tpl-lang-key" value="${lang}" placeholder="en*">
                    </div>
                    <div class="form-group sp-tpl-lang-text-group">
                        <label>Message text</label>
                        <textarea class="tpl-lang-text" rows="3">${text}</textarea>
                    </div>
                    <div class="form-group sp-tpl-lang-file-group">
                        <label>Audio file</label>
                        <input type="text" class="tpl-lang-file" value="${files[lang] || ''}" placeholder="audio/_uploads/example.wav">
                    </div>
                    <button type="button" class="btn-ghost tpl-remove-lang sp-tpl-lang-remove"
                        data-code="${code}" data-lang="${lang}">&#x2715;</button>
                </div>
            </div>`).join('');
        return `
            <div class="section-block sp-tpl-block" data-template-code="${code}">
                <div class="section-hd sp-tpl-hd">
                    <input type="text" class="tpl-code-input sp-tpl-code-input" value="${code}" placeholder="RWT">
                    <div class="section-hd-actions">
                        <button type="button" class="btn-action tpl-fire-btn" data-code="${code}">&#9654; Send Now</button>
                        <button type="button" class="btn-ghost tpl-add-lang" data-code="${code}">+ Language</button>
                        <button type="button" class="btn-ghost tpl-remove-tpl sp-tpl-remove-btn" data-code="${code}">Remove</button>
                    </div>
                </div>
                <div class="sp-template-head-grid">
                    <div class="sp-tpl-field-group">
                        <label class="sp-tpl-field-label">Template Name</label>
                        <input type="text" class="sp-tpl-text-input tpl-name" value="${tpl.name || code}">
                    </div>
                    <div class="sp-tpl-field-group">
                        <label class="sp-tpl-field-label">Description</label>
                        <input type="text" class="sp-tpl-text-input tpl-description" value="${tpl.description || ''}">
                    </div>
                </div>
                <div class="sp-tpl-meta-stack">
                    <label class="sp-tpl-checkbox-row">
                        <input type="checkbox" class="tpl-same-enabled" ${same.enabled !== false ? 'checked' : ''}>
                        <span class="sp-tpl-field-label">Enable SAME</span>
                    </label>
                    <label class="sp-tpl-checkbox-row">
                        <input type="checkbox" class="tpl-auto-enabled" ${tpl.automated?.enabled ? 'checked' : ''}>
                        <span class="sp-tpl-field-label">Automated</span>
                    </label>
                </div>
                <div class="sp-tpl-meta-stack">
                    <div class="sp-tpl-field-group">
                        <label class="sp-tpl-field-label">SAME event</label>
                        <input type="text" class="sp-tpl-text-input tpl-event" value="${tpl.sameEvent || same.event || code}">
                    </div>
                    <div class="sp-tpl-field-group">
                        <label class="sp-tpl-field-label">Expire</label>
                        <input type="text" class="sp-tpl-text-input tpl-expire" value="${tpl.sameExpire || '0015'}">
                    </div>
                </div>
                <div class="sp-tpl-meta-stack">
                    <div class="sp-tpl-field-group">
                        <label class="sp-tpl-field-label">Sender ID</label>
                        <input type="text" class="sp-tpl-text-input tpl-sender" value="${same.sender_id || ''}">
                    </div>
                    <div class="sp-tpl-field-group">
                        <label class="sp-tpl-field-label">Attention Tone</label>
                        <input type="text" class="sp-tpl-text-input tpl-tone" value="${content.attention_tone || ''}">
                    </div>
                </div>
                <div class="section-body sp-tpl-lang-body">
                    ${langRows || '<p class="sp-tpl-no-langs">No language messages.</p>'}
                </div>
            </div>`;
    }).join('');

    templateList.querySelectorAll('.tpl-remove-tpl').forEach((btn) => {
        btn.addEventListener('click', () => { delete templateData[btn.dataset.code]; renderTemplates(); });
    });
    templateList.querySelectorAll('.tpl-add-lang').forEach((btn) => {
        btn.addEventListener('click', () => {
            if (!templateData[btn.dataset.code].msg) templateData[btn.dataset.code].msg = {};
            templateData[btn.dataset.code].msg[`lang-${Date.now()}`] = '';
            renderTemplates();
        });
    });
    templateList.querySelectorAll('.tpl-remove-lang').forEach((btn) => {
        btn.addEventListener('click', () => {
            const { code, lang } = btn.dataset;
            if (templateData[code]?.msg) delete templateData[code].msg[lang];
            renderTemplates();
        });
    });
    templateList.querySelectorAll('.tpl-fire-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
            btn.disabled    = true;
            btn.textContent = 'Sending\u2026';
            try {
                await apiPost(`/same/test?event_code=${encodeURIComponent(btn.dataset.code)}`, {});
                btn.textContent = 'Sent \u2713';
                setTimeout(() => { btn.disabled = false; btn.innerHTML = '&#9654; Send Now'; }, 3000);
            } catch (err) {
                templateShowStatus(`Send failed: ${err.message}`, 'err');
                btn.disabled = false;
                btn.innerHTML = '&#9654; Send Now';
            }
        });
    });
}

function collectTemplateData() {
    const updated = cloneData(templateData);
    templateList.querySelectorAll('[data-template-code]').forEach((block) => {
        const origCode = block.dataset.templateCode;
        const code = (block.querySelector('.tpl-code-input')?.value || origCode).trim().toUpperCase();
        if (!code) return;
        const base = cloneData(updated[origCode] || makeEmptyTemplate(code));
        const expire = block.querySelector('.tpl-expire')?.value.trim() || base.sameExpire || '0015';
        const event  = block.querySelector('.tpl-event')?.value.trim().toUpperCase() || code;
        const msg    = {};
        const files  = {};
        block.querySelectorAll('.template-lang-row').forEach((row) => {
            const langKey = row.querySelector('.tpl-lang-key')?.value.trim() || '';
            const text    = row.querySelector('.tpl-lang-text')?.value || '';
            const file    = row.querySelector('.tpl-lang-file')?.value || '';
            if (langKey) {
                msg[langKey] = text;
                files[langKey] = file;
            }
        });
        delete updated[origCode];
        updated[code] = {
            ...base,
            name: block.querySelector('.tpl-name')?.value.trim() || base.name || code,
            description: block.querySelector('.tpl-description')?.value.trim() || base.description || '',
            automated: {
                ...(base.automated || {}),
                enabled: Boolean(block.querySelector('.tpl-auto-enabled')?.checked),
            },
            same: {
                ...(base.same || {}),
                enabled: Boolean(block.querySelector('.tpl-same-enabled')?.checked),
                event,
                sender_id: block.querySelector('.tpl-sender')?.value.trim() || '',
                duration: {
                    ...((base.same || {}).duration || {}),
                    hr: Math.max(0, Math.min(parseInt(expire.slice(0, 2), 10) || 0, 99)),
                    min: Math.max(0, Math.min(parseInt(expire.slice(2), 10) || 15, 59)),
                },
                content: {
                    ...((base.same || {}).content || {}),
                    attention_tone: block.querySelector('.tpl-tone')?.value.trim() || '',
                    lang: msg,
                    file: files,
                },
            },
            sameEvent: event,
            sameExpire: expire,
            msg,
            files,
        };
    });
    return updated;
}

async function loadTemplates() {
    try {
        templateData = await apiGet('/same/templates');
        renderTemplates();
    } catch {
        templateShowStatus('Could not load templates.', 'err');
    }
}

async function saveTemplates() {
    templateData = collectTemplateData();
    templateSaveBtn.disabled    = true;
    templateSaveBtn.textContent = 'Saving\u2026';
    try {
        await apiPut('/same/templates', { content: JSON.stringify(templateData, null, 2) });
        templateShowStatus('Templates saved.', 'ok');
        renderTemplates();
    } catch (err) {
        templateShowStatus(`Save failed: ${err.message}`, 'err');
    } finally {
        templateSaveBtn.disabled    = false;
        templateSaveBtn.textContent = 'Save All';
    }
}

templateAddBtn.addEventListener('click', () => {
    const code = `CUSTOM${Object.keys(templateData).length + 1}`;
    templateData[code] = makeEmptyTemplate(code);
    renderTemplates();
});
templateSaveBtn.addEventListener('click', saveTemplates);

if (!token.get()) {
    window.location.href = '/';
} else {
    Promise.all([
        (async () => {
            try {
                await fetch('/api/v1/health').then((r) => r.json());
                apiDot.dataset.state  = 'ok';
                healthPill.textContent = 'API healthy';
                const [summary, config] = await Promise.all([
                    apiGet('/summary'),
                    apiGet('/config').catch(() => ({})),
                ]);
                allFeedsData          = summary.feeds || [];
                configuredCallsign    = config.same?.sender || 'XXXXXXXX';
                if (allFeedsData.length) selectedFeedIds.add(allFeedsData[0].id);
                renderFeedPicker();
                renderLocPicker();
            } catch {
                apiDot.dataset.state   = 'err';
                healthPill.textContent = 'API unavailable';
                feedPicker.innerHTML   = '<p class="sp-picker-empty">Could not load feeds.</p>';
            }
        })(),
        apiGet('/same/event-codes').then((m) => { sameMapping     = m; }).catch(() => { sameMapping     = {}; }),
        apiGet('/same/location-names').then((n) => { locationNames = n; }).catch(() => { locationNames = {}; }),
        loadTemplates(),
    ]).then(() => {
        populateEventSelect();
        updatePreview();
        clearStatus();
        window.lucide?.createIcons();
    });
}


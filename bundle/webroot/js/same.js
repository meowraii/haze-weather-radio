import { apiCommand, apiGet } from './lib/api.js';
import { getDashboardState, refreshDashboard, waitForDashboardState } from './dashboard.js';

const TEST_CODES = new Set(['DMO', 'RWT', 'RMT']);

const ORIGINATORS = [
    { value: 'WXR', label: 'WXR - Weather' },
    { value: 'EAS', label: 'EAS - EAS Participant' },
    { value: 'CIV', label: 'CIV - Civil Authority' },
    { value: 'PEP', label: 'PEP - Primary Entry Point' },
];

const TONES = [
    { code: 'WXR', label: '1050 Hz' },
    { code: 'EAS', label: 'EAS dual tone' },
    { code: 'NPAS', label: 'Alert Ready' },
    { code: 'EGG_TIMER', label: 'Legacy Ontario' },
    { code: 'QUEBEC', label: 'Quebec' },
    { code: 'NONE', label: 'No tone' },
];

let allFeedsData = [];
let sameMapping = {};
let locationNames = {};
let alertTemplates = {};
let configuredCallsign = 'HAZE';
let selectedFeedIds = new Set();
let selectedLocationCodes = new Set();
let customLocations = [];
let selectedTone = 'WXR';
let introText = '';
let introTimer = null;
let confirmTimer = null;
let confirmPending = false;
let stateListenerBound = false;
let previewObjectUrl = '';

const TABLE_SIZE_KEYS = {
    feeds: 'haze.broadcastAlert.feedsTableHeight',
    locations: 'haze.broadcastAlert.locationsTableHeight',
};

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

function buildLayout() {
    document.getElementById('sameMain').innerHTML = `
        <div class="ba-stack">
            <section class="section-block ba-panel">
                <div class="section-hd">
                    <span>Alert Details</span>
                    <span class="section-hd-sub">SAME metadata and spoken alert text.</span>
                </div>
                <div class="section-body ba-form-grid">
                    <label class="ba-field ba-template-field">
                        <span>Alert template</span>
                        <select id="baTemplate">
                            <option value="">Custom alert</option>
                        </select>
                    </label>
                    <label class="ba-field">
                        <span>Originator</span>
                        <select id="baOriginator">
                            ${ORIGINATORS.map((item) => `<option value="${item.value}">${item.label}</option>`).join('')}
                        </select>
                    </label>
                    <label class="ba-field ba-field-wide">
                        <span>Event</span>
                        <select id="baEvent"></select>
                    </label>
                    <label class="ba-field">
                        <span>Duration</span>
                        <div class="ba-duration-row">
                            <input id="baHours" type="number" min="0" max="99" value="1" aria-label="Hours">
                            <input id="baMinutes" type="number" min="0" max="59" value="0" aria-label="Minutes">
                        </div>
                    </label>
                </div>
            </section>

            <section class="section-block ba-panel">
                <div class="section-hd">
                    <span>Routing</span>
                    <span class="section-hd-sub">Compact feed and location targets.</span>
                </div>
                <div class="section-body ba-routing-grid">
                    <div class="ba-table-box">
                        <div class="ba-table-title">
                            <span>Feeds</span>
                            <strong id="baFeedCount">0</strong>
                            <div class="ba-mini-actions">
                                <button id="baFeedsAll" class="btn-action" type="button">All</button>
                                <button id="baFeedsNone" class="btn-action" type="button">None</button>
                            </div>
                        </div>
                        <div class="ba-table-scroll" id="baFeedScroll">
                            <table class="ba-table">
                                <thead>
                                    <tr><th></th><th>Feed</th><th>ID</th><th>Queue</th></tr>
                                </thead>
                                <tbody id="baFeedRows"></tbody>
                            </table>
                        </div>
                    </div>
                    <div class="ba-table-box">
                        <div class="ba-table-title">
                            <span>SAME Locations</span>
                            <strong id="baLocationCount">0</strong>
                            <div class="ba-mini-actions">
                                <button id="baLocationsAll" class="btn-action" type="button">All</button>
                                <button id="baLocationsNone" class="btn-action" type="button">None</button>
                            </div>
                        </div>
                        <div class="ba-table-scroll ba-location-scroll" id="baLocationScroll">
                            <table class="ba-table">
                                <thead>
                                    <tr><th></th><th>Area</th><th>SAME Location</th><th>Region</th></tr>
                                </thead>
                                <tbody id="baLocationRows"></tbody>
                            </table>
                        </div>
                        <div class="ba-custom-location">
                            <input id="baCustomCode" type="text" inputmode="numeric" maxlength="6" placeholder="SAME Location">
                            <input id="baCustomName" type="text" placeholder="Name">
                            <button id="baAddCustom" class="btn-action" type="button">Add</button>
                        </div>
                    </div>
                </div>
            </section>

            <section class="section-block ba-panel">
                <div class="section-hd">
                    <span>Broadcast Audio</span>
                    <span class="section-hd-sub">Priority alert audio rendered by the playlist service.</span>
                </div>
                <div class="section-body ba-audio-grid">
                    <div class="ba-option-group">
                        <label class="ba-switch">
                            <input id="baIncludeSame" type="checkbox" checked>
                            <span>Enable SAME</span>
                        </label>
                        <label class="ba-switch">
                            <input id="baPrependIntro" type="checkbox" checked>
                            <span>Prepend SAME-to-text intro</span>
                        </label>
                        <div class="ba-tone-list" id="baToneList">
                            ${TONES.map((tone, index) => `
                                <button class="ba-tone-btn${index === 0 ? ' active' : ''}" type="button" data-tone="${tone.code}">
                                    <strong>${tone.code}</strong>
                                    <span>${tone.label}</span>
                                </button>
                            `).join('')}
                        </div>
                    </div>
                    <div class="ba-message-box">
                        <label class="ba-field">
                            <span>SAME-to-text intro</span>
                            <textarea id="baIntro" rows="3" readonly></textarea>
                        </label>
                        <label class="ba-field">
                            <span>Message text</span>
                            <textarea id="baMessage" rows="6" placeholder="Custom alert text after the generated intro"></textarea>
                        </label>
                    </div>
                </div>
            </section>

            <section class="section-block ba-panel">
                <div class="section-hd">
                    <span>Timing</span>
                    <span class="section-hd-sub">Immediate or scheduled priority insertion.</span>
                </div>
                <div class="section-body ba-timing-row">
                    <label class="ba-switch">
                        <input id="baScheduleEnabled" type="checkbox">
                        <span>Schedule for later</span>
                    </label>
                    <label class="ba-field ba-schedule-field">
                        <span>Start time</span>
                        <input id="baScheduleAt" type="datetime-local" disabled>
                    </label>
                </div>
            </section>
        </div>
    `;

    document.getElementById('sameRail').innerHTML = `
        <aside class="ba-rail-card">
            <section class="ba-rail-section">
                <span class="ba-label">Plan</span>
                <dl class="ba-plan-list">
                    <div><dt>Event</dt><dd id="baPlanEvent">ADR</dd></div>
                    <div><dt>Feeds</dt><dd id="baPlanFeeds">0</dd></div>
                    <div><dt>Locations</dt><dd id="baPlanLocations">0</dd></div>
                    <div><dt>SAME</dt><dd id="baPlanSame">Enabled</dd></div>
                    <div><dt>Timing</dt><dd id="baPlanTiming">Now</dd></div>
                </dl>
            </section>
            <section class="ba-rail-section">
                <span class="ba-label">SAME Header</span>
                <code id="baHeaderPreview" class="ba-header-preview">Select a feed and location.</code>
            </section>
            <section class="ba-rail-section">
                <span class="ba-label">SAME-to-text intro</span>
                <p id="baRailIntro" class="ba-intro-preview">Waiting for alert details.</p>
            </section>
            <section class="ba-rail-section">
                <p id="baStatus" class="status-banner ba-status"></p>
                <div class="ba-action-grid">
                    <button id="baPreviewSame" class="btn-action" type="button">Preview alert audio</button>
                    <button id="baQueue" class="btn-danger" type="button">Broadcast</button>
                </div>
                <audio id="baPreviewPlayer" class="ba-preview-player" controls hidden></audio>
            </section>
        </aside>
`;
}

buildLayout();

const templateSelect = document.getElementById('baTemplate');
const originator = document.getElementById('baOriginator');
const eventSelect = document.getElementById('baEvent');
const hoursInput = document.getElementById('baHours');
const minutesInput = document.getElementById('baMinutes');
const feedRows = document.getElementById('baFeedRows');
const locationRows = document.getElementById('baLocationRows');
const feedScroll = document.getElementById('baFeedScroll');
const locationScroll = document.getElementById('baLocationScroll');
const feedCount = document.getElementById('baFeedCount');
const locationCount = document.getElementById('baLocationCount');
const customCode = document.getElementById('baCustomCode');
const customName = document.getElementById('baCustomName');
const addCustom = document.getElementById('baAddCustom');
const includeSame = document.getElementById('baIncludeSame');
const prependIntro = document.getElementById('baPrependIntro');
const toneList = document.getElementById('baToneList');
const introBox = document.getElementById('baIntro');
const messageBox = document.getElementById('baMessage');
const scheduleEnabled = document.getElementById('baScheduleEnabled');
const scheduleAt = document.getElementById('baScheduleAt');
const planEvent = document.getElementById('baPlanEvent');
const planFeeds = document.getElementById('baPlanFeeds');
const planLocations = document.getElementById('baPlanLocations');
const planSame = document.getElementById('baPlanSame');
const planTiming = document.getElementById('baPlanTiming');
const headerPreview = document.getElementById('baHeaderPreview');
const railIntro = document.getElementById('baRailIntro');
const statusBanner = document.getElementById('baStatus');
const previewSame = document.getElementById('baPreviewSame');
const queueButton = document.getElementById('baQueue');
const previewPlayer = document.getElementById('baPreviewPlayer');

function eventCatalog() {
    const eas = sameMapping.eas || {};
    return Object.entries(eas)
        .map(([code, description]) => ({ code, description }))
        .sort((a, b) => a.code.localeCompare(b.code));
}

function templateName(key, template) {
    const event = template?.same?.event || template?.sameEvent || key;
    const name = template?.name || event || key;
    return `${name}${event && event !== name ? ` (${event})` : ''}`;
}

function populateTemplateSelect() {
    const keys = Object.keys(alertTemplates || {}).sort((a, b) => templateName(a, alertTemplates[a]).localeCompare(templateName(b, alertTemplates[b])));
    templateSelect.innerHTML = [
        '<option value="">Custom alert</option>',
        ...keys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(templateName(key, alertTemplates[key]))}</option>`),
    ].join('');
}

function populateEventSelect() {
    const events = eventCatalog();
    eventSelect.innerHTML = events.map((item) => (
        `<option value="${escapeHtml(item.code)}">${escapeHtml(item.code)} - ${escapeHtml(item.description || item.code)}</option>`
    )).join('');
    if (events.some((item) => item.code === 'DMO')) {
        eventSelect.value = 'DMO';
    }
}

function templateLocations(template) {
    return (template?.same?.locations || [])
        .map((location) => normalizeLocationCode(location?.id || location))
        .filter(Boolean);
}

function templateFeedIDs(template) {
    return (template?.automated?.target?.feed_ids || [])
        .map((id) => String(id || '').trim())
        .filter(Boolean);
}

function templateMessage(template) {
    return String(template?.msg?.en || template?.same?.content?.lang?.en || '');
}

function applyTemplate(key) {
    const template = alertTemplates[key];
    if (!template) return;
    const same = template.same || {};
    const content = same.content || {};
    const duration = same.duration || {};
    const event = String(same.event || template.sameEvent || key || '').toUpperCase();
    if (event) eventSelect.value = event;
    if (same.originator) originator.value = String(same.originator).toUpperCase();
    hoursInput.value = Number(duration.hr ?? String(template.sameExpire || '0015').slice(0, 2)) || 0;
    minutesInput.value = Number(duration.min ?? String(template.sameExpire || '0015').slice(2, 4)) || 15;
    const tone = String(content.attention_tone || same.tone || 'WXR').toUpperCase();
    selectedTone = tone;
    toneList.querySelectorAll('.ba-tone-btn').forEach((item) => item.classList.toggle('active', item.dataset.tone === tone));
    const message = templateMessage(template);
    if (message) messageBox.value = message;

    const templateFeeds = templateFeedIDs(template);
    if (templateFeeds.length) {
        if (templateFeeds.includes('*')) {
            selectedFeedIds = new Set(allFeedsData.filter((feed) => feed.enabled !== false).map((feed) => feed.id));
        } else {
            selectedFeedIds = new Set(templateFeeds);
        }
        renderFeeds();
    }

    const locations = templateLocations(template);
    if (locations.length) {
        selectedLocationCodes = normalizeSelectedLocationSet(locations);
        renderLocations();
    }
    updateAll();
    setStatus(`Loaded template: ${templateName(key, template)}.`, 'ok');
}

function eventName(code) {
    return String((sameMapping.eas || {})[String(code || '').toUpperCase()] || code || 'Alert');
}

function feedName(feed) {
    return String(feed?.name || feed?.id || 'Feed');
}

function normalizeLocationCode(value) {
    const digits = String(value || '').replace(/\D/g, '');
    if (digits.length === 5) return `0${digits}`;
    if (digits.length === 6) return digits;
    return '';
}

function locationName(code) {
    const normalized = normalizeLocationCode(code);
    if (normalized === '000000') return 'All areas';
    return String(locationNames[normalized] || customLocations.find((item) => item.code === normalized)?.name || normalized);
}

function selectedFeeds() {
    return allFeedsData.filter((feed) => selectedFeedIds.has(feed.id));
}

function feedCoverageRows(feed) {
    const rows = [];
    if (feed.same_all_locations && Array.isArray(feed.same_locations) && feed.same_locations.length) {
        return [...new Set(feed.same_locations.map(normalizeLocationCode).filter(Boolean))]
            .map((code) => ({
                code,
                name: locationName(code),
                region: code === '000000' ? 'National / all locations' : feedName(feed),
                feedID: feed.id,
            }));
    }
    const regions = Array.isArray(feed.coverage_regions) ? feed.coverage_regions : [];
    for (const region of regions) {
        const regionName = String(region.name || region.id || feedName(feed));
        const regionCode = normalizeLocationCode(region.id);
        if (regionCode) rows.push({ code: regionCode, name: regionName, region: 'Forecast region', feedID: feed.id });
        const subregions = Array.isArray(region.subregions) ? region.subregions : [];
        if (!subregions.length) {
            continue;
        }
        for (const item of subregions) {
            const code = normalizeLocationCode(item.id);
            if (!code) continue;
            rows.push({
                code,
                name: String(item.name || locationName(code)),
                region: regionName,
                feedID: feed.id,
            });
        }
    }
    if (rows.length) return rows;
    const fallback = Array.isArray(feed.same_locations) && feed.same_locations.length ? feed.same_locations : (feed.clc_codes || []);
    return [...new Set(fallback.map(normalizeLocationCode).filter(Boolean))]
        .map((code) => ({ code, name: locationName(code), region: feedName(feed), feedID: feed.id }));
}

function normalizeSelectedLocationSet(codes) {
    const normalized = [...codes].map(normalizeLocationCode).filter(Boolean);
    if (normalized.includes('000000')) return new Set(['000000']);
    return new Set(normalized);
}

function targetLocationRows() {
    const seen = new Set();
    const rows = [];
    for (const feed of selectedFeeds()) {
        for (const row of feedCoverageRows(feed)) {
            if (seen.has(row.code)) continue;
            seen.add(row.code);
            rows.push(row);
        }
    }
    for (const item of customLocations) {
        if (seen.has(item.code)) continue;
        seen.add(item.code);
        rows.push({ ...item, region: 'Custom', feedID: '' });
    }
    return rows;
}

function selectedAreaNames() {
    const byCode = new Map(targetLocationRows().map((row) => [row.code, row.name]));
    return [...selectedLocationCodes].map((code) => byCode.get(code) || locationName(code)).filter(Boolean);
}

function safeStorageGet(key) {
    try {
        return window.localStorage.getItem(key);
    } catch {
        return '';
    }
}

function safeStorageSet(key, value) {
    try {
        window.localStorage.setItem(key, value);
    } catch {
        // Storage can be unavailable in hardened/private contexts.
    }
}

function applySavedTableSize(element, key, fallbackPx) {
    if (!element) return;
    const saved = parseInt(safeStorageGet(key), 10);
    const height = Number.isFinite(saved) && saved >= 120 ? saved : fallbackPx;
    element.style.height = `${height}px`;
}

function bindResizableTableSize(element, key) {
    if (!element || !window.ResizeObserver) return;
    const observer = new ResizeObserver((entries) => {
        const height = Math.round(entries[0]?.contentRect?.height || 0);
        if (height >= 120) safeStorageSet(key, String(height));
    });
    observer.observe(element);
}

function initResizableTables() {
    applySavedTableSize(feedScroll, TABLE_SIZE_KEYS.feeds, 210);
    applySavedTableSize(locationScroll, TABLE_SIZE_KEYS.locations, 290);
    bindResizableTableSize(feedScroll, TABLE_SIZE_KEYS.feeds);
    bindResizableTableSize(locationScroll, TABLE_SIZE_KEYS.locations);
}

function renderFeeds() {
    if (!allFeedsData.length) {
        feedRows.innerHTML = '<tr><td colspan="4" class="ba-empty-cell">No feeds configured.</td></tr>';
        return;
    }
    feedRows.innerHTML = allFeedsData.map((feed) => {
        const checked = selectedFeedIds.has(feed.id) ? 'checked' : '';
        return `<tr>
            <td><input class="ba-feed-check" type="checkbox" value="${escapeHtml(feed.id)}" ${checked}></td>
            <td>${escapeHtml(feedName(feed))}</td>
            <td><code>${escapeHtml(feed.id)}</code></td>
            <td>${escapeHtml(feed.alert_queue_depth ?? 0)}</td>
        </tr>`;
    }).join('');
    feedRows.querySelectorAll('.ba-feed-check').forEach((input) => {
        input.addEventListener('change', () => {
            if (input.checked) selectedFeedIds.add(input.value);
            else selectedFeedIds.delete(input.value);
            renderLocations();
            updateAll();
        });
    });
    feedCount.textContent = selectedFeedIds.size;
}

function renderLocations() {
    const rows = targetLocationRows();
    const availableCodes = new Set(rows.map((row) => row.code));
    selectedLocationCodes = normalizeSelectedLocationSet([...selectedLocationCodes].filter((code) => availableCodes.has(code)));
    if (!rows.length) {
        locationRows.innerHTML = '<tr><td colspan="4" class="ba-empty-cell">No locations available.</td></tr>';
        locationCount.textContent = '0';
        return;
    }
    locationRows.innerHTML = rows.map((row) => {
        const checked = selectedLocationCodes.has(row.code) ? 'checked' : '';
        return `<tr>
            <td><input class="ba-location-check" type="checkbox" value="${escapeHtml(row.code)}" ${checked}></td>
            <td title="${escapeHtml(row.name)}">${escapeHtml(row.name)}</td>
            <td><code>${escapeHtml(row.code)}</code></td>
            <td title="${escapeHtml(row.region)}">${escapeHtml(row.region)}</td>
        </tr>`;
    }).join('');
    locationRows.querySelectorAll('.ba-location-check').forEach((input) => {
        input.addEventListener('change', () => {
            if (input.checked) selectedLocationCodes.add(input.value);
            else selectedLocationCodes.delete(input.value);
            selectedLocationCodes = normalizeSelectedLocationSet(selectedLocationCodes);
            if (input.value === '000000' || selectedLocationCodes.has('000000')) {
                renderLocations();
            }
            updateAll();
        });
    });
    locationCount.textContent = selectedLocationCodes.size;
}

function durationCode() {
    const hours = Math.max(0, Math.min(parseInt(hoursInput.value, 10) || 0, 99));
    const minutes = Math.max(0, Math.min(parseInt(minutesInput.value, 10) || 0, 59));
    return `${String(hours).padStart(2, '0')}${String(minutes).padStart(2, '0')}`;
}

function headerText() {
    const locs = selectedLocationCodes.has('000000') ? ['000000'] : [...selectedLocationCodes];
    const locText = (locs.length ? locs.slice(0, 31) : ['000000']).join('-');
    const now = new Date();
    const start = new Date(now.getFullYear(), 0, 0);
    const day = String(Math.floor((now - start) / 86400000)).padStart(3, '0');
    const utcHour = String(now.getUTCHours()).padStart(2, '0');
    const utcMinute = String(now.getUTCMinutes()).padStart(2, '0');
    const callsign = configuredCallsign.replace(/-/g, '/').padEnd(8).slice(0, 8);
    return `ZCZC-${originator.value}-${eventSelect.value}-${locText}+${durationCode()}-${day}${utcHour}${utcMinute}-${callsign}-`;
}

function scheduleISOString() {
    if (!scheduleEnabled.checked || !scheduleAt.value) return '';
    const date = new Date(scheduleAt.value);
    if (Number.isNaN(date.getTime())) return '';
    return date.toISOString();
}

function payload() {
    const feedIds = [...selectedFeedIds];
    const locs = selectedLocationCodes.has('000000') ? ['000000'] : [...selectedLocationCodes];
    return {
        originator: originator.value,
        event: eventSelect.value,
        same_event: eventSelect.value,
        locations: locs,
        area_names: selectedAreaNames(),
        duration: durationCode(),
        duration_hours: parseInt(hoursInput.value, 10) || 0,
        duration_minutes: parseInt(minutesInput.value, 10) || 0,
        include_same: includeSame.checked,
        tone_type: selectedTone,
        prepend_same_translation: prependIntro.checked,
        voice_message: messageBox.value.trim(),
        alert_text: messageBox.value.trim(),
        feed_ids: feedIds,
        feed_id: feedIds[0] || '',
        callsign: configuredCallsign,
        schedule_at: scheduleISOString(),
        mimic_endec: 'SAGE',
    };
}

function setStatus(message, kind = '') {
    statusBanner.textContent = message || '';
    statusBanner.className = `status-banner ba-status ${kind}`.trim();
}

function fallbackIntro() {
    const areas = selectedAreaNames();
    const areaText = areas.length > 1
        ? `${areas.slice(0, -1).join(', ')}, and ${areas.at(-1)}`
        : (areas[0] || 'the selected area');
    return `Environment Canada has issued a ${eventName(eventSelect.value)} for ${areaText}.`;
}

function refreshIntroSoon() {
    if (introTimer) window.clearTimeout(introTimer);
    introTimer = window.setTimeout(refreshIntro, 180);
}

async function refreshIntro() {
    introTimer = null;
    const current = payload();
    try {
        const result = await apiCommand('same.intro', current, 8000);
        introText = result.intro || fallbackIntro();
    } catch {
        introText = fallbackIntro();
    }
    introBox.value = introText;
    railIntro.textContent = introText;
}

function updateAll() {
    feedCount.textContent = selectedFeedIds.size;
    locationCount.textContent = selectedLocationCodes.size;
    planEvent.textContent = `${eventSelect.value} - ${eventName(eventSelect.value)}`;
    planFeeds.textContent = String(selectedFeedIds.size);
    planLocations.textContent = String(selectedLocationCodes.size);
    planSame.textContent = includeSame.checked ? `${selectedTone}` : 'Disabled';
    planTiming.textContent = scheduleEnabled.checked && scheduleAt.value ? new Date(scheduleAt.value).toLocaleString() : 'Now';
    headerPreview.textContent = includeSame.checked ? headerText() : 'SAME disabled for this broadcast.';
    queueButton.textContent = scheduleEnabled.checked ? 'Schedule Alert' : 'Broadcast';
    refreshIntroSoon();
}

function resetConfirm() {
    if (confirmTimer) window.clearInterval(confirmTimer);
    confirmTimer = null;
    confirmPending = false;
    queueButton.className = 'btn-danger';
    queueButton.disabled = false;
    queueButton.textContent = scheduleEnabled.checked ? 'Schedule Alert' : 'Broadcast';
}

function startConfirm() {
    confirmPending = true;
    let remaining = 5;
    queueButton.className = 'btn-confirm';
    queueButton.textContent = `Confirm - ${remaining}s`;
    confirmTimer = window.setInterval(() => {
        remaining -= 1;
        if (remaining <= 0) {
            resetConfirm();
        } else {
            queueButton.textContent = `Confirm - ${remaining}s`;
        }
    }, 1000);
}

function validatePayload() {
    if (!selectedFeedIds.size) return 'Select at least one feed.';
    if (!selectedLocationCodes.size) return 'Select at least one location.';
    if (!includeSame.checked && !messageBox.value.trim() && !prependIntro.checked) {
        return 'Add message text or enable the generated intro.';
    }
    if (scheduleEnabled.checked) {
        const date = new Date(scheduleAt.value);
        if (Number.isNaN(date.getTime()) || date <= new Date()) {
            return 'Choose a future schedule time.';
        }
    }
    return '';
}

async function queueBroadcast() {
    const validation = validatePayload();
    if (validation) {
        setStatus(validation, 'err');
        return;
    }
    setStatus('');
    queueButton.disabled = true;
    queueButton.textContent = scheduleEnabled.checked ? 'Scheduling...' : 'Queueing...';
    try {
        const result = await apiCommand('alert.broadcast', payload(), 20000);
        if (result.scheduled) {
            setStatus(`Scheduled for ${new Date(result.schedule_at).toLocaleString()}.`, 'ok');
        } else {
            setStatus(`Priority alert requested for ${(result.feed_ids || []).join(', ')}.`, 'ok');
        }
    } catch (err) {
        setStatus(`Broadcast failed: ${err.message}`, 'err');
    } finally {
        resetConfirm();
        refreshDashboard({ force: true }).catch(() => {});
    }
}

async function previewSameAudio() {
    const validation = validatePayload();
    if (validation) {
        setStatus(validation, 'err');
        return;
    }
    previewSame.disabled = true;
    previewSame.textContent = 'Generating...';
    try {
        const result = await apiCommand('alert.preview', payload(), 120000);
        if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
        const bytes = Uint8Array.from(atob(result.audio_base64 || ''), (char) => char.charCodeAt(0));
        previewObjectUrl = URL.createObjectURL(new Blob([bytes], { type: result.content_type || 'audio/wav' }));
        previewPlayer.src = previewObjectUrl;
        previewPlayer.hidden = false;
        headerPreview.textContent = result.same_header || (includeSame.checked ? headerText() : 'SAME disabled for this broadcast.');
        setStatus('Alert audio preview ready.', 'ok');
        previewPlayer.play().catch(() => {});
    } catch (err) {
        setStatus(`Preview failed: ${err.message}`, 'err');
    } finally {
        previewSame.disabled = false;
        previewSame.textContent = 'Preview alert audio';
    }
}

function applyPanelState(panelState) {
    const summary = panelState?.summary || {};
    const config = panelState?.config || {};
    allFeedsData = summary.feeds || [];
    configuredCallsign = config.same?.sender || configuredCallsign || 'HAZE';
    if (!selectedFeedIds.size && allFeedsData.length) {
        const firstFeed = allFeedsData.find((feed) => feed.enabled !== false) || allFeedsData[0];
        selectedFeedIds.add(firstFeed.id);
        const firstLocation = feedCoverageRows(firstFeed).find((row) => row.code);
        if (firstLocation) selectedLocationCodes.add(firstLocation.code);
    }
    renderFeeds();
    renderLocations();
    updateAll();
}

function bindStateListener() {
    if (stateListenerBound) return;
    stateListenerBound = true;
    window.addEventListener('haze:admin-state', (event) => applyPanelState(event.detail || {}));
}

function bindEvents() {
    document.getElementById('baFeedsAll').addEventListener('click', () => {
        selectedFeedIds = new Set(allFeedsData.filter((feed) => feed.enabled !== false).map((feed) => feed.id));
        renderFeeds();
        renderLocations();
        updateAll();
    });
    document.getElementById('baFeedsNone').addEventListener('click', () => {
        selectedFeedIds.clear();
        selectedLocationCodes.clear();
        renderFeeds();
        renderLocations();
        updateAll();
    });
    document.getElementById('baLocationsAll').addEventListener('click', () => {
        const rows = targetLocationRows();
        selectedLocationCodes = rows.some((row) => row.code === '000000')
            ? new Set(['000000'])
            : new Set(rows.map((row) => row.code));
        renderLocations();
        updateAll();
    });
    document.getElementById('baLocationsNone').addEventListener('click', () => {
        selectedLocationCodes.clear();
        renderLocations();
        updateAll();
    });
    addCustom.addEventListener('click', () => {
        const code = normalizeLocationCode(customCode.value);
        if (!code) return;
        if (!customLocations.some((item) => item.code === code)) {
            customLocations.push({ code, name: customName.value.trim() || locationName(code) || code });
        }
        selectedLocationCodes.add(code);
        selectedLocationCodes = normalizeSelectedLocationSet(selectedLocationCodes);
        customCode.value = '';
        customName.value = '';
        renderLocations();
        updateAll();
    });
    templateSelect.addEventListener('change', () => {
        if (!templateSelect.value) {
            setStatus('Using custom alert details.', 'ok');
            return;
        }
        applyTemplate(templateSelect.value);
    });
    toneList.addEventListener('click', (event) => {
        const button = event.target.closest('.ba-tone-btn');
        if (!button) return;
        selectedTone = button.dataset.tone;
        toneList.querySelectorAll('.ba-tone-btn').forEach((item) => item.classList.toggle('active', item === button));
        updateAll();
    });
    scheduleEnabled.addEventListener('change', () => {
        scheduleAt.disabled = !scheduleEnabled.checked;
        updateAll();
    });
    queueButton.addEventListener('click', () => {
        if (!TEST_CODES.has(eventSelect.value) && !confirmPending) {
            startConfirm();
            return;
        }
        queueBroadcast();
    });
    previewSame.addEventListener('click', previewSameAudio);
    [originator, eventSelect, hoursInput, minutesInput, includeSame, prependIntro, messageBox, scheduleAt]
        .forEach((element) => {
            element.addEventListener('input', updateAll);
            element.addEventListener('change', updateAll);
        });
}

bindEvents();
initResizableTables();

export function initSameView() {
    bindStateListener();
    Promise.all([
        apiGet('/same/event-codes').then((mapping) => { sameMapping = mapping; }).catch(() => { sameMapping = {}; }),
        apiGet('/same/location-names').then((names) => { locationNames = names; }).catch(() => { locationNames = {}; }),
        apiGet('/same/templates').then((templates) => { alertTemplates = templates || {}; }).catch(() => { alertTemplates = {}; }),
        (async () => {
            try {
                refreshDashboard().catch(() => {});
                return getDashboardState() || await waitForDashboardState();
            } catch {
                refreshDashboard({ force: true }).catch(() => {});
                return null;
            }
        })(),
    ]).then(([, , , panelState]) => {
        populateEventSelect();
        populateTemplateSelect();
        if (panelState) applyPanelState(panelState);
        else {
            renderFeeds();
            renderLocations();
            updateAll();
        }
        window.lucide?.createIcons();
    });
}

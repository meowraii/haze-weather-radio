import { panelClient } from './lib/ws-client.js';

const statusBanner = document.getElementById('dictionaryStatusBanner');
const groupCountEl = document.getElementById('dictionaryGroupCount');
const entryCountEl = document.getElementById('dictionaryEntryCount');
const pathEl = document.getElementById('dictionaryPath');
const groupSelect = document.getElementById('dictionaryGroupSelect');
const searchInput = document.getElementById('dictionarySearchInput');
const entriesBody = document.getElementById('dictionaryEntriesBody');
const rawEditor = document.getElementById('dictionaryRawEditor');
const newGroupInput = document.getElementById('dictionaryNewGroupInput');
const addGroupButton = document.getElementById('dictionaryAddGroupButton');
const deleteGroupButton = document.getElementById('dictionaryDeleteGroupButton');
const addEntryButton = document.getElementById('dictionaryAddEntryButton');
const applyRawButton = document.getElementById('dictionaryApplyRawButton');
const saveButton = document.getElementById('dictionarySaveButton');
const reloadButton = document.getElementById('dictionaryReloadButton');

let bound = false;
let dictionaryGroups = {};
let selectedGroup = '*';
let dirty = false;
let rendering = false;

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;',
    }[char]));
}

function setStatus(message, state = '') {
    statusBanner.textContent = message;
    statusBanner.className = `status-banner${state ? ` ${state}` : ''}`;
}

function sortedGroups() {
    const groups = Object.keys(dictionaryGroups);
    groups.sort((a, b) => {
        if (a === '*') return -1;
        if (b === '*') return 1;
        return a.localeCompare(b);
    });
    return groups;
}

function sortedEntries(group) {
    const entries = dictionaryGroups[group] || {};
    return Object.entries(entries).sort(([left], [right]) => left.localeCompare(right));
}

function countEntries() {
    return Object.values(dictionaryGroups).reduce((total, entries) => total + Object.keys(entries || {}).length, 0);
}

function normalizedGroups(groups) {
    const next = {};
    Object.entries(groups || {}).forEach(([group, entries]) => {
        const cleanGroup = String(group || '').trim();
        if (!cleanGroup || !entries || typeof entries !== 'object' || Array.isArray(entries)) return;
        next[cleanGroup] = {};
        Object.entries(entries).forEach(([match, replacement]) => {
            const cleanMatch = String(match || '').trim();
            const cleanReplacement = String(replacement || '').trim();
            if (cleanMatch && cleanReplacement) next[cleanGroup][cleanMatch] = cleanReplacement;
        });
    });
    if (!Object.keys(next).length) next['*'] = {};
    return next;
}

function groupsForServer() {
    collectCurrentGroup();
    return normalizedGroups(dictionaryGroups);
}

function renderMetrics() {
    groupCountEl.textContent = String(Object.keys(dictionaryGroups).length);
    entryCountEl.textContent = String(countEntries());
}

function renderGroupSelect() {
    const groups = sortedGroups();
    if (!groups.includes(selectedGroup)) selectedGroup = groups[0] || '*';
    groupSelect.innerHTML = groups.map((group) => (
        `<option value="${escapeHtml(group)}" ${group === selectedGroup ? 'selected' : ''}>${escapeHtml(group)}</option>`
    )).join('');
}

function renderRawEditor() {
    rawEditor.value = `${JSON.stringify(groupsForDisplay(), null, 2)}\n`;
}

function groupsForDisplay() {
    const out = {};
    sortedGroups().forEach((group) => {
        out[group] = {};
        sortedEntries(group).forEach(([match, replacement]) => {
            out[group][match] = replacement;
        });
    });
    return out;
}

function renderEntries() {
    const filter = searchInput.value.trim().toLowerCase();
    const rows = sortedEntries(selectedGroup).filter(([match, replacement]) => {
        if (!filter) return true;
        return match.toLowerCase().includes(filter) || replacement.toLowerCase().includes(filter);
    });
    if (!rows.length) {
        entriesBody.innerHTML = `<tr><td colspan="3" class="dictionary-empty panel-empty-cell">${filter ? 'No matching entries.' : 'No entries in this group.'}</td></tr>`;
        return;
    }
    entriesBody.innerHTML = rows.map(([match, replacement]) => `
        <tr>
            <td><input class="dictionary-entry-match" type="text" value="${escapeHtml(match)}" aria-label="Match"></td>
            <td><input class="dictionary-entry-replacement" type="text" value="${escapeHtml(replacement)}" aria-label="Replacement"></td>
            <td>
                <button class="sidebar-icon-btn dictionary-delete-entry" type="button" title="Delete entry" aria-label="Delete entry">
                    <i data-lucide="x" width="13" height="13"></i>
                </button>
            </td>
        </tr>
    `).join('');
    window.lucide?.createIcons();
}

function renderAll({ syncRaw = true } = {}) {
    rendering = true;
    renderMetrics();
    renderGroupSelect();
    renderEntries();
    if (syncRaw) renderRawEditor();
    rendering = false;
}

function collectCurrentGroup() {
    if (!selectedGroup || !entriesBody) return;
    const rows = entriesBody.querySelectorAll('tr');
    const next = {};
    rows.forEach((row) => {
        const matchInput = row.querySelector('.dictionary-entry-match');
        const replacementInput = row.querySelector('.dictionary-entry-replacement');
        if (!matchInput || !replacementInput) return;
        const match = matchInput.value.trim();
        const replacement = replacementInput.value.trim();
        if (match && replacement) next[match] = replacement;
    });
    if (!dictionaryGroups[selectedGroup]) dictionaryGroups[selectedGroup] = {};
    dictionaryGroups[selectedGroup] = next;
}

function markDirty() {
    if (rendering) return;
    dirty = true;
    setStatus('Dictionary has unsaved changes.', 'warn');
    renderMetrics();
    renderRawEditor();
}

function addEntry() {
    collectCurrentGroup();
    const entries = dictionaryGroups[selectedGroup] || {};
    let base = 'New phrase';
    let candidate = base;
    let index = 2;
    while (Object.prototype.hasOwnProperty.call(entries, candidate)) {
        candidate = `${base} ${index}`;
        index += 1;
    }
    entries[candidate] = 'spoken replacement';
    dictionaryGroups[selectedGroup] = entries;
    dirty = true;
    renderAll();
    const newInput = [...entriesBody.querySelectorAll('.dictionary-entry-match')]
        .find((input) => input.value === candidate);
    newInput?.focus();
    newInput?.select();
    setStatus('Dictionary has unsaved changes.', 'warn');
}

function addGroup() {
    collectCurrentGroup();
    const group = newGroupInput.value.trim();
    if (!group) {
        setStatus('Enter a group name first.', 'err');
        return;
    }
    if (dictionaryGroups[group]) {
        selectedGroup = group;
        renderAll();
        setStatus(`Selected existing group ${group}.`, 'warn');
        return;
    }
    dictionaryGroups[group] = {};
    selectedGroup = group;
    newGroupInput.value = '';
    dirty = true;
    renderAll();
    setStatus('Dictionary has unsaved changes.', 'warn');
}

function deleteGroup() {
    collectCurrentGroup();
    const group = selectedGroup;
    if (!group || sortedGroups().length <= 1) {
        setStatus('At least one dictionary group is required.', 'err');
        return;
    }
    delete dictionaryGroups[group];
    selectedGroup = sortedGroups()[0] || '*';
    dirty = true;
    renderAll();
    setStatus(`Deleted group ${group}. Save to apply.`, 'warn');
}

function applyRawJson() {
    try {
        const parsed = JSON.parse(rawEditor.value || '{}');
        dictionaryGroups = normalizedGroups(parsed);
        selectedGroup = sortedGroups()[0] || '*';
        dirty = true;
        renderAll({ syncRaw: false });
        setStatus('Raw JSON applied. Save to persist it.', 'warn');
    } catch (error) {
        setStatus(error.message || 'Raw JSON is invalid.', 'err');
    }
}

async function loadDictionary() {
    setStatus('Loading dictionary…');
    const payload = await panelClient.command('dictionary.get', {}, 10000);
    dictionaryGroups = normalizedGroups(payload.groups || {});
    selectedGroup = sortedGroups()[0] || '*';
    dirty = false;
    pathEl.textContent = payload.path || 'managed/dictionary.json';
    renderAll();
    const entryCount = payload.summary?.entry_count ?? countEntries();
    setStatus(`Dictionary loaded with ${entryCount} entries.`, 'ok');
}

async function saveDictionary() {
    saveButton.disabled = true;
    setStatus('Saving dictionary…');
    try {
        const payload = await panelClient.command('dictionary.save', { groups: groupsForServer() }, 15000);
        dictionaryGroups = normalizedGroups(payload.groups || {});
        selectedGroup = groupSelect.value || sortedGroups()[0] || '*';
        dirty = false;
        pathEl.textContent = payload.path || 'managed/dictionary.json';
        renderAll();
        setStatus('Dictionary saved. New TTS renders will use the updated replacements.', 'ok');
    } catch (error) {
        setStatus(error.message || 'Dictionary save failed.', 'err');
    } finally {
        saveButton.disabled = false;
    }
}

function bindEvents() {
    groupSelect.addEventListener('change', () => {
        collectCurrentGroup();
        selectedGroup = groupSelect.value;
        renderAll();
    });
    searchInput.addEventListener('input', () => {
        collectCurrentGroup();
        renderEntries();
    });
    addEntryButton.addEventListener('click', addEntry);
    addGroupButton.addEventListener('click', addGroup);
    deleteGroupButton.addEventListener('click', deleteGroup);
    applyRawButton.addEventListener('click', applyRawJson);
    saveButton.addEventListener('click', saveDictionary);
    reloadButton.addEventListener('click', () => loadDictionary().catch((error) => {
        setStatus(error.message || 'Unable to reload dictionary.', 'err');
    }));
    entriesBody.addEventListener('input', () => {
        collectCurrentGroup();
        markDirty();
    });
    entriesBody.addEventListener('click', (event) => {
        const button = event.target.closest('.dictionary-delete-entry');
        if (!button) return;
        button.closest('tr')?.remove();
        collectCurrentGroup();
        dirty = true;
        renderAll();
        setStatus('Dictionary has unsaved changes.', 'warn');
    });
    newGroupInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            addGroup();
        }
    });
    window.addEventListener('beforeunload', (event) => {
        if (!dirty) return;
        event.preventDefault();
        event.returnValue = '';
    });
}

export function initDictionaryView() {
    if (!bound) {
        bound = true;
        bindEvents();
    }
    loadDictionary().catch((error) => {
        setStatus(error.message || 'Unable to load dictionary.', 'err');
    });
}

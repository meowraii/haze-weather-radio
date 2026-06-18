import { panelClient } from './lib/ws-client.js';

const statusBanner = document.getElementById('alertsArchiveStatus');
const list = document.getElementById('alertsArchiveList');
const acceptedMetric = document.getElementById('alertsAcceptedMetric');
const expiredMetric = document.getElementById('alertsExpiredMetric');
const clearExpiredButton = document.getElementById('alertsClearExpired');
const clearAllButton = document.getElementById('alertsClearAll');
const expireAllButton = document.getElementById('alertsExpireAll');

let bound = false;
let activeTab = 'accepted';
let archiveState = {
    accepted_by_feed: [],
    rejected: [],
    expired: [],
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

function formatDateTime(value) {
    if (!value) return 'not set';
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleString([], {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
    });
}

function setStatus(text, state = 'ok') {
    statusBanner.textContent = text;
    statusBanner.dataset.state = state;
}

function recordsForTab(tab) {
    if (tab === 'accepted') return Array.isArray(archiveState.accepted_by_feed) ? archiveState.accepted_by_feed : [];
    if (tab === 'rejected') return Array.isArray(archiveState.rejected) ? archiveState.rejected : [];
    return Array.isArray(archiveState.expired) ? archiveState.expired : [];
}

function groupedAccepted(records) {
    const groups = new Map();
    for (const record of records) {
        const feedID = record.feed_id || 'unassigned';
        if (!groups.has(feedID)) groups.set(feedID, []);
        groups.get(feedID).push(record);
    }
    return [...groups.entries()];
}

function metaItems(record) {
    const items = [
        ['Event', record.event || 'unknown'],
        ['Feed', record.feed_id || 'none'],
        ['Status', record.status || record.bucket || 'unknown'],
        ['Message', record.message_type || 'unknown'],
        ['Severity', record.severity || 'unknown'],
        ['Urgency', record.urgency || 'unknown'],
        ['Certainty', record.certainty || 'unknown'],
        ['Sent', formatDateTime(record.sent)],
        ['Expires', formatDateTime(record.expires)],
    ];
    if (record.reason) items.splice(3, 0, ['Reason', record.reason]);
    if (record.audio_url) items.push(['Audio', record.audio_mime_type || 'available']);
    return items;
}

function alertCard(record) {
    const id = record.id || '';
    const feedID = record.feed_id || '';
    const headline = record.headline || record.event || 'Weather Alert';
    const areas = Array.isArray(record.areas) ? record.areas.join('; ') : '';
    const meta = metaItems(record).map(([key, value]) => `
        <span><b>${escapeHtml(key)}</b>${escapeHtml(value)}</span>
    `).join('');
    return `
        <article class="alert-card" data-alert-id="${escapeHtml(id)}" data-feed-id="${escapeHtml(feedID)}">
            <div class="alert-card-main">
                <div class="alert-card-head">
                    <div>
                        <h3>${escapeHtml(headline)}</h3>
                        <p>${escapeHtml(areas || record.sender || 'No area text available')}</p>
                    </div>
                    <span class="alert-card-time">${escapeHtml(formatDateTime(record.updated_at || record.sent))}</span>
                </div>
                <div class="alert-card-meta">${meta}</div>
                <details class="alert-details">
                    <summary>Details</summary>
                    <div class="alert-details-grid">
                        <section>
                            <h4>Description</h4>
                            <p>${escapeHtml(record.description || 'No description provided.')}</p>
                        </section>
                        <section>
                            <h4>Instruction</h4>
                            <p>${escapeHtml(record.instruction || 'No instruction provided.')}</p>
                        </section>
                        <section>
                            <h4>Identifier</h4>
                            <p>${escapeHtml(id)}</p>
                        </section>
                    </div>
                </details>
            </div>
            <div class="alert-card-actions">
                <button class="btn-action" type="button" data-archive-action="rebroadcast">
                    <i data-lucide="radio" width="13" height="13"></i>
                    Rebroadcast
                </button>
                <button class="btn-action" type="button" data-archive-action="rebroadcast_without_same">
                    <i data-lucide="volume-2" width="13" height="13"></i>
                    No SAME
                </button>
                <button class="btn-danger" type="button" data-archive-action="delete">
                    <i data-lucide="trash-2" width="13" height="13"></i>
                    Delete
                </button>
            </div>
        </article>
    `;
}

function renderAccepted(records) {
    const groups = groupedAccepted(records);
    if (!groups.length) return '<article class="alert-empty">No accepted alerts are active for any feed.</article>';
    return groups.map(([feedID, items]) => `
        <section class="alert-feed-group">
            <div class="alert-feed-group-hd">
                <strong>${escapeHtml(feedID)}</strong>
                <span>${items.length} active</span>
            </div>
            ${items.map(alertCard).join('')}
        </section>
    `).join('');
}

function renderArchive() {
    const accepted = recordsForTab('accepted');
    const expired = recordsForTab('expired');
    acceptedMetric.textContent = String(accepted.length);
    expiredMetric.textContent = String(expired.length);

    const records = recordsForTab(activeTab);
    if (activeTab === 'accepted') {
        list.innerHTML = renderAccepted(records);
    } else {
        list.innerHTML = records.length
            ? records.map(alertCard).join('')
            : `<article class="alert-empty">No ${escapeHtml(activeTab)} alerts are archived.</article>`;
    }
    window.lucide?.createIcons();
}

async function loadArchive() {
    setStatus('Loading alert archive...', 'pending');
    archiveState = await panelClient.command('alerts.archive.get', {}, 12000);
    renderArchive();
    setStatus('Alert archive loaded.', 'ok');
}

async function runAction(action, payload = {}) {
    setStatus('Sending alert archive command...', 'pending');
    await panelClient.command('alerts.archive.action', { action, ...payload }, 20000);
    await loadArchive();
}

function makeDoubleClickConfirm(button, callback) {
    let armed = false;
    let timer = null;
    const original = button.innerHTML;
    button.addEventListener('click', () => {
        if (!armed) {
            armed = true;
            button.classList.add('is-confirming');
            button.textContent = 'Click again to confirm';
            timer = window.setTimeout(() => {
                armed = false;
                button.classList.remove('is-confirming');
                button.innerHTML = original;
                window.lucide?.createIcons();
            }, 2600);
            return;
        }
        window.clearTimeout(timer);
        armed = false;
        button.classList.remove('is-confirming');
        button.disabled = true;
        callback().catch((error) => {
            setStatus(error.message || 'Alert archive command failed.', 'err');
        }).finally(() => {
            button.disabled = false;
            button.innerHTML = original;
            window.lucide?.createIcons();
        });
    });
}

function bindActionDelegates() {
    list.addEventListener('click', (event) => {
        const button = event.target.closest('[data-archive-action]');
        if (!button) return;
        const card = button.closest('.alert-card');
        const action = button.dataset.archiveAction;
        const id = card?.dataset.alertId || '';
        const feedID = card?.dataset.feedId || '';
        if (!id) return;
        if (button.dataset.confirming === '1') {
            button.dataset.confirming = '0';
            button.disabled = true;
            runAction(action, { id, feed_id: feedID }).catch((error) => {
                setStatus(error.message || 'Alert archive command failed.', 'err');
            }).finally(() => {
                button.disabled = false;
                delete button.dataset.confirming;
                renderArchive();
            });
            return;
        }
        button.dataset.confirming = '1';
        const original = button.innerHTML;
        button.dataset.originalHtml = original;
        button.textContent = 'Click again';
        window.setTimeout(() => {
            if (button.dataset.confirming === '1') {
                button.dataset.confirming = '0';
                button.innerHTML = button.dataset.originalHtml || original;
                window.lucide?.createIcons();
            }
        }, 2600);
    });
}

export function initAlertsArchiveView() {
    if (bound) {
        loadArchive().catch((error) => setStatus(error.message || 'Unable to load alert archive.', 'err'));
        return;
    }
    bound = true;
    document.querySelectorAll('[data-alert-tab]').forEach((button) => {
        button.addEventListener('click', () => {
            activeTab = button.dataset.alertTab || 'accepted';
            document.querySelectorAll('[data-alert-tab]').forEach((tab) => {
                tab.classList.toggle('active', tab === button);
            });
            renderArchive();
        });
    });
    bindActionDelegates();
    makeDoubleClickConfirm(clearExpiredButton, () => runAction('clear_expired'));
    makeDoubleClickConfirm(clearAllButton, () => runAction('clear_all'));
    makeDoubleClickConfirm(expireAllButton, () => runAction('expire_all'));
    window.addEventListener('haze:admin-state', () => {
        if (document.querySelector('.view[data-view="alerts"]')?.classList.contains('active')) {
            loadArchive().catch(() => {});
        }
    });
    loadArchive().catch((error) => setStatus(error.message || 'Unable to load alert archive.', 'err'));
}

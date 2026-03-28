const API_BASE = '/api/v1';
const TOKEN_KEY = 'haze.panel.token';
const TEST_CODES = new Set(['DMO', 'RWT', 'RMT']);

const feedSelect = document.getElementById('feedSelect');
const allFeedsCheck = document.getElementById('allFeedsCheck');
const origSelect = document.getElementById('origSelect');
const eventSelect = document.getElementById('eventSelect');
const locationsInput = document.getElementById('locationsInput');
const allLocationsCheck = document.getElementById('allLocationsCheck');
const durationHours = document.getElementById('durationHours');
const durationMinutes = document.getElementById('durationMinutes');
const toneSelect = document.getElementById('toneSelect');
const audioFile = document.getElementById('audioFile');
const uploadArea = document.getElementById('uploadArea');
const uploadPromptEl = document.getElementById('uploadPromptEl');
const uploadFilenameEl = document.getElementById('uploadFilenameEl');
const uploadActions = document.getElementById('uploadActions');
const uploadBtn = document.getElementById('uploadBtn');
const clearUploadBtn = document.getElementById('clearUploadBtn');
const uploadStatusEl = document.getElementById('uploadStatusEl');
const ttsSection = document.getElementById('ttsSection');
const fileSection = document.getElementById('fileSection');
const voiceInput = document.getElementById('voiceInput');
const airBtn = document.getElementById('airBtn');
const headerPreview = document.getElementById('headerPreview');
const statusBanner = document.getElementById('statusBanner');
const recentList = document.getElementById('recentList');
const apiDot = document.getElementById('apiDot');
const healthPill = document.getElementById('healthPill');
const themeToggle = document.getElementById('themeToggle');
const eventTableBody = document.getElementById('eventTableBody');

let token = localStorage.getItem(TOKEN_KEY) || '';
let allFeedsData = [];
let sameMapping = {};
let configuredCallsign = 'XXXXXXXX';
let uploadedFilePath = '';
let selectedFile = null;
let airConfirmPending = false;
let airConfirmTimer = null;
const recentBroadcasts = [];

const SUN_ICON = `<i data-lucide="sun" width="13" height="13"></i>`;
const MOON_ICON = `<i data-lucide="moon" width="13" height="13"></i>`;

function applyTheme(theme) {
	document.documentElement.dataset.theme = theme;
	localStorage.setItem('haze.theme', theme);
	const isDark = theme === 'dark';
	themeToggle.innerHTML = (isDark ? SUN_ICON : MOON_ICON) + `<span>${isDark ? 'Light mode' : 'Dark mode'}</span>`;
	lucide.createIcons({ nodes: [themeToggle] });
}

(function initTheme() {
	const saved = localStorage.getItem('haze.theme');
	if (saved) applyTheme(saved);
})();
lucide.createIcons();

themeToggle.addEventListener('click', () => {
	const current = document.documentElement.dataset.theme;
	const isDark = current === 'dark' || (!current && window.matchMedia('(prefers-color-scheme: dark)').matches);
	applyTheme(isDark ? 'light' : 'dark');
});

function buildUniqueEventOptions() {
	return Object.entries(sameMapping)
		.map(([code, entry]) => ({ code, label: entry.easText || code }))
		.sort((a, b) => a.code.localeCompare(b.code));
}

function populateEventSelect() {
	const options = buildUniqueEventOptions();
	eventSelect.innerHTML = options.map(({ code, label }) =>
		`<option value="${code}">${code} — ${label}</option>`
	).join('');
	eventSelect.value = 'DMO';
}

function populateEventTable() {
	const options = buildUniqueEventOptions();
	eventTableBody.innerHTML = options.map(({ code, label }) =>
		`<tr>
			<td style="padding:3px 8px;border-bottom:1px solid var(--border);font-family:var(--font-mono);font-size:0.78rem;font-weight:600;color:var(--accent);">${code}</td>
			<td style="padding:3px 8px;border-bottom:1px solid var(--border);color:var(--text-muted);">${label}</td>
		</tr>`
	).join('');
}

function buildPreview() {
	const orig = (origSelect.value || 'WXR').toUpperCase().slice(0, 3);
	const event = (eventSelect.value || 'CEM').toUpperCase().slice(0, 3);
	const locs = allLocationsCheck.checked
		? [...new Set(allFeedsData.flatMap((f) => f.clc_codes || []))]
		: locationsInput.value.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);
	const locStr = (locs.length ? locs.slice(0, 31) : ['000000']).join('-');
	const h = Math.max(0, Math.min(parseInt(durationHours.value, 10) || 0, 99));
	const m = Math.max(0, Math.min(parseInt(durationMinutes.value, 10) || 0, 59));
	const dur = (h === 0 && m === 0) ? '0100' : `${String(h).padStart(2, '0')}${String(m).padStart(2, '0')}`;
	const now = new Date();
	const doy = String(Math.floor((now - new Date(now.getFullYear(), 0, 0)) / 86400000)).padStart(3, '0');
	const utcH = String(now.getUTCHours()).padStart(2, '0');
	const utcM = String(now.getUTCMinutes()).padStart(2, '0');
	const cs = configuredCallsign.replace(/-/g, '/').padEnd(8).slice(0, 8);
	return `ZCZC-${orig}-${event}-${locStr}+${dur}-${doy}${utcH}${utcM}-${cs}-`;
}

function updatePreview() {
	headerPreview.textContent = buildPreview();
	headerPreview.className = 'same-preview has-data';
}

function showStatus(message, type) {
	statusBanner.textContent = message;
	statusBanner.className = `status-banner ${type}`;
}

function clearStatus() {
	statusBanner.className = 'status-banner';
	statusBanner.textContent = '';
}

async function apiGet(path) {
	const headers = new Headers();
	if (token) headers.set('Authorization', `Bearer ${token}`);
	const r = await fetch(`${API_BASE}${path}`, { headers });
	if (r.status === 401) { window.location.href = '/'; throw new Error('Not authenticated'); }
	if (!r.ok) throw new Error(`Request failed: ${r.status}`);
	return r.json();
}

async function apiPost(path, body) {
	const headers = new Headers({ 'Content-Type': 'application/json' });
	if (token) headers.set('Authorization', `Bearer ${token}`);
	const r = await fetch(`${API_BASE}${path}`, { method: 'POST', headers, body: JSON.stringify(body) });
	if (r.status === 401) { window.location.href = '/'; throw new Error('Not authenticated'); }
	if (!r.ok) {
		const err = await r.json().catch(() => ({ detail: 'Unknown error' }));
		throw new Error(err.detail || `Request failed: ${r.status}`);
	}
	return r.json();
}

async function apiPostForm(path, formData) {
	const headers = new Headers();
	if (token) headers.set('Authorization', `Bearer ${token}`);
	const r = await fetch(`${API_BASE}${path}`, { method: 'POST', headers, body: formData });
	if (r.status === 401) { window.location.href = '/'; throw new Error('Not authenticated'); }
	if (!r.ok) {
		const err = await r.json().catch(() => ({ detail: 'Unknown error' }));
		throw new Error(err.detail || `Request failed: ${r.status}`);
	}
	return r.json();
}

async function apiPut(path, body) {
	const headers = new Headers({ 'Content-Type': 'application/json' });
	if (token) headers.set('Authorization', `Bearer ${token}`);
	const r = await fetch(`${API_BASE}${path}`, { method: 'PUT', headers, body: JSON.stringify(body) });
	if (r.status === 401) { window.location.href = '/'; throw new Error('Not authenticated'); }
	if (!r.ok) {
		const err = await r.json().catch(() => ({ detail: 'Unknown error' }));
		throw new Error(err.detail || `Request failed: ${r.status}`);
	}
	return r.json();
}

async function loadMapping() {
	try {
		sameMapping = await apiGet('/same/event-codes');
	} catch {
		sameMapping = {};
	}
}

async function loadFeeds() {
	try {
		const health = await fetch(`${API_BASE}/health`).then((r) => r.json());
		apiDot.dataset.state = 'ok';
		healthPill.textContent = 'API healthy';
		const [summary, config] = await Promise.all([apiGet('/summary'), apiGet('/config').catch(() => ({}))]);
		allFeedsData = summary.feeds || [];
		configuredCallsign = (config.same && config.same.sender) || 'XXXXXXXX';
		if (allFeedsData.length) {
			feedSelect.innerHTML = allFeedsData.map((f) => `<option value="${f.id}">${f.name || f.id}</option>`).join('');
		} else {
			feedSelect.innerHTML = '<option value="">No feeds configured</option>';
		}
		updatePreview();
	} catch (err) {
		apiDot.dataset.state = 'err';
		healthPill.textContent = 'API unavailable';
		feedSelect.innerHTML = '<option value="">Could not load feeds</option>';
	}
}

function handleFileSelect(file) {
	if (!file) return;
	selectedFile = file;
	uploadedFilePath = '';
	uploadArea.classList.add('has-file');
	uploadPromptEl.style.display = 'none';
	uploadFilenameEl.textContent = `Selected: ${file.name}`;
	uploadFilenameEl.style.display = 'block';
	uploadActions.style.display = 'flex';
	uploadStatusEl.textContent = '';
	uploadBtn.textContent = 'Upload & Encode';
}

async function uploadFile() {
	if (!selectedFile) return;
	uploadBtn.disabled = true;
	uploadBtn.textContent = 'Encoding…';
	uploadStatusEl.textContent = '';
	try {
		const fd = new FormData();
		fd.append('file', selectedFile);
		const result = await apiPostForm('/same/upload-audio', fd);
		uploadedFilePath = result.path;
		uploadStatusEl.textContent = `Ready — encoded to ${result.sample_rate} Hz PCM WAV`;
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
	selectedFile = null;
	uploadedFilePath = '';
	audioFile.value = '';
	uploadArea.classList.remove('has-file');
	uploadPromptEl.style.display = 'block';
	uploadFilenameEl.style.display = 'none';
	uploadFilenameEl.textContent = '';
	uploadActions.style.display = 'none';
	uploadStatusEl.textContent = '';
	uploadBtn.textContent = 'Upload & Encode';
}

function resetAirButton() {
	clearInterval(airConfirmTimer);
	airConfirmPending = false;
	airConfirmTimer = null;
	airBtn.className = 'btn-danger';
	airBtn.textContent = 'Air Now';
	airBtn.disabled = false;
}

function startAirConfirm() {
	airConfirmPending = true;
	let countdown = 5;
	airBtn.className = 'btn-confirm';
	airBtn.textContent = `CONFIRM AIR — ${countdown}s`;
	airConfirmTimer = setInterval(() => {
		countdown--;
		if (countdown <= 0) {
			clearInterval(airConfirmTimer);
			resetAirButton();
		} else {
			airBtn.textContent = `CONFIRM AIR — ${countdown}s`;
		}
	}, 1000);
}

function addRecentBroadcast(result) {
	recentBroadcasts.unshift({ ...result, ts: new Date().toLocaleTimeString() });
	recentList.innerHTML = recentBroadcasts.slice(0, 16).map((b) => {
		const feedLabel = (b.feeds_aired || [b.feed_id]).join(', ');
		return `<article class="event-item" style="margin-bottom:6px;">
			<div class="event-head">
				<span class="event-kind">SAME</span>
				<span style="font-size:0.75rem;color:var(--text-muted);">${feedLabel}</span>
				<time>${b.ts}</time>
			</div>
			<p style="font-family:var(--font-mono);font-size:0.77rem;word-break:break-all;margin:0;">${b.header}</p>
		</article>`;
	}).join('');
}

async function doAir() {
	const audioMode = document.querySelector('input[name="audioMode"]:checked')?.value || 'none';

	const locations = allLocationsCheck.checked
		? [...new Set(allFeedsData.flatMap((f) => f.clc_codes || []))]
		: locationsInput.value.split(/[\n,]+/).map((s) => s.trim()).filter(Boolean);

	if (!locations.length) {
		showStatus('Enter at least one location code.', 'err');
		return;
	}

	if (audioMode === 'file' && !uploadedFilePath) {
		showStatus('Upload and encode the audio file before airing.', 'err');
		return;
	}

	clearStatus();
	airBtn.disabled = true;
	airBtn.textContent = 'Transmitting…';

	const payload = {
		feed_id: allFeedsCheck.checked ? '' : (feedSelect.value || ''),
		originator: origSelect.value,
		event: eventSelect.value,
		locations,
		duration_hours: parseInt(durationHours.value, 10) || 0,
		duration_minutes: parseInt(durationMinutes.value, 10) || 0,
		tone_type: toneSelect.value,
		voice_message: audioMode === 'tts' ? voiceInput.value.trim() : '',
		audio_file_path: audioMode === 'file' ? uploadedFilePath : '',
		air_on_all_feeds: allFeedsCheck.checked,
	};

	try {
		const result = await apiPost('/same/air', payload);
		const feedLabel = (result.feeds_aired || [result.feed_id]).join(', ');
		showStatus(`Broadcast queued on ${feedLabel}: ${result.header}`, 'ok');
		headerPreview.textContent = result.header;
		headerPreview.className = 'same-preview has-data';
		addRecentBroadcast(result);
	} catch (err) {
		showStatus(`Failed to air: ${err.message}`, 'err');
	} finally {
		resetAirButton();
	}
}

airBtn.addEventListener('click', () => {
	const code = eventSelect.value;
	if (TEST_CODES.has(code)) {
		doAir();
		return;
	}
	if (airConfirmPending) {
		resetAirButton();
		doAir();
	} else {
		startAirConfirm();
	}
});

document.querySelectorAll('input[name="audioMode"]').forEach((radio) => {
	radio.addEventListener('change', () => {
		const mode = document.querySelector('input[name="audioMode"]:checked').value;
		ttsSection.style.display = mode === 'tts' ? '' : 'none';
		fileSection.style.display = mode === 'file' ? '' : 'none';
	});
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

allFeedsCheck.addEventListener('change', () => {
	feedSelect.disabled = allFeedsCheck.checked;
	updatePreview();
});

allLocationsCheck.addEventListener('change', () => {
	if (allLocationsCheck.checked) {
		const locs = [...new Set(allFeedsData.flatMap((f) => f.clc_codes || []))];
		locationsInput.value = locs.join('\n');
		locationsInput.disabled = true;
	} else {
		locationsInput.value = '';
		locationsInput.disabled = false;
	}
	updatePreview();
});

[origSelect, eventSelect, locationsInput, durationHours, durationMinutes].forEach((el) => {
	el.addEventListener('input', updatePreview);
	el.addEventListener('change', updatePreview);
});

const templateList = document.getElementById('templateList');
const templateStatus = document.getElementById('templateStatus');
const templateAddBtn = document.getElementById('templateAddBtn');
const templateSaveBtn = document.getElementById('templateSaveBtn');

let templateData = {};

function templateShowStatus(msg, type) {
	templateStatus.textContent = msg;
	templateStatus.className = `status-banner ${type}`;
	if (type === 'ok') setTimeout(() => { templateStatus.className = 'status-banner'; templateStatus.textContent = ''; }, 3000);
}

function renderTemplates() {
	if (!Object.keys(templateData).length) {
		templateList.innerHTML = '<p style="color:var(--muted);font-size:12.5px;">No templates defined. Click "+ New Template" to add one.</p>';
		return;
	}
	templateList.innerHTML = Object.entries(templateData).map(([code, tpl]) => {
		const msgs = tpl.msg || {};
		const langRows = Object.entries(msgs).map(([lang, text]) => `
			<div class="template-lang-row" data-code="${code}" data-lang="${lang}">
				<div class="form-row" style="align-items:start;">
					<div class="form-group" style="max-width:160px;">
						<label>Language pattern</label>
						<input type="text" class="tpl-lang-key" value="${lang}" placeholder="en*">
					</div>
					<div class="form-group" style="flex:1">
						<label>Message text</label>
						<textarea class="tpl-lang-text" rows="3">${text}</textarea>
					</div>
					<button type="button" class="btn-ghost tpl-remove-lang" style="margin-top:20px;align-self:end;flex-shrink:0" data-code="${code}" data-lang="${lang}">✕</button>
				</div>
			</div>
		`).join('');
		return `
			<div class="section-block" data-template-code="${code}" style="margin-top:0;border-radius:var(--radius);">
				<div class="section-hd" style="gap:8px;">
					<input type="text" class="tpl-code-input" value="${code}" placeholder="RWT" style="width:90px;font-weight:700;background:transparent;border:1px solid var(--border);border-radius:var(--radius-sm);padding:3px 8px;color:var(--text);font-family:var(--mono-font);font-size:12px;">
					<div style="display:flex;gap:8px;align-items:center;">
						<label style="font-size:11.5px;font-weight:400;color:var(--muted);">Expire:</label>
						<input type="text" class="tpl-expire" value="${tpl.sameExpire || '0015'}" placeholder="0015" style="width:58px;background:transparent;border:1px solid var(--border);border-radius:var(--radius-sm);padding:3px 8px;color:var(--text);font-family:var(--mono-font);font-size:12px;">
					</div>
					<div style="display:flex;gap:8px;align-items:center;">
						<label style="font-size:11.5px;font-weight:400;color:var(--muted);">SAME event:</label>
						<input type="text" class="tpl-event" value="${tpl.sameEvent || code}" placeholder="RWT" style="width:58px;background:transparent;border:1px solid var(--border);border-radius:var(--radius-sm);padding:3px 8px;color:var(--text);font-family:var(--mono-font);font-size:12px;">
					</div>
					<div class="section-hd-actions">
						<button type="button" class="btn-action tpl-fire-btn" data-code="${code}" title="Fire this test now">&#9654; Send Now</button>
						<button type="button" class="btn-ghost tpl-add-lang" data-code="${code}" style="font-size:11.5px;">+ Language</button>
						<button type="button" class="btn-ghost tpl-remove-tpl" data-code="${code}" style="font-size:11.5px;color:var(--accent-err);">Remove</button>
					</div>
				</div>
				<div class="section-body" style="display:flex;flex-direction:column;gap:6px;">
					${langRows || '<p style="color:var(--muted);font-size:12px;">No language messages. Add one below.</p>'}
				</div>
			</div>
		`;
	}).join('');

	templateList.querySelectorAll('.tpl-remove-tpl').forEach((btn) => {
		btn.addEventListener('click', () => {
			const code = btn.dataset.code;
			delete templateData[code];
			renderTemplates();
		});
	});

	templateList.querySelectorAll('.tpl-add-lang').forEach((btn) => {
		btn.addEventListener('click', () => {
			const code = btn.dataset.code;
			if (!templateData[code].msg) templateData[code].msg = {};
			const newKey = `lang-${Date.now()}`;
			templateData[code].msg[newKey] = '';
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
			const code = btn.dataset.code;
			btn.disabled = true;
			btn.textContent = 'Sending…';
			try {
				await apiPost(`/same/test?event_code=${encodeURIComponent(code)}`, {});
				btn.textContent = 'Sent ✓';
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
	const updated = {};
	templateList.querySelectorAll('[data-template-code]').forEach((block) => {
		const origCode = block.dataset.templateCode;
		const codeInput = block.querySelector('.tpl-code-input');
		const code = (codeInput?.value || origCode).trim().toUpperCase();
		if (!code) return;
		const expire = block.querySelector('.tpl-expire')?.value.trim() || '0015';
		const event = block.querySelector('.tpl-event')?.value.trim().toUpperCase() || code;
		const msg = {};
		block.querySelectorAll('.template-lang-row').forEach((row) => {
			const langKey = row.querySelector('.tpl-lang-key')?.value.trim() || '';
			const text = row.querySelector('.tpl-lang-text')?.value || '';
			if (langKey) msg[langKey] = text;
		});
		updated[code] = { sameExpire: expire, sameEvent: event, msg };
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
	templateSaveBtn.disabled = true;
	templateSaveBtn.textContent = 'Saving…';
	try {
		await apiPut('/same/templates', { content: JSON.stringify(templateData, null, 2) });
		templateShowStatus('Templates saved.', 'ok');
		renderTemplates();
	} catch (err) {
		templateShowStatus(`Save failed: ${err.message}`, 'err');
	} finally {
		templateSaveBtn.disabled = false;
		templateSaveBtn.textContent = 'Save All';
	}
}

templateAddBtn.addEventListener('click', () => {
	const code = `CUSTOM${Object.keys(templateData).length + 1}`;
	templateData[code] = { sameExpire: '0015', sameEvent: code, msg: { 'en*': '' } };
	renderTemplates();
});

templateSaveBtn.addEventListener('click', saveTemplates);

if (!token) {
	window.location.href = '/';
} else {
	Promise.all([loadFeeds(), loadMapping(), loadTemplates()]).then(() => {
		populateEventSelect();
		populateEventTable();
		updatePreview();
	});
}

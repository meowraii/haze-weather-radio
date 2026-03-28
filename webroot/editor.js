const API_BASE = '/api/v1';
const TOKEN_KEY = 'haze.panel.token';

const editorTitle = document.getElementById('editorTitle');
const editorSubtitle = document.getElementById('editorSubtitle');
const sectionFilename = document.getElementById('sectionFilename');
const editorArea = document.getElementById('editorArea');
const emptyState = document.getElementById('emptyState');
const statusBanner = document.getElementById('statusBanner');
const loadBtn = document.getElementById('loadBtn');
const saveBtn = document.getElementById('saveBtn');
const themeToggle = document.getElementById('themeToggle');
const themeLabel = document.getElementById('themeLabel');

let token = localStorage.getItem(TOKEN_KEY) || '';
let currentFile = null;

const SUN_ICON = `<i data-lucide="sun" width="13" height="13"></i>`;
const MOON_ICON = `<i data-lucide="moon" width="13" height="13"></i>`;

function applyTheme(theme) {
	document.documentElement.dataset.theme = theme;
	localStorage.setItem('haze.theme', theme);
	const isDark = theme === 'dark';
	themeToggle.innerHTML = (isDark ? SUN_ICON : MOON_ICON) + `<span id="themeLabel">${isDark ? 'Light mode' : 'Dark mode'}</span>`;
	lucide.createIcons({ nodes: [themeToggle] });
}

(function initTheme() {
	const saved = localStorage.getItem('haze.theme');
	if (saved) applyTheme(saved);
})();
lucide.createIcons();

themeToggle.addEventListener('click', () => {
	const current = document.documentElement.dataset.theme;
	const isDark = current === 'dark' ||
		(current === undefined && window.matchMedia('(prefers-color-scheme: dark)').matches);
	applyTheme(isDark ? 'light' : 'dark');
});

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
	const response = await fetch(`${API_BASE}${path}`, { headers });
	if (response.status === 401) {
		window.location.href = '/';
		throw new Error('Not authenticated');
	}
	if (!response.ok) throw new Error(`Request failed: ${response.status}`);
	return response.json();
}

async function apiPut(path, body) {
	const headers = new Headers({ 'Content-Type': 'application/json' });
	if (token) headers.set('Authorization', `Bearer ${token}`);
	const response = await fetch(`${API_BASE}${path}`, {
		method: 'PUT',
		headers,
		body: JSON.stringify(body),
	});
	if (response.status === 401) {
		window.location.href = '/';
		throw new Error('Not authenticated');
	}
	if (!response.ok) {
		const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
		throw new Error(err.detail || `Request failed: ${response.status}`);
	}
	return response.json();
}

async function loadFile(filename) {
	clearStatus();
	editorArea.style.display = 'none';
	emptyState.style.display = 'block';
	emptyState.textContent = `Loading ${filename}…`;
	loadBtn.disabled = true;
	saveBtn.disabled = true;

	try {
		const data = await apiGet(`/managed/${encodeURIComponent(filename)}`);
		editorArea.value = data.content;
		editorArea.style.display = 'block';
		emptyState.style.display = 'none';
		currentFile = filename;
		sectionFilename.textContent = filename;
		editorTitle.textContent = `Editing: ${filename}`;
		loadBtn.disabled = false;
		saveBtn.disabled = false;
	} catch (error) {
		emptyState.textContent = `Failed to load ${filename}: ${error.message}`;
		loadBtn.disabled = false;
	}
}

async function saveFile() {
	if (!currentFile) return;
	saveBtn.disabled = true;
	clearStatus();
	try {
		await apiPut(`/managed/${encodeURIComponent(currentFile)}`, { content: editorArea.value });
		showStatus(`Saved ${currentFile} successfully.`, 'ok');
	} catch (error) {
		showStatus(`Save failed: ${error.message}`, 'err');
	} finally {
		saveBtn.disabled = false;
	}
}

document.querySelectorAll('[data-file]').forEach((link) => {
	link.addEventListener('click', (event) => {
		event.preventDefault();
		document.querySelectorAll('[data-file]').forEach((el) => el.classList.remove('active'));
		link.classList.add('active');
		loadFile(link.dataset.file);
	});
});

loadBtn.addEventListener('click', () => {
	if (currentFile) loadFile(currentFile);
});

saveBtn.addEventListener('click', saveFile);

editorArea.addEventListener('keydown', (event) => {
	if ((event.ctrlKey || event.metaKey) && event.key === 's') {
		event.preventDefault();
		saveFile();
	}
	if (event.key === 'Tab') {
		event.preventDefault();
		const start = editorArea.selectionStart;
		const end = editorArea.selectionEnd;
		editorArea.value = editorArea.value.substring(0, start) + '    ' + editorArea.value.substring(end);
		editorArea.selectionStart = editorArea.selectionEnd = start + 4;
	}
});

if (!token) {
	window.location.href = '/';
}

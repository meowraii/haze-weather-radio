import { apiGet, apiPut, token } from './lib/api.js';
import { initTheme } from './lib/theme.js';

const editorTitle = document.getElementById('editorTitle');
const editorSubtitle = document.getElementById('editorSubtitle');
const sectionFilename = document.getElementById('sectionFilename');
const editorArea = document.getElementById('editorArea');
const emptyState = document.getElementById('emptyState');
const statusBanner = document.getElementById('statusBanner');
const loadBtn = document.getElementById('loadBtn');
const saveBtn = document.getElementById('saveBtn');

let currentFile = null;

initTheme(document.getElementById('themeToggle'));
window.lucide?.createIcons();


function showStatus(message, type) {
	statusBanner.textContent = message;
	statusBanner.className = `status-banner ${type}`;
}

function clearStatus() {
	statusBanner.className = 'status-banner';
	statusBanner.textContent = '';
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

if (!token.get()) {
	window.location.href = '/';
}

(() => {
    const NS = 'http://www.w3.org/2000/svg';
    const ICONS = {
        'bell-ring': '<path d="M10 5a2 2 0 0 1 4 0"/><path d="M5 17h14"/><path d="M7 17V9a5 5 0 0 1 10 0v8"/><path d="M10 20a2 2 0 0 0 4 0"/><path d="M4 4 2.8 5.2"/><path d="m20 4 1.2 1.2"/>',
        'circle-play': '<circle cx="12" cy="12" r="9"/><path d="m10 8 6 4-6 4z"/>',
        copy: '<rect x="8" y="8" width="11" height="11" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v1"/>',
        'file-code-2': '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/><path d="m10 13-2 2 2 2"/><path d="m14 17 2-2-2-2"/>',
        github: '<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.1-1.2-.3-2.4-1.1-3.3.1-.3.5-1.7-.1-3.3 0 0-1-.3-3.4 1.3a11.6 11.6 0 0 0-6.2 0C6.8 2.1 5.8 2.4 5.8 2.4c-.6 1.6-.2 3-.1 3.3A5.1 5.1 0 0 0 4.6 9c0 3.5 3 5.5 6 5.5a4.8 4.8 0 0 0-1 3.5v4"/><path d="M9 18c-4.5 2-5-2-7-2"/>',
        headphones: '<path d="M3 14h3a2 2 0 0 1 2 2v5H5a2 2 0 0 1-2-2v-7a9 9 0 0 1 18 0v7a2 2 0 0 1-2 2h-3v-5a2 2 0 0 1 2-2h3"/>',
        'chart-no-axes-combined': '<path d="M12 16v5"/><path d="M16 14v7"/><path d="M20 10v11"/><path d="m22 3-8.646 8.646a.5.5 0 0 1-.708 0L9.354 8.354a.5.5 0 0 0-.708 0L2 15"/><path d="M4 18v3"/><path d="M8 14v7"/>',
        house: '<path d="m3 11 9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M9 20v-6h6v6"/>',
        link: '<path d="M10 13a5 5 0 0 0 7.1 0l2-2a5 5 0 0 0-7.1-7.1l-1.1 1.1"/><path d="M14 11a5 5 0 0 0-7.1 0l-2 2a5 5 0 0 0 7.1 7.1l1.1-1.1"/>',
        play: '<path d="m6 3 14 9-14 9z"/>',
        radio: '<path d="M4.9 19.1a10 10 0 0 1 14.2 0"/><path d="M8.5 15.5a5 5 0 0 1 7 0"/><path d="M12 12h.01"/><path d="M5 4h14v8H5z"/><path d="M8 8h8"/>',
        shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
        square: '<rect x="5" y="5" width="14" height="14" rx="2"/>',
        'volume-2': '<path d="M11 5 6 9H3v6h3l5 4z"/><path d="M15.5 8.5a5 5 0 0 1 0 7"/><path d="M19 5a10 10 0 0 1 0 14"/>',
    };

    function iconMarkup(name) {
        return ICONS[name] || '<circle cx="12" cy="12" r="8"/>';
    }

    function createIcon(node) {
        const name = node.getAttribute('data-lucide');
        if (!name) return null;
        const svg = document.createElementNS(NS, 'svg');
        const width = node.getAttribute('width') || '16';
        const height = node.getAttribute('height') || '16';
        svg.setAttribute('xmlns', NS);
        svg.setAttribute('width', width);
        svg.setAttribute('height', height);
        svg.setAttribute('viewBox', '0 0 24 24');
        svg.setAttribute('fill', 'none');
        svg.setAttribute('stroke', 'currentColor');
        svg.setAttribute('stroke-width', '2');
        svg.setAttribute('stroke-linecap', 'round');
        svg.setAttribute('stroke-linejoin', 'round');
        svg.setAttribute('aria-hidden', 'true');
        svg.setAttribute('focusable', 'false');
        svg.classList.add('lucide', `lucide-${name}`);
        svg.innerHTML = iconMarkup(name);
        return svg;
    }

    function createIcons(options = {}) {
        const roots = Array.isArray(options.nodes) && options.nodes.length ? options.nodes : [document];
        for (const root of roots) {
            const nodes = root.matches?.('[data-lucide]')
                ? [root]
                : Array.from(root.querySelectorAll?.('[data-lucide]') || []);
            for (const node of nodes) {
                const svg = createIcon(node);
                if (svg) node.replaceWith(svg);
            }
        }
    }

    window.lucide = { createIcons };
})();

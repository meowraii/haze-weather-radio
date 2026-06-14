import { createControlClient } from './ws-client.js';

export function panelWsUrl(extra = {}, options = {}) {
    const client = createControlClient();
    if (options.includeToken === false) {
        return client.url({ ...extra, token: '' }).replace(/[?&]token=&?/, (match) => match.startsWith('?') ? '?' : '');
    }
    return client.url(extra);
}

export function panelWsRequest(message, timeoutMs = 8000) {
    const client = createControlClient();
    const requestType = message.type || 'command';
    const payload = { ...message };
    delete payload.type;
    return client.request(requestType, payload, timeoutMs).finally(() => client.close());
}

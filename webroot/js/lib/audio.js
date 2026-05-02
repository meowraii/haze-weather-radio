export function pcmToWav(pcmBytes, sampleRate, numChannels) {
    const dataLen = pcmBytes.byteLength;
    const buf = new ArrayBuffer(44 + dataLen);
    const dv = new DataView(buf);
    const str = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
    str(0, 'RIFF');
    dv.setUint32(4, 36 + dataLen, true);
    str(8, 'WAVE');
    str(12, 'fmt ');
    dv.setUint32(16, 16, true);
    dv.setUint16(20, 1, true);
    dv.setUint16(22, numChannels, true);
    dv.setUint32(24, sampleRate, true);
    dv.setUint32(28, sampleRate * numChannels * 2, true);
    dv.setUint16(32, numChannels * 2, true);
    dv.setUint16(34, 16, true);
    str(36, 'data');
    dv.setUint32(40, dataLen, true);
    new Uint8Array(buf).set(pcmBytes, 44);
    return buf;
}

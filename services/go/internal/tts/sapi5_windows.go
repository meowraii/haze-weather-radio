//go:build windows

package tts

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"strings"

	"github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
)

const sapiCreateForWrite = 3

// SAPI5Provider uses installed Windows SAPI5 voices.
type SAPI5Provider struct{}

// NewSAPI5Provider creates a Windows SAPI5 provider.
func NewSAPI5Provider() *SAPI5Provider {
	return &SAPI5Provider{}
}

func (p *SAPI5Provider) ID() string { return "sapi5" }

func (p *SAPI5Provider) ListVoices(ctx context.Context) ([]Voice, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := ole.CoInitializeEx(0, ole.COINIT_APARTMENTTHREADED); err != nil {
		return nil, err
	}
	defer ole.CoUninitialize()

	voice, cleanup, err := createDispatch("SAPI.SpVoice")
	if err != nil {
		return nil, err
	}
	defer cleanup()

	tokensRaw, err := oleutil.CallMethod(voice, "GetVoices")
	if err != nil {
		return nil, err
	}
	tokens := tokensRaw.ToIDispatch()
	defer tokens.Release()

	countRaw, err := oleutil.GetProperty(tokens, "Count")
	if err != nil {
		return nil, err
	}
	count := int(countRaw.Val)
	voices := make([]Voice, 0, count)
	for idx := 0; idx < count; idx++ {
		tokenRaw, err := oleutil.CallMethod(tokens, "Item", idx)
		if err != nil {
			continue
		}
		token := tokenRaw.ToIDispatch()
		voice, ok := sapiVoiceFromToken(token)
		token.Release()
		if ok {
			voices = append(voices, voice)
		}
	}
	return voices, nil
}

func (p *SAPI5Provider) Synthesize(ctx context.Context, req Request) (Audio, error) {
	if err := ctx.Err(); err != nil {
		return Audio{}, err
	}
	if strings.TrimSpace(req.Text) == "" {
		return Audio{}, fmt.Errorf("empty synthesis text")
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := ole.CoInitializeEx(0, ole.COINIT_APARTMENTTHREADED); err != nil {
		return Audio{}, err
	}
	defer ole.CoUninitialize()

	voice, cleanupVoice, err := createDispatch("SAPI.SpVoice")
	if err != nil {
		return Audio{}, err
	}
	defer cleanupVoice()

	if req.VoiceID != "" {
		token, err := findSAPIVoiceToken(voice, req.VoiceID)
		if err != nil {
			return Audio{}, err
		}
		defer token.Release()
		if _, err := oleutil.PutPropertyRef(voice, "Voice", token); err != nil {
			return Audio{}, err
		}
	}
	if req.Rate != 0 {
		if _, err := oleutil.PutProperty(voice, "Rate", sapiRate(req.Rate)); err != nil {
			return Audio{}, err
		}
	}
	if req.Volume > 0 {
		if _, err := oleutil.PutProperty(voice, "Volume", sapiVolume(req.Volume)); err != nil {
			return Audio{}, err
		}
	}

	stream, cleanupStream, err := createDispatch("SAPI.SpFileStream")
	if err != nil {
		return Audio{}, err
	}
	defer cleanupStream()

	tmp, err := os.CreateTemp("", "haze-sapi5-*.wav")
	if err != nil {
		return Audio{}, err
	}
	path := tmp.Name()
	_ = tmp.Close()
	defer os.Remove(path)

	if _, err := oleutil.CallMethod(stream, "Open", path, sapiCreateForWrite, false); err != nil {
		return Audio{}, err
	}
	streamClosed := false
	defer func() {
		if !streamClosed {
			_, _ = oleutil.CallMethod(stream, "Close")
		}
	}()
	if _, err := oleutil.PutPropertyRef(voice, "AudioOutputStream", stream); err != nil {
		return Audio{}, err
	}
	if err := ctx.Err(); err != nil {
		return Audio{}, err
	}
	if _, err := oleutil.CallMethod(voice, "Speak", req.Text, 0); err != nil {
		return Audio{}, err
	}
	if _, err := oleutil.CallMethod(stream, "Close"); err != nil {
		return Audio{}, err
	}
	streamClosed = true

	data, err := os.ReadFile(path)
	if err != nil {
		return Audio{}, err
	}
	return Audio{Format: FormatWAV, Data: data}, nil
}

func createDispatch(progID string) (*ole.IDispatch, func(), error) {
	unknown, err := oleutil.CreateObject(progID)
	if err != nil {
		return nil, nil, err
	}
	dispatch, err := unknown.QueryInterface(ole.IID_IDispatch)
	if err != nil {
		unknown.Release()
		return nil, nil, err
	}
	return dispatch, func() {
		dispatch.Release()
		unknown.Release()
	}, nil
}

func sapiVoiceFromToken(token *ole.IDispatch) (Voice, bool) {
	idRaw, err := oleutil.GetProperty(token, "Id")
	if err != nil {
		return Voice{}, false
	}
	descRaw, err := oleutil.CallMethod(token, "GetDescription")
	if err != nil {
		return Voice{}, false
	}
	var languages []string
	if langRaw, err := oleutil.CallMethod(token, "GetAttribute", "Language"); err == nil && langRaw != nil {
		languages = sapiLanguages(langRaw.ToString())
	}
	return Voice{
		ID:       idRaw.ToString(),
		Name:     descRaw.ToString(),
		Provider: "sapi5",
		Language: languages,
	}, true
}

func findSAPIVoiceToken(voice *ole.IDispatch, requested string) (*ole.IDispatch, error) {
	requestedNorm := strings.ToLower(strings.TrimSpace(requested))
	tokensRaw, err := oleutil.CallMethod(voice, "GetVoices")
	if err != nil {
		return nil, err
	}
	tokens := tokensRaw.ToIDispatch()
	defer tokens.Release()
	countRaw, err := oleutil.GetProperty(tokens, "Count")
	if err != nil {
		return nil, err
	}
	count := int(countRaw.Val)
	var partial *ole.IDispatch
	for idx := 0; idx < count; idx++ {
		tokenRaw, err := oleutil.CallMethod(tokens, "Item", idx)
		if err != nil {
			continue
		}
		token := tokenRaw.ToIDispatch()
		voiceInfo, ok := sapiVoiceFromToken(token)
		if !ok {
			token.Release()
			continue
		}
		idNorm := strings.ToLower(voiceInfo.ID)
		nameNorm := strings.ToLower(voiceInfo.Name)
		if voiceInfo.ID == requested || idNorm == requestedNorm || nameNorm == requestedNorm || strings.HasSuffix(idNorm, "\\"+requestedNorm) {
			return token, nil
		}
		if partial == nil && strings.Contains(nameNorm, requestedNorm) {
			partial = token
			continue
		}
		token.Release()
	}
	if partial != nil {
		return partial, nil
	}
	return nil, fmt.Errorf("SAPI5 voice %q not found", requested)
}

func sapiRate(rate int) int {
	if rate < -10 {
		return -10
	}
	if rate > 10 {
		return 10
	}
	return rate
}

func sapiVolume(volume int) int {
	if volume < 0 {
		return 0
	}
	if volume > 100 {
		return 100
	}
	return volume
}

func sapiLanguages(raw string) []string {
	var windowsLCID = map[string]string{
		"409":  "en-US",
		"809":  "en-GB",
		"1009": "en-CA",
		"c0c":  "fr-CA",
		"40c":  "fr-FR",
		"c0a":  "es-ES",
		"80a":  "es-MX",
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ";")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimPrefix(strings.TrimSpace(strings.ToLower(part)), "0x")
		part = strings.TrimLeft(part, "0")
		if part == "" {
			part = "0"
		}
		if part != "" {
			if mapped, ok := windowsLCID[part]; ok {
				out = append(out, mapped)
				continue
			}
			out = append(out, part)
		}
	}
	return out
}

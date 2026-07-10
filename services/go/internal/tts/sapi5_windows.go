//go:build windows

package tts

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
)

const sapiCreateForWrite = 3
const sapi5ShimDisabledEnv = "HAZE_SAPI5_SHIM_DISABLED"
const sapi5ShimPathEnv = "HAZE_SAPI5_SHIM_PATH"

// SAPI5Provider uses installed Windows SAPI5 voices.
type SAPI5Provider struct{}

// NewSAPI5Provider creates a Windows SAPI5 provider.
func NewSAPI5Provider() *SAPI5Provider {
	return &SAPI5Provider{}
}

func (p *SAPI5Provider) ID() string { return "sapi5" }

func (p *SAPI5Provider) ListVoices(ctx context.Context) ([]Voice, error) {
	voices, nativeErr := p.listVoicesNative(ctx)
	if !sapi5ShimDisabled() {
		if shimVoices, err := listSAPI5ShimVoices(ctx); err == nil {
			voices = mergeSAPIVoices(voices, shimVoices)
		}
	}
	if len(voices) > 0 {
		return voices, nil
	}
	return nil, nativeErr
}

func (p *SAPI5Provider) Synthesize(ctx context.Context, req Request) (Audio, error) {
	req.Text = prepareSAPIText(req.Text)
	audio, nativeErr := p.synthesizeNative(ctx, req)
	if nativeErr == nil || sapi5ShimDisabled() {
		return audio, nativeErr
	}
	if strings.TrimSpace(req.VoiceID) == "" || !sapiVoiceNotFound(nativeErr) {
		return audio, nativeErr
	}
	if shimAudio, err := synthesizeWithSAPI5Shim(ctx, req); err == nil {
		return shimAudio, nil
	}
	return audio, nativeErr
}

func (p *SAPI5Provider) listVoicesNative(ctx context.Context) ([]Voice, error) {
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

func (p *SAPI5Provider) synthesizeNative(ctx context.Context, req Request) (Audio, error) {
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

type sapi5ShimSynthRequest struct {
	Text            string      `json:"text"`
	VoiceID         string      `json:"voice_id,omitempty"`
	Language        string      `json:"language,omitempty"`
	OutputFormat    AudioFormat `json:"output_format,omitempty"`
	Rate            int         `json:"rate,omitempty"`
	Volume          int         `json:"volume,omitempty"`
	SentenceSilence float64     `json:"sentence_silence,omitempty"`
}

type sapi5ShimSynthResponse struct {
	Format     AudioFormat `json:"format"`
	SampleRate int         `json:"sample_rate,omitempty"`
	Channels   int         `json:"channels,omitempty"`
	DataBase64 string      `json:"data_base64"`
}

func listSAPI5ShimVoices(ctx context.Context) ([]Voice, error) {
	shim, err := sapi5ShimPath()
	if err != nil {
		return nil, err
	}
	var voices []Voice
	if err := runSAPI5ShimJSON(ctx, shim, "list", nil, &voices); err != nil {
		return nil, err
	}
	return voices, nil
}

func synthesizeWithSAPI5Shim(ctx context.Context, req Request) (Audio, error) {
	shim, err := sapi5ShimPath()
	if err != nil {
		return Audio{}, err
	}
	payload := sapi5ShimSynthRequest{
		Text:            req.Text,
		VoiceID:         req.VoiceID,
		Language:        req.Language,
		OutputFormat:    req.OutputFormat,
		Rate:            req.Rate,
		Volume:          req.Volume,
		SentenceSilence: req.SentenceSilence,
	}
	var response sapi5ShimSynthResponse
	if err := runSAPI5ShimJSON(ctx, shim, "synthesize", payload, &response); err != nil {
		return Audio{}, err
	}
	data, err := base64.StdEncoding.DecodeString(response.DataBase64)
	if err != nil {
		return Audio{}, fmt.Errorf("SAPI5 compatibility shim returned invalid audio: %w", err)
	}
	return Audio{
		Format:     response.Format,
		SampleRate: response.SampleRate,
		Channels:   response.Channels,
		Data:       data,
	}, nil
}

func runSAPI5ShimJSON(ctx context.Context, shim string, command string, input any, output any) error {
	cmd := exec.CommandContext(ctx, shim, command)
	cmd.Env = append(os.Environ(), sapi5ShimDisabledEnv+"=1")
	if input != nil {
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return err
		}
		go func() {
			defer stdin.Close()
			_ = json.NewEncoder(stdin).Encode(input)
		}()
	}
	raw, err := cmd.Output()
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			detail := strings.TrimSpace(string(exitErr.Stderr))
			if detail != "" {
				return fmt.Errorf("SAPI5 compatibility shim failed: %s", detail)
			}
		}
		return fmt.Errorf("SAPI5 compatibility shim failed: %w", err)
	}
	if output == nil {
		return nil
	}
	if err := json.Unmarshal(raw, output); err != nil {
		return fmt.Errorf("SAPI5 compatibility shim returned invalid JSON: %w", err)
	}
	return nil
}

func sapi5ShimPath() (string, error) {
	if raw := strings.TrimSpace(os.Getenv(sapi5ShimPathEnv)); raw != "" {
		if info, err := os.Stat(raw); err == nil && !info.IsDir() {
			return raw, nil
		}
		return "", fmt.Errorf("%w: SAPI5 compatibility shim not found at %s", ErrProviderUnavailable, raw)
	}
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	path := filepath.Join(filepath.Dir(exe), "haze-sapi5-shim.exe")
	if info, err := os.Stat(path); err == nil && !info.IsDir() {
		return path, nil
	}
	return "", fmt.Errorf("%w: SAPI5 compatibility shim not bundled", ErrProviderUnavailable)
}

func sapi5ShimDisabled() bool {
	value := strings.TrimSpace(os.Getenv(sapi5ShimDisabledEnv))
	return value == "1" || strings.EqualFold(value, "true")
}

func mergeSAPIVoices(native []Voice, shim []Voice) []Voice {
	out := append([]Voice{}, native...)
	seen := make(map[string]struct{}, len(out))
	for _, voice := range out {
		seen[strings.ToLower(strings.TrimSpace(voice.ID))] = struct{}{}
		seen[strings.ToLower(strings.TrimSpace(voice.Name))] = struct{}{}
	}
	for _, voice := range shim {
		id := strings.ToLower(strings.TrimSpace(voice.ID))
		name := strings.ToLower(strings.TrimSpace(voice.Name))
		if _, ok := seen[id]; ok && id != "" {
			continue
		}
		if _, ok := seen[name]; ok && name != "" {
			continue
		}
		out = append(out, voice)
		seen[id] = struct{}{}
		seen[name] = struct{}{}
	}
	return out
}

func sapiVoiceNotFound(err error) bool {
	return err != nil && strings.Contains(strings.ToLower(err.Error()), "voice") && strings.Contains(strings.ToLower(err.Error()), "not found")
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

package tts

// DefaultProviders returns the built-in TTS provider registry.
func DefaultProviders() map[string]Provider {
	sapi := NewSAPI5Provider()
	espeak := NewESpeakProvider("")
	piper := NewPiperProvider("", "")
	kokoro := NewKokoroProvider()
	f5 := NewF5TTSProvider()
	chatterbox := NewChatterboxProvider()
	return map[string]Provider{
		sapi.ID():       sapi,
		espeak.ID():     espeak,
		piper.ID():      piper,
		kokoro.ID():     kokoro,
		f5.ID():         f5,
		chatterbox.ID(): chatterbox,
	}
}

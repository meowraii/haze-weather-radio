package tts

import "regexp"

var sapiPronunciationReplacements = []struct {
	pattern     *regexp.Regexp
	replacement string
}{
	{regexp.MustCompile(`(?i)\bwinds\b`), "windz"},
	{regexp.MustCompile(`(?i)\bkilometers\b`), "killometers"},
	{regexp.MustCompile(`(?i)\bkilometer\b`), "killometer"},
	{regexp.MustCompile(`(?i)\bkilometres\b`), "killometers"},
	{regexp.MustCompile(`(?i)\bkilometre\b`), "killometer"},
}

// prepareSAPIText applies known pronunciation workarounds for SAPI-backed voices.
func prepareSAPIText(text string) string {
	for _, entry := range sapiPronunciationReplacements {
		text = entry.pattern.ReplaceAllString(text, entry.replacement)
	}
	return text
}

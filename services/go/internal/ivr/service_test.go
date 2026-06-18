package ivr

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func TestWriteProductTwiMLDoesNotPreRenderOnCacheMiss(t *testing.T) {
	cfg := loadedConfig{
		BaseDir: t.TempDir(),
		IVR: Config{
			DefaultLanguage: "en-CA",
			DefaultPackages: []string{"current_conditions"},
		},
		Prompts: defaultPromptConfig(),
	}
	service := &Service{cfg: cfg}
	service.cache = NewProductCache(cfg, nil)

	request := httptest.NewRequest("POST", "http://ivr.test/ivr/v1/twiml", nil)
	response := httptest.NewRecorder()
	service.writeProductTwiML(response, request, ResolvedLocation{
		Code:     "06040",
		FeedID:   "sk-0001",
		Language: "en-CA",
		Name:     "Saskatoon",
	}, []string{"forecast"}, "http://ivr.test/next")

	body := response.Body.String()
	if strings.Contains(body, "unavailable") {
		t.Fatalf("product TwiML should not fail before the audio URL can render: %s", body)
	}
	if !strings.Contains(body, "/ivr/v1/prompt?line=one_moment") {
		t.Fatalf("cache miss should play one_moment before audio URL: %s", body)
	}
	if !strings.Contains(body, "/ivr/v1/audio") || !strings.Contains(body, "packages=forecast") {
		t.Fatalf("product TwiML did not include audio URL with package: %s", body)
	}
	if !strings.Contains(body, "<Redirect>http://ivr.test/next</Redirect>") {
		t.Fatalf("product TwiML did not preserve redirect: %s", body)
	}
}

func TestProductCacheKeyUsesCallerLanguageWhenMenuDoesNotOverride(t *testing.T) {
	cfg := loadedConfig{
		BaseDir: t.TempDir(),
		IVR: Config{
			DefaultLanguage: "en-CA",
			DefaultPackages: []string{"forecast"},
		},
		Prompts: defaultPromptConfig(),
	}
	cache := NewProductCache(cfg, nil)
	location := ResolvedLocation{Code: "06040", FeedID: "sk-0001", Language: "en-CA"}
	enKey, _, enLang, _, _ := cache.productCacheKey(location, []string{"forecast"})
	location.Language = "fr-CA"
	frKey, _, frLang, _, _ := cache.productCacheKey(location, []string{"forecast"})

	if enLang != "en-CA" || frLang != "fr-CA" {
		t.Fatalf("languages = %q/%q", enLang, frLang)
	}
	if enKey == frKey {
		t.Fatalf("language-specific IVR products must not share a cache key")
	}
}

func TestEntryBroadcastUsesConfiguredBroadcastPackages(t *testing.T) {
	cfg := loadedConfig{
		BaseDir: t.TempDir(),
		IVR: Config{
			DefaultLanguage:   "en-CA",
			DefaultPackages:   []string{"current_conditions"},
			BroadcastPackages: []string{"alerts", "forecast"},
		},
		Feeds: []feedXML{{ID: "sk-0001", EnabledRaw: "true"}},
		Prompts: withPromptOption(defaultPromptConfig(), "entry", menuOption{
			Digit:  "4",
			Action: "broadcast",
		}),
	}
	service := &Service{cfg: cfg}
	service.cache = NewProductCache(cfg, nil)

	request := formRequest("http://ivr.test/ivr/v1/twiml?state=entry", url.Values{"Digits": {"4"}})
	response := httptest.NewRecorder()
	service.handleEntryDigit(response, request)

	body := response.Body.String()
	if !strings.Contains(body, "packages=alerts%2Cforecast") {
		t.Fatalf("entry broadcast did not use configured broadcast packages: %s", body)
	}
}

func TestEntryBroadcastOptionPackagesOverrideConfiguredDefault(t *testing.T) {
	cfg := loadedConfig{
		BaseDir: t.TempDir(),
		IVR: Config{
			DefaultLanguage:   "en-CA",
			DefaultPackages:   []string{"current_conditions"},
			BroadcastPackages: []string{"alerts", "forecast"},
		},
		Feeds: []feedXML{{ID: "sk-0001", EnabledRaw: "true"}},
		Prompts: withPromptOption(defaultPromptConfig(), "entry", menuOption{
			Digit:    "4",
			Action:   "broadcast",
			Packages: "current_conditions,air_quality",
		}),
	}
	service := &Service{cfg: cfg}
	service.cache = NewProductCache(cfg, nil)

	request := formRequest("http://ivr.test/ivr/v1/twiml?state=entry", url.Values{"Digits": {"4"}})
	response := httptest.NewRecorder()
	service.handleEntryDigit(response, request)

	body := response.Body.String()
	if !strings.Contains(body, "packages=current_conditions%2Cair_quality") {
		t.Fatalf("entry broadcast did not use option packages: %s", body)
	}
}

func withPromptOption(cfg PromptConfig, menuID string, option menuOption) PromptConfig {
	menu := cfg.Menus[menuID]
	menu.Options = append(menu.Options, option)
	cfg.Menus[menuID] = menu
	return cfg
}

func formRequest(target string, values url.Values) *http.Request {
	request := httptest.NewRequest(http.MethodPost, target, strings.NewReader(values.Encode()))
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return request
}

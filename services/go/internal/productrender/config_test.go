package productrender

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadCombinedProductsBuildsProfilesAndText(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "products.xml")
	raw := `<?xml version="1.0" encoding="UTF-8"?>
<ProductText version="1.1">
  <defaults>
    <enabled>true</enabled>
    <reader_id>00</reader_id>
  </defaults>
  <product id="forecast" enabled="true">
    <lang iso="en-US" readerid="01">
      <text key="opener">Forecast for {site}.</text>
      <region>
        <shortterm>
          <text key="today">For today. {fc_text}</text>
        </shortterm>
        <extended>
          <text key="opener">The extended outlook.</text>
          <text key="period">For {period}. {fc_text}</text>
        </extended>
      </region>
    </lang>
    <lang iso="fr-CA" readerid="02">
      <text key="opener">Previsions pour {site}.</text>
    </lang>
  </product>
  <product id="current_conditions" enabled="true">
    <lang iso="en" readerid="03">
      <primary>
        <placeholder>The report at {location} was not available.</placeholder>
        <text>At {location},</text>
        <text>The temperature was {ctemp} degrees.</text>
      </primary>
    </lang>
  </product>
  <product id="alerts" enabled="false" />
</ProductText>`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}

	packages, productText, err := loadCombinedProducts(path)
	if err != nil {
		t.Fatal(err)
	}
	if !packages["forecast"].Enabled {
		t.Fatalf("forecast should be enabled")
	}
	if packages["forecast"].ReaderID != "01" {
		t.Fatalf("forecast reader = %q, want 01", packages["forecast"].ReaderID)
	}
	if packages["current_conditions"].ReaderID != "03" {
		t.Fatalf("current_conditions reader = %q, want 03", packages["current_conditions"].ReaderID)
	}
	if packages["alerts"].Enabled {
		t.Fatalf("alerts should be disabled")
	}
	if packages["alerts"].ReaderID != "00" {
		t.Fatalf("alerts reader = %q, want default 00", packages["alerts"].ReaderID)
	}
	if got := productText["forecast"]["opener"]["en-us"]; got != "Forecast for {site}." {
		t.Fatalf("en opener = %q", got)
	}
	if got := productText["forecast"]["opener"]["fr-ca"]; got != "Previsions pour {site}." {
		t.Fatalf("fr opener = %q", got)
	}
	if got := productText["forecast"]["today"]["en-us"]; got != "For today. {fc_text}" {
		t.Fatalf("forecast today = %q", got)
	}
	if got := productText["forecast"]["shortterm.today"]["en-us"]; got != "For today. {fc_text}" {
		t.Fatalf("forecast shortterm today = %q", got)
	}
	if got := productText["forecast"]["opener"]["en-us"]; got != "Forecast for {site}." {
		t.Fatalf("nested extended opener should not replace main opener, got %q", got)
	}
	if got := productText["forecast"]["extended.opener"]["en-us"]; got != "The extended outlook." {
		t.Fatalf("forecast extended opener = %q", got)
	}
	if got := productText["forecast"]["extended.period"]["en-us"]; got != "For {period}. {fc_text}" {
		t.Fatalf("forecast extended period = %q", got)
	}
	if got := productText["current_conditions"]["primary.placeholder"]["en"]; got != "The report at {location} was not available." {
		t.Fatalf("primary placeholder = %q", got)
	}
	if got := productText["current_conditions"]["primary.text.1"]["en"]; got != "At {location}," {
		t.Fatalf("primary text 1 = %q", got)
	}
	if got := productText["current_conditions"]["primary.text.2"]["en"]; got != "The temperature was {ctemp} degrees." {
		t.Fatalf("primary text 2 = %q", got)
	}
}

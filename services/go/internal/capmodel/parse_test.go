package capmodel

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseCAP(t *testing.T) {
	raw := []byte(`<?xml version="1.0"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>ABC-123</identifier>
  <sender>sender@example.test</sender>
  <sent>2026-06-14T12:00:00Z</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en-CA</language>
    <category>Met</category>
    <event>Severe Thunderstorm Warning</event>
    <urgency>Immediate</urgency>
    <severity>Severe</severity>
    <certainty>Likely</certainty>
    <headline>Storm warning</headline>
    <eventCode><valueName>SAME</valueName><value>SVR</value></eventCode>
    <area>
      <areaDesc>Test Region</areaDesc>
      <geocode><valueName>profile:CAP-CP:Location:0.4</valueName><value>4611045</value></geocode>
    </area>
  </info>
</alert>`)

	alert, err := ParseCAP(raw)
	if err != nil {
		t.Fatalf("ParseCAP returned error: %v", err)
	}
	if alert.Identifier != "ABC-123" {
		t.Fatalf("identifier = %q", alert.Identifier)
	}
	if len(alert.Infos) != 1 {
		t.Fatalf("infos = %d", len(alert.Infos))
	}
	if alert.Infos[0].Areas[0].Geocodes[0].Value != "4611045" {
		t.Fatalf("geocode = %q", alert.Infos[0].Areas[0].Geocodes[0].Value)
	}
	if len(alert.Infos[0].EventCodes) != 1 || alert.Infos[0].EventCodes[0].Value != "SVR" {
		t.Fatalf("event codes = %#v", alert.Infos[0].EventCodes)
	}
}

func TestParseCAPReferenceFixtures(t *testing.T) {
	matches, err := filepath.Glob(filepath.Join("..", "..", "testdata", "cap", "example_*.xml"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) == 0 {
		t.Fatal("no CAP reference fixtures found")
	}
	for _, path := range matches {
		t.Run(filepath.Base(path), func(t *testing.T) {
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			alert, err := ParseCAP(raw)
			if err != nil {
				t.Fatalf("ParseCAP returned error: %v", err)
			}
			if alert.Identifier == "" {
				t.Fatal("identifier is empty")
			}
			if len(alert.Infos) == 0 {
				t.Fatal("infos are empty")
			}
		})
	}
}

func TestParseCAPRejectsMissingRequiredHeader(t *testing.T) {
	raw := []byte(`<?xml version="1.0"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>ABC-123</identifier>
  <sender>sender@example.test</sender>
  <sent>not-a-time</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
</alert>`)

	if _, err := ParseCAP(raw); err == nil {
		t.Fatal("ParseCAP accepted invalid CAP header")
	}
}

func TestParseCAPAcceptsNAADSHeartbeatWithoutInfo(t *testing.T) {
	raw := []byte(`<?xml version="1.0"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>heartbeat-1</identifier>
  <sender>NAADS-Heartbeat</sender>
  <sent>2026-06-14T12:00:00Z</sent>
  <status>System</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <references>sender,abc,2026-06-14T11:59:00Z</references>
</alert>`)

	alert, err := ParseCAP(raw)
	if err != nil {
		t.Fatalf("ParseCAP returned error: %v", err)
	}
	if alert.Status != "System" || len(alert.Infos) != 0 {
		t.Fatalf("heartbeat parse = %#v", alert)
	}
}

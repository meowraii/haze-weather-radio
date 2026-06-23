package capingest

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
)

type cancelPublisher struct {
	count  atomic.Int32
	cancel context.CancelFunc
}

func (p *cancelPublisher) Publish(event events.Event) error {
	if event.Type == "cap.alert.received" {
		p.count.Add(1)
		p.cancel()
	}
	return nil
}

func TestPollAtomOnlyMarksSeenAfterSuccessfulPublish(t *testing.T) {
	var capRequests atomic.Int32
	var atomRequests atomic.Int32
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/feed":
			atomRequests.Add(1)
			w.Header().Set("Content-Type", "application/atom+xml")
			_, _ = fmt.Fprintf(w, `<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>tag:test:alert</id>
    <updated>2026-06-16T21:50:00Z</updated>
    <link href="%s/cap" rel="alternate"/>
  </entry>
</feed>`, server.URL)
		case "/cap":
			if capRequests.Add(1) == 1 {
				http.Error(w, "temporary miss", http.StatusBadGateway)
				return
			}
			w.Header().Set("Content-Type", "application/cap+xml")
			_, _ = w.Write([]byte(testCAPXML()))
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	atomFixture := fmt.Sprintf(`<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>tag:test:alert</id>
    <updated>2026-06-16T21:50:00Z</updated>
    <link href="%s/cap" rel="alternate"/>
  </entry>
</feed>`, server.URL)
	parsedEntries, err := ParseAtomEntries([]byte(atomFixture))
	if err != nil || len(parsedEntries) != 1 {
		t.Fatalf("bad atom fixture: entries=%#v err=%v", parsedEntries, err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	publisher := &cancelPublisher{cancel: cancel}
	poller := NewPoller(publisher)
	source := SourceConfig{
		ID:           "test",
		URL:          server.URL + "/feed",
		PollInterval: 5 * time.Millisecond,
		Timeout:      time.Second,
	}
	fetchedEntries, fetchErr := poller.fetchAtom(context.Background(), source)
	if fetchErr != nil || len(fetchedEntries) != 1 {
		t.Fatalf("fetchAtom entries=%#v err=%v atom_requests=%d", fetchedEntries, fetchErr, atomRequests.Load())
	}

	err = poller.PollAtom(ctx, source)
	if err != context.Canceled {
		t.Fatalf("PollAtom error = %v (published=%d atom_requests=%d cap_requests=%d)", err, publisher.count.Load(), atomRequests.Load(), capRequests.Load())
	}
	if publisher.count.Load() != 1 {
		t.Fatalf("published alerts = %d", publisher.count.Load())
	}
	if capRequests.Load() < 2 {
		t.Fatalf("CAP was not retried after transient failure")
	}
}

func TestFetchAtomFallsBackToSecondaryURL(t *testing.T) {
	var primaryRequests atomic.Int32
	var fallbackRequests atomic.Int32
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/primary":
			primaryRequests.Add(1)
			http.Error(w, "primary unavailable", http.StatusBadGateway)
		case "/fallback":
			fallbackRequests.Add(1)
			w.Header().Set("Content-Type", "application/atom+xml")
			_, _ = fmt.Fprintf(w, `<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>tag:test:fallback-alert</id>
    <updated>2026-06-16T21:55:00Z</updated>
    <link href="%s/cap" rel="alternate"/>
  </entry>
</feed>`, server.URL)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	poller := NewPoller(events.NewJSONLPublisher(io.Discard))
	entries, err := poller.fetchAtom(context.Background(), SourceConfig{
		ID:      "naads",
		URL:     server.URL + "/primary",
		URLs:    []string{server.URL + "/fallback"},
		Timeout: time.Second,
	})
	if err != nil {
		t.Fatalf("fetchAtom: %v", err)
	}
	if len(entries) != 1 || entries[0].ID != "tag:test:fallback-alert" {
		t.Fatalf("entries = %#v", entries)
	}
	if primaryRequests.Load() != 1 || fallbackRequests.Load() != 1 {
		t.Fatalf("requests primary=%d fallback=%d", primaryRequests.Load(), fallbackRequests.Load())
	}
}

func testCAPXML() string {
	return `<?xml version="1.0"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>urn:test:poller</identifier>
  <sender>sender@example.test</sender>
  <sent>2026-06-16T21:50:00Z</sent>
  <status>Actual</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <info>
    <language>en-CA</language>
    <category>Met</category>
    <event>thunderstorm</event>
    <urgency>Immediate</urgency>
    <severity>Moderate</severity>
    <certainty>Likely</certainty>
    <headline>yellow warning - severe thunderstorm - in effect</headline>
    <area><areaDesc>Test</areaDesc></area>
  </info>
</alert>`
}

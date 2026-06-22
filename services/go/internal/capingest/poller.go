package capingest

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
)

const (
	defaultTimeout      = 15 * time.Second
	defaultPollInterval = 30 * time.Second
	maxNAADSBodyBytes   = 5 << 20
)

// Poller fetches Atom/CAP sources and publishes normalized alert events.
type Poller struct {
	client    *http.Client
	publisher events.Publisher
}

// NewPoller creates a CAP ingest poller.
func NewPoller(publisher events.Publisher) *Poller {
	return &Poller{
		client: &http.Client{
			Timeout: defaultTimeout,
		},
		publisher: publisher,
	}
}

// FetchArchive fetches an Atom archive once and emits alert events for each CAP item.
func (p *Poller) FetchArchive(ctx context.Context, source SourceConfig) (int, error) {
	entries, err := p.fetchAtom(ctx, source)
	if err != nil {
		return 0, err
	}

	count := 0
	for _, entry := range entries {
		alert, err := p.fetchFirstCAP(ctx, source, entry.Links)
		if err != nil {
			log.Printf("[%s] CAP fetch skipped %s: %v", source.ID, entry.ID, err)
			continue
		}
		if err := p.publishAlert(source, alert); err != nil {
			return count, err
		}
		count++
	}
	return count, nil
}

// PollAtom continuously polls an Atom feed until the context is cancelled.
func (p *Poller) PollAtom(ctx context.Context, source SourceConfig) error {
	pollInterval := source.PollInterval
	if pollInterval <= 0 {
		pollInterval = defaultPollInterval
	}

	seen := map[string]string{}
	for {
		entries, err := p.fetchAtom(ctx, source)
		if err == nil {
			for _, entry := range entries {
				if previous, ok := seen[entry.ID]; ok && previous == entry.Updated {
					continue
				}
				alert, err := p.fetchFirstCAP(ctx, source, entry.Links)
				if err != nil {
					log.Printf("[%s] CAP fetch skipped %s: %v", source.ID, entry.ID, err)
					continue
				}
				if err := p.publishAlert(source, alert); err != nil {
					return err
				}
				seen[entry.ID] = entry.Updated
			}
		} else {
			log.Printf("[%s] CAP atom poll failed: %v", source.ID, err)
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(pollInterval):
		}
	}
}

func (p *Poller) fetchAtom(ctx context.Context, source SourceConfig) ([]AtomEntry, error) {
	body, err := p.httpGet(ctx, source, source.URL, "application/atom+xml, application/xml;q=0.9, */*;q=0.1")
	if err != nil {
		return nil, err
	}
	return ParseAtomEntries(body)
}

func (p *Poller) fetchFirstCAP(ctx context.Context, source SourceConfig, links []string) (Alert, error) {
	for _, link := range links {
		body, err := p.httpGet(ctx, source, link, "application/cap+xml, application/xml;q=0.9, */*;q=0.1")
		if err != nil {
			continue
		}
		alert, err := ParseCAP(body)
		if err == nil && alert.Identifier != "" {
			return alert, nil
		}
	}
	return Alert{}, errors.New("no parseable CAP alert found")
}

func (p *Poller) httpGet(ctx context.Context, source SourceConfig, url string, accept string) ([]byte, error) {
	timeout := source.Timeout
	if timeout <= 0 {
		timeout = defaultTimeout
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	request.Header.Set("Accept", accept)
	if source.UserAgent != "" {
		request.Header.Set("User-Agent", source.UserAgent)
	}

	response, err := p.client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()

	if response.StatusCode < 200 || response.StatusCode >= 300 {
		return nil, fmt.Errorf("unexpected HTTP status %s from %s", response.Status, url)
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, maxNAADSBodyBytes+1))
	if err != nil {
		return nil, err
	}
	if len(body) > maxNAADSBodyBytes {
		return nil, fmt.Errorf("response from %s exceeds %d bytes", url, maxNAADSBodyBytes)
	}
	return body, nil
}

func (p *Poller) publishAlert(source SourceConfig, alert Alert) error {
	return p.publisher.Publish(events.Event{
		ID:      alert.Identifier,
		Type:    "cap.alert.received",
		Source:  source.ID,
		Subject: alert.Identifier,
		Data: map[string]any{
			"alert": alert,
		},
	})
}

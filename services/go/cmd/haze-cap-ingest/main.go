package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/capingest"
	"github.com/meowraii/haze-weather-radio/services/go/internal/events"
	"github.com/meowraii/haze-weather-radio/services/go/internal/processguard"
)

const defaultNAADSURL = "https://rss.naad-adna.pelmorex.com/"

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(0)
	if err := run(); err != nil && !errors.Is(err, context.Canceled) {
		log.Fatalf("haze-cap-ingest: %v", err)
	}
}

func run() error {
	sourceID := flag.String("source-id", "cap", "source identifier used in emitted events")
	sourceKind := flag.String("source", "naads", "source preset: naads, nws, or custom")
	url := flag.String("url", "", "Atom feed URL")
	once := flag.Bool("once", false, "fetch once and exit")
	interval := flag.Duration("interval", 30*time.Second, "poll interval for continuous mode")
	timeout := flag.Duration("timeout", 15*time.Second, "HTTP request timeout")
	startupDelay := flag.Duration("startup-delay", 2*time.Second, "delay before first continuous poll")
	userAgent := flag.String("user-agent", "haze-weather-radio-go-cap-ingest/0.1", "HTTP user agent")
	flag.Parse()

	sourceURL := *url
	if sourceURL == "" {
		sourceURL = defaultURLForSource(*sourceKind)
	}
	if sourceURL == "" {
		return fmt.Errorf("missing --url for source %q", *sourceKind)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx = processguard.WithParent(ctx)

	publisher := events.Publisher(events.NewJSONLPublisher(os.Stdout))
	if bridgeAddr := os.Getenv("HAZE_HOST_BRIDGE_ADDR"); bridgeAddr != "" {
		publisher = events.NewHostBridgePublisher(bridgeAddr)
	}
	poller := capingest.NewPoller(publisher)
	source := capingest.SourceConfig{
		ID:           *sourceID,
		URL:          sourceURL,
		PollInterval: *interval,
		Timeout:      *timeout,
		UserAgent:    *userAgent,
	}

	if *once {
		_, err := poller.FetchArchive(ctx, source)
		return err
	}
	if *startupDelay > 0 {
		timer := time.NewTimer(*startupDelay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
	return poller.PollAtom(ctx, source)
}

func defaultURLForSource(kind string) string {
	switch kind {
	case "naads":
		return defaultNAADSURL
	case "nws":
		return "https://api.weather.gov/alerts/active.atom"
	case "custom":
		return ""
	default:
		return ""
	}
}

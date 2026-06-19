package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/dataingest"
	"github.com/meowraii/haze-weather-radio/services/go/internal/processguard"
)

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(0)
	if err := run(); err != nil && !errors.Is(err, context.Canceled) {
		log.Fatalf("haze-data-ingest: %v", err)
	}
}

func run() error {
	configPath := flag.String("config", "config.yaml", "Haze config path")
	interval := flag.Duration("interval", 45*time.Minute, "poll interval")
	timeout := flag.Duration("timeout", 20*time.Second, "HTTP request timeout")
	once := flag.Bool("once", false, "fetch once and exit")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx = processguard.WithParent(ctx)

	return dataingest.Run(ctx, dataingest.Options{
		ConfigPath: *configPath,
		Interval:   *interval,
		Timeout:    *timeout,
		Once:       *once,
	})
}

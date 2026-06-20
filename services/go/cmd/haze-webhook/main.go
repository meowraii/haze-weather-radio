package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/processguard"
	"github.com/meowraii/haze-weather-radio/services/go/internal/webhook"
)

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(0)
	if err := run(); err != nil && err != context.Canceled {
		log.Fatalf("haze-webhook: %v", err)
	}
}

func run() error {
	configPath := flag.String("config", "config.yaml", "Haze YAML configuration path")
	bridgeAddr := flag.String("bridge", os.Getenv("HAZE_HOST_BRIDGE_ADDR"), "host event bridge address")
	webhooksPath := flag.String("webhooks", "", "Discord webhooks XML path")
	timeout := flag.Duration("timeout", 15*time.Second, "Discord webhook HTTP timeout")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx = processguard.WithParent(ctx)
	return webhook.Run(ctx, webhook.Options{
		ConfigPath:   *configPath,
		BridgeAddr:   *bridgeAddr,
		WebhooksPath: *webhooksPath,
		Timeout:      *timeout,
	})
}

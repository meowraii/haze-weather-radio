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
	"github.com/meowraii/haze-weather-radio/services/go/internal/productrender"
)

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(0)
	if err := run(); err != nil && err != context.Canceled {
		log.Fatalf("haze-product-render: %v", err)
	}
}

func run() error {
	configPath := flag.String("config", "config.yaml", "Haze YAML configuration path")
	bridgeAddr := flag.String("bridge", os.Getenv("HAZE_HOST_BRIDGE_ADDR"), "host event bridge address")
	refresh := flag.Duration("refresh", 5*time.Minute, "configuration refresh interval")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx = processguard.WithParent(ctx)

	return productrender.Run(ctx, productrender.Options{
		ConfigPath: *configPath,
		BridgeAddr: *bridgeAddr,
		Refresh:    *refresh,
	})
}

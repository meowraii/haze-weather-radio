package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/meowraii/haze-weather-radio/services/go/internal/ivr"
	"github.com/meowraii/haze-weather-radio/services/go/internal/processguard"
)

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(0)
	if err := run(); err != nil && err != context.Canceled {
		log.Fatalf("haze-ivr: %v", err)
	}
}

func run() error {
	configPath := flag.String("config", "config.yaml", "Haze YAML configuration path")
	bridgeAddr := flag.String("bridge", os.Getenv("HAZE_HOST_BRIDGE_ADDR"), "host event bridge address")
	httpAddr := flag.String("http-addr", "", "override IVR provider webhook listen address")
	sipAddr := flag.String("sip-addr", "", "override SIP UDP listen address")
	cacheDir := flag.String("cache-dir", "", "override IVR cache directory")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx = processguard.WithParent(ctx)

	return ivr.Run(ctx, ivr.Options{
		ConfigPath: *configPath,
		BridgeAddr: *bridgeAddr,
		HTTPAddr:   *httpAddr,
		SIPAddr:    *sipAddr,
		CacheDir:   *cacheDir,
	})
}

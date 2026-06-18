package main

import (
	"context"
	"flag"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/playlist"
)

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(0)
	if err := run(); err != nil && err != context.Canceled {
		log.Fatalf("haze-playlist: %v", err)
	}
}

func run() error {
	configPath := flag.String("config", "config.yaml", "Haze YAML configuration path")
	bridgeAddr := flag.String("bridge", os.Getenv("HAZE_HOST_BRIDGE_ADDR"), "host event bridge address")
	tick := flag.Duration("tick", 500*time.Millisecond, "scheduler tick interval")
	lookahead := flag.Duration("lookahead", 2*time.Minute, "playlist prediction and queue lookahead")
	outDir := flag.String("out-dir", filepath.Join("runtime", "audio", "playlist"), "rendered playlist audio output directory")
	flag.Parse()

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	return playlist.Run(ctx, playlist.Options{
		ConfigPath: *configPath,
		BridgeAddr: *bridgeAddr,
		Tick:       *tick,
		Lookahead:  *lookahead,
		OutDir:     *outDir,
	})
}

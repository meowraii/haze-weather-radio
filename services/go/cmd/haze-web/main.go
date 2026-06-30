package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/meowraii/haze-weather-radio/services/go/internal/processguard"
	"github.com/meowraii/haze-weather-radio/services/go/internal/webgateway"
)

func main() {
	log.SetOutput(os.Stdout)
	log.SetFlags(0)
	if err := run(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatalf("haze-web: %v", err)
	}
}

func run() error {
	addr := flag.String("addr", "127.0.0.1:8081", "HTTP listen address")
	webroot := flag.String("webroot", "webroot", "webroot directory")
	configPath := flag.String("config", "config.yaml", "Haze config path")
	surface := flag.String("surface", "combined", "HTTP surface: public, admin, or combined")
	checkCodecs := flag.Bool("check-codecs", false, "print WebRTC codec capabilities and exit")
	requireOpus := flag.Bool("require-opus", false, "fail --check-codecs unless native Opus is available")
	flag.Parse()

	if *checkCodecs {
		capabilities := webgateway.WebRTCAudioCapabilities()
		if *requireOpus {
			opus, _ := capabilities["webrtc_opus"].(bool)
			if !opus {
				return fmt.Errorf("native Opus is not available in this haze-web build")
			}
		}
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		return encoder.Encode(capabilities)
	}

	if envFile := os.Getenv("HAZE_ENV_FILE"); envFile != "" {
		if err := webgateway.LoadDotEnv(envFile); err != nil {
			return err
		}
	} else {
		if err := webgateway.LoadDotEnv(filepath.Join(filepath.Dir(*configPath), ".env")); err != nil {
			return err
		}
		if err := webgateway.LoadDotEnv(".env"); err != nil {
			return err
		}
	}
	pprofServer := startPprofServer()

	config, err := webgateway.LoadConfig(*configPath)
	if err != nil {
		return err
	}
	tlsRuntime, err := webgateway.NewTLSRuntime(config, *configPath)
	if err != nil {
		return err
	}

	server := &http.Server{
		Addr:              *addr,
		Handler:           webgateway.NewServerWithSurface(config, *configPath, *webroot, *surface).Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		TLSConfig:         tlsRuntime.TLSConfig(),
	}
	normalizedSurface := webgateway.NormalizeSurface(*surface)
	var challengeServer *http.Server
	if tlsRuntime.HTTPChallengeEnabled(normalizedSurface) {
		challengeServer = &http.Server{
			Addr:              tlsRuntime.HTTPChallengeAddr,
			Handler:           tlsRuntime.HTTPChallengeHandler(server.Addr),
			ReadHeaderTimeout: 10 * time.Second,
		}
		go func() {
			log.Printf("haze-web acme challenge listening on %s", challengeServer.Addr)
			if err := challengeServer.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
				log.Printf("acme challenge listener failed: %v", err)
			}
		}()
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	ctx = processguard.WithParent(ctx)

	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			log.Printf("shutdown failed: %v", err)
		}
		if challengeServer != nil {
			if err := challengeServer.Shutdown(shutdownCtx); err != nil {
				log.Printf("acme challenge shutdown failed: %v", err)
			}
		}
		if pprofServer != nil {
			if err := pprofServer.Shutdown(shutdownCtx); err != nil {
				log.Printf("pprof shutdown failed: %v", err)
			}
		}
	}()

	if tlsRuntime.Enabled {
		log.Printf("haze-web %s listening with HTTPS on %s", *surface, *addr)
		if tlsRuntime.Mode == "acme" {
			return server.ListenAndServeTLS("", "")
		}
		return server.ListenAndServeTLS(tlsRuntime.CertFile, tlsRuntime.KeyFile)
	}
	log.Printf("haze-web %s listening on %s", *surface, *addr)
	return server.ListenAndServe()
}

func startPprofServer() *http.Server {
	if os.Getenv("HAZE_WEB_PPROF") == "" && os.Getenv("HAZE_WEB_PPROF_ADDR") == "" {
		return nil
	}
	addr := os.Getenv("HAZE_WEB_PPROF_ADDR")
	if addr == "" {
		addr = "127.0.0.1:6060"
	}
	server := &http.Server{
		Addr:              addr,
		Handler:           http.DefaultServeMux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		log.Printf("haze-web pprof listening on %s", addr)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("pprof listener failed: %v", err)
		}
	}()
	return server
}

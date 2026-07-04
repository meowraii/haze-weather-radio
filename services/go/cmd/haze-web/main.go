package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	_ "net/http/pprof"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
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

	handler := webgateway.NewServerWithSurface(config, *configPath, *webroot, *surface).Handler()
	server := &http.Server{
		Addr:              *addr,
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
		TLSConfig:         tlsRuntime.TLSConfig(),
	}
	normalizedSurface := webgateway.NormalizeSurface(*surface)
	var publicServers []*http.Server
	if publicPortEnabled(config, normalizedSurface) {
		publicServers = startPublicPortServers(config, handler, tlsRuntime)
	}
	publicPortServingACME := len(publicServers) > 0 && publicPortHTTPHandlesACME(config, tlsRuntime)
	var challengeServer *http.Server
	if tlsRuntime.HTTPChallengeEnabled(normalizedSurface) && !publicPortServingACME {
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
		for _, publicServer := range publicServers {
			if err := publicServer.Shutdown(shutdownCtx); err != nil {
				log.Printf("public web listener shutdown failed: %v", err)
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

func publicPortEnabled(config webgateway.Config, surface webgateway.WebSurface) bool {
	return config.Webpanel.PublicPort.Enabled && surface != webgateway.SurfaceAdmin
}

func publicPortHTTPHandlesACME(config webgateway.Config, tlsRuntime *webgateway.TLSRuntime) bool {
	return config.Webpanel.PublicPort.Enabled &&
		tlsRuntime != nil &&
		tlsRuntime.Enabled &&
		tlsRuntime.Mode == "acme" &&
		tlsRuntime.Manager != nil
}

func startPublicPortServers(config webgateway.Config, handler http.Handler, tlsRuntime *webgateway.TLSRuntime) []*http.Server {
	host := strings.TrimSpace(config.Webpanel.PublicPort.Host)
	if host == "" {
		host = "0.0.0.0"
	}
	httpPort := config.Webpanel.PublicPort.HTTPPort
	if httpPort <= 0 {
		httpPort = 80
	}
	httpsPort := config.Webpanel.PublicPort.HTTPSPort
	if httpsPort <= 0 {
		httpsPort = 443
	}

	servers := []*http.Server{}
	if tlsRuntime != nil && tlsRuntime.Enabled {
		httpsAddr := net.JoinHostPort(host, strconv.Itoa(httpsPort))
		httpsServer := &http.Server{
			Addr:              httpsAddr,
			Handler:           handler,
			ReadHeaderTimeout: 10 * time.Second,
			TLSConfig:         tlsRuntime.TLSConfig(),
		}
		servers = append(servers, httpsServer)
		go servePublicHTTPS(httpsServer, tlsRuntime)

		if tlsRuntime.RedirectHTTP || publicPortHTTPHandlesACME(config, tlsRuntime) {
			httpServer := &http.Server{
				Addr:              net.JoinHostPort(host, strconv.Itoa(httpPort)),
				Handler:           tlsRuntime.HTTPChallengeHandler(httpsAddr),
				ReadHeaderTimeout: 10 * time.Second,
			}
			servers = append(servers, httpServer)
			go servePublicHTTP(httpServer)
		}
		return servers
	}

	httpServer := &http.Server{
		Addr:              net.JoinHostPort(host, strconv.Itoa(httpPort)),
		Handler:           handler,
		ReadHeaderTimeout: 10 * time.Second,
	}
	servers = append(servers, httpServer)
	go servePublicHTTP(httpServer)
	return servers
}

func servePublicHTTP(server *http.Server) {
	log.Printf("haze-web public HTTP listening on %s", server.Addr)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Printf("public HTTP listener failed: %v", err)
	}
}

func servePublicHTTPS(server *http.Server, tlsRuntime *webgateway.TLSRuntime) {
	log.Printf("haze-web public HTTPS listening on %s", server.Addr)
	var err error
	if tlsRuntime.Mode == "acme" {
		err = server.ListenAndServeTLS("", "")
	} else {
		err = server.ListenAndServeTLS(tlsRuntime.CertFile, tlsRuntime.KeyFile)
	}
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Printf("public HTTPS listener failed: %v", err)
	}
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

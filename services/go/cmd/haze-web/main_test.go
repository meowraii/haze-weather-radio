package main

import (
	"net/http"
	"testing"
)

func TestConfigureTLSProtocolsDisablesHTTP2WhenRequested(t *testing.T) {
	t.Setenv("HAZE_WEB_DISABLE_HTTP2", "true")
	server := &http.Server{}

	if !configureTLSProtocols(server) {
		t.Fatal("configureTLSProtocols reported HTTP/2 was still enabled")
	}
	if server.TLSNextProto == nil {
		t.Fatal("TLSNextProto was not initialized")
	}
	if len(server.TLSNextProto) != 0 {
		t.Fatalf("TLSNextProto = %#v, want an empty protocol map", server.TLSNextProto)
	}
}

func TestConfigureTLSProtocolsLeavesHTTP2Default(t *testing.T) {
	t.Setenv("HAZE_WEB_DISABLE_HTTP2", "false")
	server := &http.Server{}

	if configureTLSProtocols(server) {
		t.Fatal("configureTLSProtocols reported HTTP/2 was disabled")
	}
	if server.TLSNextProto != nil {
		t.Fatalf("TLSNextProto = %#v, want nil", server.TLSNextProto)
	}
}

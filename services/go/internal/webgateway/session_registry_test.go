package webgateway

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
)

func TestRedisGroupedLoginRateIsAtomicAndResettable(t *testing.T) {
	server := miniredis.RunT(t)
	registry, err := newRedisSessionRegistry(context.Background(), "redis://"+server.Addr()+"/0", "haze:test")
	if err != nil {
		t.Fatal(err)
	}
	defer registry.Close()

	const attempts = 40
	var admitted atomic.Int32
	var wait sync.WaitGroup
	wait.Add(attempts)
	now := time.Now().UTC()
	for index := 0; index < attempts; index++ {
		index := index
		go func() {
			defer wait.Done()
			allowed, err := registry.AllowGroupedRate(
				context.Background(), "login_pair", "admin|198.51.100.90", "admin",
				fmt.Sprintf("attempt-%d", index), 5, 15*time.Minute, now,
			)
			if err != nil {
				t.Errorf("admit grouped rate entry: %v", err)
				return
			}
			if allowed {
				admitted.Add(1)
			}
		}()
	}
	wait.Wait()
	if got := admitted.Load(); got != 5 {
		t.Fatalf("Redis admitted attempts = %d, want 5", got)
	}

	if err := registry.ResetRateGroup(context.Background(), "login_pair", "admin"); err != nil {
		t.Fatal(err)
	}
	allowed, err := registry.AllowGroupedRate(
		context.Background(), "login_pair", "admin|203.0.113.22", "admin",
		"after-unlock", 5, 15*time.Minute, now,
	)
	if err != nil || !allowed {
		t.Fatalf("attempt after account rate reset: allowed=%v err=%v", allowed, err)
	}
}

func TestRedisOriginationRateUsesAtomicSlidingWindow(t *testing.T) {
	server := miniredis.RunT(t)
	registry, err := newRedisSessionRegistry(context.Background(), "redis://"+server.Addr()+"/0", "haze:test")
	if err != nil {
		t.Fatal(err)
	}
	defer registry.Close()

	now := time.Now().UTC()
	for index := 0; index < 2; index++ {
		allowed, err := registry.AllowRate(context.Background(), "origination", "account-id", fmt.Sprintf("request-%d", index), 2, time.Second, now)
		if err != nil || !allowed {
			t.Fatalf("origination request %d: allowed=%v err=%v", index, allowed, err)
		}
	}
	allowed, err := registry.AllowRate(context.Background(), "origination", "account-id", "request-3", 2, time.Second, now)
	if err != nil {
		t.Fatal(err)
	}
	if allowed {
		t.Fatal("third origination request in one second was admitted")
	}
	allowed, err = registry.AllowRate(context.Background(), "origination", "account-id", "request-4", 2, time.Second, now.Add(time.Second+time.Millisecond))
	if err != nil || !allowed {
		t.Fatalf("origination request after window: allowed=%v err=%v", allowed, err)
	}
}

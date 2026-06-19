package processguard

import (
	"context"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
)

// WithParent returns a context that is canceled when HAZE_PARENT_PID exits.
func WithParent(ctx context.Context) context.Context {
	raw := strings.TrimSpace(os.Getenv("HAZE_PARENT_PID"))
	if raw == "" {
		return ctx
	}
	pid, err := strconv.Atoi(raw)
	if err != nil || pid <= 0 {
		return ctx
	}
	childCtx, cancel := context.WithCancel(ctx)
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		defer cancel()
		for {
			select {
			case <-childCtx.Done():
				return
			case <-ticker.C:
				if !parentAlive(pid) {
					log.Printf("haze parent process %d exited; shutting down", pid)
					return
				}
			}
		}
	}()
	return childCtx
}

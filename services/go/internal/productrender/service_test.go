package productrender

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"
)

func TestRunDoesNotPanicBeforeStoreOpens(t *testing.T) {
	dir := t.TempDir()
	writeFixture(t, dir)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()

	err := Run(ctx, Options{
		ConfigPath: filepath.Join(dir, "config.yaml"),
		BridgeAddr: "",
		Refresh:    time.Second,
	})
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		t.Fatalf("Run returned %v, want context cancellation", err)
	}
}

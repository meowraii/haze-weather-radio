package webgateway

import "testing"

func TestListenerTrackerDeduplicatesOverlappingConnectionsFromOneIP(t *testing.T) {
	tracker := NewListenerTracker()
	releaseFirst, first, ok := tracker.TryAcquire("feed-1", "192.0.2.10")
	if !ok || first.Current != 1 || first.Peak != 1 {
		t.Fatalf("first acquire = %#v, ok=%v", first, ok)
	}

	releaseSecond, second, ok := tracker.TryAcquire("feed-1", "192.0.2.10")
	if !ok {
		t.Fatal("overlapping connection from the same IP was rejected")
	}
	if second.Current != 1 || second.Peak != 1 {
		t.Fatalf("overlapping acquire = %#v", second)
	}

	releaseFirst()
	if got := tracker.Snapshot()["feed-1"].Current; got != 1 {
		t.Fatalf("current listeners after first release = %d, want 1", got)
	}
	releaseSecond()
	if got := tracker.Snapshot()["feed-1"].Current; got != 0 {
		t.Fatalf("current listeners after final release = %d, want 0", got)
	}
}

func TestListenerTrackerCountsDistinctIPsAndPreservesPeak(t *testing.T) {
	tracker := NewListenerTracker()
	releaseFirst, _, _ := tracker.TryAcquire("feed-1", "192.0.2.10")
	releaseSecond, stats, ok := tracker.TryAcquire("feed-1", "192.0.2.11")
	if !ok || stats.Current != 2 || stats.Peak != 2 || stats.PeakAt.IsZero() {
		t.Fatalf("second IP acquire = %#v, ok=%v", stats, ok)
	}

	releaseFirst()
	releaseSecond()
	stats = tracker.Snapshot()["feed-1"]
	if stats.Current != 0 || stats.Peak != 2 || stats.PeakAt.IsZero() {
		t.Fatalf("released stats = %#v", stats)
	}
}

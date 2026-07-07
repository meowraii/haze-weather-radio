//go:build !linux || !cgo

package tts

// ReleaseNativeMemory is a no-op on platforms without glibc malloc trimming.
func ReleaseNativeMemory() {}

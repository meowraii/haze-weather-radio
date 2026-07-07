//go:build linux && cgo

package tts

/*
#include <malloc.h>
*/
import "C"

// ReleaseNativeMemory asks glibc to return freed native model pages to the OS.
func ReleaseNativeMemory() {
	C.malloc_trim(0)
}

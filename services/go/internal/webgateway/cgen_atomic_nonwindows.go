//go:build !windows

package webgateway

import "os"

func replaceCgenFileAtomically(temporary string, destination string) error {
	return os.Rename(temporary, destination)
}

func syncCgenDirectory(directory string) error {
	handle, err := os.Open(directory)
	if err != nil {
		return err
	}
	defer handle.Close()
	return handle.Sync()
}

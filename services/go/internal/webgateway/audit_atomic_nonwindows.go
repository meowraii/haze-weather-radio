//go:build !windows

package webgateway

import (
	"os"
	"path/filepath"
)

func replaceAuditFileAtomically(temporary string, destination string) error {
	if err := os.Rename(temporary, destination); err != nil {
		return err
	}
	directory, err := os.Open(filepath.Dir(destination))
	if err != nil {
		return err
	}
	defer directory.Close()
	return directory.Sync()
}

//go:build !windows

package processguard

import (
	"errors"
	"os"
	"syscall"
)

func parentAlive(pid int) bool {
	process, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	err = process.Signal(syscall.Signal(0))
	return err == nil || errors.Is(err, syscall.EPERM)
}

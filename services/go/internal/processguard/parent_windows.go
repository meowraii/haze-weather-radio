//go:build windows

package processguard

import "syscall"

const (
	parentSynchronize = 0x00100000
	parentWaitTimeout = 0x00000102
)

func parentAlive(pid int) bool {
	handle, err := syscall.OpenProcess(parentSynchronize, false, uint32(pid))
	if err != nil {
		return false
	}
	defer syscall.CloseHandle(handle)
	result, err := syscall.WaitForSingleObject(handle, 0)
	if err != nil {
		return false
	}
	return result == parentWaitTimeout
}

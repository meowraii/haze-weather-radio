//go:build windows

package webgateway

import (
	"errors"
	"syscall"
	"unsafe"
)

const moveFileReplaceExisting = 0x1
const moveFileWriteThrough = 0x8

var moveFileExW = syscall.NewLazyDLL("kernel32.dll").NewProc("MoveFileExW")

func replaceCgenFileAtomically(temporary string, destination string) error {
	from, err := syscall.UTF16PtrFromString(temporary)
	if err != nil {
		return err
	}
	to, err := syscall.UTF16PtrFromString(destination)
	if err != nil {
		return err
	}
	result, _, callErr := moveFileExW.Call(
		uintptr(unsafe.Pointer(from)),
		uintptr(unsafe.Pointer(to)),
		moveFileReplaceExisting|moveFileWriteThrough,
	)
	if result != 0 {
		return nil
	}
	if callErr != syscall.Errno(0) {
		return callErr
	}
	return errors.New("atomic cgen configuration replacement failed")
}

func syncCgenDirectory(string) error {
	return nil
}

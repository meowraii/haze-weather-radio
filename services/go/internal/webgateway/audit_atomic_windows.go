//go:build windows

package webgateway

import (
	"errors"
	"syscall"
	"unsafe"
)

const auditMoveFileReplaceExisting = 0x1
const auditMoveFileWriteThrough = 0x8

var auditMoveFileExW = syscall.NewLazyDLL("kernel32.dll").NewProc("MoveFileExW")

func replaceAuditFileAtomically(temporary string, destination string) error {
	from, err := syscall.UTF16PtrFromString(temporary)
	if err != nil {
		return err
	}
	to, err := syscall.UTF16PtrFromString(destination)
	if err != nil {
		return err
	}
	result, _, callErr := auditMoveFileExW.Call(
		uintptr(unsafe.Pointer(from)),
		uintptr(unsafe.Pointer(to)),
		auditMoveFileReplaceExisting|auditMoveFileWriteThrough,
	)
	if result != 0 {
		return nil
	}
	if callErr != syscall.Errno(0) {
		return callErr
	}
	return errors.New("atomic audit checkpoint replacement failed")
}

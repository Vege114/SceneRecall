"""User-restricted Windows events for cooperative CLI shutdown.

No process is killed and no token/PID file is trusted. A named kernel event is
bound to the already verified process identity and exists only while it runs.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import hashlib
from pathlib import Path
import sys
import sysconfig
import threading

import psutil


def _event_name(process: psutil.Process, data_dir: Path, port: int) -> str:
    identity = "\0".join((
        str(Path(sys.executable).resolve()).casefold(),
        str((Path(sysconfig.get_path("scripts")) / "scenerecall.exe").resolve()).casefold(),
        str(data_dir.expanduser().resolve()).casefold(), str(port),
        str(process.pid), process.create_time().hex(),
    ))
    # Global events allow another terminal/session of this same Windows user to
    # stop the service. The explicit DACL below restricts access across users.
    return "Global\\SceneRecall.Stop." + hashlib.sha256(identity.encode("utf-8")).hexdigest()


class _Events:
    """Small, typed Win32 boundary; imported safely on other platforms too."""

    def __init__(self):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.security = ctypes.WinDLL("advapi32", use_last_error=True)
        signatures = (
            (self.kernel, "GetCurrentProcess", [], wintypes.HANDLE),
            (self.kernel, "CreateEventW", [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
            (self.kernel, "OpenEventW", [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR], wintypes.HANDLE),
            (self.kernel, "SetEvent", [wintypes.HANDLE], wintypes.BOOL),
            (self.kernel, "WaitForSingleObject", [wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
            (self.kernel, "CloseHandle", [wintypes.HANDLE], wintypes.BOOL),
            (self.kernel, "LocalFree", [ctypes.c_void_p], ctypes.c_void_p),
            (self.security, "OpenProcessToken", [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)], wintypes.BOOL),
            (self.security, "GetTokenInformation", [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                   wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)], wintypes.BOOL),
            (self.security, "ConvertSidToStringSidW", [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)], wintypes.BOOL),
            (self.security, "ConvertStringSecurityDescriptorToSecurityDescriptorW",
             [wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p], wintypes.BOOL),
        )
        for library, name, arguments, result in signatures:
            function = getattr(library, name)
            function.argtypes, function.restype = arguments, result

    def _user_sid(self) -> str:
        token = wintypes.HANDLE()
        if not self.security.OpenProcessToken(self.kernel.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            size = wintypes.DWORD()
            self.security.GetTokenInformation(token, 1, None, 0, ctypes.byref(size))
            buffer = ctypes.create_string_buffer(size.value)
            if not self.security.GetTokenInformation(token, 1, buffer, size, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            # TOKEN_USER starts with SID_AND_ATTRIBUTES, whose first member is
            # a SID pointer on both 32-bit and 64-bit Windows.
            sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
            text = wintypes.LPWSTR()
            if not self.security.ConvertSidToStringSidW(sid, ctypes.byref(text)):
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                return text.value
            finally:
                self.kernel.LocalFree(text)
        finally:
            self.close(token)

    def create(self, name: str):
        class SecurityAttributes(ctypes.Structure):
            _fields_ = [("nLength", wintypes.DWORD), ("lpSecurityDescriptor", ctypes.c_void_p),
                        ("bInheritHandle", wintypes.BOOL)]

        descriptor = ctypes.c_void_p()
        # Protected DACL: only the exact current user SID and SYSTEM. In
        # particular, do not inherit a permissive parent/default object ACL.
        dacl = f"D:P(A;;GA;;;SY)(A;;GA;;;{self._user_sid()})"
        if not self.security.ConvertStringSecurityDescriptorToSecurityDescriptorW(dacl, 1,
                                                                                  ctypes.byref(descriptor), None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            attributes = SecurityAttributes(ctypes.sizeof(SecurityAttributes), descriptor, False)
            ctypes.set_last_error(0)
            handle = self.kernel.CreateEventW(ctypes.byref(attributes), True, False, name)
            error = ctypes.get_last_error()
            if not handle:
                raise ctypes.WinError(error)
            if error == 183:  # ERROR_ALREADY_EXISTS: never adopt a precreated object.
                self.close(handle)
                raise FileExistsError("SceneRecall 的安全停止通道已存在，拒绝复用。")
            return handle
        finally:
            self.kernel.LocalFree(descriptor)

    def open(self, name: str):
        handle = self.kernel.OpenEventW(0x0002, False, name)  # EVENT_MODIFY_STATE only.
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        return handle

    def signal(self, handle):
        if not self.kernel.SetEvent(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    def wait(self, handle) -> bool:
        status = self.kernel.WaitForSingleObject(handle, 100)
        if status not in (0, 258):  # WAIT_OBJECT_0 / WAIT_TIMEOUT.
            raise ctypes.WinError(ctypes.get_last_error())
        return status == 0

    def close(self, handle):
        self.kernel.CloseHandle(handle)


@contextmanager
def shutdown_control(server, data_dir: Path, port: int):
    events = _Events()
    handle = events.create(_event_name(psutil.Process(), data_dir, port))
    closed = threading.Event()

    def listen():
        while not closed.is_set():
            if events.wait(handle):
                # Uvicorn then runs the ordinary application lifespan shutdown,
                # including JobQueue.stop() and its task checkpoint handling.
                server.should_exit = True
                return

    listener = threading.Thread(target=listen, name="scenerecall-stop", daemon=True)
    listener.start()
    try:
        yield
    finally:
        closed.set()
        listener.join()
        events.close(handle)


def request_shutdown(process: psutil.Process, data_dir: Path, port: int, dry_run: bool = False):
    events = _Events()
    handle = events.open(_event_name(process, data_dir, port))
    try:
        if not dry_run:
            events.signal(handle)
    finally:
        events.close(handle)

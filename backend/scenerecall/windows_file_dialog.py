"""Modern Explorer file chooser hosted by Python's GUI process.

Only the standard library is needed. The parent sends options over stdin and
receives one JSON result over stdout; filenames never become executable code.
"""
from __future__ import annotations

import ctypes
import json
import sys
import uuid
from ctypes import wintypes


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    @classmethod
    def parse(cls, value):
        return cls.from_buffer_copy(uuid.UUID(value).bytes_le)


class _Filter(ctypes.Structure):
    _fields_ = [("name", wintypes.LPCWSTR), ("pattern", wintypes.LPCWSTR)]


def _check(result):
    if result < 0:
        raise OSError(f"Windows file chooser failed (HRESULT 0x{result & 0xffffffff:08x})")


def _method(pointer, index, result_type, *argument_types):
    table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(result_type, ctypes.c_void_p, *argument_types)(table[index])


def _release(pointer):
    if pointer:
        _method(pointer, 2, wintypes.ULONG)(pointer)


def choose_files(options):
    """Show IFileOpenDialog on this STA thread and free every COM allocation."""
    if sys.platform != "win32":
        raise OSError("The Explorer chooser requires Windows")
    ole = ctypes.WinDLL("ole32")
    ole.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    ole.CoInitializeEx.restype = ctypes.c_long
    ole.CoCreateInstance.argtypes = [ctypes.POINTER(_GUID), ctypes.c_void_p, wintypes.DWORD,
                                   ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    ole.CoCreateInstance.restype = ctypes.c_long
    ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole.CoUninitialize.argtypes = []
    shell = ctypes.WinDLL("shell32")
    shell.SHCreateItemFromParsingName.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p,
                                                ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    shell.SHCreateItemFromParsingName.restype = ctypes.c_long
    _check(ole.CoInitializeEx(None, 0x2 | 0x4))  # STA, no legacy OLE1 DDE.
    dialog = ctypes.c_void_p()
    items = ctypes.c_void_p()
    try:
        clsid = _GUID.parse("dc1c5a9c-e88a-4dde-a5a1-60f82a20aef7")
        iid = _GUID.parse("d57c7288-d4ad-4768-be02-9d969532d960")
        _check(ole.CoCreateInstance(ctypes.byref(clsid), None, 1, ctypes.byref(iid), ctypes.byref(dialog)))
        flags = wintypes.DWORD()
        _check(_method(dialog, 10, ctypes.c_long, ctypes.POINTER(wintypes.DWORD))(dialog, ctypes.byref(flags)))
        # Filesystem paths only; existing files, no process-wide CWD change.
        flags.value |= 0x40 | 0x800 | 0x1000 | 0x8
        if options["multiple"]:
            flags.value |= 0x200
        _check(_method(dialog, 9, ctypes.c_long, wintypes.DWORD)(dialog, flags))
        filters = (_Filter * 1)(_Filter("SceneRecall", ";".join("*." + ext for ext in options["extensions"])))
        _check(_method(dialog, 4, ctypes.c_long, wintypes.UINT, ctypes.POINTER(_Filter))(dialog, 1, filters))
        _check(_method(dialog, 17, ctypes.c_long, wintypes.LPCWSTR)(dialog, options["title"]))
        if options.get("directory"):
            folder = ctypes.c_void_p()
            try:
                shell_iid = _GUID.parse("43826d1e-e718-42ee-bc55-a1e261c37bfe")
                _check(shell.SHCreateItemFromParsingName(options["directory"], None, ctypes.byref(shell_iid),
                                                       ctypes.byref(folder)))
                _check(_method(dialog, 12, ctypes.c_long, ctypes.c_void_p)(dialog, folder))
            finally:
                _release(folder)
        # The server has no desktop window. Do not make an unrelated foreground
        # application (browser, editor, or terminal) the modal owner.
        result = _method(dialog, 3, ctypes.c_long, wintypes.HWND)(dialog, None)
        if result & 0xffffffff == 0x800704c7:  # HRESULT_FROM_WIN32(ERROR_CANCELLED).
            return {"paths": [], "cancelled": True}
        _check(result)
        _check(_method(dialog, 27, ctypes.c_long, ctypes.POINTER(ctypes.c_void_p))(dialog, ctypes.byref(items)))
        count = wintypes.DWORD()
        _check(_method(items, 7, ctypes.c_long, ctypes.POINTER(wintypes.DWORD))(items, ctypes.byref(count)))
        paths = []
        for index in range(count.value):
            item = ctypes.c_void_p()
            name = ctypes.c_void_p()
            try:
                _check(_method(items, 8, ctypes.c_long, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p))(
                    items, index, ctypes.byref(item)))
                _check(_method(item, 5, ctypes.c_long, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p))(
                    item, 0x80058000, ctypes.byref(name)))  # SIGDN_FILESYSPATH.
                paths.append(ctypes.wstring_at(name))
            finally:
                if name:
                    ole.CoTaskMemFree(name)
                _release(item)
        return {"paths": paths, "cancelled": False}
    finally:
        _release(items)
        _release(dialog)
        ole.CoUninitialize()


def main():
    # pythonw still receives real standard handles when its parent supplies pipes.
    # Set their codec explicitly because Windows' locale may otherwise use GBK.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    result = choose_files(json.load(sys.stdin))
    json.dump(result, sys.stdout, ensure_ascii=False)
    sys.stdout.flush()


if __name__ == "__main__":
    main()

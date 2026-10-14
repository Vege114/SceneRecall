"""Native dialog tests substitute the OS chooser; no test opens a real window."""
import base64
import json
import subprocess
import zipfile

import pytest
from fastapi.testclient import TestClient

from scenerecall import native_picker
from scenerecall.library import Library
from scenerecall.main import create_app
from scenerecall.native_picker import FilePickerBusy, FilePickerInput, FilePickerUnavailable, pick_local_files


def test_mac_multiselection_preserves_unicode_quotes_and_newlines(tmp_path, monkeypatch):
    paths = [tmp_path / '字幕 "第一集".ASS', tmp_path / "字幕\n第二集.vtt"]
    for path in paths:
        path.write_text("synthetic subtitle", encoding="utf-8")
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs == {"input": None, "capture_output": True, "text": True, "encoding": "utf-8", "check": False}
        return subprocess.CompletedProcess(command, 0, json.dumps({"paths": [str(path) for path in paths],
                                                                 "cancelled": False}, ensure_ascii=False), "")

    monkeypatch.setattr(native_picker.sys, "platform", "darwin")
    monkeypatch.setattr(native_picker.subprocess, "run", run)
    result = pick_local_files(FilePickerInput(kind="subtitle", multiple=True, directory=str(tmp_path)))
    assert result == {"paths": [str(path) for path in paths], "cancelled": False}
    assert commands[0][:4] == ["/usr/bin/osascript", "-l", "JavaScript", "-e"]
    assert commands[0][4] == native_picker._MAC_SCRIPT
    options = json.loads(commands[0][5])
    assert options["extensions"] == ["ass", "srt", "vtt"]
    assert options["multiple"] is True and options["directory"] == str(tmp_path)


def test_cancellation_and_duplicate_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(native_picker, "_open_dialog", lambda options: {"paths": [], "cancelled": True})
    assert pick_local_files(FilePickerInput()) == {"paths": [], "cancelled": True}
    video = tmp_path / "first.mp4"
    video.write_bytes(b"not probed by picker")
    monkeypatch.setattr(native_picker, "_open_dialog", lambda options: {"paths": [str(video), str(video)], "cancelled": False})
    assert pick_local_files(FilePickerInput(multiple=True))["paths"] == [str(video)]
    with pytest.raises(FilePickerUnavailable, match="数量"):
        pick_local_files(FilePickerInput(multiple=False))


@pytest.mark.parametrize("directory", ["relative", "https://example.invalid", "bad\x00path"])
def test_invalid_starting_directory_never_opens_dialog(directory, monkeypatch):
    monkeypatch.setattr(native_picker, "_open_dialog", lambda options: pytest.fail("invalid input opened dialog"))
    with pytest.raises(ValueError):
        pick_local_files(FilePickerInput(directory=directory))


def test_missing_starting_directory_opens_at_nearest_existing_parent(tmp_path, monkeypatch):
    opened = []

    def choose(options):
        opened.append(options["directory"])
        return {"paths": [], "cancelled": True}

    monkeypatch.setattr(native_picker, "_open_dialog", choose)
    assert pick_local_files(FilePickerInput(directory=str(tmp_path / "moved" / "episodes"))) == {
        "paths": [], "cancelled": True}
    assert opened == [str(tmp_path)]


def test_existing_nondirectory_starting_position_never_opens_dialog(tmp_path, monkeypatch):
    video = tmp_path / "episode.mp4"
    video.write_bytes(b"synthetic")
    monkeypatch.setattr(native_picker, "_open_dialog", lambda options: pytest.fail("file is not a starting directory"))
    for directory in (video, video / "child"):
        with pytest.raises(ValueError, match="必须是目录"):
            pick_local_files(FilePickerInput(directory=str(directory)))


def test_selected_wrong_kind_rejected_and_picker_lock_released(tmp_path, monkeypatch):
    video = tmp_path / "first.mp4"
    video.write_bytes(b"not a subtitle")
    monkeypatch.setattr(native_picker, "_open_dialog", lambda options: {"paths": [str(video)], "cancelled": False})
    with pytest.raises(ValueError, match="字幕"):
        pick_local_files(FilePickerInput(kind="subtitle"))
    assert native_picker._dialog_lock.acquire(blocking=False)
    native_picker._dialog_lock.release()


def test_open_picker_rejects_concurrent_window(monkeypatch):
    monkeypatch.setattr(native_picker, "_open_dialog", lambda options: pytest.fail("second window opened"))
    native_picker._dialog_lock.acquire()
    try:
        with pytest.raises(FilePickerBusy, match="已打开"):
            pick_local_files(FilePickerInput())
    finally:
        native_picker._dialog_lock.release()


@pytest.mark.parametrize("output", ["not json", '{"paths":[1],"cancelled":false}',
                                     '{"paths":["a"],"cancelled":true}', '{"paths":[],"cancelled":"yes"}'])
def test_malformed_native_result_is_actionable(output):
    with pytest.raises(FilePickerUnavailable, match="无法读取"):
        native_picker._json_result(subprocess.CompletedProcess([], 0, output, ""))


def test_finder_automation_permission_error_is_actionable(monkeypatch):
    monkeypatch.setattr(native_picker.sys, "platform", "darwin")
    monkeypatch.setattr(native_picker, "_run", lambda command, **kwargs: subprocess.CompletedProcess(
        command, 0, '{"paths":[],"cancelled":false,"error":"finder_permission"}', ""))
    with pytest.raises(FilePickerUnavailable, match="自动化.*Finder"):
        pick_local_files(FilePickerInput())
    assert native_picker._dialog_lock.acquire(blocking=False)
    native_picker._dialog_lock.release()


@pytest.mark.skipif(native_picker.sys.platform != "darwin", reason="JXA runtime is macOS-only")
@pytest.mark.parametrize("multiple,error", [(False, None), (True, None), (True, -128), (True, -1743)])
def test_mac_script_serialization_and_cancellation_without_real_dialog(tmp_path, multiple, error):
    # Replace only the Finder proxy; exercise Apple's real JXA Path conversion
    # and the production run handler without launching or controlling an app.
    fake_finder = r'''const finder = {
        activate: function() {},
        chooseFile: function(selection, modifiers) {
            if (selection.withPrompt !== options.title ||
                selection.defaultLocation.toString() !== options.directory ||
                selection.multipleSelectionsAllowed !== options.multiple ||
                JSON.stringify(selection.ofType) !== JSON.stringify(options.extensions) ||
                modifiers.timeout !== 86400) throw new Error('invalid chooser options');
            if (options.fakeError !== null) throw {errorNumber: options.fakeError};
            const files = options.fakePaths.map(function(path) { return Path(path); });
            return options.multiple ? files : files[0];
        }
    };'''
    script = native_picker._MAC_SCRIPT.replace("const finder = Application('com.apple.finder');", fake_finder)
    paths = [str(tmp_path / '字幕 "第一集".ass'), str(tmp_path / "字幕\n第二集.vtt")]
    options = {"title": "SceneRecall test", "extensions": ["ass", "srt", "vtt"], "directory": str(tmp_path),
               "multiple": multiple, "fakePaths": paths, "fakeError": error}
    result = subprocess.run(["/usr/bin/osascript", "-l", "JavaScript", "-e", script,
                             json.dumps(options, ensure_ascii=False)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    if error == -1743:
        assert json.loads(result.stdout) == {"paths": [], "cancelled": False, "error": "finder_permission"}
    elif error == -128:
        assert json.loads(result.stdout) == {"paths": [], "cancelled": True}
    else:
        assert json.loads(result.stdout) == {"paths": paths if multiple else paths[:1], "cancelled": False}


def test_system_picker_failure_and_linux_missing_desktop(monkeypatch):
    monkeypatch.setattr(native_picker.sys, "platform", "darwin")
    monkeypatch.setattr(native_picker, "_run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", "OS error"))
    with pytest.raises(FilePickerUnavailable, match="桌面环境"):
        pick_local_files(FilePickerInput())
    monkeypatch.setattr(native_picker.sys, "platform", "linux")
    monkeypatch.setattr(native_picker.shutil, "which", lambda command: None)
    with pytest.raises(FilePickerUnavailable, match="Zenity"):
        pick_local_files(FilePickerInput())


def test_windows_system_dialog_uses_fixed_script_and_separate_json(tmp_path, monkeypatch):
    backup = tmp_path / '备份 "$()".zip'
    backup.write_bytes(b"selection does not inspect archive")
    monkeypatch.setattr(native_picker.sys, "platform", "win32")
    monkeypatch.setattr(native_picker.shutil, "which", lambda command: "powershell.exe")

    def run(command, **kwargs):
        assert command[:4] == ["powershell.exe", "-NoProfile", "-STA", "-EncodedCommand"]
        assert base64.b64decode(command[4]).decode("utf-16-le") == native_picker._WINDOWS_SCRIPT
        options = json.loads(kwargs["input"])
        assert options["extensions"] == ["zip"] and not options["multiple"]
        return subprocess.CompletedProcess(command, 0, json.dumps({"paths": [str(backup)], "cancelled": False}), "")

    monkeypatch.setattr(native_picker, "_run", run)
    assert pick_local_files(FilePickerInput(kind="backup"))["paths"] == [str(backup)]


def test_linux_system_dialog_multiselection_and_cancel(tmp_path, monkeypatch):
    paths = [tmp_path / "one.srt", tmp_path / "two.ass"]
    for path in paths:
        path.write_bytes(b"selection only")
    monkeypatch.setattr(native_picker.sys, "platform", "linux")
    monkeypatch.setattr(native_picker.shutil, "which", lambda command: "/usr/bin/zenity")

    def run(command, **kwargs):
        assert "--multiple" in command and f"--filename={tmp_path}/" in command
        return subprocess.CompletedProcess(command, 0, "\x1f".join(str(path) for path in paths) + "\n", "")

    monkeypatch.setattr(native_picker, "_run", run)
    assert pick_local_files(FilePickerInput(kind="subtitle", multiple=True, directory=str(tmp_path)))["paths"] == [str(path) for path in paths]
    monkeypatch.setattr(native_picker, "_run", lambda command, **kwargs: subprocess.CompletedProcess(command, 1, "", ""))
    assert pick_local_files(FilePickerInput()) == {"paths": [], "cancelled": True}


def test_native_api_requires_local_authenticated_session(tmp_path, monkeypatch):
    calls = []

    def choose(payload):
        calls.append(payload)
        return {"paths": [], "cancelled": True}

    monkeypatch.setattr("scenerecall.main.pick_local_files", choose)
    app = create_app(tmp_path / "library", start_worker=False, token="native-test")
    with TestClient(app) as client:
        assert client.post("/api/local-files/pick", json={}).status_code == 401
        assert client.post("/api/backups/restore-local", json={"path": "/private/tmp/missing.zip"}).status_code == 401
        assert not calls
        client.post("/api/session", json={"token": "native-test"})
        assert client.post("/api/local-files/pick", json={}, headers={"origin": "https://evil.invalid"}).status_code == 403
        assert client.post("/api/local-files/pick", json={"kind": "other"}).status_code == 422
        assert client.post("/api/local-files/pick", json={"kind": "backup", "multiple": True}).status_code == 422
        assert client.post("/api/local-files/pick", json={"multiple": "yes"}).status_code == 422
        assert not calls
        result = client.post("/api/local-files/pick", json={"kind": "subtitle", "multiple": True})
        assert result.status_code == 200 and result.json() == {"paths": [], "cancelled": True}
        assert len(calls) == 1 and calls[0].kind == "subtitle" and calls[0].multiple


@pytest.mark.parametrize("error,status", [(FilePickerBusy("窗口已打开"), 409), (FilePickerUnavailable("无法打开"), 503)])
def test_native_api_dialog_errors(tmp_path, monkeypatch, error, status):
    def fail(payload):
        raise error
    monkeypatch.setattr("scenerecall.main.pick_local_files", fail)
    app = create_app(tmp_path / "library", start_worker=False, token="native-test")
    with TestClient(app) as client:
        result = client.post("/api/local-files/pick", json={}, headers={"authorization": "Bearer native-test"})
        assert result.status_code == status and result.json()["detail"] == str(error)


def test_restore_local_keeps_archive_and_reuses_backup_validation(tmp_path):
    source = Library(tmp_path / "source")
    project = source.organization.save_project({"name": "合成备份项目", "kind": "animation"})
    backup = source.backup()
    path = source.root / "runtime" / "backups" / f"{backup['id']}.zip"
    contents = path.read_bytes()
    app = create_app(tmp_path / "target", start_worker=False, token="native-test")
    with TestClient(app) as client:
        client.post("/api/session", json={"token": "native-test"})
        result = client.post("/api/backups/restore-local", json={"path": str(path)})
        assert result.status_code == 200, result.text
        assert path.read_bytes() == contents
        assert client.get("/api/projects").json()[0]["id"] == project["id"]
        assert client.post("/api/backups/restore-local", json={"path": "relative.zip"}).status_code == 400
        assert client.post("/api/backups/restore-local", json={"path": str(tmp_path / "missing.zip")}).status_code == 404
        bad = tmp_path / "bad.zip"
        bad.write_bytes(b"not a ZIP")
        assert client.post("/api/backups/restore-local", json={"path": str(bad)}).status_code == 400
        unsafe = tmp_path / "unsafe.zip"
        with zipfile.ZipFile(unsafe, "w") as archive:
            archive.writestr("../outside.json", "{}")
        result = client.post("/api/backups/restore-local", json={"path": str(unsafe)})
        assert result.status_code == 400 and "非法路径" in result.json()["detail"]
        assert not (tmp_path / "outside.json").exists()
        assert client.get("/api/projects").json()[0]["id"] == project["id"]

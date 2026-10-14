"""Native chooser tests; only the explicit macOS smoke opts into real windows."""
import base64
import json
import os
import subprocess
import time
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


def test_mac_dialog_runs_in_its_own_process_without_finder_automation():
    assert "ObjC.import('AppKit')" in native_picker._MAC_SCRIPT
    assert "$.NSOpenPanel.openPanel" in native_picker._MAC_SCRIPT
    assert "Application('com.apple.finder')" not in native_picker._MAC_SCRIPT
    assert "chooseFile" not in native_picker._MAC_SCRIPT
    assert "includeStandardAdditions" not in native_picker._MAC_SCRIPT


@pytest.mark.skipif(native_picker.sys.platform != "darwin", reason="JXA runtime is macOS-only")
@pytest.mark.parametrize("multiple,cancelled,directory", [(False, False, True), (True, False, True),
                                                          (True, True, True), (True, False, False)])
def test_mac_script_serialization_and_cancellation_without_real_dialog(tmp_path, multiple, cancelled, directory):
    # Replace only the application and panel. Exercise Apple's real JXA NSURL
    # conversion and the production run handler without showing a window.
    fake_application = r'''const application = {
        setActivationPolicy: function(policy) {
            if (policy !== $.NSApplicationActivationPolicyAccessory) throw new Error('invalid activation policy');
        },
        activateIgnoringOtherApps: function(activate) {
            if (!activate) throw new Error('panel would not come to foreground');
        }
    };'''
    fake_panel = r'''const panel = {
        get runModal() {
            if (this.title !== options.title ||
                (options.directory && ObjC.unwrap(this.directoryURL.path) !== options.directory) ||
                (!options.directory && this.directoryURL !== undefined) ||
                this.allowsMultipleSelection !== options.multiple ||
                JSON.stringify(this.allowedFileTypes) !== JSON.stringify(options.extensions) ||
                !this.canChooseFiles || this.canChooseDirectories || this.allowsOtherFileTypes ||
                this.treatsFilePackagesAsDirectories) throw new Error('invalid chooser options');
            return options.fakeCancelled ? $.NSModalResponseCancel : $.NSModalResponseOK;
        },
        get URLs() {
            if (options.fakeCancelled) throw new Error('cancelled panel must not read selections');
            const paths = options.multiple ? options.fakePaths : options.fakePaths.slice(0, 1);
            return $(paths.map(function(path) { return $.NSURL.fileURLWithPath(path); }));
        }
    };'''
    script = native_picker._MAC_SCRIPT.replace("const application = $.NSApplication.sharedApplication;", fake_application)
    script = script.replace("const panel = $.NSOpenPanel.openPanel;", fake_panel)
    paths = [str(tmp_path / '字幕 "第一集".ass'), str(tmp_path / "字幕\n第二集.vtt")]
    options = {"title": "SceneRecall test", "extensions": ["ass", "srt", "vtt"],
               "directory": str(tmp_path) if directory else None,
               "multiple": multiple, "fakePaths": paths, "fakeCancelled": cancelled}
    result = subprocess.run(["/usr/bin/osascript", "-l", "JavaScript", "-e", script,
                             json.dumps(options, ensure_ascii=False)], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stderr
    if cancelled:
        assert json.loads(result.stdout) == {"paths": [], "cancelled": True}
    else:
        assert json.loads(result.stdout) == {"paths": paths if multiple else paths[:1], "cancelled": False}


@pytest.mark.skipif(native_picker.sys.platform != "darwin" or os.getenv("SCENERECALL_NATIVE_PICKER_SMOKE") != "1",
                    reason="Opt-in macOS window smoke; ordinary tests never open real dialogs")
@pytest.mark.parametrize("kind", ["video", "subtitle", "backup"])
def test_mac_real_panel_is_visible_focused_and_cancellable(tmp_path, kind):
    # A native timer on the main modal run loop observes readiness and cancels
    # without choosing any files or needing access to another application.
    probe = r'''
    let attempts = 0;
    ObjC.registerSubclass({
        name: 'SceneRecallPickerSmokeProbe',
        superclass: 'NSObject',
        methods: {
            'inspectAndCancel:': {
                types: ['void', ['id']],
                implementation: function(timer) {
                    const state = {
                        visible: Boolean(panel.visible),
                        keyWindow: Boolean(panel.keyWindow),
                        applicationActive: Boolean(application.active),
                        openedSeconds: (Date.now() - startedAt) / 1000
                    };
                    attempts++;
                    if (!(state.visible && state.keyWindow && state.applicationActive) && attempts < 250) return;
                    console.log(JSON.stringify(state));
                    timer.invalidate;
                    panel.cancel(null);
                }
            }
        }
    });
    const target = $.SceneRecallPickerSmokeProbe.alloc.init;
    const timer = $.NSTimer.timerWithTimeIntervalTargetSelectorUserInfoRepeats(0.02, target, 'inspectAndCancel:', null, true);
    $.NSRunLoop.mainRunLoop.addTimerForMode(timer, $.NSModalPanelRunLoopMode);
    $.NSRunLoop.mainRunLoop.addTimerForMode(timer, $.NSRunLoopCommonModes);
'''
    script = "const startedAt = Date.now();\n" + native_picker._MAC_SCRIPT
    script = script.replace("application.activateIgnoringOtherApps(true);", "application.activateIgnoringOtherApps(true);" + probe)
    options = {"title": "SceneRecall · 文件选择自动测试", "extensions": sorted(ext[1:] for ext in native_picker.EXTENSIONS[kind]),
               "multiple": kind != "backup", "directory": str(tmp_path)}
    started = time.perf_counter()
    result = subprocess.run(["/usr/bin/osascript", "-l", "JavaScript", "-e", script,
                             json.dumps(options, ensure_ascii=False)], capture_output=True, text=True,
                            encoding="utf-8", timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"paths": [], "cancelled": True}
    states = [json.loads(line) for line in result.stderr.splitlines() if line.startswith('{"visible":')]
    assert len(states) == 1, result.stderr
    state = states[0]
    assert state["visible"] and state["keyWindow"] and state["applicationActive"]
    assert state["openedSeconds"] < 5
    print(f"{kind}: panel ready {state['openedSeconds']:.3f}s; cancelled in {time.perf_counter() - started:.3f}s")


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

import base64
import json
import subprocess
from types import SimpleNamespace

import pytest

from scenerecall import codex_cli, codex_windows


def binary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    path.chmod(0o700)
    return path.resolve()


def test_windows_discovers_desktop_cli_outside_stale_path(tmp_path, monkeypatch):
    monkeypatch.delenv("SCENERECALL_CODEX_PATH", raising=False)
    monkeypatch.setattr(codex_cli, "_runtime", lambda: "windows")
    monkeypatch.setattr(codex_cli.shutil, "which", lambda _: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "user"))
    expected = binary(tmp_path / "user/AppData/Local/Programs/OpenAI/Codex/bin/codex.exe")
    assert codex_cli._executable() == str(expected)


@pytest.mark.parametrize("layout", [
    "vendor/x86_64-pc-windows-msvc/codex/codex.exe",
    "node_modules/@openai/codex-win32-x64/vendor/x86_64-pc-windows-msvc/codex/codex.exe",
])
def test_windows_npm_cmd_resolves_native_binary_without_executing_shim(tmp_path, monkeypatch, layout):
    shim = tmp_path / "codex.cmd"
    shim.write_text("must not be executed or parsed")
    expected = binary(tmp_path / "node_modules/@openai/codex" / layout)
    monkeypatch.setattr(codex_cli, "_runtime", lambda: "windows")
    monkeypatch.setenv("SCENERECALL_CODEX_PATH", str(shim))
    assert codex_cli._executable() == str(expected)


def test_unknown_windows_shim_is_not_sent_to_a_shell(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_cli, "_runtime", lambda: "windows")
    shim = binary(tmp_path / "codex.cmd")
    monkeypatch.setenv("SCENERECALL_CODEX_PATH", str(shim))
    assert codex_cli._executable() is None


def test_wsl_prefers_windows_desktop_cli_and_supports_windows_override(tmp_path, monkeypatch):
    expected = binary(tmp_path / "desktop/codex.exe")
    monkeypatch.delenv("SCENERECALL_CODEX_PATH", raising=False)
    monkeypatch.setattr(codex_cli, "_runtime", lambda: "wsl")
    monkeypatch.setattr(codex_windows, "discover", lambda _: {"paths": [r"C:\desktop\codex.exe"]})
    monkeypatch.setattr(codex_windows, "linux_path", lambda _: str(expected))
    assert codex_cli._executable() == str(expected)
    monkeypatch.setenv("SCENERECALL_CODEX_PATH", r"C:\desktop\codex.exe")
    assert codex_cli._executable() == str(expected)


def test_wsl_windows_environment_uses_windows_home_and_keeps_secrets_out(monkeypatch):
    monkeypatch.setattr(codex_cli, "_runtime", lambda: "wsl")
    monkeypatch.setattr(codex_windows, "discover", lambda _: {
        "home": r"C:\Users\test\.codex", "profile": r"C:\Users\test", "paths": [],
    })
    monkeypatch.setenv("HOME", "/home/linux-user")
    monkeypatch.setenv("CODEX_HOME", "/home/linux-user/.codex")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("CODEX_THREAD_ID", "private-thread")
    environment = codex_cli._environment("/mnt/c/codex.exe")
    assert environment["CODEX_HOME"] == r"C:\Users\test\.codex"
    assert environment["USERPROFILE"] == r"C:\Users\test"
    assert not any(key in environment for key in ("HOME", "OPENAI_API_KEY", "CODEX_THREAD_ID"))
    assert codex_cli._environment("/usr/bin/codex")["CODEX_HOME"] == "/home/linux-user/.codex"
    monkeypatch.setenv("CODEX_HOME", r"D:\custom codex\账号")
    assert codex_cli._environment("/mnt/c/codex.exe")["CODEX_HOME"] == r"D:\custom codex\账号"


def test_wsl_model_catalog_uses_the_same_windows_home_as_the_cli(tmp_path, monkeypatch):
    windows_home = tmp_path / "windows-codex"
    windows_home.mkdir()
    (windows_home / "models_cache.json").write_text(json.dumps({"models": [
        {"slug": "windows-model", "visibility": "list", "supported_reasoning_levels": [{"effort": "high"}]},
    ]}), encoding="utf-8")
    monkeypatch.setattr(codex_cli, "_runtime", lambda: "wsl")
    monkeypatch.setattr(codex_cli, "_executable", lambda: "/mnt/c/codex.exe")
    monkeypatch.setenv("CODEX_HOME", "/home/linux-user/.codex")
    monkeypatch.setattr(codex_windows, "discover", lambda _: {
        "home": r"C:\Users\test\.codex", "profile": r"C:\Users\test", "paths": [],
    })
    converted = []

    def convert(home):
        converted.append(home)
        return str(windows_home)

    monkeypatch.setattr(codex_windows, "linux_path", convert)
    assert codex_cli._cached_model_catalog()["models"][0]["id"] == "windows-model"
    assert converted == [r"C:\Users\test\.codex"]
    monkeypatch.setenv("CODEX_HOME", r"D:\custom codex\账号")
    assert codex_cli._cached_model_catalog()["models"][0]["id"] == "windows-model"
    assert converted[-1] == r"D:\custom codex\账号"


def test_wsl_converts_all_file_arguments_without_reinterpreting_config_or_prompt(tmp_path, monkeypatch):
    paths = []

    def convert(path):
        paths.append(str(path))
        return "\\\\wsl.localhost\\Ubuntu" + str(path).replace("/", "\\")

    monkeypatch.setattr(codex_windows, "windows_path", convert)
    instruction = str(tmp_path / "instructions.txt")
    image = str(tmp_path / "image-01.png")
    schema = str(tmp_path / "schema.json")
    args = ["/mnt/c/codex.exe", "exec", "--ignore-user-config", "--image", image,
            "--output-schema", schema, "-c", "model_instructions_file=" + json.dumps(instruction),
            "-c", 'model_reasoning_effort="high"', "-m", "model-with-dash", "-"]
    result = codex_windows.command(args, tmp_path)
    assert result[:3] == [args[0], "exec", "--cd"]
    assert result[-1] == "-" and 'model_reasoning_effort="high"' in result
    assert result[result.index("-m") + 1] == "model-with-dash"
    assert paths == [image, schema, instruction, str(tmp_path)]
    encoded_instruction = next(value for value in result if value.startswith("model_instructions_file="))
    assert json.loads(encoded_instruction.split("=", 1)[1]).startswith("\\\\wsl.localhost\\Ubuntu")
    assert args[2] == "--ignore-user-config"  # Caller-owned arguments remain reusable.
    assert codex_windows.command([args[0], "exec", "--help"], tmp_path) == [args[0], "exec", "--help"]


def test_windows_path_preserves_drive_and_unc_paths(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected subprocess"))
    assert codex_windows.windows_path(r"C:\Temp\test image.png") == r"C:\Temp\test image.png"
    assert codex_windows.windows_path(r"\\wsl.localhost\Ubuntu\tmp\image.png") == r"\\wsl.localhost\Ubuntu\tmp\image.png"


def test_launcher_keeps_unicode_arguments_and_prompt_as_data(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_windows, "powershell", lambda: "/mnt/c/powershell.exe")
    monkeypatch.setattr(codex_windows, "windows_path", lambda path: "C:\\映射 目录\\" + str(path).rsplit("/", 1)[-1])
    prompt = '用户输入 $(bad); & %SECRET% "原样保留"'.encode("utf-8")
    args = ["/mnt/c/codex.exe", "exec", "-c", 'model_reasoning_effort="high"', "--image", "/tmp/中文 image.png", "-"]
    command, stdin = codex_windows.launch(args, tmp_path, prompt, {
        "CODEX_HOME": r"D:\custom codex\账号", "PATH": "/usr/bin", "OPENAI_API_KEY": "private-secret",
        "no_proxy": "localhost", "NO_PROXY": "localhost",
    })
    assert command[:2] == ["/mnt/c/powershell.exe", "-NoLogo"]
    script = base64.b64decode(command[-1]).decode("utf-16-le")
    assert "SceneRecallJob" in script and "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE" in script
    assert "private-secret" not in script and "用户输入" not in script
    payload = json.loads(stdin)
    assert base64.b64decode(payload["input"]) == prompt
    assert payload["environment"] == {"CODEX_HOME": r"D:\custom codex\账号", "NO_PROXY": "localhost"}
    assert payload["arguments"] == subprocess.list2cmdline(codex_windows.command(args, tmp_path)[1:])
    assert "中文 image.png" in payload["arguments"]


def test_discovery_is_read_only_constant_script_and_drops_linux_home(monkeypatch):
    monkeypatch.setattr(codex_windows, "_cached", None)
    monkeypatch.setattr(codex_windows.shutil, "which", lambda _: "/mnt/c/powershell.exe")
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "paths": [r"C:\Codex\codex.exe"], "home": r"C:\Users\test\.codex", "profile": r"C:\Users\test",
        }).encode())

    monkeypatch.setattr(subprocess, "run", run)
    environment = {"HOME": "/home/test", "CODEX_HOME": "/linux/.codex", "PATH": "/bin"}
    assert codex_windows.discover(environment)["paths"] == [r"C:\Codex\codex.exe"]
    assert codex_windows.discover(environment)["paths"]  # The same metadata is briefly cached.
    assert len(calls) == 1
    args, options = calls[0]
    script = base64.b64decode(args[-1]).decode("utf-16-le")
    assert "auth.json" not in script and "Get-Content" not in script
    assert "HOME" not in options["env"] and "CODEX_HOME" not in options["env"]
    assert options["timeout"] == 8 and "shell" not in options


@pytest.mark.parametrize("stdout", [b"secret-invalid", b"[]", b'{"paths":"bad"}'])
def test_discovery_ignores_unrecognized_or_private_output(monkeypatch, stdout):
    monkeypatch.setattr(codex_windows, "_cached", None)
    monkeypatch.setattr(codex_windows.shutil, "which", lambda _: "/mnt/c/powershell.exe")
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=stdout))
    result = codex_windows.discover({})
    assert result == {"paths": [], "home": None, "profile": None}


@pytest.mark.asyncio
async def test_status_exposes_backend_and_cli_runtime_without_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_cli, "_runtime", lambda: "wsl")
    monkeypatch.setattr(codex_cli, "_executable", lambda: "/mnt/c/codex.exe")

    async def inspect(*args):
        return {"installed": True, "authenticated": True, "compatible": True, "message": "ready"}

    monkeypatch.setattr(codex_cli, "_inspect", inspect)
    result = await codex_cli.status()
    assert result["runtime"] == "wsl" and result["cli_runtime"] == "windows"
    assert result["authenticated"] and result["executable"] == "/mnt/c/codex.exe"

import asyncio
import base64
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import psutil

from scenerecall import codex_cli
from scenerecall.codex_cli import CodexCLIError


def test_environment_preserves_windows_login_and_certificate_locations(monkeypatch):
    locations = {"USERPROFILE": r"C:\Users\影像 用户", "HOMEDRIVE": "C:", "HOMEPATH": r"\Users\影像 用户",
                 "APPDATA": r"C:\Users\影像 用户\AppData\Roaming", "CODEX_HOME": r"D:\Codex 目录"}
    for name, value in locations.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-inherit")
    monkeypatch.setenv("CODEX_THREAD_ID", "do-not-inherit")
    environment = codex_cli._environment()
    assert all(environment.get(name) == value for name, value in locations.items())
    assert "OPENAI_API_KEY" not in environment and "CODEX_THREAD_ID" not in environment


@pytest.mark.parametrize("suffix", [".cmd", ".CMD", ".bat"])
def test_windows_npm_shim_resolves_to_node_without_a_shell(tmp_path, monkeypatch, suffix):
    monkeypatch.setattr(codex_cli, "_WINDOWS", True)
    shim = tmp_path / f"codex{suffix}"
    shim.write_text("this batch script must never run")
    entry = tmp_path / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("// npm entry point")
    node = tmp_path / "node.exe"
    node.touch()
    monkeypatch.setattr(codex_cli.shutil, "which", lambda _: pytest.fail("prefer npm's adjacent Node"))
    assert codex_cli._command(str(shim)) == [str(node), str(entry)]


def test_windows_npm_shim_uses_node_from_path_or_rejects_missing_install(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_cli, "_WINDOWS", True)
    shim = tmp_path / "codex.cmd"
    shim.touch()
    entry = tmp_path / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    entry.parent.mkdir(parents=True)
    entry.touch()
    monkeypatch.setattr(codex_cli.shutil, "which", lambda name: r"C:\Program Files\nodejs\node.exe")
    assert codex_cli._command(str(shim)) == [r"C:\Program Files\nodejs\node.exe", str(entry)]
    monkeypatch.setattr(codex_cli.shutil, "which", lambda name: None)
    with pytest.raises(CodexCLIError, match="Node.js"):
        codex_cli._command(str(shim))
    entry.unlink()
    with pytest.raises(CodexCLIError, match="npm"):
        codex_cli._command(str(shim))


def test_native_cli_path_is_not_reinterpreted_as_a_shell_command(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_cli, "_WINDOWS", True)
    executable = str(tmp_path / "native codex.exe")
    assert codex_cli._command(executable) == [executable]


@pytest.mark.skipif(os.name != "nt", reason="Windows npm shim and Job Object integration")
@pytest.mark.asyncio
async def test_windows_npm_launch_preserves_unicode_shell_characters_and_arguments(tmp_path, monkeypatch):
    node = shutil.which("node.exe")
    if not node:
        pytest.skip("Node.js is not installed")
    directory = tmp_path / "影像 & (100%) npm"
    directory.mkdir()
    shim = directory / "codex.cmd"
    sentinel = tmp_path / "batch-was-executed"
    shim.write_text(f'@echo off\necho unsafe > "{sentinel}"\n', encoding="utf-8")
    entry = directory / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
    entry.parent.mkdir(parents=True)
    entry.write_text('process.stdin.setEncoding("utf8"); let input="";'
                     'process.stdin.on("data", chunk => input += chunk);'
                     'process.stdin.on("end", () => console.log(JSON.stringify({args:process.argv.slice(2), input})));',
                     encoding="utf-8")
    arguments = ["原始 & echo injected", '(100%) "quoted"', 'model_instructions_file="D:\\\\影像 路径\\\\a.txt"']
    code, output, errors = await codex_cli._run([str(shim), *arguments], directory=directory,
                                               timeout_s=5, stdin="Unicode 用户数据 & %PATH%".encode("utf-8"))
    assert code == 0 and errors == b""
    assert json.loads(output) == {"args": arguments, "input": "Unicode 用户数据 & %PATH%"}
    assert not sentinel.exists()


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """A real child process with deterministic output, never a model request."""
    configuration = tmp_path / "fake-config.json"
    log = tmp_path / "calls"
    log.mkdir()
    configuration.write_text("{}")
    script = tmp_path / "codex.py"
    script.write_text('''
import json
import os
import subprocess
import sys
import time
from pathlib import Path

configuration = Path(CONFIGURATION)
log = Path(LOG)
config = json.loads(configuration.read_text())
args = sys.argv[1:]
if args == ["--version"]:
    print(config.get("version", "codex-cli 0.155.1"))
elif args == ["exec", "--help"]:
    print(config.get("help", "--ignore-user-config --ignore-rules --ephemeral --json --image"))
elif args == ["features", "list"]:
    print(config.get("features", FEATURES))
elif args[:2] == ["login", "status"]:
    print(config.get("login", "Logged in using ChatGPT"), file=sys.stderr)
    sys.exit(config.get("login_code", 0))
else:
    prompt = sys.stdin.buffer.read().decode("utf-8")
    images = [Path(args[i + 1]) for i, value in enumerate(args) if value == "--image"]
    instructions = next(value.split("=", 1)[1] for value in args if value.startswith("model_instructions_file="))
    record = {"args": args, "prompt": prompt, "cwd": os.getcwd(), "pid": os.getpid(), "env": dict(os.environ),
              "created": __import__("psutil").Process().create_time(),
              "image_data": [base64_encode(image.read_bytes()) for image in images],
              "image_modes": [image.stat().st_mode & 0o777 for image in images],
              "instructions": Path(json.loads(instructions)).read_text(encoding="utf-8")}
    if "--output-schema" in args:
        schema_path = Path(args[args.index("--output-schema") + 1])
        record["output_schema"] = json.loads(schema_path.read_text())
        record["schema_mode"] = schema_path.stat().st_mode & 0o777
    scenario = config.get("scenario", "success")
    if scenario in {"orphan", "descendant"}:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 stdin=subprocess.DEVNULL)
        record["child_pid"] = child.pid
        record["child_created"] = __import__("psutil").Process(child.pid).create_time()
    # Windows append writes from separate processes can overwrite each other.
    # One atomically published record per child also keeps polling readers safe.
    pending = log / f"{os.getpid()}.tmp"
    pending.write_text(json.dumps(record), encoding="utf-8")
    pending.replace(log / f"{os.getpid()}.json")
    if scenario == "timeout":
        time.sleep(60)
    if scenario == "orphan":
        sys.exit(0)
    if scenario == "descendant":
        time.sleep(60)
    if scenario == "flood":
        print("x" * (5 * 1024 * 1024), flush=True)
        time.sleep(60)
    if scenario == "stderr_flood":
        print("secret" * (100 * 1024), file=sys.stderr, flush=True)
        time.sleep(60)
    if scenario == "invalid":
        print("secret-invalid-output")
        sys.exit(0)
    if scenario == "failure":
        print("secret-credential", file=sys.stderr)
        print(json.dumps({"type": "turn.failed", "error": {"message": "secret-credential"}}))
        sys.exit(1)
    if scenario == "slow":
        time.sleep(0.1)
    events = config.get("events", [
        {"type": "thread.started", "thread_id": "hidden-thread"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "intermediate"}},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "{\\"search_text\\": \\"回家\\"}"}},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 9}},
    ])
    for event in events:
        print(json.dumps(event), flush=True)
    sys.exit(config.get("exit_code", 0))
'''.replace("CONFIGURATION", repr(str(configuration))).replace("LOG", repr(str(log)))
        .replace("FEATURES", repr("\n".join(f"{name} stable true" for name in (
            *codex_cli._DISABLED_FEATURES, "skip_host_skill_discovery")))).replace(
                "base64_encode(image.read_bytes())", "__import__('base64').b64encode(image.read_bytes()).decode()"),
                      encoding="utf-8")
    script.chmod(0o700)
    original_command = codex_cli._command

    def command(executable):
        if Path(executable).resolve() == script.resolve():
            return [sys.executable, "-X", "utf8", str(script)]
        return original_command(executable)

    monkeypatch.setattr(codex_cli, "_command", command)
    monkeypatch.setenv("SCENERECALL_CODEX_PATH", str(script))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))

    def configure(**values):
        configuration.write_text(json.dumps(values))

    def calls():
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(log.glob("*.json"))]

    return configure, calls


@pytest.mark.asyncio
async def test_status_reports_installation_chatgpt_login_and_compatibility(cli):
    result = await codex_cli.status()
    assert result == {"installed": True, "authenticated": True, "compatible": True,
                      "auth_method": "chatgpt", "version": "0.155.1", "message": "Codex CLI 已通过 ChatGPT 登录"}


@pytest.mark.asyncio
async def test_missing_cli_and_invalid_server_override(monkeypatch, tmp_path):
    monkeypatch.setenv("SCENERECALL_CODEX_PATH", str(tmp_path / "missing"))
    assert not (await codex_cli.status())["installed"]
    with pytest.raises(CodexCLIError, match="未找到") as error:
        await codex_cli.complete("instruction", "text", "gpt-5.4", 5)
    assert error.value.request_count == 0


@pytest.mark.parametrize("settings,field,value", [
    ({"version": "codex-cli 0.154.0"}, "compatible", False),
    ({"help": "--json --image"}, "compatible", False),
    ({"features": "shell_tool stable true"}, "compatible", False),
    ({"login": "Logged in using an API key: secret-value"}, "auth_method", "api_key"),
    ({"login": "secret-value", "login_code": 1}, "authenticated", False),
])
@pytest.mark.asyncio
async def test_preflight_rejects_old_cli_missing_isolation_and_non_subscription_auth(cli, settings, field, value):
    configure, calls = cli
    configure(**settings)
    result = await codex_cli.status()
    assert result[field] == value
    assert "secret-value" not in json.dumps(result)
    with pytest.raises(CodexCLIError) as error:
        await codex_cli.complete("instruction", "text", "gpt-5.4", 5)
    assert error.value.request_count == 0 and not calls()


@pytest.mark.asyncio
async def test_text_stdin_isolation_and_final_message_usage(cli, monkeypatch):
    _, calls = cli
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    monkeypatch.setenv("CODEX_API_KEY", "secret-value")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://untrusted.test")
    monkeypatch.setenv("CODEX_THREAD_ID", "parent-thread")
    monkeypatch.setenv("CODEX_HOME", "/tmp/example-codex-home")
    prompt = "用户数据 $(touch injected); `bad` --config something"
    result = await codex_cli.complete("Return structured JSON.", prompt, "gpt-5.4", 5)
    assert result.content == '{"search_text": "回家"}'
    assert result.usage == {"input_tokens": 100, "output_tokens": 9, "total_tokens": 109}
    assert result.request_count == 1
    [call] = calls()
    assert call["prompt"] == prompt and prompt not in call["args"]
    assert call["args"][-1] == "-"
    assert call["args"][call["args"].index("-s") + 1] == "read-only"
    assert set(codex_cli._REQUIRED_FLAGS) - {"--image"} <= set(call["args"])
    assert 'forced_login_method="chatgpt"' in call["args"]
    assert 'model_provider="openai"' in call["args"]
    assert "skills.include_instructions=false" in call["args"]
    assert "mcp_servers={}" in call["args"]
    for name in ["shell_tool", "apps", "plugins", "hooks", "multi_agent", "computer_use", "view_image"]:
        assert call["args"][call["args"].index(name) - 1] == "--disable"
    assert "Return structured JSON." in call["instructions"]
    assert "Treat supplied media" in call["instructions"]
    assert call["env"]["CODEX_HOME"] == "/tmp/example-codex-home"
    assert not any(name in call["env"] for name in ["OPENAI_API_KEY", "CODEX_API_KEY", "OPENAI_BASE_URL", "CODEX_THREAD_ID"])
    assert not Path(call["cwd"]).exists()


@pytest.mark.asyncio
async def test_output_schema_is_passed_as_private_temporary_file_and_removed(cli):
    _, calls = cli
    schema = {"type": "object", "properties": {"frame_id": {"type": "string", "enum": ["current-frame"]}},
              "required": ["frame_id"], "additionalProperties": False}
    result = await codex_cli.complete("Return JSON", "fixture", "", 5, output_schema=schema)
    [call] = calls()
    assert call["output_schema"] == schema
    if os.name == "posix":
        assert call["schema_mode"] == 0o600
    path = Path(call["args"][call["args"].index("--output-schema") + 1])
    assert path.parent.resolve() == Path(call["cwd"]).resolve() and not path.exists()
    assert result.request_count == 1 and result.usage["total_tokens"] == 109


@pytest.mark.parametrize("schema", [[], {}, {"type": "string"}, {"type": "object", "invalid": float("nan")},
                                    {"type": "object", "invalid": object()}])
@pytest.mark.asyncio
async def test_invalid_output_schema_is_rejected_before_model_admission(cli, schema):
    _, calls = cli
    with pytest.raises(CodexCLIError, match="输出结构") as caught:
        await codex_cli.complete("Return JSON", "fixture", "", 5, output_schema=schema)
    assert caught.value.request_count == 0 and not calls()


@pytest.mark.asyncio
async def test_oversized_output_schema_is_rejected_before_model_admission(cli, monkeypatch):
    _, calls = cli
    monkeypatch.setattr(codex_cli, "MAX_SCHEMA_BYTES", 32)
    with pytest.raises(CodexCLIError, match="输出结构") as caught:
        await codex_cli.complete("Return JSON", "fixture", "", 5,
                                 output_schema={"type": "object", "description": "x" * 64})
    assert caught.value.request_count == 0 and not calls()


@pytest.mark.asyncio
async def test_images_preserve_frame_order_and_are_private_and_removed(cli):
    _, calls = cli
    content = []
    for index, payload in enumerate([b"first-image", b"second-image"]):
        content.extend([{"type": "text", "text": json.dumps({"frame_id": f"f{index}", "at_ms": index * 500})},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                            base64.b64encode(payload).decode()}}])
    await codex_cli.complete("Analyze frames.", content, "gpt-5.4", 5)
    [call] = calls()
    assert call["image_data"] == [base64.b64encode(data).decode() for data in [b"first-image", b"second-image"]]
    if os.name == "posix":
        assert call["image_modes"] == [0o600, 0o600]
    assert call["prompt"].index('"f0"') < call["prompt"].index("Attached image 1") < call["prompt"].index('"f1"')
    assert call["prompt"].index('"f1"') < call["prompt"].index("Attached image 2")
    assert call["args"].count("--image") == 2
    assert not Path(call["cwd"]).exists()


@pytest.mark.parametrize("content", [
    [{"type": "image_url", "image_url": {"url": "file:///etc/passwd"}}],
    [{"type": "image_url", "image_url": {"url": "https://example.test/image.png"}}],
    [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abcd="}}],
    [{"type": "audio", "data": "value"}],
    [{"type": "text", "text": 123}],
])
@pytest.mark.asyncio
async def test_rejects_unsupported_or_malformed_input_before_start(cli, content):
    _, calls = cli
    with pytest.raises(CodexCLIError) as error:
        await codex_cli.complete("instruction", content, "gpt-5.4", 5)
    assert error.value.request_count == 0 and not calls()


@pytest.mark.parametrize("scenario,message", [("failure", "推理失败"), ("invalid", "无效事件"),
                                            ("flood", "大小限制"), ("stderr_flood", "大小限制")])
@pytest.mark.asyncio
async def test_error_output_is_bounded_sanitized_and_cleaned_up(cli, scenario, message):
    configure, calls = cli
    configure(scenario=scenario)
    with pytest.raises(CodexCLIError, match=message) as error:
        await codex_cli.complete("instruction", "text", "gpt-5.4", 5)
    assert "secret" not in str(error.value)
    assert error.value.request_count == 1
    assert not Path(calls()[0]["cwd"]).exists()
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(calls()[0]["pid"], 0)


@pytest.mark.asyncio
async def test_timeout_kills_reaps_and_cleans_up(cli):
    configure, calls = cli
    configure(scenario="timeout")
    with pytest.raises(CodexCLIError, match="超时") as error:
        # Windows launches four inspection processes before model admission.
        await codex_cli.complete("instruction", "text", "gpt-5.4", 3 if os.name == "nt" else 1)
    assert error.value.request_count == 1
    [call] = calls()
    assert not Path(call["cwd"]).exists()
    _assert_process_stopped(call["pid"], call["created"])


@pytest.mark.asyncio
async def test_cancellation_kills_reaps_and_cleans_up(cli):
    configure, calls = cli
    configure(scenario="timeout")
    task = asyncio.create_task(codex_cli.complete("instruction", "text", "gpt-5.4", 10))
    async with asyncio.timeout(5):
        while not calls():
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    [call] = calls()
    assert not Path(call["cwd"]).exists()
    _assert_process_stopped(call["pid"], call["created"])


@pytest.mark.asyncio
async def test_parallel_calls_have_independent_workspaces(cli):
    configure, calls = cli
    configure(scenario="slow")
    results = await asyncio.gather(*(codex_cli.complete("instruction", f"query {index}", "gpt-5.4", 5)
                                    for index in range(3)))
    assert len(results) == 3
    records = calls()
    assert len({call["cwd"] for call in records}) == 3
    assert {call["prompt"] for call in records} == {"query 0", "query 1", "query 2"}
    assert all(not Path(call["cwd"]).exists() for call in records)


@pytest.mark.parametrize("events", [
    [{"type": "item.completed", "item": {"type": "agent_message", "text": "partial"}}],
    [{"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 3}}],
    [{"type": "turn.completed", "usage": {"input_tokens": -1, "output_tokens": 3}}],
    [{"type": "turn.completed", "usage": {"input_tokens": True, "output_tokens": 3}}],
])
@pytest.mark.asyncio
async def test_incomplete_output_and_invalid_usage_fail(cli, events):
    configure, _ = cli
    configure(events=events)
    with pytest.raises(CodexCLIError) as error:
        await codex_cli.complete("instruction", "text", "gpt-5.4", 5)
    assert error.value.request_count == 1


@pytest.mark.asyncio
async def test_failure_after_reported_usage_preserves_accounting(cli):
    configure, _ = cli
    configure(events=[{"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 3}},
                      {"type": "error", "message": "secret"}], exit_code=1)
    with pytest.raises(CodexCLIError) as error:
        await codex_cli.complete("instruction", "text", "gpt-5.4", 5)
    assert error.value.usage == {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}


@pytest.mark.asyncio
async def test_recovered_transport_error_and_default_model(cli):
    configure, calls = cli
    configure(events=[{"type": "error", "message": "Reconnecting"},
                      {"type": "item.completed", "item": {"type": "agent_message", "text": "{}"}},
                      {"type": "turn.completed", "usage": {"input_tokens": 5, "output_tokens": 3}}])
    result = await codex_cli.complete("instruction", "text", "", 5)
    assert result.content == "{}"
    assert "-m" not in calls()[0]["args"]


@pytest.mark.asyncio
async def test_timeout_kills_descendants_after_leader_exits(cli):
    configure, calls = cli
    configure(scenario="orphan")
    with pytest.raises(CodexCLIError, match="超时") as error:
        await asyncio.wait_for(codex_cli.complete("instruction", "text", "", 3 if os.name == "nt" else 1), timeout=6)
    assert error.value.request_count == 1
    [call] = calls()
    assert not Path(call["cwd"]).exists()
    _assert_process_stopped(call["child_pid"], call["child_created"])


def _assert_process_stopped(pid, created):
    try:
        child = psutil.Process(pid)
        # Zombies have stopped and closed their inherited pipes. Include create
        # time so a recycled PID never makes the assertion inspect another app.
        assert child.create_time() != created or child.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        pass


@pytest.mark.asyncio
async def test_cancellation_kills_native_descendants_and_closes_pipes(cli):
    configure, calls = cli
    configure(scenario="descendant")
    task = asyncio.create_task(codex_cli.complete("instruction", "text", "", 10))
    async with asyncio.timeout(5):
        while not calls():
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=4)
    [call] = calls()
    assert not Path(call["cwd"]).exists()
    _assert_process_stopped(call["child_pid"], call["child_created"])


@pytest.mark.asyncio
async def test_text_and_image_limits_reject_before_model_start(cli, monkeypatch):
    _, calls = cli
    monkeypatch.setattr(codex_cli, "MAX_PROMPT_BYTES", 16)
    with pytest.raises(CodexCLIError, match="大小限制"):
        await codex_cli.complete("instruction", "x" * 17, "", 5)
    monkeypatch.setattr(codex_cli, "MAX_IMAGE_BYTES", 4)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(b"12345").decode()}}
    with pytest.raises(CodexCLIError, match="大小超过"):
        await codex_cli.complete("instruction", [image], "", 5)
    assert not calls()


@pytest.fixture
def model_cache(cli):
    directory = Path(os.environ["CODEX_HOME"])
    directory.mkdir()
    path = directory / "models_cache.json"
    metadata = {
        "fetched_at": "2026-10-13T02:00:00Z",
        "identity": {"account": "private-account-metadata"},
        "models": [
            {"slug": "model-small", "display_name": "Small", "visibility": "list",
             "supported_in_api": False, "default_reasoning_level": "medium",
             "supported_reasoning_levels": [{"effort": "low"}, {"effort": "medium"}]},
            {"slug": "model-large", "display_name": "Large", "visibility": "list",
             "default_reasoning_level": "high",
             "supported_reasoning_levels": [{"effort": "high"}, {"effort": "ultra"}]},
            {"slug": "model-hidden", "display_name": "Hidden", "visibility": "hide",
             "supported_reasoning_levels": [{"effort": "low"}]},
        ],
    }
    path.write_text(json.dumps(metadata))
    return path, metadata


@pytest.mark.asyncio
async def test_model_catalog_uses_visible_cached_metadata_without_reading_credentials(cli, model_cache):
    _, calls = cli
    path, _ = model_cache
    (path.parent / "auth.json").write_text("secret-credential")
    result = await codex_cli.model_catalog()
    assert result["models"] == [
        {"id": "model-small", "name": "Small", "reasoning_efforts": ["low", "medium"],
         "default_reasoning_effort": "medium"},
        {"id": "model-large", "name": "Large", "reasoning_efforts": ["high", "ultra"],
         "default_reasoning_effort": "high"},
    ]
    assert result["source"] == "local_cache" and result["fetched_at"] == "2026-10-13T02:00:00Z"
    assert result["default_model"] is None
    assert "缓存" in result["message"]
    assert "private-account" not in json.dumps(result) and "secret-credential" not in json.dumps(result)
    assert not calls()


@pytest.mark.asyncio
async def test_model_catalog_reads_updated_cache_without_stale_process_state(model_cache):
    path, metadata = model_cache
    await codex_cli.model_catalog()
    metadata["models"][0]["supported_reasoning_levels"].append({"effort": "xhigh"})
    path.write_text(json.dumps(metadata))
    result = await codex_cli.model_catalog()
    assert result["models"][0]["reasoning_efforts"] == ["low", "medium", "xhigh"]


@pytest.mark.asyncio
async def test_model_catalog_defends_against_invalid_entries(model_cache):
    path, metadata = model_cache
    metadata["fetched_at"] = "private-secret"
    metadata["models"] = [None, [], {"visibility": "list", "slug": "--config=bad"},
                          {"visibility": "list", "slug": "valid", "display_name": {},
                           "supported_reasoning_levels": [None, {}, {"effort": []}, {"effort": "private-secret"},
                                                          {"effort": "low"}, {"effort": "low"}],
                           "default_reasoning_level": "max"},
                          {"visibility": "list", "slug": "valid"}]
    path.write_text(json.dumps(metadata))
    result = await codex_cli.model_catalog()
    assert result["models"] == [{"id": "valid", "name": "valid", "reasoning_efforts": ["low"],
                                  "default_reasoning_effort": None}]
    assert result["fetched_at"] is None
    assert "private-secret" not in json.dumps(result)


@pytest.mark.parametrize("payload", [None, b"private-secret", b"[]", b'{"models":{}}', b'{"models":[]}'])
@pytest.mark.asyncio
async def test_model_catalog_missing_or_invalid_cache_falls_back_to_cli_defaults(cli, tmp_path, payload):
    directory = tmp_path / "codex-home"
    directory.mkdir()
    if payload is not None:
        (directory / "models_cache.json").write_bytes(payload)
    result = await codex_cli.model_catalog()
    assert result["models"] == [] and result["default_model"] is None and result["source"] == "unavailable"
    assert "默认" in result["message"] and "private-secret" not in result["message"]


@pytest.mark.asyncio
async def test_model_catalog_size_is_bounded(model_cache, monkeypatch):
    monkeypatch.setattr(codex_cli, "MAX_CATALOG_BYTES", 32)
    assert (await codex_cli.model_catalog())["models"] == []


@pytest.mark.asyncio
async def test_model_catalog_unavailable_without_compatible_authenticated_cli(cli, model_cache):
    configure, _ = cli
    configure(login="secret-login-output", login_code=1)
    result = await codex_cli.model_catalog()
    assert result["models"] == [] and result["source"] == "unavailable"
    assert "secret" not in json.dumps(result)


@pytest.mark.asyncio
async def test_reasoning_effort_reaches_cli_as_safe_config_and_preserves_isolation(cli, model_cache):
    _, calls = cli
    await codex_cli.complete("Return JSON", "prompt", "model-large", 5, reasoning_effort="ultra")
    [call] = calls()
    assert 'model_reasoning_effort="ultra"' in call["args"]
    assert call["args"][call["args"].index('model_reasoning_effort="ultra"') - 1] == "-c"
    assert call["args"][call["args"].index("-m") + 1] == "model-large"
    assert set(codex_cli._REQUIRED_FLAGS) - {"--image"} <= set(call["args"])


@pytest.mark.parametrize("effort", [None, ""])
@pytest.mark.asyncio
async def test_unselected_reasoning_effort_preserves_cli_default(cli, effort):
    _, calls = cli
    await codex_cli.complete("instruction", "prompt", "", 5, reasoning_effort=effort)
    assert not any(arg.startswith("model_reasoning_effort=") for arg in calls()[0]["args"])


@pytest.mark.parametrize("effort", ["ultra", "high", 'low"\nmodel_provider="custom"', "$(touch bad)", [], True])
@pytest.mark.asyncio
async def test_invalid_or_unsupported_reasoning_rejected_before_model_start(cli, model_cache, effort):
    _, calls = cli
    with pytest.raises(CodexCLIError, match="思考等级") as error:
        await codex_cli.complete("instruction", "prompt", "model-small", 5, reasoning_effort=effort)
    assert error.value.request_count == 0 and not calls()


@pytest.mark.parametrize("model", ["", "legacy-model"])
def test_reasoning_validation_preserves_default_and_unknown_models(model_cache, model):
    codex_cli.validate_reasoning_effort(model, "ultra")


def test_reasoning_validation_allows_known_effort_when_cache_is_unavailable(cli):
    codex_cli.validate_reasoning_effort("model-small", "high")

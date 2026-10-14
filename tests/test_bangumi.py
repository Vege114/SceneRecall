import asyncio
import json

import pytest

from scenerecall.bangumi import BangumiClient, BangumiError, parse_subject_reference


@pytest.mark.parametrize("reference", [
    "348335", 348335, " https://bgm.tv/subject/348335 ",
    "https://bangumi.tv/subject/348335/?x=1#info", "http://chii.in/subject/348335",
])
def test_subject_reference(reference):
    assert parse_subject_reference(reference) == 348335


@pytest.mark.parametrize("reference", [
    "0", -1, True, "https://evil.test/subject/348335", "https://bgm.tv.evil.test/subject/348335",
    "https://bgm.tv/character/348335", "https://user@bgm.tv/subject/348335",
    "https://bgm.tv:8080/subject/348335", "https://bgm.tv/subject/1/../../2",
    "https://bgm.tv/subject/348335;touch /tmp/injected", str(2**53),
])
def test_reject_invalid_subject_reference(reference):
    with pytest.raises(BangumiError, match="请输入|正整数"):
        parse_subject_reference(reference)


class FixtureClient(BangumiClient):
    def __init__(self, results):
        super().__init__()
        self.results = results
        self.calls = []

    async def _run(self, *arguments):
        self.calls.append(arguments)
        result = self.results[arguments[:3]]
        if isinstance(result, Exception):
            raise result
        return result


async def test_cli_p1_envelopes_and_external_fields_are_normalized():
    client = FixtureClient({
        ("subject", "search", "平家物语"): {"data": [
            {"id": 348335, "name": "平家物語", "nameCN": "平家物语", "type": 2,
             "images": {"large": "https://lain.bgm.tv/cover.jpg"}},
        ]},
        ("subject", "get", "348335"): {
            "id": 348335, "name": "平家物語", "nameCN": "平家物语", "type": 2,
            "airtime": {"date": "2022-01-12"}, "summary": "条目简介", "eps": 11,
        },
        ("subject", "characters", "348335"): {"data": [
            {"character": {"id": 100314, "name": "びわ", "nameCN": "琵琶"}, "type": 1},
            {"character": {"id": 100315, "name": "平重盛", "nameCN": "平重盛"}, "type": 2},
        ], "total": 2},
        ("character", "get", "100314"): {"resource": "character", "data": {
            "id": 100314, "name": "びわ", "nameCN": "琵琶", "summary": "异色瞳少女",
            "infobox": [{"key": "别名", "values": [{"k": "英文名", "v": "Biwa"}]}],
            "images": {"large": "https://lain.bgm.tv/biwa.jpg"},
        }},
        ("character", "get", "100315"): BangumiError("暂不可读取"),
    })
    assert (await client.search("平家物语"))[0]["name_cn"] == "平家物语"
    dossier = await client.dossier(348335)
    assert dossier["subject"]["date"] == "2022-01-12"
    assert dossier["subject"]["eps"] == 11
    assert dossier["source"] == "bangumi" and dossier["fetched_at"]
    assert dossier["characters"][0]["aliases"] == ["Biwa", "びわ", "琵琶"]
    assert dossier["characters"][0]["description"] == "异色瞳少女"
    assert dossier["characters"][0]["role"] == "主角"
    assert dossier["characters"][1]["name"] == "平重盛"
    assert dossier["characters"][1]["detail_error"] == "暂不可读取"
    assert dossier["warnings"]


async def test_reject_non_anime_subject_before_character_fetch():
    client = FixtureClient({("subject", "get", "1"): {"id": 1, "name": "书籍", "type": 1}})
    with pytest.raises(BangumiError, match="不是动画"):
        await client.dossier(1)
    assert len(client.calls) == 1


async def test_cli_process_uses_literal_args_and_isolated_credential_free_environment(tmp_path, monkeypatch):
    script = tmp_path / "fake-cli.js"
    script.write_text("""
console.log(JSON.stringify({
  arguments: process.argv.slice(2),
  token: process.env.BGM_ACCESS_TOKEN || null,
  configuration: process.env.BGM_CONFIG_DIR,
  injectedNodeOptions: process.env.NODE_OPTIONS || null
}));
""")
    monkeypatch.setenv("SCENERECALL_BGM_CLI_PATH", str(script))
    monkeypatch.setenv("BGM_ACCESS_TOKEN", "private-token")
    monkeypatch.setenv("BGM_CONFIG_DIR", str(tmp_path / "personal-config"))
    monkeypatch.setenv("NODE_OPTIONS", "--bad-option")
    query = "平家物语 $(touch injected); `bad`"
    result = await BangumiClient()._run("subject", "search", query)
    assert result["arguments"] == ["--json", "subject", "search", query]
    assert result["token"] is None and result["injectedNodeOptions"] is None
    assert result["configuration"] != str(tmp_path / "personal-config")
    assert not (tmp_path / "personal-config").exists()


async def test_timeout_kills_real_child_process(tmp_path, monkeypatch):
    script = tmp_path / "slow-cli.js"
    pid_file = tmp_path / "pid"
    script.write_text("import fs from 'node:fs';\n"
                      f"fs.writeFileSync({json.dumps(str(pid_file))}, String(process.pid));\n"
                      "setInterval(() => {}, 1000);\n")
    monkeypatch.setenv("SCENERECALL_BGM_CLI_PATH", str(script))
    with pytest.raises(BangumiError, match="超时"):
        await BangumiClient(timeout_s=0.2)._run("subject", "get", "1")
    assert pid_file.exists()
    # A failed process must be reaped rather than left in the server's event loop.
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


async def test_request_concurrency_is_bounded(tmp_path, monkeypatch):
    script = tmp_path / "cli.js"
    script.write_text("setTimeout(() => console.log('{}'), 50);")
    monkeypatch.setenv("SCENERECALL_BGM_CLI_PATH", str(script))
    client = BangumiClient(max_concurrency=2)
    active = 0
    peak = 0
    execute = client._execute

    async def counted(*arguments):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            return await execute(*arguments)
        finally:
            active -= 1

    monkeypatch.setattr(client, "_execute", counted)
    await asyncio.gather(*(client._run("subject", "get", "1") for _ in range(6)))
    assert peak == 2

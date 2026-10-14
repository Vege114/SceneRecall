import asyncio
import json

import pytest

from scenerecall.bangumi import (BangumiClient, BangumiError, character_names, normalize_dossier_names,
                                parse_subject_reference)


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
    assert dossier["characters"][0]["name_en"] == "Biwa"
    assert dossier["characters"][0]["name_ja"] == "びわ"
    assert next(value for value in dossier["characters"][0]["name_variants"]
                if value["name"] == "Biwa")["language"] == "en"
    assert dossier["characters"][0]["description"] == "异色瞳少女"
    assert dossier["characters"][0]["role"] == "主角"
    assert dossier["characters"][1]["name"] == "平重盛"
    assert dossier["characters"][1]["detail_error"] == "暂不可读取"
    assert dossier["warnings"]


@pytest.mark.parametrize("envelope", ["value", "values"])
def test_nested_infobox_name_labels_preserve_language_and_prefer_english(envelope):
    raw = {"id": 7, "name": "架空人物", "infobox": [
        {"key": "别名", envelope: [{"k": "第二中文名", "v": "另一中文名"},
                                  {"k": "日文名", "v": "架空 人物"},
                                  {"k": "纯假名", "v": "かくう じんぶつ"},
                                  {"k": "罗马字", "v": "Kakuu Jinbutsu"},
                                  {"k": "英文名", "v": "Imaginary Person"},
                                  {"k": "昵称", "v": "舞台君"}]},
    ]}
    names = character_names(raw)
    assert names["name_cn"] == "另一中文名"
    assert names["name_ja"] == "架空 人物"
    assert names["name_en"] == "Imaginary Person"
    assert set(names["aliases"]) >= {"另一中文名", "架空 人物", "Kakuu Jinbutsu", "Imaginary Person", "舞台君"}
    variants = {value["name"]: value for value in names["name_variants"]}
    assert variants["另一中文名"]["language"] == "zh-Hans"
    assert variants["Kakuu Jinbutsu"]["language"] == "ja-Latn"
    assert variants["Kakuu Jinbutsu"]["kind"] == "romanization"
    assert variants["Imaginary Person"]["language"] == "en"
    assert all(value["source"] == "bangumi" and value["source_url"] == "https://bgm.tv/character/7"
               for value in variants.values())


def test_missing_latin_name_uses_only_supplied_kana_reading():
    raw = {"id": 137797, "name": "眞井霧子", "name_cn": "真井雾子", "infobox": [
        {"key": "别名", "values": [{"k": "纯假名", "v": "まい きりこ"}]},
    ]}
    names = character_names(raw)
    assert names["name_en"] == "Mai Kiriko"
    derived = next(value for value in names["name_variants"] if value["name"] == "Mai Kiriko")
    assert derived == {"name": "Mai Kiriko", "language": "ja-Latn", "kind": "romanization",
                       "source": "kana_transliteration", "source_url": "https://bgm.tv/character/137797"}
    assert character_names({"id": 137797, "name": "眞井霧子", "name_cn": "真井雾子"})["name_en"] == ""
    raw["infobox"][0]["values"][0]["v"] = "眞井 きりこ"
    assert character_names(raw)["name_en"] == ""


def test_missing_chinese_name_converts_han_script_without_inventing_kana_translation():
    names = character_names({"id": 150045, "name": "櫻木麗", "infobox": [
        {"key": "别名", "values": [{"k": "罗马字", "v": "Sakuragi Rei"}]},
    ]})
    assert names["name_cn"] == "樱木丽" and names["name_en"] == "Sakuragi Rei"
    assert {"name": "樱木丽", "language": "zh-Hans", "kind": "script_conversion",
            "source": "opencc", "source_url": "https://bgm.tv/character/150045"} in names["name_variants"]
    assert "櫻木麗" in names["aliases"]
    assert character_names({"id": 1, "name": "サクラ"})["name_cn"] == ""
    traditional = character_names({"id": 2, "name": "Sakura", "infobox": [
        {"key": "繁体中文名", "value": "櫻"},
    ]})
    assert traditional["name_cn"] == "樱"
    assert any(value["language"] == "zh-Hant" for value in traditional["name_variants"])


def test_cached_snapshot_name_upgrade_is_pure_and_idempotent():
    dossier = {"source": "bangumi", "characters": [{"id": 150045, "name": "櫻木麗", "aliases": ["旧别名"],
                                                       "infobox": []}]}
    upgraded = normalize_dossier_names(dossier)
    assert "name_schema_version" not in dossier
    assert upgraded["name_schema_version"] == 1
    assert "旧别名" in upgraded["characters"][0]["aliases"]
    assert upgraded == normalize_dossier_names(upgraded)
    converted = next(value for value in upgraded["characters"][0]["name_variants"] if value["name"] == "樱木丽")
    assert converted["source"] == "opencc"


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

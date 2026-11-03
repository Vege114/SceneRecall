from types import SimpleNamespace

import pytest

from scenerecall.character_query import CharacterQueryResolver, name_index, resolve_text
from scenerecall.characters import scope_id
from scenerecall.library import Library, atomic_json
from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine
from scenerecall.search_timing import trace_request


def profile(identity, name, scope="cast-one", aliases=()):
    return {"id": identity, "scope_id": scope, "name": name, "aliases": list(aliases),
            "description": "", "notes": "", "appearance": [], "edited_fields": ["name"],
            "user_edited": True, "created_at": "2026-11-03", "updated_at": "2026-11-03"}


def revue_names():
    return name_index([profile("karen", "爱城华恋", aliases=["愛城華恋", "Karen Aijo"]),
                       profile("hikari", "神乐光", aliases=["神楽ひかり", "Hikari Kagura"])])


@pytest.mark.parametrize(("query", "expected"), [
    ("华恋", "爱城华恋"),
    ("华恋与光站在东京塔上", "爱城华恋与神乐光站在东京塔上"),
    ("光与华恋站在东京塔上", "神乐光与爱城华恋站在东京塔上"),
    ("华恋和神楽ひかり在东京塔", "爱城华恋和神乐光在东京塔"),
    ("爱城华恋与神乐光站在东京塔上", "爱城华恋与神乐光站在东京塔上"),
    ("华恋在阳光下看光线", "爱城华恋在阳光下看光线"),
    ("华恋与光线一起出现", "爱城华恋与光线一起出现"),
    ("光站在东京塔上", "光站在东京塔上"),
    ("光", "光"),
    ("karen aijo stands by a tower", "爱城华恋 stands by a tower"),
    ("xxKaren Aijo", "xxKaren Aijo"),
    ("华恋与光在 TOKYO TOWER！", "爱城华恋与神乐光在 tokyo tower!"),
    ("爱城华恋与神乐光在 TOKYO TOWER！", "爱城华恋与神乐光在 tokyo tower!"),
])
def test_local_shorthand_and_single_character_context(query, expected):
    assert resolve_text(query, revue_names())["text"] == expected


def test_cross_work_collisions_abstain_and_explicit_alias_has_priority():
    profiles = [profile("karen", "爱城华恋"), profile("other", "另一个华恋", "cast-two")]
    assert resolve_text("华恋站在塔上", name_index(profiles))["text"] == "华恋站在塔上"
    profiles[0]["aliases"] = ["华恋"]
    assert resolve_text("华恋站在塔上", name_index(profiles))["text"] == "爱城华恋站在塔上"
    profiles[1]["aliases"] = ["华恋"]
    assert resolve_text("华恋站在塔上", name_index(profiles))["text"] == "华恋站在塔上"


def test_single_character_collision_and_explicit_alias_boundary():
    values = [profile("karen", "爱城华恋"), profile("hikari", "神乐光"), profile("other", "另一个光")]
    assert resolve_text("华恋与光站在塔上", name_index(values))["text"] == "爱城华恋与光站在塔上"
    names = name_index([profile("hikari", "神乐光", aliases=["光"])])
    assert resolve_text("光站在塔上", names)["text"] == "神乐光站在塔上"
    assert resolve_text("灯光在塔上", names)["text"] == "灯光在塔上"


def test_lexical_aliases_are_identity_based_and_exclude_ambiguous_nicknames():
    names = name_index([profile("nana", "大场奈奈", aliases=["香蕉", "班长"]),
                        profile("other", "另一角色", aliases=["班长"])])
    short = resolve_text("香蕉", names)
    full = resolve_text("大场奈奈", names)
    assert short["text"] == full["text"] == "大场奈奈"
    assert short["aliases"] == full["aliases"] == ["大场奈奈", "香蕉"]
    assert resolve_text("班长", names)["aliases"] == []


def evidence(identity="v1", text="二人在塔上", **extra):
    return {"id": identity, "record_id": identity, "asset_id": "asset-one", "work_id": "work-one",
            "kind": "visual", "text": text, "start_ms": 0, "end_ms": 1000,
            "evidence_frame_ids": ["frame-one"], **extra}


@pytest.fixture
def stored(tmp_path):
    library = Library(tmp_path / "library")
    asset = {"schema_version": "1.0", "id": "asset-one", "work_id": "work-one", "title": "合成测试",
             "series": "合成剧集", "kind": "animation", "created_at": "2026-11-03", "duration_ms": 2000,
             "version": "test", "fingerprint": "synthetic"}
    atomic_json(library.asset_dir(asset["id"]) / "manifest.json", asset)
    asset = library.get_asset(asset["id"])
    scope = scope_id(asset)
    path = library.root / "works" / scope / "characters.json"
    registry = {"schema_version": "1.0", "scope_id": scope, "revision": 0, "redirects": {},
                "profiles": {"karen": profile("karen", "爱城华恋", scope),
                             "hikari": profile("hikari", "神乐光", scope)}}
    atomic_json(path, registry)
    engine = SearchEngine(library.root / "indexes", ProviderManager(library.root / "private"))
    engine.upsert([evidence()])
    resolver = CharacterQueryResolver(library)
    engine.character_resolver = resolver
    return engine, resolver, path, registry


def test_registry_cache_invalidates_and_scope_filters_are_respected(stored):
    engine, resolver, path, registry = stored
    filters = {"asset_id": "asset-one", "kind": "visual"}
    original = resolver.revision(filters, engine)
    assert resolver.resolve("华恋与光站在塔上", filters, engine)["text"] == "爱城华恋与神乐光站在塔上"
    assert resolver.resolve("华恋", {"asset_id": "different"}, engine)["text"] == "华恋"
    engine.upsert([evidence("s1", "华恋", kind="subtitle")])
    assert resolver.resolve("华恋", {"kind": "subtitle"}, engine)["text"] == "爱城华恋"
    registry["profiles"]["karen"]["name"] = "另一角色"
    registry["revision"] += 1
    atomic_json(path, registry)
    assert resolver.revision(filters, engine) != original
    assert resolver.resolve("华恋", filters, engine)["text"] == "华恋"


@pytest.mark.asyncio
async def test_one_visual_query_source_quote_and_candidate_identity_guard(stored):
    engine, resolver, _path, _registry = stored
    queried = []
    candidate = evidence(entities=[{"id": "person", "candidate_character_id": "karen",
                                   "candidate_character_name": "爱城华恋"}])
    engine.upsert([candidate, evidence("s1", "华恋", kind="subtitle")])
    engine.visual_encoder = object()

    async def visual(query, filters, warnings):
        queried.append(query)
        return [candidate]

    async def decide(_profile, query, candidates):
        assert query == "爱城华恋"
        return SimpleNamespace(data={"candidates": [{"id": row["id"], "score": 1, "match_type": "full"}
                                                      for row in candidates]})

    engine._visual = visual
    engine.providers.decide = decide
    with trace_request() as trace:
        result = await engine.search("华恋", decision_profile_id="test")
    assert queried == ["爱城华恋"]
    assert next(row for row in result["results"] if row["id"] == "s1")["text"] == "华恋"
    hit = next(row for row in result["results"] if row["id"] == "v1")
    assert hit["match_type"] == "partial" and hit["unverified_character_names"] == ["爱城华恋"]
    assert engine.completed_query_text("华恋") == "爱城华恋"
    assert next(span for span in trace.snapshot()["spans"] if span["name"] == "query.character_names")["attributes"] == {
        "matched_count": 1, "changed": True}
    await engine.check_completed_query()


@pytest.mark.asyncio
async def test_roster_change_retries_with_new_name_resolution(stored):
    engine, resolver, path, registry = stored
    engine.visual_encoder = object()
    queries = []

    async def visual(query, filters, warnings):
        queries.append(query)
        if len(queries) == 1:
            registry["profiles"]["karen"]["name"] = "新版华恋"
            registry["revision"] += 1
            atomic_json(path, registry)
        return []

    engine._visual = visual
    await engine.search("华恋")
    assert queries == ["爱城华恋", "新版华恋"]
    assert engine.completed_query_text("华恋") == "新版华恋"
    assert engine.query_diagnostics["retries"] == 1


@pytest.mark.asyncio
async def test_invalid_roster_degrades_without_losing_literal_search(stored):
    engine, _resolver, path, _registry = stored
    path.write_text('{"broken":true}', encoding="utf-8")
    engine.upsert([evidence("s1", "华恋", kind="subtitle")])
    result = await engine.search("华恋")
    assert next(row for row in result["results"] if row["id"] == "s1")["match_type"] == "full"
    assert "角色简称解析不可用，已保留原始查询" in result["degraded"]
    assert engine.completed_query_text("华恋") == "华恋"


def test_deleted_asset_in_old_index_does_not_break_name_resolution(stored):
    engine, resolver, _path, _registry = stored
    engine.upsert([evidence("stale", asset_id="already-deleted")])
    assert resolver.resolve("华恋", {}, engine)["text"] == "爱城华恋"


SHORT_QUERY = "华恋与光站在东京塔上"
FULL_QUERY = "爱城华恋与神乐光站在东京塔上"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, "visual", "subtitle"])
@pytest.mark.parametrize("enhanced", [False, True])
async def test_aliases_have_identical_rankings_even_when_raw_hits_fill_top_fifty(stored, kind, enhanced):
    engine, _resolver, _path, _registry = stored
    record_kind = kind or "visual"
    shorts = [evidence(f"short-{i:02}", SHORT_QUERY, kind=record_kind) for i in range(60)]
    full = evidence("full", FULL_QUERY, kind=record_kind)
    engine.upsert([*shorts, full])
    calls = []
    engine.visual_encoder = object()

    async def visual(query, filters, warnings):
        calls.append(("visual", query))
        return [full, shorts[0]] if record_kind == "visual" else []

    async def interpret(_profile, query):
        calls.append(("interpret", query))
        return SimpleNamespace(data={"search_text": query, "filters": {"kind": record_kind}})

    async def decide(_profile, query, candidates):
        calls.append(("decide", query))
        return SimpleNamespace(data={"candidates": [{"id": row["id"], "score": row["score"],
                                                      "match_type": row["match_type"]} for row in candidates],
                                     "expand_search": True})

    async def compose(_profile, query, candidates):
        calls.append(("compose", query))
        return SimpleNamespace(data={"text": "按现有证据展示候选"})

    engine._visual = visual
    engine.providers.interpret, engine.providers.decide, engine.providers.compose = interpret, decide, compose
    options = {name + "_profile_id": "test" for name in ("query", "decision", "answer")} if enhanced else {}
    filters = {"kind": kind} if kind else {}
    short = await engine.search(SHORT_QUERY, filters, **options)
    full_result = await engine.search(FULL_QUERY, filters, **options)
    assert short == full_result
    assert short["results"][0]["id"] == "full"
    assert all(query == FULL_QUERY for _branch, query in calls)
    assert sum(branch == "visual" for branch, _query in calls) == 2
    if enhanced:
        assert sum(branch == "decide" for branch, _query in calls) == 4  # both ranking batches


@pytest.mark.asyncio
async def test_subtitles_match_aliases_and_partial_words_without_rewriting_dialogue(stored):
    engine, _resolver, _path, _registry = stored
    dialogue = "华恋，光在东京塔上等你。"
    engine.upsert([evidence("s1", dialogue, kind="subtitle"),
                   evidence("s2", "两个人站在东京塔上", kind="subtitle"),
                   evidence("s3", "森林", kind="subtitle")])
    short = await engine.search(SHORT_QUERY, {"kind": "subtitle"})
    full = await engine.search(FULL_QUERY, {"kind": "subtitle"})
    assert short == full
    assert next(row for row in short["results"] if row["id"] == "s1")["text"] == dialogue
    for query in ("东京塔 等你", "京塔"):
        result = await engine.search(query, {"kind": "subtitle"})
        assert next(row for row in result["results"] if row["id"] == "s1")["text"] == dialogue
        assert all(row["id"] != "s3" for row in result["results"])


@pytest.mark.asyncio
async def test_name_change_retries_query_interpretation_without_reusing_old_filters(stored):
    engine, _resolver, path, registry = stored
    calls = []

    async def interpret(_profile, query):
        calls.append(query)
        if len(calls) == 1:
            registry["profiles"]["karen"]["name"] = "新版华恋"
            registry["revision"] += 1
            atomic_json(path, registry)
            filters = {"kind": "subtitle"}
        else:
            filters = {"kind": "visual"}
        return SimpleNamespace(data={"search_text": query, "filters": filters})

    engine.providers.interpret = interpret
    await engine.search("华恋", query_profile_id="test")
    assert calls == ["爱城华恋", "新版华恋"]
    assert engine.completed_query_text("华恋") == "新版华恋"
    assert engine.query_diagnostics["retries"] == 1


@pytest.mark.asyncio
async def test_name_change_during_interpreted_snapshot_acquisition_retries_plan(stored):
    engine, _resolver, path, registry = stored
    engine.upsert([evidence("s1", "新版华恋", kind="subtitle")])
    original_token = engine.query_token
    changed = False
    calls, recalled = [], []

    def token(filters):
        nonlocal changed
        if filters.get("kind") == "subtitle" and not changed:
            changed = True
            registry["profiles"]["karen"]["name"] = "新版华恋"
            registry["revision"] += 1
            atomic_json(path, registry)
        return original_token(filters)

    async def interpret(_profile, query):
        calls.append(query)
        return SimpleNamespace(data={"search_text": query, "filters": {"kind": "subtitle"}})

    async def snapshot(*args):
        recalled.append(args[:3])
        return {"results": [], "degraded": args[7], "explanation": ""}

    engine.query_token = token
    engine.providers.interpret = interpret
    engine._search_snapshot = snapshot
    await engine.search("华恋", query_profile_id="test")
    assert calls == ["爱城华恋", "新版华恋"]
    assert recalled == [("新版华恋", "新版华恋", {"kind": "subtitle"})]
    assert engine.query_diagnostics["retries"] == 1


class ScopeQueryResolver:
    """Simulate a shorthand resolving differently after a scope restriction."""

    def revision(self, filters, _engine):
        return ("names", filters.get("kind", "all"))

    def resolve(self, query, filters, _engine):
        scope = filters.get("kind", "all")
        name = {"all": "原范围华恋", "subtitle": "字幕范围华恋", "visual": "画面范围华恋"}[scope]
        text = name if query == "华恋" else query
        return {"text": text, "aliases": [scope + "-alias"], "matches": [{}], "changed": text != query}


@pytest.mark.asyncio
async def test_interpreted_scope_change_replans_using_new_character_and_aliases(stored):
    engine, _resolver, _path, _registry = stored
    engine.character_resolver = ScopeQueryResolver()
    calls, recalled = [], []

    async def interpret(_profile, query):
        calls.append(query)
        return SimpleNamespace(data={"search_text": query + "在东京塔", "filters": {"kind": "subtitle"}})

    async def snapshot(*args):
        recalled.append((args[0], args[1], args[2], args[10]))
        return {"results": [], "degraded": args[7], "explanation": ""}

    engine.providers.interpret = interpret
    engine._search_snapshot = snapshot
    await engine.search("华恋", query_profile_id="test")
    assert calls == ["原范围华恋", "字幕范围华恋"]
    assert recalled == [("字幕范围华恋", "字幕范围华恋在东京塔", {"kind": "subtitle"}, ("subtitle-alias",))]
    assert engine.completed_query_text("华恋") == "字幕范围华恋"
    assert engine.query_diagnostics["retries"] == 0


@pytest.mark.asyncio
async def test_repeated_interpretation_scope_changes_fall_back_to_user_scope(stored):
    engine, _resolver, _path, _registry = stored
    engine.character_resolver = ScopeQueryResolver()
    calls, recalled = [], []

    async def interpret(_profile, query):
        calls.append(query)
        kind = "subtitle" if len(calls) == 1 else "visual"
        return SimpleNamespace(data={"search_text": query + "旧解释", "filters": {"kind": kind}})

    async def snapshot(*args):
        recalled.append((args[0], args[1], args[2], args[10]))
        return {"results": [], "degraded": args[7], "explanation": ""}

    engine.providers.interpret = interpret
    engine._search_snapshot = snapshot
    result = await engine.search("华恋", query_profile_id="test")
    assert calls == ["原范围华恋", "字幕范围华恋"]
    assert recalled == [("原范围华恋", "原范围华恋", {}, ("all-alias",))]
    assert engine.completed_query_text("华恋") == "原范围华恋"
    assert "查询理解反复改变角色范围，已使用原筛选与本地角色解析" in result["degraded"]

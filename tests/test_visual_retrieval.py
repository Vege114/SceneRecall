import copy
import hashlib

import pytest
from PIL import Image

from scenerecall.providers import ProviderManager
from scenerecall.search import SearchEngine, merge_visual_candidates
from scenerecall.visual_index import evidence_digest


class Encoder:
    batch_size = 16

    def __init__(self, *, model="fixture-a", vectors=None, query=None):
        self.model = model
        self.vectors = vectors or {}
        self.query = query or [1., 0., 0.]
        self.image_calls = []
        self.text_calls = []
        self.on_images = None
        self.on_texts = None
        self.fail_call = None

    def identity(self):
        return {"model": self.model, "weights_sha256": "fixture-weights", "preprocessing": "fixture-rgb-v1"}

    async def encode_images(self, frames):
        self.image_calls.append([frame["id"] for frame in frames])
        if self.on_images:
            self.on_images()
        if self.fail_call == len(self.image_calls):
            raise RuntimeError("fixture encoder failed")
        return [self.vectors.get(frame["id"], [1., 0., 0.]) for frame in frames]

    async def encode_texts(self, texts):
        self.text_calls.append(texts)
        if self.on_texts:
            self.on_texts()
        return [self.query for _ in texts]


@pytest.fixture
def engine(tmp_path):
    return SearchEngine(tmp_path / "index", ProviderManager(tmp_path / "settings"))


def evidence(tmp_path, name, *, asset="asset-one", at=200, text="本地画面", extra=None):
    path = tmp_path / (name + ".png")
    Image.new("RGB", (12, 8), color=(len(name) * 17 % 255, at % 255, 50)).save(path)
    frame = {"id": "frame-" + name, "at_ms": at, "pts": at, "time_base": [1, 1000],
             "path": str(path)}
    record = {"id": "window-" + name, "asset_id": asset, "kind": "visual",
              "evidence_type": "retrieval_window", "source_fingerprint": "source-" + asset,
              "start_ms": at // 1000 * 1000, "end_ms": at // 1000 * 1000 + 1000,
              "text": text, "frame_ids": [frame["id"]], "evidence_frame_ids": [frame["id"]],
              "ordered_frames": [{key: frame[key] for key in ("id", "at_ms", "pts", "time_base")}],
              **(extra or {})}
    return record, frame


@pytest.mark.asyncio
async def test_visual_recall_uses_pixels_original_query_and_max_frame_with_source_seek(engine, tmp_path):
    record, first = evidence(tmp_path, "one", at=100)
    _, second = evidence(tmp_path, "two", at=700)
    record["frame_ids"].append(second["id"])
    record["evidence_frame_ids"].append(second["id"])
    record["ordered_frames"].append({key: second[key] for key in ("id", "at_ms", "pts", "time_base")})
    before = copy.deepcopy(record)
    encoder = Encoder(vectors={first["id"]: [0., 1., 0.], second["id"]: [2., 0., 0.]})
    engine.visual_encoder = encoder
    built = await engine.index_visual([record], [first, second])
    result = await engine.search("甲没有把红杯递给乙")
    assert built["records"] == 1 and built["encoded_frames"] == 2
    assert len(result["results"]) == 1
    hit = result["results"][0]
    assert hit["id"] == record["id"] and hit["hit_frame_id"] == second["id"]
    assert hit["seek_ms"] == 700 and hit["evidence_ids"] == [second["id"]]
    assert hit["match_type"] == "partial" and hit["visual_similarity"] == pytest.approx(1)
    assert encoder.text_calls == [["甲没有把红杯递给乙"]]
    assert record == before
    await engine.search("甲没有把红杯递给乙")
    assert len(encoder.text_calls) == 1


@pytest.mark.asyncio
async def test_visual_and_fts_rankings_fuse_without_duplicate_windows(engine, tmp_path):
    a, fa = evidence(tmp_path, "a", text="红色的杯子", at=200)
    b, fb = evidence(tmp_path, "b", text="绿色树林", at=1200)
    engine.visual_encoder = Encoder(vectors={fa["id"]: [0.8, 0.6, 0], fb["id"]: [1, 0, 0]})
    await engine.index_visual([a, b], [fa, fb])
    result = await engine.search("杯子")
    assert [row["id"] for row in result["results"]] == [a["id"], b["id"]]
    assert result["results"][0]["score"] > result["results"][1]["score"]
    assert all(row["match_type"] == "partial" for row in result["results"])


@pytest.mark.asyncio
async def test_model_preprocessing_and_query_spaces_are_isolated(engine, tmp_path):
    row, frame = evidence(tmp_path, "model")
    first = Encoder()
    engine.visual_encoder = first
    await engine.index_visual([row], [frame])
    assert (await engine.search("海边"))["results"]
    second = Encoder(model="fixture-b", query=[0, 1, 0], vectors={frame["id"]: [0, 1, 0]})
    engine.visual_encoder = second
    result = await engine.search("海边")
    assert not result["results"] and any("尚未建立视觉索引" in text for text in result["degraded"])
    await engine.index_visual([row], [frame])
    assert (await engine.search("海边"))["results"][0]["id"] == row["id"]
    assert len(second.image_calls) == 1 and len(second.text_calls) == 1
    assert len(engine.status()["visual_namespaces"]) == 2


@pytest.mark.asyncio
async def test_visual_cache_ignores_display_names_but_tracks_pixels_region_source_and_time(engine, tmp_path):
    row, frame = evidence(tmp_path, "cache")
    encoder = Encoder()
    engine.visual_encoder = encoder
    await engine.index_visual([row], [frame])
    renamed = {**row, "text": "人工修正后的名字", "aliases": ["别名"]}
    cached = await engine.index_visual([renamed], [frame])
    assert cached["cache_hits"] == 1 and cached["encoded_frames"] == 0
    cropped = {**frame, "bbox": [0, 0, 0.5, 1]}
    assert (await engine.index_visual([renamed], [cropped]))["encoded_frames"] == 1
    Image.new("RGB", (12, 8), color="red").save(frame["path"])
    assert (await engine.index_visual([renamed], [cropped]))["encoded_frames"] == 1
    changed_source = {**renamed, "source_fingerprint": "different-source"}
    assert (await engine.index_visual([changed_source], [cropped]))["encoded_frames"] == 1
    assert len(encoder.image_calls) == 4


@pytest.mark.asyncio
async def test_incremental_publication_and_failure_resume_keep_completed_windows(engine, tmp_path):
    one, first = evidence(tmp_path, "first", at=200)
    two, second = evidence(tmp_path, "second", at=1200)
    encoder = Encoder()
    encoder.batch_size, encoder.fail_call = 1, 2
    engine.visual_encoder = encoder
    progress = []
    with pytest.raises(RuntimeError, match="fixture encoder"):
        await engine.index_visual([one, two], [first, second], progress=progress.append)
    assert [row["id"] for row in (await engine.search("海边"))["results"]] == [one["id"]]
    assert progress[-1]["published_records"] == 1
    encoder.fail_call = None
    done = await engine.index_visual([one, two], [first, second])
    assert done["cache_hits"] == 1 and done["encoded_frames"] == 1 and done["records"] == 2
    assert encoder.image_calls == [[first["id"]], [second["id"]], [second["id"]]]


@pytest.mark.asyncio
async def test_delete_during_encoding_refuses_stale_publication(engine, tmp_path):
    row, frame = evidence(tmp_path, "deleted")
    engine.upsert([row])
    encoder = Encoder()
    encoder.on_images = lambda: engine.remove_assets([row["asset_id"]])
    engine.visual_encoder = encoder
    with pytest.raises(ValueError, match="资料发生变化"):
        await engine.index_visual([row], [frame])
    assert engine.status()["records"] == 0
    assert not (await engine.search("海边"))["results"]


@pytest.mark.asyncio
async def test_filtering_precedes_visual_top50_and_deletion_removes_only_owned_manifest(engine, tmp_path):
    items = [evidence(tmp_path, f"row-{index}", at=index * 1000 + 100) for index in range(55)]
    selected, selected_frame = evidence(tmp_path, "selected", asset="small-work",
                                       extra={"project_id": "project", "tags": ["tagged"],
                                              "entities": [{"id": "p1", "name": "小明", "aliases": ["阿明"]}]})
    encoder = Encoder(vectors={selected_frame["id"]: [0.5, 0.5, 0.]})
    engine.visual_encoder = encoder
    await engine.index_visual([row for row, _ in items], [frame for _, frame in items])
    await engine.index_visual([selected], [selected_frame])
    result = await engine.search("海边", {"project_id": "project", "character": "阿明", "tag": "tagged"})
    assert [row["id"] for row in result["results"]] == [selected["id"]]
    engine.remove_assets(["asset-one"])
    assert engine.status()["visual_namespaces"][0]["records"] == 1
    assert [row["id"] for row in (await engine.search("海边"))["results"]] == [selected["id"]]


@pytest.mark.asyncio
async def test_updated_evidence_skips_stale_vectors_while_name_projection_remains_current(engine, tmp_path):
    row, frame = evidence(tmp_path, "updated")
    engine.visual_encoder = Encoder()
    await engine.index_visual([row], [frame])
    changed = {**row, "text": "最新的人物名"}
    engine.upsert([changed])
    assert (await engine.search("海边"))["results"][0]["text"] == "最新的人物名"
    engine.upsert([{**changed, "source_fingerprint": "new-source"}])
    assert not (await engine.search("海边"))["results"]


@pytest.mark.asyncio
async def test_source_edit_or_deletion_during_query_does_not_return_old_snapshot(engine, tmp_path):
    row, frame = evidence(tmp_path, "query-race")
    encoder = Encoder()
    engine.visual_encoder = encoder
    await engine.index_visual([row], [frame])
    encoder.on_texts = lambda: engine.upsert([{**row, "source_fingerprint": "new-source"}])
    assert not (await engine.search("另一处海边"))["results"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["timestamp", "missing-frame", "nan-vector", "dimensions"])
async def test_invalid_evidence_or_vectors_never_publish(engine, tmp_path, invalid):
    row, frame = evidence(tmp_path, "invalid")
    encoder = Encoder()
    if invalid == "timestamp":
        row["ordered_frames"][0]["at_ms"] += 1
    elif invalid == "missing-frame":
        frame["id"] = "missing"
    elif invalid == "nan-vector":
        encoder.vectors[frame["id"]] = [float("nan"), 0, 0]
    else:
        encoder.vectors[frame["id"]] = []
    with pytest.raises(ValueError):
        await engine.index_visual([row], [frame], encoder=encoder)
    assert engine.status()["records"] == 0


@pytest.mark.asyncio
async def test_visual_index_and_query_cache_survive_restart(engine, tmp_path):
    row, frame = evidence(tmp_path, "restart")
    engine.visual_encoder = Encoder()
    await engine.index_visual([row], [frame])
    await engine.search("海边")
    reopened = SearchEngine(engine.index_dir, engine.providers)
    reopened.visual_encoder = Encoder()
    assert (await reopened.search("海边"))["results"][0]["id"] == row["id"]
    assert not reopened.visual_encoder.text_calls
    rebuilt = await reopened.index_visual([row], [frame])
    assert rebuilt["encoded_frames"] == 0 and not reopened.visual_encoder.image_calls


def reference(tmp_path, name, concept, *, color=40, status="confirmed"):
    path = tmp_path / (name + "-reference.png")
    Image.new("RGB", (12, 8), color=(color, 100, 90)).save(path)
    return {"id": name, "concept_id": concept, "asset_id": "work-scope", "path": str(path),
            "source_fingerprint": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pts": 0, "time_base": [1, 1], "status": status}


@pytest.mark.asyncio
async def test_picture_selected_concepts_reuse_current_features_and_do_not_bind_observations(engine, tmp_path):
    row, first = evidence(tmp_path, "reference-window", at=100)
    _, second = evidence(tmp_path, "reference-last", at=700)
    row["frame_ids"].append(second["id"])
    row["evidence_frame_ids"].append(second["id"])
    row["ordered_frames"].append({key: second[key] for key in ("id", "at_ms", "pts", "time_base")})
    clock = reference(tmp_path, "clock", "concept-clock")
    room = reference(tmp_path, "room", "concept-room", color=80)
    candidate = reference(tmp_path, "candidate", "concept-untrusted", color=100, status="candidate")
    encoder = Encoder(vectors={first["id"]: [0, 1, 0], second["id"]: [1, 0, 0],
                               clock["id"]: [1, 0, 0], room["id"]: [0, 0, 1]})
    engine.visual_encoder = encoder
    before = copy.deepcopy(row)
    await engine.index_visual([row], [first, second])
    matched = await engine.match_visual_references([row], [clock, room, candidate])
    assert matched == {row["id"]: [{"concept_id": "concept-clock", "similarity": pytest.approx(1.)}]}
    assert encoder.image_calls == [[first["id"], second["id"]], ["clock", "room"]]
    assert not encoder.text_calls and row == before
    assert "concept_bindings" not in engine.filtered_records()[0]


@pytest.mark.asyncio
async def test_reference_features_cache_only_missing_images_and_model_spaces_stay_isolated(engine, tmp_path):
    row, frame = evidence(tmp_path, "reference-cache")
    first = reference(tmp_path, "ref-first", "concept-first")
    second = reference(tmp_path, "ref-second", "concept-second", color=180)
    encoder = Encoder()
    engine.visual_encoder = encoder
    await engine.index_visual([row], [frame])
    await engine.match_visual_references([row], [first])
    await engine.match_visual_references([row], [first])
    await engine.match_visual_references([row], [first, second])
    assert encoder.image_calls == [[frame["id"]], ["ref-first"], ["ref-second"]]
    other = Encoder(model="different-space")
    assert await engine.match_visual_references([row], [first], encoder=other) == {}
    assert not other.image_calls
    await engine.index_visual([row], [frame], encoder=other)
    await engine.match_visual_references([row], [first], encoder=other)
    assert other.image_calls == [[frame["id"]], ["ref-first"]]


@pytest.mark.asyncio
async def test_picture_reference_candidates_are_unique_concepts_and_bounded_to_five(engine, tmp_path):
    row, frame = evidence(tmp_path, "five-concepts")
    refs = [reference(tmp_path, f"candidate-{index}", f"concept-{index}", color=index * 20 + 20)
            for index in range(7)]
    refs.append(reference(tmp_path, "same-concept-another-view", "concept-0", color=230))
    engine.visual_encoder = Encoder()
    await engine.index_visual([row], [frame])
    result = (await engine.match_visual_references([row], refs))[row["id"]]
    assert len(result) == 5
    assert [item["concept_id"] for item in result] == [f"concept-{index}" for index in range(5)]
    assert all(set(item) == {"concept_id", "similarity"} for item in result)


@pytest.mark.asyncio
async def test_reference_sha_and_region_are_part_of_cache_identity(engine, tmp_path):
    row, frame = evidence(tmp_path, "ref-pixels")
    ref = reference(tmp_path, "ref-checked", "concept-checked")
    encoder = Encoder()
    engine.visual_encoder = encoder
    await engine.index_visual([row], [frame])
    await engine.match_visual_references([row], [ref])
    await engine.match_visual_references([row], [{**ref, "box": [0, 0, 0.5, 1]}])
    assert encoder.image_calls == [[frame["id"]], [ref["id"]], [ref["id"]]]
    Image.new("RGB", (12, 8), color="blue").save(ref["path"])
    with pytest.raises(ValueError, match="SHA256"):
        await engine.match_visual_references([row], [ref])


@pytest.mark.asyncio
async def test_deleted_or_changed_current_frames_cannot_select_reference_dossiers(engine, tmp_path):
    row, frame = evidence(tmp_path, "ref-stale")
    ref = reference(tmp_path, "ref-deleted", "concept-deleted")
    encoder = Encoder()
    engine.visual_encoder = encoder
    await engine.index_visual([row], [frame])
    await engine.match_visual_references([row], [ref])
    engine.upsert([{**row, "source_fingerprint": "replacement-source"}])
    assert await engine.match_visual_references([row], [ref]) == {}
    engine.remove_assets([row["asset_id"]])
    assert await engine.match_visual_references([row], [ref]) == {}


@pytest.mark.asyncio
async def test_held_frame_keeps_original_pts_and_explicit_span_in_evidence(engine, tmp_path):
    row, frame = evidence(tmp_path, "held-frame", at=200)
    row.update(start_ms=6000, end_ms=12000,
               held_frame_spans=[{"frame_id": frame["id"], "start_ms": 6000, "end_ms": 12000}])
    original = copy.deepcopy(row)
    engine.visual_encoder = Encoder()
    await engine.index_visual([row], [frame])
    hit = (await engine.search("海边"))["results"][0]
    assert hit["seek_ms"] == 200 and hit["hit_frame_id"] == frame["id"]
    assert row == original and row["ordered_frames"][0]["pts"] == 200
    changed = copy.deepcopy(row)
    changed["held_frame_spans"][0]["end_ms"] = 13000
    assert evidence_digest(changed) != evidence_digest(row)
    engine.upsert([changed])
    assert not (await engine.search("海边"))["results"]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["missing", "incomplete", "wrong-frame", "future-frame", "multiple"])
async def test_held_frame_requires_complete_single_previous_frame_declaration(engine, tmp_path, invalid):
    row, frame = evidence(tmp_path, "held-invalid", at=200)
    row.update(start_ms=6000, end_ms=12000,
               held_frame_spans=[{"frame_id": frame["id"], "start_ms": 6000, "end_ms": 12000}])
    frames = [frame]
    if invalid == "missing":
        row.pop("held_frame_spans")
    elif invalid == "incomplete":
        row["held_frame_spans"][0]["end_ms"] = 11999
    elif invalid == "wrong-frame":
        row["held_frame_spans"][0]["frame_id"] = "another-frame"
    elif invalid == "future-frame":
        frame["at_ms"] = frame["pts"] = 13000
        row["ordered_frames"][0].update(at_ms=13000, pts=13000)
    elif invalid == "multiple":
        _, second = evidence(tmp_path, "held-second", at=300)
        frames.append(second)
        row["frame_ids"].append(second["id"])
        row["evidence_frame_ids"].append(second["id"])
        row["ordered_frames"].append({key: second[key] for key in ("id", "at_ms", "pts", "time_base")})
    with pytest.raises(ValueError):
        await engine.index_visual([row], frames, encoder=Encoder())
    assert engine.status()["records"] == 0


def test_repeated_held_frame_merges_preserving_representative_and_source_spans(tmp_path):
    first, frame = evidence(tmp_path, "merge-held", at=200)
    first.update(start_ms=0, end_ms=6000, hit_frame_id=frame["id"], seek_ms=200, score=.9)
    second = {**copy.deepcopy(first), "id": "held-next-window", "start_ms": 6000, "end_ms": 12000, "score": .5}
    original = copy.deepcopy([first, second])
    merged = merge_visual_candidates([first, second])
    assert len(merged) == 1 and merged[0]["id"] == first["id"] and merged[0]["score"] == .9
    assert merged[0]["seek_ms"] == 200 and merged[0]["end_ms"] == 12000
    assert merged[0]["member_ids"] == [first["id"], second["id"]]
    assert merged[0]["source_spans"] == [
        {"record_id": first["id"], "start_ms": 0, "end_ms": 6000},
        {"record_id": second["id"], "start_ms": 6000, "end_ms": 12000}]
    assert merged[0]["ordered_frames"] == first["ordered_frames"]
    assert merged[0]["frame_ids"] == [frame["id"]] and merged[0]["merge_basis"] == "same_hit_frame"
    assert [first, second] == original


def event_candidate(tmp_path, name, start, *, actor="alice", target="bob", action="递给", polarity="positive"):
    row, frame = evidence(tmp_path, name, at=start + 200)
    row.update(entities=[{"id": name + "-p1", "character_id": actor, "association_status": "confirmed"},
                         {"id": name + "-p2", "character_id": target, "association_status": "confirmed"},
                         {"id": name + "-o1"}],
               concept_bindings=[{"entity_id": name + "-o1", "concept_id": "cup", "status": "confirmed"}],
               events=[{"action": action, "actor_ids": [name + "-p1"], "recipient_id": name + "-p2",
                        "object_id": name + "-o1", "polarity": polarity, "temporal": True,
                        "evidence_frame_ids": [frame["id"]]}])
    return row


def test_same_event_merges_only_with_stable_roles_and_preserves_ordered_evidence(tmp_path):
    first = event_candidate(tmp_path, "event-first", 0)
    second = event_candidate(tmp_path, "event-second", 1000)
    result = merge_visual_candidates([second, first])
    assert len(result) == 1 and result[0]["id"] == second["id"]
    assert result[0]["merge_basis"] == "same_stable_event"
    assert result[0]["member_ids"] == [first["id"], second["id"]]
    assert result[0]["frame_ids"] == first["frame_ids"] + second["frame_ids"]
    assert result[0]["events"] == second["events"]  # Never manufacture continuity.


@pytest.mark.parametrize("difference", ["action", "polarity", "roles", "anonymous", "work", "kind"])
def test_different_or_unresolved_events_never_merge(tmp_path, difference):
    first = event_candidate(tmp_path, "different-first", 0)
    second = event_candidate(tmp_path, "different-second", 1000)
    if difference == "action":
        second["events"][0]["action"] = "拿回"
    elif difference == "polarity":
        second["events"][0]["polarity"] = "negative"
    elif difference == "roles":
        second["entities"][0]["character_id"], second["entities"][1]["character_id"] = "bob", "alice"
    elif difference == "anonymous":
        second["entities"][1]["character_id"] = None
    elif difference == "work":
        second["asset_id"] = "another-work"
    elif difference == "kind":
        second["kind"] = "scene"
    assert len(merge_visual_candidates([first, second])) == 2


@pytest.mark.parametrize("start,end", [(2001, 3001), (1000, 12001)])
def test_visual_merge_rejects_long_gap_or_range(tmp_path, start, end):
    first = event_candidate(tmp_path, "range-first", 0)
    second = event_candidate(tmp_path, "range-second", 1000)
    second.update(start_ms=start, end_ms=end)
    assert len(merge_visual_candidates([first, second])) == 2


@pytest.mark.asyncio
async def test_search_merges_repeated_source_frame_without_counting_duplicate_windows(engine, tmp_path):
    first, frame = evidence(tmp_path, "search-merge-held", at=200)
    first.update(start_ms=0, end_ms=6000)
    second = {**copy.deepcopy(first), "id": "search-held-next", "start_ms": 6000, "end_ms": 12000,
              "held_frame_spans": [{"frame_id": frame["id"], "start_ms": 6000, "end_ms": 12000}]}
    engine.visual_encoder = Encoder()
    await engine.index_visual([first, second], [frame])
    result = await engine.search("海边")
    assert len(result["results"]) == 1
    assert result["results"][0]["member_ids"] == [first["id"], second["id"]]
    assert result["results"][0]["match_type"] == "partial"

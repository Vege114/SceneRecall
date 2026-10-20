"""Bounded query review uses real image serialization and validation, no model."""
import asyncio
import json
import time
from contextlib import nullcontext
from copy import deepcopy

import pytest

from scenerecall.providers import ProviderManager
from scenerecall.library import atomic_json, read_json
from scenerecall.retrieval_verification import verify_candidates
from test_scene_jobs import SyntheticSceneProvider
from test_visual_identity import frame, visual  # noqa: F401


class VerificationTransport(SyntheticSceneProvider):
    _images = staticmethod(ProviderManager._images)

    def __init__(self, *, delay=0, mutate=None):
        super().__init__()
        self.delay = delay
        self.mutate = mutate
        self.cancelled = 0

    def use_profile(self, _profile):
        return nullcontext()

    async def _chat(self, profile, capability, instruction, content, validator, **options):
        payload = json.loads(content[0]["text"])
        self.calls.append({"profile": profile, "instruction": instruction, "content": deepcopy(content),
                           "payload": payload, "options": options})
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        judgments = [{"id": candidate["id"], "match_type": "full", "missing_constraints": [],
                      "evidence_frame_ids": [f["id"] for f in candidate["ordered_frames"]]}
                     for candidate in payload["candidates"]]
        if self.mutate:
            self.mutate(judgments)
        return self._result(validator({"candidates": judgments}))


@pytest.fixture
def verification_input(visual):  # noqa: F811
    library, _, _ = visual
    results = []
    for number in range(4):
        frames = []
        for offset in range(8):
            timestamp = number * 2000 + offset * 200
            item = frame(library, f"frame_{number}_{offset}", at_ms=timestamp)
            frames.append({key: item[key] for key in ("id", "at_ms")})
        results.append({"id": f"unit_{number}", "asset_id": "asset_one", "source_fingerprint": "synthetic-video",
            "start_ms": number * 2000, "end_ms": number * 2000 + 2000, "ordered_frames": frames,
            "entities": [], "events": [], "subtitle_text": "", "summary": "合成蓝色背景", "match_type": "candidate"})
    return library, {"results": results, "degraded": [], "explanation": "尚未视觉核对"}


@pytest.mark.asyncio
async def test_one_request_bounded_candidates_images_and_success_cache(verification_input):
    library, response = verification_input
    original = deepcopy(response)
    provider = VerificationTransport()
    question = "阿明没有把红色杯子递给小雪，而是递给阿兰"
    first = await verify_candidates(library, provider, "vision", question, response)
    assert response == original
    assert first["verification"]["status"] == "completed"
    assert first["verification"]["request_count"] == 1
    assert first["verification"]["candidate_count"] == 3
    call = provider.calls[0]
    assert call["payload"]["original_query"] == question
    assert call["options"]["validation_attempts"] == 1
    assert call["options"]["request_attempts"] == 1
    assert len(call["payload"]["candidates"]) == 3
    assert sum(item["type"] == "image_url" for item in call["content"]) == 12
    assert first["results"][3]["match_type"] == "candidate"
    second = await verify_candidates(library, provider, "vision", question, response)
    assert second["verification"]["status"] == "cache_hit"
    assert second["verification"]["request_count"] == 0
    assert len(provider.calls) == 1
    # A change to negation must not share the cached judgment.
    await verify_candidates(library, provider, "vision", "阿明把红色杯子递给小雪", response)
    assert len(provider.calls) == 2

    # The source pixels can stay identical while a setting binding is revoked.
    response["results"][0]["concept_bindings"] = [{"concept_id": "prop_one", "status": "revoked", "revision": 2}]
    changed = await verify_candidates(library, provider, "vision", question, response)
    assert changed["verification"]["status"] == "completed"
    assert len(provider.calls) == 3


@pytest.mark.asyncio
async def test_single_request_deadline_cancels_transport_and_never_caches_timeout(verification_input):
    library, response = verification_input
    provider = VerificationTransport(delay=10)
    started = time.monotonic()
    result = await verify_candidates(library, provider, "vision", "蓝色天空", response, deadline_s=.05)
    assert time.monotonic() - started < 1.5
    assert result["verification"]["status"] == "timeout"
    assert result["verification"]["deadline_ms"] == 50
    assert result["results"] == response["results"]
    assert provider.cancelled == 1 and len(provider.calls) == 1
    assert not list((library.root / "cache" / "retrieval-verification").glob("*.json"))
    provider.delay = 0
    resumed = await verify_candidates(library, provider, "vision", "蓝色天空", response, deadline_s=.05)
    assert resumed["verification"]["status"] == "completed"
    assert len(provider.calls) == 2


@pytest.mark.asyncio
async def test_invalid_cross_candidate_evidence_fails_once_and_preserves_originals(verification_input):
    library, response = verification_input

    def wrong_evidence(judgments):
        judgments[0]["evidence_frame_ids"] = judgments[1]["evidence_frame_ids"]

    provider = VerificationTransport(mutate=wrong_evidence)
    result = await verify_candidates(library, provider, "vision", "谁正在转身", response)
    assert result["verification"]["status"] == "failed"
    assert result["results"] == response["results"] and len(provider.calls) == 1
    assert not list((library.root / "cache" / "retrieval-verification").glob("*.json"))


@pytest.mark.asyncio
async def test_temporal_full_needs_distinct_ordered_source_times(verification_input):
    library, response = verification_input

    def insufficient(judgments):
        judgments[0]["evidence_frame_ids"] = judgments[0]["evidence_frame_ids"][:1]
        judgments[1]["evidence_frame_ids"].reverse()

    result = await verify_candidates(library, VerificationTransport(mutate=insufficient), "vision",
                                    "人物先转身再把红色杯子递给另一人", response)
    assert result["results"][0]["match_type"] == "partial"
    assert result["results"][1]["match_type"] == "partial"
    assert result["results"][2]["match_type"] == "full"


@pytest.mark.asyncio
async def test_vfr_held_source_frame_shared_by_windows_is_serialized_only_once(verification_input):
    library, response = verification_input
    held = response["results"][0]["ordered_frames"][:1]
    response["results"] = [
        {**response["results"][number], "ordered_frames": deepcopy(held)} for number in range(3)
    ]
    provider = VerificationTransport()
    result = await verify_candidates(library, provider, "vision", "蓝色天空", response)
    assert result["verification"]["status"] == "completed"
    assert len(provider.calls) == 1
    assert sum(item["type"] == "image_url" for item in provider.calls[0]["content"]) == 1
    assert len(result["results"]) == 3


@pytest.mark.asyncio
async def test_unavailable_evidence_degrades_review_without_failing_search(verification_input):
    library, response = verification_input
    missing = response["results"][0]["ordered_frames"][0]["id"]
    library.frame_path("asset_one", missing).unlink()
    provider = VerificationTransport()
    result = await verify_candidates(library, provider, "vision", "蓝色天空", response)
    assert result["verification"]["status"] in {"failed", "unavailable"}
    assert result["results"] == response["results"] and provider.calls == []


@pytest.mark.asyncio
async def test_corrupt_cached_judgment_cannot_promote_unprovided_frame_evidence(verification_input):
    library, response = verification_input
    provider = VerificationTransport()
    first = await verify_candidates(library, provider, "vision", "蓝色天空", response)
    assert first["verification"]["status"] == "completed"
    cache = next((library.root / "cache" / "retrieval-verification").glob("*.json"))
    cached = read_json(cache)
    cached["data"]["candidates"][0]["evidence_frame_ids"] = ["not_a_provided_source_frame"]
    atomic_json(cache, cached)
    result = await verify_candidates(library, provider, "vision", "蓝色天空", response)
    assert result["verification"]["status"] in {"failed", "unavailable"}
    assert result["results"] == response["results"]
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_expired_image_preparation_never_submits_a_model_request(verification_input, monkeypatch):
    library, response = verification_input
    provider = VerificationTransport()
    serialize = provider._images

    def slow_images(images):
        time.sleep(.25)
        return serialize(images)

    monkeypatch.setattr(provider, "_images", slow_images)
    result = await verify_candidates(library, provider, "vision", "蓝色天空", response, deadline_s=.02)
    assert result["verification"]["status"] == "timeout"
    assert provider.calls == []
    assert result["results"] == response["results"]

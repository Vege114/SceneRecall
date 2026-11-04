"""Threshold selection survives canonical HTTP enrichment without a result cap."""
import json

import httpx
import pytest
from pydantic import ValidationError

from scenerecall.library import atomic_jsonl
from scenerecall.main import create_app
from scenerecall.models import SearchInput
from test_storage_security import ASSET_ID, seed_asset


@pytest.mark.parametrize("value", [-1, 101, float('nan'), float('inf'), -float('inf')])
def test_threshold_contract_rejects_invalid_numbers(value):
    with pytest.raises(ValidationError):
        SearchInput(query="threshold", min_match_score=value)


@pytest.mark.asyncio
async def test_threshold_api_keeps_all_eligible_sources_and_applied_selection(tmp_path):
    app = create_app(tmp_path / 'library', start_worker=False, token='threshold-test-token')
    library, engine = app.state.library, app.state.search
    engine.visual_encoder = None
    seed_asset(library)
    cues = [{"id": f"cue_{index:03d}", "start_ms": index * 10000, "end_ms": index * 10000 + 1000,
             "text": f"thresholdneedle {index:03d}", "language": "en", "source": "embedded",
             "review_status": "confirmed"} for index in range(65)]
    atomic_jsonl(library.asset_dir(ASSET_ID) / 'subtitles/external.jsonl', cues)
    engine.sync_asset(library.records(ASSET_ID))
    headers = {"Authorization": "Bearer threshold-test-token"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://testserver') as client:
        response = await client.post('/api/search', headers=headers, json={
            "query": "thresholdneedle", "filters": {"kind": "subtitle"}, "min_match_score": 0, "limit": 1})
        assert response.status_code == 200, response.text
        result = response.json()
        assert len(result['results']) == 65
        assert {row['record_id'] for row in result['results']} == {row['id'] for row in cues}
        assert all(row['match_score'] > 0 for row in result['results'])
        assert result['results'][0]['match_score'] == 100
        order = [(-row['match_score'], row['id']) for row in result['results']]
        assert order == sorted(order)
        assert result['selection']['min_match_score'] == 0
        assert result['selection']['returned_count'] == 65
        assert result['selection']['score_basis'] == 'relative_rrf_v1'
        trace = json.loads(response.headers['X-Search-Timing'])
        assert trace['attributes']['min_match_score'] == 0
        assert trace['attributes']['result_count'] == 65
        # Strictly above 100 excludes even the best candidate.
        empty = await client.post('/api/search', headers=headers, json={
            "query": "thresholdneedle", "filters": {"kind": "subtitle"}, "min_match_score": 100})
        assert empty.status_code == 200, empty.text
        assert empty.json()['results'] == []
        assert empty.json()['selection']['returned_count'] == 0
        # The explicit legacy count interface remains compatible for other clients.
        legacy = await client.post('/api/search', headers=headers, json={
            "query": "thresholdneedle", "filters": {"kind": "subtitle"}, "limit": 3})
        assert legacy.status_code == 200, legacy.text
        assert len(legacy.json()['results']) == 3

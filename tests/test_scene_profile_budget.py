from copy import deepcopy
import json

import pytest

from scenerecall.characters import BANGUMI_REFERENCE_POLICY
from scenerecall.providers import ProviderManager
from scenerecall.scene_jobs import SceneJobMixin, canonical, encoded_size


def profile(index, *, description="参考背景" * 300):
    source = {"source": "bangumi", "subject_id": 214265, "character_id": index + 1,
              "url": f"https://bgm.tv/character/{index + 1}", "fetched_at": "2026-10-18T00:00:00Z",
              "image_url": f"https://lain.bgm.tv/pic/crt/l/character_{index + 1}.jpg"}
    aliases = [f"角色{index}别名{number}" for number in range(5)]
    return {"id": f"character_{index:03d}", "name": f"角色{index}", "aliases": aliases,
            "description": description, "reference_description": description,
            "notes": "人工备注" * 100, "appearance": ["短发，白色校服" * 30] * 4,
            "observed_traits": ["短发，白色校服" * 30] * 4,
            "external_source": source, "external_sources": [],
            "reference_policy": BANGUMI_REFERENCE_POLICY, "reference_only": True,
            "identity_status": "unverified_prior", "presence_verified": False,
            "name_variants": [{"name": alias, "language": "ja", "kind": "alias", "source": "bangumi",
                               "source_url": source["url"]} for alias in aliases],
            "created_at": "2026-10-18T00:00:00Z", "updated_at": "2026-10-18T00:00:00Z",
            "representative": None}


def test_duplicate_reference_description_retains_full_text_and_complete_aliases():
    entry = profile(0, description="参考背景" * 400)
    entry["aliases"] = [f"完整别名{number}" for number in range(12)]
    context = {"profiles": [entry], "reference_frames": [], "retrieval": {"reference_only": True}}
    original = deepcopy(context)

    bounded = SceneJobMixin._bounded_scene_profiles(context)

    assert bounded["profiles"][0]["description"] == entry["description"]
    assert bounded["profiles"][0]["aliases"] == entry["aliases"]
    assert "reference_description" not in bounded["profiles"][0]
    assert bounded["profiles"][0]["reference_description_same_as"] == "description"
    assert bounded["profiles"][0]["reference_policy"] == BANGUMI_REFERENCE_POLICY
    assert bounded["dossier_detail_compacted"] is True
    assert "dossier_detail_truncated" not in bounded
    assert context == original


def test_whole_cast_keeps_short_biographies_notes_traits_and_prior_policies():
    context = {"profiles": [profile(index) for index in reversed(range(14))],
               "reference_frames": [], "retrieval": {"identity_status": "unverified_prior",
                                                       "reference_only": True, "presence_verified": False}}
    original = deepcopy(context)
    assert encoded_size(canonical(context)) > 18000

    bounded = SceneJobMixin._bounded_scene_profiles(context)

    assert encoded_size(canonical(bounded)) <= 18000
    assert len(bounded["profiles"]) == 14
    assert bounded["dossier_detail_truncated"] is True
    assert not bounded.get("dossier_detail_omitted")
    assert bounded["retrieval"] == context["retrieval"]
    assert bounded["reference_policy"] == BANGUMI_REFERENCE_POLICY
    text = ProviderManager.character_context_text(bounded)
    serialized = json.loads(text)
    assert len(text.encode("utf-8")) <= 18000
    for field in ("reference_policy", "dossier_detail_compacted", "dossier_detail_truncated",
                  "dossier_metadata_omitted", "retrieval"):
        assert serialized[field] == bounded[field]
    assert len(serialized["profiles"]) == 14
    assert all(entry["reference_policy"] == "context.reference_policy" and entry["reference_only"]
               and entry["identity_status"] == "unverified_prior" and entry["presence_verified"] is False
               for entry in serialized["profiles"])
    for entry, before in zip(bounded["profiles"], reversed(context["profiles"])):
        assert entry["id"] == before["id"]
        assert entry["aliases"] == before["aliases"]
        assert entry["name"] == before["name"]
        assert entry["description"].startswith("参考背景")
        assert entry["notes"].startswith("人工备注")
        assert entry["appearance"] and entry["observed_traits"]
        assert "description" in entry["dossier_truncated_fields"]
        assert entry["reference_description_same_as"] == "description"
        assert entry["external_source"] == {key: value for key, value in before["external_source"].items()
                                             if key not in {"fetched_at", "image_url"}}
        assert entry["reference_policy"] == "context.reference_policy"
        for field in ("external_sources", "reference_only",
                      "identity_status", "presence_verified"):
            assert entry[field] == before[field]
        assert "name_variants" not in entry  # Every name/source remains present above.
    assert context == original


def test_distinct_reference_biography_and_nonduplicate_provenance_survive():
    entry = profile(0, description="人工描述" * 10000)
    entry["reference_description"] = "外部背景" * 10000
    entry["name_variants"][0]["source_url"] = "https://other.invalid/character/1"
    entry["name_variants"][1]["name"] = "独有名称"

    bounded = SceneJobMixin._bounded_scene_profiles({"profiles": [entry]})
    result = bounded["profiles"][0]

    assert result["description"].startswith("人工描述")
    assert result["reference_description"].startswith("外部背景")
    assert "reference_description_same_as" not in result
    assert {"description", "reference_description"} <= set(result["dossier_truncated_fields"])
    assert result["name_variants"][0] == entry["name_variants"][0]
    assert result["name_variants"][1] == {"name": "独有名称", "language": "ja", "kind": "alias"}
    assert result["reference_policy"] == "context.reference_policy"
    assert bounded["reference_policy"] == BANGUMI_REFERENCE_POLICY
    assert encoded_size(canonical(bounded)) <= 18000


def test_large_cast_explicit_fallback_retains_full_roster_and_prior_policies():
    entries = [{"id": f"character_{index:03d}", "name": f"人物{index}", "aliases": [f"别名{index}"],
                "description": "参考背景" * 1000, "notes": "人工备注" * 1000,
                "reference_only": True, "identity_status": "unverified_prior", "presence_verified": False,
                "reference_policy": "Prior reference only; no current observation is established."}
               for index in range(70)]

    bounded = SceneJobMixin._bounded_scene_profiles({"profiles": entries})

    assert encoded_size(canonical(bounded)) <= 18000
    assert bounded["dossier_detail_omitted"] is True
    assert len(bounded["profiles"]) == len(entries)
    for result, entry in zip(bounded["profiles"], entries):
        assert "description" not in result
        assert result["reference_policy"] == "context.reference_policy"
        assert bounded["reference_policy"] == entry["reference_policy"]
        for field in ("id", "name", "aliases", "reference_only", "identity_status", "presence_verified"):
            assert result[field] == entry[field]


def test_oversized_names_and_aliases_raise_without_silently_losing_identities():
    context = {"profiles": [{"id": "character_1", "name": "人物", "aliases": ["完整别名" * 2000]}]}
    original = deepcopy(context)

    with pytest.raises(ValueError, match="角色未被静默截断"):
        SceneJobMixin._bounded_scene_profiles(context)

    assert context == original

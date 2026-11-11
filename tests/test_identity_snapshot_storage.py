import json

import pytest

from scenerecall.identity_storage import compact_identity_snapshot
from scenerecall.library import Library, atomic_json, digest
from test_compact_person_crops import compact


def test_compact_new_and_old_snapshots_preserve_all_evidence_and_receipts(tmp_path):
    library = Library(tmp_path / "library")
    asset = library.asset_dir("episode")
    atomic_json(asset / "manifest.json", {"id": asset.name})
    latest = asset / "identity/visual/latest.json"
    receipt = asset / "analyses/run/retrieval-local-identity.json"
    visual = {"asset_id": asset.name, "status": "completed",
              "regions": [{"caption": '  引号 " Hi \n\t  ', "score": .01234567, "unknown": [True, None, -3]}]}
    saved = {"input_signature": "frozen", "visual": visual}
    atomic_json(latest, visual)
    atomic_json(receipt, saved)
    # Existing documents can contain number spellings and escaping that a
    # generic JSON decoder would normalize. Preserve them byte-for-byte.
    old = latest.read_bytes().replace(b"0.01234567", b"1.234567000E-2")
    latest.write_bytes(old)
    old_total = len(old) + receipt.stat().st_size
    unrelated = asset / "identity/visual/crops/region_keep.png"
    unrelated.parent.mkdir()
    unrelated.write_bytes(b"unrelated")
    counts = compact.run(library.root, mode="verify", identity_json_only=True)["totals"]
    assert counts["verified_savings_bytes"] > 0 and counts["saved_bytes"] == 0
    assert latest.read_bytes() == old
    result = compact.run(library.root, mode="apply", identity_json_only=True)["totals"]
    assert result["saved_bytes"] == counts["verified_savings_bytes"] == old_total - latest.stat().st_size - receipt.stat().st_size
    assert digest(json.loads(latest.read_bytes())) == digest(visual)
    assert b"1.234567000E-2" in latest.read_bytes()
    assert json.loads(receipt.read_bytes()) == saved and unrelated.read_bytes() == b"unrelated"
    assert compact.run(library.root, mode="apply", identity_json_only=True)["totals"]["saved_bytes"] == 0
    atomic_json(latest, visual, compact=True)
    assert json.loads(latest.read_bytes()) == visual and b"\n" not in latest.read_bytes()


@pytest.mark.parametrize("raw", [b'invalid json', b'{"asset_id":"other","regions":[]}', b'[]'])
def test_invalid_snapshot_keeps_original_bytes(tmp_path, raw):
    path = tmp_path / "latest.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        compact_identity_snapshot(path, "episode", apply=True)
    assert path.read_bytes() == raw

from copy import deepcopy

import pytest

from scenerecall import library as library_module
from scenerecall.library import atomic_json, read_json
from test_characters import cast, observation  # noqa: F401


def many_frames(library, count=12):
    identifiers = ["frame_one"]
    for index in range(1, count):
        identifier = f"frame_{index}"
        path = library.asset_dir("asset_one") / "frames" / f"{identifier}.jpg"
        path.write_bytes(b"synthetic storage evidence; no image decoding")
        library.register_frame("asset_one", {"id": identifier, "path": str(path), "at_ms": index})
        identifiers.append(identifier)
    return identifiers


def test_apply_observation_reads_one_frame_registry_and_matches_legacy_validation(cast, monkeypatch):  # noqa: F811
    library, store = cast
    frames = many_frames(library)
    identity = store.create("asset_one", {"name": "Aoi"})["id"]
    record = observation(character_id=identity, evidence_frame_ids=frames,
                         portrait_frame_id=frames[-1], portrait_box=[0, 0, .5, .5])
    record["evidence_frame_ids"] = frames
    original = deepcopy(record)
    registry_path = library.asset_dir("asset_one") / "frames" / "registry.json"
    reads = []
    original_read = library_module.read_json

    def counted_read(path, *args, **kwargs):
        if path == registry_path:
            reads.append(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(library_module, "read_json", counted_read)
    batched = store.apply_observation("asset_one", record)
    assert len(reads) == 1
    assert batched["evidence_frame_ids"] == frames
    assert batched["entities"][0]["character_id"] == identity
    assert batched["entities"][0]["portrait_frame_id"] == frames[-1]

    original_frame_paths = library.frame_paths

    def legacy_paths(asset_id, frame_ids, **_kwargs):
        return {frame_id: original_frame_paths(asset_id, [frame_id])[frame_id] for frame_id in frame_ids}

    reads.clear()
    monkeypatch.setattr(library, "frame_paths", legacy_paths)
    legacy = store.apply_observation("asset_one", record)
    assert len(reads) == len(frames)
    assert batched == legacy
    assert record == original


@pytest.mark.parametrize("broken", ["unregistered", "missing_file", "outside_asset"])
def test_batch_validates_last_evidence_and_rejects_missing_or_escaping_paths(cast, broken):  # noqa: F811
    library, store = cast
    frames = many_frames(library)
    record = observation(evidence_frame_ids=frames, name=None)
    record["evidence_frame_ids"] = frames
    registry_path = library.asset_dir("asset_one") / "frames" / "registry.json"
    registry = read_json(registry_path)
    if broken == "unregistered":
        registry.pop(frames[-1])
    elif broken == "missing_file":
        (library.asset_dir("asset_one") / registry[frames[-1]]["path"]).unlink()
    else:
        registry[frames[-1]]["path"] = "../outside.jpg"
        (library.asset_dir("asset_one").parent / "outside.jpg").write_bytes(b"foreign evidence")
    atomic_json(registry_path, registry)
    with pytest.raises(ValueError if broken == "outside_asset" else FileNotFoundError):
        store.apply_observation("asset_one", record)
    assert record["evidence_frame_ids"] == frames
    assert store.catalog("asset_one")["profiles"] == []

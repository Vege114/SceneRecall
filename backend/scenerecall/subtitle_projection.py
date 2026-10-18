"""Compact duplicate subtitle effects in the disposable search projection only.

Source cues, annotations and evidence files remain untouched. Adjacent cues are
deliberately separate: without ASS style information, two touching equal lines
may be two actual utterances rather than animation fragments.
"""
from __future__ import annotations

import copy
import json
from collections import defaultdict


# Everything else participates in the semantic key, including asset, language,
# source track, entities, text, and any future annotation or interpretation fields.
_SOURCE_FIELDS = {
    "id", "record_id", "start_ms", "end_ms", "source_start_ms", "source_end_ms",
    "source_cue_index", "evidence_frame_ids", "thumbnail", "source_record_ids", "source_spans",
}


def _identity(record: dict) -> tuple[str, str]:
    return str(record.get("record_id") or record["id"]), str(record["id"])


def _time_order(record: dict) -> tuple:
    return record["start_ms"], record["end_ms"], *_identity(record)


def _protected(record: dict) -> bool:
    # Each collectible or manually interpreted cue must keep its own identity,
    # even when two such cues currently happen to contain the same metadata.
    return bool(record.get("favorite") or record.get("note") or record.get("user_confirmed")
                or record.get("review_status") == "user_confirmed")


def _source_spans(record: dict) -> list[dict]:
    if record.get("source_spans"):
        return record["source_spans"]
    span = {"record_id": record.get("record_id") or record["id"],
            "start_ms": record["start_ms"], "end_ms": record["end_ms"]}
    for key in ("source_start_ms", "source_end_ms"):
        if key in record:
            span[key] = record[key]
    return [span]


def _merge(rows: list[dict]) -> dict:
    # Identity does not depend on the expanded time range, so repeated compaction
    # and compaction of an already compacted subset retain an original cue ID.
    result = copy.deepcopy(min(rows, key=_identity))
    result["start_ms"] = min(row["start_ms"] for row in rows)
    result["end_ms"] = max(row["end_ms"] for row in rows)
    spans = {}
    for row in rows:
        for span in _source_spans(row):
            key = json.dumps(span, sort_keys=True, ensure_ascii=False, allow_nan=False)
            spans[key] = span
    result["source_spans"] = [copy.deepcopy(span) for span in sorted(
        spans.values(), key=lambda span: (span["start_ms"], span["end_ms"], str(span["record_id"])))]
    identities = [span["record_id"] for span in result["source_spans"]]
    identities.extend(identity for row in sorted(rows, key=_time_order)
                      for identity in (row.get("source_record_ids") or [row.get("record_id") or row["id"]]))
    result["source_record_ids"] = list(dict.fromkeys(identities))
    result["evidence_frame_ids"] = list(dict.fromkeys(
        frame for row in sorted(rows, key=_time_order) for frame in row.get("evidence_frame_ids", [])))
    if not result.get("thumbnail") and result["evidence_frame_ids"]:
        result["thumbnail"] = result["evidence_frame_ids"][0]
    return result


def compact_subtitle_records(records: list[dict]) -> list[dict]:
    """Merge strictly overlapping, semantically identical subtitle cue chains.

    Work is O(N log N), rather than comparing every cue with every other cue.
    A merged row occupies its first source row's position; untouched rows retain
    their exact values and relative order. Original dictionaries are never edited.
    """
    groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    output: list[tuple[int, dict]] = []
    for position, record in enumerate(records):
        if record.get("kind") != "subtitle" or record.get("is_context") or _protected(record):
            output.append((position, record))
            continue
        key = json.dumps({key: value for key, value in record.items() if key not in _SOURCE_FIELDS},
                         sort_keys=True, ensure_ascii=False, allow_nan=False)
        groups[key].append((position, record))

    for group in groups.values():
        group.sort(key=lambda item: _time_order(item[1]))
        chain: list[tuple[int, dict]] = []
        end_ms = -1

        def publish() -> None:
            if len(chain) == 1:
                output.append(chain[0])
            elif chain:
                output.append((min(position for position, _ in chain), _merge([row for _, row in chain])))

        for item in group:
            row = item[1]
            if chain and row["start_ms"] >= end_ms:
                publish()
                chain = []
            chain.append(item)
            end_ms = max(end_ms, row["end_ms"]) if len(chain) > 1 else row["end_ms"]
        publish()
    output.sort(key=lambda item: item[0])
    return [record for _, record in output]

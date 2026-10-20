"""Conservative source-frame seeking for an already observed local action.

This is a bounded Chinese query grammar, not another event detector. Unconsumed
roles, attributes, locations, negation or multiple actions keep the frame seek.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from fractions import Fraction


_ACTIONS = {
    "转身": ("turn", "unary"), "转过身": ("turn", "unary"),
    "进入": ("enter", "place"), "走进": ("enter", "place"), "进到": ("enter", "place"),
    "离开": ("leave", "place"), "走出": ("leave", "place"),
    "拿起": ("pick_up", "object"), "拾起": ("pick_up", "object"),
    "放下": ("put_down", "object"), "接过": ("receive", "object"), "接住": ("catch", "object"),
    "递出": ("hand_out", "object"), "递给": ("give", "transfer"), "交给": ("give", "transfer"),
}
_ACTION = re.compile("|".join(sorted(_ACTIONS, key=len, reverse=True)))
_UNSUPPORTED = re.compile(r"不|没|未|无|别|非|先|然后|再|同时|接着|之后|随后|并且|以及|或者|或|又|"
                          r"\b(?:not|never|without|before|after|then)\b", re.IGNORECASE)
_BOUND = {None, "confirmed", "auto_bound", "human_confirmed", "manual_confirmed"}
_ROLES = ("actor_ids", "target_ids", "object_id", "recipient_id")


def _text(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value or "")).casefold())


def _query(value):
    value = _text(value).strip("。？！?!，,；;")
    for prefix in ("请帮我找到", "请帮我查找", "帮我找到", "请找到", "帮我找", "查找", "搜索", "找到", "检索", "找"):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    for suffix in ("的片段", "的画面", "的镜头", "片段", "画面", "镜头"):
        if value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    return value


def _role_ids(event, role):
    value = event.get(role)
    if value is None:
        return []
    if role.endswith("_ids"):
        return value if isinstance(value, list) and all(isinstance(item, str) for item in value) else None
    return [value] if isinstance(value, str) and value else None


def _labels(record):
    labels = {}
    entities = {entity["id"]: entity for entity in record.get("entities", []) if entity.get("id")}
    for identity, entity in entities.items():
        names = [entity.get("description")]
        status = entity.get("binding_status", entity.get("identity_status", entity.get("association_status")))
        if entity.get("character_id") and status in _BOUND:
            names.extend([entity.get("character_name"), entity.get("name"), *entity.get("aliases", [])])
        for name in names:
            if name and _text(name):
                labels.setdefault(_text(name), set()).add(identity)
    for binding in record.get("concept_bindings", []):
        identity = binding.get("entity_id")
        if binding.get("status") != "confirmed" or identity not in entities:
            continue
        for name in [binding.get("concept_name"), *binding.get("aliases", [])]:
            if name and _text(name):
                labels.setdefault(_text(name), set()).add(identity)
    return entities, {name: next(iter(ids)) for name, ids in labels.items() if len(ids) == 1}


def _role_matches(fragment, identities, labels, *, actor=False):
    if actor:
        for suffix in ("正在", "开始", "正", "在"):
            if fragment.endswith(suffix):
                fragment = fragment[:-len(suffix)]
                break
    if not fragment:
        return True  # No role constraint was present in this query.
    if not identities:
        return False
    if fragment in labels:
        return set(identities) == {labels[fragment]}
    pieces = re.split("[和与、]", fragment)
    return (len(pieces) > 1 and all(piece in labels for piece in pieces)
            and len(set(pieces)) == len(pieces) and set(identities) == {labels[piece] for piece in pieces})


def _phrase_matches(phrase, match, grammar, roles, labels):
    before, after = phrase[:match.start()], phrase[match.end():]
    if after.endswith("了"):
        after = after[:-1]
    if grammar == "unary":
        return not after and _role_matches(before, roles["actor_ids"], labels, actor=True)
    if grammar == "place":
        target = roles["target_ids"] or roles["object_id"]
        return (_role_matches(before, roles["actor_ids"], labels, actor=True)
                and _role_matches(after, target, labels))
    # The 把/将 construction fixes the object before the verb, preserving who
    # gives what to whom instead of accepting mere name co-occurrence.
    disposal = re.split("[把将]", before)
    if len(disposal) > 2:
        return False
    if len(disposal) == 2:
        if (not disposal[1] or not roles["object_id"]
                or not _role_matches(disposal[0], roles["actor_ids"], labels, actor=True)
                or not _role_matches(disposal[1], roles["object_id"], labels)):
            return False
        return (not after if grammar == "object" else
                bool(after) and _role_matches(after, roles["recipient_id"], labels))
    target = roles["recipient_id"] if grammar == "transfer" else roles["object_id"]
    return (_role_matches(before, roles["actor_ids"], labels, actor=True)
            and _role_matches(after, target, labels))


def _matches(query, event, entities, labels):
    query_actions = list(_ACTION.finditer(query))
    action = _text(event.get("action"))
    event_actions = list(_ACTION.finditer(action))
    if len(query_actions) != 1 or len(event_actions) != 1 or _UNSUPPORTED.search(action):
        return False
    query_match, event_match = query_actions[0], event_actions[0]
    family, grammar = _ACTIONS[query_match.group()]
    if _ACTIONS[event_match.group()][0] != family:
        return False
    roles = {role: _role_ids(event, role) for role in _ROLES}
    if any(value is None or any(identity not in entities for identity in value) for value in roles.values()):
        return False
    if not roles["actor_ids"]:
        return False
    # Action prose has to fit the same bounded grammar. A bare token inside
    # "attempts to turn" must not override explicit uncertainty in the phrase.
    return (_phrase_matches(action, event_match, grammar, roles, labels)
            and _phrase_matches(query, query_match, grammar, roles, labels))


def _supported(record, event):
    if event.get("temporal") is not True or event.get("polarity", "positive") != "positive" or event.get("uncertainty"):
        return None
    start, end = event.get("start_ms"), event.get("end_ms")
    lower, upper = record.get("start_ms"), record.get("end_ms")
    if record.get("source_spans"):
        origin = next((span for span in record["source_spans"]
                       if span.get("record_id") == (record.get("record_id") or record.get("id"))), None)
        if origin is None:
            return None
        lower, upper = origin.get("start_ms"), origin.get("end_ms")
    if any(type(value) is not int for value in (start, end, lower, upper)) or not lower <= start < end <= upper:
        return None
    frames = {frame["id"]: frame for frame in record.get("ordered_frames", [])}
    ids = event.get("evidence_frame_ids")
    if (not isinstance(ids, list) or len(ids) < 2 or len(set(ids)) != len(ids)
            or not set(ids) <= set(record.get("evidence_frame_ids", []))
            or any(identity not in frames for identity in ids)):
        return None
    selected, pts_times = [frames[identity] for identity in ids], []
    for frame in selected:
        base = frame.get("time_base")
        if (type(frame.get("pts")) is not int or not isinstance(base, (list, tuple)) or len(base) != 2
                or any(type(value) is not int or value <= 0 for value in base) or type(frame.get("at_ms")) is not int):
            return None
        timestamp = Fraction(frame["pts"]) * Fraction(*base)
        if round(timestamp * 1000) != frame["at_ms"] or not start <= frame["at_ms"] <= end:
            return None
        pts_times.append(timestamp)
    if (selected[0]["at_ms"] != start or len(set(pts_times)) != len(pts_times)
            or pts_times != sorted(pts_times) or len({frame["at_ms"] for frame in selected}) < 2):
        return None
    return selected


def event_seek(query: str, canonical_record: dict) -> dict | None:
    """Return one evidenced action onset, or keep the existing visual seek."""
    record = canonical_record
    if (record.get("kind") != "visual" or record.get("evidence_type") != "retrieval_window"
            or record.get("provenance", {}).get("evidence_kind") != "current_visual"
            or not isinstance(query, str) or _UNSUPPORTED.search(query)):
        return None
    text = _query(query)
    entities, labels = _labels(record)
    matching = [event for event in record.get("events", []) if _matches(text, event, entities, labels)]
    if len(matching) != 1:
        return None
    event = matching[0]
    frames = _supported(record, event)
    if frames is None:
        return None
    participants = []
    for role in _ROLES:
        for identity in _role_ids(event, role):
            entity = entities[identity]
            concepts = [binding["concept_id"] for binding in record.get("concept_bindings", [])
                        if binding.get("status") == "confirmed" and binding.get("entity_id") == identity]
            participants.append({"role": role, "entity_id": identity,
                                 "character_id": entity.get("character_id"), "concept_ids": concepts})
    return {"seek_ms": event["start_ms"], "seek_basis": "local_event",
            "event_evidence": {"record_id": record.get("record_id") or record["id"],
                **{key: copy.deepcopy(event.get(key)) for key in ("action", *_ROLES, "polarity", "start_ms", "end_ms",
                                                                "evidence_frame_ids")},
                "participants": participants, "ordered_frames": copy.deepcopy(frames)}}

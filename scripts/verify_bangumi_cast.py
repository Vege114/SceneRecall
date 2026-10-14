#!/usr/bin/env python3
"""Verify an actual Bangumi cast against an isolated TV-series import.

Example (run from the installed development environment):
    python scripts/verify_bangumi_cast.py --media-dir /path/to/TV \
        --subject 214265 --expected-episodes 12 --output test-results/cast-check \
        --provider-settings /path/to/library/private --vision-times-ms 70000,90000,110000

Only top-level numbered video files are imported. External simplified-Chinese
subtitles are preferred when the media has no text track; traditional-Chinese
sidecars are independently parsed as well. Source media and subtitles are never
modified. The output is a private verification library, not a redistributable
fixture: it contains subtitle text and, when requested, a few video frames and
provider results. Keep it under an ignored directory such as test-results/.
Invalid source ASS events remain a strict import failure unless the explicit
--filter-invalid-ass-events flag creates a labelled local derivative. That mode
records original rejection and omission counts; it does not establish support
for importing the unmodified ASS.

The optional vision probe calls the real configured provider. It never replaces
the user's settings, copies credentials, or processes every episode with a model.
Explicitly labelled injected observations exercise policy edges separately from
the real model response. Exit 0 means all required assertions passed; 1 means an
assertion failed or the optional provider probe could not complete.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import re
from pathlib import Path
import sys

import pysubs2

from scenerecall import media, subtitles
from scenerecall.anime_profiles import AnimeProfileStore
from scenerecall.bangumi import BangumiClient, parse_subject_reference
from scenerecall.characters import CharacterStore, external_sources
from scenerecall.library import Library, atomic_json, digest, read_json, source_snapshot
from scenerecall.models import AssetInput
from scenerecall.providers import PROMPT_VERSION, Profile, ProviderError, ProviderManager


def log(value: str) -> None:
    print(value, flush=True)


def inventory(directory: Path, pattern: str, expected: int) -> list[tuple[int, Path]]:
    numbered = re.compile(pattern)
    found = []
    for path in directory.iterdir():
        match = numbered.search(path.name)
        if path.is_file() and path.suffix.lower() in {".mkv", ".mp4", ".mov"} and match:
            found.append((int(match[1]), path))
    found.sort(key=lambda pair: pair[0])
    if [number for number, _ in found] != list(range(1, expected + 1)):
        raise ValueError(f"Expected exactly episodes 1–{expected} at the top level")
    return found


def roster_ids(profiles: list[dict], subject_id: int) -> set[int]:
    return {source["character_id"] for profile in profiles for source in external_sources(profile)
            if source.get("source") == "bangumi" and source.get("subject_id") == subject_id}


def check(report: dict, label: str, passed: bool, **details) -> None:
    item = {"check": label, "passed": bool(passed), **details}
    report["checks"].append(item)
    log(f"{'PASS' if passed else 'FAIL'} {label}")


def prepare_sidecar(path: Path, duration: int, args: argparse.Namespace, episode: int,
                    suffix: str) -> tuple[Path, list[dict], dict]:
    """Never alter sources; an explicit flag permits a labelled derived fixture."""
    details = {"suffix": suffix, "derived_subtitle": False,
               "source_validation": {"status": "passed", "excluded_events": 0}}
    try:
        cues = subtitles.parse_subtitles(path, duration)
        return path, cues, details
    except subtitles.SubtitleError as exc:
        if not args.filter_invalid_ass_events:
            raise
        source_error = str(exc)
    parsed = pysubs2.SSAFile.from_string(subtitles._read(path), format_="ass")
    excluded = [event for event in parsed if not event.is_comment and event.end <= event.start]
    if not excluded:
        raise subtitles.SubtitleError("ASS failed for a reason other than an empty/inverted event interval")
    retained = [event for event in parsed if event not in excluded]
    before = [event.as_dict() for event in retained]
    parsed.events = retained
    target = args.output / "derived-subtitles" / f"episode_{episode:02}_{suffix.strip('.').replace('.', '_')}.ass"
    target.parent.mkdir(parents=True, exist_ok=True)
    parsed.save(target, encoding="utf-8", format_="ass")
    if [event.as_dict() for event in pysubs2.load(target)] != before:
        raise ValueError("Derived ASS failed the retained-event preservation check")
    cues = subtitles.parse_subtitles(target, duration)
    details.update(derived_subtitle=True, derived_path=str(target.relative_to(args.output)),
                   source_validation={"status": "rejected", "error": source_error,
                                      "excluded_events": len(excluded),
                                      "excluded_styles": dict(Counter(event.style for event in excluded)),
                                      "retained_events": len(retained), "retained_events_preserved": True})
    return target, cues, details


def record(asset: dict, frame: dict, label: str, entity: dict) -> dict:
    return {"id": "verification_" + label, "asset_id": asset["id"], "run_id": "run_policy_verification",
            "window_id": "window_" + label, "shot_id": "shot_policy_verification",
            "start_ms": frame["at_ms"], "end_ms": frame["at_ms"] + 1,
            "summary": "Explicit injected policy-verification fixture; not a model observation",
            "evidence_frame_ids": [frame["id"]], "entities": [{"id": "entity_" + label,
            "kind": "person", "is_character": True, "name": None, "character_id": None,
            "match_confidence": 0, "match_reason": "Injected unmatched-identity test fixture",
            "observed_traits": ["Injected test fixture"], "description": "Injected policy test entity",
            "evidence_frame_ids": [frame["id"]], "portrait_frame_id": None,
            "portrait_box": None, **entity}], "verification_fixture": True}


def save_policy_fixture(library: Library, characters: CharacterStore, item: dict) -> dict:
    with library.lock:
        linked = characters.apply_observation(item["asset_id"], item)
        library.save_observation(item["asset_id"], linked)
    return linked


def policy_checks(report: dict, library: Library, characters: CharacterStore, asset: dict,
                  frame: dict, source_ids: set[int], subject_id: int) -> None:
    before = characters.catalog(asset["id"])
    linked = save_policy_fixture(library, characters, record(asset, frame, "unmatched", {}))
    after = characters.catalog(asset["id"])
    check(report, "unmatched entity retains evidence without creating a dossier",
          linked["entities"][0].get("character_id") is None
          and linked["entities"][0]["description"] == "Injected policy test entity"
          and linked["entities"][0]["evidence_frame_ids"] == [frame["id"]]
          and {row["id"] for row in before["profiles"]} == {row["id"] for row in after["profiles"]})
    named = record(asset, frame, "named_outside_roster", {"name": "Verification Outside Roster",
                   "match_confidence": .95, "match_reason": "Verification Outside Roster: injected named fixture"})
    named_linked = save_policy_fixture(library, characters, named)
    check(report, "a named observation outside Bangumi cannot add a dossier",
          named_linked["entities"][0].get("character_id") is None
          and named_linked["entities"][0]["evidence_frame_ids"] == [frame["id"]]
          and {p["id"] for p in characters.catalog(asset["id"])["profiles"]}
          == {p["id"] for p in before["profiles"]})

    # A raw legacy profile and immutable observation reproduce the old polluted
    # library shape. Only this isolated output library receives injected data.
    scope = after["scope_id"]
    registry = characters._read(scope)
    legacy_id = "character_verification_legacy_pollution"
    legacy = deepcopy(next(iter(registry["profiles"].values())))
    legacy.update(id=legacy_id, name="Injected legacy unnamed extra", name_cn="", name_ja="", name_en="",
                  aliases=[], appearance=[],
                  description="Explicit legacy-pollution fixture", notes="", user_edited=False,
                  edited_fields=[], representative=None)
    for field in ("external_source", "external_sources", "reference_description", "identity_origins", "imported_from"):
        legacy.pop(field, None)
    registry["profiles"][legacy_id] = legacy
    atomic_json(characters._path(scope), registry)
    legacy_record = record(asset, frame, "legacy", {"character_id": legacy_id,
                            "match_confidence": .9, "match_reason": "Injected legacy reference"})
    legacy_record["character_scope_id"] = scope
    legacy_record["character_revision"] = registry["revision"]
    # Save directly, as this represents already-committed historical evidence.
    library.save_observation(asset["id"], legacy_record)
    projected = characters.catalog(asset["id"])
    context = characters.context(asset["id"])
    decorated = characters.decorate_observations(asset["id"], [legacy_record])[0]
    check(report, "old pollution is excluded from catalog and model context",
          legacy_id not in {p["id"] for p in projected["profiles"]}
          and legacy_id not in {p["id"] for p in context["profiles"]}
          and roster_ids(projected["profiles"], subject_id) == source_ids)
    check(report, "old polluted evidence is preserved without a projected identity",
          legacy_id in characters._read(scope)["profiles"]
          and read_json(library.asset_dir(asset["id"]) / "analyses" / legacy_record["run_id"]
                        / "observations" / (legacy_record["id"] + ".json"))["entities"][0]["character_id"] == legacy_id
          and decorated["entities"][0].get("character_id") is None
          and decorated["entities"][0]["evidence_frame_ids"] == [frame["id"]])


def configured_provider(settings_dir: Path, output: Path) -> tuple[ProviderManager, dict]:
    settings_dir = settings_dir.expanduser().resolve(strict=True)
    settings = read_json(settings_dir / "settings.json", {})
    profiles = read_json(settings_dir / "profiles.json", {}).get("profiles", [])
    selected_id = settings.get("bindings", {}).get("vision")
    profile = next((p for p in profiles if p.get("id") == selected_id), None)
    if not profile:
        raise ValueError("The configured library has no vision binding")
    if profile.get("provider_type", "openai_compatible") != "codex_cli" and profile.get("secret_mode") not in {"none", "env"}:
        raise ValueError("This verifier supports Codex CLI or API none/env credentials; no credentials are copied")
    providers = ProviderManager(output / "provider-settings")
    # Validated profiles contain no API key; Codex authentication stays in its own
    # credential store, and env references keep the actual key out of the output.
    selected = providers.upsert(Profile.model_validate(profile).model_dump())
    return providers, selected


async def vision_probe(report: dict, args: argparse.Namespace, library: Library,
                       characters: CharacterStore, asset: dict, frames: list[dict]) -> None:
    if args.provider_settings is None:
        report["vision"] = {"status": "not_requested"}
        return
    try:
        providers, profile = configured_provider(args.provider_settings, args.output)
        context = characters.context(asset["id"])
        start = min(f["at_ms"] for f in frames)
        end = max(f["at_ms"] for f in frames) + 1
        cues = [cue for cue in library.subtitles(asset["id"])
                if cue["start_ms"] < end and cue["end_ms"] > start]
        log(f"Running real vision probe on episode 1: {len(frames)} frames")
        result = await providers.analyze(profile["id"], frames, start, end, context,
                                         {"start_ms": start, "end_ms": end, "cues": cues}, None)
        atomic_json(args.output / "real-vision-response.json", result.data)
        item = {**result.data, "id": "verification_real_vision", "asset_id": asset["id"],
                "run_id": "run_real_vision", "window_id": "window_real_vision", "shot_id": "shot_real_vision",
                "start_ms": start, "end_ms": end, "provenance": {"provider_type": profile["provider_type"],
                "model": profile["model"], "prompt_version": PROMPT_VERSION,
                "character_context_digest": digest(ProviderManager.character_context_text(context))}}
        with library.lock:
            linked = characters.apply_observation(asset["id"], item, expected_revision=context["revision"])
            library.save_observation(asset["id"], linked)
        known_ids = {p["id"] for p in context["profiles"]}
        entities = linked["entities"]
        matched = [entity for entity in entities if entity.get("character_id")]
        unmatched = [entity for entity in entities if entity.get("is_character") and not entity.get("character_id")]
        check(report, "real vision uses only Bangumi roster identities",
              bool(entities) and all(entity.get("character_id") in known_ids for entity in matched)
              and {p["id"] for p in characters.catalog(asset["id"])["profiles"]} == known_ids)
        report["vision"] = {"status": "passed", "provider_type": profile["provider_type"],
                            "model": profile["model"] or "configured CLI default", "frames": len(frames),
                            "requested_timestamps_ms": args.vision_times_ms,
                            "actual_timestamps_ms": [frame["at_ms"] for frame in frames],
                            "entities": len(entities), "matched_characters": len(matched),
                            "unmatched_characters": len(unmatched), "request_count": result.request_count,
                            "usage": result.usage, "cost": result.cost,
                            "matched_names": [next(p["name"] for p in context["profiles"]
                                                   if p["id"] == entity["character_id"]) for entity in matched]}
    except Exception as exc:
        report["vision"] = {"status": "blocked", "error_type": type(exc).__name__}
        if isinstance(exc, ProviderError):
            report["vision"]["error"] = str(exc)
        check(report, "real vision probe completed", False, error_type=type(exc).__name__)
        # Application error messages are safe; unexpected external diagnostics
        # are intentionally never copied into the shareable summary.
        log(f"Real vision probe blocked: {type(exc).__name__}")


async def verify(args: argparse.Namespace) -> dict:
    args.media_dir = args.media_dir.expanduser().resolve(strict=True)
    args.output = args.output.expanduser().resolve()
    if args.output == args.media_dir or args.media_dir in args.output.parents:
        raise ValueError("Verification output must be outside the media input directory")
    library_root = args.output / "library"
    if library_root.exists() and not args.resume:
        raise ValueError("Output library already exists; choose a fresh output or explicitly --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = inventory(args.media_dir, args.episode_pattern, args.expected_episodes)
    sources = [path for _, path in paths]
    sidecars = [p for path in sources for suffix in (".sc.ass", ".tc.ass")
                if (p := path.with_suffix(suffix)).is_file()]
    snapshots = {path: source_snapshot(path) for path in sources + sidecars}
    report = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "subject_id": args.subject, "checks": [], "episodes": [],
              "limits": ["Real model analysis is limited to supplied episode-1 timestamps.",
                         "Roster/profile and subtitle checks cover every imported episode.",
                         "Explicit injected observations test unmatched and legacy policy edges.",
                         "Provider identity accuracy is not established by this sampled smoke test."]}
    check(report, "top-level TV inventory", True, episodes=[number for number, _ in paths],
          nested_media_excluded=True)
    library = Library(library_root)
    anime = AnimeProfileStore(library, client=BangumiClient(timeout_s=args.bangumi_timeout))
    log(f"Reading public Bangumi subject {args.subject}")
    if args.dossier_cache:
        dossier = read_json(args.dossier_cache.expanduser().resolve(strict=True))
        if dossier.get("subject", {}).get("id") != args.subject:
            raise ValueError("Dossier cache does not match the requested subject")
        report["bangumi_read"] = "explicit public-read cache"
    else:
        # Bypass AnimeProfileStore's cache so each verification performs a real
        # public adapter read. Its snapshot is persisted only in this output.
        dossier = await anime.client.dossier(args.subject)
        report["bangumi_read"] = "live public adapter"
    atomic_json(args.output / "bangumi-dossier.json", dossier)
    source_ids = {role["id"] for role in dossier["characters"]}
    check(report, "Bangumi dossier has a unique cast", bool(source_ids)
          and len(source_ids) == len(dossier["characters"]), characters=len(source_ids),
          warnings=len(dossier.get("warnings", [])), subject_name=dossier["subject"]["name"])
    platform = dossier["subject"].get("platform") or {}
    platform_type = platform.get("type") if isinstance(platform, dict) else platform
    check(report, "Bangumi subject matches the TV episode inventory",
          dossier["subject"].get("eps") == args.expected_episodes and str(platform_type).lower() == "tv")
    projects = library.organization.projects()
    if args.resume and projects:
        if len(projects) != 1:
            raise ValueError("Verification library has unexpected projects")
        project = projects[0]
    else:
        project = library.organization.save_project({"name": dossier["subject"].get("name_cn")
                    or dossier["subject"]["name"], "kind": "animation", "tags": ["cast-verification"]})
    anime.apply_project(project["id"], {"mode": "bangumi", "subject_id": args.subject}, dossier)
    characters = CharacterStore(library)
    assets = []
    existing = {asset.get("episode"): asset for asset in library.list_assets()}
    for number, path in paths:
        log(f"Importing TV episode {number}/{len(paths)}")
        sc = path.with_suffix(".sc.ass")
        duration = existing[number]["duration_ms"] if number in existing else media.probe(path)["duration_ms"]
        prepared_tracks = {}
        for suffix in (".sc.ass", ".tc.ass"):
            sidecar = path.with_suffix(suffix)
            if sidecar.is_file():
                prepared_tracks[suffix] = prepare_sidecar(sidecar, duration, args, number, suffix)
        if number in existing:
            asset = existing[number]
            if not args.resume or library.source_path(asset["id"]) != path:
                raise ValueError("Verification library contains an unexpected episode source")
        else:
            asset = library.register(AssetInput(video_path=str(path), project_id=project["id"],
                title=f"{project['name']} {number:02}", kind="animation", season=1, episode=number,
                subtitle_mode="external" if sc.exists() else "auto",
                **({"subtitle_path": str(prepared_tracks[".sc.ass"][0])} if sc.exists() else {})))
        assets.append(asset)
        cues = library.subtitles(asset["id"])
        tracks = []
        for suffix, (_, parsed, details) in prepared_tracks.items():
            tracks.append({**details, "cues": len(parsed), "passed": bool(parsed)
                          and all(0 <= cue["start_ms"] < cue["end_ms"] <= asset["duration_ms"] for cue in parsed)})
        def cue_projection(rows):
            return Counter((cue["start_ms"], cue["end_ms"], cue["text"]) for cue in rows)
        imported_matches = (cue_projection(cues) == cue_projection(prepared_tracks[".sc.ass"][1])
                            if ".sc.ass" in prepared_tracks else bool(cues))
        catalog = characters.catalog(asset["id"])
        names_complete = all(p.get("name_cn") and p.get("name_ja") and p.get("name_en")
                             for p in catalog["profiles"])
        names_searchable = all(p[key] in [p["name"], *p.get("aliases", [])]
                               for p in catalog["profiles"] for key in ("name_cn", "name_ja", "name_en")
                               if p.get(key))
        item = {"episode": number, "duration_ms": asset["duration_ms"], "subtitle_mode": asset["subtitle_mode"],
                "imported_cues": len(cues), "external_tracks": tracks, "roster_count": len(catalog["profiles"]),
                "names_complete": bool(names_complete), "scope_id": catalog["scope_id"],
                "derived_subtitle": any(track["derived_subtitle"] for track in tracks),
                "multilingual_names_in_aliases": names_searchable,
                "imported_matches_prepared_subtitle": imported_matches}
        report["episodes"].append(item)
        check(report, f"episode {number} subtitles, Bangumi-only cast, and multilingual names",
              bool(cues) and imported_matches and all(track["passed"] for track in tracks)
              and roster_ids(catalog["profiles"], args.subject) == source_ids
              and len(catalog["profiles"]) == len(source_ids) and names_complete and names_searchable)
        atomic_json(args.output / "report.json", report)
    check(report, "all episodes share one cast registry",
          len({episode["scope_id"] for episode in report["episodes"]}) == 1
          and len(library.list_assets()) == args.expected_episodes)
    rejected_tracks = [track for item in report["episodes"] for track in item["external_tracks"]
                       if track["source_validation"]["status"] == "rejected"]
    report["source_validation"] = {"unmodified_ass_passed": not rejected_tracks,
                                   "rejected_ass_files": len(rejected_tracks),
                                   "explicit_filter_enabled": args.filter_invalid_ass_events,
                                   "excluded_events": sum(track["source_validation"]["excluded_events"]
                                                          for track in rejected_tracks)}
    if rejected_tracks:
        report["limits"].append("Original ASS import rejected inverted karaoke intervals; only explicitly derived subtitles were imported for affected episodes.")
    first = assets[0]
    times = args.vision_times_ms or [1000]
    outputs = [library.asset_dir(first["id"]) / "frames" / f"verification_{at}.jpg" for at in times]
    frames = media.extract_frames(sources[0], times, outputs)
    for frame in frames:
        library.register_frame(first["id"], frame)
    # Complete the real provider probe before injecting synthetic legacy state.
    await vision_probe(report, args, library, characters, first, frames)
    policy_checks(report, library, characters, first, frames[0], source_ids, args.subject)
    check(report, "source media and both subtitle sidecars remain unchanged",
          all(source_snapshot(path) == value for path, value in snapshots.items()), files=len(snapshots))
    report["status"] = "passed" if all(item["passed"] for item in report["checks"]) else "failed"
    report["checks_passed"] = sum(item["passed"] for item in report["checks"])
    report["checks_failed"] = sum(not item["passed"] for item in report["checks"])
    atomic_json(args.output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--subject", type=parse_subject_reference, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=12)
    parser.add_argument("--episode-pattern", default=r"\[(\d{2})\]")
    parser.add_argument("--bangumi-timeout", type=float, default=30)
    parser.add_argument("--dossier-cache", type=Path, help="Explicit previously fetched public dossier; report labels cache use")
    parser.add_argument("--provider-settings", type=Path, help="Read existing private model settings; enables a real vision probe")
    parser.add_argument("--vision-times-ms", default="", help="Comma-separated increasing episode-1 times, at most 8 frames")
    parser.add_argument("--resume", action="store_true", help="Reuse only this verification output library")
    parser.add_argument("--filter-invalid-ass-events", action="store_true",
                        help="Explicitly derive local ASS fixtures omitting only nonpositive-duration events; never modifies original subtitles")
    args = parser.parse_args()
    try:
        args.vision_times_ms = [int(value) for value in args.vision_times_ms.split(",") if value]
        if (args.expected_episodes < 1 or args.bangumi_timeout <= 0 or len(args.vision_times_ms) > 8
                or args.vision_times_ms != sorted(set(args.vision_times_ms))
                or any(value < 0 for value in args.vision_times_ms)):
            raise ValueError("Episode count, timeout or bounded frame times are invalid")
        if args.provider_settings and not args.vision_times_ms:
            raise ValueError("The real provider probe requires explicit --vision-times-ms")
        report = asyncio.run(verify(args))
    except Exception as exc:
        log(f"Verification stopped: {type(exc).__name__}")
        if isinstance(exc, (subtitles.SubtitleError, ProviderError)):
            log(str(exc))
        return 1
    log(json.dumps({key: report[key] for key in ("status", "subject_id", "checks_passed", "checks_failed")}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())

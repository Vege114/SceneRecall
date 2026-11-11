"""Source-backed knowledge retrieval and authoritative film evidence projection."""
from __future__ import annotations

from collections import defaultdict
from time import perf_counter

from .execution import run_in_thread
from .knowledge_recall import KnowledgeRecallResult, KnowledgeSnapshot, WorkEntityRecall
from .knowledge_store import target_key
from .library import digest
from .search import SearchSnapshotChanged, normalize
from .search_timing import aggregate_operation, timed, timing_count, trace_span


def confirmed_character_evidence(record, identities, character_scope=None):
    """Only current v2 identity decisions may establish a character appearance."""
    evidence, bindings, matched = [], [], []
    current_frames = set(record.get("evidence_frame_ids", []))
    for entity in record.get("entities", []):
        binding = entity.get("identity_binding") or {}
        identity = entity.get("character_id")
        if (identity not in identities or binding.get("character_id") != identity
                or binding.get("status") not in {"human_confirmed", "auto_bound"}
                or binding.get("policy_version") != "identity-binding-v2" or not binding.get("id")
                or character_scope is not None and binding.get("character_scope_id") != character_scope):
            continue
        frames = entity.get("evidence_frame_ids", [])
        if not frames or not set(frames) <= current_frames:
            continue
        evidence.extend(frames)
        bindings.append(binding["id"])
        matched.append(identity)
    return list(dict.fromkeys(evidence)), list(dict.fromkeys(bindings)), list(dict.fromkeys(matched))


class KnowledgeService:
    def __init__(self, store, runtime):
        self.knowledge, self.runtime = store, runtime
        self.store = store.store
        self.library = self.store.library
        self.baseline = WorkEntityRecall(self.store)

    @timed("knowledge.scopes")
    def scopes(self, filters, engine):
        explicit = filters.get("_knowledge_scope")
        memberships = self.store._memberships()["assets"]
        if explicit:
            self.knowledge.scope_catalog(explicit)
        owners = engine.filtered_asset_ids(filters)
        timing_count("selected_assets", len(owners))
        scopes = sorted({memberships[asset] for asset in owners if asset in memberships})
        timing_count("selected_scopes", len(scopes))
        if explicit:
            asset_filter = filters.get("asset_id")
            if asset_filter and memberships.get(asset_filter) != explicit:
                return []
            return [explicit] if not scopes or explicit in scopes else []
        return scopes

    def _snapshot(self, filters, engine, scopes):
        snapshot = self.baseline._query_snapshot(filters, engine)
        allowed = set(scopes)
        assets = {asset: scope for asset, scope in snapshot.asset_scopes.items() if scope in allowed}
        return KnowledgeSnapshot(snapshot.token, {scope: catalog for scope, catalog in snapshot.catalogs.items()
                                 if scope in allowed}, assets,
                                 {asset: value for asset, value in snapshot.fingerprints.items() if asset in assets})

    @timed("knowledge.revision.identity")
    def _identity_token(self, scope):
        # Identity projection depends on decisions, references, calibration and
        # active receipts, not just character names. File identities avoid model IO.
        from .characters import scope_id
        paths = set()
        with aggregate_operation("scope_view"):
            assets = self.store.scope_view(scope)["asset_ids"]
        timing_count("scope_assets", len(assets))
        for asset in assets:
            try:
                asset_row = self.library.get_asset(asset)
            except (FileNotFoundError, ValueError):
                continue
            base = self.library.asset_dir(asset)
            identity = base / "identity" / "visual"
            # Immutable observations are authenticated by active.json; mutable
            # identity inputs have explicit publication files. Scanning every
            # crop/feature/history file makes revision checks scale with all
            # detections and is unrelated to the active evidence contract.
            paths.update([identity / "decisions.json", identity / "settings.json", identity / "regions",
                          identity / "matches", base / "retrieval" / "active.json",
                          base / "frames" / "registry.json", base / "manifest.json"])
            character_root = self.library.root / "works" / scope_id(asset_row)
            gallery = character_root / "reference-images"
            paths.update([character_root / "characters.json", gallery / "gallery.json", gallery])
            with aggregate_operation("file_stat", identity=str(gallery)):
                is_gallery = gallery.is_dir()
            if is_gallery:
                with aggregate_operation("file_glob", identity=str(gallery)):
                    paths.update(gallery.glob("*.png"))
        stamps = []
        timing_count("identity_paths", len(paths))
        for path in paths:
            try:
                with aggregate_operation("file_stat", identity=str(path)):
                    stat = path.stat()
                stamps.append((str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
            except FileNotFoundError:
                timing_count("file_stat_missing")
                stamps.append((str(path), None))
        return digest(sorted(stamps))

    def revision(self, filters, engine):
        scopes = self.scopes(filters, engine)
        timing_count("revision_scopes", len(scopes))
        values = []
        for scope in scopes:
            projection = self.knowledge.projection(scope)
            # Busy/error text are transient operational state, not index versions.
            with trace_span("knowledge.revision.runtime") as runtime_phase:
                status = self.runtime.status(projection)
                runtime_phase.set(enabled=bool(status.get("enabled")), configured=bool(status.get("configured")))
            runtime_version = {key: status.get(key) for key in (
                "enabled", "configured", "status", "generation", "signature", "version", "model", "embedding_dim")}
            values.append((scope, projection["signature"], runtime_version, self._identity_token(scope)))
        with trace_span("knowledge.revision.baseline"):
            baseline = self.baseline.revision(filters, engine)
        return digest([baseline, values, filters.get("_knowledge_mode", "baseline"),
                       filters.get("_knowledge_target")])

    def search(self, query, filters, engine):
        """Compatibility entry point for clients that explicitly request legacy recall."""
        return self.baseline.search(query, filters, engine)

    @staticmethod
    def _retrieved_cards(result, cards, alignments=()):
        """Resolve document provenance and explicitly reviewed graph identities.

        A returned display name is only a lookup key into this generation's
        tracking plus a human-reviewed alignment. It is never a name join to
        the formal concept/character catalog. Relationship keys are the literal
        directed ``source->target`` pair shown in the graph review UI.
        """
        data = result.get("data", {})
        if not isinstance(data, dict):
            raise ValueError("知识检索结果格式无效")
        if "data" in data:
            data = data["data"]
        mapping = result.get("mappings", {})
        if not isinstance(data, dict) or not isinstance(mapping, dict):
            raise ValueError("知识派生映射格式无效")
        chunks = mapping.get("chunks", {})
        tracking = result.get("graph_tracking", {})
        if not isinstance(chunks, dict) or not isinstance(tracking, dict):
            raise ValueError("知识派生映射格式无效")
        for collection in (data, tracking):
            for field in ("entities", "relationships", "chunks"):
                rows = collection.get(field, [])
                if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                    raise ValueError("知识派生映射格式无效")
        selected, unmapped = [], 0
        available = {row["card_id"]: row for row in cards
                     if row.get("review_status", "reviewed") == "reviewed"
                     and row.get("effective_status", "reviewed") == "reviewed"}

        def chunk_cards(chunk_id):
            entry = chunks.get(chunk_id)
            entries = entry if isinstance(entry, list) else [entry] if entry else []
            if any(not isinstance(item, dict) or not isinstance(item.get("card_ids", []), list)
                   or any(not isinstance(card_id, str) for card_id in item.get("card_ids", [])) for item in entries):
                raise ValueError("知识文本块映射格式无效")
            ids = list(dict.fromkeys(cid for item in entries for cid in item.get("card_ids", [])))
            return ids, [available[cid] for cid in ids if cid in available]

        def append_card(card, *, status, basis, upstream_key=None):
            existing = next((row for row in selected if row["card_id"] == card["card_id"]), None)
            if existing is None:
                existing = {**card, "mapping_status": status, "mapping_basis": basis}
                selected.append(existing)
            elif basis == "reviewed_graph_alignment":
                # An explicit, provenance-backed alignment can resolve a shared
                # document block's ambiguity for this one approved target.
                existing.update(mapping_status=status, mapping_basis=basis)
            if upstream_key is not None:
                keys = existing.setdefault("aligned_upstream_keys", [])
                if upstream_key not in keys:
                    keys.append(upstream_key)

        # Chunk provenance remains useful even when a graph node is unresolved.
        # These hits assert the reviewed card's primary target, not the node's
        # identity. Query-local reference_id is deliberately ignored.
        chunk_ids = [row.get("chunk_id") for row in data.get("chunks", [])]
        for row in [*data.get("entities", []), *data.get("relationships", [])]:
            chunk_ids.extend(str(row.get("source_id", "")).split("<SEP>"))
        for chunk_id in dict.fromkeys(chunk_ids):
            _, valid = chunk_cards(chunk_id)
            if not valid:
                if chunk_id:
                    unmapped += 1
                continue
            ambiguous = len({target_key(row["target_ref"]) for row in valid}) != 1
            for row in valid:
                append_card(row, status="ambiguous" if ambiguous else "reviewed", basis="document_card")

        def node_key(row):
            return row.get("entity_name") or row.get("id")

        def edge_key(row):
            source = row.get("src_id", row.get("source"))
            target = row.get("tgt_id", row.get("target"))
            return source + "->" + target if isinstance(source, str) and isinstance(target, str) else None

        tracked = defaultdict(list)
        for kind, key_function in (("entities", node_key), ("relationships", edge_key)):
            for row in tracking.get(kind, []):
                key = key_function(row)
                if isinstance(key, str) and key:
                    tracked[(kind, key)].append(row)
        reviewed = defaultdict(list)
        for alignment in alignments:
            if alignment.get("status") == "reviewed":
                reviewed[alignment.get("upstream_key")].append(alignment)

        for kind, key_function in (("entities", node_key), ("relationships", edge_key)):
            for returned in data.get(kind, []):
                key = key_function(returned)
                if not isinstance(key, str) or not key:
                    continue
                candidates = reviewed.get(key, [])
                # A repeated upstream name/edge or conflicting approved mapping
                # cannot become an identity merely because one row was first.
                provenance = tracked.get((kind, key), [])
                if len(candidates) != 1 or len(provenance) != 1:
                    continue
                alignment = candidates[0]
                support = alignment.get("supporting_card_ids", [])
                try:
                    targets = {target_key(target) for target in alignment.get("target_refs", [])}
                except (KeyError, TypeError, ValueError):
                    continue
                if len(targets) != 1 or not support or not set(support) <= set(available):
                    continue
                versions = alignment.get("supporting_card_versions", {})
                if any(versions.get(card_id) != available[card_id].get("version") for card_id in support):
                    continue
                full_chunks = provenance[0].get("supporting_chunk_ids", [])
                if (not isinstance(full_chunks, list) or not full_chunks
                        or any(not isinstance(chunk_id, str) for chunk_id in full_chunks)):
                    continue
                tracked_cards, tracking_valid = set(), True
                for chunk_id in full_chunks:
                    ids, valid = chunk_cards(chunk_id)
                    if not ids or len(valid) != len(ids):
                        tracking_valid = False
                        break
                    tracked_cards.update(ids)
                if not tracking_valid or not set(support) <= tracked_cards:
                    continue
                for card_id in support:
                    card = available[card_id]
                    if target_key(card["target_ref"]) not in targets:
                        continue
                    if kind == "relationships" and card.get("kind") != "relation":
                        continue
                    append_card(card, status="reviewed", basis="reviewed_graph_alignment", upstream_key=key)
        return selected, unmapped

    def _concept_records(self, hits, filters, engine, scopes=None, extra_selected=None):
        snapshot = self._snapshot(filters, engine, scopes if scopes is not None else self.scopes(filters, engine))
        selected = defaultdict(dict)
        for scope, concepts in (extra_selected or {}).items():
            selected[scope].update(concepts)
        for rank, hit in enumerate(hits, 1):
            target = hit["target_ref"]
            if target.get("kind") == "concept" and hit.get("mapping_status", "reviewed") == "reviewed":
                selected[hit["work_scope_id"]].setdefault(target["concept_id"], 1 / (60 + rank))
        ids = sorted({binding["record_id"] for scope, concepts in selected.items() if scope in snapshot.catalogs
                      for concept in concepts for binding in snapshot.catalogs[scope]["by_concept"].get(concept, [])
                      if snapshot.asset_scopes.get(binding["asset_id"]) == scope})
        records = engine.filtered_records(filters, record_ids=ids)
        return self.baseline.project(records, selected, snapshot, filters)

    def _character_records(self, hits, query, filters, engine, explicit=None):
        wanted, named = {}, {}
        for hit in hits:
            target = hit["target_ref"]
            if target.get("kind") != "character" or hit.get("mapping_status", "reviewed") != "reviewed":
                continue
            # Complex ability/relationship questions cannot expand all appearances.
            names = hit.get("target_names", []) or [hit.get("target_name", "")]
            if explicit and target_key(explicit) == target_key(target):
                wanted[target_key(target)] = target
            elif normalize(query) in {normalize(name) for name in names if name}:
                named[target_key(target)] = target
        if len(named) == 1:
            wanted.update(named)
        if not wanted:
            return []
        results = []
        memberships = self.store._memberships()["assets"]
        allowed_scopes = set(self.scopes(filters, engine))
        for target in wanted.values():
            scoped = {**filters, "character_id": target["character_id"]}
            candidates = engine.filtered_records(scoped)
            for owner in sorted({row["asset_id"] for row in candidates}):
                if memberships.get(owner) not in allowed_scopes:
                    continue
                ids = {row["id"] for row in candidates if row["asset_id"] == owner}
                for record in self.library.records(owner, record_ids=ids):
                    if record["id"] not in ids or not self.store._matches_filters(record, filters):
                        continue
                    evidence, bindings, matched = confirmed_character_evidence(
                        record, {target["character_id"]}, target["character_scope_id"])
                    if evidence:
                        results.append({**record, "match_basis": "knowledge_character_appearance",
                            "matched_character_ids": matched, "identity_binding_ids": bindings, "evidence_ids": evidence,
                            "matched_character_scope_id": target["character_scope_id"],
                            "event_verified": False, "match_type": "partial", "retrieval_branches": ["concept"]})
        return list({row["id"]: row for row in results}.values())[:None if filters.get("_all_candidates") else 50]

    async def asearch(self, query, filters, engine) -> KnowledgeRecallResult:
        started = perf_counter()
        all_candidates = bool(filters.get("_all_candidates"))
        scopes = await run_in_thread(self.scopes, filters, engine)
        snapshot = await run_in_thread(self._snapshot, filters, engine, scopes)
        legacy_selected, _ = await run_in_thread(self.baseline.candidate_refs, query, snapshot,
                                                limit=None if all_candidates else 5)
        lexical, semantic, diagnostics = [], [], {"mode": filters.get("_knowledge_mode", "baseline"), "degraded": []}
        projections = {}
        for scope in scopes:
            projections[scope] = await run_in_thread(self.knowledge.projection, scope)
            lexical.extend(await run_in_thread(self.knowledge.lexical_hits, scope, query,
                                               limit=None if all_candidates else 50, projection=projections[scope]))
        explicit = filters.get("_knowledge_target")
        if explicit:
            # An explicit navigation target must not depend on its display name
            # overlapping the user's question. It is still source/review gated.
            for projection in projections.values():
                for card in projection["cards"]:
                    if target_key(card["target_ref"]) == target_key(explicit) and card["card_id"] not in {
                            hit["card_id"] for hit in lexical}:
                        lexical.insert(0, {**card, "score": 0, "mapping_status": "reviewed",
                                           "retrieval_source": "explicit_target"})
        if diagnostics["mode"] == "lightrag":
            if len(scopes) != 1:
                diagnostics["degraded"].append("请选择单一作品版本后使用 LightRAG；已回退词法知识")
            else:
                from .knowledge_runtime import KnowledgeUnavailable
                if all_candidates:
                    diagnostics["degraded"].append(
                        "LightRAG 仍受 top_k=20、chunk_top_k=20 限制，不能保证穷尽；本地知识与影片候选未作数量截断")
                try:
                    response = await self.runtime.query(projections[scopes[0]], query)
                    current = await run_in_thread(self.knowledge.projection, scopes[0])
                    if current["signature"] != projections[scopes[0]]["signature"]:
                        raise SearchSnapshotChanged("知识来源或映射已变化，请重试")
                    semantic, unmapped = self._retrieved_cards(response, projections[scopes[0]]["cards"],
                                                              projections[scopes[0]].get("alignments", []))
                    diagnostics.update(response.get("diagnostics", {}), generation=response.get("generation"), unmapped=unmapped)
                except SearchSnapshotChanged:
                    raise
                except (KnowledgeUnavailable, OSError, ValueError, RuntimeError) as exc:
                    diagnostics["degraded"].append("LightRAG 不可用，已回退词法知识：" + str(exc))
        scores, by_card, target_scores = defaultdict(float), {}, defaultdict(float)
        lexical_targets = {scope: {"concept:" + identity: score for identity, score in concepts.items()}
                           for scope, concepts in legacy_selected.items()}
        for row in lexical:
            scoped_targets = lexical_targets.setdefault(row["work_scope_id"], {})
            identity = target_key(row["target_ref"])
            scoped_targets[identity] = max(scoped_targets.get(identity, -1), row.get("score", 0))
        if explicit:
            for scoped_targets in lexical_targets.values():
                if target_key(explicit) in scoped_targets:
                    scoped_targets[target_key(explicit)] = float("inf")
        for scope, ranking in lexical_targets.items():
            for rank, (identity, _) in enumerate(sorted(ranking.items(), key=lambda item: (-item[1], item[0])), 1):
                target_scores[(scope, identity)] += 1 / (60 + rank)
        for branch, ranking in (("lexical", lexical), ("lightrag", semantic)):
            diagnostics[branch + "_card_ids"] = [row["card_id"] for row in ranking]
            seen = set()
            for rank, row in enumerate(ranking, 1):
                key = (row["work_scope_id"], row["card_id"])
                by_card[key] = {**row, "mapping_status": row.get("mapping_status", "reviewed")}
                scores[key] += 1 / (60 + rank)
                target = (row["work_scope_id"], target_key(row["target_ref"]))
                if branch == "lightrag" and target not in seen and row.get("mapping_status", "reviewed") == "reviewed":
                    target_scores[target] += 1 / (60 + rank)
                    seen.add(target)
        allowed = set()
        for scope in scopes:
            allowed.update(key for key, _ in sorted(((key, score) for key, score in target_scores.items() if key[0] == scope),
                                                   key=lambda item: (-item[1], item[0]))[:None if all_candidates else 5])
        hits = [{**by_card[key], "knowledge_score": scores[key]} for key in sorted(scores, key=lambda key: (-scores[key], key))]
        film_hits = [hit for hit in hits if (hit["work_scope_id"], target_key(hit["target_ref"])) in allowed]
        mapping_started = perf_counter()
        selected = defaultdict(dict)
        for scope, identity in allowed:
            if identity.startswith("concept:"):
                selected[scope][identity[len("concept:"):]] = target_scores[(scope, identity)]
        records = await run_in_thread(self._concept_records, film_hits, filters, engine, scopes, selected)
        records.extend(await run_in_thread(self._character_records, film_hits, query, filters, engine,
                                          filters.get("_knowledge_target")))
        # Both paths share one target selection, one current projection and one outer
        # concept vote; legacy rows cannot overwrite the mapped evidence.
        unique = {}
        for record in records:
            unique.setdefault(record["id"], record)
        records = list(unique.values())[:None if all_candidates else 50]
        for hit in hits:
            target = hit["target_ref"]
            hit["has_film_evidence"] = any(target.get("concept_id") in row.get("matched_concept_ids", [])
                or target.get("character_id") in row.get("matched_character_ids", []) for row in records)
            hit["evidence_type"] = "source_statement"
        diagnostics.update(mapping_and_evidence_ms=(perf_counter() - mapping_started) * 1000,
                           total_ms=(perf_counter() - started) * 1000)
        return {"knowledge_hits": hits, "record_candidates": records, "diagnostics": diagnostics}

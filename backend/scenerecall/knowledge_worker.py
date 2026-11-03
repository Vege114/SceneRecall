"""LightRAG 1.5.7 stdio worker, executable without SceneRecall dependencies.

Only derived storage is writable here. Models are callbacks to the owning app;
no credentials, official concept IDs or occurrence decisions are stored by RAG.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid

PROTOCOL_OUTPUT = sys.stdout
sys.stdout = sys.stderr
MAX_MESSAGE = 32 * 1024 * 1024
VERSION = "1.5.7"


def send(value):
    line = json.dumps(value, ensure_ascii=False, allow_nan=False)
    if len(line.encode("utf-8")) > MAX_MESSAGE:
        raise ValueError("response exceeds protocol limit")
    PROTOCOL_OUTPUT.buffer.write((line + "\n").encode("utf-8"))
    PROTOCOL_OUTPUT.buffer.flush()


def read_stdin(descriptor):
    if os.name != "nt":
        return os.read(descriptor, 65536)
    # Windows CRT _read holds the descriptor lock while blocked. ML imports
    # call _isatty on stdin and take the same lock, even with unbuffered I/O.
    # ReadFile avoids both Python's and the CRT's stream locks.
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                               ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
    kernel.ReadFile.restype = wintypes.BOOL
    buffer = ctypes.create_string_buffer(65536)
    count = wintypes.DWORD()
    if not kernel.ReadFile(msvcrt.get_osfhandle(descriptor), buffer, len(buffer), ctypes.byref(count), None):
        if ctypes.get_last_error() in {109, 232}:
            return b""
        raise ctypes.WinError(ctypes.get_last_error())
    return buffer.raw[:count.value]


def probe():
    print("Checking LightRAG runtime dependencies", file=sys.stderr, flush=True)
    version = importlib.metadata.version("lightrag-hku")
    if version != VERSION:
        raise ValueError("LightRAG version must be " + VERSION)
    # Import on explicit connection check, not application startup. Missing
    # core packages fail here rather than during a user's first search.
    from lightrag import LightRAG  # noqa: F401
    print("LightRAG imported", file=sys.stderr, flush=True)
    import tiktoken
    import tiktoken.load

    def offline_file(*args, **kwargs):
        raise ValueError("Tokenizer cache is missing; run setup_knowledge.ps1")

    # read_file_cached reads the prepared cache before invoking this download
    # function. A cache miss is a setup error, never an implicit network call.
    tiktoken.load.read_file = offline_file
    tiktoken.encoding_for_model("gpt-4o-mini")
    dependencies = {name: importlib.metadata.version(name) for name in
                    ("lightrag-hku", "numpy", "nano-vectordb", "networkx", "tiktoken", "pydantic",
                     "fastembed", "onnxruntime", "tokenizers")}
    return {"version": version, "python": sys.version.split()[0], "dependencies": dependencies}


class Worker:
    def __init__(self):
        self.rag = None
        self.identity = None
        self.pending = {}
        self.operations = set()
        self.active = False
        self.parent_id = None
        self.local_model = self.local_tokenizer = None
        self.local_path = None

    def load_local(self, directory):
        root = Path(directory).resolve()
        if self.local_path is not None:
            if self.local_path != root:
                raise ValueError("local embedding snapshot is immutable within a worker")
            return
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if (manifest.get("model") != "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                or manifest.get("revision") != "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
                or manifest.get("dimension") != 384):
            raise ValueError("unsupported local embedding manifest")
        for name, expected in manifest["files"].items():
            if Path(name).name != name:
                raise ValueError("invalid local model filename")
            digest = hashlib.sha256()
            with (root / name).open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    digest.update(block)
            if digest.hexdigest() != expected["sha256"]:
                raise ValueError("local embedding file hash mismatch")
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer
        self.local_model = TextEmbedding(model_name=manifest["model"], specific_model_path=str(root),
            local_files_only=True, providers=["CPUExecutionProvider"], threads=2)
        self.local_tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.local_tokenizer.no_truncation()
        self.local_tokenizer.no_padding()
        self.local_path = root

    def local_vectors(self, directory, texts):
        import numpy as np
        self.load_local(directory)
        windows, groups = [], []
        for text in texts:
            encoded = self.local_tokenizer.encode(text, add_special_tokens=False)
            offsets = encoded.offsets
            indices = []
            if len(offsets) <= 126:
                indices.append((len(windows), max(1, len(offsets))))
                windows.append(text)
            else:
                # The selected model has a 128-token context. Preserve long
                # inputs with overlapping windows rather than silently losing
                # the end of a card or query to FastEmbed's truncation.
                for start in range(0, len(offsets), 100):
                    end = min(len(offsets), start + 126)
                    piece = text[offsets[start][0]:offsets[end - 1][1]]
                    indices.append((len(windows), end - start))
                    windows.append(piece)
                    if end == len(offsets):
                        break
            groups.append(indices)
        vectors = list(self.local_model.embed(windows, batch_size=32, parallel=None))
        output = []
        for indices in groups:
            vector = np.average(np.asarray([vectors[index] for index, _ in indices]), axis=0,
                                weights=[weight for _, weight in indices]).astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if not np.isfinite(vector).all() or not norm > 0:
                raise ValueError("invalid local embedding vector")
            output.append((vector / norm).tolist())
        return {"vectors": output, "diagnostics": {"input_count": len(texts), "window_count": len(windows)}}

    async def local_embed(self, params):
        started = time.perf_counter()
        result = await asyncio.to_thread(self.local_vectors, params["local_model_dir"], params["texts"])
        result["diagnostics"]["duration_ms"] = (time.perf_counter() - started) * 1000
        return result

    async def callback(self, method, params):
        identity = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[identity] = future
        try:
            send({"type": "callback", "id": identity, "parent_id": self.parent_id,
                  "method": method, "params": params})
            return await future
        finally:
            self.pending.pop(identity, None)

    async def initialize(self, params):
        probe()
        identity = params["identity"]
        if self.rag is not None:
            if identity != self.identity:
                raise ValueError("worker scope/generation is immutable")
            return
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        async def complete(prompt, system_prompt=None, history_messages=None, **kwargs):
            # Extract role is used for both extraction and merge summaries.
            # The upstream summarization system prompt identifies the latter;
            # retain role evidence rather than counting both as keywords.
            text = ((system_prompt or "") + "\n" + str(prompt)).lower()
            purpose = "knowledge_merge_summary" if "synthesize a list of descriptions" in text else "knowledge_extract"
            return await self.callback("complete", {"prompt": prompt, "system_prompt": system_prompt or "",
                "history_messages": history_messages or [], "purpose": purpose})

        async def keyword(prompt, system_prompt=None, history_messages=None, **kwargs):
            return await self.callback("complete", {"prompt": prompt, "system_prompt": system_prompt or "",
                "history_messages": history_messages or [], "purpose": "knowledge_keyword"})

        async def forbidden_answer(*args, **kwargs):
            raise ValueError("final answer generation is disabled")

        async def embed(texts, **kwargs):
            import numpy as np
            purpose = "knowledge_embed_query" if kwargs.get("context") == "query" else "knowledge_embed_document"
            if params.get("embedding_backend") == "local":
                result = await self.local_embed({"local_model_dir": params["local_model_dir"], "texts": list(texts)})
                send({"type": "event", "event": "model_finished", "purpose": purpose, "backend": "local",
                      "request_count": 0, "local_call_count": 1, "usage": None, "cost": 0,
                      **result["diagnostics"]})
                return np.asarray(result["vectors"], dtype=np.float32)
            vectors = await self.callback("embed", {"texts": list(texts), "purpose": purpose,
                                                    "context": kwargs.get("context")})
            return np.asarray(vectors, dtype=np.float32)

        workspace = "sr_" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:40]
        self.rag = LightRAG(working_dir=params["working_dir"], workspace=workspace,
            kv_storage="JsonKVStorage", doc_status_storage="JsonDocStatusStorage",
            graph_storage="NetworkXStorage", vector_storage="NanoVectorDBStorage",
            llm_model_func=forbidden_answer,
            role_llm_configs={"extract": {"func": complete, "max_async": 1},
                              "keyword": {"func": keyword, "max_async": 1}},
            embedding_func=EmbeddingFunc(embedding_dim=params["embedding_dim"], max_token_size=8192,
                                         func=embed, supports_asymmetric=True),
            embedding_func_max_async=1, embedding_batch_num=32, max_parallel_insert=1,
            chunk_token_size=1200, chunk_overlap_token_size=100,
            entity_extraction_use_json=True, enable_llm_cache=True)
        self.identity = identity
        await self.rag.initialize_storages()

    async def build(self, params):
        await self.initialize(params)
        cards = params["cards"]
        ids = ["sr_doc_" + card["card_id"] for card in cards]
        await self.rag.ainsert([card["text"] for card in cards], ids=ids,
                               file_paths=["sr_" + card["card_id"] + ".txt" for card in cards])
        documents, chunks = {}, {}
        for identity, card in zip(ids, cards):
            status = await self.rag.doc_status.get_by_id(identity)
            if not status or str(getattr(status.get("status"), "value", status.get("status"))) != "processed":
                raise ValueError("document processing did not complete")
            chunk_ids = status.get("chunks_list") or []
            if not chunk_ids:
                raise ValueError("processed document has no chunks")
            stored = await self.rag.text_chunks.get_by_ids(chunk_ids)
            if len(stored) != len(chunk_ids) or any(not row for row in stored):
                raise ValueError("document chunk tracking incomplete")
            documents[identity] = {"card_id": card["card_id"], "source_refs": card.get("source_refs", []),
                                   "projection_hash": hashlib.sha256(card["text"].encode()).hexdigest()}
            for chunk_id, row in zip(chunk_ids, stored):
                if row.get("full_doc_id") != identity:
                    raise ValueError("chunk belongs to another document")
                entry = chunks.setdefault(chunk_id, {"doc_id": identity, "card_ids": [],
                                                      "source_refs": card.get("source_refs", [])})
                if card["card_id"] not in entry["card_ids"]:
                    entry["card_ids"].append(card["card_id"])
        from lightrag.utils import make_relation_chunk_key
        entities, relationships = [], []
        for node in await self.rag.chunk_entity_relation_graph.get_all_nodes():
            tracking = await self.rag.entity_chunks.get_by_id(node["id"])
            entities.append({**node, "supporting_chunk_ids": (tracking or {}).get("chunk_ids", []),
                             "alignment_status": "unresolved"})
        for edge in await self.rag.chunk_entity_relation_graph.get_all_edges():
            tracking = await self.rag.relation_chunks.get_by_id(make_relation_chunk_key(edge["source"], edge["target"]))
            relationships.append({**edge, "supporting_chunk_ids": (tracking or {}).get("chunk_ids", []),
                                  "alignment_status": "unresolved"})
        return {"mappings": {"documents": documents, "chunks": chunks},
                "candidates": {"entities": entities, "relationships": relationships}}

    async def query(self, params):
        await self.initialize(params)
        from lightrag import QueryParam
        started = time.perf_counter()
        result = await self.rag.aquery_data(params["query"], QueryParam(mode=params.get("mode", "mix"),
            top_k=20, chunk_top_k=20, enable_rerank=False, include_references=True))
        # Empty, short and no-keyword cases can be failure objects rather than
        # exceptions. Preserve status so the application can distinguish them.
        return {"identity": self.identity, "data": result.get("data", {}),
                "diagnostics": {"retrieval_ms": (time.perf_counter() - started) * 1000,
                                "upstream_status": result.get("status"),
                                "metadata": result.get("metadata", {})}}

    async def request(self, row):
        identity = row["id"]
        if self.active:
            send({"id": identity, "error": "worker busy"})
            return
        self.active, self.parent_id = True, identity
        try:
            method, params = row["method"], row.get("params", {})
            if method == "probe":
                result = probe()
            elif method == "local_embed":
                result = await self.local_embed(params)
            elif method == "build":
                result = await self.build(params)
            elif method == "query":
                result = await self.query(params)
            elif method == "close":
                if self.rag is not None:
                    await self.rag.finalize_storages()
                result = {"closed": True}
            else:
                raise ValueError("unknown operation")
            send({"id": identity, "result": result})
        except Exception as exc:
            # Never forward source material, credentials or full framework
            # exception arguments across the application API boundary.
            print(type(exc).__name__ + ": knowledge operation failed", file=sys.stderr)
            send({"id": identity, "error": type(exc).__name__})
        finally:
            self.active, self.parent_id = False, None

    async def run(self):
        buffered = b""
        descriptor = sys.stdin.fileno()
        try:
            while received := await asyncio.to_thread(read_stdin, descriptor):
                # Do not hold sys.stdin's BufferedReader lock in a background
                # readline while importing ML packages. Several dependencies
                # inspect stdin.isatty() during import and would deadlock.
                buffered += received
                if len(buffered) > MAX_MESSAGE and b"\n" not in buffered:
                    raise ValueError("request exceeds protocol limit")
                while b"\n" in buffered:
                    line, buffered = buffered.split(b"\n", 1)
                    if len(line) > MAX_MESSAGE:
                        raise ValueError("request exceeds protocol limit")
                    row = json.loads(line)
                    if row.get("type") == "callback_result":
                        future = self.pending.get(row.get("id"))
                        if future is None or future.done():
                            raise ValueError("unknown callback response")
                        if row.get("error"):
                            future.set_exception(ValueError("model callback failed"))
                        else:
                            future.set_result(row.get("result"))
                    else:
                        task = asyncio.create_task(self.request(row))
                        self.operations.add(task)
                        task.add_done_callback(self.operations.discard)
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(EOFError("owner disconnected"))
            for task in list(self.operations):
                task.cancel()
            await asyncio.gather(*self.operations, return_exceptions=True)
            if self.rag is not None:
                with contextlib.suppress(Exception):
                    await self.rag.finalize_storages()


if __name__ == "__main__":
    # Import native runtimes before a Windows stdin read is outstanding. NumPy
    # DLL initialization inspects CRT standard handles and can otherwise block
    # behind a pending pipe read. All later callbacks use already loaded code.
    probe()
    from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: F401
    from lightrag.kg.networkx_impl import NetworkXStorage  # noqa: F401
    worker = Worker()
    if len(sys.argv) == 3 and sys.argv[1] == "--local-model-dir":
        worker.load_local(sys.argv[2])
    asyncio.run(worker.run())

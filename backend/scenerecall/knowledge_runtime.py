"""Supervised, optional LightRAG runtime. Formal knowledge never lives here."""
from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from .library import atomic_json, digest, now, read_json, safe_id
from .providers import ProviderError

LIGHTRAG_VERSION = "1.5.7"
LIGHTRAG_COMMIT = "28ff1b05f2ac3f3e6fa14dd2cd33656579bd0c9c"
IMPLEMENTATION = "knowledge-worker-v1"
MAX_MESSAGE = 32 * 1024 * 1024
LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LOCAL_REVISION = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"


class KnowledgeUnavailable(ValueError):
    def __init__(self, message, code="unavailable"):
        super().__init__(message)
        self.code = code


class KnowledgeRuntime:
    def __init__(self, library, providers):
        self.library, self.providers = library, providers
        self.path = library.root / "private" / "knowledge.json"
        self.root = library.root / "works" / "knowledge-indexes"
        self._process = self._reader = self._stderr = None
        self._key = None
        self._pending = {}
        self._callbacks = set()
        self._busy = False
        self._ledger = []
        self._receipt = None
        self._operation_id = None
        self._callback_limit = asyncio.Semaphore(1)
        self._last_error = None
        self._stderr_tail = ""
        self._protocol_error = ""

    def settings(self):
        executable = os.environ.get("SCENERECALL_KNOWLEDGE_PYTHON", "")
        return {"enabled": False, "python": executable, "text_profile_id": None,
                "embedding_profile_id": None, "text_model": "", "reasoning_effort": None, "text_thinking": None,
                "embedding_backend": "local", "local_model_dir": str(Path(__file__).resolve().parents[2]
                    / ".cache" / "knowledge-models" / "multilingual-minilm-l12-v2"),
                "embedding_dim": None, "query_timeout_s": 60.0, **read_json(self.path, {})}

    def configure(self, payload):
        allowed = {"enabled", "python", "text_profile_id", "embedding_profile_id", "text_model",
                   "reasoning_effort", "text_thinking", "query_timeout_s", "embedding_backend", "local_model_dir"}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError("知识配置包含未声明字段")
        value = {**self.settings(), **payload}
        if type(value["enabled"]) is not bool:
            raise ValueError("知识启用状态无效")
        if value["embedding_backend"] not in {"local", "api"}:
            raise ValueError("知识 embedding 后端须为 local 或 api")
        for key in ("python", "text_model", "local_model_dir"):
            if not isinstance(value[key], str) or len(value[key]) > 2000:
                raise ValueError("知识模型或 Python 配置无效")
        for key in ("text_profile_id", "embedding_profile_id"):
            if value[key] is not None:
                safe_id(value[key])
        timeout = value["query_timeout_s"]
        if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 1 <= timeout <= 600:
            raise ValueError("知识查询超时须为 1–600 秒")
        if value["reasoning_effort"] not in {None, "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}:
            raise ValueError("知识模型思考等级无效")
        if value["text_thinking"] not in {None, "enabled", "disabled"}:
            raise ValueError("知识文本 thinking 选项无效")
        if any(value.get(key) != self.settings().get(key) for key in allowed - {"query_timeout_s", "enabled"}):
            value.pop("preflight_signature", None)
            value["embedding_dim"] = None
        atomic_json(self.path, value)
        return self.settings()

    def _configuration(self, *, require_preflight=True):
        value = self.settings()
        if not value["enabled"]:
            raise KnowledgeUnavailable("知识后端未启用", "disabled")
        executable = shutil.which(value["python"]) if value["python"] else None
        if not executable:
            raise KnowledgeUnavailable("请先安装并选择独立知识 Python 环境", "configuration")
        try:
            text = self.providers.get(value["text_profile_id"])
        except (ProviderError, KeyError, TypeError):
            raise KnowledgeUnavailable("请配置知识文本连接", "configuration") from None
        if "query" not in text.get("capabilities", []):
            raise KnowledgeUnavailable("文本连接需要 query 能力", "configuration")
        if value["text_thinking"] is not None and text.get("provider_type") == "codex_cli":
            raise KnowledgeUnavailable("Codex 连接使用思考等级，不支持 API thinking 选项", "configuration")
        local_identity = None
        if value["embedding_backend"] == "api":
            try:
                embedding = self.providers.get(value["embedding_profile_id"])
            except (ProviderError, KeyError, TypeError):
                raise KnowledgeUnavailable("请配置 embedding API 连接", "configuration") from None
            if embedding.get("provider_type") == "codex_cli" or "embedding" not in embedding.get("capabilities", []):
                raise KnowledgeUnavailable("embedding 必须使用独立 API 连接", "configuration")
        else:
            directory = Path(value["local_model_dir"]).expanduser()
            if not directory.is_absolute():
                raise KnowledgeUnavailable("本地 embedding 模型须为绝对路径", "configuration")
            manifest = read_json(directory / "manifest.json", {})
            if (manifest.get("model") != LOCAL_MODEL or manifest.get("revision") != LOCAL_REVISION
                    or manifest.get("dimension") != 384 or not manifest.get("files")):
                raise KnowledgeUnavailable("请先运行 download_knowledge_embedding.py 准备固定版本模型", "configuration")
            stamps = {}
            for filename in manifest["files"]:
                if Path(filename).name != filename:
                    raise KnowledgeUnavailable("本地 embedding 文件清单无效", "configuration")
                try:
                    info = (directory / filename).stat()
                except FileNotFoundError:
                    raise KnowledgeUnavailable("本地 embedding 模型文件缺失", "configuration") from None
                stamps[filename] = [info.st_size, info.st_mtime_ns]
            local_identity = {"manifest": manifest, "files": stamps, "path": str(directory.resolve())}
            embedding = {"id": "local_multilingual_minilm", "provider_type": "local", "model": LOCAL_MODEL}
        model = value["text_model"] or text.get("model")
        if not isinstance(model, str) or not model.strip():
            raise KnowledgeUnavailable("请选择明确的知识文本模型后测试连接", "configuration")
        if text.get("provider_type") != "codex_cli" and model != text.get("model"):
            raise KnowledgeUnavailable("API 模型须与所选连接的模型一致", "configuration")
        text = {**text, "model": model, "reasoning_effort": value["reasoning_effort"]
                if value["reasoning_effort"] is not None else text.get("reasoning_effort")}
        identity = {"implementation": IMPLEMENTATION, "lightrag": LIGHTRAG_VERSION,
                    "text": {**{k: text.get(k) for k in ("id", "provider_type", "base_url", "model", "reasoning_effort")},
                             "thinking": value["text_thinking"]},
                    "embedding": {k: embedding.get(k) for k in ("id", "provider_type", "base_url", "model")},
                    "embedding_dim": value.get("embedding_dim"), "chunk_size": 1200, "chunk_overlap": 100,
                    "entity_extraction_use_json": True, "top_k": 20, "chunk_top_k": 20, "rerank": False,
                    "runtime": value.get("runtime"),
                    "local_embedding": local_identity,
                    "worker_hash": digest(Path(__file__).with_name("knowledge_worker.py").read_text(encoding="utf-8"))}
        if require_preflight and (not value.get("embedding_dim") or value.get("preflight_signature") != digest(identity)):
            raise KnowledgeUnavailable("知识模型配置已变化，请重新测试连接", "configuration")
        return {**value, "python": executable, "text_profile": text, "embedding_profile": embedding,
                "identity": identity}

    def signature(self, projection):
        config = self._configuration()
        return digest([projection["signature"], config["identity"]])

    def _scope_root(self, projection):
        return self.root / safe_id(projection["work_scope_id"]) / safe_id(projection["work_version_id"])

    def status(self, projection=None):
        result = {"enabled": self.settings()["enabled"], "busy": self._busy,
                  "version": LIGHTRAG_VERSION, "commit": LIGHTRAG_COMMIT, "error": self._last_error}
        try:
            config = self._configuration()
            result.update(configured=True, model=config["text_profile"]["model"],
                          embedding_dim=config["embedding_dim"])
            if projection is not None:
                manifest = read_json(self._scope_root(projection) / "active.json", {})
                expected = self.signature(projection)
                ready = manifest.get("signature") == expected and manifest.get("status") == "ready"
                result.update(status="ready" if ready else "stale" if manifest else "not_built",
                              generation=manifest.get("generation"), signature=expected)
        except (KnowledgeUnavailable, ValueError, OSError) as exc:
            result.update(configured=False, status=getattr(exc, "code", "configuration"), error=str(exc))
        return result

    def _enter(self, operation):
        if self._busy:
            raise KnowledgeUnavailable("知识后端正在处理其他任务，已回退词法", "busy")
        self._busy = True
        self._operation_id = uuid.uuid4().hex
        self._ledger = []
        self._receipt = self.root / "receipts" / (self._operation_id + ".jsonl")
        self._record({"event": "operation_started", "operation": operation})

    def _record(self, value):
        row = {"operation_id": self._operation_id, "at": now(), **value}
        self._ledger.append(row)
        if self._receipt:
            self._receipt.parent.mkdir(parents=True, exist_ok=True)
            with self._receipt.open("a", encoding="utf-8") as output:
                output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())

    async def _send(self, value):
        encoded = (json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
        if len(encoded) > MAX_MESSAGE or self._process is None:
            raise KnowledgeUnavailable("知识消息过大或工作进程未就绪", "protocol")
        self._process.stdin.write(encoded)
        await self._process.stdin.drain()

    async def _start(self, config, key):
        if self._process is not None and self._key == key and self._process.returncode is None:
            return
        await self._stop()
        command = [config["python"], "-u", str(Path(__file__).with_name("knowledge_worker.py"))]
        if config["embedding_backend"] == "local":
            command.extend(["--local-model-dir", config["local_model_dir"]])
        if os.name == "nt":
            command = [getattr(sys, "_base_executable", sys.executable), "-I",
                       str(Path(__file__).with_name("_windows_process.py")), *command]
        try:
            environment = {**os.environ, "TIKTOKEN_CACHE_DIR": str(Path(config["python"]).parent.parent / "tokenizer-cache"),
                           "PIP_NO_INDEX": "1", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
            self._process = await asyncio.create_subprocess_exec(
                *command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, limit=MAX_MESSAGE,
                env=environment,
                creationflags=0x08000000 if os.name == "nt" else 0)
        except (OSError, RuntimeError) as exc:
            raise KnowledgeUnavailable("无法启动独立知识进程；请检查 Python 环境", "worker") from exc
        self._key = key
        self._reader = asyncio.create_task(self._read(config))
        self._stderr = asyncio.create_task(self._discard_stderr())

    @staticmethod
    def _embedding_options(config):
        return {"embedding_backend": config["embedding_backend"],
                "local_model_dir": config.get("local_model_dir")}

    async def _discard_stderr(self):
        # Framework diagnostics may contain source text. Do not expose raw logs
        # as public API errors, or let a full pipe deadlock the worker.
        process = self._process
        while block := await process.stderr.read(8192):
            self._stderr_tail = (self._stderr_tail + block.decode("utf-8", errors="replace"))[-8192:]

    async def _read(self, config):
        process = self._process
        try:
            while line := await process.stdout.readline():
                if len(line) > MAX_MESSAGE:
                    raise ValueError("oversized protocol message")
                row = json.loads(line)
                if row.get("type") == "callback":
                    task = asyncio.create_task(self._callback(row, config))
                    self._callbacks.add(task)
                    task.add_done_callback(self._callbacks.discard)
                elif row.get("type") == "event":
                    self._record({key: value for key, value in row.items() if key != "type"})
                elif row.get("id") in self._pending:
                    future = self._pending[row["id"]]
                    if not future.done():
                        if row.get("error"):
                            future.set_exception(KnowledgeUnavailable("知识工作进程处理失败：" + row["error"], "worker"))
                        else:
                            future.set_result(row.get("result"))
                else:
                    raise ValueError("unexpected protocol message")
        except (ValueError, OSError, asyncio.IncompleteReadError) as exc:
            self._protocol_error = type(exc).__name__ + ": " + str(exc) + " / " + repr(locals().get("line", b"")[:300])
            self._last_error = "知识工作进程通信中断"
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(KnowledgeUnavailable("知识工作进程已退出", "worker"))

    async def _callback(self, row, config):
        callback_id = row.get("id")
        started = time.perf_counter()
        params = row.get("params", {})
        purpose = params.get("purpose", "knowledge_extract")
        profile = config["embedding_profile"] if row.get("method") == "embed" else config["text_profile"]
        self._record({"event": "model_started", "callback_id": callback_id, "purpose": purpose,
                      "profile_id": profile["id"], "model": profile["model"]})
        result = None
        error = None
        failure = None
        try:
            async with self._callback_limit:
                # API adapters read live profiles. Reject changes before a
                # call instead of labeling a different live model as frozen.
                if self._configuration()["identity"] != config["identity"]:
                    raise KnowledgeUnavailable("模型连接在任务期间发生变化", "configuration")
                if row.get("method") == "embed":
                    result = await self.providers.embed(profile["id"], params["texts"])
                    if any(len(vector) != config["embedding_dim"] for vector in result.data):
                        raise KnowledgeUnavailable("embedding 维度已变化，请重新测试连接并建库", "configuration")
                elif row.get("method") == "complete":
                    with self.providers.use_profile(profile):
                        result = await self.providers.complete_text(profile["id"], system_prompt=params.get("system_prompt", ""),
                            prompt=params["prompt"], history_messages=params.get("history_messages", []), purpose=purpose,
                            thinking_mode=config["text_thinking"])
                else:
                    raise KnowledgeUnavailable("知识工作进程请求了不支持的模型操作", "protocol")
            await self._send({"type": "callback_result", "id": callback_id, "result": result.data})
        except asyncio.CancelledError:
            error = "cancelled; in-flight usage unavailable"
            raise
        except Exception as exc:
            failure = exc
            error = "模型回调失败"
            with contextlib.suppress(Exception):
                await self._send({"type": "callback_result", "id": callback_id, "error": error})
        finally:
            self._record({"event": "model_finished", "callback_id": callback_id, "purpose": purpose,
                          "duration_ms": (time.perf_counter() - started) * 1000,
                          "request_count": result.request_count if result else getattr(failure, "request_count", None),
                          "usage": result.usage if result else getattr(failure, "usage", None),
                          "cost": result.cost if result else getattr(failure, "cost", None),
                          "usage_incomplete": result is None,
                          "error": error})

    async def _call(self, method, params, timeout):
        identity = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending[identity] = future
        try:
            async with asyncio.timeout(timeout):
                await self._send({"id": identity, "method": method, "params": params})
                return await future
        except TimeoutError:
            await self._stop()
            raise KnowledgeUnavailable("知识检索或模型调用超时，已回退词法", "timeout") from None
        except asyncio.CancelledError:
            await self._stop()
            raise
        except KnowledgeUnavailable as exc:
            await self._stop()
            exc.diagnostics = {"calls": copy.deepcopy(self._ledger), "operation_id": self._operation_id}
            raise
        finally:
            self._pending.pop(identity, None)

    async def _stop(self):
        process, self._process = self._process, None
        if process is not None:
            if process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()
            await process.wait()
        for task in (self._reader, self._stderr, *list(self._callbacks)):
            if task is not None:
                task.cancel()
        await asyncio.gather(*(task for task in (self._reader, self._stderr, *list(self._callbacks))
                               if task is not None), return_exceptions=True)
        self._reader = self._stderr = self._key = None
        self._callbacks.clear()

    async def close(self):
        if self._process is not None and not self._busy:
            with contextlib.suppress(Exception):
                await self._call("close", {}, 5)
        await self._stop()

    async def preflight(self):
        self._enter("preflight")
        active_model = None
        try:
            config = self._configuration(require_preflight=False)
            await self._start(config, ("preflight",))
            probe = await self._call("probe", {}, 60)
            if probe.get("version") != LIGHTRAG_VERSION:
                raise KnowledgeUnavailable("知识环境中的 LightRAG 版本不匹配", "configuration")
            started = time.perf_counter()
            active_model = "knowledge_preflight"
            with self.providers.use_profile(config["text_profile"]):
                result = await self.providers.complete_text(config["text_profile_id"], system_prompt="Return the word ready.",
                    prompt="Connection test.", purpose="knowledge_preflight", thinking_mode=config["text_thinking"])
            self._record({"event": "model_finished", "purpose": "knowledge_preflight", "usage": result.usage,
                          "request_count": result.request_count, "cost": result.cost,
                          "duration_ms": (time.perf_counter() - started) * 1000})
            active_model = None
            if config["embedding_backend"] == "local":
                embedded = await self._call("local_embed", {**self._embedding_options(config),
                    "texts": ["知识连接测试 / knowledge / 知識"]}, 90)
                dimension = len(embedded["vectors"][0])
                self._record({"event": "model_finished", "purpose": "knowledge_embed_preflight",
                              "backend": "local", "usage": None, "request_count": 0, "cost": 0,
                              **embedded["diagnostics"]})
            else:
                active_model = "knowledge_embed_preflight"
                result = await self.providers.embed(config["embedding_profile_id"], ["知识连接测试 / knowledge / 知識"])
                self._record({"event": "model_finished", "purpose": "knowledge_embed_preflight", "usage": result.usage,
                              "request_count": result.request_count, "cost": result.cost})
                dimension = len(result.data[0])
                active_model = None
            config["identity"]["embedding_dim"] = dimension
            config["identity"]["runtime"] = probe
            value = {**self.settings(), "embedding_dim": dimension,
                     "preflight_signature": digest(config["identity"]), "tested_at": now(),
                     "runtime": probe}
            atomic_json(self.path, value)
            self._last_error = None
            return {"status": "ready", "embedding_dim": dimension, "runtime": probe, "calls": list(self._ledger)}
        except Exception as exc:
            if active_model:
                self._record({"event": "model_finished", "purpose": active_model, "error": "连接测试失败",
                              "request_count": getattr(exc, "request_count", None),
                              "usage": getattr(exc, "usage", None), "cost": getattr(exc, "cost", None),
                              "usage_incomplete": True})
            exc.diagnostics = {"calls": copy.deepcopy(self._ledger), "operation_id": self._operation_id}
            self._last_error = str(exc) if isinstance(exc, (KnowledgeUnavailable, ProviderError)) else "知识连接测试失败"
            raise
        finally:
            await self._stop()
            self._record({"event": "operation_finished"})
            self._busy = False

    async def build(self, projection, progress=None, is_current=None):
        self._enter("build")
        started = time.perf_counter()
        try:
            config = self._configuration()
            if not projection.get("cards"):
                raise KnowledgeUnavailable("尚无已核对且来源有效的资料卡", "empty")
            signature = self.signature(projection)
            generation = "gen_" + uuid.uuid4().hex[:20]
            directory = self._scope_root(projection) / generation
            directory.mkdir(parents=True, exist_ok=False)
            identity = {"work_scope_id": projection["work_scope_id"],
                        "work_version_id": projection["work_version_id"], "generation": generation}
            await self._start(config, tuple(identity.values()))
            if progress:
                value = progress({"stage": "knowledge_index", "completed": 0, "total": len(projection["cards"])})
                if hasattr(value, "__await__"):
                    await value
            result = await self._call("build", {"identity": identity, "working_dir": str(directory),
                **self._embedding_options(config), "embedding_dim": config["embedding_dim"], "cards": projection["cards"]},
                max(600, len(projection["cards"]) * 600))
            mapping = result["mappings"]
            if set(mapping.get("documents", {})) != {"sr_doc_" + row["card_id"] for row in projection["cards"]}:
                raise KnowledgeUnavailable("知识文档映射不完整，未发布索引", "mapping")
            atomic_json(directory / "mappings.json", mapping)
            atomic_json(directory / "graph-candidates.json", result.get("candidates", {}))
            manifest = {"status": "ready", "version": LIGHTRAG_VERSION, "implementation": IMPLEMENTATION,
                        **identity, "signature": signature, "knowledge_signature": projection["signature"],
                        "model_identity": config["identity"], "created_at": now(),
                        "duration_ms": (time.perf_counter() - started) * 1000,
                        "card_count": len(projection["cards"]), "chunk_count": len(mapping.get("chunks", {})),
                        "diagnostics": copy.deepcopy(self._ledger)}
            atomic_json(directory / "manifest.json", manifest)
            # Source writes also run on FastAPI worker threads. Hold the
            # authoritative lock across the final check and pointer publish.
            with getattr(self.library, "lock", contextlib.nullcontext()):
                if (is_current is not None and not is_current()) or signature != self.signature(projection):
                    raise KnowledgeUnavailable("建库期间知识或模型已变化，未发布旧索引", "stale")
                atomic_json(self._scope_root(projection) / "active.json", manifest)
            if progress:
                value = progress({"stage": "knowledge_index", "completed": len(projection["cards"]),
                                  "total": len(projection["cards"])})
                if hasattr(value, "__await__"):
                    await value
            self._last_error = None
            return manifest
        except Exception as exc:
            exc.diagnostics = {"calls": copy.deepcopy(self._ledger), "operation_id": self._operation_id}
            raise
        finally:
            self._record({"event": "operation_finished"})
            self._busy = False

    async def query(self, projection, query, mode="mix"):
        self._enter("query")
        started = time.perf_counter()
        try:
            config = self._configuration()
            signature = self.signature(projection)
            manifest = read_json(self._scope_root(projection) / "active.json", {})
            if manifest.get("status") != "ready" or manifest.get("signature") != signature:
                raise KnowledgeUnavailable("知识索引未建立或已过期，已回退词法", "stale")
            if mode not in {"mix", "naive"}:
                raise ValueError("首轮仅支持 mix 和评估用 naive")
            identity = {key: manifest[key] for key in ("work_scope_id", "work_version_id", "generation")}
            directory = self._scope_root(projection) / safe_id(manifest["generation"])
            mappings = read_json(directory / "mappings.json")
            graph_tracking = read_json(directory / "graph-candidates.json", {"entities": [], "relationships": []})
            if not isinstance(mappings, dict):
                raise KnowledgeUnavailable("知识映射文件缺失", "mapping")
            async with asyncio.timeout(config["query_timeout_s"]):
                await self._start(config, tuple(identity.values()))
                result = await self._call("query", {"identity": identity, "working_dir": str(directory),
                    **self._embedding_options(config), "embedding_dim": config["embedding_dim"],
                    "query": query, "mode": mode}, config["query_timeout_s"])
            if signature != self.signature(projection):
                raise KnowledgeUnavailable("查询期间知识模型配置已变化", "stale")
            if result.get("identity") != identity:
                raise KnowledgeUnavailable("知识结果作用域不匹配", "scope")
            self._last_error = None
            return {"data": result["data"], "mappings": mappings, "graph_tracking": graph_tracking,
                    "generation": manifest["generation"],
                    "diagnostics": {"mode": mode, "duration_ms": (time.perf_counter() - started) * 1000,
                                    "calls": copy.deepcopy(self._ledger), **result.get("diagnostics", {})}}
        except TimeoutError:
            await self._stop()
            raise KnowledgeUnavailable("知识查询超时，已回退词法", "timeout") from None
        except asyncio.CancelledError:
            await self._stop()
            raise
        except Exception as exc:
            exc.diagnostics = {"calls": copy.deepcopy(self._ledger), "operation_id": self._operation_id}
            raise
        finally:
            self._record({"event": "operation_finished"})
            self._busy = False

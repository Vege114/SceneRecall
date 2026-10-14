"""Public, bounded Bangumi reads through the project's pinned bgm-cli dependency."""
from __future__ import annotations

import asyncio
import copy
import json
import math
import os
import re
import shutil
import signal
import tempfile
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLI_ENTRYPOINT = PROJECT_ROOT / "node_modules/@aronnaxlin/bgm-cli/src/cli.js"
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 64 * 1024
PAGE_SIZE = 100
MAX_CHARACTERS = 1000
NAME_SCHEMA_VERSION = 1
_HOSTS = {"bgm.tv", "bangumi.tv", "chii.in", "www.bgm.tv", "www.bangumi.tv", "www.chii.in"}
_CHINESE_NAME_KEYS = {"中文名", "简体中文名", "繁体中文名", "第二中文名", "中文名称"}
_JAPANESE_NAME_KEYS = {"日文名", "日本語名", "日本名"}
_ENGLISH_NAME_KEYS = {"英文名", "英语名", "English name"}
_ROMAN_NAME_KEYS = {"罗马字", "罗马音", "罗马字名", "Romanized name", "Romaji"}
_READING_KEYS = {"纯假名", "假名", "读音", "ふりがな"}
_ALIAS_KEYS = {"别名", "别称", "昵称"}
_NAME_KEYS = (_CHINESE_NAME_KEYS | _JAPANESE_NAME_KEYS | _ENGLISH_NAME_KEYS | _ROMAN_NAME_KEYS
              | _READING_KEYS | _ALIAS_KEYS)


class BangumiError(ValueError):
    """Actionable failure safe to display to a local application user."""

    def __init__(self, message: str, *, code: str = "bangumi_error"):
        super().__init__(message)
        self.code = code


def parse_subject_reference(value: str | int) -> int:
    """Accept a positive subject ID or a subject URL on a known Bangumi host."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise BangumiError("请输入 Bangumi 条目链接或正整数条目 ID", code="invalid_reference")
    reference = str(value).strip()
    if len(reference) > 2048:
        raise BangumiError("Bangumi 条目链接或 ID 过长", code="invalid_reference")
    if re.fullmatch(r"[0-9]+", reference):
        subject_id = int(reference)
    else:
        try:
            url = urlsplit(reference)
            match = re.fullmatch(r"/subject/([0-9]+)/?", url.path)
            valid = (url.scheme in {"http", "https"} and url.hostname in _HOSTS and match
                     and url.username is None and url.password is None and url.port is None)
        except ValueError:
            valid = False
        if not valid:
            raise BangumiError("请输入 bgm.tv、bangumi.tv 或 chii.in 的 /subject/ 条目链接",
                               code="invalid_reference")
        subject_id = int(match.group(1))
    if not 0 < subject_id <= 2**53 - 1:
        raise BangumiError("Bangumi 条目 ID 必须是有效的正整数", code="invalid_reference")
    return subject_id


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _id(value: Any) -> int:
    if isinstance(value, bool):
        raise BangumiError("Bangumi 返回的资料缺少有效 ID", code="invalid_response")
    try:
        identifier = int(value)
    except (ValueError, TypeError, OverflowError):
        raise BangumiError("Bangumi 返回的资料缺少有效 ID", code="invalid_response") from None
    if identifier <= 0 or str(identifier) != str(value):
        raise BangumiError("Bangumi 返回的资料缺少有效 ID", code="invalid_response")
    return identifier


def _image(value: dict) -> str | None:
    images = value.get("images") or {}
    if not isinstance(images, dict):
        return None
    for size in ("large", "common", "medium", "small", "grid"):
        image = _text(images.get(size))
        if image.startswith("//"):
            image = "https:" + image
        if image.startswith(("https://", "http://")):
            return image
    return None


def _infobox_entries(infobox: Any):
    """Retain the nested k labels used by both Bangumi API infobox formats."""
    for entry in infobox if isinstance(infobox, list) else []:
        if not isinstance(entry, dict):
            continue
        key = _text(entry.get("key"))
        raw = entry.get("values", entry.get("value", []))
        for value in raw if isinstance(raw, list) else [raw]:
            text = _text(value.get("v", value.get("value"))) if isinstance(value, dict) else _text(value)
            label = _text(value.get("k", value.get("key"))) if isinstance(value, dict) else ""
            if text:
                yield key, label or key, text


def _infobox_values(infobox: Any, keys: set[str]) -> list[str]:
    values = []
    for key, label, text in _infobox_entries(infobox):
        if key in keys or label in keys:
            if text not in values:
                values.append(text)
    return values


def _kana(value: str) -> bool:
    return bool(re.fullmatch(r"[\u3041-\u3096\u30a1-\u30faー\s・･]+", value))


def _han_name(value: str) -> bool:
    return bool(re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\s·・]+", value))


def _latin_name(value: str) -> bool:
    letters = [char for char in value if char.isalpha()]
    return bool(letters) and all("LATIN" in unicodedata.name(char, "") for char in letters)


@lru_cache(maxsize=1)
def _kana_converter():
    from pykakasi import kakasi
    return kakasi()


@lru_cache(maxsize=1)
def _han_converter():
    from opencc import OpenCC
    return OpenCC("t2s")


def _romanize_reading(reading: str) -> str:
    # Only transliterate an existing kana reading. Kanji dictionaries can guess
    # the wrong personal-name reading, and a transliteration is not an official
    # English localization. Preserve the source's word order and separators.
    if not _kana(reading):
        return ""
    words = re.split(r"[\s・･]+", reading.strip())
    result = " ".join("".join(part["hepburn"] for part in _kana_converter().convert(word)).capitalize()
                      for word in words if word)
    return result if _latin_name(result) else ""


def character_names(raw: dict) -> dict:
    """Source-backed multilingual display names, with explicit conversion provenance."""
    url = _text(raw.get("url")) or f"https://bgm.tv/character/{_id(raw.get('id'))}"
    entries = [(label, text) for key, label, text in _infobox_entries(raw.get("infobox"))
               if key in _NAME_KEYS or label in _NAME_KEYS]
    variants = []
    for variant in raw.get("name_variants", []) or []:
        if (isinstance(variant, dict) and _text(variant.get("name"))
                and all(isinstance(variant.get(field), str)
                        for field in ("language", "kind", "source", "source_url"))):
            variants.append(copy.deepcopy(variant))

    def add(name, language, kind="alias", source="bangumi", *, preserve=False):
        if not name:
            return
        # A derived name loaded from a cache must retain its original provenance.
        if any(item["name"] == name and item["language"] == language
               and (preserve or item["source"] == source) for item in variants):
            return
        variants.append({"name": name, "language": language, "kind": kind,
                         "source": source, "source_url": url})

    for label, text in entries:
        if label in _CHINESE_NAME_KEYS:
            add(text, "zh-Hant" if label == "繁体中文名" else "zh-Hans")
        elif label in _JAPANESE_NAME_KEYS:
            add(text, "ja")
        elif label in _ENGLISH_NAME_KEYS:
            add(text, "en")
        elif label in _ROMAN_NAME_KEYS:
            add(text, "ja-Latn", "romanization")
        elif label in _READING_KEYS:
            add(text, "ja", "reading")
        else:
            add(text, "ja" if re.search(r"[\u3041-\u30fa]", text) else "und")

    def first(keys):
        return next((text for label, text in entries if label in keys), "")

    def stored(field, language):
        value = _text(raw.get(field))
        # Older normalized snapshots may have a derived display name. Fresh
        # infobox names should replace it, while repeated local upgrades must
        # never relabel a conversion as an upstream name.
        if any(item["name"] == value and item["language"] == language and item["source"] == "bangumi"
               for item in variants):
            return value
        if any(item["name"] == value and item["source"] != "bangumi" for item in variants):
            return ""
        return value

    original = _text(raw.get("name"))
    chinese = (_text(raw.get("nameCN")) or stored("name_cn", "zh-Hans")
               or first(_CHINESE_NAME_KEYS - {"繁体中文名"}) or _text(raw.get("name_cn")))
    japanese = _text(raw.get("name_ja") or raw.get("nameJA")) or first(_JAPANESE_NAME_KEYS)
    if not japanese and (re.search(r"[\u3041-\u30fa]", original) or first(_READING_KEYS)):
        japanese = original
    english = (_text(raw.get("nameEN")) or first(_ENGLISH_NAME_KEYS)
               or stored("name_en", "en") or first(_ROMAN_NAME_KEYS) or _text(raw.get("name_en")))
    add(original, "ja" if japanese else "und", "primary", preserve=True)
    add(chinese, "zh-Hans", "primary", preserve=True)
    add(japanese, "ja", "primary", preserve=True)
    if english and not any(item["name"] == english for item in variants):
        add(english, "en", "primary")

    han_source = first({"繁体中文名"}) or original
    if not chinese and _han_name(han_source):
        chinese = _han_converter().convert(han_source)
        add(chinese, "zh-Hans", "script_conversion", "opencc")
    if not english:
        latin = next((text for _, text in entries if _latin_name(text)), "")
        if not latin and _latin_name(original):
            latin = original
        if latin:
            english = latin
        else:
            reading = first(_READING_KEYS) or (original if _kana(original) else "")
            english = _romanize_reading(reading)
            add(english, "ja-Latn", "romanization", "kana_transliteration")

    aliases = []
    for value in [*(raw.get("aliases", []) or []),
                  *_infobox_values(raw.get("infobox"), _NAME_KEYS - _CHINESE_NAME_KEYS),
                  original, chinese, japanese, english,
                  *(item["name"] for item in variants)]:
        text = _text(value)
        if text and text not in aliases:
            aliases.append(text)
    return {"name_cn": chinese, "name_ja": japanese, "name_en": english,
            "name_variants": variants, "aliases": aliases}


def normalize_dossier_names(dossier: dict) -> dict:
    """Upgrade cached and durable snapshots locally, without another network read."""
    result = copy.deepcopy(dossier)
    for character in result.get("characters", []):
        character.update(character_names(character))
    result["name_schema_version"] = NAME_SCHEMA_VERSION
    return result


def _subject(raw: dict) -> dict:
    identifier = _id(raw.get("id"))
    airtime = raw.get("airtime") if isinstance(raw.get("airtime"), dict) else {}
    return {
        **raw,
        "id": identifier,
        "name": _text(raw.get("name")),
        "name_cn": _text(raw.get("name_cn") or raw.get("nameCN")),
        "date": _text(raw.get("date") or airtime.get("date")),
        "summary": _text(raw.get("summary")),
        "image_url": _image(raw),
        "url": f"https://bgm.tv/subject/{identifier}",
    }


def _character(raw: dict, relation: dict) -> dict:
    identifier = _id(raw.get("id"))
    infobox = raw.get("infobox", [])
    name = _text(raw.get("name"))
    role = relation.get("type", relation.get("relation", ""))
    return {
        "id": identifier,
        "name": name,
        **character_names(raw),
        "description": _text(raw.get("summary") or raw.get("description") or raw.get("info")),
        "image_url": _image(raw),
        "url": f"https://bgm.tv/character/{identifier}",
        "role": {1: "主角", 2: "配角", 3: "客串"}.get(role, _text(role)),
        "relation_type": role,
        "infobox": infobox if isinstance(infobox, list) else [],
        "casts": relation.get("casts", relation.get("actors", [])),
    }


def _environment(config_dir: Path) -> dict[str, str]:
    allowed = {
        "PATH", "HOME", "SYSTEMROOT", "WINDIR", "PATHEXT", "COMSPEC", "APPDATA", "LOCALAPPDATA",
        "TMPDIR", "TEMP", "TMP", "LANG", "LC_ALL", "LC_CTYPE", "SSL_CERT_FILE", "SSL_CERT_DIR",
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "BGM_PROXY",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    # Public reads do not need the user's Bangumi tokens or personal CLI settings.
    environment["BGM_CONFIG_DIR"] = str(config_dir)
    environment["BGM_USER_AGENT"] = os.environ.get(
        "SCENERECALL_BGM_USER_AGENT", "SceneRecall/0.2 (local anime catalogue; bgm-cli/1.1.7)")
    return environment


def _configured_timeout() -> float:
    try:
        timeout = float(os.environ.get("SCENERECALL_BGM_TIMEOUT_S", "20"))
    except ValueError:
        raise BangumiError("SCENERECALL_BGM_TIMEOUT_S 必须是正数", code="configuration") from None
    if not math.isfinite(timeout) or timeout <= 0:
        raise BangumiError("SCENERECALL_BGM_TIMEOUT_S 必须是正数", code="configuration")
    return timeout


class BangumiClient:
    def __init__(self, *, timeout_s: float | None = None, max_concurrency: int = 4):
        self.timeout_s = _configured_timeout() if timeout_s is None else timeout_s
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise BangumiError("Bangumi 请求超时必须是正数", code="configuration")
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency <= 0:
            raise BangumiError("Bangumi 最大并发必须是正整数", code="configuration")
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def search(self, query: str) -> list[dict]:
        query = _text(query)
        if not query or len(query) > 200 or any(ord(char) < 32 for char in query) or query.startswith("-"):
            raise BangumiError("请输入 1 至 200 个字符的动画名称", code="invalid_query")
        result = await self._run("subject", "search", query, "--type", "anime", "--limit", "10")
        candidates = []
        for raw in self._items(result):
            if raw.get("type", 2) != 2:
                continue
            subject = _subject(raw)
            candidates.append({key: subject[key] for key in (
                "id", "name", "name_cn", "date", "summary", "image_url", "url")})
        return candidates

    async def dossier(self, subject_id: str | int) -> dict:
        identifier = parse_subject_reference(subject_id)
        try:
            # A stalled large catalogue must not leave a local request hanging indefinitely.
            async with asyncio.timeout(max(120, self.timeout_s * 3)):
                return await self._dossier(identifier)
        except TimeoutError:
            raise BangumiError("Bangumi 档案读取超时，请稍后重试", code="timeout") from None

    async def _dossier(self, identifier: int) -> dict:
        raw_subject = await self._run("subject", "get", str(identifier))
        if not isinstance(raw_subject, dict) or _id(raw_subject.get("id")) != identifier:
            raise BangumiError("Bangumi 返回的条目信息不匹配", code="invalid_response")
        if raw_subject.get("type") != 2:
            raise BangumiError("该 Bangumi 条目不是动画，请提供动画条目链接", code="not_anime")
        relations: list[dict] = []
        seen = set()
        warnings = []
        for offset in range(0, MAX_CHARACTERS, PAGE_SIZE):
            result = await self._run("subject", "characters", str(identifier),
                                     "--limit", str(PAGE_SIZE), "--offset", str(offset))
            page = self._items(result)
            new = 0
            for relation in page:
                raw = relation.get("character", relation)
                if not isinstance(raw, dict):
                    raise BangumiError("Bangumi 返回的角色列表格式不正确", code="invalid_response")
                character_id = _id(raw.get("id"))
                if character_id not in seen:
                    relations.append(relation)
                    seen.add(character_id)
                    new += 1
            total = result.get("total") if isinstance(result, dict) else None
            if len(page) < PAGE_SIZE or new == 0 or (isinstance(total, int) and len(seen) >= total):
                break
        else:
            warnings.append(f"角色数量超过单次读取上限 {MAX_CHARACTERS}，本次保存前 {MAX_CHARACTERS} 个角色")

        async def detail(relation: dict) -> tuple[dict, str | None]:
            raw = relation.get("character", relation)
            character_id = _id(raw.get("id"))
            try:
                response = await self._run("character", "get", str(character_id))
                enriched = response.get("data", response) if isinstance(response, dict) else None
                if not isinstance(enriched, dict) or _id(enriched.get("id")) != character_id:
                    raise BangumiError("Bangumi 返回的角色资料不匹配", code="invalid_response")
                return _character({**raw, **enriched}, relation), None
            except BangumiError as error:
                character = _character(raw, relation)
                character["detail_error"] = str(error)
                return character, f"角色 {character['name_cn'] or character['name']} 的详细资料未取得：{error}"

        detailed = await asyncio.gather(*(detail(relation) for relation in relations))
        warnings.extend(message for _, message in detailed if message)
        return {
            "source": "bangumi",
            "name_schema_version": NAME_SCHEMA_VERSION,
            "subject": _subject(raw_subject),
            "characters": [character for character, _ in detailed],
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
        }

    @staticmethod
    def _items(result: Any) -> list[dict]:
        items = result if isinstance(result, list) else result.get("data") if isinstance(result, dict) else None
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise BangumiError("Bangumi 返回的列表格式不正确", code="invalid_response")
        return items

    async def _run(self, *arguments: str) -> Any:
        configured = os.environ.get("SCENERECALL_BGM_CLI_PATH")
        entrypoint = Path(configured).expanduser().resolve() if configured else CLI_ENTRYPOINT
        node = shutil.which("node")
        if not node or not entrypoint.is_file():
            raise BangumiError("未找到项目的 bgm-cli，请安装 Node.js 20+ 并在项目根目录执行 npm ci",
                               code="not_installed")
        async with self._semaphore:
            with tempfile.TemporaryDirectory(prefix="scenerecall-bgm-") as directory:
                return await self._execute([node, str(entrypoint), "--json", *arguments], Path(directory))

    async def _execute(self, arguments: list[str], directory: Path) -> Any:
        process = None
        tasks = []

        async def read(stream: asyncio.StreamReader, limit: int) -> bytes:
            output = bytearray()
            while chunk := await stream.read(65536):
                output.extend(chunk)
                if len(output) > limit:
                    raise BangumiError("bgm-cli 输出超过大小限制", code="output_limit")
            return bytes(output)

        try:
            async with asyncio.timeout(self.timeout_s):
                process = await asyncio.create_subprocess_exec(
                    *arguments, cwd=directory, env=_environment(directory),
                    stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, start_new_session=os.name == "posix",
                )
                tasks = [asyncio.create_task(read(process.stdout, MAX_OUTPUT_BYTES)),
                         asyncio.create_task(read(process.stderr, MAX_DIAGNOSTIC_BYTES)),
                         asyncio.create_task(process.wait())]
                stdout, stderr, returncode = await asyncio.gather(*tasks)
            if returncode:
                # Avoid exposing CLI diagnostics that may contain proxy URLs, credentials or HTML.
                diagnostic = stderr.decode("utf-8", errors="replace").lower()
                if "404" in diagnostic or "not found" in diagnostic:
                    message, code = "Bangumi 未找到该条目或资料暂不可公开读取", "not_found"
                elif "429" in diagnostic or "too many" in diagnostic:
                    message, code = "Bangumi 请求过于频繁，请稍后重试", "rate_limited"
                else:
                    message, code = "Bangumi 读取失败，请检查网络或 BGM_PROXY 代理配置后重试", "unavailable"
                raise BangumiError(message, code=code)
            try:
                return json.loads(stdout)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raise BangumiError("bgm-cli 返回了无法解析的 JSON", code="invalid_response") from None
        except TimeoutError:
            raise BangumiError("Bangumi 请求超时，请稍后重试或调整 SCENERECALL_BGM_TIMEOUT_S",
                               code="timeout") from None
        except OSError:
            raise BangumiError("无法启动项目的 bgm-cli，请检查 Node.js 与依赖安装", code="not_installed") from None
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if process is not None:
                try:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    elif process.returncode is None:
                        process.kill()
                except ProcessLookupError:
                    pass
                await process.communicate()

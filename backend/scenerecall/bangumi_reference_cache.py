"""Explicit, bounded caching of stored Bangumi images as unconfirmed references.

No URLs come from a request payload. HTTPS connections pin a public DNS result
while retaining certificate verification for the permitted image hostname.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import hashlib
import http.client
import ipaddress
import json
import socket
import ssl
import threading
import time
from urllib.parse import urljoin, urlsplit, urlunsplit

from .characters import CharacterStore
from .visual_identity import MAX_IMAGE_BYTES, ReferenceGallery, _image, _png

IMAGE_HOSTS = frozenset({"lain.bgm.tv"})
MAX_REDIRECTS = 3
IMAGE_TIMEOUT_SECONDS = 15
BATCH_TIMEOUT_SECONDS = 60
MAX_IMAGES = 20
_DNS_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bangumi-image-dns")
_FAKE_IP_RANGE = ipaddress.ip_network("198.18.0.0/15")
_DOH_HOST = "cloudflare-dns.com"
_DOH_ADDRESS = "1.1.1.1"
_DOH_PATH = "/dns-query?name=lain.bgm.tv&type=A"
_DOH_MAX_BYTES = 16 * 1024
_DOH_CACHE = {}
_DOH_LOCK = threading.Lock()


class CacheError(ValueError):
    """A bounded diagnostic that never includes remote headers or credentials."""


def checked_image_url(value):
    if (not isinstance(value, str) or not value or len(value) > 4096
            or any(ord(char) < 33 or ord(char) == 127 for char in value) or "\\" in value):
        raise CacheError("Bangumi 图片地址格式无效")
    try:
        parsed = urlsplit(value)
        if (parsed.scheme != "https" or parsed.hostname not in IMAGE_HOSTS
                or parsed.username is not None or parsed.password is not None
                or parsed.port not in (None, 443) or parsed.fragment):
            raise ValueError
    except ValueError:
        raise CacheError("只允许已导入资料中的 Bangumi 官方 HTTPS 图片地址") from None
    return urlunsplit(("https", parsed.hostname, parsed.path or "/", parsed.query, ""))


def _remaining(deadline):
    value = deadline - time.monotonic()
    if value <= 0:
        raise CacheError("Bangumi 图片缓存超过本次时间限制")
    return value


def _public_addresses(host, deadline):
    timeout = min(5, _remaining(deadline))
    future = _DNS_POOL.submit(socket.getaddrinfo, host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
    try:
        records = future.result(timeout=timeout)
    except (FutureTimeout, OSError):
        future.cancel()
        raise CacheError("Bangumi 图片域名解析失败或超时") from None
    addresses = []
    for family, _, _, _, destination in records:
        if family not in (socket.AF_INET, socket.AF_INET6):
            continue
        address = destination[0]
        try:
            ipaddress.ip_address(address)
        except ValueError:
            raise CacheError("Bangumi 图片域名返回无效地址") from None
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise CacheError("Bangumi 图片域名没有可用的公开地址")
    # Some local VPNs synthesize this RFC 2544 range for all DNS answers.
    # Never connect to it. A fixed, authenticated public resolver may replace
    # an exclusively FakeIP answer; ordinary private/mixed answers still fail.
    if all(ipaddress.ip_address(value) in _FAKE_IP_RANGE for value in addresses):
        return _doh_addresses(host, deadline)
    if any(not ipaddress.ip_address(value).is_global or ipaddress.ip_address(value).is_multicast
           or ipaddress.ip_address(value).is_reserved for value in addresses):
        raise CacheError("Bangumi 图片域名指向非公开网络，已拒绝连接")
    return sorted(addresses, key=lambda value: ipaddress.ip_address(value).version)


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, address, timeout):
        super().__init__(host, port=443, timeout=timeout, context=ssl.create_default_context())
        self.public_address = address

    def connect(self):
        # The hostname is never resolved again between the public-address check
        # and connect. TLS still authenticates the original allowed hostname.
        connection = socket.create_connection((self.public_address, 443), self.timeout)
        try:
            self.sock = self._context.wrap_socket(connection, server_hostname=self.host)
        except BaseException:
            connection.close()
            raise


def _doh_addresses(host, deadline):
    """Resolve only the fixed image name through a fixed public TLS endpoint."""
    if host != "lain.bgm.tv":
        raise CacheError("公共 DNS 回退仅支持 Bangumi 官方图片域名")
    if not _DOH_LOCK.acquire(timeout=_remaining(deadline)):
        raise CacheError("Bangumi 公共 DNS 解析等待超时")
    try:
        if _DOH_CACHE.get("expires", 0) > time.monotonic():
            return list(_DOH_CACHE["addresses"])
        query_deadline = time.monotonic() + min(5, _remaining(deadline))
        connection = _PinnedHTTPS(_DOH_HOST, _DOH_ADDRESS, _remaining(query_deadline))
        response, active_socket = None, [None]

        def expire():
            stream = connection.sock or active_socket[0]
            if stream is not None:
                try:
                    stream.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

        expiry = threading.Timer(_remaining(query_deadline), expire)
        expiry.daemon = True
        expiry.start()
        try:
            connection.request("GET", _DOH_PATH, headers={"Accept": "application/dns-json",
                "Accept-Encoding": "identity", "User-Agent": "SceneRecall/0.3 (Bangumi image DNS)"})
            active_socket[0] = connection.sock
            response = connection.getresponse()
            if response.status != 200:  # Redirects are deliberately never followed.
                raise CacheError("Bangumi 公共 DNS 未返回有效响应，已拒绝跳转或失败结果")
            content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type not in {"application/dns-json", "application/json"}:
                raise CacheError("Bangumi 公共 DNS 响应类型无效")
            length = response.getheader("Content-Length")
            if length is not None and (not length.isascii() or not length.isdigit() or int(length) > _DOH_MAX_BYTES):
                raise CacheError("Bangumi 公共 DNS 响应超过大小限制")
            data = bytearray()
            while True:
                remaining = _remaining(query_deadline)
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                block = response.read1(min(4096, _DOH_MAX_BYTES + 1 - len(data)))
                if not block:
                    break
                data.extend(block)
                if len(data) > _DOH_MAX_BYTES:
                    raise CacheError("Bangumi 公共 DNS 响应超过大小限制")
            if length is not None and len(data) != int(length):
                raise CacheError("Bangumi 公共 DNS 响应不完整")
            payload = json.loads(data)
        except (OSError, http.client.HTTPException, json.JSONDecodeError, UnicodeError):
            raise CacheError("Bangumi 公共 DNS 解析失败或超时") from None
        finally:
            expiry.cancel()
            if response is not None:
                response.close()
            connection.close()
        if (not isinstance(payload, dict) or type(payload.get("Status")) is not int or payload["Status"] != 0
                or payload.get("TC") is True or not isinstance(payload.get("Question"), list)
                or len(payload["Question"]) != 1 or not isinstance(payload["Question"][0], dict)
                or payload["Question"][0].get("name") not in {host, host + "."}
                or type(payload["Question"][0].get("type")) is not int or payload["Question"][0]["type"] != 1
                or not isinstance(payload.get("Answer"), list) or not 1 <= len(payload["Answer"]) <= 16):
            raise CacheError("Bangumi 公共 DNS 问题或回答不匹配")
        addresses, ttl = [], 300
        for answer in payload["Answer"]:
            if (not isinstance(answer, dict) or answer.get("name") not in {host, host + "."}
                    or type(answer.get("type")) is not int or answer["type"] != 1
                    or type(answer.get("TTL")) is not int or answer["TTL"] < 0
                    or not isinstance(answer.get("data"), str)):
                raise CacheError("Bangumi 公共 DNS 只接受匹配域名的 IPv4 回答")
            try:
                ip = ipaddress.ip_address(answer["data"])
            except ValueError:
                raise CacheError("Bangumi 公共 DNS 地址无效") from None
            if ip.version != 4 or not ip.is_global or ip.is_multicast or ip.is_reserved:
                raise CacheError("Bangumi 公共 DNS 返回非公开 IPv4 地址，已拒绝连接")
            if str(ip) not in addresses:
                addresses.append(str(ip))
            ttl = min(ttl, answer["TTL"])
        _remaining(deadline)
        _DOH_CACHE.update(addresses=addresses, expires=time.monotonic() + ttl)
        return list(addresses)
    finally:
        _DOH_LOCK.release()


def download_image(url, *, timeout=IMAGE_TIMEOUT_SECONDS, resolver=None, connection_factory=None):
    """Follow only checked HTTPS redirects; read at most 20 MiB of image bytes."""
    current = checked_image_url(url)
    deadline = time.monotonic() + min(float(timeout), IMAGE_TIMEOUT_SECONDS)
    resolver = resolver or _public_addresses
    connection_factory = connection_factory or _PinnedHTTPS
    for hop in range(MAX_REDIRECTS + 1):
        current = checked_image_url(current)
        parsed = urlsplit(current)
        addresses = resolver(parsed.hostname, deadline)
        connection = connection_factory(parsed.hostname, addresses[0], _remaining(deadline))
        active_socket = [None]
        response = None

        def expire():
            # A socket timeout alone can be reset by a slow stream. Shutdown at
            # the absolute deadline also interrupts headers/chunk-size reads.
            for stream in (connection.sock, active_socket[0]):
                if stream is not None:
                    try:
                        stream.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass

        expiry = threading.Timer(_remaining(deadline), expire)
        expiry.daemon = True
        expiry.start()
        try:
            connection.request("GET", urlunsplit(("", "", parsed.path, parsed.query, "")), headers={
                "User-Agent": "SceneRecall/0.3 (explicit Bangumi reference image cache)",
                "Accept": "image/*", "Accept-Encoding": "identity",
            })
            active_socket[0] = connection.sock
            response = connection.getresponse()
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location")
                if not location or hop == MAX_REDIRECTS:
                    raise CacheError("Bangumi 图片跳转次数过多或缺少目标地址")
                if any(ord(char) < 33 or ord(char) == 127 for char in location) or "\\" in location:
                    raise CacheError("Bangumi 图片跳转地址格式无效")
                current = checked_image_url(urljoin(current, location))
                continue
            if response.status != 200:
                raise CacheError(f"Bangumi 图片请求失败（HTTP {response.status}）")
            content_type = response.getheader("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type and not content_type.startswith("image/"):
                raise CacheError("Bangumi 图片响应不是图像")
            length = response.getheader("Content-Length")
            if length is not None and (not length.isascii() or not length.isdigit() or int(length) > MAX_IMAGE_BYTES):
                raise CacheError("Bangumi 图片大小无效或超过 20 MiB")
            content = bytearray()
            while True:
                remaining = _remaining(deadline)
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                block = response.read1(min(65536, MAX_IMAGE_BYTES + 1 - len(content)))
                if not block:
                    break
                content.extend(block)
                if len(content) > MAX_IMAGE_BYTES:
                    raise CacheError("Bangumi 图片超过 20 MiB")
            if not content or length is not None and len(content) != int(length):
                raise CacheError("Bangumi 图片内容不完整")
            return bytes(content)
        finally:
            expiry.cancel()
            if response is not None:
                response.close()
            connection.close()
    raise CacheError("Bangumi 图片跳转次数过多")


def _stored_url(profile):
    source = profile.get("external_source") or {}
    return source.get("image_url") if source.get("source") == "bangumi" else None


class BangumiReferenceCache:
    def __init__(self, library, gallery=None, *, characters=None, downloader=None):
        self.library = library
        self.gallery = gallery or ReferenceGallery(library)
        self.characters = characters or CharacterStore(library)
        self.downloader = downloader or download_image
        self._busy = threading.Lock()

    def run(self, asset_id, max_images=12):
        if type(max_images) is not int or not 1 <= max_images <= MAX_IMAGES:
            raise ValueError("每次缓存图片数量须为 1–20")
        if not self._busy.acquire(blocking=False):
            raise ValueError("参考图缓存正在运行，请稍后重试")
        try:
            return self._run(asset_id, max_images)
        finally:
            self._busy.release()

    def _run(self, asset_id, max_images):
        catalog = self.characters.catalog(asset_id)
        result = {"imported": 0, "existing": 0, "skipped": 0, "errors": [], "references": []}
        deadline = time.monotonic() + BATCH_TIMEOUT_SECONDS
        attempted = 0
        for profile in sorted(catalog["profiles"], key=lambda row: row["id"]):
            url = _stored_url(profile)
            if not url or attempted >= max_images:
                result["skipped"] += 1
                continue
            attempted += 1
            identity = profile["id"]
            try:
                checked_image_url(url)
                content = self.downloader(url, timeout=min(IMAGE_TIMEOUT_SECONDS, _remaining(deadline)))
                # Use the gallery's exact RGB/PNG normalization for its content
                # hash; re-encoded JPEGs of the same image cannot add duplicates.
                normalized = _png(_image(content))
                if len(normalized) > MAX_IMAGE_BYTES:
                    raise CacheError("转换后的 Bangumi 参考图超过 20 MiB")
                content_hash = hashlib.sha256(normalized).hexdigest()
                with self.library.lock:
                    current = self.characters.catalog(asset_id)
                    latest = next((row for row in current["profiles"] if row["id"] == identity), None)
                    if current["scope_id"] != catalog["scope_id"] or not latest or _stored_url(latest) != url:
                        raise CacheError("角色所属作品或 Bangumi 图片资料已变更，请重试")
                    existing = next((row for row in self.gallery.list(asset_id, identity)["references"]
                        if row["source"].get("kind") == "external_cached" and row["source"].get("url") == url
                        and row["sha256"] == content_hash), None)
                    if existing is not None:
                        reference = existing  # Never reactivate a revoked reference.
                        result["existing"] += 1
                    else:
                        reference = self.gallery.add(asset_id, identity, content, source_kind="external_cached",
                            source_url=url, confirmed=False, single_character=False,
                            source={"download_sha256": hashlib.sha256(content).hexdigest()})
                        result["imported"] += 1
                result["references"].append({key: value for key, value in reference.items() if key != "path"})
            except Exception as error:
                result["skipped"] += 1
                reason = str(error) if isinstance(error, CacheError) else f"图片读取或保存失败（{type(error).__name__}）"
                result["errors"].append({"character_id": identity, "reason": reason})
        return result

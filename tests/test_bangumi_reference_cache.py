"""Network-boundary and candidate-provenance tests using only synthetic images."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from scenerecall import bangumi_reference_cache as cache  # noqa: E402
from scenerecall.characters import CharacterStore, scope_id  # noqa: E402
from scenerecall.library import Library, atomic_json  # noqa: E402
from scenerecall.visual_identity import ReferenceGallery  # noqa: E402


def picture(color=90):
    image = Image.new("RGB", (128, 160), (color, 40, 80))
    draw = ImageDraw.Draw(image)
    for index in range(8, 128, 9):
        draw.line((index, 0, index, 160), fill=(200, 220, 40), width=2)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class Response:
    def __init__(self, status=200, content=b"image-bytes", headers=None):
        self.status, self.content = status, io.BytesIO(content)
        self.headers = headers or {}
        self.reads = 0

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read1(self, size):
        self.reads += 1
        return self.content.read(size)

    def close(self):
        self.content.close()


class Connection:
    def __init__(self, response):
        self.response, self.sock, self.closed = response, Mock(), False
        self.requests = []

    def request(self, method, path, headers):
        self.requests.append((method, path, headers))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class DownloadTests(unittest.TestCase):
    def fetch(self, responses, url="https://lain.bgm.tv/pic/crt/l/one.jpg", **kwargs):
        connections = [Connection(response) for response in responses]
        factory = Mock(side_effect=connections)
        resolver = Mock(return_value=["93.184.216.34"])
        result = cache.download_image(url, resolver=resolver, connection_factory=factory, **kwargs)
        return result, connections, factory, resolver

    def test_only_exact_https_image_host_without_credentials(self):
        bad = ("http://lain.bgm.tv/a", "https://localhost/a", "https://127.0.0.1/a",
               "https://[::1]/a", "https://lain.bgm.tv.attacker.invalid/a", "https://evil.lain.bgm.tv/a",
               "https://user:secret@lain.bgm.tv/a", "https://@lain.bgm.tv/a", "https://lain.bgm.tv:444/a",
               "https://lain.bgm.tv./a", "https://lain.bgm.tv/a\n", "https://lain.bgm.tv/a#fragment")
        for url in bad:
            with self.subTest(url=url), self.assertRaises(cache.CacheError):
                cache.checked_image_url(url)
        self.assertEqual(cache.checked_image_url("https://lain.bgm.tv:443/a?size=2"),
                         "https://lain.bgm.tv/a?size=2")

    def test_relative_redirect_checked_again_and_never_sends_credentials(self):
        result, connections, factory, resolver = self.fetch([
            Response(302, headers={"Location": "/pic/crt/l/two.jpg"}),
            Response(content=b"safe", headers={"Content-Type": "image/jpeg", "Content-Length": "4"}),
        ])
        self.assertEqual(result, b"safe")
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(resolver.call_count, 2)
        self.assertTrue(all(connection.closed for connection in connections))
        for connection in connections:
            headers = connection.requests[0][2]
            self.assertFalse({"Cookie", "Authorization", "Proxy-Authorization"} & set(headers))
        self.assertEqual(connections[1].requests[0][1], "/pic/crt/l/two.jpg")

    def test_every_redirect_must_stay_on_allowed_https_host(self):
        for target in ("http://lain.bgm.tv/a", "https://127.0.0.1/a", "https://user@lain.bgm.tv/a",
                       "https://other.bgm.tv/a", "//attacker.invalid/a", "https:\n//lain.bgm.tv/a"):
            connection = Connection(Response(302, headers={"Location": target}))
            factory = Mock(return_value=connection)
            with self.subTest(target=target), self.assertRaises(cache.CacheError):
                cache.download_image("https://lain.bgm.tv/a", resolver=Mock(return_value=["93.184.216.34"]),
                                     connection_factory=factory)
            self.assertEqual(factory.call_count, 1)
            self.assertTrue(connection.closed)

    def test_redirect_loop_is_bounded(self):
        factory = Mock(side_effect=lambda *args: Connection(Response(302, headers={"Location": "/again"})))
        with self.assertRaises(cache.CacheError):
            cache.download_image("https://lain.bgm.tv/a", resolver=Mock(return_value=["93.184.216.34"]),
                                 connection_factory=factory)
        self.assertEqual(factory.call_count, cache.MAX_REDIRECTS + 1)

    def test_private_or_mixed_dns_answers_rejected(self):
        for address in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fe80::1", "224.0.0.1"):
            answer = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
                      (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]
            with self.subTest(address=address), patch.object(cache.socket, "getaddrinfo", return_value=answer):
                with self.assertRaises(cache.CacheError):
                    cache._public_addresses("lain.bgm.tv", time.monotonic() + 5)

    def test_connect_pins_checked_address_and_authenticates_hostname(self):
        connection = cache._PinnedHTTPS("lain.bgm.tv", "93.184.216.34", 4)
        connection._context = Mock()
        raw = Mock()
        with patch.object(cache.socket, "create_connection", return_value=raw) as connect:
            connection.connect()
        connect.assert_called_once_with(("93.184.216.34", 443), 4)
        connection._context.wrap_socket.assert_called_once_with(raw, server_hostname="lain.bgm.tv")

    def test_declared_and_streamed_byte_bounds(self):
        response = Response(content=b"small", headers={"Content-Length": str(cache.MAX_IMAGE_BYTES + 1)})
        with self.assertRaises(cache.CacheError):
            self.fetch([response])
        self.assertEqual(response.reads, 0)
        with patch.object(cache, "MAX_IMAGE_BYTES", 4), self.assertRaises(cache.CacheError):
            self.fetch([Response(content=b"12345")])

    def test_incomplete_or_nonimage_response_rejected(self):
        for response in (Response(content=b"a", headers={"Content-Length": "8"}), Response(content=b""),
                         Response(content=b"html", headers={"Content-Type": "text/html"}), Response(403)):
            with self.subTest(status=response.status), self.assertRaises(cache.CacheError):
                self.fetch([response])

    def test_expired_timeout_does_not_connect_or_resolve_dns(self):
        factory = Mock()
        with self.assertRaises(cache.CacheError):
            cache.download_image("https://lain.bgm.tv/a", timeout=0,
                                 resolver=Mock(return_value=["93.184.216.34"]), connection_factory=factory)
        factory.assert_not_called()
        with patch.object(cache.socket, "getaddrinfo") as resolve, self.assertRaises(cache.CacheError):
            cache._public_addresses("lain.bgm.tv", time.monotonic() - 1)
        resolve.assert_not_called()

    def test_absolute_deadline_interrupts_slow_response_headers(self):
        interrupted = threading.Event()
        connection = Connection(Response(content=b"safe"))
        connection.sock.shutdown.side_effect = lambda *_: interrupted.set()

        def slow_headers():
            self.assertTrue(interrupted.wait(.5))
            return connection.response

        connection.getresponse = slow_headers
        with self.assertRaises(cache.CacheError):
            cache.download_image("https://lain.bgm.tv/a", timeout=.03,
                                 resolver=Mock(return_value=["93.184.216.34"]),
                                 connection_factory=Mock(return_value=connection))
        self.assertTrue(connection.closed)


class DoHFallbackTests(unittest.TestCase):
    def setUp(self):
        cache._DOH_CACHE.clear()

    @staticmethod
    def payload():
        return {"Status": 0, "TC": False, "Question": [{"name": "lain.bgm.tv.", "type": 1}],
                "Answer": [{"name": "lain.bgm.tv.", "type": 1, "TTL": 600, "data": address}
                           for address in ("172.67.73.67", "104.26.8.23", "104.26.9.23")]}

    @staticmethod
    def dns_response(payload):
        return Response(content=json.dumps(payload).encode(), headers={"Content-Type": "application/dns-json"})

    def test_exclusive_vpn_fakeip_uses_fixed_doh_then_pins_public_image_ip(self):
        records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("198.18.1.242", 443))]
        doh = Connection(self.dns_response(self.payload()))
        image = Connection(Response(content=b"public-image"))
        image_factory = Mock(return_value=image)
        with patch.object(cache.socket, "getaddrinfo", return_value=records), \
                patch.object(cache, "_PinnedHTTPS", return_value=doh) as resolver_connection:
            content = cache.download_image("https://lain.bgm.tv/character.jpg", connection_factory=image_factory)
        self.assertEqual(content, b"public-image")
        self.assertEqual(resolver_connection.call_args.args[:2], ("cloudflare-dns.com", "1.1.1.1"))
        self.assertEqual(doh.requests[0][1], "/dns-query?name=lain.bgm.tv&type=A")
        self.assertEqual(image_factory.call_args.args[:2], ("lain.bgm.tv", "172.67.73.67"))
        self.assertTrue(doh.closed and image.closed)

    def test_normal_public_dns_retains_original_path_without_doh(self):
        records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.26.8.23", 443))]
        with patch.object(cache.socket, "getaddrinfo", return_value=records) as resolve, \
                patch.object(cache, "_doh_addresses") as doh:
            self.assertEqual(cache._public_addresses("lain.bgm.tv", time.monotonic() + 5), ["104.26.8.23"])
        resolve.assert_called_once_with("lain.bgm.tv", 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
        doh.assert_not_called()

    def test_ordinary_private_or_mixed_fake_answers_still_rejected(self):
        for addresses in (("10.0.0.1",), ("127.0.0.1",), ("198.18.1.242", "104.26.8.23"),
                          ("198.18.1.242", "192.168.1.1")):
            records = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443)) for address in addresses]
            with self.subTest(addresses=addresses), patch.object(cache.socket, "getaddrinfo", return_value=records), \
                    patch.object(cache, "_doh_addresses") as doh, self.assertRaises(cache.CacheError):
                cache._public_addresses("lain.bgm.tv", time.monotonic() + 5)
            doh.assert_not_called()

    def test_bad_doh_answers_cannot_select_private_or_unrelated_addresses(self):
        invalid = []
        for address in ("10.0.0.1", "198.18.1.242", "127.0.0.1", "::1", "224.0.0.1", "not-an-ip"):
            payload = self.payload()
            payload["Answer"][1]["data"] = address
            invalid.append(payload)
        for field, value in (("Status", 3), ("Status", False), ("TC", True), ("Answer", [])):
            payload = self.payload()
            payload[field] = value
            invalid.append(payload)
        payload = self.payload()
        payload["Question"][0]["name"] = "attacker.invalid."
        invalid.append(payload)
        payload = self.payload()
        payload["Answer"][0]["name"] = "attacker.invalid."
        invalid.append(payload)
        payload = self.payload()
        payload["Answer"][0]["type"] = 5
        invalid.append(payload)
        for payload in invalid:
            connection = Connection(self.dns_response(payload))
            with patch.object(cache, "_PinnedHTTPS", return_value=connection), self.assertRaises(cache.CacheError):
                cache._doh_addresses("lain.bgm.tv", time.monotonic() + 5)
            self.assertTrue(connection.closed)
        with self.assertRaises(cache.CacheError):
            cache._doh_addresses("unrelated.bgm.tv", time.monotonic() + 5)

    def test_doh_redirects_timeouts_and_oversized_responses_fail_closed(self):
        for response in (Response(302, headers={"Location": "https://attacker.invalid/resolve"}),
                         Response(content=b"x", headers={"Content-Type": "application/dns-json",
                                                         "Content-Length": str(cache._DOH_MAX_BYTES + 1)}),
                         Response(content=b"bad-json", headers={"Content-Type": "text/html"})):
            connection = Connection(response)
            with patch.object(cache, "_PinnedHTTPS", return_value=connection) as factory, self.assertRaises(cache.CacheError):
                cache._doh_addresses("lain.bgm.tv", time.monotonic() + 5)
            self.assertEqual(factory.call_count, 1)
            self.assertTrue(connection.closed)
        connection = Connection(Response())
        connection.getresponse = Mock(side_effect=TimeoutError("fixture"))
        with patch.object(cache, "_PinnedHTTPS", return_value=connection), self.assertRaises(cache.CacheError):
            cache._doh_addresses("lain.bgm.tv", time.monotonic() + 5)
        self.assertTrue(connection.closed)

    def test_doh_respects_remaining_image_budget(self):
        interrupted = threading.Event()
        connection = Connection(self.dns_response(self.payload()))
        connection.sock.shutdown.side_effect = lambda *_: interrupted.set()

        def slow_headers():
            self.assertTrue(interrupted.wait(.5))
            return connection.response

        connection.getresponse = slow_headers
        with patch.object(cache, "_PinnedHTTPS", return_value=connection), self.assertRaises(cache.CacheError):
            cache._doh_addresses("lain.bgm.tv", time.monotonic() + .03)
        self.assertTrue(connection.closed)

    def test_public_doh_answers_cached_no_longer_than_ttl_or_five_minutes(self):
        connection = Connection(self.dns_response(self.payload()))
        before = time.monotonic()
        with patch.object(cache, "_PinnedHTTPS", return_value=connection) as factory:
            first = cache._doh_addresses("lain.bgm.tv", time.monotonic() + 5)
            second = cache._doh_addresses("lain.bgm.tv", time.monotonic() + 5)
        self.assertEqual(first, second)
        self.assertEqual(factory.call_count, 1)
        self.assertLessEqual(cache._DOH_CACHE["expires"] - before, 301)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="bangumi-reference-test-")
        self.addCleanup(self.temporary.cleanup)
        self.library = Library(Path(self.temporary.name) / "library")
        asset = {"schema_version": "1.0", "id": "asset_one", "work_id": "work_one", "title": "Fixture",
                 "series": "", "kind": "animation", "season": None, "episode": None,
                 "created_at": "2026-10-15T00:00:00Z", "duration_ms": 10000, "version": "test", "fingerprint": "fixture"}
        atomic_json(self.library.asset_dir("asset_one") / "manifest.json", asset)
        self.characters = CharacterStore(self.library)
        dossier = {"source": "bangumi", "subject": {"id": 42, "name": "Fixture"},
                   "fetched_at": "2026-10-15T00:00:00Z", "characters": [
                       {"id": 101, "name": "First", "image_url": "https://lain.bgm.tv/first.jpg"},
                       {"id": 102, "name": "Second", "image_url": "https://lain.bgm.tv/second.jpg"},
                       {"id": 103, "name": "No image"}]}
        self.characters.seed_bangumi(scope_id(asset), dossier)
        self.gallery = ReferenceGallery(self.library)
        self.download = Mock(return_value=picture())
        self.service = cache.BangumiReferenceCache(self.library, self.gallery, characters=self.characters,
                                                  downloader=self.download)

    def test_candidate_provenance_idempotency_and_no_fabricated_presence(self):
        first = self.service.run("asset_one")
        self.assertEqual((first["imported"], first["existing"], first["skipped"]), (2, 0, 1))
        self.assertEqual(first["errors"], [])
        for reference in first["references"]:
            self.assertEqual(reference["status"], "candidate")
            self.assertIsNone(reference["confirmation_source"])
            self.assertFalse(reference["quality"]["single_character"])
            self.assertFalse(reference["quality"]["accepted"])
            self.assertEqual(reference["source"]["kind"], "external_cached")
            self.assertEqual(reference["source"]["download_sha256"], cache.hashlib.sha256(picture()).hexdigest())
            self.assertNotIn("path", reference)
        again = self.service.run("asset_one")
        self.assertEqual((again["imported"], again["existing"], again["skipped"]), (0, 2, 1))
        self.assertEqual({r["id"] for r in first["references"]}, {r["id"] for r in again["references"]})
        self.assertEqual(len(self.gallery.list("asset_one")["references"]), 2)
        self.assertEqual(self.gallery.trusted("asset_one"), [])
        for profile in self.characters.catalog("asset_one")["profiles"]:
            self.assertEqual(profile["appearance_count"], 0)
            self.assertIsNone(profile["representative"])

    def test_revoked_same_image_is_not_reactivated(self):
        reference = self.service.run("asset_one", 1)["references"][0]
        self.gallery.update("asset_one", reference["character_id"], reference["id"], status="revoked", reason="fixture")
        again = self.service.run("asset_one", 1)
        self.assertEqual(again["existing"], 1)
        self.assertEqual(again["references"][0]["status"], "revoked")

    def test_changed_image_content_creates_new_candidate(self):
        self.service.run("asset_one", 1)
        self.download.return_value = picture(160)
        result = self.service.run("asset_one", 1)
        self.assertEqual(result["imported"], 1)
        self.assertEqual(len(self.gallery.list("asset_one")["references"]), 2)

    def test_limit_and_invalid_max_images_never_expand_downloads(self):
        result = self.service.run("asset_one", 1)
        self.assertEqual(self.download.call_count, 1)
        self.assertEqual(result["skipped"], 2)
        for count in (0, 21, True, "12"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.service.run("asset_one", count)
        self.assertEqual(self.download.call_count, 1)

    def test_image_validation_failure_is_per_character(self):
        self.download.side_effect = [b"not-an-image", picture()]
        result = self.service.run("asset_one")
        self.assertEqual((result["imported"], result["skipped"]), (1, 2))
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(len(self.gallery.list("asset_one")["references"]), 1)

    def test_bad_metadata_url_is_not_fetched_and_other_characters_continue(self):
        registry = self.characters._read(scope_id(self.library.get_asset("asset_one")))
        profiles = sorted(registry["profiles"].values(), key=lambda row: row["id"])
        first = next(row for row in profiles if row["external_source"].get("image_url"))
        first["external_source"]["image_url"] = "https://127.0.0.1/private"
        atomic_json(self.characters._path(registry["scope_id"]), registry)
        result = self.service.run("asset_one")
        self.assertEqual(result["imported"], 1)
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(self.download.call_count, 1)

    def test_stale_scope_during_fetch_does_not_write(self):
        original = self.characters.catalog("asset_one")
        changed = {**original, "scope_id": "different_scope"}
        with patch.object(self.characters, "catalog", side_effect=[original, changed]):
            result = self.service.run("asset_one", 1)
        self.assertEqual(result["imported"], 0)
        self.assertEqual(len(result["errors"]), 1)
        self.assertEqual(self.gallery.list("asset_one")["references"], [])

    def test_explicit_route_body_and_no_fetch_on_read(self):
        if importlib.util.find_spec("fastapi") is None or importlib.util.find_spec("httpx") is None:
            self.skipTest("FastAPI/httpx unavailable in this standalone stdlib test runtime")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from scenerecall.visual_identity_api import register_visual_identity_routes
        app = FastAPI()
        register_visual_identity_routes(app, self.library, None)
        app.state.bangumi_reference_cache = self.service
        client = TestClient(app)
        endpoint = "/api/assets/asset_one/references/cache-bangumi"
        self.assertEqual(client.get("/api/assets/asset_one/identity/settings").status_code, 200)
        self.download.assert_not_called()
        for payload in ({"url": "https://lain.bgm.tv/client.jpg"}, {"max_images": True}, {"max_images": 21}):
            self.assertEqual(client.post(endpoint, json=payload).status_code, 422)
        self.download.assert_not_called()
        response = client.post(endpoint, json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["imported"], 2)


if __name__ == "__main__":
    unittest.main()

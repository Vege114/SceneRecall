"""Explicit, immutable download of the v1 CPU knowledge embedding model.

Run using any Python 3.12 interpreter. Inference uses only these verified local
files; it never calls this script or Hugging Face download APIs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
REPOSITORY = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
REVISION = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
FILES = ("config.json", "model_optimized.onnx", "ort_config.json", "special_tokens_map.json",
         "tokenizer.json", "tokenizer_config.json", "unigram.json", "README.md")


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def download(destination):
    root = Path(destination).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    request = Request(f"https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true",
                      headers={"User-Agent": "SceneRecall-knowledge-setup/1"})
    with urlopen(request, timeout=60) as response:
        metadata = json.load(response)
    if metadata["sha"] != REVISION:
        raise ValueError("model revision mismatch")
    siblings = {row["rfilename"]: row for row in metadata["siblings"]}

    def fetch(name):
        expected = siblings[name].get("lfs", {}).get("sha256")
        target = root / name
        if expected and target.is_file() and sha256(target) == expected:
            return name, {"sha256": expected, "bytes": target.stat().st_size}
        temporary = target.with_suffix(target.suffix + ".partial")
        for attempt in range(3):
            try:
                url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
                with urlopen(Request(url, headers={"User-Agent": "SceneRecall-knowledge-setup/1"}), timeout=120) as response:
                    with temporary.open("wb") as output:
                        while block := response.read(1024 * 1024):
                            output.write(block)
                actual = sha256(temporary)
                if expected and actual != expected:
                    raise ValueError("download hash mismatch: " + name)
                temporary.replace(target)
                return name, {"sha256": actual, "bytes": target.stat().st_size}
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        raise RuntimeError("download failed")

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = dict(pool.map(fetch, FILES))
    manifest = {"model": MODEL, "repository": REPOSITORY, "revision": REVISION, "dimension": 384,
                "license": "apache-2.0", "files": files, "runtime": "fastembed==0.9.1",
                "device": "cpu", "long_input": "token_windows_126_stride_100_mean_normalize_v1"}
    temporary = root / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(root / "manifest.json")
    print(json.dumps({"model_dir": str(root), "revision": REVISION,
                      "bytes": sum(row["bytes"] for row in files.values())}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    download(parser.parse_args().destination)

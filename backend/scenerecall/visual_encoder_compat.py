"""Explicitly verified execution-only upgrades of the same numerical encoder.

Do not ignore arbitrary implementation hashes. Only exact released source
fingerprints listed here may reuse an existing vector space; any later source
edit remains isolated until its numerical equivalence is verified separately.
"""
from __future__ import annotations

import hashlib
import json


CANONICAL_IMPLEMENTATION_HASH = "87cb5b38a78bc180991a5ec97e3b1ca0831c08fea05bda074d9b1b6653241994"
# The previous release and this execution-only upgrade, including the LF/CRLF
# checkout variants of the two fingerprinted files. Canary: all 533 real-frame
# JPEGs, FP32 vectors and 48 search results were bit/exactly equal; processor
# tensors and OOM sub-batches were also checked independently. Any code edit
# outside these exact fingerprints automatically gets its own vector space.
EQUIVALENT_IMPLEMENTATION_HASHES = (
    "87cb5b38a78bc180991a5ec97e3b1ca0831c08fea05bda074d9b1b6653241994",
    "28e539ea549a2bd1c2944b0a8f50f310fc7ff42d896b7b3efcc04dc47ac2c23c",
    "b9c46b0bae86247147f6d7c4912306baf3269620ac857b37744c77aa815555fa",
    "cefad830c399749643973e80b5ed272af0d3b492d7c4f465d4237f85e8085402",
    "0bdb0056fea47d20ebf36d3c8cd9815a34e150a2ccd4ef225bc7a87129d53fbe",
    "103aa91793e7de9647104ff73feb74834592363227889bad1afa909cb0790672",
    "e33369534386d631f01bce481fbd7d7af5a4fcf49565e04610a9e5d0e5463fe0",
    "f90433455a3b956d0a01c6565f389d3940d807cdc3e4208c68b7d4056afe4777",
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _compatible(identity):
    return (isinstance(identity, dict) and identity.get("implementation") == "retrieval-encoder-v1"
            and identity.get("implementation_hash") in EQUIVALENT_IMPLEMENTATION_HASHES
            and identity.get("signature") == _digest({k: v for k, v in identity.items() if k != "signature"}))


def _with_implementation(identity, implementation_hash):
    result = {k: v for k, v in identity.items() if k != "signature"}
    result["implementation_hash"] = implementation_hash
    return {**result, "signature": _digest(result)}


def canonical_identity(identity):
    if not _compatible(identity):
        return identity
    return _with_implementation(identity, CANONICAL_IMPLEMENTATION_HASH)


def cache_identities(identity):
    """Prefer the portable identity, while keeping existing Windows caches readable."""
    if not _compatible(identity):
        return [identity]
    hashes = dict.fromkeys([CANONICAL_IMPLEMENTATION_HASH, *EQUIVALENT_IMPLEMENTATION_HASHES])
    return [_with_implementation(identity, value) for value in hashes]


def compatible_identities(left, right):
    return canonical_identity(left) == canonical_identity(right)

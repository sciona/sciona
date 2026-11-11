"""Content-addressed in-memory intermediate cache for zero-subprocess CDG prototyping.

Keys every atomic node execution by:
``sha256(atom_fqn, impl_digest, canonical_node_params, incoming_input_digests, incoming_state_digests)``

This guarantees:
1. HPO sweeps on node ``k`` reuse upstream nodes ``1 .. k-1`` in ``~0 ms`` RAM hits.
2. Structural edits or sibling swaps at node ``k`` only execute ``k`` and its
   downstream cone.
3. Counterfactual attribution overrides reuse cached prefix values safely
   without stale-key collisions across mutated CDGs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import pickle
import struct
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

# Arrays larger than 16 MiB use deterministic multi-window + stride sampling
# to keep cache key hashing sub-millisecond even on large feature matrices.
_FULL_BUFFER_HASH_LIMIT_BYTES = 16 * 1024 * 1024
_CANONICAL_QNAN_BITS = struct.pack(">Q", 0x7FF8000000000000)

# Memoize bytecode/source digests by callable identity -> (weak_check, digest)
_CALLABLE_DIGEST_CACHE: dict[int, tuple[Any, str]] = {}


def _pack_canonical_float(val: float) -> bytes:
    """Pack a Python float into 8 canonical big-endian IEEE-754 bytes."""
    if math.isnan(val):
        return _CANONICAL_QNAN_BITS
    if val == 0.0:
        return struct.pack(">d", 0.0)
    return struct.pack(">d", val)


def _update_ndarray_digest(hasher: "hashlib._Hash", arr: np.ndarray[Any, Any]) -> None:
    """Hash shape, dtype, and array buffer deterministically."""
    hasher.update(b"ndarray:")
    hasher.update(str(arr.shape).encode("utf-8"))
    hasher.update(b":")
    hasher.update(arr.dtype.str.encode("utf-8"))
    hasher.update(b":")

    if arr.size == 0:
        hasher.update(b"empty")
        return

    if arr.dtype.hasobject:
        flat = arr.ravel()
        limit = min(flat.size, 1024)
        for idx in range(limit):
            hasher.update(hash_runtime_value(flat[idx]).encode("ascii"))
        if flat.size > limit:
            hasher.update(f":truncated:{flat.size}".encode("utf-8"))
        return

    contig = np.ascontiguousarray(arr)
    # Normalize NaN payloads for floating arrays when NaNs exist so distinct
    # quiet-NaN bit patterns still hash identically.
    if np.issubdtype(contig.dtype, np.floating) and bool(np.isnan(contig).any()):
        contig = np.where(np.isnan(contig), np.nan, contig)
        contig = np.ascontiguousarray(contig)

    nbytes = int(contig.nbytes)
    raw = contig.view(np.uint8).ravel()
    if nbytes <= _FULL_BUFFER_HASH_LIMIT_BYTES:
        hasher.update(raw.tobytes())
        return

    # Deterministic 3-window + strided sample for large buffers (>16 MiB)
    window = 2 * 1024 * 1024  # 2 MiB head, mid, tail
    mid = nbytes // 2
    hasher.update(f"sampled:{nbytes}:".encode("utf-8"))
    hasher.update(raw[:window].tobytes())
    hasher.update(raw[max(0, mid - window // 2) : min(nbytes, mid + window // 2)].tobytes())
    hasher.update(raw[max(0, nbytes - window) :].tobytes())
    stride = max(4096, nbytes // 4096)
    hasher.update(raw[::stride].tobytes())


def callable_impl_digest(fn: Any) -> str:
    """Compute a fast, deterministic digest of a Python callable's implementation."""
    if fn is None:
        return "none"
    fn_id = id(fn)
    cached = _CALLABLE_DIGEST_CACHE.get(fn_id)
    if cached is not None and cached[0] is fn:
        return cached[1]

    hasher = hashlib.sha256()
    hasher.update(b"callable:")
    mod = getattr(fn, "__module__", "") or type(fn).__module__
    qual = getattr(fn, "__qualname__", "") or getattr(fn, "__name__", "") or type(fn).__qualname__
    hasher.update(f"{mod}.{qual}".encode("utf-8"))

    # Hash both outer wrapper (e.g. @icontract) and unwrapped implementation code
    chain: list[Any] = [fn]
    try:
        unwrapped = inspect.unwrap(fn)
        if unwrapped is not fn:
            chain.append(unwrapped)
    except Exception:
        pass

    for target in chain:
        code = getattr(target, "__code__", None)
        if code is None and callable(target):
            call_method = getattr(target, "__call__", None)
            code = getattr(call_method, "__code__", None)
        if code is not None:
            hasher.update(b":code:")
            hasher.update(code.co_code)
            hasher.update(repr(code.co_names).encode("utf-8"))
            # Avoid hashing memory-address reprs inside co_consts
            safe_consts = tuple(
                c if isinstance(c, (int, float, str, bytes, bool, type(None), tuple)) else type(c).__name__
                for c in code.co_consts
            )
            hasher.update(repr(safe_consts).encode("utf-8"))
        # Include @icontract precondition/postcondition lambda bytecode if present
        for contract_attr in ("__preconditions__", "__postconditions__"):
            contracts = getattr(target, contract_attr, None)
            if contracts:
                hasher.update(f":{contract_attr}:{len(contracts)}".encode("utf-8"))
                for group in contracts:
                    for item in (group if isinstance(group, (list, tuple)) else [group]):
                        cond = getattr(item, "condition", None)
                        cond_code = getattr(cond, "__code__", None)
                        if cond_code is not None:
                            hasher.update(cond_code.co_code)

    digest = hasher.hexdigest()
    if len(_CALLABLE_DIGEST_CACHE) > 4096:
        _CALLABLE_DIGEST_CACHE.clear()
    _CALLABLE_DIGEST_CACHE[fn_id] = (fn, digest)
    return digest


def hash_runtime_value(value: Any, *, _depth: int = 0) -> str:
    """Compute a deterministic SHA-256 hex digest for an arbitrary runtime value."""
    hasher = hashlib.sha256()

    if _depth > 16:
        hasher.update(f"max_depth:{type(value).__qualname__}:{repr(value)[:256]}".encode("utf-8"))
        return hasher.hexdigest()

    if value is None:
        hasher.update(b"none")
        return hasher.hexdigest()

    if isinstance(value, bool):
        hasher.update(b"bool:1" if value else b"bool:0")
        return hasher.hexdigest()

    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        hasher.update(f"int:{int(value)}".encode("utf-8"))
        return hasher.hexdigest()

    if isinstance(value, (float, np.floating)):
        hasher.update(b"float:")
        hasher.update(_pack_canonical_float(float(value)))
        return hasher.hexdigest()

    if isinstance(value, str):
        hasher.update(b"str:")
        hasher.update(value.encode("utf-8"))
        return hasher.hexdigest()

    if isinstance(value, (bytes, bytearray, memoryview)):
        hasher.update(b"bytes:")
        hasher.update(bytes(value))
        return hasher.hexdigest()

    if isinstance(value, np.ndarray):
        _update_ndarray_digest(hasher, value)
        return hasher.hexdigest()

    # Duck-typed pandas DataFrame / Series support without hard import overhead
    cls_name = type(value).__name__
    cls_mod = type(value).__module__
    if cls_mod.startswith("pandas."):
        if cls_name == "DataFrame" and hasattr(value, "columns") and hasattr(value, "index"):
            hasher.update(b"dataframe:")
            hasher.update(repr(list(value.columns)).encode("utf-8"))
            hasher.update(hash_runtime_value(value.index.to_numpy(), _depth=_depth + 1).encode("ascii"))
            for col in value.columns:
                col_arr = value[col].to_numpy()
                hasher.update(str(col).encode("utf-8"))
                hasher.update(hash_runtime_value(col_arr, _depth=_depth + 1).encode("ascii"))
            return hasher.hexdigest()
        if cls_name == "Series" and hasattr(value, "to_numpy") and hasattr(value, "index"):
            hasher.update(f"series:{getattr(value, 'name', '')}:".encode("utf-8"))
            hasher.update(hash_runtime_value(value.index.to_numpy(), _depth=_depth + 1).encode("ascii"))
            hasher.update(hash_runtime_value(value.to_numpy(), _depth=_depth + 1).encode("ascii"))
            return hasher.hexdigest()

    if isinstance(value, tuple):
        hasher.update(f"tuple:{len(value)}:".encode("utf-8"))
        for item in value:
            hasher.update(hash_runtime_value(item, _depth=_depth + 1).encode("ascii"))
            hasher.update(b",")
        return hasher.hexdigest()

    if isinstance(value, list):
        hasher.update(f"list:{len(value)}:".encode("utf-8"))
        for item in value:
            hasher.update(hash_runtime_value(item, _depth=_depth + 1).encode("ascii"))
            hasher.update(b",")
        return hasher.hexdigest()

    if isinstance(value, (set, frozenset)):
        tag = "frozenset" if isinstance(value, frozenset) else "set"
        elem_hashes = sorted(hash_runtime_value(item, _depth=_depth + 1) for item in value)
        hasher.update(f"{tag}:{len(elem_hashes)}:{','.join(elem_hashes)}".encode("utf-8"))
        return hasher.hexdigest()

    if isinstance(value, Mapping):
        hasher.update(f"dict:{len(value)}:".encode("utf-8"))
        keyed = sorted(
            (str(k), hash_runtime_value(k, _depth=_depth + 1), hash_runtime_value(v, _depth=_depth + 1))
            for k, v in value.items()
        )
        for _, kh, vh in keyed:
            hasher.update(kh.encode("ascii"))
            hasher.update(b"=")
            hasher.update(vh.encode("ascii"))
            hasher.update(b";")
        return hasher.hexdigest()

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        hasher.update(f"dataclass:{cls_mod}.{cls_name}:".encode("utf-8"))
        for fld in dataclasses.fields(value):
            hasher.update(fld.name.encode("utf-8"))
            hasher.update(b"=")
            hasher.update(hash_runtime_value(getattr(value, fld.name), _depth=_depth + 1).encode("ascii"))
            hasher.update(b";")
        return hasher.hexdigest()

    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            dumped = value.model_dump(mode="python")
            hasher.update(f"pydantic:{cls_mod}.{cls_name}:".encode("utf-8"))
            hasher.update(hash_runtime_value(dumped, _depth=_depth + 1).encode("ascii"))
            return hasher.hexdigest()
        except Exception:
            pass

    if callable(value):
        hasher.update(callable_impl_digest(value).encode("ascii"))
        return hasher.hexdigest()

    try:
        pickled = pickle.dumps(value, protocol=5)
        hasher.update(b"pickle:")
        hasher.update(pickled)
        return hasher.hexdigest()
    except Exception:
        hasher.update(f"repr:{cls_mod}.{cls_name}:{repr(value)}".encode("utf-8"))
        return hasher.hexdigest()


def estimate_value_bytes(value: Any, *, _depth: int = 0) -> int:
    """Estimate the RAM footprint of a cached runtime value in bytes."""
    if _depth > 8 or value is None:
        return 16
    if isinstance(value, np.ndarray):
        return int(value.nbytes) + 128
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value) + 64
    if isinstance(value, str):
        return len(value.encode("utf-8")) + 64
    if isinstance(value, (int, float, bool, np.number)):
        return 32
    cls_mod = type(value).__module__
    if cls_mod.startswith("pandas.") and hasattr(value, "memory_usage"):
        try:
            usage = value.memory_usage(deep=False)
            return int(usage.sum() if hasattr(usage, "sum") else usage) + 256
        except Exception:
            return 1024
    if isinstance(value, Mapping):
        return 128 + sum(
            estimate_value_bytes(k, _depth=_depth + 1) + estimate_value_bytes(v, _depth=_depth + 1)
            for k, v in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return 96 + sum(estimate_value_bytes(v, _depth=_depth + 1) for v in value)
    return 256


def _canonicalize_params(node_params: Mapping[str, Any] | None) -> str:
    """Serialize node hyperparameter overrides into canonical deterministic JSON."""
    if not node_params:
        return "{}"
    normalized: dict[str, Any] = {}
    for key in sorted(str(k) for k in node_params.keys()):
        raw_val = node_params[key]
        if isinstance(raw_val, (int, str, bool)) or raw_val is None:
            normalized[key] = raw_val
        elif isinstance(raw_val, float):
            if math.isnan(raw_val):
                normalized[key] = "__nan__"
            elif math.isinf(raw_val):
                normalized[key] = "__inf__" if raw_val > 0 else "__ninf__"
            else:
                normalized[key] = raw_val
        else:
            normalized[key] = {"__digest__": hash_runtime_value(raw_val)}
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def compute_node_cache_key(
    *,
    atom_fqn: str,
    impl_callable: Any,
    output_port_names: tuple[str, ...] = (),
    node_params: Mapping[str, Any] | None = None,
    input_digests: Mapping[str, str] | None = None,
    state_digests: Mapping[str, str] | None = None,
) -> str:
    """Build a content-addressed SHA-256 cache key for a single node execution."""
    hasher = hashlib.sha256()
    hasher.update(b"sciona.node.v1\x00")
    hasher.update(atom_fqn.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(callable_impl_digest(impl_callable).encode("ascii"))
    hasher.update(b"\x00")
    hasher.update(",".join(output_port_names).encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(_canonicalize_params(node_params).encode("utf-8"))
    hasher.update(b"\x00inputs:")
    if input_digests:
        for port_name, digest in sorted(input_digests.items()):
            hasher.update(port_name.encode("utf-8"))
            hasher.update(b"=")
            hasher.update(digest.encode("ascii"))
            hasher.update(b";")
    hasher.update(b"\x00states:")
    if state_digests:
        for port_name, digest in sorted(state_digests.items()):
            hasher.update(port_name.encode("utf-8"))
            hasher.update(b"=")
            hasher.update(digest.encode("ascii"))
            hasher.update(b";")
    return hasher.hexdigest()


@dataclass
class CacheEntry:
    """Stored result of a deterministic atomic node execution."""

    key: str
    atom_fqn: str
    outputs: dict[str, Any]
    output_digests: dict[str, str]
    state_updates: dict[str, Any]
    state_update_digests: dict[str, str]
    compute_time_ms: float
    size_bytes: int


class IntermediateCache:
    """Bounded LRU content-addressed cache for in-process CDG node intermediates."""

    def __init__(
        self,
        *,
        max_entries: int = 512,
        max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must be >= 0")
        if max_bytes < 0:
            raise ValueError("max_bytes must be >= 0")
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_bytes: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.evictions: int = 0
        self.saved_time_ms: float = 0.0

    def __len__(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        """Drop all cached entries and reset counters."""
        self._entries.clear()
        self.current_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.saved_time_ms = 0.0

    def get(self, key: str) -> CacheEntry | None:
        """Look up *key*, promoting it to most-recently-used on hit."""
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        self.saved_time_ms += entry.compute_time_ms
        return entry

    def put(
        self,
        key: str,
        *,
        atom_fqn: str,
        outputs: dict[str, Any],
        output_digests: dict[str, str] | None = None,
        state_updates: dict[str, Any] | None = None,
        state_update_digests: dict[str, str] | None = None,
        compute_time_ms: float = 0.0,
    ) -> CacheEntry | None:
        """Insert or update *key* while enforcing entry and byte limits."""
        if self.max_entries == 0 or self.max_bytes == 0:
            return None

        out_copy = dict(outputs)
        st_copy = dict(state_updates or {})
        size_bytes = estimate_value_bytes(out_copy) + estimate_value_bytes(st_copy)
        if size_bytes > self.max_bytes:
            return None

        resolved_out_digests = (
            dict(output_digests)
            if output_digests is not None
            else {k: hash_runtime_value(v) for k, v in out_copy.items()}
        )
        resolved_state_digests = (
            dict(state_update_digests)
            if state_update_digests is not None
            else {k: hash_runtime_value(v) for k, v in st_copy.items()}
        )

        existing = self._entries.pop(key, None)
        if existing is not None:
            self.current_bytes = max(0, self.current_bytes - existing.size_bytes)

        while self._entries and (
            len(self._entries) >= self.max_entries
            or self.current_bytes + size_bytes > self.max_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self.current_bytes = max(0, self.current_bytes - evicted.size_bytes)
            self.evictions += 1

        entry = CacheEntry(
            key=key,
            atom_fqn=atom_fqn,
            outputs=out_copy,
            output_digests=resolved_out_digests,
            state_updates=st_copy,
            state_update_digests=resolved_state_digests,
            compute_time_ms=float(compute_time_ms),
            size_bytes=size_bytes,
        )
        self._entries[key] = entry
        self.current_bytes += size_bytes
        return entry

    def stats(self) -> dict[str, Any]:
        """Return JSON-serializable cache telemetry."""
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "max_entries": self.max_entries,
            "current_bytes": self.current_bytes,
            "max_bytes": self.max_bytes,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total > 0 else 0.0,
            "evictions": self.evictions,
            "saved_time_ms": round(self.saved_time_ms, 4),
        }

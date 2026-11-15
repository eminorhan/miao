"""TensorStore handle creation and management."""

from __future__ import annotations

import re
from pathlib import Path

import tensorstore as ts

from miao.zarr_meta import StoreFormat

_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")


def is_url(path: str | Path) -> bool:
    """True when `path` is a URL that tensorstore opens as a key-value store, such as
    `gs://bucket/dir`, `s3://bucket/dir`, `https://host/dir` or `file:///dir`, and False when it is
    a local path."""
    return isinstance(path, str) and _URL.match(path) is not None


def join_path(path: str | Path, *parts: str) -> str | Path:
    """`path` / part / ...: a `Path` for a local path, a string for a URL (empty parts skipped)."""
    if is_url(path):
        parts = tuple(p.strip("/") for p in parts)
        return "/".join([path.rstrip("/"), *(p for p in parts if p)])
    return Path(path).joinpath(*parts)


def _kvstore(directory: str | Path) -> dict | str:
    """The tensorstore key-value store of a directory: the file driver, or the URL itself."""
    if is_url(directory):
        return directory.rstrip("/") + "/"
    return {"driver": "file", "path": str(directory)}


def create_context(cache_bytes: int = 1 << 30, concurrency: int = 64) -> ts.Context:
    """Create a shared tensorstore Context with cache pool."""
    return ts.Context(
        {
            "cache_pool": {"total_bytes_limit": cache_bytes},
            "file_io_concurrency": {"limit": concurrency},
        }
    )


def open_store(
    path: str | Path,
    store_format: StoreFormat,
    context: ts.Context | None = None,
    fixed_index: dict[int, int] | None = None,
) -> ts.TensorStore:
    """Open a tensorstore handle for one pyramid level.

    Args:
        path: The level's array directory (zarr, N5), or for Neuroglancer precomputed the volume
            directory joined with the level's key. A local path, or a URL (see `is_url`).
        store_format: "zarr2", "zarr3", "n5" or "precomputed".
        context: Optional tensorstore context with cache settings.
        fixed_index: Optional {dimension: index} to pin; those dimensions are removed from the
            returned handle (see `zarr_meta.fix_axes`).
    """
    if not is_url(path):
        path = Path(path)
    if store_format == "precomputed":
        # One `info` file describes every level of a precomputed volume; a level is its key.
        if is_url(path):
            volume, _, key = path.rstrip("/").rpartition("/")
        else:
            volume, key = path.parent, path.name
        spec = {
            "driver": "neuroglancer_precomputed",
            "kvstore": _kvstore(volume),
            "scale_metadata": {"key": key},
        }
    else:
        spec = {
            "driver": {"zarr2": "zarr", "zarr3": "zarr3", "n5": "n5"}[store_format],
            "kvstore": _kvstore(path),
        }

    kwargs: dict = {"open": True}
    if context is not None:
        kwargs["context"] = context

    store = ts.open(spec, **kwargs).result()
    if store_format == "precomputed":
        # Its domain starts at the level's voxel_offset, and it always has a channel dimension;
        # miao indexes from 0 and describes a single-channel volume as x, y, z (miao.formats).
        store = store[ts.d[:].translate_to[0]]
        if store.shape[-1] == 1:
            store = store[ts.d["channel"][0]]
    if fixed_index:
        dims = sorted(fixed_index)
        store = store[ts.d[tuple(dims)][tuple(fixed_index[d] for d in dims)]]
    return store

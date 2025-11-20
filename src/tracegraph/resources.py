"""Resource adapters: fingerprint whatever a tool read or wrote.

A resource key is `scheme:locator`, e.g. `fs:shop/pagination.py`. An event's read set
maps keys to fingerprints taken when the tool ran; the Validator re-fingerprints them
later to tell whether anything a conclusion rests on has changed.

Locators are relative to an adapter's root, so the same key resolves in any checkout of
the same project (that is what makes cross-session validation work).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

from tracegraph.canonical import canonical_json, content_hash

ABSENT = "absent"

# A tool's resource declaration: args -> (keys it reads, keys it writes).
ToolResources = Callable[[dict], tuple[list[str], list[str]]]


def normalize_locator(path: str) -> str:
    norm = os.path.normpath(path.strip() or ".")
    if norm.startswith("..") or os.path.isabs(norm):
        raise ValueError(f"locator escapes the root: {path!r}")
    return norm


class ResourceAdapter(Protocol):
    schemes: tuple[str, ...]

    def fingerprint(self, scheme: str, locator: str) -> str: ...


class FilesystemAdapter:
    """`fs:<file>` hashes file bytes; `fsdir:<dir>` hashes the recursive file listing.

    Hashes are cached by (mtime_ns, size), so re-validating an unchanged tree is cheap.
    """

    schemes = ("fs", "fsdir")

    def __init__(self, root: str | os.PathLike[str], ignore: Iterable[str] = ("__pycache__", ".git")) -> None:
        self.root = Path(root).resolve()
        self.ignore = frozenset(ignore)
        self._cache: dict[Path, tuple[int, int, str]] = {}

    def _file_hash(self, path: Path) -> str:
        st = path.stat()
        cached = self._cache.get(path)
        if cached and cached[:2] == (st.st_mtime_ns, st.st_size):
            return cached[2]
        digest = content_hash(path.read_bytes())
        self._cache[path] = (st.st_mtime_ns, st.st_size, digest)
        return digest

    def files_under(self, locator: str) -> list[str]:
        base = self.root / normalize_locator(locator)
        if base.is_file():
            return [normalize_locator(locator)]
        if not base.is_dir():
            return []
        return sorted(
            str(p.relative_to(self.root))
            for p in base.rglob("*")
            if p.is_file() and not (self.ignore & set(p.relative_to(self.root).parts))
        )

    def fingerprint(self, scheme: str, locator: str) -> str:
        path = self.root / normalize_locator(locator)
        if scheme == "fs":
            return self._file_hash(path) if path.is_file() else ABSENT
        if scheme == "fsdir":
            return content_hash(canonical_json(self.files_under(locator))) if path.is_dir() else ABSENT
        raise KeyError(scheme)


class GitAdapter:
    """`git:<path>` is the git blob ID of the working-copy file (`git hash-object`)."""

    schemes = ("git",)

    def __init__(self, repo: str | os.PathLike[str]) -> None:
        self.repo = Path(repo).resolve()

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.repo, check=True, capture_output=True, text=True
        ).stdout.strip()

    def fingerprint(self, scheme: str, locator: str) -> str:
        path = self.repo / normalize_locator(locator)
        if not path.is_file():
            return ABSENT
        return "git:" + self._git("hash-object", "--", str(path.relative_to(self.repo)))

    def head(self) -> str:
        return self._git("rev-parse", "HEAD")


class ResourceRegistry:
    def __init__(self, *adapters: ResourceAdapter) -> None:
        self._by_scheme: dict[str, ResourceAdapter] = {}
        for adapter in adapters:
            for scheme in adapter.schemes:
                self._by_scheme[scheme] = adapter

    def fingerprint(self, key: str) -> str:
        scheme, _, locator = key.partition(":")
        adapter = self._by_scheme.get(scheme)
        if adapter is None:
            raise KeyError(f"no adapter for resource scheme {scheme!r}")
        return adapter.fingerprint(scheme, locator)

    def snapshot(self, keys: Iterable[str]) -> dict[str, str]:
        return {key: self.fingerprint(key) for key in sorted(set(keys))}

    def changed(self, read_set: dict[str, str]) -> dict[str, tuple[str, str]]:
        """Keys whose fingerprint differs now: {key: (then, now)}."""
        out = {}
        for key, then in read_set.items():
            try:
                now = self.fingerprint(key)
            except KeyError:
                continue  # no adapter in this context; can't judge
            if now != then:
                out[key] = (then, now)
        return out


def filesystem_tool_resources(adapter: FilesystemAdapter) -> dict[str, ToolResources]:
    """Resource declarations for the common read_file / list_files / grep / write_file tools."""

    def read_file(args: dict) -> tuple[list[str], list[str]]:
        return [f"fs:{normalize_locator(args['path'])}"], []

    def list_files(args: dict) -> tuple[list[str], list[str]]:
        return [f"fsdir:{normalize_locator(args.get('path', '.'))}"], []

    def grep(args: dict) -> tuple[list[str], list[str]]:
        scope = normalize_locator(args.get("path", "."))
        files = [f for f in adapter.files_under(scope) if f.endswith(".py")]
        return [f"fsdir:{scope}"] + [f"fs:{f}" for f in files], []

    def write_file(args: dict) -> tuple[list[str], list[str]]:
        key = f"fs:{normalize_locator(args['path'])}"
        return [], [key]

    return {"read_file": read_file, "list_files": list_files, "grep": grep, "write_file": write_file}

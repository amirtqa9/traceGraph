"""Content-addressed blob store (filesystem backend).

Large payloads (tool outputs, file contents) live here; LangGraph's Store and
checkpoints only ever hold `sha256:` references to them.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from tracegraph.canonical import content_hash


class BlobStore:
    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, ref: str) -> Path:
        algo, _, digest = ref.partition(":")
        if algo != "sha256" or len(digest) != 64:
            raise ValueError(f"not a blob ref: {ref!r}")
        return self.root / digest[:2] / digest[2:]

    def put(self, data: bytes) -> str:
        ref = content_hash(data)
        path = self._path(ref)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        return ref

    def get(self, ref: str) -> bytes:
        return self._path(ref).read_bytes()

    def has(self, ref: str) -> bool:
        return self._path(ref).exists()

    def refs(self) -> list[str]:
        return sorted(
            f"sha256:{p.parent.name}{p.name}" for p in self.root.glob("??/*") if p.is_file() and len(p.name) == 62
        )

    def delete(self, ref: str) -> int:
        """Remove a blob; returns bytes freed (0 if it was not there)."""
        path = self._path(ref)
        if not path.exists():
            return 0
        size = path.stat().st_size
        path.unlink()
        return size

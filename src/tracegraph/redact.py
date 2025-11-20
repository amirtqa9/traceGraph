"""Secret redaction at ingestion.

Tool outputs and arguments are scrubbed before they reach the blob store, the event log
or the checkpointed message history: anything content-addressed is effectively permanent,
so secrets must never get there. Patterns are conservative; extend `PATTERNS` per project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S)),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"sk-(?:proj-)?[A-Za-z0-9_\-]{20,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    # Secret-looking names assigned a quoted literal: PASSWORD = "...", "api_key": "...".
    # Unquoted values are left alone so code like `token = get_token()` stays readable.
    ("assignment", re.compile(
        r"(?i)((?:api[_-]?key|secret|password|passwd|token|access[_-]?key)[\"']?\s*[:=]\s*[\"'])"
        r"(?!\[REDACTED)([^\s\"']{8,})")),
    # .env style: SOME_SECRET_KEY=value on its own line.
    ("env_assignment", re.compile(
        r"(?im)^(\s*(?:export\s+)?[A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD)[A-Z0-9_]*=)(?![\"']?\[REDACTED)(\S{8,})$")),
)


@dataclass(frozen=True)
class Redaction:
    text: str
    kinds: tuple[str, ...]


def redact_text(text: str) -> Redaction:
    kinds: list[str] = []
    for kind, pattern in PATTERNS:
        if kind in ("assignment", "env_assignment"):
            text, n = pattern.subn(lambda m: f"{m.group(1)}[REDACTED:{kind}]", text)
        else:
            text, n = pattern.subn(f"[REDACTED:{kind}]", text)
        kinds += [kind] * n
    return Redaction(text, tuple(sorted(set(kinds))))


def redact_value(value: Any) -> tuple[Any, tuple[str, ...]]:
    """Redact strings anywhere inside JSON-like data."""
    if isinstance(value, str):
        r = redact_text(value)
        return r.text, r.kinds
    if isinstance(value, dict):
        kinds: set[str] = set()
        out = {}
        for k, v in value.items():
            out[k], found = redact_value(v)
            kinds.update(found)
        return out, tuple(sorted(kinds))
    if isinstance(value, list):
        kinds = set()
        items = []
        for v in value:
            item, found = redact_value(v)
            items.append(item)
            kinds.update(found)
        return items, tuple(sorted(kinds))
    return value, ()

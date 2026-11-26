import subprocess

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from spike.agent import build_agent
from tracegraph.redact import redact_text, redact_value
from tracegraph.resources import (
    ABSENT,
    FilesystemAdapter,
    GitAdapter,
    ResourceRegistry,
    filesystem_tool_resources,
    normalize_locator,
)

from conftest import ScriptedChatModel, call


def test_fs_fingerprints_files_and_listings(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "b.py").write_text("y = 2\n")
    reg = ResourceRegistry(FilesystemAdapter(tmp_path))
    before = reg.snapshot(["fs:a.py", "fsdir:pkg", "fs:missing.py"])
    assert before["fs:missing.py"] == ABSENT
    assert reg.changed(before) == {}

    (tmp_path / "a.py").write_text("x = 2\n")
    (tmp_path / "pkg" / "c.py").write_text("")
    changed = reg.changed(before)
    assert set(changed) == {"fs:a.py", "fsdir:pkg"}
    assert changed["fs:a.py"][0] == before["fs:a.py"]


def test_locators_cannot_escape_the_root():
    assert normalize_locator("./shop/../shop/x.py") == "shop/x.py"
    with pytest.raises(ValueError):
        normalize_locator("../etc/passwd")
    with pytest.raises(ValueError):
        normalize_locator("/etc/passwd")


def test_unknown_scheme_is_unavailable_not_matched(tmp_path):
    reg = ResourceRegistry(FilesystemAdapter(tmp_path))
    cmp = reg.compare({"http:example.com": "sha256:x"})
    assert cmp.changed == {} and cmp.unavailable == ("http:example.com",)
    assert not cmp.matched


def test_git_adapter_uses_blob_ids(tmp_path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f.txt").write_text("hello\n")
    git = GitAdapter(tmp_path)
    expected = subprocess.run(["git", "hash-object", "f.txt"], cwd=tmp_path, capture_output=True, text=True).stdout.strip()
    assert git.fingerprint("git", "f.txt") == f"git:{expected}"
    assert git.fingerprint("git", "nope.txt") == ABSENT


def test_grep_reads_every_scanned_file(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "b.txt").write_text("")
    fs = FilesystemAdapter(tmp_path)
    # Default: every file, so a grep over any file type gets a complete read set.
    assert filesystem_tool_resources(fs)["grep"]({"pattern": "x"}) == (["fsdir:.", "fs:a.py", "fs:b.txt"], [])
    decl = filesystem_tool_resources(fs, grep_glob="*.py")
    assert decl["grep"]({"pattern": "x"}) == (["fsdir:.", "fs:a.py"], [])
    assert decl["write_file"]({"path": "out.txt"}) == ([], ["fs:out.txt"])


@pytest.mark.parametrize("secret,kind", [
    ("sk-ant-api03-" + "a" * 40, "anthropic_key"),
    ("AKIA" + "ABCDEFGHIJKLMNOP", "aws_access_key"),
    ("ghp_" + "x" * 36, "github_token"),
    ("-----BEGIN RSA PRIVATE KEY-----\nabc\n-----END RSA PRIVATE KEY-----", "private_key"),
])
def test_redaction_patterns(secret, kind):
    out = redact_text(f"before {secret} after")
    assert secret not in out.text and kind in out.kinds and "before" in out.text


def test_assignment_redaction_keeps_the_name():
    out = redact_text('DB_PASSWORD="hunter2hunter2"\nport = 5432')
    assert out.text == 'DB_PASSWORD="[REDACTED:assignment]"\nport = 5432'


def test_env_lines_are_redacted_but_code_is_not():
    out = redact_text("export STRIPE_SECRET_KEY=abcd1234efgh\nDEBUG=true")
    assert out.text == "export STRIPE_SECRET_KEY=[REDACTED:env_assignment]\nDEBUG=true"
    code = "token = get_token(request)\npassword = form.get('password')"
    assert redact_text(code).text == code


def test_redact_value_walks_nested_data():
    value, kinds = redact_value({"cmd": ["curl", "-H", "token: 'abcdefghijkl123'"], "n": 3})
    assert value["cmd"][2] == "token: '[REDACTED:assignment]'" and value["n"] == 3 and kinds == ("assignment",)


def test_recorder_records_read_sets_and_redacts(harness, tmp_path):
    tg, _ = harness
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "settings.py").write_text('API_KEY = "sk-ant-api03-' + "z" * 40 + '"\nDEBUG = True\n')

    steps = iter([call("read_file", path="settings.py"), AIMessage("done")])
    from langgraph.checkpoint.memory import InMemorySaver
    agent = build_agent(ScriptedChatModel(policy=lambda m: next(steps)), ws, tg,
                        checkpointer=InMemorySaver(), store=tg.events.store)
    out = agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "r1"}})

    [event] = tg.events.for_thread("r1")
    assert set(event.read_set) == {"fs:settings.py"}
    assert event.redactions == ["anthropic_key"]
    stored = tg.blobs.get(event.output_ref).decode()
    shown = next(m for m in out["messages"] if isinstance(m, ToolMessage)).content
    for text in (stored, shown):
        assert "zzzz" not in text and "[REDACTED:anthropic_key]" in text and "DEBUG = True" in text


def test_tool_errors_become_results_not_crashes(harness, tmp_path):
    """Real models send bad input (gpt-oss sent grep an unterminated regex in the first live run)."""
    tg, _ = harness
    from langgraph.checkpoint.memory import InMemorySaver

    steps = iter([call("grep", pattern="def (paginate"), call("read_file", path="nope.py"), AIMessage("done")])
    agent = build_agent(ScriptedChatModel(policy=lambda m: next(steps)), tmp_path, tg,
                        checkpointer=InMemorySaver(), store=tg.events.store)
    out = agent.invoke({"messages": [HumanMessage("go")]}, {"configurable": {"thread_id": "e1"}})
    errors = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert [m.status for m in errors] == ["error", "error"]
    assert [e.status for e in tg.events.for_thread("e1")] == ["error", "error"]

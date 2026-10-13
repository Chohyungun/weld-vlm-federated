import json

import pytest

from corpus.production.common import (
    assert_gate_closed,
    canonical_bytes,
    iter_jsonl,
    parse_json,
    sha256_file,
    write_json_new,
)


def test_canonical_is_order_independent():
    assert canonical_bytes({"b": "한글", "a": 1}) == canonical_bytes({"a": 1, "b": "한글"})
    assert not canonical_bytes({}).endswith(b"\n")
    with pytest.raises(ValueError):
        canonical_bytes({"bad": float("nan")})


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_ambiguous_json_rejected(raw):
    with pytest.raises(ValueError):
        parse_json(raw)


def test_new_file_does_not_replace(tmp_path):
    path = tmp_path / "receipt.json"
    write_json_new(path, {"a": 1})
    digest = sha256_file(path)
    with pytest.raises(FileExistsError):
        write_json_new(path, {"a": 2})
    assert sha256_file(path) == digest
    assert json.loads(path.read_text()) == {"a": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_jsonl_rejects_blank_and_non_objects(tmp_path):
    path = tmp_path / "rows.jsonl"
    for raw in ('{}\n\n', '[]\n', '{"a":1,"a":2}\n'):
        path.write_text(raw, encoding="utf-8")
        with pytest.raises(ValueError):
            list(iter_jsonl(path))


def test_gate_is_enforced(tmp_path):
    (tmp_path / ".git").mkdir()
    assert_gate_closed(tmp_path)
    (tmp_path / ".git" / "CTO_GATE_OPEN").touch()
    with pytest.raises(ValueError):
        assert_gate_closed(tmp_path)


@pytest.mark.parametrize("exit_code", [0, 3, 4, 5])
def test_cli_preserves_stage_stop_code(tmp_path, monkeypatch, capsys, exit_code):
    from corpus.production import infer
    from corpus.production.__main__ import main

    config = tmp_path / "config.json"
    write_json_new(config, {})
    monkeypatch.setattr(infer, "run_stage", lambda *args, **kwargs: {"exit_code": exit_code})
    assert main(["generate", "--run", str(tmp_path), "--config", str(config)]) == exit_code
    assert json.loads(capsys.readouterr().out) == {"exit_code": exit_code}

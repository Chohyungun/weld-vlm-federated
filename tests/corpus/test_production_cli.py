"""prepare CLI의 원문·이용 확인·검수 입력 연결. 모델과 실데이터는 사용하지 않는다."""

import json

import pytest

from corpus.production import prepare
from corpus.production.__main__ import main
from corpus.production.common import write_json_new


@pytest.mark.parametrize("provided", [False, True])
def test_prepare_source_inputs_are_forwarded_without_inferred_defaults(tmp_path, monkeypatch, capsys, provided):
    config = {"schema_version": 1}
    config_path = tmp_path / "config.json"
    write_json_new(config_path, config)
    pairs, sources, run = (tmp_path / name for name in ("pairs.jsonl", "pointer.jsonl", "run"))
    argv = ["prepare", "--pairs", str(pairs), "--sources", str(sources),
            "--config", str(config_path), "--run", str(run)]
    expected = {}
    for option, keyword, filename in (
        ("--source-text", "source_text_path", "full.jsonl"),
        ("--source-registry", "source_registry_path", "registry.yaml"),
        ("--source-rights", "source_rights_path", "rights.jsonl"),
        ("--source-review", "source_review_path", "review.jsonl"),
    ):
        expected[keyword] = tmp_path / filename if provided else None
        if provided:
            argv.extend([option, str(expected[keyword])])
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return {"run_dir": str(run), "counts": {"source_qa": {"train": 0, "val": 0}}}

    monkeypatch.setattr(prepare, "prepare_run", capture)
    assert main(argv) == 0
    assert calls == [((pairs, sources, config, run), expected)]
    output = capsys.readouterr()
    assert json.loads(output.out)["counts"]["source_qa"] == {"train": 0, "val": 0}
    assert not run.exists()


def test_prepare_cli_reports_bank_refusal(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.json"
    write_json_new(config_path, {})

    def refuse(*args, **kwargs):
        raise prepare.PrepareRefused("source_bank:registry_sha256", "등록부 지문 불일치")

    monkeypatch.setattr(prepare, "prepare_run", refuse)
    assert main(["prepare", "--pairs", str(tmp_path / "pairs.jsonl"),
                 "--sources", str(tmp_path / "pointer.jsonl"), "--config", str(config_path),
                 "--run", str(tmp_path / "run"), "--source-registry", str(tmp_path / "registry.yaml")]) == 2
    output = capsys.readouterr()
    assert "source_bank:registry_sha256" in output.err
    assert not output.out
    assert not (tmp_path / "run").exists()

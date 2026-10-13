"""본실험 등록값(`configs/train_registration.yaml`)의 판독 · 맞대기 · 착수 고정 결속 — 모델 없이."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vlm import pilot_vlm
from vlm import registration as R
from vlm.pin import PinMismatch


def _uni(**over):
    base = {"num_rounds": 1, "local_epochs": 1, "total_epochs": 1, "max_new_tokens": 512, "seeds": (7, 8, 9)}
    base.update(over)
    return SimpleNamespace(**base)


def _rule(tmp: Path) -> Path:
    p = tmp / "early_coord_check.yaml"
    p.write_bytes(b"kind: early_coord_check\nversion: 3\n")
    return p


def _reg(tmp: Path, rule: Path, **over) -> dict:
    body = {"kind": R.KIND, "version": 1, "seed_indices": [1], "num_rounds": 1, "local_epochs": 1, "total_epochs": 1,
            "recipe": "json_only", "eval_gen_limit": 512, "quantization": dict(pilot_vlm.QUANT_4BIT),
            "micro_batch": pilot_vlm.MICRO_BATCH, "grad_accum": pilot_vlm.GRAD_ACCUM, "lr": pilot_vlm.LR,
            "early_coord_check_rules": {"version": 3, "sha256": hashlib.sha256(rule.read_bytes()).hexdigest()},
            "init_adapter_digest": None}
    body.update(over)
    return body


def _write(tmp: Path, body: dict) -> Path:
    p = tmp / "train_registration.yaml"
    p.write_text(yaml.safe_dump(body, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


def test_맞는_등록값은_파일_지문과_값을_낸다(tmp_path, monkeypatch):
    rule = _rule(tmp_path)
    p = _write(tmp_path, _reg(tmp_path, rule))
    monkeypatch.setattr("vlm.uni_config.load_uni_config", lambda path=None: _uni())
    got = R.registration_probe(None, path=p, rule_path=rule)
    assert got["registration_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert got["registration"]["eval_gen_limit"] == 512 and got["registration"]["recipe"] == "json_only"


@pytest.mark.parametrize(("over", "uni", "needle"), [
    ({"micro_batch": 2}, {}, "micro_batch"),
    ({"grad_accum": 16}, {}, "grad_accum"),
    ({"lr": 2e-4}, {}, "lr"),
    ({"quantization": {**dict(pilot_vlm.QUANT_4BIT), "bnb_4bit_quant_type": "fp4"}}, {}, "quantization"),
    ({"recipe": "two_task_mix"}, {}, "recipe"),
    ({}, {"num_rounds": None}, "num_rounds"),
    ({"num_rounds": 2, "total_epochs": 2}, {}, "num_rounds"),
    ({}, {"max_new_tokens": 128}, "eval_gen_limit"),
    ({"seed_indices": [4]}, {}, "seed_indices"),
    ({"seed_indices": [1, 1]}, {}, "seed_indices"),
    ({"early_coord_check_rules": {"version": 2, "sha256": "0" * 64}}, {}, "early_coord_check_rules"),
    ({"init_adapter_digest": "abc"}, {}, "init_adapter_digest"),
])
def test_코드_설정_규칙_파일과_다르면_어긋남을_든다(tmp_path, over, uni, needle):
    rule = _rule(tmp_path)
    bad = R.registration_problems(_reg(tmp_path, rule, **over), uni=_uni(**uni), rule_path=rule)
    assert any(needle in s for s in bad), bad


@pytest.mark.parametrize("case", ["missing_file", "extra_key", "missing_key", "kind", "version"])
def test_꼴이_다른_등록_파일은_읽지_않는다(tmp_path, case):
    rule = _rule(tmp_path)
    body = _reg(tmp_path, rule)
    if case == "extra_key":
        body["lora_r"] = 16
    elif case == "missing_key":
        del body["lr"]
    elif case == "kind":
        body["kind"] = "early_coord_check"
    elif case == "version":
        body["version"] = 2
    p = tmp_path / "없다.yaml" if case == "missing_file" else _write(tmp_path, body)
    with pytest.raises(R.RegistrationRejected):
        R.load_registration(p)


def test_등록_거부는_착수_고정의_거부라_세_자리가_닫힌_쪽으로_멈춘다(tmp_path, monkeypatch):
    """`pin` · 본실험 시작 문 · 조기 점검이 모두 `PinMismatch` 를 잡는다 — 등록 거부도 그 갈래다."""
    assert issubclass(R.RegistrationRejected, PinMismatch)
    rule = _rule(tmp_path)
    p = _write(tmp_path, _reg(tmp_path, rule, micro_batch=2))
    monkeypatch.setattr("vlm.uni_config.load_uni_config", lambda path=None: _uni())
    with pytest.raises(PinMismatch) as exc:
        R.registration_probe(None, path=p, rule_path=rule)
    assert exc.value.fields == ["registration"] and "micro_batch" in str(exc.value)


def test_착수_고정의_공통_칸이_등록값을_싣는다(tmp_path, monkeypatch):
    from vlm import pin
    from vlm.uni_config import UniConfig

    cfg = UniConfig(model_id="m", model_revision="r", pairs_path=None, pairs_digest=None,
                    chat_template_kwargs={"enable_thinking": False}, max_new_tokens=512, batch_size=1, num_rounds=1,
                    local_epochs=1, total_epochs=1, seeds=(7,), snapshot_digest=None, processor_kwargs={},
                    prompt_path=None, coord_space="ABS_ORIG")
    monkeypatch.setattr("vlm.uni_config.load_uni_config", lambda path=None: cfg)
    rule = _rule(tmp_path)
    p = _write(tmp_path, _reg(tmp_path, rule))
    monkeypatch.setattr(R, "REPO_ROOT", tmp_path)                      # 규칙 파일의 기본 자리를 시험 폴더로
    monkeypatch.setattr("vlm.ecc.RULE_PATH", Path(rule.name))
    common = pin.common_probe(None, registration_path=p)
    assert common["registration_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert common["registration"]["micro_batch"] == pilot_vlm.MICRO_BATCH
    with pytest.raises(PinMismatch):
        pin.common_probe(None, registration_path=tmp_path / "없다.yaml")


def test_4bit_적재_인자는_한_상수에서_나온다():
    import inspect

    src = inspect.getsource(pilot_vlm)
    assert 'q = dict(QUANT_4BIT)' in src and "load_in_4bit=True, bnb_4bit_quant_type=\"nf4\"" not in src
    assert dict(pilot_vlm.QUANT_4BIT) == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                         "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_use_double_quant": True}


def test_local_은_참여자_하나만_돌_수_있고_다른_명령은_참여자_인자를_받지_않는다(tmp_path, monkeypatch, capsys):
    import scripts.main_uni as M
    import vlm.train_cell as TC
    from tests.test_main_uni import _cfg_yaml

    p = _cfg_yaml(tmp_path)
    calls = []
    monkeypatch.setattr(TC, "run_uni_local_cell", lambda clients, *, spec: calls.append(tuple(clients)) or {})
    monkeypatch.setattr(M, "_cell_call", lambda fn, stage, n, out: fn())
    assert M.main(["local", "--seed", "1", "--client", "C2"], out=tmp_path / "out", config_path=p) == 0
    assert M.main(["local", "--seed", "1"], out=tmp_path / "out", config_path=p) == 0
    assert calls == [("C2",), ("C1", "C2", "C3")]
    assert M.main(["central", "--seed", "1", "--client", "C1"], out=tmp_path / "out", config_path=p) == 2
    assert "client_on_other" in capsys.readouterr().err


def test_확인_명령은_맞으면_0_어긋나면_2_와_사유를_낸다(tmp_path, monkeypatch, capsys):
    rule = _rule(tmp_path)
    monkeypatch.setattr("vlm.uni_config.load_uni_config", lambda path=None: _uni())
    monkeypatch.setattr(R, "REPO_ROOT", tmp_path)
    monkeypatch.setattr("vlm.ecc.RULE_PATH", Path(rule.name))
    ok = _write(tmp_path, _reg(tmp_path, rule))
    assert R.main(["--path", str(ok)]) == 0
    assert hashlib.sha256(ok.read_bytes()).hexdigest() in capsys.readouterr().out
    bad = _write(tmp_path, _reg(tmp_path, rule, grad_accum=16))
    assert R.main(["--path", str(bad)]) == 2
    assert "grad_accum" in capsys.readouterr().err

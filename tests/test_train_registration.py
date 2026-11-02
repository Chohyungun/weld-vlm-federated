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
    base = {"num_rounds": 1, "local_epochs": 1, "total_epochs": 1, "max_new_tokens": 512, "seeds": (7, 8, 9),
            "prompt_path": "vlm/prompts/unified_v2_absorig.txt", "model_id": "tiny", "model_revision": "r1",
            "pairs_digest": "d" * 64}
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def v2_table(tmp_path, monkeypatch):
    """새 레시피의 행별 토큰 표 — 합성 표 · 출처(합성 토크나이저 환경에서 낸 꼴, `_uni` 의 모델 · 페어 digest). 등록의 표 검사가
    실물 표 · 실물 캐시에 기대지 않게 한다."""
    from tests.uni_fakes import use_v2_token_table

    return use_v2_token_table(monkeypatch, tmp_path, pairs_digest="d" * 64)


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
    ({"init_adapter_digest": "g" * 64}, {}, "init_adapter_digest"),                     # 길이는 맞고 16진이 아니다(§137 M-02)
    ({"init_adapter_digest": "A" * 64}, {}, "init_adapter_digest"),                     # sha256 16진은 소문자다
    ({"init_adapter_digest": "a" * 64, "seed_indices": [1, 2]}, {}, "init_adapter_digest"),   # 시드마다 초기 어댑터가 다르다
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
                    prompt_path="vlm/prompts/unified_v2_absorig.txt", coord_space="ABS_ORIG")
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
    from tests.uni_fakes import use_registration

    use_registration(monkeypatch, tmp_path)                             # 시드 경계는 임시 등록으로
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


def test_초기_어댑터_digest_가_있으면_시드_하나와_소문자_16진이면_꼴은_맞다(tmp_path):
    rule = _rule(tmp_path)
    reg = _reg(tmp_path, rule, init_adapter_digest="0123456789abcdef" * 4)
    assert R.registration_problems(reg, uni=_uni(), rule_path=rule) == []


def test_결속은_등록의_시드_안과_실물_초기_어댑터를_본다():
    """§137 M-01 · M-02 — 시드 번호는 등록의 seed_indices 안이어야 하고, 등록에 digest 가 있으면 실물과 같아야 한다."""
    reg = {"seed_indices": [1], "init_adapter_digest": None}
    assert R.binding_problems(reg, seed_index=1) == []
    assert R.binding_problems(reg, seed_index=1, init_digest="f" * 64) == []        # 등록이 null 이면 digest 는 보지 않는다
    for bad in (2, 0, True, "1", None):
        assert any("seed_indices" in s for s in R.binding_problems(reg, seed_index=bad)), bad
    pinned = {**reg, "init_adapter_digest": "a" * 64}
    assert R.binding_problems(pinned, seed_index=1, init_digest="a" * 64) == []
    assert any("init_adapter_digest" in s for s in R.binding_problems(pinned, seed_index=1, init_digest="b" * 64))
    assert R.binding_problems(pinned, seed_index=1) == []                             # digest 를 안 주면 시드만(명령줄)


@pytest.mark.parametrize("seeds, ok", [([1], True), ([2, 1], True), ([], False), ([1, 1], False), ([0], False),
                                       ([True], False), (["1"], False), (1, False)])
def test_등록의_시드_번호는_서로_다른_양의_정수_목록이다(tmp_path, monkeypatch, seeds, ok):
    rule = _rule(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "train_registration.yaml").write_text(
        yaml.safe_dump(_reg(tmp_path, rule, seed_indices=seeds), allow_unicode=True, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(R, "REPO_ROOT", tmp_path)
    if ok:
        assert R.registered_seed_indices() == sorted(seeds)
    else:
        with pytest.raises(R.RegistrationRejected):
            R.registered_seed_indices()


def _cli_world(tmp_path, monkeypatch, seeds):
    from tests.test_main_uni import _cfg_yaml

    rule = _rule(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "train_registration.yaml").write_text(
        yaml.safe_dump(_reg(tmp_path, rule, seed_indices=seeds), allow_unicode=True, sort_keys=False), encoding="utf-8")
    monkeypatch.setattr(R, "REPO_ROOT", tmp_path)
    return _cfg_yaml(tmp_path)


@pytest.mark.parametrize("cmd", ["initadapter", "pin", "local", "central", "fed", "close-ledger", "gate-check",
                                 "ecc-prepare", "ecc", "ecc-accept", "all"])
def test_명령줄은_등록하지_않은_시드를_어느_명령으로도_시작하지_않는다(cmd, tmp_path, monkeypatch, capsys):
    """§137 M-01 — 설정의 시드표(셋)가 아니라 등록의 seed_indices([1])가 경계다. 단계 함수에 닿기 전에 멈춘다."""
    import scripts.main_uni as M

    p = _cli_world(tmp_path, monkeypatch, [1])
    for name in ("stage_initadapter", "stage_pin", "stage_local", "stage_central", "stage_fed", "stage_gate_check",
                 "stage_ecc_prepare", "stage_ecc", "stage_ecc_accept", "cmd_all"):
        monkeypatch.setattr(M, name, lambda *a, **k: pytest.fail("등록 밖 시드로 단계를 불렀다"))
    argv = [cmd, "--seed", "2"] + (["--tag", "uni_local_C1"] if cmd == "close-ledger" else [])
    assert M.main(argv, out=tmp_path / "out", config_path=p) == 2
    assert "seed_not_registered" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_all_의_기본은_설정의_시드표가_아니라_등록의_시드다(tmp_path, monkeypatch):
    import scripts.main_uni as M

    seen = []
    monkeypatch.setattr(M, "cmd_all", lambda cfg, seeds, **k: seen.append(list(seeds)))
    p = _cli_world(tmp_path, monkeypatch, [1])
    assert M.main(["all"], out=tmp_path / "out", config_path=p) == 0
    assert M.main(["all", "--seed", "1"], out=tmp_path / "out", config_path=p) == 0
    assert seen == [[1], [1]]                                             # 설정의 시드표는 셋이다(11 · 12 · 13)


def test_등록_파일이_없으면_명령줄이_시작하지_않는다(tmp_path, monkeypatch, capsys):
    import scripts.main_uni as M
    from tests.test_main_uni import _cfg_yaml

    monkeypatch.setattr(R, "REPO_ROOT", tmp_path)                       # 등록 파일이 없는 자리
    monkeypatch.setattr(M, "stage_pin", lambda *a, **k: pytest.fail("등록 없이 단계를 불렀다"))
    assert M.main(["pin", "--seed", "1"], out=tmp_path / "out", config_path=_cfg_yaml(tmp_path)) == 2
    assert "[registration]" in capsys.readouterr().err


@pytest.mark.parametrize("seeds", [[2], [3], [1, 3]])
def test_시드_경계는_등록의_값이다_개수로_1_부터_세지_않는다(tmp_path, monkeypatch, capsys, seeds):
    """'개수로 1..k' 변이(고치기 전 `range(1, len(…)+1)` 꼴)를 잡는다 — 등록이 [2] 면 경계는 [2] 다."""
    import scripts.main_uni as M

    seen = []
    monkeypatch.setattr(M, "cmd_all", lambda cfg, s, **k: seen.append(list(s)))
    monkeypatch.setattr(M, "stage_pin", lambda *a, **k: pytest.fail("등록 밖 시드로 단계를 불렀다"))
    p = _cli_world(tmp_path, monkeypatch, seeds)
    assert R.registered_seed_indices() == sorted(seeds)
    assert M.main(["all"], out=tmp_path / "out", config_path=p) == 0 and seen == [sorted(seeds)]
    outside = next(n for n in (1, 2, 3) if n not in seeds)
    assert M.main(["pin", "--seed", str(outside)], out=tmp_path / "out", config_path=p) == 2
    assert "seed_not_registered" in capsys.readouterr().err
    assert any("seed_indices" in s for s in R.binding_problems({"seed_indices": seeds}, seed_index=outside))
    assert R.binding_problems({"seed_indices": seeds, "init_adapter_digest": None}, seed_index=max(seeds)) == []


# ================================================================ 학습 레시피(02 2판 §3-5 · 지시 1022b)
def test_2차_레시피의_등록은_맞고_실행기는_그_레시피를_읽는다(tmp_path, monkeypatch, v2_table):
    rule = _rule(tmp_path)
    p = _write(tmp_path, _reg(tmp_path, rule, recipe="json_line_then_basis"))
    monkeypatch.setattr("vlm.uni_config.load_uni_config",
                        lambda path=None: _uni(prompt_path="vlm/prompts/unified_v3_1_basis.txt"))    # 2 차 체제의 프롬프트
    assert R.registration_probe(None, path=p, rule_path=rule)["registration"]["recipe"] == "json_line_then_basis"
    assert R.registered_recipe(p) == "json_line_then_basis"
    bad = _write(tmp_path, _reg(tmp_path, rule, recipe="two_task_mix"))
    with pytest.raises(R.RegistrationRejected, match="recipe"):
        R.registered_recipe(bad)


def test_보존한_1차_등록은_1차_레시피다(tmp_path):
    from tests.registration_history import historical_registration

    p = tmp_path / "old_registration.yaml"
    p.write_bytes(historical_registration())
    assert R.registered_recipe(p) == "json_only"


def test_지금_등록은_봉인_토큰_표와_같은_레시피다():
    import json

    from vlm.budget import PROVENANCE_SUFFIX, token_table_of
    from vlm.uni_config import load_uni_config

    reg = R.load_registration()[0]
    table, digest = token_table_of(reg["recipe"])
    assert hashlib.sha256((ROOT / table).read_bytes()).hexdigest() == digest
    prov = json.loads((ROOT / (str(table) + PROVENANCE_SUFFIX)).read_bytes())
    assert reg["recipe"] == prov["recipe"] == "json_line_then_basis"
    assert R.registration_problems(reg, uni=load_uni_config()) == []


@pytest.mark.parametrize(("reg_recipe", "run_recipe", "bad"), [
    ("json_only", "json_only", False),
    ("json_line_then_basis", "json_line_then_basis", False),
    ("json_only", "json_line_then_basis", True),            # 실행 설정이 등록과 갈린 채 학습하지 않는다
    ("json_line_then_basis", "json_only", True),
    ("json_only", None, False),                              # 레시피를 주지 않으면 시드만 본다(명령줄의 이른 경계)
])
def test_실행의_레시피가_등록과_다르면_결속이_어긋난다(reg_recipe, run_recipe, bad):
    got = R.binding_problems({"seed_indices": [1], "recipe": reg_recipe}, seed_index=1, recipe=run_recipe)
    assert bool(got) is bad and all("recipe" in s for s in got)


# ================================================================ 체제 짝(지시 1022c 덧붙임 A) — 레시피 × 프롬프트 × 한도
V2_PROMPT, V3_PROMPT = "vlm/prompts/unified_v2_absorig.txt", "vlm/prompts/unified_v3_1_basis.txt"


def test_실제_등록의_레시피만_2차로_바꾸면_설정이_그대로여도_거부한다(tmp_path, monkeypatch):
    """탐침 재현 — 보존된 1차 등록 바이트에서 recipe 한 줄만 바꾸고 보존된 base.yaml(프롬프트 v2 · 512)을 둔다.
    고치기 전에는 `[]` 이었다. 새 레시피의 표는 합성(실제 설정의 모델 · 페어 digest 로 낸 꼴)이라 거부 사유는 체제 짝만 남는다."""
    from vlm import pilot_vlm

    monkeypatch.setattr(pilot_vlm, "GRAD_ACCUM", 32)                    # 기록 등록(누적 32 시절)과 맞대는 시험 — 10-31 누적 8 변경 뒤에도 그 시절 코드값으로
    from tests.uni_fakes import use_v2_token_table
    from vlm.ecc import RULE_PATH
    from vlm.uni_config import load_uni_config

    from tests.registration_history import historical_base, historical_registration

    raw = historical_registration()
    assert raw.count(b"recipe: json_only") == 1
    p = tmp_path / "train_registration.yaml"
    p.write_bytes(raw.replace(b"recipe: json_only", b"recipe: json_line_then_basis"))
    uni = load_uni_config(historical_base())
    use_v2_token_table(monkeypatch, tmp_path, pairs_digest=uni.pairs_digest, model_id=uni.model_id,
                       revision=uni.model_revision)
    assert R.registration_problems(R.registration_from_bytes(raw), uni=uni,
                                   rule_path=ROOT / RULE_PATH) == []
    bad = R.registration_problems(R.load_registration(p)[0], uni=uni, rule_path=ROOT / RULE_PATH)
    assert bad and all("json_line_then_basis" in b for b in bad) and any(V3_PROMPT in b for b in bad)


@pytest.mark.parametrize(("recipe", "prompt", "limit", "needle"), [
    ("json_only", V2_PROMPT, 512, None),
    ("json_line_then_basis", V3_PROMPT, 1280, None),
    ("json_line_then_basis", V3_PROMPT, 1024, None),
    ("json_only", V3_PROMPT, 512, "프롬프트"),                    # 1 차 레시피 + v3
    ("json_line_then_basis", V2_PROMPT, 1280, "프롬프트"),         # 2 차 레시피 + v2
    ("json_only", V2_PROMPT, 1024, "한도는 512"),                 # 1 차는 1 차 등록의 한도
    ("json_line_then_basis", V3_PROMPT, None, "명시한 양의 정수"),  # 2 차 한도는 명시
])
def test_등록은_레시피_프롬프트_한도의_짝이_맞아야_선다(tmp_path, v2_table, recipe, prompt, limit, needle):
    rule = _rule(tmp_path)
    reg = _reg(tmp_path, rule, recipe=recipe, eval_gen_limit=limit)
    bad = R.registration_problems(reg, uni=_uni(prompt_path=prompt, max_new_tokens=limit), rule_path=rule)
    if needle is None:
        assert bad == []
    else:
        assert len(bad) == 1 and needle in bad[0] and recipe in bad[0]


def test_2차_등록은_행별_토큰_표의_출처가_지금_토크나이저와_다르면_거부한다(tmp_path, monkeypatch, v2_table):
    """외부 검토 200 M-01 — 등록 확인 · 착수 고정(`registration_probe`)이 표 출처의 신원을 지금 환경과 맞댄다. 페어 digest · 레시피 ·
    표는 그대로다. 1 차 등록은 표의 신원을 보지 않는다(v1: 신원 미기록)."""
    rule = _rule(tmp_path)
    uni = _uni(prompt_path=V3_PROMPT, max_new_tokens=1024)
    reg = _reg(tmp_path, rule, recipe="json_line_then_basis", eval_gen_limit=1024)
    assert R.registration_problems(reg, uni=uni, rule_path=rule) == []
    snap = tmp_path / "_hf_snapshot"
    (snap / "tokenizer_config.json").write_bytes(b'{"tokenizer_class": "Other"}\n')     # 같은 판 이름 · 다른 토크나이저
    bad = R.registration_problems(reg, uni=uni, rule_path=rule)
    assert len(bad) == 1 and "행별 토큰 표" in bad[0] and "tokenizer_config.json" in bad[0] and "json_line_then_basis" in bad[0]
    monkeypatch.setattr("vlm.uni_config.load_uni_config", lambda path=None: uni)
    with pytest.raises(R.RegistrationRejected, match="tokenizer_config.json"):           # 착수 고정의 공통 칸을 쓰지 않는다
        R.registration_probe(None, path=_write(tmp_path, reg), rule_path=rule)
    assert R.registration_problems(_reg(tmp_path, rule), uni=_uni(), rule_path=rule) == []   # 1 차는 그대로다


def test_체제_짝의_프롬프트는_자리가_아니라_모델이_받는_문자열로_맞댄다(tmp_path):
    from vlm.pilot_vlm import load_prompt

    copy = tmp_path / "copy_of_v3.txt"
    copy.write_bytes((ROOT / V3_PROMPT).read_bytes())                  # 다른 자리 · 같은 바이트 — 같은 체제다
    assert R.regime_problems("json_line_then_basis", prompt_sha256=load_prompt(copy)[1]) == []
    for recipe in R.TRAIN_RECIPES:                                      # 고정 상수 = 그 파일이 모델에 주는 문자열의 sha256
        assert R.regime_prompt_sha256(recipe) == load_prompt(ROOT / R.REGIME_PROMPTS[recipe])[1]
    assert R.regime_problems("two_task_mix", prompt_sha256="0" * 64)        # 짝이 없는 레시피
    assert R.regime_problems("json_only", prompt_sha256=load_prompt(ROOT / V2_PROMPT)[1], gen_limit=True)   # bool 한도
    assert set(R.REGIME_PROMPTS) == set(R.REGIME_LIMITS) == set(R.TRAIN_RECIPES)

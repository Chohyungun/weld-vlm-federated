"""2 차 체제의 등록 사슬 — 02 2판 §3-5 의 등록 사슬 표 · 외부 검토 194 M6(지시 1022b). 모델 · GPU 없이.

정본(`configs/*` · 봉인 표 · 저장소 등록)을 덮지 않는 **임시 2 차 등록**으로 사슬을 끝까지 돈다:
레시피 · 프롬프트 v3.1 · 한도 · 행별 토큰 표 · 페어 → 학습 등록 → 규칙 파일 해시 → `common_probe` · 착수 고정 → 실행기(칸 명세 ·
연합 실행 설정) → 중앙 · 로컬 칸이 `train_rounds` 에 실제로 넘기는 값과 연합 클라이언트가 받는 값 → 학습 설정 원문 · 재개 신원 →
본 칸의 시작 문. export 가 학습 설정 원문의 계약에서 레시피를 되찾는 것은 `tests/test_export_main_eval.py` 가 같은 합성 세계에서 본다.

거부 — 규칙 파일만 바뀐 경우 · 등록 해시만 바뀐 경우 · 새 착수 고정에 옛 체제의 허가 · 면제를 붙인 경우. 옛 체제(1 차 — `json_only`
· 프롬프트 v2 · 512)의 세계도 같은 꼴로 지어 그 착수 고정의 허가 · 면제를 쓴다.

면제 — 새 체제의 착수 고정은 면제를 받지 않는다(지시 1022c, 외부 검토 200 C-01). 1 차에서 받은 옛 다섯 칸 면제 · 다른 착수 고정을
가리킨 면제 · 승인 대상이 빠진 새 면제 · 지금 착수 고정을 가리킨 새 꼴(새 꼴은 사용자 결정 전) 모두 면제 파일을 읽거나 키 기록을
만들기 전에 거부되고 실행 폴더도 생기지 않는다. 새 체제는 착수 허가(통과 파일)로 선다. 1 차 체제의 옛 면제는 그대로 받는다(회귀).

실행기의 실행 묶음(`run_profile`)은 이 시험에서 임시 등록을 가리키게 바꿔 끼운다 — 묶음 파일 자체의 규칙(저장소 안의 등록 벌)은
`tests/test_sens_profile.py` 가 본다.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import main_uni as M
from tests import test_ecc_train as ET
from tests.test_ecc_waiver import _waiver
from tests.test_main_uni import _cfg_yaml
from tests.test_prompt_v3 import V2_SHA as PROMPT_V2_SHA
from tests.test_prompt_v3_1 import V3_1_SHA as PROMPT_V3_SHA          # 2 차 체제의 판 — v3.1(C 08 2 판 Q5)
from tests.test_uni_review_fixes import pairs_snapshot
from tests.uni_fakes import CountingLoader, fake_tokenizer_env, make_rows, write_pairs
from vlm import budget as B
from vlm import ecc as E
from vlm import pilot_vlm
from vlm import registration as R
from vlm.init_adapter import build_initial_adapter
from vlm.pin import build_pin, common_probe, init_probe, write_pin
from vlm.train_cell import CellRejected, main_start_gate

V1, V2 = "json_only", "json_line_then_basis"
PROMPTS = {V1: "vlm/prompts/unified_v2_absorig.txt", V2: "vlm/prompts/unified_v3_1_basis.txt"}
PROMPT_SHAS = {V1: PROMPT_V2_SHA, V2: PROMPT_V3_SHA}
#: 2 차 한도는 사용자 결정 전 — 시험 값(1 차 512 와 다르기만 하면 된다).
LIMITS = {V1: 512, V2: 1024}
SIZES = (("C1", 3), ("C2", 2), ("C3", 1))
#: 코드 · 환경 측정만 대역 — 작업 트리는 커밋 전이라 실물 코드 측정은 더럽다고 낸다. 공통 칸 · 초기 어댑터는 실물로 잰다.
CODE_ENV = {"code": lambda: {"commit": "c" * 40, "dirty": []},
            "env": lambda: {"torch": "t", "cuda": None, "gpu_name": None}}


@pytest.fixture(autouse=True)
def _no_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def chain(tmp_path, monkeypatch):
    """규칙 파일 하나(두 체제 공통) 위에 1 차 · 2 차 세계를 각자의 폴더 · 등록 · 출력 뿌리로 짓는다."""
    rule = ET._rule_file(tmp_path)
    monkeypatch.setattr(E, "RULE_PATH", rule)
    fake_tokenizer_env(monkeypatch, tmp_path)                          # 표 출처의 토크나이저 신원 — 합성 스냅샷(모델 tiny@r1)
    profiles: dict[Path, dict] = {}
    monkeypatch.setattr(M, "run_profile", lambda out: profiles[Path(out)])
    worlds = {r: _world(tmp_path / f"w_{r}", r, rule, profiles, monkeypatch) for r in (V1, V2)}
    return SimpleNamespace(tmp=tmp_path, rule=rule, w1=worlds[V1], w2=worlds[V2])


def _world(d: Path, recipe: str, rule: Path, profiles: dict, monkeypatch) -> SimpleNamespace:
    d.mkdir(parents=True)
    rows = [{**r, "width_px": 64, "height_px": 32, "target_text": f"{r['image_id']} 의 서술이다."}
            for c, n in SIZES for r in make_rows(d / "img", c, n)]
    write_pairs(d / "pairs.jsonl", rows)
    digest = pairs_snapshot(d)
    if recipe == V1:
        sums = {"C1": 10, "C2": 20, "C3": 30}                                   # 1 차 · f = 1 은 설정 값 그대로(표를 읽지 않는다)
    else:
        # 새 레시피의 표 — 계측 함수 그대로(`vlm.token_table`), 토크나이저만 글자 → UTF-8 바이트의 합성. 출처의 신원은 합성
        # 스냅샷의 것(`fake_tokenizer_env`) — 설정 모델(tiny@r1)의 지금 신원과 같다.
        from vlm.token_table import count_rows, provenance, table_bytes, tokenizer_identity

        counts = count_rows(rows, lambda ts: [list(t.encode("utf-8")) for t in ts], recipe=V2)
        raw = table_bytes(counts)
        table = d / "t2" / "per_row.csv"
        table.parent.mkdir()
        table.write_bytes(raw)
        prov = provenance(recipe=V2, counts=counts, table=raw, pairs_path=(d / "pairs.jsonl").as_posix(),
                          pairs_sha256=_sha(d / "pairs.jsonl"), pairs_digest=digest,
                          tokenizer_identity=tokenizer_identity("tiny", "r1"))
        (d / "t2" / ("per_row.csv" + B.PROVENANCE_SUFFIX)).write_text(json.dumps(prov), encoding="utf-8")
        monkeypatch.setitem(B.TOKEN_TABLES, V2, (table, hashlib.sha256(raw).hexdigest()))
        sums = {c: v["supervised_tokens"] for c, v in prov["per_client"].items()}
    cfg_path = _cfg_yaml(d, uni_model={"id": "tiny", "revision": "r1"},
                         uni_train_budget={"num_rounds": 1, "local_epochs": 1, "total_epochs": 1},
                         uni_max_new_tokens=LIMITS[recipe], uni_prompt_path=PROMPTS[recipe],
                         uni_expected_supervised_tokens=sums)
    reg = {"kind": R.KIND, "version": 1, "seed_indices": [1], "num_rounds": 1, "local_epochs": 1, "total_epochs": 1,
           "recipe": recipe, "eval_gen_limit": LIMITS[recipe], "quantization": dict(pilot_vlm.QUANT_4BIT),
           "micro_batch": pilot_vlm.MICRO_BATCH, "grad_accum": pilot_vlm.GRAD_ACCUM, "lr": pilot_vlm.LR,
           "early_coord_check_rules": {"version": yaml.safe_load(rule.read_bytes())["version"], "sha256": _sha(rule)},
           "init_adapter_digest": None}
    reg_path = d / "train_registration.yaml"
    reg_path.write_text(yaml.safe_dump(reg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    out = d / "out"
    profiles[out] = {"train_registration": reg_path, "run_name": f"r{1 if recipe == V1 else 2}"}
    cfg = M.load_main_config(cfg_path)
    spec = M.spec_for(cfg, 1, out=out, config_path=cfg_path)
    build_initial_adapter(model_id="tiny", seed=int(cfg.seeds[0]), cache_path=spec.init_adapter_path, revision="r1",
                          purpose="rehearsal", model_loader=CountingLoader())
    pin = build_pin(init_adapter_path=spec.init_adapter_path, config_path=cfg_path, registration_path=reg_path,
                    probes=dict(CODE_ENV))
    pin_sha = write_pin(spec.pin_path, pin)
    return SimpleNamespace(d=d, recipe=recipe, rows=rows, digest=digest, sums=sums, cfg_path=cfg_path, cfg=cfg,
                           reg_path=reg_path, out=out, spec=spec, pin=pin, pin_sha=pin_sha,
                           init_digest=init_probe(spec.init_adapter_path)["init_adapter_digest"])


def _gate(w, spec=None):
    return main_start_gate(spec or w.spec, init_digest=w.init_digest, pin_probes=dict(CODE_ENV))


def _code(fn) -> str:
    with pytest.raises(CellRejected) as exc:
        fn()
    return exc.value.code


# ================================================================ 사슬 — 등록에서 시작 문까지
def test_2차_등록은_코드_설정_규칙_파일과_맞고_공통_칸이_v3_프롬프트와_새_레시피를_싣는다(chain):
    w = chain.w2
    got = R.registration_probe(w.cfg_path, path=w.reg_path)
    assert got["registration"]["recipe"] == V2 and got["registration"]["eval_gen_limit"] == LIMITS[V2]
    assert got["registration_sha256"] == _sha(w.reg_path)
    common = common_probe(w.cfg_path, registration_path=w.reg_path)
    assert common["prompt_sha256"] == PROMPT_V3_SHA and common["registration"]["recipe"] == V2
    assert w.pin["common"] == common                                       # 착수 고정의 공통 칸 = 지금 실측
    c1 = common_probe(chain.w1.cfg_path, registration_path=chain.w1.reg_path)
    assert c1["prompt_sha256"] == PROMPT_V2_SHA and c1["registration"]["recipe"] == V1
    assert chain.w1.pin_sha != w.pin_sha


def test_실행기의_칸_명세와_연합_실행_설정이_2차_등록에서_온다(chain, monkeypatch):
    from flwr.common.config import fuse_dicts, parse_config_args

    from fl.uni_fed import fed_identity
    from fl.uni_run_config import client_run_cfg, down_config

    w = chain.w2
    s = w.spec
    assert (s.recipe, s.prompt_path, s.run_stamp, s.train_registration_path) == (V2, PROMPTS[V2], "r2_s1", w.reg_path)
    assert s.expected_supervised_tokens == w.sums                         # 새 표의 참여자 합(f = 1 에서도 표와 맞댐)
    assert (chain.w1.spec.recipe, chain.w1.spec.prompt_path) == (V1, PROMPTS[V1])
    # 연합 — 실행기 → flwr 실행 설정 → 서버 `down_config` → 클라이언트 `client_run_cfg`
    import tomllib

    declared = {**tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["flwr"]["app"]["config"],
                "uni-target-recipe": ""}                                     # 뿌리 파일의 선언(변경 요청)을 모사
    monkeypatch.setattr(M, "pyproject_run_keys", lambda path=None: frozenset(declared))
    run_cfg = fuse_dicts(declared, parse_config_args([M.fed_run_config(w.cfg, 1, out=w.out)]))
    uni = client_run_cfg(down_config(lambda k, d: run_cfg.get(k, d)))
    assert (uni["recipe"], uni["prompt_path"], uni["expected_supervised_tokens"]) == (V2, PROMPTS[V2], w.sums)
    ident = fed_identity(run_id="r", seed_index=1, seed_value=s.seed_value, purpose="main", num_rounds=1, local_epochs=1,
                         total_epochs=1, model_id=s.model_id, model_revision=s.model_revision, pairs_digest=s.pairs_digest,
                         prompt_sha256=PROMPT_V3_SHA, chat_template_kwargs=s.chat_template_kwargs,
                         coord_space=s.coord_space, init_digest=w.init_digest, snapshot_digest=s.snapshot_digest,
                         plan_sha256=None, client_tags=["C1", "C2", "C3"], processor_kwargs=s.processor_kwargs,
                         expected_supervised_tokens=w.sums, recipe=uni["recipe"])
    assert ident["target_contract_sha256"] == pilot_vlm.target_contract_sha256(V2)


def test_중앙_로컬_칸은_명세의_레시피와_프롬프트로_train_rounds_를_부르고_원문_신원이_새_계약을_말한다(chain, monkeypatch):
    """본실험 목적은 대역을 받지 않으므로(설계) 실행기 명세에서 **목적 · 경로만** 리허설로 바꾼 사본으로 돈다 — 레시피 · 프롬프트 ·
    모델 · 페어 · 템플릿 · 좌표 · 프로세서는 실행기가 낸 값 그대로다."""
    from vlm.train_cell import run_uni_central_cell, run_uni_local_cell

    w = chain.w2
    calls: list[dict] = []
    real = pilot_vlm.train_rounds

    def spy(**kw):
        calls.append({k: kw.get(k) for k in ("recipe", "prompt_path", "model_id", "chat_template_kwargs")})
        return real(**kw)

    monkeypatch.setattr(pilot_vlm, "train_rounds", spy)
    root = w.d / "reh-20261023T000000"
    loader = CountingLoader()
    reh = replace(w.spec, purpose="rehearsal", run_stamp="t1", train_root=root / "train",
                  init_adapter_path=root / "init" / "initial.npz", resume_root=root / "resume", run_root=root,
                  non_main_parent=w.d, plan_sha256="a" * 64, pin_path=None, ecc_permit_path=None, config_path=None,
                  train_registration_path=None, expected_supervised_tokens=None)
    build_initial_adapter(model_id="tiny", seed=reh.seed_value, cache_path=reh.init_adapter_path, revision="r1",
                          purpose="rehearsal", model_loader=loader)
    by_client = {c: [r for r in w.rows if r["client"] == c] for c, _ in SIZES}
    cen = run_uni_central_cell(spec=reh, model_loader=loader, rows=w.rows)
    loc = run_uni_local_cell(["C3"], spec=reh, model_loader=loader, rows_by_client=by_client)
    assert calls and all(c == {"recipe": V2, "prompt_path": PROMPTS[V2], "model_id": "tiny",
                               "chat_template_kwargs": {"enable_thinking": False}} for c in calls)
    v2 = pilot_vlm.target_contract_sha256(V2)
    for res in (cen, loc["C3"]):
        m = json.loads((res.cell_dir / "adapter_last.meta.json").read_text(encoding="utf-8"))
        tc = json.loads(m["train_config"])
        assert (tc["target_contract_sha256"], tc["prompt_sha256"]) == (v2, PROMPT_V3_SHA)
        assert m["identity"]["target_contract_sha256"] == v2
        assert m["metrics"]["resume_identity"]["target_contract_sha256"] == v2
        assert m["metrics"]["resume_identity"]["prompt_sha256"] == PROMPT_V3_SHA


def _permit_for(w, rule: Path) -> dict:
    """그 세계의 착수 고정으로 받은 점검 통과 — 그 시드 폴더에 사본과 키 기록(평가 쪽 실물 `build_output` · 학습 쪽 `accept_permit`)."""
    rules = E.load_check_rules(rule)["rules"]
    expected = {"start_adapter_digest": w.init_digest, "pin_sha256": w.pin_sha, "check_run_id": "uni_central-ecc",
                "seed": 1}
    src = w.d / "early_coord_check.json"
    src.write_bytes(ET._permit_bytes(w.d, rules, expected))
    E.accept_permit(src, rules=rules, expected=expected, seed_dir=w.spec.ecc_permit_path.parent,
                    permit_name=w.spec.ecc_permit_path.name, attempts_parent=w.d / "ecc_parent",
                    attempt_row={"run_id": "uni_central-ecc", "pin_sha256": w.pin_sha, "last_step": 100,
                                 "hours": 0.5, "won": 1500}, extra={"attempt": 1, "ecc_root": ET.ECC_ROOT})
    return expected


def test_본_칸의_시작_문은_새_착수_고정과_새_착수_허가로_서고_레시피가_갈리면_거부한다(chain):
    w = chain.w2
    _permit_for(w, chain.rule)
    gate = _gate(w)
    assert gate["pin_sha256"] == w.pin_sha and "ecc_permit_sha256" in gate and "ecc_permit_waiver_sha256" not in gate
    assert _code(lambda: _gate(w, replace(w.spec, recipe=V1))) == "registration_binding"


# ================================================================ 토큰 표 출처의 신원(지시 1023a — 외부 검토 200 M-01)
def _prov_path() -> Path:
    table = Path(B.TOKEN_TABLES[V2][0])
    return table.with_name(table.name + B.PROVENANCE_SUFFIX)


@pytest.mark.parametrize(("what", "needle"), [
    ("transformers", "transformers"),                     # 같은 페어 · 레시피 · 합 · 다른 꾸러미 판으로 계측한 표
    ("files", "tokenizer.json"),                          # 같은 판 이름 · 다른 토크나이저 바이트로 계측한 표
    ("counter", "계측 코드"),                              # 다른 계측 코드로 낸 표
])
def test_새_체제의_표_출처가_지금_토크나이저_계측_코드와_다르면_등록_실행기_가드_시작_문이_모델_전에_멈춘다(chain, capsys,
                                                                                  what, needle):
    w = chain.w2
    rec = json.loads(_prov_path().read_text(encoding="utf-8"))
    sums_before = {c: v["supervised_tokens"] for c, v in rec["per_client"].items()}
    if what == "counter":
        rec["counter"] = {**rec["counter"], "sha256_lf": "0" * 64}
    elif what == "files":
        rec["tokenizer"] = {**rec["tokenizer"], "files": {**rec["tokenizer"]["files"], "tokenizer.json": "0" * 64}}
    else:
        rec["tokenizer"] = {**rec["tokenizer"], "transformers": "0.0.0-other"}
    _prov_path().write_text(json.dumps(rec), encoding="utf-8")
    assert {c: v["supervised_tokens"] for c, v in rec["per_client"].items()} == sums_before == w.sums
    assert R.main(["--config", str(w.cfg_path), "--path", str(w.reg_path)]) == 2          # 등록 확인 — 착수 고정을 쓰지 않는다
    assert needle in capsys.readouterr().err
    with pytest.raises(M.MainRefused) as exc:                                          # 실행기 명세(감독 토큰 기대값)
        M.spec_for(w.cfg, 1, out=w.out, config_path=w.cfg_path)
    assert exc.value.reason == "token_table_provenance"
    with pytest.raises(M.MainRefused) as exc:                                          # 실행 가드 — 칸 · 연합 · 착수 고정 명령
        M._guarded(w.spec)
    assert exc.value.reason == "main_config_mismatch" and needle in capsys.readouterr().err
    assert _code(lambda: _gate(w)) == "pin_mismatch"                                   # 본 칸의 시작 · 재개 문(착수 고정 재측정)
    n_before = sum(1 for _ in w.d.rglob("adapter_last.npz"))
    from vlm.train_cell import check_before_load

    with pytest.raises(CellRejected) as exc:                                           # 칸 — 모델 적재 전의 가드
        check_before_load(w.spec)
    assert exc.value.code == "main_config_mismatch"
    assert sum(1 for _ in w.d.rglob("adapter_last.npz")) == n_before


def test_1차_체제는_토크나이저_신원을_재지_않고_지금처럼_지난다(chain, monkeypatch, capsys):
    """v1: 신원 미기록 — 1 차 표는 봉인 폴더 안이라 출처 파일이 없다. 1 차 경로는 신원을 재지 않고 등록 · 실행기 · 가드 · 착수 고정
    대조를 지금처럼 지난다(회귀)."""
    from vlm import token_table as T
    from vlm.pin import check_pin

    def boom(*a, **k):
        raise AssertionError("1 차 체제는 토크나이저 신원을 재지 않는다")

    monkeypatch.setattr(T, "tokenizer_identity", boom)
    monkeypatch.setattr(T, "_cached_file", boom)
    w = chain.w1
    assert R.main(["--config", str(w.cfg_path), "--path", str(w.reg_path)]) == 0
    assert "v1: 신원 미기록" in capsys.readouterr().out
    assert M.spec_for(w.cfg, 1, out=w.out, config_path=w.cfg_path).expected_supervised_tokens == w.sums
    M._guarded(w.spec)
    now = build_pin(init_adapter_path=w.spec.init_adapter_path, config_path=w.cfg_path, registration_path=w.reg_path,
                    probes=dict(CODE_ENV))
    assert check_pin(w.spec.pin_path, now) == w.pin_sha
    assert "token_table" not in json.dumps(now["common"])                              # 1 차 착수 고정의 공통 칸은 그대로다


# ================================================================ 거부 — 규칙 파일만 · 등록 해시만 · 옛 허가 · 면제
def test_규칙_파일만_바뀌면_등록이_어긋나고_시작_문이_거부한다(chain):
    w = chain.w2
    spec = w.spec                                                         # 착수 고정 대조가 허가 · 면제보다 먼저다
    reg_before = w.reg_path.read_bytes()
    ET._rule_file(chain.tmp, gen_limit=256)                                 # 같은 자리 · 다른 바이트
    assert w.reg_path.read_bytes() == reg_before
    bad = R.registration_problems(R.load_registration(w.reg_path)[0], uni=__import__("vlm.uni_config", fromlist=["x"])
                                  .load_uni_config(w.cfg_path), rule_path=chain.rule)
    assert any("early_coord_check_rules" in b for b in bad)
    assert _code(lambda: _gate(w, spec)) == "pin_mismatch"


def test_등록_해시만_바뀌면_값이_같아도_시작_문이_거부한다(chain):
    w = chain.w2
    spec = w.spec                                                         # 착수 고정 대조가 허가 · 면제보다 먼저다
    values = R.load_registration(w.reg_path)[0]
    w.reg_path.write_bytes(w.reg_path.read_bytes() + "# 값은 같다\n".encode())
    assert R.load_registration(w.reg_path)[0] == values
    assert R.registration_probe(w.cfg_path, path=w.reg_path)["registration_sha256"] != w.pin["common"]["registration_sha256"]
    assert _code(lambda: _gate(w, spec)) == "pin_mismatch"


def _old_permit(chain) -> tuple[Path, dict]:
    """옛 체제(1 차 세계)의 착수 고정으로 받은 점검 통과 — 1 차 시드 폴더에 사본과 키 기록."""
    w1 = chain.w1
    expected = _permit_for(w1, chain.rule)
    assert _gate(w1)["pin_sha256"] == w1.pin_sha                            # 옛 착수 고정에서는 선다
    return w1.spec.ecc_permit_path.parent, expected


@pytest.mark.parametrize("how", ["copy", "copy_rekey", "point"])
def test_새_착수_고정에_옛_허가를_붙이면_거부한다(chain, how):
    """`copy` — 옛 사본 · 키 기록을 새 시드 폴더로 · `copy_rekey` — 키 기록의 착수 고정 칸까지 새 값으로 고쳐 씀 ·
    `point` — 새 칸의 허가 자리를 옛 시드 폴더의 사본으로."""
    import shutil

    old_dir, _ = _old_permit(chain)
    w = chain.w2
    new_dir = w.spec.ecc_permit_path.parent
    spec = w.spec
    if how == "point":
        spec = replace(spec, ecc_permit_path=old_dir / w.spec.ecc_permit_path.name)
        want = "ecc_permit_key_mismatch"
    else:
        for name in (w.spec.ecc_permit_path.name, E.PERMIT_KEY_FILE):
            shutil.copyfile(old_dir / name, new_dir / name)
        want = "ecc_permit_key_mismatch"
        if how == "copy_rekey":
            kp = new_dir / E.PERMIT_KEY_FILE
            key = json.loads(kp.read_text(encoding="utf-8"))
            kp.write_text(json.dumps({**key, "pin_sha256": w.pin_sha, "start_adapter_digest": w.init_digest}),
                          encoding="utf-8")
            want = "ecc_permit_rejected"                                    # 통과 파일 안의 착수 고정이 다르다
    assert _code(lambda: _gate(w, spec)) == want


def test_새_착수_고정에_옛_면제의_키_기록을_붙여도_키를_보기_전에_거부한다(chain):
    import shutil

    w1, w = chain.w1, chain.w2
    old = _waiver(w1.d)
    assert "ecc_permit_waiver_sha256" in _gate(w1, replace(w1.spec, ecc_permit_waiver_path=old))   # 옛 체제에서 받음
    new_dir = w.spec.ecc_permit_path.parent
    new_dir.mkdir(parents=True, exist_ok=True)
    key = new_dir / E.WAIVER_KEY_FILE
    shutil.copyfile(w1.spec.ecc_permit_path.parent / E.WAIVER_KEY_FILE, key)
    before = key.read_bytes()
    assert _code(lambda: _gate(w, replace(w.spec, ecc_permit_waiver_path=old))) == "ecc_waiver_regime"
    assert key.read_bytes() == before                                       # 붙인 키 기록도 건드리지 않는다


def _seed_files(w) -> list[str]:
    d = w.spec.ecc_permit_path.parent
    return sorted(p.name for p in d.iterdir()) if d.exists() else []


@pytest.mark.parametrize("case", ["old_waiver_reused", "other_pin", "target_missing", "new_form_current_pin"])
def test_새_체제의_착수_고정은_면제를_키_기록_전에_거부하고_실행_폴더를_만들지_않는다(chain, case):
    """(a) 1 차에서 받은 옛 다섯 칸 면제 파일을 새 뿌리의 새 착수 고정에 처음 줌 · (b) 다른 착수 고정(1 차)을 가리킨 면제 ·
    (c) 승인 대상이 빠진 새 면제(다섯 칸) · (d) 지금 착수 고정을 가리킨 새 꼴 — 새 면제 꼴은 사용자 결정 전이라 이것도 받지 않는다."""
    from vlm.train_cell import CENTRAL_TAG, LOCAL_TAGS, cell_dir

    w1, w = chain.w1, chain.w2
    if case == "old_waiver_reused":
        p = _waiver(w1.d)
        assert "ecc_permit_waiver_sha256" in _gate(w1, replace(w1.spec, ecc_permit_waiver_path=p))   # 1 차에서는 받았다
    elif case == "other_pin":
        p = _waiver(w.d, name="other_pin.json", pin_sha256=w1.pin_sha)
    elif case == "target_missing":
        p = _waiver(w.d, name="new_decision.json", decided_at="2026-10-23T09:00:00+09:00")
    else:
        p = _waiver(w.d, name="new_form.json", pin_sha256=w.pin_sha)
    before = _seed_files(w)
    assert _code(lambda: _gate(w, replace(w.spec, ecc_permit_waiver_path=p))) == "ecc_waiver_regime"
    assert _seed_files(w) == before and E.WAIVER_KEY_FILE not in before          # 키 기록 0 — 시드 폴더가 그대로다
    assert not any(cell_dir(w.spec, t).exists() for t in (CENTRAL_TAG, *LOCAL_TAGS.values()))   # 실행 폴더 없음
    assert not M.fed_dir(w.out, 1).exists()


def test_면제_함수는_새_레시피면_면제_파일을_읽거나_키를_만들기_전에_거부한다(tmp_path):
    w = _waiver(tmp_path)
    sd = tmp_path / "seed1"
    kw = {"seed_dir": sd, "permit_path": None, "pin_sha256": "a" * 64, "seed": 1, "start_adapter_digest": "d" * 64}
    for path in (tmp_path / "없는_면제.json", w):                       # 파일을 읽기 전이다 — 없는 파일이어도 같은 사유
        with pytest.raises(E.EccRefused) as exc:
            E.check_waiver(path, **kw, recipe=V2)
        assert exc.value.code == "ecc_waiver_regime" and not sd.exists()
    with pytest.raises(TypeError):
        E.check_waiver(w, **kw)                                           # 레시피를 주지 않는 호출은 서지 않는다
    assert E.check_waiver(w, **kw, recipe=V1)["waiver_sha256"] == _sha(w)    # 1 차는 그대로 — 키 기록이 선다
    assert sorted(x.name for x in sd.iterdir()) == [E.WAIVER_KEY_FILE]


def test_1차_체제의_옛_면제는_그대로_받는다(chain):
    w1 = chain.w1
    old = _waiver(w1.d)
    gate = _gate(w1, replace(w1.spec, ecc_permit_waiver_path=old))
    assert gate["pin_sha256"] == w1.pin_sha and gate["ecc_permit_waiver_sha256"] == _sha(old)
    key = json.loads((w1.spec.ecc_permit_path.parent / E.WAIVER_KEY_FILE).read_text(encoding="utf-8"))
    assert (key["pin_sha256"], key["waiver_sha256"]) == (w1.pin_sha, _sha(old))
    assert _gate(w1, replace(w1.spec, ecc_permit_waiver_path=old)) == gate      # 재개 — 같은 기록에 같은 값


# ================================================================ A3 대체 승인(D7, 사용자 결정 10-31) — 실제 시작 문
def test_새_체제의_시작_문은_A3_대체_승인서를_받고_1_차_꼴은_여전히_거부한다(chain, tmp_path):
    """2 차 칸 명세의 면제 자리에 A3 대체 승인서(합성 저장소 · 증거 — `tests/test_ecc_substitute.make_world`)를 주면 실제 시작 문이 받고
    `start_gate` 에 승인서 sha 를 싣는다. 같은 자리에 1 차 꼴 면제 파일을 주면 종전처럼 키 기록 전에 `ecc_waiver_regime`. 1 차 칸은
    대체 승인서를 받지 않는다(1 차 꼴 검사로 가서 꼴이 틀려 거부)."""
    from tests.test_ecc_substitute import make_world

    w1, w = chain.w1, chain.w2
    sub = make_world(tmp_path / "sub")
    gate = main_start_gate(replace(w.spec, ecc_permit_waiver_path=sub.path), init_digest=w.init_digest,
                           pin_probes=dict(CODE_ENV), substitute_repo=sub.repo)
    assert gate["ecc_a3_substitute_sha256"] == hashlib.sha256(sub.raw).hexdigest() and gate["pin_sha256"] == w.pin_sha
    assert "ecc_permit_sha256" not in gate and gate["ecc_a3_substitute"]["a3_commit"] == sub.a3
    old = _waiver(w.d, name="old_form.json", pin_sha256=w.pin_sha)
    seed_dir = w.spec.ecc_permit_path.parent
    assert _code(lambda: _gate(w, replace(w.spec, ecc_permit_waiver_path=old))) == "ecc_waiver_regime"
    assert E.WAIVER_KEY_FILE not in [p.name for p in seed_dir.iterdir()]
    assert _code(lambda: main_start_gate(replace(w1.spec, ecc_permit_waiver_path=sub.path), init_digest=w1.init_digest,
                                         pin_probes=dict(CODE_ENV), substitute_repo=sub.repo)) == "ecc_waiver_form"

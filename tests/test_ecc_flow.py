"""조기 좌표 점검의 **한 흐름** — `tests/test_main_uni._cfg_yaml` 의 설정 세계로 공식 CLI 단계를 차례로 돈다(지시 1013i 의 3-a).

pin → ecc-prepare(대상 목록 · planned 줄 · ECC-local PIN) → ecc(step 2 도달 · 종료 4) → ecc(재개 → step 4) → ecc-accept(평가 쪽
`build_output` 의 통과 파일) → 본 칸의 시작 문(gate-check). 거부 — 시도 수 초과 · 대상 목록 불일치 · 규칙 지문 변경 · 루트 밖 PIN ·
네 검사 산출 파일 실물 없음 · 판별력 영수증 실물 없음 · 상한 인자 NaN · 점검 export 의 비용 가드.

대역은 모델 적재기뿐이다(`standin_allowed` 는 점검 칸 설정에서만 켠다 — 시험이 명시적으로 바꾼다). 설정 대조(`main_config_problems`) ·
착수 고정의 공통 칸(`common_probe` — 등록 파일 포함) · 평가 쪽 규칙 · 통과 파일 검증은 **실물**이다. 코드 · 환경 칸만 결정적 값으로
바꾼다(작업 트리가 커밋과 다른 개발 중에도 돌게 — 실물 `code_probe` · `common_probe` 는 `tests/test_pin_real_probes.py` 가 임시 git
저장소에서 본다).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.main_uni as M
from evaluation import early_coord_check as D
from evaluation.eval_list import canonical_bytes, validate_eval_list
from tests.test_main_uni import _cfg_yaml
from tests.uni_fakes import CountingLoader, make_rows, write_pairs
from vlm import ecc as E
from vlm import pilot_vlm, pin, registration
from vlm.exposure import find_planned
from vlm.train_cell import CENTRAL_TAG, cell_dir, cell_run_id

TGT = validate_eval_list(canonical_bytes([f"synth:t{i:03d}" for i in range(5)]))
STEPS = [2, 4, 8]


@pytest.fixture(autouse=True)
def _small_world(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(pilot_vlm, "GRAD_ACCUM", 2)
    monkeypatch.setattr(pin, "code_probe", lambda repo=None: {"commit": "c" * 40, "dirty": []})
    monkeypatch.setattr(pin, "env_probe", lambda repo=None: {"torch": "t", "cuda": None, "gpu_name": None, "driver": None})


def _rules(tmp: Path, **over) -> Path:
    """커밋된 평가 쪽 규칙 파일의 칸을 시험 값으로 — 판별력 영수증은 평가 쪽 4판 계약대로 실물로 쓴다(§119 I-05 ·
    `verify_discrimination`: 입력은 이 규칙의 `discrimination_inputs`, 반복 1, 고른 순차 규칙의 사다리 하나 — 합성)."""
    b = yaml.safe_load(D.RULES_PATH.read_bytes().decode("utf-8"))
    b.update({"steps": STEPS, "sequential_rule": "single", "max_attempts": 1,
              "target_list": {"sha256": TGT.file_sha256, "n": len(TGT)}})
    b["limits"] = {**b["limits"], "max_hours": 30, "max_cost_krw": 90000}
    b.update(over)
    probe = {**b, "discrimination": {"path": "x.json", "sha256": "0" * 64}}
    rules = D.load_rules(yaml.safe_dump(probe, allow_unicode=True, sort_keys=False).encode("utf-8"))
    cell = {"false_pass": 0, "false_fail": 0, "positive_runs": 1, "negative_runs": 1}
    disc = tmp / "discrimination.json"
    disc.write_text(json.dumps({"inputs": D.discrimination_inputs(rules), "reps": 1,
                                "tally": {"synthetic": {rules.sequential_rule: cell}}}, ensure_ascii=False), encoding="utf-8")
    b["discrimination"] = {"path": str(disc), "sha256": hashlib.sha256(disc.read_bytes()).hexdigest()}
    p = tmp / "early_coord_check.yaml"
    p.write_bytes(yaml.safe_dump(b, allow_unicode=True, sort_keys=False).encode("utf-8"))
    return p


def _registration(tmp: Path, rule: Path, uni: dict) -> Path:
    reg = {"kind": registration.KIND, "version": 1, "seed_indices": [1], "num_rounds": uni["num_rounds"],
           "local_epochs": uni["local_epochs"], "total_epochs": uni["total_epochs"], "recipe": "json_only",
           "eval_gen_limit": uni["uni_max_new_tokens"], "quantization": dict(pilot_vlm.QUANT_4BIT),
           "micro_batch": pilot_vlm.MICRO_BATCH, "grad_accum": pilot_vlm.GRAD_ACCUM, "lr": pilot_vlm.LR,
           "early_coord_check_rules": {"version": yaml.safe_load(rule.read_bytes())["version"],
                                       "sha256": hashlib.sha256(rule.read_bytes()).hexdigest()},
           "init_adapter_digest": None}
    (tmp / "configs").mkdir(exist_ok=True)
    p = tmp / "configs" / "train_registration.yaml"
    p.write_text(yaml.safe_dump(reg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return p


class World:
    """설정 · 규칙 · 등록 · 페어(합성 그림) · 출력 자리 — 모두 tmp 아래. 공통 초기 어댑터는 대역 적재기로 만든다."""

    def __init__(self, tmp: Path, monkeypatch):
        from vlm.init_adapter import build_initial_adapter

        self.tmp, self.out = tmp, tmp / "out"
        self.rows = make_rows(tmp / "img", "C1", 7) + make_rows(tmp / "img", "C2", 4) + make_rows(tmp / "img", "C3", 1)
        write_pairs(tmp / "pairs.jsonl", self.rows)                   # `_cfg_yaml` 의 계약(SNAPSHOT)이 이 바이트를 잰다
        self.rule = _rules(tmp)
        budget = {"num_rounds": 1, "local_epochs": 2, "total_epochs": 2}
        self.cfg_path = _cfg_yaml(tmp, uni_train_budget=budget, uni_max_new_tokens=512)
        _registration(tmp, self.rule, {**budget, "uni_max_new_tokens": 512})
        monkeypatch.setattr(registration, "REPO_ROOT", tmp)           # 등록 파일 · 규칙 파일의 기본 자리를 tmp 로
        monkeypatch.setattr(E, "RULE_PATH", self.rule)
        self.cfg = M.load_main_config(self.cfg_path)
        self.loader = CountingLoader()
        self.ledger = tmp / "exposure" / "uni_exposure_ledger.jsonl"
        self.target = tmp / "targets.txt"
        self.target.write_bytes(canonical_bytes(list(TGT.ids)))
        self.parent = tmp / "early_check"
        self.parent.mkdir()
        orig = M.spec_for_ecc                                          # 점검 칸만 대역 적재기를 받는다 — 본 칸은 그대로
        monkeypatch.setattr(M, "spec_for_ecc", lambda *a, **k: replace(orig(*a, **k), standin_allowed=True,
                                                                         non_main_parent=self.parent))
        self.main = M.spec_for(self.cfg, 1, out=self.out, config_path=self.cfg_path)
        build_initial_adapter(model_id=self.main.model_id, seed=self.main.seed_value, cache_path=self.main.init_adapter_path,
                              revision=self.main.model_revision, purpose=E.PURPOSE, model_loader=self.loader,
                              standin_allowed=True)
        self.pin_sha = M.stage_pin(self.cfg, 1, out=self.out, config_path=self.cfg_path)

    def root(self, stamp: str = "ecc-20261013T000000") -> Path:
        return self.parent / stamp

    def ecc_spec(self, root: Path):
        return M.spec_for_ecc(self.cfg, 1, root, config_path=self.cfg_path, non_main_parent=self.parent)

    def prepare(self, root: Path):
        return M.stage_ecc_prepare(self.cfg, 1, root, target_list=self.target, out=self.out, config_path=self.cfg_path,
                                   non_main_parent=self.parent, exposure_ledger=self.ledger)

    def ecc(self, root: Path, **kw):
        kw.setdefault("spent_hours", 0.5)
        kw.setdefault("next_hours", 0.5)
        return M.stage_ecc(self.cfg, 1, root, out=self.out, config_path=self.cfg_path, non_main_parent=self.parent,
                           model_loader=self.loader, **kw)

    def accept(self, root: Path, permit: Path):
        return M.stage_ecc_accept(self.cfg, 1, root, permit, out=self.out, config_path=self.cfg_path,
                                  non_main_parent=self.parent)

    def permit(self, root: Path, *, seed: int = 1) -> Path:
        rules = E.load_check_rules()["rules"]
        ecc = self.ecc_spec(root)
        expected = {"start_adapter_digest": pin.init_probe(ecc.init_adapter_path)["init_adapter_digest"],
                    "pin_sha256": self.pin_sha, "check_run_id": cell_run_id(ecc, CENTRAL_TAG), "seed": seed}
        d = self.tmp / "mf"
        d.mkdir(exist_ok=True)
        files = {}
        for n in D.MODEL_FREE_CHECKS:
            f = d / f"{n}.json"
            f.write_text(json.dumps({"name": n}), encoding="utf-8")
            files[n] = {"path": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        mf = [{"name": n, "status": D.PASS, "n": len(TGT), "n_bad": 0, "n_ids": len(TGT), "ids_set_sha256": TGT.set_sha256}
              for n in D.MODEL_FREE_CHECKS]
        steps = [{"status": D.PASS, "reason": "", "n_images": len(TGT), "ids_set_sha256": TGT.set_sha256}]
        ev = {"lineage": {"snapshot": "s", "pairs": "p", "r0": "r"}, "model_free_files": files,
              "step_outputs": [{"step": 2, "raw_sha256": "g", "parsed_sha256": "q", "consumed_sha256": "c"}],
              "attempt": 1, "step_adapter_digests": ["w2"], "processor_cfg_sha256": "pc", "model_input_wh": [1280, 704],
              **expected}
        p = self.tmp / "early_coord_check.json"
        p.write_bytes(D.canonical(D.build_output(rules=rules, target=TGT, model_free=mf, steps=steps, evidence=ev)))
        return p


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_한_흐름_pin_준비_점검_재개_통과_시작문(world: World):
    root = world.root()
    world.prepare(root)
    # ① 준비 — 대상 목록의 planned 줄 · ECC-local PIN · 준비 기록(step 목록 · 대상 지문 · 시도 번호)
    row = find_planned(world.ledger, run_id=root.name, list_kind="gen", list_file_sha256=TGT.file_sha256)
    assert row is not None and row["purpose"] == E.PURPOSE and row["n"] == len(TGT) and row["seed_index"] == 1
    assert pin.read_pin(root / M.PIN_FILE)[1] == world.pin_sha
    rec = json.loads((root / E.PREPARE_FILE).read_text(encoding="utf-8"))
    assert rec["steps"] == STEPS and rec["attempt"] == 1 and rec["target_list_sha256"] == TGT.file_sha256
    # ② 점검 — 설정 대조 · 착수 고정 대조는 실물, step 2 에서 멈춘다(종료 4)
    res = world.ecc(root)
    assert res["exit_code"] == 4 and res["step"] == 2 and world.loader.n >= 1
    # ③ 재개 — step 4 까지. 원장은 비용 줄 → 시작 · step · 멈춤 → 비용 줄 → 재개 · step · 멈춤
    res = world.ecc(root, resume_step=2, spent_hours=1.0)
    assert res["exit_code"] == 4 and res["step"] == 4
    events = [r["event"] for r in E.ledger_rows(cell_dir(world.ecc_spec(root), CENTRAL_TAG))]
    assert events == ["budget_ok", "start", "step", "stop", "budget_ok", "resume", "step", "stop"]
    # ④ 통과 — 평가 쪽 build_output 의 통과 파일을 평가 쪽 verify_permit 으로 받는다
    key = world.accept(root, world.permit(root))
    seed_dir = Path(world.main.pin_path).parent
    assert key["seed"] == 1 and key["pin_sha256"] == world.pin_sha and (seed_dir / E.PERMIT_KEY_FILE).is_file()
    attempts = [json.loads(x) for x in (world.parent / E.ATTEMPTS_FILE).read_text(encoding="utf-8").splitlines()]
    assert attempts[-1]["reason"] == "pass" and attempts[-1]["last_step"] == 4
    assert attempts[-1]["train_loop_hours"] == attempts[-1]["hours"] and attempts[-1]["spent_hours_last_budget"] == 1.0
    # ⑤ 본 칸의 시작 문 — gate-check(모델을 올리지 않는다) · 명령줄
    gate = M.stage_gate_check(world.cfg, 1, out=world.out, config_path=world.cfg_path)
    assert gate["pin_sha256"] == world.pin_sha and gate["permit_expected"]["seed"] == 1
    assert M.main(["gate-check", "--seed", "1"], out=world.out, config_path=world.cfg_path) == 0
    assert M.main(["gate-check", "--seed", "1", "--ecc-root", "x"], out=world.out, config_path=world.cfg_path) == 2


def test_준비는_시도_수_대상_목록을_보고_점검은_규칙_지문을_본다(world: World):
    a, b = world.root(), world.root("ecc-20261013T000001")
    world.prepare(a)
    with pytest.raises(SystemExit) as exc:                             # 허용 시도 수 1 — 두 번째 루트를 만들지 않는다
        world.prepare(b)
    assert exc.value.reason == "ecc_attempts_exhausted"
    shutil.rmtree(a)                                                   # 준비 기록이 사라지면 시도 수도 준다(보수적 셈의 뜻)
    world.target.write_bytes(canonical_bytes(list(TGT.ids)[:4]))       # 대상 목록이 규칙과 다르다 — 아무것도 쓰지 않는다
    with pytest.raises(SystemExit) as exc:
        world.prepare(b)
    assert exc.value.reason == "ecc_target_list" and not b.exists()
    world.target.write_bytes(canonical_bytes(list(TGT.ids)))
    world.prepare(b)
    world.ecc(b)
    world.rule.write_bytes(world.rule.read_bytes() + b"\n# changed\n")  # 규칙 지문이 준비 때와 다르면 재개하지 않는다
    with pytest.raises(SystemExit) as exc:
        world.ecc(b, resume_step=2, spent_hours=1.0)
    assert exc.value.reason == "ecc_rules_changed"


def test_점검_실행기는_준비가_옮긴_루트_안의_PIN_사본으로만_돈다(world: World):
    root = world.root()
    world.prepare(root)
    (root / M.PIN_FILE).unlink()                                        # §119 I-06 — 사본이 없으면 시작하지 않는다
    with pytest.raises(SystemExit) as exc:
        world.ecc(root)
    assert exc.value.reason == "pin_mismatch"
    shutil.copyfile(world.main.pin_path, root / M.PIN_FILE)
    assert world.ecc(root)["step"] == 2


def test_통과_파일의_네_검사_산출_파일_실물이_없으면_받지_않는다(world: World):
    root = world.root()
    world.prepare(root)
    world.ecc(root)
    permit = world.permit(root)
    (world.tmp / "mf" / "gold_echo.json").unlink()                      # §119 I-04 — 경로는 적혔는데 실물이 없다
    with pytest.raises(SystemExit) as exc:
        world.accept(root, permit)
    assert exc.value.reason == "ecc_permit_files"
    obj = json.loads(permit.read_text(encoding="utf-8"))
    obj["evidence"]["model_free_files"] = {n: {} for n in D.MODEL_FREE_CHECKS}
    permit.write_text(json.dumps(obj), encoding="utf-8")                # 빈 항목 넷 — 평가 쪽의 None == None 에 닿기 전에
    with pytest.raises(SystemExit) as exc:
        world.accept(root, permit)
    assert exc.value.reason == "ecc_permit_files"
    assert not (Path(world.main.pin_path).parent / E.PERMIT_KEY_FILE).exists()


def test_규칙의_판별력_영수증_실물을_본다(world: World):
    """평가 쪽 4판의 `verify_discrimination` 이 먼저 본다 — 학습 쪽은 그 거부를 `ecc_rules_rejected` 로 싣는다."""
    disc = world.tmp / "discrimination.json"
    disc.write_text("{}", encoding="utf-8")                              # 바이트가 적힌 sha 와 다르다(§119 I-05)
    with pytest.raises(E.EccRefused) as exc:
        E.load_check_rules()
    assert exc.value.code == "ecc_rules_rejected" and "DISCRIMINATION_SHA" in str(exc.value)
    disc.unlink()
    with pytest.raises(E.EccRefused) as exc:
        E.load_check_rules()
    assert exc.value.code == "ecc_rules_rejected" and "DISCRIMINATION_MISSING" in str(exc.value)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, True])
def test_상한_실행_인자는_유한_양수만_받는다(world: World, bad):
    rules = E.load_check_rules()["rules"]
    with pytest.raises(E.EccRefused) as exc:
        E.effective_limits(rules, max_hours=bad)
    assert exc.value.code == "ecc_budget_args"
    led = world.tmp / "led"
    with pytest.raises(E.EccRefused) as exc:                            # §119 I-09 — NaN 둘로 상한이 사라지지 않는다
        E.budget_check(led, spent_hours=100.0, next_hours=1.0, max_hours=float("nan"), max_won=float("nan"),
                       won_per_hour=3000.0, what="start")
    assert exc.value.code == "ecc_budget_args" and not (led / E.LEDGER_FILE).exists()
    with pytest.raises(E.EccRefused) as exc:
        E.budget_check(led, spent_hours=float("nan"), next_hours=1.0, max_hours=1.0, max_won=3000.0,
                       won_per_hour=3000.0, what="start")
    assert exc.value.code == "ecc_budget_args" and not (led / E.LEDGER_FILE).exists()


def test_점검_export_는_모델을_올리기_전에_비용_가드를_지난다(world: World, tmp_path):
    import scripts.export_uni as X

    root = world.root()
    world.prepare(root)
    world.ecc(root)
    adir = cell_dir(world.ecc_spec(root), CENTRAL_TAG)
    base = ["--purpose", "early_coord_check", "--check-step", "2", "--mode", "model", "--tag", "uni_central",
            "--seed-index", "1", "--receipt", str(tmp_path / "r.json"), "--list", str(world.target),
            "--snapshot", str(tmp_path / "s"), "--folder", str(tmp_path / "f"), "--run-root", str(root),
            "--adapter-dir", str(adir), "--device", "cpu"]
    loaded = []
    seams = {"load_generator": lambda *a, **k: loaded.append(1), "measure_env": lambda *a, **k: {}}
    assert X.main(base, **seams) == X.EXIT_REFUSED                      # 비용 인자 없이는 돌지 않는다
    assert X.main([*base, "--spent-hours", "40", "--next-hours", "1"], **seams) == X.EXIT_REFUSED   # 상한 30 h 초과
    assert [r["event"] for r in E.ledger_rows(adir)][-1] == "ecc_budget_exceeded" and loaded == []
    assert X.main([*base, "--spent-hours", "1", "--next-hours", "1"], **seams) == X.EXIT_REFUSED  # 가드를 지나고 영수증에서 멈춘다
    assert [r["event"] for r in E.ledger_rows(adir)][-1] == "budget_ok" and loaded == []
    assert X.main(["--purpose", "main", "--mode", "model", "--tag", "uni_central", "--seed-index", "1",
                   "--receipt", "r", "--list", "l", "--snapshot", "s", "--folder", "f", "--device", "cpu",
                   "--spent-hours", "1"], **seams) == X.EXIT_REFUSED   # 본실험은 비용 인자를 받지 않는다

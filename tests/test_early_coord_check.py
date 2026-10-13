"""조기 좌표 점검(`evaluation/early_coord_check.py`) — 미니스펙 10번 3판의 평가 쪽 구현 시험.

합성 정답 · 합성 id 만 쓴다. 실물 평가 자산 · 봉인 자산을 열지 않는다. 진단 검사기의 판정을 실제로 부르는 시험은 몇 개뿐이다.
착수 허가 · 대상 덮음의 음성은 원저자 아닌 검토(19b)의 탐침을 고정한 것이다 — 고치기 전 코드에서 떨어진다.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import early_coord_check as E  # noqa: E402
from evaluation import frame_diag as F  # noqa: E402
from evaluation import frame_diag_cases as C  # noqa: E402
from evaluation import frame_diag_rules as FR  # noqa: E402
from evaluation.eval_list import canonical_bytes, validate_eval_list  # noqa: E402

RAW = E.RULES_PATH.read_bytes()
P, X, I = E.PASS, E.FAIL, E.INDETERMINATE


def body() -> dict:
    return yaml.safe_load(RAW.decode("utf-8"))


def raw_of(b: dict) -> bytes:
    return yaml.safe_dump(b, allow_unicode=True, sort_keys=False).encode("utf-8")


def code_of(fn) -> tuple[str, str]:
    with pytest.raises(E.EarlyCheckRejected) as e:
        fn()
    return e.value.code, str(e.value)


def elist(ids):
    return validate_eval_list(canonical_bytes(list(ids)))


_SYNTH_DIR = Path(tempfile.mkdtemp(prefix="ecc_disc_"))


def synthetic_receipt(b: dict, *, rule: str = "single", reps: int = 3, tally=None) -> dict:
    """합성 판별력 영수증 — 규칙 파일 `b` 와 같은 입력으로 돌았다고 적은 작은 파일. {path, sha256} 을 돌려준다(절대 경로)."""
    base = dict(b)
    base.update({"sequential_rule": None, "discrimination": None})
    inputs = E.discrimination_inputs(E.load_rules(raw_of(base)))
    cell = {"false_pass": 0, "false_fail": 0, "positive_runs": 5, "negative_runs": 10}
    body_ = {"inputs": inputs, "reps": reps, "chosen_rule": rule,
             "tally": tally if tally is not None else {"합성 사다리": {"single": dict(cell), "two_consecutive": dict(cell)}}}
    raw = json.dumps(body_, ensure_ascii=False, sort_keys=True).encode("utf-8")
    path = _SYNTH_DIR / (hashlib.sha256(raw).hexdigest()[:16] + ".json")
    path.write_bytes(raw)
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def ready_rules(target=None, **over) -> E.EarlyRules:
    b = body()
    b.update({"sequential_rule": "single", "max_attempts": 2,
              "limits": {"max_hours": 3, "max_cost_krw": 20000, "price_krw_per_hour": 2500, "price_source": "합성", "safety_factor": 1.2}})
    if target is not None:
        b["target_list"] = {"sha256": target.file_sha256, "n": len(target)}
    b.update(over)
    if "discrimination" not in over:
        b["discrimination"] = synthetic_receipt(b)
    return E.load_rules(raw_of(b))


# ---------------------------------------------------------------- 규칙 파일

def test_커밋한_규칙_파일을_읽는다() -> None:
    r = E.load_rules(RAW)
    assert r.steps == (100, 200, 400) and r.gen_limit == 512 and r.target_n == 417
    assert (r.frame.min_pairs, r.frame.min_pair_groups, r.frame.min_groups) == (100, 20, 20)
    assert r.target_sha256 == "b7986fb78b9726dc4b24c892f78691e465e7be39125b636a2c818f381338f58e"
    assert r.model_input_wh == (1280, 704)
    assert set(r.limits) == E.LIMIT_KEYS and r.limits["price_krw_per_hour"] == 2500 and r.limits["safety_factor"] == 1.2


def test_진단_규칙_파일과_서로_받지_않는다() -> None:
    with pytest.raises(FR.RulesFileError):
        FR.load_rules_file(RAW)
    diag_like = FR.canonical({"version": "2", "rules": body()["rules"]})
    assert code_of(lambda: E.load_rules(diag_like))[0] == "RULES_KIND"


@pytest.mark.parametrize(("key", "value", "code"), [
    ("steps", [200, 100, 400], "RULES_STEPS"),
    ("steps", [], "RULES_STEPS"),
    ("n", 300, "RULES_N"),
    ("gen_limit", 0, "RULES_GEN_LIMIT"),
    ("sequential_rule", "majority", "RULES_SEQUENTIAL"),
    ("max_attempts", 0, "RULES_ATTEMPTS"),
    ("checker_sha256", "0" * 64, "CHECKER_CHANGED"),
    ("unexposed", {"unit": "image", "must_pass": True}, "RULES_UNEXPOSED"),
    ("unexposed", {"unit": "group", "must_pass": False}, "RULES_UNEXPOSED"),
    ("model_free_checks", ["original_size", "processor_input", "coord_roundtrip"], "RULES_CHECKS"),
    ("format_gate", {"min_parse_rate": 0}, "RULES_FORMAT_GATE"),
    ("limits", {"max_hours": 0, "max_cost_krw": None, "price_krw_per_hour": 2500, "price_source": "x", "safety_factor": 1.2},
     "RULES_LIMITS"),
    ("limits", {"max_hours": 1, "max_cost_krw": 1}, "RULES_LIMITS"),
    ("limits", {"max_hours": 1, "max_cost_krw": 1, "price_krw_per_hour": 0, "price_source": "x", "safety_factor": 1.2},
     "RULES_LIMITS"),
    ("limits", {"max_hours": 1, "max_cost_krw": 1, "price_krw_per_hour": 2500, "price_source": " ", "safety_factor": 1.2},
     "RULES_LIMITS"),
    ("limits", {"max_hours": 1, "max_cost_krw": 1, "price_krw_per_hour": 2500, "price_source": "x", "safety_factor": 0.9},
     "RULES_LIMITS"),
    ("version", 2, "RULES_VERSION"),
])
def test_규칙_파일의_어긋난_칸을_거부한다(key, value, code) -> None:
    b = body()
    b[key] = value
    assert code_of(lambda: E.load_rules(raw_of(b)))[0] == code


def test_모르는_칸과_빠진_칸을_거부한다() -> None:
    b = body()
    b["extra"] = 1
    assert code_of(lambda: E.load_rules(raw_of(b)))[0] == "RULES_KEYS"
    b = body()
    del b["limits"]
    assert code_of(lambda: E.load_rules(raw_of(b)))[0] == "RULES_KEYS"


def test_순차_규칙을_적으면_판별력_영수증도_적는다() -> None:
    b = body()
    b["sequential_rule"] = "single"
    b["discrimination"] = None
    assert code_of(lambda: E.load_rules(raw_of(b)))[0] == "RULES_DISCRIMINATION"


def test_미확정_값이_있으면_시작하지_않는다() -> None:
    b = body()
    b.update({"sequential_rule": None, "discrimination": None, "max_attempts": 2,
              "limits": {"max_hours": 3, "max_cost_krw": 20000, "price_krw_per_hour": 2500, "price_source": "합성", "safety_factor": 1.2}})
    assert code_of(lambda: E.require_ready(E.load_rules(raw_of(b))))[0] == "SEQUENTIAL_UNDECIDED"
    assert code_of(lambda: E.require_ready(ready_rules(max_attempts=None)))[0] == "ATTEMPTS_UNSET"
    unset = {"max_hours": None, "max_cost_krw": 1, "price_krw_per_hour": 2500, "price_source": "x", "safety_factor": 1.2}
    assert code_of(lambda: E.require_ready(ready_rules(limits=unset)))[0] == "LIMITS_UNSET"
    E.require_ready(ready_rules())


def test_판별력_영수증이_있으면_규칙_파일과_같은_입력으로_돌았다() -> None:
    disc = body()["discrimination"]
    if disc is None:
        pytest.skip("순차 규칙이 미확정 — 영수증을 규칙 파일에 걸지 않았다")
    path = ROOT / disc["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == disc["sha256"]
    receipt = json.loads(path.read_text(encoding="utf-8"))
    rules = E.load_rules(RAW)
    assert receipt["inputs"] == E.discrimination_inputs(rules)
    # 2판 영수증은 규칙을 고르지 않았다(`chosen_rule` null — 14-2 의 0 을 어느 규칙도 못 채웠다). 규칙은 등록 결정이고,
    # 영수증은 그 규칙의 사다리별 양성 · 음성 실행을 담아야 한다(4판 · §119 I-05)
    assert all(rules.sequential_rule in by_rule for by_rule in receipt["tally"].values())
    assert E.verify_discrimination(rules) == {s: {k: by_rule[rules.sequential_rule][k] for k in ("false_pass", "false_fail", "positive_runs", "negative_runs")}
                                             for s, by_rule in receipt["tally"].items()}


# ---------------------------------------------------------------- 대상 목록

def _pair(i, split, n_defects):
    return {"image_id": i, "split": split, "skeleton": {"defects": [{"type": "x"}] * n_defects}}


def test_대상은_학습_결함_장에서_R0_를_뺀_정렬_목록이다() -> None:
    rows = [_pair("synth:t3", "train", 1), _pair("synth:t1", "train", 2), _pair("synth:t2", "train", 0),
            _pair("synth:v1", "val", 1), _pair("synth:t4", "train", 1), _pair("synth:t1", "train", 2)]
    assert E.target_ids(iter(rows), {"synth:t4"}) == ["synth:t1", "synth:t3"]


def test_학습_뷰_밖의_id_는_건너뛰지_않고_거부한다() -> None:
    split_of = {"synth:t1": "train", "synth:t2": "train", "synth:v1": "val"}
    el = elist(["synth:t1", "synth:e1", "synth:x9", "synth:v1"])
    code, msg = code_of(lambda: E.check_target_list(el, split_of))
    assert code == "TARGET_NOT_TRAIN" and "뷰에 없음 2개" in msg and "학습 아님 1개" in msg
    assert not any(i in msg for i in el.ids), "사유에 id 를 싣지 않는다"
    E.check_target_list(elist(["synth:t1", "synth:t2"]), split_of)


def test_규칙의_대상과_다른_목록을_거부한다() -> None:
    assert code_of(lambda: E.check_target_list(elist(["synth:t1"]), {"synth:t1": "train"}, E.load_rules(RAW)))[0] \
        == "TARGET_MISMATCH"


def test_대상_생성기는_학습_분할만_읽는_판독기를_쓴다() -> None:
    src = (ROOT / "scripts/probe/early_coord_check_targets.py").read_text(encoding="utf-8")
    assert 'read_manifest_columns(a.snapshot, ["image_id", "split"], split_filter=("train",))' in src
    assert "check_target_list(el, split_of, rules)" in src


# ---------------------------------------------------------------- 모델 없이 가르는 네 검사

def test_프로세서_입력_크기는_격자_곱하기_patch_다() -> None:
    assert E.processor_input_wh((1, 44, 80), 16) == (1280, 704)
    assert E.image_tokens((1, 44, 80), 2) == 880
    ok = E.check_processor_input({"a": (1, 44, 80)}, patch=16, merge=2, model_input_wh=(1280, 704))
    assert ok["status"] == P and ok["image_tokens"] == [880] and ok["n_ids"] == 1
    bad = E.check_processor_input({"a": (1, 44, 80), "b": (1, 45, 80)}, patch=16, merge=2, model_input_wh=(1280, 704))
    assert bad["status"] == X and bad["n_bad"] == 1
    assert code_of(lambda: E.image_tokens((1, 3, 3), 2))[0] == "GRID_NOT_DIVISIBLE"


def test_원본_크기의_어긋남과_빠짐을_센다() -> None:
    man = {"a": (1280, 720), "b": (1280, 720)}
    assert E.check_original_size({"a": (1280, 720), "b": (1280, 720)}, man)["status"] == P
    r = E.check_original_size({"a": (1280, 721)}, man)
    assert r["status"] == X and r["n_bad"] == 2


def test_좌표_변환_왕복() -> None:
    golds = {"a": ((1280, 720), [(10.0, 20.0, 50.0, 60.0), (100.4, 200.6, 300.0, 400.0)])}
    assert E.check_coord_roundtrip(golds, input_wh=(1280, 704), coord_space="ABS_ORIG")["status"] == P
    assert E.check_coord_roundtrip(golds, input_wh=(1280, 704), coord_space="NORM_1000")["status"] == P


def test_정답_에코는_허용치를_인자로_받는다() -> None:
    golds = C.gold_set(30, 5)
    assert E.check_gold_echo(C.identity(golds), budget_px=0.0)["status"] == P
    assert E.check_gold_echo(C.resize_y(golds), budget_px=0.5)["status"] == X
    assert E.check_gold_echo(C.count_mismatch(golds, 3), budget_px=0.5)["status"] == X
    with pytest.raises(TypeError):
        E.check_gold_echo(C.identity(golds))


def test_네_검사가_모두_있어야_통과다() -> None:
    four = [{"name": n, "status": P} for n in E.MODEL_FREE_CHECKS]
    assert E.model_free_ok(four)
    assert not E.model_free_ok(four[:3])
    assert not E.model_free_ok(four[:3] + [{"name": "gold_echo", "status": X}])


# ---------------------------------------------------------------- step 의 판정 · 대상 덮음

def _target_of(golds):
    return elist([im.image_id for im in golds])


def test_형식_관문은_판정기를_부르지_않고_판정_불가다(monkeypatch) -> None:
    monkeypatch.setattr(F, "judge", lambda *a, **k: pytest.fail("형식 관문 아래에서 판정기를 불렀다"))
    golds = C.gold_set(20, 1)
    v = E.step_verdict(C.identity(golds), set(), E.load_rules(RAW), parse_rate=0.9, target=_target_of(golds))
    assert v["status"] == I and v["reason"] == E.REASON_FORMAT


def test_학습한_장만_맞히면_통과가_아니다() -> None:
    golds = C.gold_set(417, 3)
    groups = sorted({im.group_id for im in golds})
    exposed = set(groups[: len(groups) // 5])
    imgs = [replace(im, pred=im.gold) if im.group_id in exposed else replace(im, pred=()) for im in golds]
    v = E.step_verdict(imgs, exposed, E.load_rules(RAW), parse_rate=1.0, target=_target_of(golds))
    assert v["status"] != P and v["unexposed"]["status"] != P
    assert v["pair_rate_exposed"] == 1.0 and v["pair_rate_unexposed"] == 0.0


def test_옳은_프레임은_통과_리사이즈는_불통과() -> None:
    rules, golds = E.load_rules(RAW), C.gold_set(417, 4)
    exposed, tgt = {im.group_id for im in golds[:40]}, _target_of(golds)
    assert E.step_verdict(C.identity(golds), exposed, rules, parse_rate=1.0, target=tgt)["status"] == P
    assert E.step_verdict(C.resize_y(golds), exposed, rules, parse_rate=1.0, target=tgt)["status"] == X


@pytest.mark.parametrize("cut", ["subset", "outside", "duplicate"])
def test_대상_전부가_아니면_step_판정을_하지_않는다(cut, monkeypatch) -> None:
    monkeypatch.setattr(F, "judge", lambda *a, **k: pytest.fail("덮음 검사 전에 판정기를 불렀다"))
    golds = C.gold_set(417, 4)
    tgt = _target_of(golds)
    imgs = C.identity(golds)
    if cut == "subset":
        imgs = imgs[:200]                                       # 19b: 417 장 가운데 200 장으로 통과했다
    elif cut == "outside":
        imgs = imgs[:-1] + [replace(imgs[-1], image_id="eval:x1")]   # 19b: 대상 밖 id 로 통과했다
    else:
        imgs = imgs + [imgs[0]]
    code, msg = code_of(lambda: E.step_verdict(imgs, set(), E.load_rules(RAW), parse_rate=1.0, target=tgt))
    assert code == "TARGET_COVERAGE" and "eval:x1" not in msg


def test_비노출_미리_세기() -> None:
    rules = E.load_rules(RAW)
    group_of = {f"t{i}": f"g{i // 2}" for i in range(400)} | {"c1": "g0", "c2": "g1"}
    boxes = {f"t{i}": 1 for i in range(400)}
    out = E.unexposed_counts([{"c1"}, {"c1", "c2"}, {f"t{i}" for i in range(380)}], group_of, boxes, rules)
    assert [o["n_images"] for o in out] == [398, 396, 20]
    assert [o["floors_reachable"] for o in out] == [True, True, False]


# ---------------------------------------------------------------- 순차 결정

@pytest.mark.parametrize(("seq", "single", "two"), [
    ([P, I, P], (P, 0), (I, 2)),
    ([I, I, P], (P, 2), (I, 2)),
    ([I, X, P], (X, 1), (X, 1)),
    ([P, X], (P, 0), (X, 1)),
    ([P, P], (P, 0), (P, 1)),
    ([I, I, I], (I, 2), (I, 2)),
])
def test_순차_결정(seq, single, two) -> None:
    for rule, (outcome, at) in ((E.SINGLE, single), (E.TWO_CONSECUTIVE, two)):
        d = E.decide(seq, 3, rule)
        assert (d["outcome"], d["step_index"]) == (outcome, at), rule


def test_마지막_step_만_통과면_두_step_연속은_판정_불가() -> None:
    d = E.decide([I, I, P], 3, E.TWO_CONSECUTIVE)
    assert d["outcome"] == I and d["reason"] == E.REASON_NO_NEXT_STEP


def test_step_이_남았으면_이어_간다() -> None:
    assert E.decide([I], 3, E.SINGLE)["outcome"] == E.CONTINUE
    assert E.decide([P], 3, E.TWO_CONSECUTIVE)["outcome"] == E.CONTINUE
    assert code_of(lambda: E.decide([I, I, I, I], 3, E.SINGLE))[0] == "STEPS_EXCEEDED"
    assert code_of(lambda: E.decide([I], 3, None))[0] == "SEQUENTIAL_UNDECIDED"


def test_네_칸은_네_검사와_점검_통과가_함께일_때만_연다() -> None:
    four = [{"name": n, "status": P} for n in E.MODEL_FREE_CHECKS]
    assert E.opens_four_cells(four, {"outcome": P})
    assert not E.opens_four_cells(four, {"outcome": I})
    assert not E.opens_four_cells(four[:3], {"outcome": P})


def test_불통과의_동작은_검토와_보존을_적는다() -> None:
    assert "운영 책임자가" in E.ACTIONS[X] and "옆으로 옮겨 보존한다" in E.ACTIONS[X]
    assert not any(w in v for v in E.ACTIONS.values() for w in ("총괄", "사용자"))   # 공개 자료의 런타임 문자열(§119 M-07)


# ---------------------------------------------------------------- 산출물 · 착수 허가 · 채점 거부

TGT = elist([f"synth:t{i:03d}" for i in range(5)])
EXPECT = {"start_adapter_digest": "w0", "pin_sha256": "pin", "check_run_id": "run-1", "seed": 1}


def _mf(status=P, cover=TGT):
    return [{"name": n, "status": status if n == "gold_echo" else P, "n": len(cover), "n_bad": 0,
             "n_ids": len(cover), "ids_set_sha256": cover.set_sha256} for n in E.MODEL_FREE_CHECKS]


def _step(status=P, cover=TGT):
    return {"status": status, "reason": "", "n_images": len(cover), "ids_set_sha256": cover.set_sha256}


def _evidence(**over):
    ev = {"lineage": {"snapshot": "s", "pairs": "p", "r0": "r"},
          "model_free_files": {n: {"path": f"{n}.json", "sha256": hashlib.sha256(n.encode()).hexdigest()} for n in E.MODEL_FREE_CHECKS},
          "step_outputs": [{"step": 100, "raw_sha256": "g", "parsed_sha256": "q", "consumed_sha256": "c"}],
          "step_adapter_digests": ["w100"], "model_input_wh": [1280, 704], "attempt": 1, "processor_cfg_sha256": "pc",
          **EXPECT}
    ev.update(over)
    return ev


def _files(obj):
    return lambda p: {f["path"]: f["sha256"] for f in obj["evidence"]["model_free_files"].values()}.get(p)


def _output(rules=None, steps=None, model_free=None, **ev_over):
    rules = rules or ready_rules(TGT)
    obj = E.build_output(rules=rules, target=TGT, model_free=model_free or _mf(), steps=steps or [_step()],
                         evidence=_evidence(**ev_over))
    return rules, obj, E.canonical(obj)


def _verify(base_rules, raw, obj, **over):
    kw = {"expected_sha256": hashlib.sha256(raw).hexdigest(), "rules": base_rules, "expected": dict(EXPECT),
          "file_sha256_of": _files(obj)}
    kw.update(over)
    return E.verify_permit(raw, **kw)


def test_산출물은_채점_대상이_아니고_증거_묶음을_싣는다() -> None:
    rules, obj, raw = _output()
    assert (obj["kind"], obj["purpose"], obj["split"], obj["scoreable"]) == (E.KIND, E.KIND, "train", False)
    assert obj["decision"]["outcome"] == P and obj["opens_four_cells"] is True
    assert obj["evidence"]["target_set_sha256"] == TGT.set_sha256
    assert _verify(rules, raw, obj)["kind"] == E.KIND


@pytest.mark.parametrize(("change", "code"), [
    ("subset_step", "TARGET_COVERAGE"), ("subset_check", "TARGET_COVERAGE"), ("attempt", "ATTEMPTS_EXCEEDED"),
    ("not_ready", "SEQUENTIAL_UNDECIDED"), ("other_target", "TARGET_MISMATCH"), ("missing_evidence", "EVIDENCE_MISSING"),
])
def test_산출을_만들지_않는_경우(change, code) -> None:
    half = elist(TGT.ids[:2])
    kw = {}
    if change == "subset_step":
        kw["steps"] = [_step(cover=half)]
    elif change == "subset_check":
        kw["model_free"] = _mf(cover=half)
    elif change == "attempt":
        kw["attempt"] = 3
    elif change == "not_ready":
        b = body()
        b.update({"sequential_rule": None, "discrimination": None})     # 4판 실물은 준비됐다 — 미정을 명시한다
        b["target_list"] = {"sha256": TGT.file_sha256, "n": len(TGT)}
        kw["rules"] = E.load_rules(raw_of(b))
    elif change == "other_target":
        kw["rules"] = ready_rules()
    else:
        kw["lineage"] = None
    if change == "missing_evidence":
        ev = _evidence()
        del ev["lineage"]
        assert code_of(lambda: E.build_output(rules=ready_rules(TGT), target=TGT, model_free=_mf(), steps=[_step()],
                                              evidence=ev))[0] == code
        return
    assert code_of(lambda: _output(**kw))[0] == code


@pytest.mark.parametrize(("change", "code"), [
    ("forged_outcome", "PERMIT_RECOMPUTE"),        # 19b: 판정 불가를 통과로 고쳐 쓴 유효 JSON
    ("forged_cells", "PERMIT_RECOMPUTE"),          # 19b: 네 검사 불통과인데 네 칸만 참으로
    ("empty_expected", "PERMIT_EXPECTED_INCOMPLETE"),  # 19b: 빈 기대값
    ("partial_expected", "PERMIT_EXPECTED_INCOMPLETE"),  # 19b: 실행 · 출발점을 뺀 기대값
    ("no_file_check", "PERMIT_ARGS"),              # 19b: 파일 대조 없이
    ("rules_not_ready", "ATTEMPTS_UNSET"),         # 19b: 허용 시도 수 null
    ("bytes_changed", "PERMIT_SHA"),
    ("other_run", "PERMIT_MISMATCH"), ("changed_adapter", "PERMIT_MISMATCH"),
    ("missing_check_file", "PERMIT_MISMATCH"), ("other_rules", "PERMIT_MISMATCH"),
    ("purpose", "PERMIT_KIND"), ("split", "PERMIT_KIND"), ("unreadable", "PERMIT_FORMAT"),
    ("rules_sha256_passed", "PERMIT_EXPECTED_UNKNOWN"),   # 3판: 규칙은 rules 로 온다 — 대조값에 넘기면 거부
    ("processor_passed", "PERMIT_EXPECTED_UNKNOWN"),      # 3판: 프로세서는 착수 고정이 묶는다
])
def test_착수_허가의_음성(change, code) -> None:
    rules, obj, raw = _output()
    over = {}

    def forged(o):
        r = E.canonical(o)
        return r, {"expected_sha256": hashlib.sha256(r).hexdigest()}

    if change == "forged_outcome":
        _, o, _ = _output(steps=[_step(I)])
        raw, over = forged({**o, "decision": {**obj["decision"]}, "opens_four_cells": True})
    elif change == "forged_cells":
        _, o, _ = _output(model_free=_mf(X))
        raw, over = forged({**o, "opens_four_cells": True})
    elif change == "empty_expected":
        over = {"expected": {}}
    elif change == "partial_expected":
        over = {"expected": {k: v for k, v in EXPECT.items() if k not in ("check_run_id", "start_adapter_digest")}}
    elif change == "no_file_check":
        over = {"file_sha256_of": None}
    elif change == "rules_not_ready":
        b = body()
        b.update({"sequential_rule": "single", "max_attempts": None,
                  "limits": {"max_hours": 3, "max_cost_krw": 20000, "price_krw_per_hour": 2500, "price_source": "합성", "safety_factor": 1.2},
                  "target_list": {"sha256": TGT.file_sha256, "n": len(TGT)}})
        b["discrimination"] = synthetic_receipt(b)
        over = {"rules": E.load_rules(raw_of(b))}
    elif change == "bytes_changed":
        over = {"expected_sha256": hashlib.sha256(raw).hexdigest()}
        raw = raw.replace(b'"seed":1', b'"seed":2')
    elif change == "other_run":
        over = {"expected": {**EXPECT, "check_run_id": "run-2"}}
    elif change == "changed_adapter":
        over = {"expected": {**EXPECT, "start_adapter_digest": "w0-other"}}
    elif change == "missing_check_file":
        over = {"file_sha256_of": lambda p: None if p == "gold_echo.json" else _files(obj)(p)}
    elif change == "other_rules":
        over = {"rules": ready_rules(TGT, max_attempts=3)}
    elif change == "rules_sha256_passed":
        over = {"expected": {**EXPECT, "rules_sha256": rules.raw_sha256}}
    elif change == "processor_passed":
        over = {"expected": {**EXPECT, "processor_cfg_sha256": "pc"}}
    elif change in ("purpose", "split"):
        raw, over = forged({**obj, change: "frame_diag" if change == "purpose" else "val"})
    else:
        raw = raw[:-3] + b"x}\n"
        over = {"expected_sha256": hashlib.sha256(raw).hexdigest()}
    assert code_of(lambda: _verify(rules, raw, obj, **over))[0] == code


def test_묶음_검증기는_점검_목적의_곁_파일을_받지_않는다() -> None:
    from evaluation import bundle_v14 as B

    out: list = []
    B._check_sidecar_schema({"mode": "model", "purpose": E.KIND}, Path("x.sidecar.json"), out)
    assert any(r.code == B.RejectCode.SIDECAR_SCHEMA and r.detail == {"not_vocab": "purpose"} for r in out)


def test_본채점_진입점의_목적에_점검이_없다() -> None:
    """등록 모듈은 점검 목적을 받는다(분할 train — export 가 지나야 한다, 1013o 의 1). **본채점 진입점**의 choices · 탐색 목적 ·
    채점 거부는 그대로다 — 점검 산출은 채점 대상이 아니다."""
    from evaluation.entry_gate import PROBE_PURPOSES
    from evaluation.prereg_unified import PURPOSES, SPLIT_OF_PURPOSE, UnifiedRegistration, require_probe_scoring

    assert E.KIND in PURPOSES and SPLIT_OF_PURPOSE[E.KIND] == "train"
    assert E.KIND not in PROBE_PURPOSES
    src = (ROOT / "scripts/probe/score_unified.py").read_text(encoding="utf-8")
    assert 'choices=("main", "rehearsal", "frame_diag")' in src
    with pytest.raises(E.EarlyCheckRejected):
        E.refuse_for_scoring({"purpose": E.KIND})
    r = UnifiedRegistration()
    r.generation.purpose = E.KIND
    with pytest.raises(ValueError):
        require_probe_scoring(r, None)


def test_착수_허가의_대조값은_넷이고_프로세서는_묻지_않는다() -> None:
    assert E.PERMIT_EXPECTED_KEYS == ("start_adapter_digest", "pin_sha256", "check_run_id", "seed")
    rules, obj, raw = _output()
    assert _verify(rules, raw, obj)["evidence"]["processor_cfg_sha256"] == "pc"   # 증거에는 남고 대조값에는 없다


@pytest.mark.parametrize("seed", [0, "1", 1.0, None])
def test_증거의_seed_는_시드_번호다(seed) -> None:
    assert code_of(lambda: _output(seed=seed))[0] == "EVIDENCE_SEED"


def test_상한은_규칙_파일이_정본이고_실행_인자는_좁히기만_한다() -> None:
    rules = ready_rules()
    got = E.effective_limits(rules)
    assert (got["max_hours"], got["max_cost_krw"], got["cost_krw_of_hour"]) == (3.0, 20000.0, 3000.0)
    assert E.effective_limits(rules, max_hours=2, max_cost_krw=10000)["max_hours"] == 2.0
    assert code_of(lambda: E.effective_limits(rules, max_hours=4))[0] == "LIMITS_WIDENED"
    assert code_of(lambda: E.effective_limits(rules, max_cost_krw=30000))[0] == "LIMITS_WIDENED"
    real = E.load_rules(RAW)                                            # 4판 — 실물 규칙 파일은 준비됐다
    assert E.effective_limits(real)["max_hours"] == float(real.limits["max_hours"])


# ---------------------------------------------------------------- 4판 (2026-10-13) — §119 I-04 · I-05 · I-09

def test_판_4_의_규칙_파일은_시작할_수_있고_영수증_실물과_맞는다() -> None:
    """커밋된 규칙 파일 — 순차 규칙 · 시도 수 · 상한이 차 있고, 판별력 영수증이 실물로 있어 입력이 같다."""
    rules = E.load_rules(RAW)
    assert E.RULES_VERSION == "4" and rules.sequential_rule in E.SEQUENTIAL_RULES and rules.max_attempts == 1
    assert rules.limits["max_hours"] > 0 and rules.limits["max_cost_krw"] > 0
    E.require_ready(rules)
    summary = E.verify_discrimination(rules)
    assert summary and all(v["positive_runs"] > 0 and v["negative_runs"] > 0 for v in summary.values())


@pytest.mark.parametrize("case", ["missing", "sha", "inputs", "reps", "no_rule"])
def test_판별력_영수증은_실물을_본다(case) -> None:
    b = body()
    b.update({"sequential_rule": "single", "max_attempts": 1,
              "limits": {"max_hours": 3, "max_cost_krw": 20000, "price_krw_per_hour": 2500, "price_source": "합성", "safety_factor": 1.2}})
    if case == "missing":
        b["discrimination"] = {"path": str(_SYNTH_DIR / "없는.json"), "sha256": "0" * 64}
        want = "DISCRIMINATION_MISSING"
    elif case == "sha":
        d = synthetic_receipt(b)
        b["discrimination"] = {"path": d["path"], "sha256": "1" * 64}
        want = "DISCRIMINATION_SHA"
    elif case == "inputs":
        other = dict(b)
        other["steps"] = [5, 6]
        b["discrimination"] = synthetic_receipt(other)
        want = "DISCRIMINATION_INPUTS"
    elif case == "reps":
        b["discrimination"] = synthetic_receipt(b, reps=0)
        want = "DISCRIMINATION_EMPTY"
    else:
        cell = {"false_pass": 0, "false_fail": 0, "positive_runs": 5, "negative_runs": 10}
        b["discrimination"] = synthetic_receipt(b, tally={"s": {"two_consecutive": cell}})
        want = "DISCRIMINATION_EMPTY"
    assert code_of(lambda: E.require_ready(E.load_rules(raw_of(b))))[0] == want


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 0, True])
def test_상한_실행_인자는_유한한_양수다(bad) -> None:
    rules = ready_rules()
    assert code_of(lambda: E.effective_limits(rules, max_hours=bad))[0] == "LIMITS_INVALID"
    assert code_of(lambda: E.effective_limits(rules, max_cost_krw=bad))[0] == "LIMITS_INVALID"
    assert E.effective_limits(rules, max_hours=1.0)["max_hours"] == 1.0


def test_네_검사의_산출_파일_지문은_빈_칸이면_열리지_않는다() -> None:
    """기대값과 실제값이 함께 비어(None == None) 통과하던 길(§119 I-04) — 만드는 쪽 · 읽는 쪽 둘 다 닫는다."""
    with pytest.raises(E.EarlyCheckRejected) as e:
        E._check_model_free_files({n: {} for n in E.MODEL_FREE_CHECKS})
    assert e.value.code == "EVIDENCE_MISSING"
    with pytest.raises(E.EarlyCheckRejected):
        E._check_model_free_files({n: {"path": f"{n}.json", "sha256": f"{n}-sha"} for n in E.MODEL_FREE_CHECKS})
    ok = {n: {"path": f"{n}.json", "sha256": hashlib.sha256(n.encode()).hexdigest()} for n in E.MODEL_FREE_CHECKS}
    E._check_model_free_files(ok)

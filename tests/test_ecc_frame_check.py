"""본실험 본채점의 프레임 표지(`evaluation/ecc_frame_check.py`) · 등록 계약 — 목적 main 의 `canary.frame_diag_sha256` 은
조기 좌표 점검 허가 파일의 sha256 이고(의사결정로그 10-07), null 이면 등록이 서지 않는다."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import early_coord_check as E
from evaluation import ecc_frame_check as FC
from evaluation.prereg_unified import MAIN_FRAME_DIAG_MEANING, UnifiedRegistration
from tests import test_early_coord_check as TEC


def _permit(rules=None, *, step=E.PASS, edit=None) -> tuple[E.EarlyRules, bytes]:
    rules = rules or TEC.ready_rules(TEC.TGT)
    obj = E.build_output(rules=rules, target=TEC.TGT, model_free=TEC._mf(), steps=[TEC._step(status=step)],
                         evidence=TEC._evidence())
    if edit:
        edit(obj)
    return rules, E.canonical(obj)


def _check(rules, raw, registered=None):
    return FC.frame_check_by_permit(raw, registered_sha256=registered or hashlib.sha256(raw).hexdigest(), rules=rules)


def test_통과_파일이면_표지와_허가_sha_를_낸다() -> None:
    rules, raw = _permit()
    out = _check(rules, raw)
    assert (out["status"], out["ecc_permit_sha256"], out["rules_sha256"]) == (
        FC.FRAME_CHECKED_BY_ECC, hashlib.sha256(raw).hexdigest(), rules.raw_sha256)
    assert (out["attempt"], out["seed"], out["pin_sha256"]) == (1, 1, "pin")


@pytest.mark.parametrize(("make", "code"), [
    (lambda: (*_permit(), "0" * 64), "FRAME_PERMIT_HASH"),
    (lambda: _permit(edit=lambda o: o.update(kind="frame_diag")), "FRAME_PERMIT_KIND"),
    (lambda: _permit(edit=lambda o: o.update(split="val")), "FRAME_PERMIT_KIND"),
    (lambda: (TEC.ready_rules(TEC.TGT, max_attempts=3), _permit()[1]), "FRAME_PERMIT_RULES"),
    (lambda: _permit(edit=lambda o: o["decision"].update(step_index=9)), "FRAME_PERMIT_RECOMPUTE"),
    (lambda: _permit(step=E.FAIL), "FRAME_PERMIT_NOT_PASSED"),
    (lambda: (TEC.ready_rules(TEC.TGT), b"{not json"), "FRAME_PERMIT_FORMAT"),
])
def test_등록과_다르거나_통과가_아닌_허가_파일은_거부한다(make, code) -> None:
    got = make()
    rules, raw = got[0], got[1]
    registered = got[2] if len(got) > 2 else None
    with pytest.raises(E.EarlyCheckRejected) as exc:
        _check(rules, raw, registered)
    assert exc.value.code == code


def test_목적_main_에서_칸의_뜻은_허가_sha_이고_null_과_틀린_꼴은_그대로_거부한다() -> None:
    assert MAIN_FRAME_DIAG_MEANING == "ecc_permit_sha256"
    r = UnifiedRegistration()
    r.generation.purpose = "main"
    assert "canary.frame_diag_sha256" in r.missing(side="scoring")                  # null 은 '진단 보류' 가 아니다
    r.canary.frame_diag_sha256 = "x" * 64
    assert any(p.startswith("canary.frame_diag_sha256") for p in r.invalid(side="scoring"))
    r.canary.frame_diag_sha256 = hashlib.sha256(_permit()[1]).hexdigest()
    assert "canary.frame_diag_sha256" not in r.missing(side="scoring")
    assert not any(p.startswith("canary.frame_diag_sha256") for p in r.invalid(side="scoring"))
    assert json.loads(json.dumps(r.scoring_payload()))["canary"]["frame_diag_sha256"] == r.canary.frame_diag_sha256


# ---------------------------------------------------------------- 허가 면제 — 모델 없는 통과 산출 + 면제 파일의 묶음(의사결정로그 2026-10-17)
SECTION = "ECC 모델 점검 허가를 기다리지 않고 다섯 칸을 연다 — 사용자 결정(2026-10-17 06:0x)"
LOG = f"# 의사결정로그\n\n**{SECTION}** 본문.\n"


def _admission(rules, edit=None) -> bytes:
    obj = {"checks": [{"name": n, "status": E.PASS, "n": 5, "n_bad": 0, "n_ids": 5} for n in E.MODEL_FREE_CHECKS],
           "passed": True, "rules_sha256": rules.raw_sha256, "target_list_sha256": rules.target_sha256, "gpu_calls": 0}
    if edit:
        edit(obj)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def _waiver(adm: bytes, edit=None) -> bytes:
    obj = {"decision": "모델 점검 허가를 기다리지 않고 다섯 칸을 연다", "decision_log_section": SECTION,
           "model_free_admission_sha256": hashlib.sha256(adm).hexdigest(), "step100_geometry_sha256": None,
           "decided_at": "2026-10-17T06:05:00+09:00"}
    if edit:
        edit(obj)
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")


def _mf_check(rules, adm, wv, *, registered=None, log=LOG):
    return FC.frame_check_by_model_free(adm, wv, registered_sha256=registered or FC.model_free_bundle_sha256(adm, wv),
                                        rules=rules, decision_log=log)


def test_허가_면제면_두_파일의_묶음을_보고_모델_없는_검사만이라고_표지한다() -> None:
    rules = TEC.ready_rules(TEC.TGT)
    adm = _admission(rules)
    wv = _waiver(adm)
    out = _mf_check(rules, adm, wv)
    assert out["status"] == FC.FRAME_CHECKED_MODEL_FREE_ONLY == "FRAME_CHECKED_MODEL_FREE_ONLY"
    assert (out["admission_sha256"], out["waiver_sha256"]) == (hashlib.sha256(adm).hexdigest(), hashlib.sha256(wv).hexdigest())
    assert out["frame_diag_sha256"] == FC.model_free_bundle_sha256(adm, wv) and "미세조정" in out["scope"]
    body = {"kind": FC.BUNDLE_KIND, "admission_sha256": out["admission_sha256"], "waiver_sha256": out["waiver_sha256"]}
    assert out["frame_diag_sha256"] == hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert FC.model_free_bundle_sha256(adm + b" ", wv) != out["frame_diag_sha256"] != FC.model_free_bundle_sha256(adm, wv + b" ")


def _bad_admission(edit):
    def make(rules):
        adm = _admission(rules, edit)
        return adm, _waiver(adm)
    return make


def _bad_waiver(edit):
    def make(rules):
        adm = _admission(rules)
        return adm, _waiver(adm, edit)
    return make


@pytest.mark.parametrize(("make", "kw", "code"), [
    (lambda r: (_admission(r), _waiver(_admission(r))), {"registered": "0" * 64}, "FRAME_MODEL_FREE_HASH"),
    (lambda r: (b"{nope", b"{}"), {}, "FRAME_MODEL_FREE_FORMAT"),
    (_bad_admission(lambda o: o["checks"][3].update(status=E.FAIL)), {}, "FRAME_MODEL_FREE_NOT_PASSED"),
    (_bad_admission(lambda o: o.update(passed=False)), {}, "FRAME_MODEL_FREE_NOT_PASSED"),
    (_bad_admission(lambda o: o["checks"][2].update(n_bad=1)), {}, "FRAME_MODEL_FREE_NOT_PASSED"),
    (_bad_admission(lambda o: o.update(rules_sha256="1" * 64)), {}, "FRAME_MODEL_FREE_RULES"),
    (_bad_waiver(lambda o: o.pop("decided_at")), {}, "FRAME_WAIVER_FORM"),
    (_bad_waiver(lambda o: o.update(extra=1)), {}, "FRAME_WAIVER_FORM"),
    (_bad_waiver(lambda o: o.update(decision=" ")), {}, "FRAME_WAIVER_FORM"),
    (_bad_waiver(lambda o: o.update(decided_at="어제")), {}, "FRAME_WAIVER_FORM"),
    (_bad_waiver(lambda o: o.update(model_free_admission_sha256="2" * 64)), {}, "FRAME_WAIVER_BINDING"),
    (lambda r: (_admission(r), _waiver(_admission(r))), {"log": "# 의사결정로그\n\n다른 결정만 있다\n"}, "FRAME_WAIVER_DECISION"),
])
def test_허가_면제의_파일이_등록_규칙_결정과_다르면_거부한다(make, kw, code) -> None:
    rules = TEC.ready_rules(TEC.TGT)
    adm, wv = make(rules)
    with pytest.raises(E.EarlyCheckRejected) as exc:
        _mf_check(rules, adm, wv, **kw)
    assert exc.value.code == code


def test_면제_파일의_칸은_학습_쪽과_같고_뜻_상수가_있다() -> None:
    from evaluation.prereg_unified import MAIN_FRAME_DIAG_MODEL_FREE

    assert MAIN_FRAME_DIAG_MODEL_FREE == "model_free_admission_waiver_bundle_sha256"
    ecc = pytest.importorskip("vlm.ecc")
    if not hasattr(ecc, "WAIVER_FIELDS"):
        pytest.skip("학습 쪽 면제 칸이 이 트리에 아직 없다")
    assert tuple(ecc.WAIVER_FIELDS) == FC.WAIVER_FIELDS

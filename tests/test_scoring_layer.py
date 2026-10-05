"""통합형 채점 층 — 진입점 미니스펙 3판 1-2 나 · 채점 층(M-5) · 1-1-다 · 2 절 · 4-2.

지키는 것.

1. 채점 층의 함수는 `VerifiedBundle` 이 아닌 것을 받으면 `TypeError` — 목록(`SCORING_LAYER`)의 **모든** 함수가, **첫 문장에서**.
2. 에코 채점은 FP = FN = 0 · TP = 정답 박스 수 · 좌표 최대 차 ≤ 허용일 때만 통과하고 다섯 수를 싣는다.
3. 문자 일치는 생성문을 다시 읽고 독립 역변환으로 좌표를 다시 내 맞댄다 — 좌표가 다르면 `literal_match_failed`,
   유지 · 폐기 판정이 다르면 `parser_disagreement`.
4. ③ 은 실패율이 **기록되면** 통과(0 이 아니어도), ⑤ 는 지연 · 길이 요약이 있으면 통과이고 한도 도달 5 % 초과는 경보다.
5. 적대적 한 장 셋이 실제로 거부된다.

묶음은 `tests/test_bundle_v14.py` 의 정상 묶음 도우미로 만든다 — 검증기를 실제로 지나 `VerifiedBundle` 을 얻는다.
"""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest

from evaluation import scoring_layer as L
from evaluation.bundle_v14 import VerifiedBundle, require_verified
from tests.test_bundle_v14 import IDS, make, row

KNOWN = ["100", "2011", "2012", "301", "401", "402"]
CLASSES = ["100", "2011", "301", "401"]
RULE = "decoupled_v2"
BOX = [10.0, 20.0, 30.0, 40.0]


def gen_row(image_id: str, *, boxes=((("2011"), BOX),), text_boxes=None, latency=800.0, n_new=12,
            gen_stop="eos", parse_error=None, dropped=0, **over) -> dict:
    """모델 · 에코 줄 하나 — 생성문(`text`)과 파싱 · 역변환 결과(`bbox_px_parsed`)가 짝이다(ABS_ORIG 는 항등)."""
    tb = boxes if text_boxes is None else text_boxes
    text = json.dumps({"defects": [{"iso_code": c, "bbox_2d": list(b)} for c, b in tb],
                       "verdict": "불합격", "cited_clauses": ["C-1"]}, ensure_ascii=False)
    parsed = None if parse_error else {"defects": [{"iso_code": c, "bbox_px": list(b)} for c, b in boxes],
                                       "verdict": "불합격", "cited_clauses": ["C-1"]}
    r = row(image_id, text=text, bbox_px_parsed=parsed, parse_error=parse_error, n_new_tokens=n_new,
            latency_ms=latency, n_bad_items_dropped=dropped, raw_output_ref=None, gen_stop=gen_stop)
    r.update(over)
    return r


def verified(tmp_path: Path, rows, *, echo=False, meta_over=None, **kw) -> VerifiedBundle:
    """리허설은 배치 1 이다(쓰는 쪽 2판) — 도우미의 기본 배치 크기(8)를 1 로 둔다. 배치 시험은 덮어쓴다."""
    return make(tmp_path, rows=rows, echo=echo, meta_over={"batch_size": 1, **(meta_over or {})}, **kw).verify()


def gold_of(ids, box=BOX):
    return {i: [("2011", list(box))] for i in ids}


# ---------------------------------------------------------------- 1 채점 층의 규칙

def test_require_verified_는_검증을_지난_묶음만_받는다(tmp_path) -> None:
    vb = verified(tmp_path, [gen_row(i) for i in IDS])
    assert require_verified(vb) is vb
    for bad in ({}, None, "path.jsonl", [vb]):
        with pytest.raises(TypeError):
            require_verified(bad)


KWARGS = {
    "records_from_verified": {"coupling_rule": RULE, "known_iso_codes": KNOWN, "scoring_iso_codes": CLASSES},
    "score_echo": {"gold": {}, "classes": CLASSES, "max_coord_diff": 0.0, "coupling_rule": RULE,
                   "known_iso_codes": KNOWN},
    "check_literal": {"known_iso_codes": KNOWN, "scoring_iso_codes": CLASSES, "tolerance": 0.0005},
    "judge_rehearsal_gates": {"coupling_rule": RULE, "known_iso_codes": KNOWN, "scoring_iso_codes": CLASSES},
    "adversarial_probe": {"coupling_rule": RULE, "known_iso_codes": KNOWN, "scoring_iso_codes": CLASSES},
    "rehearsal_metric_smoke": {"gold": {}, "classes": CLASSES, "coupling_rule": RULE, "known_iso_codes": KNOWN},
}


@pytest.mark.parametrize("name", L.SCORING_LAYER)
def test_채점_층의_함수는_검증을_지나지_않은_것을_받지_않는다(name: str) -> None:
    fn = getattr(L, name)
    bad = [{"lines": []}] if name == "judge_rehearsal_gates" else {"lines": []}
    with pytest.raises(TypeError, match="VerifiedBundle"):
        fn(bad, **KWARGS[name])


@pytest.mark.parametrize("name", L.SCORING_LAYER)
def test_채점_층의_함수는_첫_문장에서_require_verified_를_부른다(name: str) -> None:
    """독스트링 다음 **첫 실행 문장** 안에 호출이 있다 — 늦게 부르면 그 사이에 검증 안 된 값이 쓰인다."""
    src = inspect.getsource(getattr(L, name))
    fn = ast.parse(src).body[0]
    body = [s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    body = [s for s in body if not isinstance(s, ast.ImportFrom)]
    calls = {n.func.id for n in ast.walk(body[0]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "require_verified" in calls, name


def test_채점_층의_목록에_없는_함수는_그물_이름을_부르지_않는다() -> None:
    """모듈 안에서 `build_counts` · `adapt_unified_main` 을 부르는 함수는 목록 안에만 있다."""
    tree = ast.parse(Path(L.__file__).read_text(encoding="utf-8"))
    net = {"build_counts", "adapt_unified_main", "score_records", "score_m1"}
    for fn in tree.body:
        if isinstance(fn, ast.FunctionDef) and not fn.name.startswith("_"):
            used = {n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
            if used & net:
                assert fn.name in L.SCORING_LAYER, fn.name


# ---------------------------------------------------------------- 2 에코 채점

def test_에코가_정답과_같으면_통과하고_다섯_수를_싣는다(tmp_path) -> None:
    vb = verified(tmp_path, [gen_row(i) for i in IDS], echo=True)
    out = L.score_echo(vb, **KWARGS["score_echo"] | {"gold": gold_of(IDS)})
    assert out.as_dict() | {"reasons": []} == {"status": "pass", "n_expected": 3, "tp": 3, "fp": 0, "fn": 0,
                                               "max_coord_diff": 0.0, "reasons": []}


def test_좌표가_허용보다_어긋나면_불통과(tmp_path) -> None:
    shifted = [10.0, 20.0, 30.0, 41.0]
    vb = verified(tmp_path, [gen_row(i, boxes=(("2011", shifted),)) for i in IDS], echo=True)
    out = L.score_echo(vb, **KWARGS["score_echo"] | {"gold": gold_of(IDS)})
    assert out.status == "fail" and out.max_coord_diff == 1.0 and out.tp == 3


def test_남는_예측이_있으면_불통과(tmp_path) -> None:
    rows = [gen_row(i, boxes=(("2011", BOX), ("401", [50, 50, 60, 60]))) for i in IDS]
    vb = verified(tmp_path, rows, echo=True)
    out = L.score_echo(vb, **KWARGS["score_echo"] | {"gold": gold_of(IDS)})
    assert out.status == "fail" and out.fp == 3


def test_정답_뷰에_없는_에코_id_는_멈춘다(tmp_path) -> None:
    vb = verified(tmp_path, [gen_row(i) for i in IDS], echo=True)
    with pytest.raises(ValueError, match="정답 뷰에 없는"):
        L.score_echo(vb, **KWARGS["score_echo"] | {"gold": gold_of(IDS[:2])})


def test_에코_채점은_모델_묶음을_받지_않는다(tmp_path) -> None:
    vb = verified(tmp_path, [gen_row(i) for i in IDS])
    with pytest.raises(ValueError, match="에코 묶음만"):
        L.score_echo(vb, **KWARGS["score_echo"] | {"gold": gold_of(IDS)})


# ---------------------------------------------------------------- 3 문자 일치

def test_생성문과_좌표가_같으면_통과(tmp_path) -> None:
    vb = verified(tmp_path, [gen_row(i) for i in IDS])
    out = L.check_literal(vb, **KWARGS["check_literal"])
    assert out.status == "pass" and out.n_checked == 3


def test_좌표가_생성문과_다르면_문자_일치_실패(tmp_path) -> None:
    rows = [gen_row(IDS[0], boxes=(("2011", [10.0, 20.0, 30.0, 40.01]),), text_boxes=(("2011", BOX),))]
    rows += [gen_row(i) for i in IDS[1:]]
    out = L.check_literal(verified(tmp_path, rows), **KWARGS["check_literal"])
    assert out.as_dict()["mismatches"] == [{"line": 1, "code": "literal_match_failed"}]


def test_유지_판정이_다르면_파서_불일치(tmp_path) -> None:
    rows = [gen_row(IDS[0], text_boxes=(("2011", BOX), ("401", [1, 1, 5, 5])))] + [gen_row(i) for i in IDS[1:]]
    out = L.check_literal(verified(tmp_path, rows), **KWARGS["check_literal"])
    assert out.as_dict()["mismatches"] == [{"line": 1, "code": "parser_disagreement"}]


def test_폐기_수만_달라도_파서_불일치(tmp_path) -> None:
    """유지 항목이 같아도 상류가 버렸다고 적은 수가 참조 파서와 다르면 두 파서의 판정이 갈린 것이다."""
    rows = [gen_row(IDS[0], dropped=1)] + [gen_row(i) for i in IDS[1:]]
    out = L.check_literal(verified(tmp_path, rows), **KWARGS["check_literal"])
    assert out.as_dict()["mismatches"] == [{"line": 1, "code": "parser_disagreement"}]


def test_레코드_실패_줄은_건너뛰고_센다(tmp_path) -> None:
    rows = [gen_row(IDS[0], parse_error="no_json", text="not json")] + [gen_row(i) for i in IDS[1:]]
    out = L.check_literal(verified(tmp_path, rows), **KWARGS["check_literal"])
    assert (out.n_checked, out.n_skipped_record_fail, out.status) == (2, 1, "pass")


# ---------------------------------------------------------------- 4 ③ · ⑤

def test_실패율이_0_이_아니어도_기록되면_통과(tmp_path) -> None:
    rows = [gen_row(IDS[0], parse_error="no_json", text="x")] + [gen_row(i) for i in IDS[1:]]
    g = L.judge_rehearsal_gates([verified(tmp_path, rows)], **KWARGS["judge_rehearsal_gates"])
    assert g["③"].status == "pass"
    s = g["③"].summary
    assert s["parse_fail_rate"] == pytest.approx(1 / 3)
    assert s["combined_violation_rate"] == pytest.approx(s["parse_fail_rate"] + s["clause_error_rate"])


def test_게이트3_의_검사는_다섯이고_3_5_가_어긋나면_불통과(tmp_path, monkeypatch) -> None:
    """검토(10-02 병합 뒤) 4 — ③-5(합산 위반율 = 파싱 실패율 + 조항 오류율의 재계산)를 지우면 떨어지게.
    실제 입력에서는 두 값이 같은 셈에서 나와 어긋날 수 없다 — 그래서 비교를 거짓으로 바꿔 판정이 그 검사에 묶였는지 본다."""
    rows = [gen_row(i) for i in IDS]
    g = L.judge_rehearsal_gates([verified(tmp_path, rows)], **KWARGS["judge_rehearsal_gates"])
    assert set(g["③"].checks) == {"③-1", "③-2", "③-3", "③-4", "③-5"}
    assert g["③"].checks["③-5"]["pass"] is True and g["③"].status == "pass"
    monkeypatch.setattr(L.math, "isclose", lambda *a, **k: False)
    g = L.judge_rehearsal_gates([verified(tmp_path / "b", rows)], **KWARGS["judge_rehearsal_gates"])
    assert g["③"].checks["③-5"]["pass"] is False and g["③"].status == "fail"


def test_지연과_길이의_요약(tmp_path) -> None:
    rows = [gen_row(i, latency=100.0 * (k + 1), n_new=10 + k) for k, i in enumerate(IDS)]
    g = L.judge_rehearsal_gates([verified(tmp_path, rows)], **KWARGS["judge_rehearsal_gates"])
    s = g["⑤"].summary
    assert g["⑤"].status == "pass"
    assert s["latency_ms"]["n"] == 3 and s["latency_ms"]["max"] == 300.0 and s["n_new_tokens"]["min"] == 10
    assert s["gen_stop_counts"] == {"eos": 3} and s["length_ratio"] == 0.0


def test_에코_묶음은_세지_않는다(tmp_path) -> None:
    model = verified(tmp_path / "m", [gen_row(i) for i in IDS])
    echo = verified(tmp_path / "e", [gen_row(i, latency=1.0) for i in IDS], echo=True)
    g = L.judge_rehearsal_gates([model, echo], **KWARGS["judge_rehearsal_gates"])
    assert g["⑤"].summary["latency_ms"]["n"] == 3


def test_한도_도달이_5_퍼센트를_넘으면_경보다(tmp_path) -> None:
    rows = [gen_row(IDS[0], gen_stop="length")] + [gen_row(i) for i in IDS[1:]]
    g = L.judge_rehearsal_gates([verified(tmp_path, rows)], **KWARGS["judge_rehearsal_gates"])
    assert g["⑤"].status == "pass" and g["⑤"].alerts


def test_배치가_1_보다_크면_장당의_정의를_요구한다(tmp_path) -> None:
    vb = verified(tmp_path, [gen_row(i) for i in IDS], meta_over={"batch_size": 4})
    g = L.judge_rehearsal_gates([vb], **KWARGS["judge_rehearsal_gates"])
    assert g["⑤"].checks["⑤-4"]["pass"] is False
    vb2 = verified(tmp_path / "b", [gen_row(i) for i in IDS], meta_over={"batch_size": 4, "latency_basis": "per_call"})
    assert L.judge_rehearsal_gates([vb2], **KWARGS["judge_rehearsal_gates"])["⑤"].checks["⑤-4"]["pass"]


def test_지연이_빠진_줄이_있으면_불통과(tmp_path) -> None:
    rows = [gen_row(IDS[0], latency_ms=None)] + [gen_row(i) for i in IDS[1:]]
    g = L.judge_rehearsal_gates([verified(tmp_path, rows)], **KWARGS["judge_rehearsal_gates"])
    assert g["⑤"].checks["⑤-1"]["pass"] is False


def test_생성_길이가_빠진_줄은_스키마_검사가_먼저_잡는다(tmp_path) -> None:
    """`n_new_tokens` 는 1.4 의 필수 필드다 — 그 줄은 어댑터가 배관 고장으로 멈추고 ③-1 이 불통과다."""
    rows = [gen_row(IDS[0], n_new_tokens=None)] + [gen_row(i) for i in IDS[1:]]
    g = L.judge_rehearsal_gates([verified(tmp_path, rows)], **KWARGS["judge_rehearsal_gates"])
    assert g["③"].checks["③-1"]["pass"] is False


# ---------------------------------------------------------------- 4′ 리허설의 지표 — 값 없음

def _numbers(obj) -> list:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [obj]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in _numbers(v)]
    if isinstance(obj, (list, tuple)):
        return [n for v in obj for n in _numbers(v)]
    return []


def test_리허설의_지표는_끝났는가와_키만_내고_값은_싣지_않는다(tmp_path) -> None:
    vb = verified(tmp_path, [gen_row(i) for i in IDS])
    out = L.rehearsal_metric_smoke(vb, **KWARGS["rehearsal_metric_smoke"] | {"gold": gold_of(IDS)})
    assert out["completed"] is True and "macro_f1" in out["keys"]
    assert _numbers(out) == [], "리허설 산출에 수가 실리면 그 수로 주 지표를 고르게 된다(4-2)"


def test_리허설의_지표_계산이_깨지면_끝나지_않았다고_적는다(tmp_path) -> None:
    vb = verified(tmp_path, [gen_row(i) for i in IDS])
    out = L.rehearsal_metric_smoke(vb, **KWARGS["rehearsal_metric_smoke"] | {"gold": gold_of(IDS[:1])})
    assert out["completed"] is False and out["error"] == "KeyError"


# ---------------------------------------------------------------- 5 적대적 한 장

def test_적대적_한_장_셋이_실제로_거부된다(tmp_path) -> None:
    vb = verified(tmp_path, [gen_row(i) for i in IDS])
    out = L.adversarial_probe(vb, **KWARGS["adversarial_probe"])
    assert out["status"] == "pass"
    assert out["fired"] == {"missing_required_field": True, "parse_error_out_of_vocab": True,
                            "coord_space_mismatch": True}


def test_검사가_안_걸리면_적대적_한_장이_그렇게_적는다(tmp_path, monkeypatch) -> None:
    """탐침이 검사의 발화를 **지어내지 않는다** — 검사를 끈 채로 돌리면 불통과로 나와야 한다."""
    import evaluation.bundle_v14 as B

    vb = verified(tmp_path, [gen_row(i) for i in IDS])
    monkeypatch.setattr(B, "_check_line_env", lambda *a, **k: None)
    monkeypatch.setattr(L, "adapt_unified_main", lambda *a, **k: None)
    out = L.adversarial_probe(vb, **KWARGS["adversarial_probe"])
    assert out["status"] == "fail"
    assert out["fired"] == {"missing_required_field": False, "parse_error_out_of_vocab": False,
                            "coord_space_mismatch": False}

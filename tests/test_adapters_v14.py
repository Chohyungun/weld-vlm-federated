"""1.4 어댑터 — 07번 미니스펙 §13-3 · §14-2.

여기서 가르는 것은 하나다. **모델의 실패와 배관의 고장은 다른 사실이다.**

상류가 정해진 필드를 정해진 자료형으로 싣지 않은 것은 모델이 못한 일이 아니다. 그것을 레코드
실패로 적으면 배관 고장이 모델 점수로 기록되고, 그 점수는 다시 칸 비교에 들어간다. 그래서
배관 고장은 세지 않고 **예외로 멈춘다.** 사유와 파싱 결과가 서로 맞지 않는 줄(짝 규칙 위반)도
배관 고장이다 — 앞 판은 그것을 어댑터의 레코드 실패로 셌고 이 파일의 시험이 그 거동을 고정했다.

결합 규칙도 같은 이유로 기본값이 없다. 판정·인용이 깨져도 박스는 채점한다는 것이 `decoupled_v2`
인데, 그것이 기본값이면 "정하지 않고 돌린 것"과 "정해서 돌린 것"이 구분되지 않는다.
"""

from __future__ import annotations

import json

import pytest

from data.label_map import load_label_map
from evaluation.adapters_v14 import (
    COUPLING_RULE_V2,
    STOP_UNDETERMINED,
    AdaptReportV14,
    UpstreamContractError,
    adapt_unified_main,
    decode_line,
)
from evaluation.params import CLASS_NAMES

_LM = load_label_map()
SCORING = [_LM.iso_code(n) for n in CLASS_NAMES]
"""채점 4클래스 — 채점기(`evaluation.cells`)와 같은 규칙. 앞 판은 사상표 코드를 정렬해 앞 넷을 잘랐다."""
CODE = SCORING[0]
KNOWN = _LM.iso_codes(include_alt=True)
TAG = "uni_local_C1"
SEED = 20260828


def line(**over) -> str:
    """계약을 지키는 줄 하나. 시험이 고치고 싶은 것만 덮어쓴다."""
    row = {
        "image_id": "aihub000001",
        "gen_stop": "eos",
        "n_new_tokens": 42,
        "coord_space": "ABS_ORIG",
        "coord_cfg_hash": "c0ffee",
        "latency_ms": 1200.0,
        "raw_output_ref": "gen.jsonl:1",
        "bbox_px_parsed": {
            "defects": [{"iso_code": CODE, "bbox_px": [10.0, 20.0, 40.0, 60.0]}],
            "verdict": "불합격",
            "cited_clauses": ["IACS-47-3.1"],
        },
    }
    row.update(over)
    return json.dumps(row, ensure_ascii=False)


def failed(reason: str, **over) -> str:
    """상류가 레코드 실패로 적은 줄 — 짝 규칙대로 파싱 결과가 null 이다."""
    return line(parse_error=reason, bbox_px_parsed=None, **over)


def run(*lines: str, **kw) -> AdaptReportV14:
    kw.setdefault("tag", TAG)
    kw.setdefault("seed", SEED)
    kw.setdefault("coupling_rule", COUPLING_RULE_V2)
    kw.setdefault("known_iso_codes", KNOWN)
    kw.setdefault("scoring_iso_codes", SCORING)
    return adapt_unified_main(lines, **kw)


# ---------------------------------------------------------------- 결합 규칙

def test_결합_규칙에_기본값이_없다() -> None:
    with pytest.raises(TypeError):
        adapt_unified_main([], tag=TAG, seed=SEED, known_iso_codes=KNOWN,
                           scoring_iso_codes=SCORING)  # type: ignore[call-arg]


def test_다른_규칙으로는_1_4_레코드를_만들지_않는다() -> None:
    with pytest.raises(ValueError, match="decoupled_v2"):
        run(line(), coupling_rule="coupled_v1")


def test_규칙_id_가_보고에_그대로_실린다() -> None:
    assert run(line()).as_dict()["coupling_rule"] == COUPLING_RULE_V2


# ---------------------------------------------------------------- 배관 고장

@pytest.mark.parametrize("raw", ["", "   ", "\t"])
def test_빈_줄은_멈춘다(raw: str) -> None:
    """줄 수가 계약의 일부다 — 조용히 건너뛰면 줄 수 검사가 무의미해진다."""
    with pytest.raises(UpstreamContractError, match="빈 줄"):
        run(raw)


def test_JSON_이_아닌_줄은_멈춘다() -> None:
    with pytest.raises(UpstreamContractError, match="JSON 이 아니다"):
        run("{ 망가진")


def test_객체가_아닌_줄은_멈춘다() -> None:
    with pytest.raises(UpstreamContractError, match="객체가 아니다"):
        run("[1, 2]")


@pytest.mark.parametrize("bad", [None, "", 3])
def test_image_id_가_없으면_멈춘다(bad) -> None:
    with pytest.raises(UpstreamContractError, match="image_id"):
        run(line(image_id=bad))


@pytest.mark.parametrize("bad", ["stopped", None, "EOS"])
def test_계약_밖_gen_stop_은_멈춘다(bad) -> None:
    with pytest.raises(UpstreamContractError, match="gen_stop"):
        run(line(gen_stop=bad))


@pytest.mark.parametrize("bad", [-1, 1.0, "42", None, True])
def test_n_new_tokens_가_0_이상의_정수가_아니면_멈춘다(bad) -> None:
    """`True` 도 막는다 — 참/거짓이 토큰 수 자리에 들어오면 값이 1 로 읽힌다."""
    with pytest.raises(UpstreamContractError, match="n_new_tokens"):
        run(line(n_new_tokens=bad))


@pytest.mark.parametrize("bad", [-1, 1.5, "2", True])
def test_폐기_수가_0_이상의_정수가_아니면_멈춘다(bad) -> None:
    with pytest.raises(UpstreamContractError, match="n_bad_items_dropped"):
        run(line(n_bad_items_dropped=bad))


def test_계약_밖_parse_error_는_멈춘다() -> None:
    with pytest.raises(UpstreamContractError, match="parse_error"):
        run(line(parse_error="unknown_iso_code"))


def test_멈출_때_어느_줄인지_말한다() -> None:
    with pytest.raises(UpstreamContractError, match=rf"{TAG}:2"):
        run(line(), "{ 망가진")


# ---------------------------------------------------------------- 결합 해제

def test_판정이_깨져도_박스를_채점한다() -> None:
    rep = run(line(bbox_px_parsed={"defects": [{"iso_code": CODE, "bbox_px": [1, 2, 3, 4]}],
                                   "verdict": "아마도", "cited_clauses": ["A"]}))
    (rec,) = rep.records
    assert rec.parse_ok is True, "판정이 깨졌다고 레코드를 버리지 않는다"
    assert rec.verdict is None
    assert rec.clause_error == ["verdict_invalid"]
    assert len(rec.defects) == 1
    assert rep.n_record_failures == 0


def test_인용이_깨져도_박스를_채점한다() -> None:
    rep = run(line(bbox_px_parsed={"defects": [], "verdict": "합격", "cited_clauses": "A"}))
    (rec,) = rep.records
    assert rec.parse_ok is True
    assert rec.cited_clauses == []
    assert rec.clause_error == ["citations_invalid"]


def test_둘_다_깨지면_둘_다_적힌다() -> None:
    rep = run(line(bbox_px_parsed={"defects": []}))
    assert rep.records[0].clause_error == ["citations_invalid", "verdict_invalid"]
    assert rep.clause_errors == {"citations_invalid": 1, "verdict_invalid": 1}


def test_조항_위반과_레코드_실패는_서로소다() -> None:
    """같은 줄을 두 번 세면 합산 위반율이 부풀려진다."""
    rep = run(failed("truncated"), line(bbox_px_parsed={"defects": []}))
    assert rep.n_record_failures == 1
    assert rep.n_clause_error_records == 1
    d = rep.as_dict()
    assert d["combined_violation_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------- 레코드 실패

def test_상류가_보낸_실패는_그대로_승계한다() -> None:
    rep = run(failed("truncated"))
    (rec,) = rep.records
    assert rec.parse_ok is False
    assert rec.parse_error == "truncated"
    assert rep.upstream_failures == {"truncated": 1}
    assert rep.n_record_failures == 1


def test_레코드_실패를_새로_판정하는_칸이_없다() -> None:
    """레코드 실패는 상류가 적은 것뿐이다. 언제나 0 인 칸을 두면 '재서 0 이었다' 로 읽힌다."""
    d = run(line()).as_dict()
    assert "adapter_parse_failures" not in d
    assert not hasattr(AdaptReportV14(), "adapter_failures")


# ---------------------------------------------------------------- 짝 규칙 — 배관 고장으로 멈춘다

@pytest.mark.parametrize("parsed", [None, "x", [], {"defects": "x"}, {}])
def test_사유가_없는데_파싱_결과가_비거나_틀리면_멈춘다(parsed) -> None:
    """앞 판은 이것을 어댑터의 `schema_violation` 으로 셌다 — 쓰는 쪽의 결함이 모델 실패율에 들어갔다."""
    with pytest.raises(UpstreamContractError, match=rf"{TAG}:1: parse_error 가 null 인데"):
        run(line(bbox_px_parsed=parsed))


@pytest.mark.parametrize("parsed", [
    {"defects": [], "verdict": "합격", "cited_clauses": []},
    {"defects": "x"},
    {},
    "x",
])
def test_사유가_있는데_파싱_결과도_있으면_멈춘다(parsed) -> None:
    with pytest.raises(UpstreamContractError, match="bbox_px_parsed 가 null 이 아니다"):
        run(line(parse_error="truncated", bbox_px_parsed=parsed))


def test_짝_규칙을_어긴_줄이_하나면_전체를_멈춘다() -> None:
    """앞 줄이 멀쩡해도 보고를 내지 않는다 — 셈에서 빠진 줄이 있는 보고는 실패율이 틀린다."""
    with pytest.raises(UpstreamContractError, match=rf"{TAG}:2"):
        run(line(), line(bbox_px_parsed=None))


def test_실패한_줄이_여럿이면_레코드도_여럿이다() -> None:
    """한 줄의 값이 다른 줄에 새어 들면 실패 레코드가 서로 같아진다."""
    rep = run(failed("no_json", image_id="a"),
              failed("json_decode", image_id="b"))
    assert [r.image_id for r in rep.records] == ["a", "b"]
    assert [r.parse_error for r in rep.records] == ["no_json", "json_decode"]
    assert rep.upstream_failures == {"no_json": 1, "json_decode": 1}


# ---------------------------------------------------------------- 종료 사유

def test_정할_수_없는_종료는_값을_지어내지_않고_멈춘다() -> None:
    """앞 판은 `length` 로 채워 레코드를 만들었다 — 시험 이름과 반대로 지어낸 값을 고정했다.
    그 표시가 있는 묶음은 묶음 검증이 먼저 거부한다. 여기까지 왔으면 검증을 거치지 않은 것이다."""
    with pytest.raises(UpstreamContractError, match=STOP_UNDETERMINED):
        run(line(gen_stop=STOP_UNDETERMINED))


def test_종료_사유와_실패_여부의_교차표() -> None:
    rep = run(failed("truncated", gen_stop="length"),
              line(gen_stop="eos"),
              line(gen_stop="length"))
    assert rep.parse_by_stop == {"length|fail": 1, "eos|ok": 1, "length|ok": 1}


def test_종료_사유_건수를_센다() -> None:
    rep = run(line(gen_stop="eos"), line(gen_stop="length"), line(gen_stop="eos"))
    assert rep.gen_stop_counts == {"eos": 2, "length": 1}


# ---------------------------------------------------------------- 박스

def test_신뢰도를_지어내지_않는다() -> None:
    d = run(line()).records[0].defects[0]
    assert d.score is None
    assert d.retrieved is None, "통합형에는 검색을 붙이지 않는다"


def test_크기는_긴_축이다() -> None:
    d = run(line()).records[0].defects[0]
    assert d.size_px == pytest.approx(40.0)
    assert d.size_basis == "major_axis"


def test_채점_밖_코드는_버리고_무엇을_버렸는지_센다() -> None:
    out = sorted(KNOWN - set(SCORING))
    if not out:
        pytest.skip("채점 밖 코드가 없다")
    rep = run(line(bbox_px_parsed={"defects": [{"iso_code": out[0], "bbox_px": [1, 2, 3, 4]}],
                                   "verdict": "합격", "cited_clauses": []}))
    assert rep.n_out_of_scope == 1
    assert rep.out_of_scope_codes[out[0]] == 1
    assert rep.records[0].defects == []


def test_사상표에_없는_코드는_따로_센다() -> None:
    rep = run(line(bbox_px_parsed={"defects": [{"iso_code": "9999", "bbox_px": [1, 2, 3, 4]}],
                                   "verdict": "합격", "cited_clauses": []}))
    assert rep.n_unknown_code == 1
    assert rep.records[0].defects == []


def test_경계_이탈은_세기만_하고_판정을_바꾸지_않는다() -> None:
    rep = run(line(bbox_px_parsed={"defects": [{"iso_code": CODE, "bbox_px": [0, 0, 5000, 10]}],
                                   "verdict": "합격", "cited_clauses": []}),
              image_size={"aihub000001": (1280, 720)})
    assert rep.n_boxes == 1
    assert rep.n_boxes_out_of_bounds == 1
    assert rep.records[0].parse_ok is True
    assert len(rep.records[0].defects) == 1, "버리지 않는다"


def test_이미지_크기를_주지_않으면_이탈은_재지_않은_것이다() -> None:
    """앞 판은 0 을 냈다 — '재지 않았다' 와 '0 이었다' 가 산출에서 같은 글자였다."""
    rep = run(line(bbox_px_parsed={"defects": [{"iso_code": CODE, "bbox_px": [0, 0, 5000, 10]}],
                                   "verdict": "합격", "cited_clauses": []}))
    d = rep.as_dict()
    assert rep.n_boxes == 1
    assert d["out_of_bounds_measured"] is False
    assert d["n_boxes_out_of_bounds"] is None
    assert d["n_boxes_size_unknown"] is None


def test_크기표에_없는_이미지의_박스는_따로_센다() -> None:
    rep = run(line(bbox_px_parsed={"defects": [{"iso_code": CODE, "bbox_px": [0, 0, 5000, 10]}],
                                   "verdict": "합격", "cited_clauses": []}),
              image_size={"다른장": (1280, 720)})
    d = rep.as_dict()
    assert d["out_of_bounds_measured"] is True
    assert d["n_boxes_out_of_bounds"] == 0
    assert d["n_boxes_size_unknown"] == 1


def test_상류가_버린_항목_수를_합산한다() -> None:
    """정상 줄에서 공통 정책이 더 버리는 것이 없으므로 정확히 3 이다."""
    assert run(line(n_bad_items_dropped=3)).n_bad_items == 3


# ---------------------------------------------------------------- 레코드 규칙 위반은 줄 번호와 함께 멈춘다

@pytest.mark.parametrize("over", [
    {"coord_space": None},
    {"coord_cfg_hash": ""},
    {"gen_stop": "eos", "n_new_tokens": 0},
])
def test_레코드_규칙_위반은_줄_번호와_함께_배관_고장으로_멈춘다(over) -> None:
    """앞 판은 레코드 생성자의 검증 오류가 줄 번호 없이 그대로 올라왔다."""
    with pytest.raises(UpstreamContractError, match=rf"{TAG}:2: 레코드 규칙 위반"):
        run(line(), line(**over))


def test_실패_줄의_레코드_규칙_위반도_줄_번호와_함께_멈춘다() -> None:
    with pytest.raises(UpstreamContractError, match=rf"{TAG}:1: 레코드 규칙 위반"):
        run(failed("no_json", coord_space="ABS_WHATEVER"))


# ---------------------------------------------------------------- 보고

def test_줄이_없으면_비율이_0_이고_나눗셈이_터지지_않는다() -> None:
    d = run().as_dict()
    assert d["n_lines"] == 0 and d["n_records"] == 0
    assert d["record_failure_rate"] == 0.0
    assert d["clause_error_rate"] == 0.0
    assert d["combined_violation_rate"] == 0.0


def test_태그에서_칸과_참여자가_나온다() -> None:
    assert run(line(), tag="uni_local_C2").records[0].client == "C2"
    assert run(line(), tag="uni_central").records[0].client is None
    assert run(line(), tag="uni_central").records[0].cell == "uni_central"


def test_본실험_태그가_아니면_거부한다() -> None:
    with pytest.raises(ValueError, match="태그가 아니다"):
        run(line(), tag="uni_local_c1")


def test_인용은_이미지별로_남는다() -> None:
    rep = run(line(image_id="a"), line(image_id="b", bbox_px_parsed={
        "defects": [], "verdict": "합격", "cited_clauses": []}))
    assert rep.citations["a"] == ["IACS-47-3.1"]
    assert rep.citations["b"] == []
    assert rep.as_dict()["n_images_with_citation"] == 1


# ---------------------------------------------------------------- 해독

@pytest.mark.parametrize(("raw", "reason"), [
    ("결함이 없습니다", "no_json"),
    ("", "no_json"),
    ('{"a": 1', "truncated"),
    ('{"a": "열린 문자열', "truncated"),
    ('{"a": }', "json_decode"),
])
def test_해독_실패는_사유로_돌아온다(raw: str, reason: str) -> None:
    obj, why = decode_line(raw)
    assert obj is None and why == reason


def test_앞뒤_말이_붙어_있어도_첫_객체를_꺼낸다() -> None:
    obj, why = decode_line('판정: {"verdict": "합격"} 이상입니다')
    assert why is None and obj == {"verdict": "합격"}


def test_너무_긴_줄은_해독을_시도하지_않는다() -> None:
    obj, why = decode_line("{" + "a" * (5 << 20) + "}")
    assert obj is None and why == "json_decode"


def test_깊은_중첩에도_죽지_않는다() -> None:
    """입력이 **여는 중괄호를 품어야** 해독기에 닿는다. 앞 판은 대괄호만 넣어 `no_json` 에서 끝났고,
    기대가 사유 셋을 다 받아 그 사실을 가렸다."""
    raw = '{"defects":' + "[" * 30000
    with pytest.raises(RecursionError):                 # 전제 — 표준 해독기는 여기서 죽는다
        json.JSONDecoder().raw_decode(raw)
    obj, why = decode_line(raw)
    assert obj is None and why == "json_decode"


def test_수천_자리_정수에도_죽지_않는다() -> None:
    raw = '{"bbox_2d": [' + "9" * 5000 + ", 1, 2, 3]}"
    with pytest.raises(ValueError) as exc:              # 전제 — JSONDecodeError 가 아닌 ValueError 다
        json.JSONDecoder().raw_decode(raw)
    assert not isinstance(exc.value, json.JSONDecodeError)
    obj, why = decode_line(raw)
    assert obj is None and why == "json_decode"

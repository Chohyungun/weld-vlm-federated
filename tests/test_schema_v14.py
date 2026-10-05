"""계약 #4 의 1.4 판 — 통합형 본실험 전용. 07번 미니스펙 §12-4 · §13 · §14-2.

1.4 는 1.3 을 **고치지 않고 곁에** 선다. 시험이 지키는 것은 넷이다.

1. 1.4 에서만 필수인 것(`gen_stop`·`n_new_tokens`·`clause_error`·좌표 규약)이 **기본값 없이** 필수다 —
   선택 필드로 두고 검증기로 막으면 JSON 스키마의 `required` 에 드러나지 않아 스키마 파일이 모델보다 약해진다.
2. 새 결합 규칙의 불변식 — 레코드 실패와 조항 축 위반은 **서로소**다.
3. 판본이 섞이지 않는다 — 1.3 줄에 새 필드가 있으면 거부, 옛 되읽기는 1.4 를 읽지 못한다.
4. 1.4 클래스는 1.3 클래스의 **하위 클래스가 아니다**(부모 타입 자리에서 직렬화하면 새 필드가 경고 없이 빠진다).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from evaluation.adapters import read_records
from evaluation.schema import PredictionRecord
from evaluation.schema_v14 import (
    SCHEMA_VERSION_V14,
    UNI_MAIN_TAGS,
    UNIFIED_CELLS_V14,
    PredictionRecordV14,
    assert_single_version,
    failed_record_v14,
    json_schema_v14,
    read_any_records,
    tag_parts,
)

REPO = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO / "evaluation/prediction.schema.v1_4.json"


def rec14(**overrides) -> dict:
    base = {
        "schema_version": "1.4",
        "image_id": "synthetic:0001",
        "cell": "uni_central",
        "client": None,
        "seed": 1001,
        "defects": [{"iso_code": "2011", "bbox_px": [10.0, 20.0, 40.0, 55.0]}],
        "verdict": "불합격",
        "cited_clauses": ["IACS47-3.2.1"],
        "parse_ok": True,
        "parse_error": None,
        "clause_error": [],
        "gen_stop": "eos",
        "n_new_tokens": 57,
        "coord_space": "ABS_ORIG",
        "coord_cfg_hash": "c" * 64,
    }
    base.update(overrides)
    return base


def rec13(**overrides) -> dict:
    base = {
        "schema_version": "1.3", "image_id": "synthetic:0001", "cell": "uni_central", "seed": 1001,
        "defects": [{"iso_code": "2011", "bbox_px": [10.0, 20.0, 40.0, 55.0]}],
        "verdict": "불합격", "cited_clauses": ["IACS47-3.2.1"], "parse_ok": True,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------- 상수와 태그

def test_판본과_칸과_태그() -> None:
    assert SCHEMA_VERSION_V14 == "1.4"
    assert UNIFIED_CELLS_V14 == ("uni_local", "uni_central", "uni_fed")
    assert UNI_MAIN_TAGS == ("uni_local_C1", "uni_local_C2", "uni_local_C3", "uni_central", "uni_fed")


@pytest.mark.parametrize(("tag", "expected"), [
    ("uni_local_C1", ("uni_local", "C1")), ("uni_local_C3", ("uni_local", "C3")),
    ("uni_central", ("uni_central", None)), ("uni_fed", ("uni_fed", None)),
])
def test_태그는_명시_표로_푼다(tag: str, expected: tuple) -> None:
    assert tag_parts(tag) == expected


@pytest.mark.parametrize("tag", ["uni_local", "uni_local_c1", "uni_local_C4", "sep_fed", "uni_central_C1", ""])
def test_표에_없는_태그는_거부한다(tag: str) -> None:
    with pytest.raises(ValueError, match="태그"):
        tag_parts(tag)


# ---------------------------------------------------------------------------- 필수 필드

def test_정상_레코드가_선다() -> None:
    r = PredictionRecordV14.model_validate(rec14())
    assert r.schema_version == "1.4"
    assert r.iso_codes == frozenset({"2011"})


@pytest.mark.parametrize("field", [
    "schema_version", "clause_error", "gen_stop", "n_new_tokens", "coord_space", "coord_cfg_hash",
    "parse_error", "client", "verdict",
])
def test_1_4_의_필수_필드에는_기본값이_없다(field: str) -> None:
    payload = rec14()
    del payload[field]
    with pytest.raises(ValidationError):
        PredictionRecordV14.model_validate(payload)


def test_필수성이_json_스키마의_required_에_드러난다() -> None:
    required = set(json_schema_v14()["required"])
    assert {"schema_version", "clause_error", "gen_stop", "n_new_tokens", "coord_space",
            "coord_cfg_hash", "parse_error", "client", "verdict"} <= required


@pytest.mark.parametrize("bad", [True, "12", 12.0, -1, None])
def test_n_new_tokens_는_0_이상의_엄격한_정수다(bad: object) -> None:
    with pytest.raises(ValidationError):
        PredictionRecordV14.model_validate(rec14(n_new_tokens=bad))


@pytest.mark.parametrize("bad", ["EOS", "stop", "", None])
def test_gen_stop_은_두_값뿐이다(bad: object) -> None:
    with pytest.raises(ValidationError):
        PredictionRecordV14.model_validate(rec14(gen_stop=bad))


def test_종료_토큰으로_끝났으면_토큰이_하나는_있다() -> None:
    with pytest.raises(ValidationError, match="eos"):
        PredictionRecordV14.model_validate(rec14(gen_stop="eos", n_new_tokens=0))


def test_여분_필드는_거부한다() -> None:
    with pytest.raises(ValidationError):
        PredictionRecordV14.model_validate(rec14(score_hint=0.5))


# ---------------------------------------------------------------------------- 칸과 참여자

def test_1_4_는_통합형_세_칸_전용이다() -> None:
    for cell in ("sep_local", "sep_central", "sep_fed"):
        with pytest.raises(ValidationError):
            PredictionRecordV14.model_validate(rec14(cell=cell))


def test_uni_local_이면_참여자가_필수이고_그_밖에서는_금지다() -> None:
    assert PredictionRecordV14.model_validate(rec14(cell="uni_local", client="C2")).client == "C2"
    with pytest.raises(ValidationError, match="uni_local"):
        PredictionRecordV14.model_validate(rec14(cell="uni_local", client=None))
    for cell in ("uni_central", "uni_fed"):
        with pytest.raises(ValidationError, match="client"):
            PredictionRecordV14.model_validate(rec14(cell=cell, client="C1"))


def test_통합형에는_검색_결과가_없다() -> None:
    d = [{"iso_code": "2011", "bbox_px": [1.0, 2.0, 3.0, 4.0], "retrieved": ["X-1"]}]
    for cell, client in (("uni_local", "C1"), ("uni_central", None), ("uni_fed", None)):
        with pytest.raises(ValidationError, match="retrieved"):
            PredictionRecordV14.model_validate(rec14(cell=cell, client=client, defects=d))


# ---------------------------------------------------------------------------- 새 결합 규칙의 불변식

def test_레코드_실패는_빈_예측이고_조항_축_표시가_없다() -> None:
    ok = rec14(parse_ok=False, parse_error="truncated", defects=[], verdict=None, cited_clauses=[],
               gen_stop="length", n_new_tokens=2368)
    assert PredictionRecordV14.model_validate(ok).iso_codes == frozenset()
    for broken in (
        {"defects": [{"iso_code": "2011", "bbox_px": [1.0, 2.0, 3.0, 4.0]}]},
        {"clause_error": ["verdict_invalid"]},
        {"verdict": "판정불가"},
        {"cited_clauses": ["A-1"]},
        {"parse_error": None},
    ):
        with pytest.raises(ValidationError):
            PredictionRecordV14.model_validate({**ok, **broken})


def test_성공_레코드에는_실패_사유가_없다() -> None:
    with pytest.raises(ValidationError, match="parse_error"):
        PredictionRecordV14.model_validate(rec14(parse_error="json_decode"))


def test_판정이_비면_verdict_invalid_이고_그_역도_참이다() -> None:
    PredictionRecordV14.model_validate(rec14(verdict=None, clause_error=["verdict_invalid"]))
    with pytest.raises(ValidationError, match="verdict"):
        PredictionRecordV14.model_validate(rec14(verdict=None))
    with pytest.raises(ValidationError, match="verdict"):
        PredictionRecordV14.model_validate(rec14(clause_error=["verdict_invalid"]))


def test_인용이_틀렸으면_인용_목록은_비어_있다() -> None:
    PredictionRecordV14.model_validate(rec14(cited_clauses=[], clause_error=["citations_invalid"]))
    with pytest.raises(ValidationError, match="cited_clauses"):
        PredictionRecordV14.model_validate(rec14(clause_error=["citations_invalid"]))


def test_판정만_틀렸으면_유효한_인용은_남는다() -> None:
    r = PredictionRecordV14.model_validate(rec14(verdict=None, clause_error=["verdict_invalid"]))
    assert r.cited_clauses == ["IACS47-3.2.1"]
    assert len(r.defects) == 1


@pytest.mark.parametrize("bad", [
    ["verdict_invalid", "verdict_invalid"], ["verdict_invalid", "citations_invalid"], ["other"], "verdict_invalid",
])
def test_clause_error_는_정렬된_중복_없는_목록이다(bad: object) -> None:
    payload = rec14(verdict=None, cited_clauses=[], clause_error=bad)
    with pytest.raises(ValidationError):
        PredictionRecordV14.model_validate(payload)


def test_둘_다_틀린_경우를_적을_수_있다() -> None:
    r = PredictionRecordV14.model_validate(
        rec14(verdict=None, cited_clauses=[], clause_error=["citations_invalid", "verdict_invalid"]))
    assert r.clause_error == ["citations_invalid", "verdict_invalid"]


@pytest.mark.parametrize("err", ["unknown_iso_code", "bbox_invalid", "join_missing"])
def test_상류_실패_어휘는_넷뿐이다(err: str) -> None:
    payload = rec14(parse_ok=False, parse_error=err, defects=[], verdict=None, cited_clauses=[])
    with pytest.raises(ValidationError):
        PredictionRecordV14.model_validate(payload)


def test_실패_레코드_생성기는_생성자를_거친다() -> None:
    r = failed_record_v14("synthetic:0002", cell="uni_local", client="C3", seed=1002, error="no_json",
                          gen_stop="eos", n_new_tokens=3, coord_space="ABS_ORIG", coord_cfg_hash="c" * 64,
                          latency_ms=12.5)
    assert (r.parse_ok, r.defects, r.verdict, r.clause_error) == (False, [], None, [])
    assert PredictionRecordV14.model_validate_json(r.model_dump_json()) == r
    with pytest.raises(ValidationError):
        failed_record_v14("synthetic:0002", cell="uni_local", client="C3", seed=1002, error="no_json",
                          gen_stop=None, n_new_tokens=3, coord_space="ABS_ORIG", coord_cfg_hash="c" * 64)


# ---------------------------------------------------------------------------- 1.3 과의 관계

def test_1_4_클래스는_1_3_클래스의_하위_클래스가_아니다() -> None:
    assert PredictionRecord not in PredictionRecordV14.__mro__
    assert not isinstance(PredictionRecordV14.model_validate(rec14()), PredictionRecord)


def test_같은_결함이면_두_판본의_파생_속성이_같다() -> None:
    defects = [{"iso_code": "401", "bbox_px": [1.0, 2.0, 3.0, 4.0]}, {"iso_code": "2011", "bbox_px": [5.0, 6.0, 7.0, 8.0]},
               {"iso_code": "2011", "bbox_px": [9.0, 9.0, 12.0, 12.0]}]
    a = PredictionRecord.model_validate(rec13(defects=defects))
    b = PredictionRecordV14.model_validate(rec14(defects=defects))
    assert a.iso_codes == b.iso_codes
    assert a.pairs() == b.pairs()


def test_직렬화와_되읽기가_고정점이다() -> None:
    r = PredictionRecordV14.model_validate(rec14(cell="uni_local", client="C1", latency_ms=812.4))
    once = r.model_dump_json()
    assert PredictionRecordV14.model_validate_json(once).model_dump_json() == once


# ---------------------------------------------------------------------------- 판본 유니온

def _line(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def test_판본으로_갈라_읽는다() -> None:
    recs = read_any_records([_line(rec13()), _line(rec14())])
    assert [type(r) for r in recs] == [PredictionRecord, PredictionRecordV14]


def test_1_3_줄에_새_필드가_있으면_거부한다() -> None:
    with pytest.raises(ValidationError):
        read_any_records([_line(rec13(gen_stop="eos", n_new_tokens=3))])
    with pytest.raises(ValidationError):
        read_any_records([_line(rec13(cell="uni_local", client="C1"))])


def test_1_4_줄에_필드가_빠지면_거부한다() -> None:
    payload = rec14()
    del payload["gen_stop"]
    with pytest.raises(ValidationError):
        read_any_records([_line(payload)])


@pytest.mark.parametrize("version", [None, "1.5", "1.2", 1.4])
def test_모르는_판본은_거부한다(version: object) -> None:
    payload = rec14()
    if version is None:
        del payload["schema_version"]
    else:
        payload["schema_version"] = version
    with pytest.raises(ValidationError):
        read_any_records([_line(payload)])


def test_옛_되읽기는_1_4_를_읽지_못한다() -> None:
    """옛 경로가 1.4 를 읽게 되면 옛 결합 규칙으로 1.4 가 조용히 채점되는 길이 열린다."""
    with pytest.raises(ValidationError):
        read_records([json.dumps(rec14(), ensure_ascii=False)])


def test_빈_줄은_오류다() -> None:
    with pytest.raises(ValueError, match="빈 줄"):
        read_any_records([_line(rec14()), b""])


def test_판본이_섞인_입력을_가려낸다() -> None:
    a = PredictionRecord.model_validate(rec13())
    b = PredictionRecordV14.model_validate(rec14())
    assert assert_single_version([b, b]) == "1.4"
    assert assert_single_version([a]) == "1.3"
    with pytest.raises(ValueError, match="섞"):
        assert_single_version([a, b])
    with pytest.raises(ValueError, match="비어"):
        assert_single_version([])


# ---------------------------------------------------------------------------- 스키마 파일과 임포트

def test_커밋된_1_4_스키마_파일이_모델과_같다() -> None:
    assert SCHEMA_PATH.exists(), f"{SCHEMA_PATH.name} 가 없다 — scripts/gen_prediction_schema_v14.py 를 돌린다"
    assert json.loads(SCHEMA_PATH.read_text(encoding="utf-8")) == json_schema_v14()
    assert b"\r" not in SCHEMA_PATH.read_bytes()


def test_1_4_스키마는_별도의_id_를_갖는다() -> None:
    s = json_schema_v14()
    assert s["$id"].endswith("prediction.schema.v1_4.json")
    assert s["properties"]["schema_version"]["const"] == "1.4"


def test_새_모듈은_첫_임포트로_불러도_서고_무거운_스택을_끌지_않는다() -> None:
    code = "import sys, evaluation.schema_v14; print('torch' in sys.modules, 'evaluation.cells' in sys.modules)"
    out = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", code], cwd=REPO, capture_output=True,
                         text=True, encoding="utf-8", timeout=120, check=False)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False False"

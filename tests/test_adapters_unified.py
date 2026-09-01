"""통합형 어댑터 + 단일 채점기 — 66번 신규.

어댑터는 칸을 구분해도 되는 유일한 지점이고, 그 뒤로는 다섯 칸이 같은 함수를 탄다.
여기서 고정하는 것은 **어댑터가 답을 고쳐주지 않는다**는 성질이다.
"""

from __future__ import annotations

import json

import pytest

from evaluation.adapters import adapt_unified_generations, read_records
from evaluation.schema import PredictionRecord
from evaluation.score import coord_health, failure_breakdown, score_records

CODES = ("100", "2011", "301", "401", "402", "2012")


def line(image_id: str, **kw) -> str:
    row = {
        "image_id": image_id,
        "text": "",
        "bbox_px_parsed": kw.pop("parsed", {
            "defects": [], "verdict": "판정불가", "cited_clauses": [],
        }),
        "parse_error": kw.pop("parse_error", None),
        "coord_space": "NORM_1000",
        "coord_cfg_hash": "deadbeef",
        "latency_ms": 12.5,
    }
    row.update(kw)
    return json.dumps(row, ensure_ascii=False)


def adapt(lines, **kw):
    return adapt_unified_generations(
        lines, cell=kw.pop("cell", "uni_central"), seed=20260828,
        known_iso_codes=CODES, **kw,
    )


def test_정상_레코드는_그대로_통과하고_좌표를_바꾸지_않는다() -> None:
    parsed = {
        "defects": [{"iso_code": "2011", "bbox_px": [624.64, 349.2, 652.8, 375.84]}],
        "verdict": "판정불가", "cited_clauses": ["KRA27-T15"],
    }
    rep = adapt([line("img1", parsed=parsed)])
    assert len(rep.records) == 1
    rec = rep.records[0]
    assert rec.parse_ok is True
    # float 유지 — 정수화는 채점기의 최종 1회이고 어댑터가 하지 않는다
    assert rec.defects[0].bbox_px == (624.64, 349.2, 652.8, 375.84)
    assert rec.coord_space == "NORM_1000"
    assert rec.cited_clauses == ["KRA27-T15"]
    assert rep.citations["img1"] == ["KRA27-T15"]


def test_상류_파싱실패는_재시도없이_오답으로_남고_사유가_보존된다() -> None:
    rep = adapt([line("img1", parse_error="truncated", parsed=None)])
    rec = rep.records[0]
    assert rec.parse_ok is False and rec.parse_error == "truncated"
    assert rec.defects == []          # 빈 예측 → 미검출로 계상된다
    assert rep.upstream_failures == {"truncated": 1}
    assert rep.adapter_failures == {}
    # 실패해도 레코드가 사라지지 않는다 — 사라지면 오답보다 낙관적으로 잡힌다
    assert failure_breakdown(rep.records)["n_parse_fail"] == 1


def test_계약_밖_사유는_조용히_삼키지_않고_멈춘다() -> None:
    with pytest.raises(ValueError, match="계약 밖 parse_error"):
        adapt([line("img1", parse_error="gpu_oom", parsed=None)])


def test_미지_iso_코드는_오답이지_보정_대상이_아니다() -> None:
    parsed = {"defects": [{"iso_code": "9999", "bbox_px": [1, 1, 2, 2]}],
              "verdict": "판정불가", "cited_clauses": []}
    rep = adapt([line("img1", parsed=parsed)])
    assert rep.adapter_failures == {"unknown_iso_code": 1}
    assert rep.records[0].defects == []


def test_퇴화_bbox_는_bbox_invalid_로_오답_처리된다() -> None:
    parsed = {"defects": [{"iso_code": "2011", "bbox_px": [5, 5, 5, 9]}],
              "verdict": "판정불가", "cited_clauses": []}
    rep = adapt([line("img1", parsed=parsed)])
    assert rep.adapter_failures == {"bbox_invalid": 1}


def test_enum_밖_verdict_는_스키마_위반이다() -> None:
    parsed = {"defects": [], "verdict": "아마도 합격", "cited_clauses": []}
    rep = adapt([line("img1", parsed=parsed)])
    assert rep.adapter_failures == {"schema_violation": 1}


def test_경계_이탈_박스는_세기만_하고_버리지도_자르지도_않는다() -> None:
    parsed = {"defects": [{"iso_code": "2011", "bbox_px": [-4.0, 10.0, 1400.0, 200.0]}],
              "verdict": "판정불가", "cited_clauses": []}
    rep = adapt([line("img1", parsed=parsed)], image_size={"img1": (1280, 720)})
    assert rep.n_boxes == 1 and rep.n_boxes_out_of_bounds == 1
    assert rep.records[0].parse_ok is True
    # 클리핑하면 IoU 가 올라가는 방향으로만 작동한다 — 좌표는 그대로여야 한다
    assert rep.records[0].defects[0].bbox_px == (-4.0, 10.0, 1400.0, 200.0)


def test_통합형은_retrieved_를_붙이지_않는다() -> None:
    parsed = {"defects": [{"iso_code": "2011", "bbox_px": [1, 1, 5, 5]}],
              "verdict": "판정불가", "cited_clauses": []}
    rep = adapt([line("img1", parsed=parsed)], cell="uni_fed")
    assert rep.records[0].defects[0].retrieved is None
    assert rep.records[0].client is None


def test_왕복_직렬화가_레코드를_보존한다() -> None:
    parsed = {"defects": [{"iso_code": "301", "bbox_px": [1.5, 2.5, 30.5, 40.5]}],
              "verdict": "판정불가", "cited_clauses": ["KRA27-T16"]}
    rep = adapt([line("img1", parsed=parsed)])
    back = read_records([r.model_dump_json() for r in rep.records])
    assert back == rep.records


def _rec(image_id: str, boxes) -> PredictionRecord:
    return PredictionRecord(
        schema_version="1.3", image_id=image_id, cell="uni_central", seed=1,
        defects=[{"iso_code": c, "bbox_px": b} for c, b in boxes],
        verdict="판정불가", cited_clauses=[], parse_ok=True,
    )


def test_단일_채점기가_통합형_레코드도_그대로_채점한다() -> None:
    gold_codes = {"a": {"2011"}, "b": set()}
    gold_boxes = {"a": [("2011", (10.0, 10.0, 20.0, 20.0))], "b": []}
    recs = [_rec("a", [("2011", (10.0, 10.0, 20.0, 20.0))]), _rec("b", [])]
    m = score_records(recs, gold_codes, gold_boxes, ["100", "2011", "301", "401"])
    assert m["bbox_iou_matched_only"] == pytest.approx(1.0)
    assert m["miss_rate"] == pytest.approx(0.0)
    assert m["skipped_classes"] == ["100", "301", "401"]


def test_좌표_건강_판정이_미검출과_붕괴를_가른다() -> None:
    healthy = coord_health({"bbox_iou_matched_only": 0.44, "n_matched": 134,
                            "n_gold": 1281, "coord_suspect": False})
    assert healthy["verdict"].startswith("건강")

    collapsed = coord_health({"bbox_iou_matched_only": 0.055, "n_matched": 400,
                              "n_gold": 1281, "coord_suspect": True})
    assert "붕괴" in collapsed["verdict"]

    silent = coord_health({"bbox_iou_matched_only": 0.0, "n_matched": 0,
                           "n_gold": 1281, "coord_suspect": False})
    assert "판정 불가" in silent["verdict"]

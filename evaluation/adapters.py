"""원시 출력 → 계약 #4 레코드. **칸을 구분해도 되는 유일한 지점**(schema.py 모듈 주석).

어댑터는 원시 출력 형식의 차이만 흡수한다. 지표 계산은 어댑터 뒤에 있는 단일 채점기
(`evaluation/score.py`)가 전담하고, 여기서는 어떤 지표도 계산하지 않는다.

**좌표를 변환하지 않는다.** C 가 `to_px` 로 역변환을 끝낸 `bbox_px`(원본 픽셀, float)를
그대로 받는다 — 이중 역변환을 구조적으로 막는 장치다(스펙 §3-4). 이미지 경계 이탈은
**세기만 하고 버리지 않는다**: 클리핑은 IoU 를 올리는 방향으로만 작동해 답을 고쳐주는
셈이고, 실패로 처리하면 나쁜 예측이 파싱 실패로 둔갑해 실패율이 거짓말을 한다.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import get_args

from evaluation.schema import (
    SCHEMA_VERSION,
    Cell,
    ParseError,
    PredictionRecord,
    Verdict,
    failed_record,
)

_VERDICTS = set(get_args(Verdict))
_PARSE_ERRORS = set(get_args(ParseError))


@dataclass
class AdaptReport:
    """어댑터 통과 결과. 버린 것이 없다는 것을 건수로 증명한다."""

    records: list[PredictionRecord] = field(default_factory=list)
    n_lines: int = 0
    adapter_failures: dict[str, int] = field(default_factory=dict)
    """어댑터가 **새로** 판정한 실패(원시 파일의 `parse_error` 와 별개)."""
    upstream_failures: dict[str, int] = field(default_factory=dict)
    """C 가 이미 기록해 보낸 실패. 재시도 없이 그대로 오답 처리한다."""
    n_boxes: int = 0
    n_boxes_out_of_bounds: int = 0
    """원본 이미지 경계를 벗어난 박스 수. 좌표계 진단 신호이며 채점에는 개입하지 않는다."""
    citations: dict[str, list[str]] = field(default_factory=dict)
    """image_id → 생성문이 인용한 조항 ID. 무근거 인용률의 입력이다."""

    def as_dict(self) -> dict:
        return {
            "n_lines": self.n_lines,
            "n_records": len(self.records),
            "upstream_parse_failures": self.upstream_failures,
            "adapter_parse_failures": self.adapter_failures,
            "n_boxes": self.n_boxes,
            "n_boxes_out_of_bounds": self.n_boxes_out_of_bounds,
            "n_images_with_citation": sum(1 for v in self.citations.values() if v),
        }


def adapt_unified_generations(
    lines: Iterable[str],
    *,
    cell: Cell,
    seed: int,
    known_iso_codes: Iterable[str],
    image_size: Mapping[str, tuple[int, int]] | None = None,
) -> AdaptReport:
    """통합형 `generations.jsonl` → 계약 #4 레코드.

    Args:
        lines: 원시 파일의 각 줄.
        cell: `uni_central` / `uni_fed`.
        seed: C 의 파일럿 시드.
        known_iso_codes: `label_map.yaml` 의 코드 집합. 하드코딩하지 않는다(불변조건 1-8).
        image_size: image_id → (W, H). 경계 이탈 **집계용**이며 판정에는 쓰지 않는다.

    검증은 엄격하게 한다 — enum 위반·미지 코드·퇴화 박스는 오답 처리하고 사유를 남긴다.
    **어떤 필드값도 보정하지 않는다.**
    """
    codes = set(known_iso_codes)
    rep = AdaptReport()

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        rep.n_lines += 1
        row = json.loads(raw)
        image_id = str(row["image_id"])
        common = {
            "coord_space": row.get("coord_space"),
            "coord_cfg_hash": row.get("coord_cfg_hash"),
            "latency_ms": row.get("latency_ms"),
        }

        def fail(reason: str, *, upstream: bool) -> None:
            bucket = rep.upstream_failures if upstream else rep.adapter_failures
            bucket[reason] = bucket.get(reason, 0) + 1
            rec = failed_record(image_id, cell, seed, reason)  # type: ignore[arg-type]
            rep.records.append(rec.model_copy(update=common))

        upstream_err = row.get("parse_error")
        if upstream_err is not None:
            if upstream_err not in _PARSE_ERRORS:
                raise ValueError(
                    f"{image_id}: 계약 밖 parse_error {upstream_err!r} — 스키마를 먼저 맞춘다"
                )
            fail(str(upstream_err), upstream=True)
            continue

        parsed = row.get("bbox_px_parsed")
        if not isinstance(parsed, dict):
            fail("schema_violation", upstream=False)
            continue

        verdict = parsed.get("verdict")
        cited = parsed.get("cited_clauses", [])
        if verdict not in _VERDICTS:
            fail("schema_violation", upstream=False)
            continue
        if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
            fail("schema_violation", upstream=False)
            continue

        raw_defects = parsed.get("defects")
        if not isinstance(raw_defects, list):
            fail("schema_violation", upstream=False)
            continue

        defects: list[dict] = []
        reason: str | None = None
        for d in raw_defects:
            code = str(d.get("iso_code", ""))
            box = d.get("bbox_px")
            if code not in codes:
                reason = "unknown_iso_code"
                break
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                reason = "bbox_invalid"
                break
            x1, y1, x2, y2 = (float(v) for v in box)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x1 >= x2 or y1 >= y2:
                reason = "bbox_invalid"
                break
            defects.append({
                "iso_code": code,
                "bbox_px": [x1, y1, x2, y2],
                "score": None,          # 생성 모델은 신뢰도를 내지 않는다. 지어내지 않는다
                "size_px": max(x2 - x1, y2 - y1),
                "size_basis": "major_axis",
                "retrieved": None,      # 통합형은 검색을 붙이지 않는다(스키마 교차검증)
            })
        if reason is not None:
            fail(reason, upstream=False)
            continue

        wh = (image_size or {}).get(image_id)
        for d in defects:
            rep.n_boxes += 1
            if wh is None:
                continue
            x1, y1, x2, y2 = d["bbox_px"]
            if x1 < 0 or y1 < 0 or x2 > wh[0] or y2 > wh[1]:
                rep.n_boxes_out_of_bounds += 1

        rep.citations[image_id] = list(cited)
        rep.records.append(PredictionRecord(
            schema_version=SCHEMA_VERSION,
            image_id=image_id, cell=cell, client=None, seed=seed,
            defects=defects,                # type: ignore[arg-type]
            verdict=verdict,                # type: ignore[arg-type]
            cited_clauses=list(cited),
            parse_ok=True,
            **common,                       # type: ignore[arg-type]
        ))
    return rep


def read_records(lines: Iterable[str]) -> list[PredictionRecord]:
    """이미 계약 #4 로 저장된 jsonl 을 되읽는다(65번 산출물 재채점 경로).

    보정 없이 그대로 검증한다 — 실패하면 예외다. 저장 시점에 통과한 레코드가 되읽기에서
    깨지면 그것은 채점 대상이 아니라 배관 고장이다.
    """
    out: list[PredictionRecord] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        out.append(PredictionRecord.model_validate_json(raw))
    return out

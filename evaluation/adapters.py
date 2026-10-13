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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import get_args

from evaluation.policy import DEFECT_ITEM_POLICY, filter_defect_items
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
    n_bad_items: int = 0
    """형식이 깨져 **항목만** 버린 결함 수. 레코드는 살아 있다(80번 D7)."""
    n_unknown_code: int = 0
    """`label_map` 에 없는 코드. 환각 신호다."""
    n_out_of_scope: int = 0
    """`label_map` 에는 있으나 채점 4클래스 밖. 버리되 센다(80번 D8)."""
    out_of_scope_codes: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_lines": self.n_lines,
            "n_records": len(self.records),
            "upstream_parse_failures": self.upstream_failures,
            "adapter_parse_failures": self.adapter_failures,
            "n_boxes": self.n_boxes,
            "n_boxes_out_of_bounds": self.n_boxes_out_of_bounds,
            "n_images_with_citation": sum(1 for v in self.citations.values() if v),
            "discard_policy": DEFECT_ITEM_POLICY,
            "n_bad_items_dropped": self.n_bad_items,
            "n_unknown_code_dropped": self.n_unknown_code,
            "n_out_of_scope_dropped": self.n_out_of_scope,
            "out_of_scope_codes": dict(sorted(self.out_of_scope_codes.items())),
        }


def adapt_unified_generations(
    lines: Iterable[str],
    *,
    cell: Cell,
    seed: int,
    known_iso_codes: Iterable[str],
    scoring_iso_codes: Iterable[str] | None = None,
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
    scoring = set(scoring_iso_codes) if scoring_iso_codes is not None else codes
    rep = AdaptReport()

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        rep.n_lines += 1
        row = json.loads(raw)
        image_id = str(row["image_id"])
        upstream_bad = row.get("n_bad_items_dropped", 0)
        if type(upstream_bad) is not int or upstream_bad < 0:
            raise ValueError(f"{image_id}: n_bad_items_dropped must be a nonnegative integer")
        rep.n_bad_items += upstream_bad
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

        # **항목 단위 폐기.** 이전 판은 결함 하나가 깨지면 `break` 로 레코드 전체를
        # 폐기했다. 분리형은 그 박스만 건너뛰고 나머지를 살리므로, 같은 이미지가 칸에
        # 따라 "전량 미검출" 또는 "일부 검출"이 됐다(80번 D7). 정책은 이제
        # `evaluation.policy` 한 곳에 있고 두 계열이 같은 함수를 부른다.
        filt = filter_defect_items(
            raw_defects, known_codes=codes, scoring_codes=scoring,
        )
        defects = [
            {
                "iso_code": d["iso_code"],
                "bbox_px": d["bbox_px"],
                "score": None,          # 생성 모델은 신뢰도를 내지 않는다. 지어내지 않는다
                "size_px": max(d["bbox_px"][2] - d["bbox_px"][0],
                               d["bbox_px"][3] - d["bbox_px"][1]),
                "size_basis": "major_axis",
                "retrieved": None,      # 통합형은 검색을 붙이지 않는다(스키마 교차검증)
            }
            for d in filt.kept
        ]
        rep.n_bad_items += filt.n_bad_item
        rep.n_unknown_code += filt.n_unknown_code
        rep.n_out_of_scope += filt.n_out_of_scope
        for c, k in filt.out_of_scope_codes.items():
            rep.out_of_scope_codes[c] = rep.out_of_scope_codes.get(c, 0) + k

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


# --------------------------------------------------------------------------------------
# 본실험 검출 export → 계약 #4  (13_spec_D §2-3 · 10번 §4)
# --------------------------------------------------------------------------------------

DETECTION_EXPORT_TAGS: tuple[tuple[str, str, str | None], ...] = (
    ("sep_local_c0", "sep_local", "C1"),
    ("sep_local_c1", "sep_local", "C2"),
    ("sep_local_c2", "sep_local", "C3"),
    ("sep_central", "sep_central", None),
    ("sep_fed", "sep_fed", None),
)
"""C 의 export 태그 → (칸, 클라이언트). **클라이언트 번호가 여기서 이름으로 바뀐다.**

C 는 `client_idx` 0·1·2 를 파일명에 쓰고(`scripts/main_det.py:_export_targets`), 계약 #4 는
`C1`·`C2`·`C3` 를 쓴다. 두 이름을 잇는 표가 코드에 없으면 채점이 클라이언트를 밀어서
읽어도 아무 데서도 안 걸린다 — RQ3 의 클라이언트별 이득이 통째로 뒤바뀐다.
근거: `evaluation.detect_infer.CHECKPOINTS` 가 파일럿에서 이미 쓰던 같은 대응
(`sep_local_c0.npz` ↔ `C1`)이고, C 의 `client-tags = "C1,C2,C3"` 도 같은 순서다.
"""


def adapt_detection_export(
    lines: Iterable[str],
    *,
    cell: Cell,
    client: str | None,
    seed: int,
    class_names: Iterable[str],
    iso_code_of,
    image_size: Mapping[str, tuple[int, int]] | None = None,
) -> AdaptReport:
    """본실험 검출 `{tag}.detections.jsonl` → 계약 #4 레코드.

    입력 한 줄: `{image_id, boxes:[{cls, xyxy_px, conf}], coord_space, coord_cfg_hash}`
    (13_spec_D §2-3, C 가 `conf=0.01` 하한으로 낸다).

    **파일럿의 `evaluation.detect_infer._record_from_result` 와 같은 매핑을 쓴다.** 그쪽은
    Ultralytics 결과 객체에서, 이쪽은 C 가 저장한 jsonl 에서 만들 뿐 만들어지는 레코드는
    같은 모양이어야 한다 — 파일럿과 본실험의 채점 입력이 갈리면 두 실험을 비교할 수 없다.
    같은 규칙 셋:

    - `cls` 정수 → `class_names[cls]` → `iso_code_of(...)`. 클래스 순서는 C 가 nc=4 로
      주입한 순서이며 `ScoringParams.class_names` 가 정본이다.
    - 퇴화 박스(x1≥x2 또는 y1≥y2)는 **항목만** 버리고 센다. 스키마가 거부하므로 만들지
      않는 것이고, 레코드는 살린다(80번 D7 의 항목 단위 폐기).
    - 경계 이탈은 **세기만 한다.** 클리핑은 IoU 를 올리는 방향으로만 작동한다(모듈 주석).
    - `verdict="판정불가"` · `cited_clauses=[]` — 판정부(⑤) 미실행, 검출 축 선채점.

    **좌표를 변환하지 않고 `coord_space` 를 덮어쓰지도 않는다.** 파일이 말하는 값을 그대로
    레코드에 싣는다 — 다르면 `coord_space_contract` 게이트가 판정한다. 여기서 정정하면
    그 게이트가 영원히 통과한다.
    """
    names = list(class_names)
    rep = AdaptReport()

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        rep.n_lines += 1
        row = json.loads(raw)
        image_id = str(row["image_id"])
        boxes = row.get("boxes")
        if not isinstance(boxes, list):
            rep.adapter_failures["schema_violation"] = (
                rep.adapter_failures.get("schema_violation", 0) + 1)
            rep.records.append(failed_record(
                image_id, cell, seed, "schema_violation", client=client,
            ).model_copy(update={"coord_space": row.get("coord_space"),
                                 "coord_cfg_hash": row.get("coord_cfg_hash")}))
            continue

        wh = (image_size or {}).get(image_id)
        defects = []
        for b in boxes:
            try:
                cls = int(b["cls"])
                x1, y1, x2, y2 = (float(v) for v in b["xyxy_px"])
                score = float(b["conf"])
            except (KeyError, TypeError, ValueError):
                rep.n_bad_items += 1
                continue
            if not 0 <= cls < len(names):
                # 학습 클래스 수와 채점 클래스 수가 갈린 것이다. 지어내지 않고 센다.
                rep.n_unknown_code += 1
                continue
            if x1 >= x2 or y1 >= y2:
                rep.n_bad_items += 1
                continue
            rep.n_boxes += 1
            if wh is not None and (x1 < 0 or y1 < 0 or x2 > wh[0] or y2 > wh[1]):
                rep.n_boxes_out_of_bounds += 1
            defects.append({
                "iso_code": iso_code_of(names[cls]),
                "bbox_px": [x1, y1, x2, y2],
                "score": score,
                "size_px": max(x2 - x1, y2 - y1),
                "size_basis": "major_axis",
                "retrieved": None,
            })

        rep.records.append(PredictionRecord(
            schema_version=SCHEMA_VERSION,
            image_id=image_id, cell=cell, client=client, seed=seed,
            defects=defects,                    # type: ignore[arg-type]
            verdict="판정불가",                  # 판정부(⑤) 미실행 — 검출 축 선채점
            cited_clauses=[],
            parse_ok=True,
            coord_space=row.get("coord_space"),
            coord_cfg_hash=row.get("coord_cfg_hash"),
        ))
    return rep

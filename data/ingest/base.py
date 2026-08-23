"""ingest 어댑터 공통 인터페이스. 스펙 §3.

어댑터는 manifest 컬럼을 직접 쓰지 않는다. `ImageRecord` 만 만들고, 컬럼 직렬화는
`records_to_frames()` 한 곳에서만 일어난다 — 어댑터가 늘어도 스키마가 갈라지지 않게 하는
장치다(스펙 §3-1).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import pandas as pd
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from data.label_map import LabelMap
from data.manifest_io import ANNOTATION_COLUMNS, MANIFEST_COLUMNS, VerdictMode


class DefectRecord(BaseModel):
    """결함 인스턴스 1개. 기하 필드는 위치 라벨이 없는 데이터셋에서 전부 None 이다."""

    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=0)
    src_label_raw: str
    defect_type: str
    iso_code: str
    polygon: list[tuple[float, float]] | None = None
    bbox_px: tuple[int, int, int, int] | None = None
    area_px: float | None = None
    major_axis_px: float | None = None
    minor_axis_px: float | None = None
    equiv_diameter_px: float | None = None
    major_axis_mm: float | None = None
    equiv_diameter_mm: float | None = None
    geom_valid: bool = True
    geom_flags: str = ""


class ImageRecord(BaseModel):
    """이미지 1장. §2-1 컬럼과 1:1 대응하되 분할·묶음 컬럼은 후속 단계가 채운다."""

    model_config = ConfigDict(extra="forbid")

    image_id: str
    source: str
    rel_path: str
    sha256: str = Field(min_length=64, max_length=64)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    modality: str
    material: str
    label_type: str
    has_localization: bool
    thickness_mm: float | None = None
    thickness_source: str = "none"
    px_per_mm: float | None = None
    scale_source: str = "none"
    quality_level: str | None = None
    ingest_version: str
    defects: list[DefectRecord] = Field(default_factory=list)
    #: 원본이 알려주는 묶음 식별자(필름·모원본·연속촬영). dedup 의 E2 엣지로 들어간다.
    #: 계약 #2 컬럼이 아니다 — group_id 는 dedup 단계가 이 값과 pHash 를 합쳐 만든다.
    group_key: str | None = None
    reject_reason: str | None = None
    notes: str = ""

    @field_validator("rel_path")
    @classmethod
    def _posix(cls, v: str) -> str:
        if "\\" in v:
            raise ValueError("rel_path 는 POSIX 슬래시를 쓴다")
        return v


@dataclass(frozen=True)
class RawItem:
    """원본 1건. 어댑터가 discover() 로 낸다. 읽기 전용이다."""

    path: Path
    rel_path: str
    label: str
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    """데이터셋이 무엇을 할 수 있는지. §4-2 의 전역 플래그 재료."""

    localization: bool
    thickness_mm: bool
    pixel_scale: bool
    verdict_mode: VerdictMode
    counts: dict[str, int]


@runtime_checkable
class IngestAdapter(Protocol):
    source: str
    version: str

    def discover(self, raw_root: Path) -> Iterator[RawItem]: ...
    def parse(self, item: RawItem, label_map: LabelMap) -> ImageRecord: ...
    def capabilities(self, records: Sequence[ImageRecord]) -> Capabilities: ...


# --------------------------------------------------------------------------------------
# 공통 유틸 — 원본을 읽기만 한다
# --------------------------------------------------------------------------------------

#: 보유율 임계. 이 미만은 반올림해서 true 로 올리지 않는다 (부분 보유 위험, 스펙 §4-2).
COVERAGE_THRESHOLD = 0.95


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_image_path(rel_path: str, rel_base: Path) -> Path:
    """manifest.rel_path → 실제 파일 경로. **경로 복원의 단일 창구.**

    호출부가 `raw_root.parent.parent` 같은 계산을 직접 하면 어긋난다 — 실제로 어긋났다.
    `rel_base` 는 discover() 에 넘긴 값과 같아야 하고, 기본 레이아웃에서는 저장소 루트다.
    """
    return Path(rel_base) / rel_path


def image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return int(im.width), int(im.height)


def derive_capabilities(records: Sequence[ImageRecord]) -> Capabilities:
    """전역 플래그는 행 수준의 집계여야 한다 — 두 층위가 어긋나면 검증기가 실패시킨다."""
    n = len(records)
    n_thick = sum(1 for r in records if r.thickness_mm is not None)
    n_scale = sum(1 for r in records if r.px_per_mm is not None)
    n_quality = sum(1 for r in records if r.quality_level is not None)
    n_loc = sum(1 for r in records if r.has_localization)

    localization = n > 0 and n_loc == n
    has_thick = n > 0 and (n_thick / n) >= COVERAGE_THRESHOLD
    has_scale = n > 0 and (n_scale / n) >= COVERAGE_THRESHOLD
    # 안전 기본값은 clause_only 다. 가정값 채택(conditional)은 CTO 승인 사항이므로
    # ingest 가 스스로 conditional 로 올리지 않는다.
    mode = VerdictMode.ABSOLUTE if (has_thick and has_scale) else VerdictMode.CLAUSE_ONLY
    return Capabilities(
        localization=localization,
        thickness_mm=has_thick,
        pixel_scale=has_scale,
        verdict_mode=mode,
        counts={
            "images_total": n,
            "with_thickness": n_thick,
            "with_pixel_scale": n_scale,
            "with_quality_level": n_quality,
        },
    )


def drop_exact_duplicates(
    records: Sequence[ImageRecord],
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """바이트 동일(sha256 일치) 이미지를 하나만 남긴다. (유지분, 폐기분) 을 돌려준다.

    묶음(E1 엣지)이 이미 **누수**는 막는다 — 같은 바이트는 반드시 한 묶음이라 학습·평가로
    갈리지 않는다. 그런데 묶음은 **개수**를 고치지 않는다. 복제본을 그대로 두면

    - 표본 수가 부풀고(파일 수 ≠ 고유 이미지 수) 논문의 규모 서술이 틀리며,
    - 복제된 이미지만 학습에서 2회 노출되어 클래스 분포가 조용히 기울고,
    - `R × E = N` 학습량 등가 계산의 N 이 실제 고유 표본과 어긋난다.

    그래서 누수 방어(묶음)와 별개로 ingest 에서 한 번 걷어낸다.
    남기는 쪽은 `rel_path` 사전순 첫 번째다 — 입력 순서에 의존하지 않는 결정론적 규칙이다.
    """
    seen: dict[str, ImageRecord] = {}
    kept: list[ImageRecord] = []
    dropped: list[ImageRecord] = []
    for rec in sorted(records, key=lambda r: r.rel_path):
        if rec.sha256 in seen:
            dropped.append(rec)
        else:
            seen[rec.sha256] = rec
            kept.append(rec)
    return kept, dropped


def records_to_frames(
    records: Sequence[ImageRecord], label_map: LabelMap
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ImageRecord → (manifest, annotations). **컬럼 직렬화의 유일한 지점이다.**

    묶음(group_*)·층화(strata_key)·분할(split/client/eval_subset) 컬럼은 여기서 비워 두고
    dedup·split 단계가 채운다. 순서가 뒤바뀌면 안 되므로 후속 단계가 없으면 검증기가 잡는다.
    """
    m_rows: list[dict[str, Any]] = []
    a_rows: list[dict[str, Any]] = []

    for rec in records:
        if rec.reject_reason:
            continue
        types = sorted({d.defect_type for d in rec.defects})
        codes = sorted({d.iso_code for d in rec.defects})
        raws = sorted({d.src_label_raw for d in rec.defects})
        m_rows.append(
            {
                "image_id": rec.image_id,
                "source": rec.source,
                "rel_path": rec.rel_path,
                "sha256": rec.sha256,
                "width_px": rec.width_px,
                "height_px": rec.height_px,
                "modality": rec.modality,
                "material": rec.material,
                "has_defect": bool(rec.defects),
                "n_defects": len(rec.defects),
                "defect_types": ";".join(types),
                "iso_codes": ";".join(codes),
                "src_labels_raw": ";".join(raws),
                "label_type": rec.label_type,
                "has_localization": rec.has_localization,
                "phash_hex": pd.NA,
                "group_id": pd.NA,
                "group_size": pd.NA,
                "strata_key": pd.NA,
                "split": pd.NA,
                "client": pd.NA,
                "eval_subset": pd.NA,
                "thickness_mm": rec.thickness_mm,
                "thickness_source": rec.thickness_source,
                "px_per_mm": rec.px_per_mm,
                "scale_source": rec.scale_source,
                "quality_level": rec.quality_level,
                "ingest_version": rec.ingest_version,
                "label_map_version": label_map.version,
                "notes": rec.notes,
            }
        )
        for d in rec.defects:
            bbox = d.bbox_px or (None, None, None, None)
            a_rows.append(
                {
                    "ann_id": f"{rec.image_id}#{d.seq}",
                    "image_id": rec.image_id,
                    "src_label_raw": d.src_label_raw,
                    "defect_type": d.defect_type,
                    "iso_code": d.iso_code,
                    "polygon_json": (
                        json.dumps([[x, y] for x, y in d.polygon], separators=(",", ":"))
                        if d.polygon
                        else pd.NA
                    ),
                    "bbox_x1_px": bbox[0], "bbox_y1_px": bbox[1],
                    "bbox_x2_px": bbox[2], "bbox_y2_px": bbox[3],
                    "area_px": d.area_px,
                    "major_axis_px": d.major_axis_px,
                    "minor_axis_px": d.minor_axis_px,
                    "equiv_diameter_px": d.equiv_diameter_px,
                    "major_axis_mm": d.major_axis_mm,
                    "equiv_diameter_mm": d.equiv_diameter_mm,
                    "geom_valid": d.geom_valid,
                    "geom_flags": d.geom_flags,
                }
            )

    manifest = pd.DataFrame(m_rows, columns=list(MANIFEST_COLUMNS))
    annotations = pd.DataFrame(a_rows, columns=list(ANNOTATION_COLUMNS))
    return manifest, annotations

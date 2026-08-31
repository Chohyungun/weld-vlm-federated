"""계약 #2 — 매니페스트 스키마 + **유일하게 승인된 로더·조인·잠금 경로.**

CTO 게이트 #4 조건:
  1. `manifest.csv` 와 `annotations.csv` 를 하나의 `SNAPSHOT.sha256` 에 함께 잠근다.
  2. **조인은 이 모듈의 `join_defects()` 로만 한다.** 하류 트랙(C·D)이 각자 조인 로직을
     짜면 조인 규칙이 트랙마다 갈라진다. 로더가 계약의 일부다.

하류 트랙이 쓰는 것은 이 4개다.

    from data.manifest_io import load_snapshot, join_defects, MetricStatus

    snap = load_snapshot("data/mock/mock_aihub_v1")     # 해시 검증 포함
    snap.manifest      # 이미지 1장 = 1행
    snap.annotations   # 결함 인스턴스 1개 = 1행
    snap.capabilities  # 전역 런타임 플래그 (verdict_mode 등)
    joined = join_defects(snap)                          # 승인된 유일한 조인

`pandas.read_csv` 를 직접 부르지 마라. 빈 문자열이 NaN 으로 바뀌어 "정상 이미지"와
"결측"이 구별되지 않는다 (스펙 §2-3).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

# --------------------------------------------------------------------------------------
# 스키마 — 컬럼 순서가 계약이다. 바이트 재현성이 여기 걸려 있다 (스펙 §6-9).
# --------------------------------------------------------------------------------------

MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "source",
    "rel_path",
    "sha256",
    "width_px",
    "height_px",
    "modality",
    "material",
    "has_defect",
    "n_defects",
    "defect_types",
    "iso_codes",
    "src_labels_raw",
    "label_type",
    "has_localization",
    "phash_hex",
    "group_id",
    "group_size",
    "strata_key",
    "split",
    "client",
    "eval_subset",
    "thickness_mm",
    "thickness_source",
    "px_per_mm",
    "scale_source",
    "quality_level",
    "ingest_version",
    "label_map_version",
    "notes",
)

ANNOTATION_COLUMNS: tuple[str, ...] = (
    "ann_id",
    "image_id",
    "src_label_raw",
    "defect_type",
    "iso_code",
    "polygon_json",
    "bbox_x1_px",
    "bbox_y1_px",
    "bbox_x2_px",
    "bbox_y2_px",
    "area_px",
    "major_axis_px",
    "minor_axis_px",
    "equiv_diameter_px",
    "major_axis_mm",
    "equiv_diameter_mm",
    "geom_valid",
    "geom_flags",
)

#: 빈 문자열이 **유의미한 값**인 컬럼. 정상 이미지의 `defect_types` 는 "" 이고 NaN 이 아니다.
#: 이 목록 때문에 `keep_default_na=False` 로 읽어야 한다.
EMPTY_STRING_IS_MEANINGFUL: frozenset[str] = frozenset(
    {"defect_types", "iso_codes", "src_labels_raw", "geom_flags", "notes"}
)

_MANIFEST_STR = (
    "image_id", "source", "rel_path", "sha256", "modality", "material",
    "defect_types", "iso_codes", "src_labels_raw", "label_type", "phash_hex",
    "group_id", "strata_key", "split", "client", "eval_subset",
    "thickness_source", "scale_source", "quality_level", "ingest_version", "notes",
)
_MANIFEST_INT = ("width_px", "height_px", "n_defects", "group_size", "label_map_version")
_MANIFEST_BOOL = ("has_defect", "has_localization")
_MANIFEST_FLOAT = ("thickness_mm", "px_per_mm")

_ANN_STR = ("ann_id", "image_id", "src_label_raw", "defect_type", "iso_code",
            "polygon_json", "geom_flags")
_ANN_INT = ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px")
_ANN_BOOL = ("geom_valid",)
_ANN_FLOAT = ("area_px", "major_axis_px", "minor_axis_px", "equiv_diameter_px",
              "major_axis_mm", "equiv_diameter_mm")

#: `client` 이 null 인 것은 결측이 아니라 "평가셋이라 클라이언트가 없음"이다.
NULLABLE_STR_COLUMNS: frozenset[str] = frozenset({"client", "eval_subset", "quality_level"})

MANIFEST_FILENAME = "manifest.csv"
ANNOTATIONS_FILENAME = "annotations.csv"
CAPABILITIES_FILENAME = "data_capabilities.yaml"
SNAPSHOT_FILENAME = "SNAPSHOT.sha256"

#: 이미지별 출처(전처리 유래) 표. P9 교차 출처 오탐이 본실험 산출물로 승격되면서
#: 진행 로그(encode_progress.jsonl)가 아니라 **스냅샷 안**에 있어야 하는 값이 됐다.
#: 선택적 멤버다 — 있으면 SNAPSHOT 에 함께 잠기고, 없는 옛 스냅샷도 그대로 검증된다.
TILES_FILENAME = "tiles.csv"
TILES_COLUMNS: tuple[str, ...] = ("image_id", "provenance", "reason")

#: 타일링 reason → 출처. 결함/정상 구분 없이 전 이미지에 적용된다.
REASON_TO_PROVENANCE: dict[str, str] = {
    "ok": "N-crop",                       # 원래부터 1280×720 이던 이미지
    "tiled": "N-tile",                    # 파노라마에서 잘라낸 타일
    "oversized_band_cropped": "N-band",   # 밴드가 타일보다 커서 밴드 중심 크롭
}

#: 하나의 스냅샷에 함께 잠기는 파일 (게이트 조건 1). 순서 = SNAPSHOT.sha256 기록 순서.
SNAPSHOT_MEMBERS: tuple[str, ...] = (MANIFEST_FILENAME, ANNOTATIONS_FILENAME, CAPABILITIES_FILENAME)

FLOAT_FORMAT = "%.2f"


class VerdictMode(str, Enum):
    """판정 과업의 범위. 두께·스케일 메타데이터 유무로 결정된다 (스펙 §4-2)."""

    ABSOLUTE = "absolute"
    CONDITIONAL = "conditional"
    CLAUSE_ONLY = "clause_only"


class MetricStatus(str, Enum):
    """지표를 산출할 수 없을 때 결과표에 넣는 값. **0 으로 채우지 마라.**

    결측을 0 으로 채워 평균에 넣으면 다섯 칸 비교가 통째로 무의미해진다.
    """

    NO_LOCALIZATION = "N/A(no_localization)"
    NO_SCALE = "N/A(no_scale)"
    NOT_APPLICABLE = "N/A"


class ManifestError(ValueError):
    pass


class SnapshotVerificationError(ManifestError):
    """SNAPSHOT.sha256 과 실제 파일이 불일치. 실험 결과가 바뀐 것이다."""


# --------------------------------------------------------------------------------------
# 읽기
# --------------------------------------------------------------------------------------


def _read_csv(path: Path, columns: tuple[str, ...], strs, ints, bools, floats) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,   # ← 빈 문자열을 NaN 으로 바꾸지 않는다
        na_values=[],
        encoding="utf-8",
    )
    missing = set(columns) - set(df.columns)
    extra = set(df.columns) - set(columns)
    if missing or extra:
        raise ManifestError(f"{path}: 컬럼 불일치. 누락={sorted(missing)} 초과={sorted(extra)}")
    if tuple(df.columns) != columns:
        raise ManifestError(f"{path}: 컬럼 순서가 계약과 다르다. 바이트 재현성이 깨진다")

    out = df.copy()
    for col in strs:
        series = df[col].astype("string")
        # 빈 문자열이 유의미한 컬럼("정상"은 defect_types="")은 그대로 두고,
        # nullable 컬럼(평가셋이라 client 가 없음 등)만 NA 로 바꾼다.
        if col in NULLABLE_STR_COLUMNS:
            series = series.replace("", pd.NA)
        out[col] = series
    for col in ints:
        out[col] = pd.to_numeric(df[col].replace("", pd.NA), errors="raise").astype("Int64")
    for col in floats:
        out[col] = pd.to_numeric(df[col].replace("", pd.NA), errors="raise").astype("Float64")
    for col in bools:
        mapped = df[col].map({"True": True, "False": False, "true": True, "false": False})
        if mapped.isna().any():
            bad = sorted(set(df.loc[mapped.isna(), col]))
            raise ManifestError(f"{path}: {col} 에 bool 로 읽을 수 없는 값 {bad}")
        out[col] = mapped.astype("boolean")
    return out[list(columns)]


def read_manifest(path: Path | str) -> pd.DataFrame:
    return _read_csv(
        Path(path), MANIFEST_COLUMNS, _MANIFEST_STR, _MANIFEST_INT, _MANIFEST_BOOL, _MANIFEST_FLOAT
    )


def read_annotations(path: Path | str) -> pd.DataFrame:
    return _read_csv(
        Path(path), ANNOTATION_COLUMNS, _ANN_STR, _ANN_INT, _ANN_BOOL, _ANN_FLOAT
    )


def read_tiles(path: Path | str) -> pd.DataFrame:
    """이미지별 출처 표. 컬럼·순서는 계약이다."""
    df = _read_csv(Path(path), TILES_COLUMNS, TILES_COLUMNS, (), (), ())
    bad = sorted(set(df["provenance"]) - set(REASON_TO_PROVENANCE.values()))
    if bad:
        raise ManifestError(f"{path}: 모르는 출처 값 {bad}")
    return df


def read_capabilities(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass(frozen=True)
class Snapshot:
    """잠긴 스냅샷 하나. 세 파일이 항상 함께 온다."""

    root: Path
    snapshot_id: str
    manifest: pd.DataFrame
    annotations: pd.DataFrame
    capabilities: dict[str, Any]
    #: 이미지별 출처. 동결 스냅샷에만 있고 옛 스냅샷은 None 이다.
    tiles: pd.DataFrame | None = None

    @property
    def verdict_mode(self) -> VerdictMode:
        return VerdictMode(self.capabilities["capabilities"]["verdict_mode"])

    @property
    def has_localization(self) -> bool:
        return bool(self.capabilities["capabilities"]["localization"])

    def can_score(self, metric: Literal["map", "bbox_iou", "verdict_consistency"]) -> bool:
        """이 스냅샷으로 해당 지표를 산출할 수 있는가. False 면 MetricStatus 를 기록한다."""
        if metric in ("map", "bbox_iou"):
            return self.has_localization
        return self.verdict_mode is not VerdictMode.CLAUSE_ONLY


def load_snapshot(root: Path | str, *, verify: bool = True) -> Snapshot:
    """스냅샷 디렉터리를 읽는다. 기본으로 해시를 검증한다.

    `verify=False` 는 스냅샷 생성 중간 단계에서만 쓴다. 학습·평가 코드는 절대 끄지 않는다.
    """
    root = Path(root)
    if verify:
        verify_snapshot(root)
    caps = read_capabilities(root / CAPABILITIES_FILENAME)
    tiles_path = root / TILES_FILENAME
    return Snapshot(
        root=root,
        snapshot_id=caps["snapshot_id"],
        manifest=read_manifest(root / MANIFEST_FILENAME),
        annotations=read_annotations(root / ANNOTATIONS_FILENAME),
        capabilities=caps,
        tiles=read_tiles(tiles_path) if tiles_path.exists() else None,
    )


# --------------------------------------------------------------------------------------
# 조인 — 승인된 유일한 경로 (게이트 조건 2)
# --------------------------------------------------------------------------------------

#: 조인 결과에 붙는 행 종류. 결측 3종(N1/N2/N3)을 명시적으로 갈라준다.
ROW_KIND_DEFECT = "defect"
ROW_KIND_NORMAL = "normal"


def join_defects(
    snapshot: Snapshot | None = None,
    *,
    manifest: pd.DataFrame | None = None,
    annotations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """manifest × annotations 를 결함 인스턴스 단위 long 프레임으로 조인한다.

    **하류 트랙은 이 함수만 쓴다. 직접 merge 하지 마라.** 조인 규칙:

    - `manifest.image_id` LEFT JOIN `annotations.image_id`
    - 결함 0개인 이미지도 **1행 남는다** (annotation 컬럼 전부 null, `row_kind="normal"`)
    - 반환 프레임에 `row_kind` 컬럼이 추가된다. 결측을 `ann_id.isna()` 로 판별하지 말고
      아래 표대로 판별하라.

    | 결측 종류 | 판별 |
    |---|---|
    | N1 위치 라벨 없음 (RIAWELC) | `has_localization == False` |
    | N2 결함 없음 (정상 이미지)   | `row_kind == "normal"` |
    | N3 mm 환산 불가             | `px_per_mm.isna()` |

    N1 과 N2 를 섞으면 mAP 가 인위적으로 낮아진다. N1 은 지표에서 **제외**하고,
    N2 는 정답이 공집합인 정상 표본으로 **포함**한다.
    """
    if snapshot is not None:
        manifest, annotations = snapshot.manifest, snapshot.annotations
    if manifest is None or annotations is None:
        raise TypeError("snapshot 또는 (manifest, annotations) 를 넘겨라")

    overlap = (set(manifest.columns) & set(annotations.columns)) - {"image_id"}
    if overlap:
        raise ManifestError(f"컬럼 이름이 두 파일에 겹친다: {sorted(overlap)}")

    joined = manifest.merge(annotations, on="image_id", how="left", validate="one_to_many")
    joined["row_kind"] = joined["has_defect"].map(
        {True: ROW_KIND_DEFECT, False: ROW_KIND_NORMAL}
    ).astype("string")
    return joined


def defect_free_images(manifest: pd.DataFrame) -> pd.DataFrame:
    """정상 이미지(N2)만. 검출 학습에서 "박스 없음" 정답으로 쓴다."""
    return manifest.loc[~manifest["has_defect"].fillna(False)]


def localizable(manifest: pd.DataFrame) -> pd.DataFrame:
    """위치 지표를 산출할 수 있는 행만(N1 제외)."""
    return manifest.loc[manifest["has_localization"].fillna(False)]


def split_view(manifest: pd.DataFrame, split: str, client: str | None = None) -> pd.DataFrame:
    """split/client 로 자른 뷰. 문자열 비교를 하류에 흩뿌리지 않기 위한 단일 창구."""
    view = manifest.loc[manifest["split"] == split]
    if client is not None:
        view = view.loc[view["client"] == client]
    return view


# --------------------------------------------------------------------------------------
# 쓰기 · 잠금
# --------------------------------------------------------------------------------------


def _canonical_csv(df: pd.DataFrame, path: Path, columns: tuple[str, ...], sort_by: str) -> None:
    """바이트 재현 가능한 CSV 로 쓴다 (스펙 §6-9).

    UTF-8 BOM 없음 / 개행 LF 고정 / 컬럼 순서 고정 / 행 정렬 고정 / 결측은 빈 문자열 /
    소수 2자리 고정. win32 의 CRLF 자동 변환이 여기서 재현성을 깨뜨리므로 명시한다.
    """
    out = df.loc[:, list(columns)].sort_values(sort_by, kind="stable").reset_index(drop=True)
    for col in out.columns:
        if str(out[col].dtype) == "boolean":
            out[col] = out[col].map({True: "True", False: "False"}).astype("string")
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        out.to_csv(fh, index=False, na_rep="", float_format=FLOAT_FORMAT, lineterminator="\n")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_snapshot(
    root: Path | str,
    manifest: pd.DataFrame,
    annotations: pd.DataFrame,
    capabilities: dict[str, Any],
    tiles: pd.DataFrame | None = None,
) -> str:
    """구성 파일을 쓰고 **하나의 SNAPSHOT.sha256 에 함께 잠근다** (게이트 조건 1).

    `tiles` 를 주면 출처 표도 함께 잠긴다 (동결 스냅샷). 반환값은 스냅샷
    다이제스트(파일 해시들의 해시)다. 논문에 싣는 값이 이것이다.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    _canonical_csv(manifest, root / MANIFEST_FILENAME, MANIFEST_COLUMNS, "sha256")
    _canonical_csv(annotations, root / ANNOTATIONS_FILENAME, ANNOTATION_COLUMNS, "ann_id")
    with (root / CAPABILITIES_FILENAME).open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(capabilities, fh, allow_unicode=True, sort_keys=False)

    members = list(SNAPSHOT_MEMBERS)
    if tiles is not None:
        _canonical_csv(tiles, root / TILES_FILENAME, TILES_COLUMNS, "image_id")
        members.append(TILES_FILENAME)

    lines = [f"{file_sha256(root / name)}  {name}" for name in members]
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    with (root / SNAPSHOT_FILENAME).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
        fh.write(f"# snapshot_digest {digest}\n")
    return digest


def verify_snapshot(root: Path | str) -> str:
    """SNAPSHOT.sha256 대조. 불일치면 예외.

    잠금은 OS 읽기 전용 속성이 아니라 이 검증이다 (열린질문 Q7 확정). worktree·정션
    환경에서 읽기 전용 속성은 도구마다 다르게 동작해 오히려 사고를 낸다.
    """
    root = Path(root)
    snap = root / SNAPSHOT_FILENAME
    if not snap.exists():
        raise SnapshotVerificationError(f"{snap} 이 없다. 잠기지 않은 스냅샷은 쓰지 않는다")

    recorded: dict[str, str] = {}
    recorded_digest: str | None = None
    for line in snap.read_text(encoding="utf-8").splitlines():
        if line.startswith("# snapshot_digest "):
            recorded_digest = line.split()[-1]
            continue
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        recorded[name.strip()] = digest

    # 필수 멤버는 전부 있어야 하고, 허용된 선택 멤버(tiles.csv) 외의 것은 올 수 없다.
    required, optional = set(SNAPSHOT_MEMBERS), {TILES_FILENAME}
    if not required <= set(recorded) or set(recorded) - required - optional:
        raise SnapshotVerificationError(
            f"{snap}: 잠긴 파일 목록이 계약과 다르다. 기록={sorted(recorded)} "
            f"계약={sorted(required)} (+선택 {sorted(optional)}). 파일은 함께 잠겨야 한다"
        )
    members = list(SNAPSHOT_MEMBERS) + [n for n in (TILES_FILENAME,) if n in recorded]

    for name in members:
        actual = file_sha256(root / name)
        if actual != recorded[name]:
            raise SnapshotVerificationError(
                f"{root / name} 의 해시가 다르다. 기록={recorded[name][:12]} 실제={actual[:12]}. "
                "매니페스트가 바뀌었다면 실험 결과가 바뀐 것이다"
            )

    lines = [f"{recorded[name]}  {name}" for name in members]
    digest = hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()
    if recorded_digest is not None and recorded_digest != digest:
        raise SnapshotVerificationError("snapshot_digest 가 파일별 해시와 정합하지 않다")
    return digest

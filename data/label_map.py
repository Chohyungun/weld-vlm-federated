"""계약 #1 로더 — `configs/label_map.yaml`.

라벨 문자열을 코드에 하드코딩하지 않는다(docs/개발규약.md 불변조건 1-8). 모든 트랙이 이 모듈로
사상표를 읽는다. 직접 yaml.safe_load 하지 말 것 — 검증이 빠진다.

    from data.label_map import load_label_map
    lm = load_label_map()
    lm.to_defect_type("aihub71761", "기공")   # -> "porosity"
    lm.iso_code("porosity")                    # -> "2011"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABEL_MAP = REPO_ROOT / "configs" / "label_map.yaml"


class LabelMapError(ValueError):
    """사상표 자체가 잘못된 경우. 조용히 넘기지 않는다."""


class UnmappedLabelError(LabelMapError):
    """원본 라벨이 사상표에 없는 경우 (unmapped_policy: fail)."""


@dataclass(frozen=True)
class DefectType:
    key: str
    iso_code: str
    iso_code_alt: tuple[str, ...]
    name_ko: str
    name_en: str
    train_class_id: int


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label_field: str | None
    normal_labels: frozenset[str]
    mapping: dict[str, str]
    unmapped_policy: str


@dataclass(frozen=True)
class EvalSpace:
    key: str
    defect_types: tuple[str, ...]
    excluded: tuple[str, ...]
    reason: str


class LabelMap:
    """검증을 통과한 사상표. 불변 객체로 다룬다."""

    def __init__(self, raw: dict[str, Any], path: Path) -> None:
        self.path = path
        self.version: int = raw["version"]
        self.iso_standard: str = raw["iso_standard"]
        self.verdict_levels: tuple[str, ...] = tuple(raw["verdict_levels"])

        self.defect_types: dict[str, DefectType] = {
            key: DefectType(
                key=key,
                iso_code=str(spec["iso_code"]),
                iso_code_alt=tuple(str(c) for c in spec.get("iso_code_alt") or ()),
                name_ko=spec["name_ko"],
                name_en=spec["name_en"],
                train_class_id=int(spec["train_class_id"]),
            )
            for key, spec in raw["defect_types"].items()
        }
        self.sources: dict[str, SourceSpec] = {
            key: SourceSpec(
                key=key,
                label_field=spec.get("label_field"),
                normal_labels=frozenset(spec.get("normal_labels") or ()),
                mapping=dict(spec.get("mapping") or {}),
                unmapped_policy=spec.get("unmapped_policy", "fail"),
            )
            for key, spec in raw["sources"].items()
        }
        self.eval_spaces: dict[str, EvalSpace] = {
            key: EvalSpace(
                key=key,
                defect_types=tuple(spec["defect_types"]),
                excluded=tuple(spec.get("excluded") or ()),
                reason=spec.get("reason", ""),
            )
            for key, spec in raw["eval_spaces"].items()
        }
        self._validate()

    # ---- 조회 ----------------------------------------------------------------

    def iso_code(self, defect_type: str) -> str:
        return self.defect_types[defect_type].iso_code

    def train_class_id(self, defect_type: str) -> int:
        return self.defect_types[defect_type].train_class_id

    def iso_codes(self, *, include_alt: bool = True) -> frozenset[str]:
        """사상표가 인정하는 ISO 6520-1 코드 전체 집합.

        `limits.csv` 로더 게이트 V1(`defect_code ∈ 사상표`)이 참조하는 접근자다.
        소비하는 쪽이 `{d.iso_code for d in defect_types.values()}` 를 각자 재구성하면
        **별칭 코드(2012 등)를 포함할지가 갈린다.** 그 판단은 이 계약에서 한 번만 한다.

        `include_alt=True` 가 기본이다 — 기공의 2012(균일 분포 기공)는 사상표가 인정하는
        코드이고, 어느 자산 경로에서 쓰는지는 소비 트랙의 결정 사항이지 유효성 문제가 아니다.
        """
        codes: set[str] = set()
        for dt in self.defect_types.values():
            codes.add(dt.iso_code)
            if include_alt:
                codes.update(dt.iso_code_alt)
        return frozenset(codes)

    def defect_type_for_code(self, iso_code: str) -> str:
        """ISO 코드 → L2 키. 별칭 코드도 대표 유형으로 해석한다."""
        for key, dt in self.defect_types.items():
            if iso_code == dt.iso_code or iso_code in dt.iso_code_alt:
                return key
        raise LabelMapError(
            f"ISO 코드 {iso_code!r} 가 {self.path} 에 없다. 사상표 밖 결함은 "
            "limits.csv 에 넣지 않고 out_of_scope 로 기록한다"
        )

    def is_normal_label(self, source: str, raw_label: str) -> bool:
        return raw_label in self.sources[source].normal_labels

    def to_defect_type(self, source: str, raw_label: str) -> str | None:
        """원본 라벨 → L2 키. 정상 라벨이면 None.

        사상표에 없으면 unmapped_policy 에 따라 예외를 던진다. 경고로 넘기지 않는다 —
        조용히 스킵하면 라벨이 사라진 것을 아무도 모른다.
        """
        spec = self.sources[source]
        if raw_label in spec.normal_labels:
            return None
        if raw_label in spec.mapping:
            return spec.mapping[raw_label]
        if spec.unmapped_policy == "fail":
            raise UnmappedLabelError(
                f"source={source!r} 의 원본 라벨 {raw_label!r} 가 {self.path} 에 없다. "
                f"사상표를 고치거나 데이터를 확인하라 (unmapped_policy=fail)."
            )
        return None

    def dense_ids(self, eval_space: str) -> dict[str, int]:
        """축소된 라벨 공간 안에서만 쓰는 조밀 ID.

        `train_class_id` 를 재부여하지 않는다 — 원래 ID 는 그대로 두고 매핑을 반환해
        호출부가 로깅할 수 있게 한다 (스펙 §1-3-2).
        """
        keys = sorted(
            self.eval_spaces[eval_space].defect_types,
            key=lambda k: self.defect_types[k].train_class_id,
        )
        return {key: i for i, key in enumerate(keys)}

    # ---- 검증 ----------------------------------------------------------------

    def _validate(self) -> None:
        if not self.defect_types:
            raise LabelMapError("defect_types 가 비어 있다")

        # L1/L2 계층 분리 — 정상이 결함 유형에 섞여서는 안 된다
        for level in self.verdict_levels:
            if level in self.defect_types:
                raise LabelMapError(
                    f"판정 수준 {level!r} 가 defect_types 에 있다. L1/L2 계층 분리 위반"
                )

        for key, dt in self.defect_types.items():
            if not key.replace("_", "").isalnum() or not key.isascii() or key != key.lower():
                raise LabelMapError(f"defect_types 키 {key!r} 는 소문자 ASCII snake_case 여야 한다")
            if not dt.iso_code.isdigit():
                raise LabelMapError(f"{key}: iso_code {dt.iso_code!r} 가 숫자가 아니다")

        ids = sorted(dt.train_class_id for dt in self.defect_types.values())
        if ids != list(range(len(ids))):
            raise LabelMapError(f"train_class_id 가 0..N-1 조밀·유일이 아니다: {ids}")

        # 대표 코드와 별칭이 서로 충돌하지 않아야 한다
        seen: dict[str, str] = {}
        for key, dt in self.defect_types.items():
            for code in (dt.iso_code, *dt.iso_code_alt):
                if code in seen and seen[code] != key:
                    raise LabelMapError(
                        f"ISO 코드 {code!r} 가 {seen[code]!r} 와 {key!r} 에 중복 배정됐다"
                    )
                seen[code] = key

        for src, spec in self.sources.items():
            if spec.unmapped_policy not in {"fail", "skip"}:
                raise LabelMapError(f"{src}: unmapped_policy {spec.unmapped_policy!r} 는 정의되지 않았다")
            for raw, target in spec.mapping.items():
                if target not in self.defect_types:
                    raise LabelMapError(f"{src}: {raw!r} → {target!r} 인데 {target!r} 가 L2 에 없다")
            overlap = spec.normal_labels & set(spec.mapping)
            if overlap:
                raise LabelMapError(f"{src}: {sorted(overlap)} 가 정상 라벨과 결함 사상에 동시에 있다")

        for name, space in self.eval_spaces.items():
            unknown = set(space.defect_types) - set(self.defect_types)
            if unknown:
                raise LabelMapError(f"eval_spaces.{name}: L2 에 없는 키 {sorted(unknown)}")
            both = set(space.defect_types) & set(space.excluded)
            if both:
                raise LabelMapError(f"eval_spaces.{name}: {sorted(both)} 가 포함·제외에 동시에 있다")
            covered = set(space.defect_types) | set(space.excluded)
            missing = set(self.defect_types) - covered
            if missing:
                raise LabelMapError(
                    f"eval_spaces.{name}: {sorted(missing)} 의 포함/제외가 미결정이다. "
                    "라벨 공간 축소는 명시적이어야 한다"
                )


def load_label_map(path: Path | str | None = None) -> LabelMap:
    p = Path(path) if path is not None else DEFAULT_LABEL_MAP
    with p.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return LabelMap(raw, p)

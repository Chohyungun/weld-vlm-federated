"""정답 조항 목록 — 계약 #3의 세 소비처 중 두 번째. 스펙 §6.

`corpus/derived/gold_clauses.csv`(B의 `derive_gold_clauses` 산출물)를 읽어 이미지별
정답 쌍으로 사영한다. **D가 `limits.csv` 를 직접 순회해 만들지 않는다**(B 스펙 §1-4
금지 규칙 ④). 파생물을 직접 수정하지도 않는다 — 전문가가 조항 오류를 찾으면 원천을
고치고 전체를 다시 파생한다.

**조회 키에 `inspection_method` 가 들어간다**(게이트 #13 결정 L). 축 없이 키를 만들면
표면 기준 조항이 내부 검사 정답으로 등재되고, 그러면 채점기가 잘못된 정답과 대조하면서도
형식상 정상으로 보인다.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from evaluation.metrics.clause import GoldPair

METHOD_ANY = "ALL"

KEY_FIELDS = (
    "inspection_method",
    "defect_code",
    "material",
    "quality_scheme",
    "quality_level",
    "limit_type",
)
"""유일성 키의 비-구간 축. 두께는 구간이라 별도로 처리한다.
검사축이 **첫 축**인 것이 계약 #3 변경의 핵심이다."""


class GoldLookupError(LookupError):
    """정답 조항을 특정할 수 없다. fallback 하지 않는다 — 조용한 오답이 더 위험하다."""


@dataclass(frozen=True)
class GoldEntry:
    """`derive_gold_clauses()` 한 행."""

    defect_code: str
    material: str
    inspection_method: str
    thickness_min: Decimal | None
    thickness_max: Decimal | None
    quality_scheme: str
    quality_level: str
    limit_type: str
    clause_id: str
    rule_id: str

    def covers_thickness(self, t: Decimal | None) -> bool:
        """반구간 `[min, max)`. 양끝 포함으로 두면 경계 두께에서 두 행이 걸린다."""
        if self.thickness_min is None and self.thickness_max is None:
            return True
        if t is None:
            return False
        if self.thickness_min is not None and t < self.thickness_min:
            return False
        return not (self.thickness_max is not None and t >= self.thickness_max)

    def matches_method(self, method: str) -> bool:
        return self.inspection_method in (method, METHOD_ANY)


def _dec(v: object) -> Decimal | None:
    return None if v is None or v == "" else Decimal(str(v))


def read_derived_csv(path: str | Path) -> tuple[dict[str, str], ...]:
    """B 파생물 CSV 를 읽는다. **선두 `#` 주석 줄을 건너뛴다.**

    `gold_clauses.csv` 는 격리 경고·원천 sha256·재파생 명령을 `#` 주석 3줄로 이고 있다
    (62번). 그대로 `DictReader` 에 넣으면 첫 주석이 헤더가 되어 컬럼이 통째로 어긋난다.
    """
    text = Path(path).read_text(encoding="utf-8")
    body = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return tuple(csv.DictReader(body))


def entries_from_derived(rows: Iterable[Mapping[str, object]]) -> tuple[GoldEntry, ...]:
    """B의 `derive_gold_clauses()` 출력을 그대로 받는다. 이 변환이 유일한 진입 지점이다."""
    return tuple(
        GoldEntry(
            defect_code=str(r["defect_code"]),
            material=str(r["material"]),
            inspection_method=str(r["inspection_method"]),
            thickness_min=_dec(r.get("thickness_min")),
            thickness_max=_dec(r.get("thickness_max")),
            quality_scheme=str(r["quality_scheme"]),
            quality_level=str(r["quality_level"]),
            limit_type=str(r["limit_type"]),
            clause_id=str(r["clause_id"]),
            rule_id=str(r["rule_id"]),
        )
        for r in rows
    )


def assert_unique(entries: Sequence[GoldEntry]) -> None:
    """유일성 검사 — 위반 시 **빌드 실패 + B 회부**. 경고로 넘기지 않는다.

    누가 언제 검사하는지가 없으면 "조용한 다대다"가 그대로 통과한다(트랙 C의 R-재현
    Minor #2). 검사축이 키에 들어간 것이 핵심이다 — 축이 없으면 표면·내부 기공 행이
    같은 키로 보여 중복 검사를 통과해 버린다.
    """
    seen: dict[tuple, str] = {}
    for e in entries:
        key = (
            e.inspection_method, e.defect_code, e.material,
            str(e.thickness_min), str(e.thickness_max),
            e.quality_scheme, e.quality_level, e.limit_type,
        )
        prev = seen.get(key)
        if prev is not None and prev != e.clause_id:
            raise ValueError(
                f"정답 조항 목록에 같은 키의 조항이 둘이다: {prev} / {e.clause_id} "
                f"(키={key}) — 원천 limits.csv 의 canonical 지정을 고쳐 재파생한다"
            )
        seen[key] = e.clause_id


def lookup(
    entries: Sequence[GoldEntry],
    *,
    defect_code: str,
    material: str,
    inspection_method: str,
    quality_scheme: str,
    quality_level: str,
    thickness_mm: Decimal | None,
    limit_type: str | None = None,
) -> GoldEntry:
    """조합으로 정답 조항 1건을 특정한다. **fallback 금지.**"""
    hits = [
        e for e in entries
        if e.defect_code == defect_code
        and e.material in (material, METHOD_ANY)
        and e.matches_method(inspection_method)
        and e.quality_scheme == quality_scheme
        and e.quality_level == quality_level
        and (limit_type is None or e.limit_type == limit_type)
        and e.covers_thickness(thickness_mm)
    ]
    key = (
        f"defect={defect_code}, material={material}, method={inspection_method}, "
        f"scheme={quality_scheme}, level={quality_level}, t={thickness_mm}"
    )
    if not hits:
        raise GoldLookupError(f"정답 조항 없음 ({key})")
    clause_ids = {h.clause_id for h in hits}
    if len(clause_ids) > 1:
        raise GoldLookupError(
            f"정답 조항 복수 {sorted(clause_ids)} — limit_type 을 명시한다 ({key})"
        )
    return hits[0]


@dataclass(frozen=True)
class ImageContext:
    """채점 대상 이미지 한 장의 조회 맥락. manifest 에서 온다."""

    image_id: str
    inspection_method: str
    """manifest `modality` (`RT`/`VT`). 주 실험은 RT 한정."""
    material: str
    thickness_mm: Decimal | None
    quality_scheme: str
    quality_level: str


def build_gold_pairs(
    entries: Sequence[GoldEntry],
    contexts: Iterable[ImageContext],
    gt_codes: Mapping[str, Sequence[str]],
) -> tuple[tuple[GoldPair, ...], dict[str, int]]:
    """(이미지 × 결함코드) 정답 쌍을 만든다.

    Returns:
        `(쌍들, 건너뛴 사유별 건수)`. 조회 실패를 예외로 던지지 않고 세는 이유는,
        한 조합이 비었다고 전체 채점이 멈추면 나머지 지표까지 못 내기 때문이다.
        대신 **건너뛴 건수를 반드시 보고**한다.
    """
    pairs: list[GoldPair] = []
    skipped: dict[str, int] = {}
    for ctx in contexts:
        for code in sorted(set(gt_codes.get(ctx.image_id, ()))):
            try:
                e = lookup(
                    entries,
                    defect_code=code,
                    material=ctx.material,
                    inspection_method=ctx.inspection_method,
                    quality_scheme=ctx.quality_scheme,
                    quality_level=ctx.quality_level,
                    thickness_mm=ctx.thickness_mm,
                )
            except GoldLookupError as exc:
                reason = "복수 매칭" if "복수" in str(exc) else "정답 조항 없음"
                skipped[reason] = skipped.get(reason, 0) + 1
                continue
            pairs.append(GoldPair(ctx.image_id, code, e.clause_id, row_id=e.rule_id))
    return tuple(pairs), skipped

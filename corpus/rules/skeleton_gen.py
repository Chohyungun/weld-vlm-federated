"""판정 골격 생성기 — 함정 구간 #6 (스펙 §4 전체).

스펙: 코퍼스 합성 스펙
§4-2 (함수 분해·결정론·fail-closed), §4-3 (조합 열거·수치 샘플링), §4-4 (verdict·margin·
널러빌리티), §4-5 (골격 JSON), §4-6 (D4 경로·조립), §4-6-1 (정상 페어), §4-7 (verdict_mode),
§4-9 (G2 사후 감사).

원칙:
- 전 함수 순수 (전역 상태·시각·I/O 없음, 사상표 로더 제외). 난수는 주입 rng 만.
- 수치는 전부 Decimal, 0.01 그리드 정수 인덱스에서 **직접 추출** (연속 샘플 후 반올림 금지).
- 난수 스트림은 조합 정체성 기반: SeedSequence((seed, combo_hash64)),
  combo_hash64 = sha256(조합키 문자열).digest()[:8] 빅엔디언 정수. 위치 spawn 금지.
- 판정·한계 산출은 corpus.rules.limit_eval 의 judge/effective_limit/aggregate_verdicts 만
  사용한다 — 재구현 금지 (§1-4).
- fail-closed 적용 범위 (§4-2): (c) 경로 내부 불변식 위반 = SkeletonGenError 로 전체 중단.
  D4 외부 입력 이상 = Quarantined 격리 배출 + 생성 계속.
- verdict_mode 게이팅 (§4-7): judge 계산 자체는 항상 수행하고 **직렬화 출력만** 게이트한다.
  (c) 경로는 항상 full.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields as _dc_fields
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping, Optional, Sequence, Union

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

# 결함 어휘 락의 정본은 numeric_lock 하나다 (§4-6-1 ④). 같은 이름의 헬퍼를 여기에도 두면
# 정규화 규칙이 두 벌이 되어 한쪽만 전각·NFD 표기를 놓친다 — 재수출만 한다 (재구현 금지).
from corpus.generate.numeric_lock import find_defect_tokens
from corpus.rules.limit_eval import (
    aggregate_verdicts,
    applicable_row,
    effective_limit,
    judge,
)
from corpus.rules.schema import (
    FAIL,
    PASS,
    InspectionMethod,
    LimitOp,
    LimitRow,
    LimitRule,
    LimitsTable,
    LimitType,
    Material,
    Q,
    QualityScheme,
    RatioBasis,
    Scope,
    Unit,
    Verdict,
    VerdictMode,
    quantize,
)

__all__ = [
    "GENERATOR_VERSION",
    "DEFAULT_INSPECTION_METHOD",
    "BUCKET_PASS", "BUCKET_FAIL", "BUCKET_BOUNDARY_LOW", "BUCKET_BOUNDARY_EQ",
    "BUCKET_BOUNDARY_HIGH", "STRATA", "BOUNDARY_STRATA",
    "GenConfig", "config_sha256", "FEASIBILITY_MIN_L",
    "SkeletonGenError", "Combo", "combo_key", "combo_hash64", "combo_hash8",
    "enumerate_combos", "a_basis_hold_rows", "allocate_counts", "decompose_counts",
    "bucket_interval", "sample_measurement", "thickness_floor_idx",
    "SkeletonFlags", "Provenance", "Skeleton", "build_skeleton",
    "generate_corpus_skeletons", "skeleton_from_json", "skeletons_jsonl", "skeletons_sha256",
    "D4Label", "D4Assumptions", "Quarantined", "skeleton_from_label", "skeletons_from_labels",
    "PairGold", "assemble_pair",
    "NormalSelection", "select_normal_images",
    "load_defect_lexicon", "find_defect_tokens", "normal_pair_violations",
    "AuditReport", "audit_skeletons",
]

GENERATOR_VERSION = "skeleton_gen/1.0.0"

# 검사 방법 축 기본값 (§1-2 5a). 주 실험이 RT 이므로 행 선택은 {RT, ALL} 로 좁힌다.
# 축은 두 겹으로 강제한다: 테이블을 LimitsTable.for_inspection 뷰로 좁히고,
# applicable_row 에도 같은 축을 넘긴다 (뷰 주입 규율 하나에 기대지 않는다).
DEFAULT_INSPECTION_METHOD = InspectionMethod.RT

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABEL_MAP_PATH = _REPO_ROOT / "configs" / "label_map.yaml"

# ---------------------------------------------------------------------------
# 버킷 (§4-3). 버킷은 "의도"이고 verdict 는 judge() 의 계산 결과다 — 역산 금지.
# ---------------------------------------------------------------------------

BUCKET_PASS = "pass"                    # (0, 0.9L)
BUCKET_FAIL = "fail"                    # (1.1L, 2.5L]
BUCKET_BOUNDARY_LOW = "boundary_low"    # [0.9L, L−0.01]
BUCKET_BOUNDARY_EQ = "boundary_eq"      # {L}
BUCKET_BOUNDARY_HIGH = "boundary_high"  # [L+0.01, 1.1L]

STRATA: tuple[str, ...] = (
    BUCKET_PASS, BUCKET_FAIL, BUCKET_BOUNDARY_LOW, BUCKET_BOUNDARY_EQ, BUCKET_BOUNDARY_HIGH,
)
BOUNDARY_STRATA: tuple[str, ...] = (
    BUCKET_BOUNDARY_LOW, BUCKET_BOUNDARY_EQ, BUCKET_BOUNDARY_HIGH,
)

# 분기 플래그 어휘 (§4-7). (c) 샘플링 경로의 두께·수준은 실이미지 출처가 아니라
# 합성 시나리오의 샘플값이므로 "sampled" 를 쓴다 — 스펙 enum {metadata, assumed, n/a} 는
# 실이미지 경로 정의라 (c) 확장값은 보고서에 "결정 필요"로 명시한다.
SOURCE_SAMPLED = "sampled"
SOURCE_METADATA = "metadata"
SOURCE_ASSUMED = "assumed"
SOURCE_NA = "n/a"

VERDICT_TYPE_CONFIRMED = "확정"
VERDICT_TYPE_CONDITIONAL = "조건부"

# quarantine 사유 코드 (§4-2 fail-closed 범위, §4-6, §4-6-1)
QUARANTINE_UNMAPPED_DEFECT = "unmapped_defect"
QUARANTINE_EXCLUDED_2012 = "excluded_2012"
QUARANTINE_DEGENERATE_POLYGON = "degenerate_polygon"
QUARANTINE_DEGENERATE_SIZE = "degenerate_size"
QUARANTINE_MISSING_THICKNESS = "missing_thickness"
QUARANTINE_MISSING_SCALE = "missing_scale"
QUARANTINE_NO_APPLICABLE_ROW = "no_applicable_row"
QUARANTINE_AMBIGUOUS_CONSTRAINT = "ambiguous_constraint"
QUARANTINE_RATIO_NOT_MEASURABLE = "ratio_not_measurable"
QUARANTINE_BASIS_A_HOLD = "basis_a_hold"

QUARANTINE_INVALID_NUMERIC_TYPE = "invalid_numeric_type"

DISCARD_DEFECT_HALLUCINATION = "defect_hallucination"

# V10 실현성 하한 (그리드 10칸)과 같은 값 — 로더의 보고 기준과 생성기의 t 하한 보정이
# 같은 상수를 봐야 "보고된 행이 정확히 보정 대상"이 된다.
FEASIBILITY_MIN_L = Decimal("0.1")


class SkeletonGenError(Exception):
    """(c) 경로 내부 불변식 위반 — 부분 산출 없이 전체 중단 (§4-2)."""


# ---------------------------------------------------------------------------
# 샘플링 상수 주입 (§4-3·열린 질문 12 — 코드 하드코딩 금지, configs 확정 전 스펙 기본값)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenConfig:
    """샘플링 상수. 확정값은 configs/corpus_gen.yaml 이 단일 소스가 되며 여기 기본값은
    스펙 수치다 (2.5L·40/40/20·8/4/8·상한 40·총량 1만). t_sample_max·F4 범위는 파일럿
    확정 전 안전 기본값."""

    total: int = 10_000                      # (c) 목표 총량 (§7-1)
    combo_cap: int = 40                      # 조합별 상한 (§4-3)
    ratio_pass: int = 40                     # 합격 40%
    ratio_fail: int = 40                     # 불합격 40%
    ratio_boundary_low: int = 8              # 경계 내부 8/4/8 (§4-3)
    ratio_boundary_eq: int = 4
    ratio_boundary_high: int = 8
    boundary_low_mult: Decimal = Decimal("0.9")
    boundary_high_mult: Decimal = Decimal("1.1")
    fail_upper_mult: Decimal = Decimal("2.5")   # 불합격 상한 2.5L (configs)
    t_sample_max: Decimal = Decimal("100.00")   # +∞ 구간의 t 샘플 상한 (configs)
    f4_size_min: Decimal = Decimal("0.10")      # F4 크기 범위 (configs, S1-10)
    f4_size_max: Decimal = Decimal("5.00")
    prop_edge_frac: Decimal = Decimal("0.1")    # prop 계열 t 경계 근방 배치 비율 (§4-3)
    percent_max: Decimal = Decimal("100.00")    # F5 상한 min(2.5L, 100%)
    d4_quality_scheme: str = "iso5817"          # D4 기준 품질수준 (열린 질문 4)
    d4_quality_level: str = "C"
    normal_ratio: Decimal = Decimal("0.5")      # 정상 = 결함 페어 수 × 1/2 (§4-6-1)
    verdict_skew_warn: Decimal = Decimal("0.65")  # G2 verdict 쏠림 경고 65:35
    bucket_tolerance_pp: Decimal = Decimal("2")   # G2 버킷 비율 ±2%p


def config_sha256(config: GenConfig) -> str:
    """GenConfig 의 정본 해시 — provenance.configs_sha256 에 각인된다."""
    payload = {f.name: str(getattr(config, f.name)) for f in _dc_fields(config)}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


_DEFAULT_CONFIG = GenConfig()


# ---------------------------------------------------------------------------
# 그리드 헬퍼 — 0.01 그리드 정수 인덱스 (§4-3 그리드 직접 추출)
# ---------------------------------------------------------------------------


def _idx(d: Decimal) -> int:
    """양자화된 Decimal → 그리드 정수 인덱스 (0.01 = 1칸)."""
    return int(quantize(d) / Q)


def _dec(idx: int) -> Decimal:
    return (Decimal(idx) * Q).quantize(Q)


def _ceil_i(d: Decimal) -> int:
    return int(d.to_integral_value(rounding=ROUND_CEILING))


def _floor_i(d: Decimal) -> int:
    return int(d.to_integral_value(rounding=ROUND_FLOOR))


# ---------------------------------------------------------------------------
# 조합 정체성 (§4-2) — 시드·sample_id 의 기준
# ---------------------------------------------------------------------------


class Combo(tuple):
    """실존 조합 1건. row 를 동반한 NamedTuple 대용 (hashable)."""

    __slots__ = ()

    def __new__(cls, key: str, hash64: int, hash8: str, row: LimitRow):
        return super().__new__(cls, (key, hash64, hash8, row))

    @property
    def key(self) -> str:
        return self[0]

    @property
    def hash64(self) -> int:
        return self[1]

    @property
    def hash8(self) -> str:
        return self[2]

    @property
    def row(self) -> LimitRow:
        return self[3]


def combo_key(row: LimitRow) -> str:
    """조합키 문자열 (§4-2): defect|t_min|t_max|scheme|level|clause_id.

    t 는 로드 시점 0.01 양자화 문자열 그대로, t_max 공란은 "+inf".
    """
    t_max = "+inf" if row.thickness_max is None else str(row.thickness_max)
    return (
        f"{row.defect_code}|{row.thickness_min}|{t_max}|"
        f"{row.quality_scheme.value}|{row.quality_level}|{row.clause_id}"
    )


def _key_digest(key: str) -> bytes:
    return hashlib.sha256(key.encode("utf-8")).digest()


def combo_hash64(key: str) -> int:
    """sha256(조합키).digest()[:8] 빅엔디언 정수 — SeedSequence 엔트로피."""
    return int.from_bytes(_key_digest(key)[:8], "big")


def combo_hash8(key: str) -> str:
    """sample_id 접두 8자리 (조합해시8, §4-5)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


def _inspection_view(
    table: LimitsTable, inspection_method: InspectionMethod | str | None
) -> LimitsTable:
    """검사 방법 축 행 선택 뷰 (§1-2 5a). None 은 "이미 좁혀진 테이블"이라는 내부 신호다 —
    공개 진입점(generate_corpus_skeletons)은 None 을 거부한다."""
    return table if inspection_method is None else table.for_inspection(inspection_method)


def _query_axis(
    explicit: InspectionMethod | str | None, row_or_sk: Any
) -> InspectionMethod:
    """applicable_row 에 넘길 질의 축. 명시 축이 있으면 그것이 우선한다.

    명시 축이 없으면(이미 좁혀진 테이블) 대상의 축을 쓰되, `ALL`(검사 방법과 무관한 조항)은
    질의 값이 될 수 없으므로 기본 축으로 조회한다 — ALL 행은 어느 축 질의에도 응하므로
    같은 행이 잡힌다. 같은 조합에 축이 다른 행이 함께 있으면 복수 매칭으로 시끄럽게 실패한다.
    """
    if explicit is not None:
        return InspectionMethod(explicit)
    m = row_or_sk.inspection_method
    return DEFAULT_INSPECTION_METHOD if m is InspectionMethod.ALL else m


def a_basis_hold_rows(
    table: LimitsTable,
    inspection_method: InspectionMethod | str | None = DEFAULT_INSPECTION_METHOD,
) -> tuple[LimitRow, ...]:
    """ratio_basis=a 행 — 평가 규약 미확정으로 보류 큐 (열린 질문 3). 열거에서 제외."""
    return tuple(
        r for r in _inspection_view(table, inspection_method).active_canonical
        if r.ratio_basis is RatioBasis.A
    )


def enumerate_combos(
    table: LimitsTable,
    inspection_method: InspectionMethod | str | None = DEFAULT_INSPECTION_METHOD,
) -> tuple[Combo, ...]:
    """실존 조합 열거 (§4-3). scope=active ∧ canonical 만, 외삽 금지, 정렬 고정.

    검사 방법 축(§1-2 5a)으로 먼저 {inspection_method, ALL} 만 남긴다 — 표면(VT) 기준과
    내부(RT) 기준이 같은 결함코드로 병존하므로, 축 없이 열거하면 주 실험(RT)의 corpus 에
    표면 기준 판정이 섞여 들어간다.

    ratio_basis=a 행은 보류(열린 질문 3)로 제외한다 — a_basis_hold_rows 로 건수 확인.
    조합키가 충돌하면(sample_id 유일성 붕괴) fail-closed 로 전체 거부한다.
    """
    rows = [
        r for r in _inspection_view(table, inspection_method).active_canonical
        if r.ratio_basis is not RatioBasis.A
    ]
    combos = [Combo(combo_key(r), combo_hash64(combo_key(r)), combo_hash8(combo_key(r)), r)
              for r in rows]
    combos.sort(key=lambda c: (c.key, c.row.material.value, c.row.limit_type.value, c.row.rule_id))
    seen: dict[str, str] = {}
    for c in combos:
        if c.key in seen:
            raise SkeletonGenError(
                f"[enumerate_combos] 조합키 충돌: {c.key!r} ← {seen[c.key]} / {c.row.rule_id} "
                "— sample_id 유일성 붕괴 (조합키에 material·limit_type·inspection_method "
                "미포함, 결정 필요)"
            )
        seen[c.key] = c.row.rule_id
    return tuple(combos)


# ---------------------------------------------------------------------------
# 수량 배분 (§4-3)
# ---------------------------------------------------------------------------


def allocate_counts(
    combos: Sequence[Combo],
    total: int = _DEFAULT_CONFIG.total,
    cap: int = _DEFAULT_CONFIG.combo_cap,
) -> dict[Combo, int]:
    """총량 균등 배분 + 조합별 상한. 잔여는 정렬순 라운드로빈.

    상한 미달로 총량에 못 미치면 상한을 올리지 않고 **총량을 줄여** 반환한다
    (sum(counts) = min(total, len(combos)×cap) — 축소 여부는 합계로 보고).
    """
    if total < 0 or cap <= 0:
        raise SkeletonGenError(f"[allocate_counts] total={total}, cap={cap} 비정상")
    ordered = list(combos)
    n = len(ordered)
    if n == 0:
        return {}
    eff_total = min(total, n * cap)
    base = eff_total // n
    counts = {c: base for c in ordered}
    rem = eff_total - base * n
    for c in ordered:
        if rem == 0:
            break
        if counts[c] < cap:
            counts[c] += 1
            rem -= 1
    return counts


def decompose_counts(
    n: int, limit_rule: LimitRule, config: GenConfig = _DEFAULT_CONFIG
) -> dict[str, int]:
    """조합 배분 n 건의 버킷 분해 (§4-3).

    - F4(none_permitted): 40/40/20 미적용, 전량 불합격.
    - 그 외: 40/40/8/4/8 을 n 전체에 최대잔여법으로 배분 (스펙의 "round(0.48)=0" 산정이
      전체 n 기준임을 따른다) 후, n≥3 이면 =L·하경계·상경계 각 1건 최소 보장을 강제한다
      (부족분은 표본이 가장 많은 층에서 가져온다 — 결정론 타이브레이크).
    - n<3: 우선순위 =L → 상경계 → 하경계.
    - 불변식: 층 합계 = n.
    """
    if n < 0:
        raise SkeletonGenError(f"[decompose_counts] n={n} < 0")
    counts = {s: 0 for s in STRATA}
    if n == 0:
        return counts
    if limit_rule is LimitRule.NONE_PERMITTED:
        counts[BUCKET_FAIL] = n
        return counts
    if n < 3:
        for s in (BUCKET_BOUNDARY_EQ, BUCKET_BOUNDARY_HIGH, BUCKET_BOUNDARY_LOW)[:n]:
            counts[s] = 1
        return counts

    weights = {
        BUCKET_PASS: config.ratio_pass,
        BUCKET_FAIL: config.ratio_fail,
        BUCKET_BOUNDARY_LOW: config.ratio_boundary_low,
        BUCKET_BOUNDARY_EQ: config.ratio_boundary_eq,
        BUCKET_BOUNDARY_HIGH: config.ratio_boundary_high,
    }
    total_w = sum(weights.values())
    quotas = {s: Decimal(n * w) / Decimal(total_w) for s, w in weights.items()}
    for s in STRATA:
        counts[s] = _floor_i(quotas[s])
    rem = n - sum(counts.values())
    frac_order = sorted(
        STRATA, key=lambda s: (-(quotas[s] - counts[s]), STRATA.index(s))
    )
    for s in frac_order[:rem]:
        counts[s] += 1

    # 최소 보장 (S1-3): =L·하·상 각 ≥ 1
    for target in (BUCKET_BOUNDARY_EQ, BUCKET_BOUNDARY_LOW, BUCKET_BOUNDARY_HIGH):
        if counts[target] >= 1:
            continue
        donors = [
            s for s in STRATA
            if s != target and counts[s] > (1 if s in BOUNDARY_STRATA else 0)
        ]
        donors.sort(key=lambda s: (-counts[s], STRATA.index(s)))
        if not donors:
            raise SkeletonGenError(f"[decompose_counts] n={n}: 최소 보장 불능")
        counts[donors[0]] -= 1
        counts[target] += 1

    assert sum(counts.values()) == n, "층 합계 = n 불변식 붕괴"
    return counts


# ---------------------------------------------------------------------------
# 버킷 구간·수치 샘플링 (§4-3 그리드 직접 추출)
# ---------------------------------------------------------------------------


def _basis_value(row: LimitRow, t: Decimal) -> Optional[Decimal]:
    """prop 계열의 기준 치수. s 는 완전용입 맞대기 가정 s=t (열린 질문 3), a 는 보류."""
    if row.limit_rule not in (LimitRule.PROP_T, LimitRule.PROP_T_CAP):
        return None
    if row.ratio_basis is RatioBasis.A:
        raise SkeletonGenError(
            f"[basis] ratio_basis=a 행({row.rule_id})은 평가 보류 — 열거에서 제외됐어야 한다"
        )
    return t  # t 그대로, s 는 s=t 가정


def _basis_source(row: LimitRow) -> Optional[str]:
    return "assumed_s_eq_t" if row.ratio_basis is RatioBasis.S else None


def bucket_interval(
    row: LimitRow, L: Decimal, bucket: str, config: GenConfig = _DEFAULT_CONFIG
) -> tuple[int, int]:
    """버킷의 그리드 정수 인덱스 구간 [lo, hi] (양끝 포함). 빈 구간이면 전체 중단.

    합격 (0, 0.9L) / 하경계 [0.9L, L−0.01] / =L {L} / 상경계 [L+0.01, 1.1L] /
    불합격 (1.1L, 2.5L]. 비율형은 상한 min(·, 100%). =L 외 경계 표본은 L 과 최소
    0.01(그리드 1칸) 차이가 구간 정의 자체로 보장된다.
    """
    Li = _idx(L)
    low_edge = config.boundary_low_mult * Li
    high_edge = config.boundary_high_mult * Li
    fail_top = config.fail_upper_mult * Li
    if bucket == BUCKET_PASS:
        lo, hi = 1, _ceil_i(low_edge) - 1
    elif bucket == BUCKET_BOUNDARY_LOW:
        lo, hi = _ceil_i(low_edge), Li - 1
    elif bucket == BUCKET_BOUNDARY_EQ:
        lo, hi = Li, Li
    elif bucket == BUCKET_BOUNDARY_HIGH:
        lo, hi = Li + 1, _floor_i(high_edge)
    elif bucket == BUCKET_FAIL:
        lo, hi = _floor_i(high_edge) + 1, _floor_i(fail_top)
    else:
        raise SkeletonGenError(f"[bucket_interval] 미정의 버킷: {bucket!r}")
    if row.unit is Unit.PERCENT:
        hi = min(hi, _idx(config.percent_max))
    if lo > hi:
        raise SkeletonGenError(
            f"[bucket_interval] 빈 버킷 {bucket} — {row.rule_id}, L={L} "
            "(V10 실현성 검사를 통과했는지 확인하라)"
        )
    return lo, hi


def sample_measurement(
    row: LimitRow,
    t: Decimal,
    bucket: str,
    rng: np.random.Generator,
    config: GenConfig = _DEFAULT_CONFIG,
) -> Decimal:
    """0.01 그리드 위 직접 추출 (§4-3). 연속 샘플 후 반올림 금지.

    F4(none_permitted)는 버킷 무관 전량 불합격 — configs [f4_size_min, f4_size_max].
    """
    if row.limit_rule is LimitRule.NONE_PERMITTED:
        if bucket != BUCKET_FAIL:
            raise SkeletonGenError(
                f"[sample_measurement] F4 행({row.rule_id})의 버킷은 fail 만 — got {bucket!r}"
            )
        lo, hi = _idx(config.f4_size_min), _idx(config.f4_size_max)
        if lo > hi:
            raise SkeletonGenError(f"[sample_measurement] F4 범위 뒤집힘: {lo}..{hi}")
    else:
        L = effective_limit(row, _basis_value(row, quantize(t)))
        assert L is not None
        lo, hi = bucket_interval(row, L, bucket, config)
    k = int(rng.integers(lo, hi + 1))
    return _dec(k)


def thickness_floor_idx(row: LimitRow, config: GenConfig = _DEFAULT_CONFIG) -> int:
    """prop 계열 행에서 유효 한계 L ≥ 실현성 하한을 만족하는 최소 t 그리드 인덱스.

    prop 계열은 L = factor × t 라 t 가 0 근방이면 L 이 0 이 되고 어떤 버킷 구간도
    정의되지 않는다. 두께 구간이 0 에서 시작하는 행(파일럿 CSV 의 표면 기공 0.25t)은
    경계 두께 배치가 t=0.00 을 강제로 넣으므로 시드와 무관하게 생성이 전량 중단됐다
    (검토 N2). 구간 하한을 실현 가능한 최소 그리드 값으로 올려 축 전체를 살린다.
    올린 사실은 감사 보고서(audit_skeletons)의 경고로 남는다 — 조용한 보정이 아니다.

    const 계열은 t 와 무관하므로 하한을 올려도 실현 불가가 해소되지 않는다 (그 경우는
    V10 이 보고한 대로 생성 시점에 전체 중단된다).
    """
    if row.limit_rule not in (LimitRule.PROP_T, LimitRule.PROP_T_CAP):
        return _idx(row.thickness_min)
    if row.limit_rule is LimitRule.PROP_T_CAP and row.limit_cap < FEASIBILITY_MIN_L:
        # L = min(factor·t, cap) ≤ cap 이라 어떤 t 로도 하한을 넘지 못한다. t 보정으로
        # 살릴 수 없는 행이므로 여기서 끊어 원인을 가리킨다 (아래 단계의 "빈 버킷"보다 낫다).
        raise SkeletonGenError(
            f"[t 하한] 실현성 미달 — {row.rule_id}: limit_cap={row.limit_cap} < "
            f"{FEASIBILITY_MIN_L} 라 t 보정으로 살릴 수 없다 "
            "(V10 실현성 보고 대상 — scope=excluded 로 배제하라)"
        )
    need = (FEASIBILITY_MIN_L / row.limit_factor).quantize(Q, rounding=ROUND_CEILING)
    return max(_idx(row.thickness_min), _idx(need))


def _sample_thicknesses(
    row: LimitRow, n: int, rng: np.random.Generator, config: GenConfig
) -> list[Decimal]:
    """t 를 0.01 그리드에서 직접 샘플링 (§4-3, S2-3). 반열림 [min, max) → 인덱스 [lo, hi].

    prop 계열은 10%(올림)를 구간 하한 정확값·상한−0.01 에 교대로 배치해 행 선택 로직
    자체를 데이터에 노출한다. 이때의 하한은 thickness_floor_idx 로 보정된 값이다.
    +∞ 구간의 상한은 configs t_sample_max.
    """
    lo = _idx(row.thickness_min)
    if row.thickness_max is not None:
        hi = _idx(row.thickness_max) - 1
    else:
        hi = _idx(config.t_sample_max)
    if hi < lo:
        raise SkeletonGenError(
            f"[t 샘플] 빈 두께 구간 — {row.rule_id}: idx [{lo}, {hi}] "
            "(t_sample_max 가 thickness_min 보다 작은지 확인)"
        )
    idxs: list[int] = []
    if row.limit_rule in (LimitRule.PROP_T, LimitRule.PROP_T_CAP):
        lo = thickness_floor_idx(row, config)
        if lo > hi:
            raise SkeletonGenError(
                f"[t 샘플] 실현성 미달 — {row.rule_id}: 구간 [{row.thickness_min}, "
                f"{row.thickness_max}) 전체에서 L < {FEASIBILITY_MIN_L} "
                "(V10 실현성 보고 대상 — scope=excluded 로 배제하라)"
            )
        n_edge = min(n, _ceil_i(Decimal(n) * config.prop_edge_frac))
        for j in range(n_edge):
            idxs.append(lo if j % 2 == 0 else hi)
    for _ in range(n - len(idxs)):
        idxs.append(int(rng.integers(lo, hi + 1)))
    return [_dec(k) for k in idxs]


# ---------------------------------------------------------------------------
# 골격 스키마 (§4-5) — 직렬화 키 순서·수치 표기 고정
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """오염 역추적 경로 (§4-5). D4 는 난수가 없어 seed=None."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    limits_sha256: str
    generator_version: str = GENERATOR_VERSION
    seed: Optional[int] = None
    configs_sha256: str

    def to_json_dict(self) -> dict:
        return {
            "limits_sha256": self.limits_sha256,
            "generator_version": self.generator_version,
            "seed": self.seed,
            "configs_sha256": self.configs_sha256,
        }


@dataclass(frozen=True)
class SkeletonFlags:
    """build_skeleton 의 분기 플래그 (§4-5·§4-7)."""

    source: str                              # sampled | measured
    verdict_mode: VerdictMode
    sample_bucket: Optional[str] = None      # (c) 만. D4 는 None
    image_id: Optional[str] = None
    defect_instance_id: Optional[str] = None
    thickness_source: str = SOURCE_SAMPLED
    scale_source: str = SOURCE_NA
    quality_source: str = SOURCE_SAMPLED


# 직렬화 키 순서 (고정 — 변경은 승인 사항, §4-5)
SKELETON_KEYS: tuple[str, ...] = (
    "sample_id", "source", "image_id", "defect_instance_id",
    "defect_code", "material", "inspection_method", "quality_scheme", "quality_level",
    "size_mm", "measured_value", "measured_unit", "thickness_mm",
    "clause_id", "limit_type", "limit_rule", "limit_op", "unit",
    "limit_value", "limit_factor", "limit_cap", "ratio_basis", "basis_source",
    "verdict", "margin", "verdict_type", "sample_bucket",
    "thickness_source", "scale_source", "quality_source", "verdict_mode",
    "provenance",
)

_DECIMAL_KEYS = frozenset({
    "size_mm", "measured_value", "thickness_mm", "limit_value",
    "limit_factor", "limit_cap", "margin",
})
_GATED_KEYS = ("verdict", "margin", "verdict_type")  # §4-7 출력 게이트 대상


class Skeleton(BaseModel):
    """골격 1건. verdict·margin 은 judge() 계산값을 **항상 보유**하고 (§4-7:
    judge 계산 자체는 수행), clause_only 게이트는 to_json_dict 직렬화에서만 적용된다."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    source: Literal["sampled", "measured"]
    image_id: Optional[str] = None
    defect_instance_id: Optional[str] = None
    defect_code: str
    material: Material
    # D 채점기가 인용 조항으로 후보를 좁힌 뒤 행을 재선택할 때 쓰는 축 (§1-4 rows_for_clause).
    # 골격에 실려 있지 않으면 재선택에서 표면·내부 행이 다시 섞인다.
    inspection_method: InspectionMethod
    quality_scheme: QualityScheme
    quality_level: str
    size_mm: Optional[Decimal] = None
    measured_value: Optional[Decimal] = None   # 비율형(%) 측정값 — 단위 혼입 스키마 차단
    measured_unit: Optional[str] = None
    thickness_mm: Decimal
    clause_id: str
    limit_type: LimitType
    limit_rule: LimitRule
    limit_op: LimitOp
    unit: Optional[Unit] = None
    limit_value: Optional[Decimal] = None      # 유효 한계 L (F4 는 None)
    limit_factor: Optional[Decimal] = None
    limit_cap: Optional[Decimal] = None
    ratio_basis: Optional[RatioBasis] = None
    basis_source: Optional[str] = None         # s=t 가정 기록 (열린 질문 3)
    verdict: Optional[Verdict] = None          # judge 계산값 (직렬화 게이트)
    margin: Optional[Decimal] = None
    verdict_type: Optional[str] = None         # 확정 | 조건부 (직렬화 게이트)
    sample_bucket: Optional[str] = None
    thickness_source: str
    scale_source: str
    quality_source: str
    verdict_mode: VerdictMode
    provenance: Provenance

    def to_json_dict(self) -> dict:
        """키 순서·수치 표기(양자화 문자열) 고정 직렬화. clause_only 는 verdict·margin·
        verdict_type 을 null 로 게이트한다 (§4-7)."""
        gate = self.verdict_mode is VerdictMode.CLAUSE_ONLY
        out: dict[str, Any] = {}
        for k in SKELETON_KEYS:
            if k == "provenance":
                out[k] = self.provenance.to_json_dict()
                continue
            v = getattr(self, k)
            if gate and k in _GATED_KEYS:
                v = None
            if v is None:
                out[k] = None
            elif k in _DECIMAL_KEYS:
                out[k] = str(v)
            elif hasattr(v, "value"):
                out[k] = v.value
            else:
                out[k] = v
        return out


def skeleton_from_json(d: Mapping[str, Any]) -> Skeleton:
    """to_json_dict 역변환 (T23 round-trip). verdict enum 위반은 pydantic 이 거부한다."""
    kwargs: dict[str, Any] = {}
    for k in SKELETON_KEYS:
        v = d.get(k)
        if k == "provenance":
            kwargs[k] = Provenance(**v)
        elif v is not None and k in _DECIMAL_KEYS:
            kwargs[k] = Decimal(str(v))
        else:
            kwargs[k] = v
    return Skeleton(**kwargs)


def skeletons_jsonl(skeletons: Sequence[Skeleton]) -> str:
    return "".join(
        json.dumps(s.to_json_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
        for s in skeletons
    )


def skeletons_sha256(skeletons: Sequence[Skeleton]) -> str:
    return hashlib.sha256(skeletons_jsonl(skeletons).encode("utf-8")).hexdigest()


def build_skeleton(
    row: LimitRow,
    measured: Decimal,
    t: Decimal,
    flags: SkeletonFlags,
    sample_id: str,
    provenance: Provenance,
    config: GenConfig = _DEFAULT_CONFIG,
) -> Skeleton:
    """골격 조립 (§4-2). judge·effective_limit 은 limit_eval 공용 함수만 쓴다.

    널러빌리티 매트릭스 (§4-4): 길이·직경(mm) → size_mm / 비율(percent) → measured_value /
    F4 → size_mm 만, L·margin 은 None.
    """
    tq = quantize(t)
    mq = quantize(measured)
    basis = _basis_value(row, tq)
    L = effective_limit(row, basis)  # F4 = None
    judgment = judge(row, mq, basis)

    is_ratio = row.unit is Unit.PERCENT
    # 세 출처 중 하나라도 가정이면 조건부다. quality_source 를 빼면, 판정을 가른 허용치가
    # 가정 품질수준(configs d4_quality_level)에서 나온 D4 골격이 "확정"으로 표기된다 —
    # 수준 가정이 바뀌면 verdict 도 바뀌므로 확정이 아니다 (검토 N7).
    verdict_type = (
        VERDICT_TYPE_CONDITIONAL
        if SOURCE_ASSUMED in (
            flags.thickness_source, flags.scale_source, flags.quality_source
        )
        else VERDICT_TYPE_CONFIRMED
    )
    return Skeleton(
        sample_id=sample_id,
        source=flags.source,  # type: ignore[arg-type]
        image_id=flags.image_id,
        defect_instance_id=flags.defect_instance_id,
        defect_code=row.defect_code,
        material=row.material,
        inspection_method=row.inspection_method,
        quality_scheme=row.quality_scheme,
        quality_level=row.quality_level,
        size_mm=None if is_ratio else mq,
        measured_value=mq if is_ratio else None,
        measured_unit="percent" if is_ratio else None,
        thickness_mm=tq,
        clause_id=row.clause_id,
        limit_type=row.limit_type,
        limit_rule=row.limit_rule,
        limit_op=row.limit_op,
        unit=row.unit,
        limit_value=L,
        limit_factor=row.limit_factor,
        limit_cap=row.limit_cap,
        ratio_basis=row.ratio_basis,
        basis_source=_basis_source(row),
        verdict=judgment.verdict,
        margin=judgment.margin,
        verdict_type=verdict_type,
        sample_bucket=flags.sample_bucket,
        thickness_source=flags.thickness_source,
        scale_source=flags.scale_source,
        quality_source=flags.quality_source,
        verdict_mode=flags.verdict_mode,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# (c) 판정추론 경로 (§4-3) — 항상 full, 내부 이상은 전체 중단
# ---------------------------------------------------------------------------


def generate_corpus_skeletons(
    table: LimitsTable,
    seed: int,
    total: Optional[int] = None,
    cap: Optional[int] = None,
    config: GenConfig = _DEFAULT_CONFIG,
    inspection_method: InspectionMethod | str | None = DEFAULT_INSPECTION_METHOD,
) -> tuple[Skeleton, ...]:
    """(c) 경로 골격 전량 생성. 같은 (table sha, seed, config)면 출력 jsonl sha 가 바이트
    동일하다 (T19). 조합별 rng 는 정체성 시드라 무관 행 추가에 불변이다 (T21).

    검사 방법 축은 열거·행 재조회·질의에 모두 같은 값을 쓴다. 축 미지정(None)은 거부한다 —
    축 없이 생성하면 표면(VT) 기준 판정과 내부(RT) 기준 판정이 한 corpus 에 섞이고,
    그 corpus 로 학습한 모델의 판정 정합성은 틀린 채로 높게 나온다."""
    if inspection_method is None:
        raise SkeletonGenError(
            "[generate] 검사 방법 축 미지정 — RT 또는 VT 를 지정하라 "
            "(§1-2 5a: 축 없이 생성하면 표면·내부 기준이 한 corpus 에 섞인다)"
        )
    axis = InspectionMethod(inspection_method)
    total = config.total if total is None else total
    cap = config.combo_cap if cap is None else cap
    table = _inspection_view(table, inspection_method)
    combos = enumerate_combos(table, inspection_method=None)
    counts = allocate_counts(combos, total, cap)
    prov = Provenance(
        limits_sha256=table.sha256,
        generator_version=GENERATOR_VERSION,
        seed=seed,
        configs_sha256=config_sha256(config),
    )
    out: list[Skeleton] = []
    for combo in combos:
        n = counts.get(combo, 0)
        if n == 0:
            continue
        row = combo.row
        rng = np.random.default_rng(np.random.SeedSequence((seed, combo.hash64)))
        strata = decompose_counts(n, row.limit_rule, config)
        buckets = [b for b in STRATA for _ in range(strata[b])]
        # 버킷 순서를 조합 rng 로 섞는다. STRATA 고정 순서로 두면 _sample_thicknesses 가
        # 앞쪽에 몰아넣은 경계 두께 표본이 항상 합격 버킷과 짝지어져, 구간 경계 두께에서
        # 대소 비교가 갈리는 사례(=L·불합격)가 corpus 에 0건이 된다 (검토 N8).
        # 같은 (seed, 조합)이면 같은 순열이므로 결정론은 유지된다.
        if len(buckets) > 1:
            buckets = [buckets[i] for i in rng.permutation(len(buckets))]
        ts = _sample_thicknesses(row, n, rng, config)
        for i, (bucket, t) in enumerate(zip(buckets, ts)):
            measured = sample_measurement(row, t, bucket, rng, config)
            flags = SkeletonFlags(
                source="sampled",
                verdict_mode=VerdictMode.FULL,  # (c) 경로는 항상 full (§4-7)
                sample_bucket=bucket,
                thickness_source=SOURCE_SAMPLED,
                scale_source=SOURCE_NA,
                quality_source=SOURCE_SAMPLED,
            )
            sk = build_skeleton(
                row, measured, t, flags,
                sample_id=f"{combo.hash8}-{i:03d}",
                provenance=prov, config=config,
            )
            # 불변식 (S2-3): 저장 t 재조회 = 생성 행. 내부 위반은 전체 중단.
            recheck = applicable_row(
                table, row.defect_code, row.material, axis, row.quality_scheme,
                row.quality_level, sk.thickness_mm, limit_type=row.limit_type,
            )
            if recheck.rule_id != row.rule_id:
                raise SkeletonGenError(
                    f"[generate] 저장 t 재조회 행 불일치: {row.rule_id} → {recheck.rule_id}"
                )
            out.append(sk)
    return tuple(out)


# ---------------------------------------------------------------------------
# D4 실측 경로 (§4-6) — 난수 없음, 외부 입력 이상은 quarantine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class D4Label:
    """이미지 결함 인스턴스 1건 (manifest·라벨 파생). size_px 는 폴리곤 실측 픽셀 크기."""

    image_id: str
    defect_instance_id: str
    defect_type: str                       # 사상표 L2 키(porosity 등) 또는 ISO 코드
    material: str                          # ST | AL
    size_px: Optional[Decimal]
    thickness_mm: Optional[Decimal] = None  # 메타데이터 두께 (있으면)


@dataclass(frozen=True)
class D4Assumptions:
    """D4 가정값 — configs 단일점 (S1-4). 현재 verdict_mode=clause_only 고정 (§4-7),
    가정 두께·스케일은 conditional 승격 시에만 발효된다."""

    quality_scheme: str = "iso5817"
    quality_level: str = "C"
    # 검사 방법 축 (§1-2 5a). D1 은 RT 촬영본이므로 기본값 RT — 가정이 아니라 데이터 사실이다.
    inspection_method: InspectionMethod | str = DEFAULT_INSPECTION_METHOD
    thickness_mm: Optional[Decimal] = None
    scale_mm_per_px: Optional[Decimal] = None
    verdict_mode: VerdictMode = VerdictMode.CLAUSE_ONLY


@dataclass(frozen=True)
class Quarantined:
    """D4 외부 입력 이상 격리 레코드 (§4-2). 사유 코드·건수는 G2 감사와 논문에 보고."""

    image_id: str
    defect_instance_id: Optional[str]
    reason: str
    detail: str = ""


@lru_cache(maxsize=4)
def _label_map(path_str: str) -> dict:
    return yaml.safe_load(Path(path_str).read_text(encoding="utf-8")) or {}


def _resolve_iso_code(defect_type: str, label_map: Mapping) -> Optional[str]:
    """사상표 L2 키 또는 ISO 코드 → ISO 코드. 사상표 밖이면 None (하드코딩 금지)."""
    dts = label_map.get("defect_types") or {}
    if defect_type in dts:
        return str(dts[defect_type]["iso_code"])
    codes: set[str] = set()
    for spec in dts.values():
        codes.add(str(spec["iso_code"]))
        codes.update(str(c) for c in (spec.get("iso_code_alt") or []))
    return defect_type if defect_type in codes else None


def skeleton_from_label(
    table: LimitsTable,
    label: D4Label,
    scale: Optional[Decimal],
    assumptions: D4Assumptions,
    provenance: Optional[Provenance] = None,
    label_map_path: Optional[Path] = None,
    config: GenConfig = _DEFAULT_CONFIG,
) -> Union[Skeleton, Quarantined]:
    """D4 경로 (§4-6). 난수 없음 — manifest 동일이면 출력 동일.

    scale = 메타데이터 픽셀→mm 스케일. None 이면 assumptions.scale_mm_per_px(가정값)로
    fallback 하고 scale_source=assumed·verdict_type=조건부 (T25). judge 는 항상 계산하고
    clause_only 출력 게이트는 직렬화에서 적용된다 (§4-7).
    외부 입력 이상은 사유 코드를 붙여 Quarantined 로 격리 배출한다 — 전체 중단 없음.
    행 선택은 assumptions.inspection_method 축 뷰에서만 한다 (§1-2 5a).
    """
    table = _inspection_view(table, assumptions.inspection_method)
    axis = InspectionMethod(assumptions.inspection_method)

    # 외부 입력의 타입 이상은 코드 버그가 아니라 데이터 이상이다 (§4-2). float 이 섞이면
    # quantize 의 float 금지 가드보다 곱셈이 먼저 터져 원인 불명의 TypeError 로 배치 전체가
    # 중단됐다 (검토 N5). 진입부에서 걸러 사유 코드를 붙인 격리로 배출한다.
    for name, value in (("size_px", label.size_px),
                        ("thickness_mm", label.thickness_mm),
                        ("scale", scale),
                        ("assumed_thickness_mm", assumptions.thickness_mm),
                        ("assumed_scale_mm_per_px", assumptions.scale_mm_per_px)):
        if value is None or isinstance(value, (Decimal, int)) and not isinstance(value, bool):
            continue
        return Quarantined(
            label.image_id, label.defect_instance_id, QUARANTINE_INVALID_NUMERIC_TYPE,
            f"{name}={value!r} 타입 {type(value).__name__} — Decimal 필수 (§1-3 float 금지)",
        )

    lm = _label_map(str(label_map_path or _LABEL_MAP_PATH))
    code = _resolve_iso_code(label.defect_type, lm)
    if code is None:
        return Quarantined(label.image_id, label.defect_instance_id,
                           QUARANTINE_UNMAPPED_DEFECT, f"defect_type={label.defect_type!r}")
    primary_codes = {
        str(spec["iso_code"]) for spec in (lm.get("defect_types") or {}).values()
    }
    if code not in primary_codes:
        # 기공 1:N 해소 (S2-4): D4 는 대표 코드(2011) 확정, alt 코드(2012 — 비율 한계)는
        # 분모 실측 불능 → 선언 배제. 사상표의 iso_code/iso_code_alt 구분과 상호 잠금.
        return Quarantined(label.image_id, label.defect_instance_id,
                           QUARANTINE_EXCLUDED_2012,
                           f"사상표 alt 코드 {code} — 면적 비율 분모 실측 불능, D4 배제")

    if label.size_px is None or label.size_px <= 0:
        return Quarantined(label.image_id, label.defect_instance_id,
                           QUARANTINE_DEGENERATE_POLYGON, f"size_px={label.size_px}")

    if scale is not None:
        scale_val, scale_source = scale, SOURCE_METADATA
    elif assumptions.scale_mm_per_px is not None:
        scale_val, scale_source = assumptions.scale_mm_per_px, SOURCE_ASSUMED
    else:
        return Quarantined(label.image_id, label.defect_instance_id,
                           QUARANTINE_MISSING_SCALE,
                           "스케일 부재 — judge 입력 불능 (verdict_mode 승격 전 가정값 금지)")
    size_mm = quantize(label.size_px * scale_val)
    if size_mm <= 0:
        return Quarantined(label.image_id, label.defect_instance_id,
                           QUARANTINE_DEGENERATE_SIZE, f"size_mm={size_mm}")

    if label.thickness_mm is not None:
        t, thickness_source = quantize(label.thickness_mm), SOURCE_METADATA
    elif assumptions.thickness_mm is not None:
        t, thickness_source = quantize(assumptions.thickness_mm), SOURCE_ASSUMED
    else:
        return Quarantined(label.image_id, label.defect_instance_id,
                           QUARANTINE_MISSING_THICKNESS,
                           "두께 부재 — judge 입력 불능 (verdict_mode 승격 전 가정값 금지)")

    try:
        row = applicable_row(
            table, code, label.material, axis, assumptions.quality_scheme,
            assumptions.quality_level, t,
        )
    except LookupError as e:
        reason = (
            QUARANTINE_AMBIGUOUS_CONSTRAINT if "복수" in str(e)
            else QUARANTINE_NO_APPLICABLE_ROW
        )
        return Quarantined(label.image_id, label.defect_instance_id, reason, str(e))

    if row.unit is Unit.PERCENT:
        return Quarantined(label.image_id, label.defect_instance_id,
                           QUARANTINE_RATIO_NOT_MEASURABLE, row.rule_id)
    if row.ratio_basis is RatioBasis.A:
        return Quarantined(label.image_id, label.defect_instance_id,
                           QUARANTINE_BASIS_A_HOLD, row.rule_id)

    flags = SkeletonFlags(
        source="measured",
        verdict_mode=assumptions.verdict_mode,
        sample_bucket=None,
        image_id=label.image_id,
        defect_instance_id=label.defect_instance_id,
        thickness_source=thickness_source,
        scale_source=scale_source,
        quality_source=SOURCE_ASSUMED,  # 라벨에 지정 수준이 없다 (S1-4)
    )
    prov = provenance or Provenance(
        limits_sha256=table.sha256, seed=None, configs_sha256=config_sha256(config)
    )
    try:
        return build_skeleton(
            row, size_mm, t, flags,
            sample_id=f"{label.image_id}-{label.defect_instance_id}",
            provenance=prov, config=config,
        )
    except ValueError as e:
        # judge 의 외부 입력 기인 예외(퇴화 크기 등)는 D4 에서 격리다 (§4-2)
        return Quarantined(label.image_id, label.defect_instance_id,
                           QUARANTINE_DEGENERATE_SIZE, str(e))


def skeletons_from_labels(
    table: LimitsTable,
    labels: Sequence[D4Label],
    scale: Optional[Decimal],
    assumptions: D4Assumptions,
    provenance: Optional[Provenance] = None,
    label_map_path: Optional[Path] = None,
    config: GenConfig = _DEFAULT_CONFIG,
) -> tuple[tuple[Skeleton, ...], tuple[Quarantined, ...]]:
    """D4 배치 — 이상치 1건이 전체를 중단시키지 않는다 (T31).

    타입 이상은 skeleton_from_label 진입부가 이미 격리하지만, 예상하지 못한 형태의 외부
    입력까지 배치를 죽이지 않도록 TypeError 도 격리 대상으로 받는다 (§4-2 fail-closed 범위).
    """
    oks: list[Skeleton] = []
    bad: list[Quarantined] = []
    for label in labels:
        try:
            r = skeleton_from_label(table, label, scale, assumptions,
                                    provenance, label_map_path, config)
        except TypeError as e:
            r = Quarantined(label.image_id, label.defect_instance_id,
                            QUARANTINE_INVALID_NUMERIC_TYPE, str(e))
        (oks if isinstance(r, Skeleton) else bad).append(r)  # type: ignore[arg-type]
    return tuple(oks), tuple(bad)


# ---------------------------------------------------------------------------
# 이미지 단위 조립 (§4-6 assemble_pair) — 정상 이미지도 같은 경로 (§4-6-1)
# ---------------------------------------------------------------------------

PAIR_KEYS: tuple[str, ...] = (
    "image_id", "defects", "verdict", "cited_clauses", "verdict_type",
    "quality_level", "source", "sample_id", "verdict_mode", "provenance",
)


class PairGold(BaseModel):
    """이미지 단위 gold (§4-6). verdict 는 aggregate_verdicts 계산값을 보유하고
    clause_only 게이트는 직렬화에서 적용한다. margin 은 이미지 수준에 두지 않는다."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    image_id: str
    defects: tuple[Skeleton, ...]
    verdict: Optional[Verdict] = None
    cited_clauses: tuple[str, ...]
    verdict_type: Optional[str] = None
    quality_level: Optional[str] = None
    verdict_mode: VerdictMode
    sample_id: str
    provenance: Optional[Provenance] = None

    def to_json_dict(self) -> dict:
        gate = self.verdict_mode is VerdictMode.CLAUSE_ONLY
        return {
            "image_id": self.image_id,
            "defects": [s.to_json_dict() for s in self.defects],
            "verdict": None if gate else self.verdict,
            "cited_clauses": list(self.cited_clauses),
            "verdict_type": None if gate else self.verdict_type,
            "quality_level": self.quality_level,
            "source": "measured",
            "sample_id": self.sample_id,
            "verdict_mode": self.verdict_mode.value,
            "provenance": self.provenance.to_json_dict() if self.provenance else None,
        }


def assemble_pair(
    image_id: str,
    skeletons: Sequence[Skeleton],
    verdict_mode: VerdictMode = VerdictMode.FULL,
    quality_level: Optional[str] = None,
    provenance: Optional[Provenance] = None,
) -> PairGold:
    """이미지 단위 조립 (§4-6, S2-5). 다결함 verdict 는 **aggregate_verdicts 만** 사용
    (보수 규칙 — 재구현 금지). 정상 이미지는 skeletons=[] 로 같은 경로를 지난다:
    aggregate_verdicts([]) = 합격, cited_clauses = [] (§4-6-1).
    cited_clauses = 정렬 고정 합집합(중복 제거). margin 은 defects[]별 유지."""
    for s in skeletons:
        if s.verdict_mode is not verdict_mode:
            raise SkeletonGenError(
                f"[assemble_pair] 골격 verdict_mode 불일치: {s.sample_id} "
                f"{s.verdict_mode.value} ≠ {verdict_mode.value}"
            )
        if s.image_id is not None and s.image_id != image_id:
            raise SkeletonGenError(
                f"[assemble_pair] image_id 불일치: {s.image_id} ≠ {image_id}"
            )
    verdicts = [s.verdict for s in skeletons]
    if any(v is None for v in verdicts):
        raise SkeletonGenError("[assemble_pair] 골격 verdict 계산값 결손 — judge 미수행 골격")
    image_verdict = aggregate_verdicts(verdicts)  # 빈 목록 ⇒ 합격
    cited = tuple(sorted({s.clause_id for s in skeletons}))
    if skeletons:
        verdict_type = (
            VERDICT_TYPE_CONDITIONAL
            if any(s.verdict_type == VERDICT_TYPE_CONDITIONAL for s in skeletons)
            else VERDICT_TYPE_CONFIRMED
        )
        levels = {s.quality_level for s in skeletons}
        ql = quality_level or (levels.pop() if len(levels) == 1 else None)
        sample_id = f"{image_id}-pair"
    else:
        verdict_type = VERDICT_TYPE_CONFIRMED  # 크기 비교가 없어 가정과 무관 (§4-6-1)
        ql = quality_level
        sample_id = f"{image_id}-normal"
    return PairGold(
        image_id=image_id,
        defects=tuple(skeletons),
        verdict=image_verdict,
        cited_clauses=cited,
        verdict_type=verdict_type,
        quality_level=ql,
        verdict_mode=verdict_mode,
        sample_id=sample_id,
        provenance=provenance,
    )


# ---------------------------------------------------------------------------
# 정상 이미지 선별 (§4-6-1 ①)
# ---------------------------------------------------------------------------


class NormalSelection(BaseModel):
    """normal_selection.json 의 내용 (S4 스냅샷 대상)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    ratio: str
    manifest_snapshot: Optional[str] = None
    clients: dict[str, tuple[str, ...]]
    shortfalls: dict[str, int]  # 후보 부족으로 목표에 못 미친 건수 (보고용)

    def to_json_dict(self) -> dict:
        return {
            "seed": self.seed,
            "ratio": self.ratio,
            "manifest_snapshot": self.manifest_snapshot,
            "clients": {c: list(v) for c, v in sorted(self.clients.items())},
            "shortfalls": dict(sorted(self.shortfalls.items())),
        }


def select_normal_images(
    manifest: Any,
    ratio: Decimal,
    seed: int,
    defect_pair_counts: Mapping[str, int],
    manifest_snapshot: Optional[str] = None,
) -> NormalSelection:
    """클라이언트별 정상 이미지 다운샘플 (§4-6-1). 시드 고정·재현 가능 (T35).

    manifest: 레코드 시퀀스(dict: image_id, split, client, n_defects) 또는 DataFrame.
    defect_pair_counts: 클라이언트별 **검증 통과 결함 페어 수** (비율의 분모 — manifest 에
    없어 인자로 받는다. 스펙 시그니처 (manifest, ratio, seed) 의 확장, 보고서 명시).
    목표 = floor(결함 페어 수 × ratio). 후보 부족 시 전량 선택 + shortfall 기록.
    """
    if hasattr(manifest, "to_dict"):
        records = manifest.to_dict("records")
    else:
        records = list(manifest)
    candidates: dict[str, list[str]] = {}
    for r in records:
        if str(r["split"]) != "train" or int(r.get("n_defects", 0)) != 0:
            continue
        candidates.setdefault(str(r["client"]), []).append(str(r["image_id"]))
    clients: dict[str, tuple[str, ...]] = {}
    shortfalls: dict[str, int] = {}
    for client in sorted(defect_pair_counts):
        pool = sorted(candidates.get(client, []))
        k = int((Decimal(int(defect_pair_counts[client])) * ratio)
                .to_integral_value(rounding=ROUND_FLOOR))
        if k > len(pool):
            shortfalls[client] = k - len(pool)
            k = len(pool)
        h = int.from_bytes(
            hashlib.sha256(f"normal|{client}".encode("utf-8")).digest()[:8], "big"
        )
        rng = np.random.default_rng(np.random.SeedSequence((seed, h)))
        picked = rng.choice(len(pool), size=k, replace=False) if k else []
        clients[client] = tuple(sorted(pool[int(i)] for i in picked))
    return NormalSelection(
        seed=seed,
        ratio=str(ratio),
        manifest_snapshot=manifest_snapshot,
        clients=clients,
        shortfalls=shortfalls,
    )


# ---------------------------------------------------------------------------
# 결함 어휘 락 (§4-6-1 ④) — 사상표에서 파생, 라벨 문자열 하드코딩 금지
# ---------------------------------------------------------------------------


def load_defect_lexicon(label_map_path: Optional[Path] = None) -> frozenset[str]:
    """사상표(계약 #1)의 결함 명칭(한/영)·ISO 코드(+alt) 사전."""
    lm = _label_map(str(label_map_path or _LABEL_MAP_PATH))
    tokens: set[str] = set()
    for spec in (lm.get("defect_types") or {}).values():
        tokens.add(str(spec["iso_code"]))
        tokens.update(str(c) for c in (spec.get("iso_code_alt") or []))
        for key in ("name_ko", "name_en"):
            if spec.get(key):
                tokens.add(str(spec[key]))
    return frozenset(tokens)


def normal_pair_violations(
    record: Mapping[str, Any], verdict_mode: VerdictMode
) -> tuple[str, ...]:
    """정상 페어 레코드 스키마 강제 (§4-6-1 ⑤ 1단계 + T34 개정판).

    defects=[] ⇒ cited_clauses=[] ∧ verdict = (full·conditional 에서만) 합격,
    clause_only 에서는 null.
    """
    violations: list[str] = []
    if record.get("defects"):
        violations.append("defects_nonempty")
    if record.get("cited_clauses"):
        violations.append("cited_clauses_nonempty")
    v = record.get("verdict")
    if verdict_mode is VerdictMode.CLAUSE_ONLY:
        if v is not None:
            violations.append("verdict_not_null")
    elif v != PASS:
        violations.append("verdict_mismatch")
    return tuple(violations)


# ---------------------------------------------------------------------------
# G2 사후 감사 (§4-9)
# ---------------------------------------------------------------------------


class AuditReport(BaseModel):
    """audit_skeletons 결과. ok=False 면 스냅샷 확정 금지."""

    model_config = ConfigDict(frozen=True)

    ok: bool
    failures: tuple[str, ...]
    warnings: tuple[str, ...]
    n_skeletons: int
    bucket_totals: dict[str, int]
    combo_counts: dict[str, int]          # rule_id → 표본 수
    verdict_counts: dict[str, int]
    n_assumed: int
    n_quarantined: int
    n_a_basis_held: int
    skeletons_sha256: str


def audit_skeletons(
    skeletons: Sequence[Skeleton],
    table: LimitsTable,
    quarantined: Sequence[Quarantined] = (),
    config: GenConfig = _DEFAULT_CONFIG,
    expected_counts: Optional[Mapping[str, int]] = None,  # rule_id → n (없으면 전 조합 ≥1)
    inspection_method: InspectionMethod | str | None = DEFAULT_INSPECTION_METHOD,
) -> AuditReport:
    """G2 사후 감사: 버킷 비율(±2%p) / 커버리지 결손 0 / 상한 준수 / 전 골격 재계산
    일치 / verdict 쏠림(>65:35) 경고 / assumed·quarantine 건수 / 출력 sha256 기록.

    조합별 버킷은 decompose_counts 재계산과의 **정확 일치**로 검사한다 (소규모 조합은
    최소 보장 때문에 ±2%p 로는 판정 불능 — 비율 검사는 전체 풀 기준, F4 조합 제외).

    행 재선택은 생성 때와 같은 검사 방법 축 뷰에서 한다. 축이 다른 골격 묶음을 넣으면
    재선택이 실패해 감사가 떨어진다 — 그 편이 조용히 통과하는 것보다 낫다.
    """
    table = _inspection_view(table, inspection_method)
    failures: list[str] = []
    warnings: list[str] = []
    bucket_totals = {s: 0 for s in STRATA}
    combo_counts: dict[str, int] = {}
    combo_buckets: dict[str, dict[str, int]] = {}
    combo_rules: dict[str, LimitRule] = {}
    verdict_counts: dict[str, int] = {PASS: 0, FAIL: 0}
    n_assumed = 0

    combo_min_t: dict[str, Decimal] = {}

    for sk in skeletons:
        # 재계산 일치 (T22·T29): 저장 t 로 행 재선택 → clause·L·verdict·margin 재도출
        try:
            row = applicable_row(
                table, sk.defect_code, sk.material, _query_axis(inspection_method, sk),
                sk.quality_scheme, sk.quality_level, sk.thickness_mm,
                limit_type=sk.limit_type,
            )
        except (LookupError, ValueError) as e:
            failures.append(f"{sk.sample_id}: 행 재선택 실패 — {e}")
            continue
        if row.clause_id != sk.clause_id:
            failures.append(
                f"{sk.sample_id}: clause 불일치 {sk.clause_id} ≠ {row.clause_id}"
            )
            continue
        if row.inspection_method is not sk.inspection_method:
            failures.append(
                f"{sk.sample_id}: inspection_method 불일치 "
                f"{sk.inspection_method.value} ≠ {row.inspection_method.value}"
            )
        basis = _basis_value(row, sk.thickness_mm)
        L = effective_limit(row, basis)
        if L != sk.limit_value:
            failures.append(f"{sk.sample_id}: L 재계산 불일치 {sk.limit_value} ≠ {L}")
        measured = sk.measured_value if sk.measured_value is not None else sk.size_mm
        judgment = judge(row, measured, basis)
        if judgment.verdict != sk.verdict or judgment.margin != sk.margin:
            failures.append(f"{sk.sample_id}: judge 재계산 불일치")
        if sk.verdict in verdict_counts:
            verdict_counts[sk.verdict] += 1
        if SOURCE_ASSUMED in (sk.thickness_source, sk.scale_source, sk.quality_source):
            n_assumed += 1

        if sk.source != "sampled":
            continue
        combo_counts[row.rule_id] = combo_counts.get(row.rule_id, 0) + 1
        combo_rules[row.rule_id] = row.limit_rule
        prev_t = combo_min_t.get(row.rule_id)
        if prev_t is None or sk.thickness_mm < prev_t:
            combo_min_t[row.rule_id] = sk.thickness_mm
        cb = combo_buckets.setdefault(row.rule_id, {s: 0 for s in STRATA})
        if sk.sample_bucket not in STRATA:
            failures.append(f"{sk.sample_id}: 미정의 버킷 {sk.sample_bucket!r}")
            continue
        cb[sk.sample_bucket] += 1
        bucket_totals[sk.sample_bucket] += 1
        # 버킷 소속 (그리드 인덱스 기준, T15 와 동일 규약)
        if row.limit_rule is LimitRule.NONE_PERMITTED:
            lo, hi = _idx(config.f4_size_min), _idx(config.f4_size_max)
        else:
            lo, hi = bucket_interval(row, L, sk.sample_bucket, config)
        mi = _idx(measured)
        if not (lo <= mi <= hi):
            failures.append(
                f"{sk.sample_id}: 버킷 침범 — {sk.sample_bucket} [{lo},{hi}] 밖 {mi}"
            )

    # 상한·조합별 분해 일치·커버리지
    for rid, n in sorted(combo_counts.items()):
        if n > config.combo_cap:
            failures.append(f"조합 {rid}: 상한 초과 {n} > {config.combo_cap}")
        expected = decompose_counts(n, combo_rules[rid], config)
        if combo_buckets[rid] != expected:
            failures.append(
                f"조합 {rid}: 버킷 분해 불일치 {combo_buckets[rid]} ≠ {expected}"
            )
        if expected_counts is not None and rid in expected_counts \
                and n != expected_counts[rid]:
            failures.append(f"조합 {rid}: 배분 불일치 {n} ≠ {expected_counts[rid]}")
    sampled_any = any(sk.source == "sampled" for sk in skeletons)
    if sampled_any:
        # table 은 이미 축 뷰다 — 여기서 또 좁히면 VT 감사가 통째로 빈다
        enumerated = {c.row.rule_id for c in enumerate_combos(table, inspection_method=None)}
        expected_rids = (
            {r for r, n in expected_counts.items() if n > 0}
            if expected_counts is not None
            else enumerated
        )
        missing = expected_rids - set(combo_counts)
        extra = set(combo_counts) - enumerated
        if missing:
            failures.append(f"커버리지 결손: {sorted(missing)}")
        if extra:
            failures.append(f"외삽(유령 조합): {sorted(extra)}")

    # prop 계열 t 하한 보정 보고 (§4-3, 검토 N2): 실현성 하한 때문에 구간 하한보다
    # 위에서 샘플링한 조합을 감사 보고서에 남긴다. 조용한 보정으로 두면 "구간 [0, ∞) 를
    # 다 훑었다"는 잘못된 전제로 커버리지를 읽게 된다.
    for rid in sorted(combo_min_t):
        rows = [r for r in table.rows if r.rule_id == rid]
        if not rows or rows[0].limit_rule not in (LimitRule.PROP_T, LimitRule.PROP_T_CAP):
            continue
        floor_t = _dec(thickness_floor_idx(rows[0], config))
        if floor_t > rows[0].thickness_min:
            warnings.append(
                f"t 하한 보정: {rid} — 구간 하한 {rows[0].thickness_min} → {floor_t} "
                f"(그 아래는 L < {FEASIBILITY_MIN_L} 로 버킷 정의 불능)"
            )

    # 전체 버킷 비율 ±2%p (F4 조합 제외 — 의도가 전량 불합격).
    # 기대값은 상수 40/40/20 이 아니라 조합별 배분 n 의 이론 분해 합 sum(decompose_counts(n))
    # 이다. 소규모 배분에서는 §4-3 최소 보장(=L·상·하 각 1건)이 비율보다 우선하므로 상수와
    # 비교하면 규칙대로 동작한 생성물이 감사에서 떨어진다 (검토 N4). 이론값이 명목
    # 비율에서 벗어나는 것 자체는 경고로 보고해 "조합당 배분이 작다"는 사실이 드러나게 한다.
    non_f4 = {
        s: sum(cb[s] for rid, cb in combo_buckets.items()
               if combo_rules[rid] is not LimitRule.NONE_PERMITTED)
        for s in STRATA
    }
    n_ratio = sum(non_f4.values())
    if n_ratio:
        theory = {s: 0 for s in STRATA}
        for rid, n in combo_counts.items():
            if combo_rules[rid] is LimitRule.NONE_PERMITTED:
                continue
            for s, v in decompose_counts(n, combo_rules[rid], config).items():
                theory[s] += v
        nominal = {
            BUCKET_PASS: Decimal(config.ratio_pass),
            BUCKET_FAIL: Decimal(config.ratio_fail),
            "boundary": Decimal(
                config.ratio_boundary_low + config.ratio_boundary_eq
                + config.ratio_boundary_high
            ),
        }
        def _group3(src: Mapping[str, int]) -> dict[str, Decimal]:
            return {
                BUCKET_PASS: Decimal(src[BUCKET_PASS]),
                BUCKET_FAIL: Decimal(src[BUCKET_FAIL]),
                "boundary": Decimal(sum(src[s] for s in BOUNDARY_STRATA)),
            }

        realized, expected = _group3(non_f4), _group3(theory)
        n_theory = sum(theory.values())
        for k, nominal_pct in nominal.items():
            pct = realized[k] * 100 / n_ratio
            target = (expected[k] * 100 / n_theory) if n_theory else nominal_pct
            if abs(pct - target) > config.bucket_tolerance_pp:
                failures.append(
                    f"버킷 비율 이탈: {k} {pct:.2f}% "
                    f"(이론 배분 {target:.2f}% ± {config.bucket_tolerance_pp}%p)"
                )
            elif abs(target - nominal_pct) > config.bucket_tolerance_pp:
                warnings.append(
                    f"버킷 비율 목표 미달성: {k} 이론 {target:.2f}% ↔ 명목 {nominal_pct}% "
                    "— 조합당 배분이 작아 최소 보장(=L·상·하 각 1건)이 비율보다 우선한다"
                )

    # verdict 쏠림 경고 (>65:35)
    n_v = verdict_counts[PASS] + verdict_counts[FAIL]
    if n_v:
        for lbl, cnt in verdict_counts.items():
            if Decimal(cnt) / n_v > config.verdict_skew_warn:
                warnings.append(
                    f"verdict 쏠림: {lbl} {cnt}/{n_v} > {config.verdict_skew_warn}"
                )

    return AuditReport(
        ok=not failures,
        failures=tuple(failures),
        warnings=tuple(warnings),
        n_skeletons=len(skeletons),
        bucket_totals=bucket_totals,
        combo_counts=combo_counts,
        verdict_counts=verdict_counts,
        n_assumed=n_assumed,
        n_quarantined=len(quarantined),
        n_a_basis_held=len(a_basis_hold_rows(table, inspection_method=None)),
        skeletons_sha256=skeletons_sha256(tuple(skeletons)),
    )

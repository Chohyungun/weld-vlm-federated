"""3단계 검증 · 1단계 — 규칙 검사 (프로그램, 전수). 스펙 §6-1.

자산별 검사 (실패분은 폐기, 재생성 금지, 통과율 기록 — §6):

| 검사 | (c)·D4 | (b) QA |
|---|---|---|
| 스키마 | 널러빌리티 매트릭스(§4-4) + verdict enum | 질문/답/근거 구절 ID |
| ISO 코드 | 사상표 집합 실존 | 해당 시 동일 |
| 조항 실존 | clause_id ∈ limits(공식 로더 경유 LimitsTable) + 색인 등재 목록 | 인용 시 동일 |
| 부등식 재계산 | limit_eval.judge 재도출 일치 (재구현 금지 — import) | — |
| 조합 실존 | applicable_row (active ∧ canonical, fallback 금지) | — |
| 수치 잠금 | §5-4 전체 재실행 (생성 스크립트 버그의 이중 안전망) | — |
| 근거 실존 | — | 근거 구절 ID ∈ parse 산출물 |
| 중복 | — | 질문 정규화 후 완전 일치 제거 |
| 길이 | max_len 초과 폐기 (자르지 않는다) | 동일 |

- limits 는 LimitsTable 로 주입한다 (limits.csv 직접 읽기 금지 — §1-4 금지 규칙 ①).
- **검사 방법 축**(§1-2 5a)은 골격의 `inspection_method` 로 조회하고, `inspection_method`
  인자를 주면 그 축의 레코드만 통과시킨다. 축이 식별 필드에 없으면 축을 위조해도 1단계를
  통과하고, 축 없이 조회하면 RT/VT 병존 테이블에서 정상 레코드가 '복수 행 매칭'으로 폐기된다.
- verdict_mode 게이팅 (§4-7): D4·정상 페어는 verdict_mode 인자를 받아 clause_only 면
  verdict·margin 이 None 인지 강제한다 (judge 재계산 자체는 수행해 평가 가능성을 확인,
  출력 비교만 게이트). (c) 경로 골격은 항상 full 이다.
- 길이 검사는 문자 수 기준 근사다 — 토크나이저 기준이 필요하면 length_fn 을 주입한다.
- allowed_standards 는 수치 잠금의 표준 식별자 화이트리스트(§5-4)를 그대로 통과시킨다.
  None 이면 결함 레코드는 모듈 기본값(결함 코드 체계), 정상 페어는 공집합을 쓴다 —
  정상 페어는 인용 자체가 금지이기 때문이다. 템플릿이 다른 표준을 인용해야 하면 여기에
  주입한다 (코드 수정이 아니라 설정으로 다루기 위한 인자다).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Callable, Collection, Mapping, Optional, Sequence

from corpus.generate.numeric_lock import (
    STANDARD_WHITELIST_DEFAULT,
    LockResult,
    check_normal_lock,
    check_numeric_lock,
    check_pair_lock,
    find_artifacts,
    find_verdict_implying,
    to_decimal,
)
from corpus.rules.limit_eval import aggregate_verdicts, applicable_row, effective_limit, judge
from corpus.rules.schema import FAIL, PASS, LimitRow, LimitsTable

__all__ = [
    "R_SCHEMA",
    "R_ISO",
    "R_CLAUSE",
    "R_CLAUSE_MISMATCH",
    "R_COMBO",
    "R_RECOMPUTE",
    "R_EVIDENCE",
    "R_DUP",
    "R_LONG",
    "R_NORMAL",
    "R_PAIR",
    "R_ARTIFACT",
    "RecordResult",
    "Stage1Report",
    "normalize_question",
    "check_record_c",
    "check_record_pair",
    "check_record_qa",
    "run_stage1",
]

# ---------------------------------------------------------------------------
# 사유 코드
# ---------------------------------------------------------------------------

R_SCHEMA = "schema_violation"
R_ISO = "iso_code_unknown"
R_CLAUSE = "clause_not_registered"
R_CLAUSE_MISMATCH = "clause_mismatch"
R_COMBO = "combo_not_found"
R_RECOMPUTE = "recompute_mismatch"
R_EVIDENCE = "evidence_not_found"
R_DUP = "duplicate_question"
R_LONG = "too_long"
R_NORMAL = "normal_schema_violation"
R_PAIR = "pair_assembly_violation"
#: 생성문 아티팩트 (80번 G3-1). 값은 맞는데 **표기**가 [자료] 밖인 경우다 —
#: 수치 검사는 후행 0 동치라 원리적으로 못 잡는다.
R_ARTIFACT = "artifact_violation"

_LOCK_PREFIX = "numeric_lock:"

_MATERIALS = {"ST", "AL", "ALL"}
_METHODS = {"RT", "VT", "ALL"}
_SCHEMES = {"iso5817", "iso10675", "iacs", "ks", "none"}
_LIMIT_TYPES = {"길이", "직경", "비율", "불허"}
_LIMIT_RULES = {"const", "prop_t", "prop_t_cap", "none_permitted"}
# 골격의 식별 필드. inspection_method 가 여기 없으면 축을 VT 로 위조하거나 키를 지워도
# 1단계를 통과한다 — 행 선택이 조용히 다른 축의 기준으로 넘어가는 경로다 (§1-2 5a).
_IDENTITY_KEYS = (
    "defect_code",
    "material",
    "inspection_method",
    "quality_scheme",
    "quality_level",
    "limit_type",
    "limit_rule",
    "clause_id",
)
# 주 실험 축. 골격이 ALL(검사 방법 무관 조항)일 때의 조회 축으로만 쓴다 — ALL 행은 어느
# 축 질의에도 응하므로 같은 행이 잡히고, 축이 다른 행과 함께 있으면 복수 매칭으로 실패한다.
_DEFAULT_METHOD = "RT"


@dataclass(frozen=True)
class RecordResult:
    index: int
    sample_id: Optional[str]
    ok: bool
    reasons: tuple[str, ...]
    detail: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage1Report:
    asset: str
    n_in: int
    n_pass: int
    n_fail: int
    fail_reasons: dict[str, int]
    results: tuple[RecordResult, ...]

    def stage_summary(self) -> dict:
        """report.build_validation_report 의 stages 항목으로 바로 쓰는 형태."""
        return {"n_in": self.n_in, "n_pass": self.n_pass, "fail_reasons": dict(self.fail_reasons)}


class _Check:
    """사유·상세 수집기."""

    def __init__(self) -> None:
        self.reasons: list[str] = []
        self.detail: list[str] = []

    def add(self, reason: str, msg: str) -> None:
        self.reasons.append(reason)
        self.detail.append(f"[{reason}] {msg}")

    def merge_lock(self, lock: LockResult) -> None:
        for r in lock.reasons:
            self.reasons.append(_LOCK_PREFIX + r)
        for d in lock.detail:
            self.detail.append(f"[{_LOCK_PREFIX.rstrip(':')}] {d}")

    def result(self, index: int, sample_id: Optional[str]) -> RecordResult:
        seen: list[str] = []
        for r in self.reasons:
            if r not in seen:
                seen.append(r)
        return RecordResult(index, sample_id, not seen, tuple(seen), tuple(self.detail))


def _get_dec(rec: Mapping, key: str, ck: _Check, *, required: bool) -> Optional[Decimal]:
    v = rec.get(key)
    if v is None:
        if required:
            ck.add(R_SCHEMA, f"{key} 결손 (널러빌리티 매트릭스 위반)")
        return None
    try:
        return to_decimal(v)
    except (ValueError, TypeError, InvalidOperation) as e:
        ck.add(R_SCHEMA, f"{key} 수치 아님: {e}")
        return None


def _forbid(rec: Mapping, key: str, ck: _Check, why: str) -> None:
    if rec.get(key) is not None:
        ck.add(R_SCHEMA, f"{key} 는 {why} 에서 null 강제인데 기재됨")


# ---------------------------------------------------------------------------
# 골격 공통 검사 (스키마 → ISO → 조항 → 조합 → 부등식 재계산)
# ---------------------------------------------------------------------------


def _check_skeleton(
    sk: Mapping,
    table: LimitsTable,
    ck: _Check,
    *,
    label_codes: Optional[Collection[str]],
    clause_registry: Optional[Collection[str]],
    gated: bool,
    inspection_method: Optional[str] = None,
) -> Optional[LimitRow]:
    """골격 1건 검사. gated=True(clause_only) 면 verdict·margin 은 None 강제,
    judge 재계산은 수행하되 출력 비교는 생략한다 (§4-7).

    inspection_method: 호출자가 기대 축을 못박을 때 쓴다. 주면 골격의 축과 일치해야 하고,
    주지 않으면 골격에 실린 축으로 조회한다 — 어느 쪽이든 축 없는 조회는 하지 않는다.
    """
    # -- 스키마: 식별 필드 --
    fatal = False
    for key in _IDENTITY_KEYS:
        v = sk.get(key)
        if not isinstance(v, str) or not v:
            ck.add(R_SCHEMA, f"{key} 결손 또는 문자열 아님")
            fatal = True
    if fatal:
        return None
    if sk["material"] not in _MATERIALS:
        ck.add(R_SCHEMA, f"material enum 위반: {sk['material']!r}")
        fatal = True
    if sk["inspection_method"] not in _METHODS:
        ck.add(R_SCHEMA, f"inspection_method enum 위반: {sk['inspection_method']!r}")
        fatal = True
    elif inspection_method is not None and sk["inspection_method"] not in (
        inspection_method, "ALL"
    ):
        ck.add(
            R_SCHEMA,
            f"inspection_method={sk['inspection_method']!r} ↔ 기대 축 {inspection_method!r} "
            "(표면·내부 기준이 섞였다)",
        )
        fatal = True
    if sk["quality_scheme"] not in _SCHEMES:
        ck.add(R_SCHEMA, f"quality_scheme enum 위반: {sk['quality_scheme']!r}")
        fatal = True
    if sk["limit_type"] not in _LIMIT_TYPES:
        ck.add(R_SCHEMA, f"limit_type enum 위반: {sk['limit_type']!r}")
        fatal = True
    if sk["limit_rule"] not in _LIMIT_RULES:
        ck.add(R_SCHEMA, f"limit_rule enum 위반: {sk['limit_rule']!r}")
        fatal = True
    t = _get_dec(sk, "thickness_mm", ck, required=True)
    if fatal or t is None:
        return None

    rule = sk["limit_rule"]
    limit_type = sk["limit_type"]

    # -- 스키마: 널러빌리티 매트릭스 (§4-4) --
    if rule == "none_permitted":
        size = _get_dec(sk, "size_mm", ck, required=True)
        measured = size
        _forbid(sk, "measured_value", ck, "none_permitted")
        _forbid(sk, "limit_value", ck, "none_permitted")
        _forbid(sk, "margin", ck, "none_permitted")
    elif limit_type == "비율":
        measured = _get_dec(sk, "measured_value", ck, required=True)
        _forbid(sk, "size_mm", ck, "비율(F5)")
        _get_dec(sk, "limit_value", ck, required=True)
    else:
        size = _get_dec(sk, "size_mm", ck, required=True)
        measured = size
        _forbid(sk, "measured_value", ck, "길이·직경")
        _get_dec(sk, "limit_value", ck, required=True)

    # -- 스키마: verdict·margin (verdict_mode 게이트, §4-7) --
    verdict = sk.get("verdict")
    if gated:
        if verdict is not None:
            ck.add(R_SCHEMA, f"clause_only 게이트인데 verdict={verdict!r} (null 강제)")
        if sk.get("margin") is not None:
            ck.add(R_SCHEMA, "clause_only 게이트인데 margin 기재됨 (null 강제)")
    else:
        if verdict not in (PASS, FAIL):
            ck.add(R_SCHEMA, f"verdict enum 위반: {verdict!r} (합격/불합격 2종 외 금지)")
        if rule != "none_permitted" and sk.get("margin") is None:
            ck.add(R_SCHEMA, "margin 결손 (full 모드 수치 한계 행)")

    # -- ISO 코드 ∈ 사상표 --
    codes = set(label_codes) if label_codes is not None else {r.defect_code for r in table.rows}
    if sk["defect_code"] not in codes:
        ck.add(R_ISO, f"defect_code={sk['defect_code']} 사상표 집합에 없음")

    # -- 조항 실존: limits(로더 경유) + 색인 등재 목록 --
    table_clauses = {r.clause_id for r in table.rows}
    if sk["clause_id"] not in table_clauses:
        ck.add(R_CLAUSE, f"clause_id={sk['clause_id']} limits 테이블에 없음")
    if clause_registry is not None and sk["clause_id"] not in set(clause_registry):
        ck.add(R_CLAUSE, f"clause_id={sk['clause_id']} 색인 등재 조항 목록에 없음")

    # -- 조합 실존 (active ∧ canonical, 검사 방법 축 포함, fallback 금지) --
    method = sk["inspection_method"]
    if method == "ALL":
        method = inspection_method or _DEFAULT_METHOD
    try:
        row = applicable_row(
            table,
            sk["defect_code"],
            sk["material"],
            method,
            sk["quality_scheme"],
            sk["quality_level"],
            t,
            limit_type=limit_type,
        )
    except (LookupError, ValueError) as e:
        ck.add(R_COMBO, f"조합 실존 위반: {e}")
        return None
    if row.clause_id != sk["clause_id"]:
        ck.add(R_CLAUSE_MISMATCH, f"행 clause_id={row.clause_id} ↔ 골격 {sk['clause_id']}")
    if row.inspection_method.value != sk["inspection_method"]:
        ck.add(
            R_SCHEMA,
            f"행 inspection_method={row.inspection_method.value} ↔ "
            f"골격 {sk['inspection_method']}",
        )

    # -- 부등식 재계산 (limit_eval 공유 함수 — 재구현 금지) --
    try:
        expected_L = effective_limit(row, basis_value=t)
    except ValueError as e:
        ck.add(R_RECOMPUTE, f"유효 한계 산출 불능: {e}")
        return row
    sk_L = sk.get("limit_value")
    if (expected_L is None) != (sk_L is None):
        ck.add(R_RECOMPUTE, f"limit_value 재계산 불일치: 재계산={expected_L} ↔ 골격={sk_L}")
    elif expected_L is not None and to_decimal(sk_L) != expected_L:
        ck.add(R_RECOMPUTE, f"limit_value 재계산 불일치: 재계산={expected_L} ↔ 골격={sk_L}")

    if measured is None:
        return row
    try:
        j = judge(row, measured, basis_value=t)
    except ValueError as e:
        ck.add(R_SCHEMA, f"측정값 평가 불능: {e}")
        return row
    if not gated:
        if verdict in (PASS, FAIL) and j.verdict != verdict:
            ck.add(R_RECOMPUTE, f"verdict 재계산 불일치: judge={j.verdict} ↔ 골격={verdict}")
        sk_margin = sk.get("margin")
        if (j.margin is None) != (sk_margin is None):
            ck.add(R_RECOMPUTE, f"margin 재계산 불일치: judge={j.margin} ↔ 골격={sk_margin}")
        elif j.margin is not None and sk_margin is not None:
            try:
                if to_decimal(sk_margin) != j.margin:
                    ck.add(R_RECOMPUTE, f"margin 재계산 불일치: judge={j.margin} ↔ 골격={sk_margin}")
            except (ValueError, TypeError, InvalidOperation) as e:
                ck.add(R_SCHEMA, f"margin 수치 아님: {e}")
    return row



def _artifact_checks(text: str, basis: Optional[str], gated: bool, ck: "_Check") -> None:
    """생성문 아티팩트 + clause_only 판정 함의 표현 (80번 G3-1·G2-6).

    `basis` 는 그 레코드의 [자료] 문자열이다. 자료에 있는 표기는 아티팩트가 아니다 —
    자료가 그렇게 줬으면 옮겨 적은 것이 맞고, 고칠 곳은 자료 쪽이다.

    **`None` 이면 검사하지 않는다.** 자료 없이 표기만 막으면 판정 기준이 없어 "무엇에
    비해 아티팩트인가"를 말할 수 없다. 빈 문자열은 다르다 — 자료가 비었다는 뜻이라 어떤
    표기도 면제되지 않는다. 생성 경로가 자료를 넘기는지는 배선 시험이 강제한다
    (`tests/corpus/test_canonical_wiring.py`).
    """
    if basis is None:
        return
    hits = find_artifacts(text, basis=basis)
    if hits:
        ck.add(R_ARTIFACT, f"생성문 아티팩트: {list(hits)}")
    if gated:
        imply = find_verdict_implying(text)
        if imply:
            ck.add(R_ARTIFACT, f"clause_only 인데 판정 함의 표현: {list(imply)}")


def _check_length(text: str, ck: _Check, max_len: Optional[int], length_fn: Callable[[str], int]) -> None:
    if max_len is not None and length_fn(text) > max_len:
        ck.add(R_LONG, f"길이 초과: {length_fn(text)} > {max_len} (자르지 않고 폐기)")


# ---------------------------------------------------------------------------
# 자산별 검사기
# ---------------------------------------------------------------------------


def check_record_c(
    record: Mapping,
    table: LimitsTable,
    *,
    verdict_mode: str = "full",
    basis: Optional[str] = None,
    label_codes: Optional[Collection[str]] = None,
    clause_registry: Optional[Collection[str]] = None,
    allowed_standards: Optional[Collection[str]] = None,
    inspection_method: Optional[str] = None,
    max_len: Optional[int] = None,
    length_fn: Callable[[str], int] = len,
    index: int = 0,
) -> RecordResult:
    """(c) 판정추론 레코드 = 골격 + 생성문.

    inspection_method 를 주면 그 축의 레코드만 통과시킨다 (§1-2 5a). 주지 않으면 골격에
    실린 축으로 행을 조회한다 — 어느 쪽이든 축 없는 조회는 없으므로, RT·VT 가 병존하는
    테이블을 그대로 넘겨도 정상 레코드가 '복수 행 매칭'으로 폐기되지 않는다.

    verdict_mode 는 기본 full 이다 (§4-7: (c) 경로 골격은 full 이 정본). `clause_only` 는
    합부를 말하지 않는 축의 corpus 를 같은 검사기로 받기 위한 게이트이며, verdict·margin
    null 강제 + 판정 함의 표현 금지가 함께 걸린다.

    basis 는 그 레코드의 [자료] 문자열이다 (`corpus.generate.basis.render_basis`).
    아티팩트 게이트가 "자료에 없는 표기"를 판정하는 기준이라 비워 두면 검사가 헐거워진다.
    """
    if verdict_mode not in ("full", "conditional", "clause_only"):
        raise ValueError(f"verdict_mode 위반: {verdict_mode!r}")
    gated = verdict_mode == "clause_only"
    ck = _Check()
    _check_skeleton(
        record, table, ck, label_codes=label_codes, clause_registry=clause_registry,
        gated=gated, inspection_method=inspection_method,
    )
    text = record.get("text")
    if not isinstance(text, str) or not text.strip():
        ck.add(R_SCHEMA, "text(생성문) 결손")
    else:
        std = STANDARD_WHITELIST_DEFAULT if allowed_standards is None else allowed_standards
        ck.merge_lock(check_numeric_lock(text, record, allowed_standards=std))
        _artifact_checks(text, basis, gated, ck)
        _check_length(text, ck, max_len, length_fn)
    return ck.result(index, record.get("sample_id"))


def check_record_pair(
    record: Mapping,
    table: LimitsTable,
    *,
    verdict_mode: str = "clause_only",
    basis: Optional[str] = None,
    label_codes: Optional[Collection[str]] = None,
    clause_registry: Optional[Collection[str]] = None,
    defect_lexicon: Collection[str] = (),
    allowed_standards: Optional[Collection[str]] = None,
    inspection_method: Optional[str] = None,
    max_len: Optional[int] = None,
    length_fn: Callable[[str], int] = len,
    index: int = 0,
) -> RecordResult:
    """D4 페어 (정상 페어 포함 — defects=[] 가 같은 경로를 지난다, §4-6-1).

    verdict_mode ∈ {full, conditional, clause_only}. clause_only 면 verdict·margin 은
    이미지·결함 수준 모두 None 강제 (§4-7). conditional 은 full 과 같은 값 강제이며
    가정값 본문 명시 검사는 2단계(Judge) 소관이다.
    """
    if verdict_mode not in ("full", "conditional", "clause_only"):
        raise ValueError(f"verdict_mode 위반: {verdict_mode!r}")
    gated = verdict_mode == "clause_only"
    ck = _Check()

    defects = record.get("defects")
    cited = record.get("cited_clauses")
    if not isinstance(defects, (list, tuple)):
        ck.add(R_SCHEMA, "defects 배열 결손")
        return ck.result(index, record.get("sample_id"))
    if not isinstance(cited, (list, tuple)):
        ck.add(R_SCHEMA, "cited_clauses 배열 결손")
        cited = []
    verdict = record.get("verdict")
    text = record.get("text")

    if not defects:
        # ---- 정상 이미지 페어 (§4-6-1 ⑤): defects=[] ⇒ 인용 공집합 + 모드별 verdict ----
        if list(cited):
            ck.add(R_NORMAL, f"정상 페어인데 cited_clauses={list(cited)} (공집합 강제)")
        expected = None if gated else aggregate_verdicts([])  # full ⇒ 합격
        if verdict != expected:
            ck.add(R_NORMAL, f"정상 페어 verdict={verdict!r} ↔ 기대 {expected!r} (mode={verdict_mode})")
        if not isinstance(text, str) or not text.strip():
            ck.add(R_SCHEMA, "text(생성문) 결손")
        else:
            ck.merge_lock(
                check_normal_lock(
                    text,
                    defect_lexicon=defect_lexicon,
                    thickness_mm=record.get("thickness_mm"),
                    quality_level=str(record.get("quality_level") or ""),
                    expected_verdict=expected,
                    allowed_standards=() if allowed_standards is None else allowed_standards,
                )
            )
            _artifact_checks(text, basis, gated, ck)
            _check_length(text, ck, max_len, length_fn)
        return ck.result(index, record.get("sample_id"))

    # ---- 결함 페어: 결함별 골격 검사 + 이미지 수준 조립 검사 (§4-6) ----
    defect_verdicts: list[str] = []
    clause_ids: list[str] = []
    for sk in defects:
        row = _check_skeleton(
            sk, table, ck, label_codes=label_codes, clause_registry=clause_registry,
            gated=gated, inspection_method=inspection_method,
        )
        if isinstance(sk.get("clause_id"), str):
            clause_ids.append(sk["clause_id"])
        v = sk.get("verdict")
        if v in (PASS, FAIL):
            defect_verdicts.append(v)
        del row

    want_cited = sorted(set(clause_ids))
    if list(cited) != want_cited:
        ck.add(R_PAIR, f"cited_clauses={list(cited)} ↔ 정렬 합집합 {want_cited}")

    if gated:
        if verdict is not None:
            ck.add(R_PAIR, f"clause_only 게이트인데 이미지 verdict={verdict!r} (null 강제)")
    else:
        if len(defect_verdicts) == len(defects):
            agg = aggregate_verdicts(defect_verdicts)
            if verdict != agg:
                ck.add(R_PAIR, f"이미지 verdict={verdict!r} ↔ aggregate_verdicts={agg!r}")
        elif verdict not in (PASS, FAIL):
            ck.add(R_SCHEMA, f"이미지 verdict enum 위반: {verdict!r}")

    if not isinstance(text, str) or not text.strip():
        ck.add(R_SCHEMA, "text(생성문) 결손")
    else:
        ck.merge_lock(
            check_pair_lock(
                text,
                defects,
                image_verdict=verdict if not gated else None,
                allowed_standards=(
                    STANDARD_WHITELIST_DEFAULT if allowed_standards is None else allowed_standards
                ),
            )
        )
        _artifact_checks(text, basis, gated, ck)
        _check_length(text, ck, max_len, length_fn)
    return ck.result(index, record.get("sample_id"))


_WS_RE = re.compile(r"\s+")


def normalize_question(q: str) -> str:
    """(b) QA 중복 제거용 질문 정규화: NFKC → casefold → 공백 제거."""
    return _WS_RE.sub("", unicodedata.normalize("NFKC", q).casefold())


def check_record_qa(
    record: Mapping,
    *,
    passage_ids: Optional[Collection[str]] = None,
    seen: Optional[set[str]] = None,
    max_len: Optional[int] = None,
    length_fn: Callable[[str], int] = len,
    index: int = 0,
) -> RecordResult:
    """(b) QA 레코드: 질문/답/근거 구절 ID 스키마 + 근거 실존 + 중복 + 길이.

    passage_ids=None 이면 근거 실존 검사를 건너뛴다 (parse 산출물 주입 전 단계) —
    본 검증 실행에서는 반드시 주입해야 한다.
    """
    ck = _Check()
    q = record.get("question")
    a = record.get("answer")
    ev = record.get("evidence_passage_ids")
    if not isinstance(q, str) or not q.strip():
        ck.add(R_SCHEMA, "question 결손")
        q = ""
    if not isinstance(a, str) or not a.strip():
        ck.add(R_SCHEMA, "answer 결손")
        a = ""
    if not isinstance(ev, (list, tuple)) or not ev or not all(isinstance(e, str) and e for e in ev):
        ck.add(R_SCHEMA, "evidence_passage_ids 결손 (비어 있지 않은 문자열 배열 필수)")
        ev = []
    if passage_ids is not None:
        known = set(passage_ids)
        for e in ev:
            if e not in known:
                ck.add(R_EVIDENCE, f"근거 구절 ID 실존 위반: {e}")
    if q and seen is not None:
        key = normalize_question(q)
        if key in seen:
            ck.add(R_DUP, "질문 정규화 후 완전 일치 중복")
        else:
            seen.add(key)
    if q or a:
        _check_length(q + a, ck, max_len, length_fn)
    return ck.result(index, record.get("sample_id"))


# ---------------------------------------------------------------------------
# 배치 러너
# ---------------------------------------------------------------------------


def run_stage1(
    records: Sequence[Mapping],
    *,
    asset: str,
    table: Optional[LimitsTable] = None,
    basis_fn: Optional[Callable[[Mapping], str]] = None,
    label_codes: Optional[Collection[str]] = None,
    clause_registry: Optional[Collection[str]] = None,
    defect_lexicon: Collection[str] = (),
    allowed_standards: Optional[Collection[str]] = None,
    passage_ids: Optional[Collection[str]] = None,
    verdict_mode: Optional[str] = None,
    inspection_method: Optional[str] = None,
    max_len: Optional[int] = None,
    length_fn: Callable[[str], int] = len,
) -> Stage1Report:
    """자산(asset ∈ {c, d4, b}) 전수 1단계 검사. 실패분은 폐기 대상 (재생성 금지).

    inspection_method: 기대 검사 방법 축 (§1-2 5a). 주면 다른 축의 레코드를 폐기한다.
    basis_fn: 레코드 → [자료] 문자열. 아티팩트 게이트의 판정 기준이다 (80번 G3-1).
      생성기가 `corpus.generate.basis.render_basis` 를 넘긴다 — 생성 프롬프트·판정
      프롬프트·이 검사가 **같은 자료**를 보게 하는 것이 요점이다 (G4-1).
    """
    if asset not in ("c", "d4", "b"):
        raise ValueError(f"asset 위반: {asset!r} (c | d4 | b)")
    if asset in ("c", "d4") and table is None:
        raise ValueError("(c)·D4 검사는 LimitsTable 주입 필수 (limits.csv 직접 읽기 금지)")
    # 자산별 기본 모드. (c) 는 full 이 정본이고(§4-7) D4 는 clause_only 가 현재 축이다.
    # 한 값을 두 자산에 공유하면 (c) 골격의 verdict 가 통째로 스키마 위반이 된다.
    if verdict_mode is None:
        verdict_mode = "clause_only" if asset == "d4" else "full"

    results: list[RecordResult] = []
    seen: set[str] = set()
    for i, rec in enumerate(records):
        basis = basis_fn(rec) if basis_fn is not None else None
        if asset == "c":
            r = check_record_c(
                rec, table, verdict_mode=verdict_mode, basis=basis,
                label_codes=label_codes, clause_registry=clause_registry,
                allowed_standards=allowed_standards, inspection_method=inspection_method,
                max_len=max_len, length_fn=length_fn, index=i,
            )
        elif asset == "d4":
            r = check_record_pair(
                rec, table, verdict_mode=verdict_mode, basis=basis, label_codes=label_codes,
                clause_registry=clause_registry, defect_lexicon=defect_lexicon,
                allowed_standards=allowed_standards, inspection_method=inspection_method,
                max_len=max_len, length_fn=length_fn, index=i,
            )
        else:
            r = check_record_qa(
                rec, passage_ids=passage_ids, seen=seen, max_len=max_len,
                length_fn=length_fn, index=i,
            )
        results.append(r)

    counter: Counter[str] = Counter()
    for r in results:
        for reason in r.reasons:
            counter[reason] += 1
    n_pass = sum(1 for r in results if r.ok)
    return Stage1Report(
        asset=asset,
        n_in=len(results),
        n_pass=n_pass,
        n_fail=len(results) - n_pass,
        fail_reasons=dict(sorted(counter.items())),
        results=tuple(results),
    )

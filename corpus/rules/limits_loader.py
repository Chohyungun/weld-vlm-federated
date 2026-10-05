"""corpus/limits.csv 공식 로더 — G0 V0~V11 내장, 실패 시 로드 자체가 거부된다.

스펙: 11_spec_B_코퍼스합성.md §1-2 (G0 검증 게이트), §1-3 (표현 규약), §1-4 (단일 소스).

금지 규칙 (§1-4 ①): limits_loader 외 경로로 limits.csv 를 읽지 않는다.

fail-closed: 위반 시 LimitsValidationError (위반 행 번호 목록 포함), 부분 통과 없음.
pilot=True 는 V0(승격 해시 결속)·검수 컬럼 검사·V5(sources 부재 시)만 면제하고
테이블에 pilot=True 를 표시 + 로거 경고를 남긴다. 그 외 G0 검사는 전부 수행한다.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import unicodedata
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import yaml
from pydantic import ValidationError

from corpus.rules.limit_eval import effective_limit
from corpus.rules.schema import (
    ISO5817_LEVEL_ORDER,
    InspectionMethod,
    LimitRow,
    LimitRule,
    LimitsTable,
    Material,
    Q,
    QualityScheme,
    RatioBasis,
    Scope,
    Unit,
    quantize,
)

__all__ = ["load_limits", "coverage_report", "LimitsValidationError", "Violation", "EXPECTED_COLUMNS"]

logger = logging.getLogger("corpus.rules.limits_loader")

# 스펙 §1-2 컬럼 전수 (순서 포함 — 계약 #3 동결, 변경은 승인 사항)
# inspection_method 는 결정 #13 승인분(스펙 v1.3 §1-2 5a) — 동결 후 첫 계약 변경이다.
EXPECTED_COLUMNS: tuple[str, ...] = (
    "rule_id", "canonical", "scope", "defect_code", "material", "inspection_method",
    "thickness_min", "thickness_max", "quality_scheme", "quality_level",
    "limit_type", "limit_rule", "limit_value", "limit_factor", "limit_cap",
    "ratio_basis", "limit_op", "unit", "clause_id", "source_doc",
    "source_page", "source_table_id", "source_row_label", "limit_expr",
    "note", "restated_from", "filled_h", "filled_v", "extraction_path",
    "verified_by", "verified_date",
)

# 공란('') → None 으로 해석하는 컬럼. 나머지는 필수값 (공란이면 행 검증 실패).
_OPTIONAL_COLUMNS = frozenset({
    "thickness_max", "limit_value", "limit_factor", "limit_cap", "ratio_basis",
    "unit", "source_page", "source_table_id", "source_row_label", "limit_expr",
    "note", "restated_from", "verified_by", "verified_date",
})

# NFKC 정규화 대상 (전각 숫자·㎜ 등 호환 문자 → 표준형). 그 외 컬럼은 NFC 만.
_NFKC_COLUMNS = frozenset({
    "thickness_min", "thickness_max", "limit_value", "limit_factor", "limit_cap",
    "source_page", "unit",
})

_UNIT_ALIASES = {"mm": "mm", "%": "percent", "percent": "percent"}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LABEL_MAP_PATH = _REPO_ROOT / "configs" / "label_map.yaml"
_CORPUS_GEN_CONFIG_PATH = _REPO_ROOT / "configs" / "corpus_gen.yaml"

_FEASIBILITY_MIN_L = Decimal("0.1")  # V10: 그리드 10칸
_RATIO_FAIL_MULT = Decimal("1.1")    # V10 비율 상한: 불합격 버킷 하단 계수 (§4-3)
# V10 비율 상한: 표본 상한 min(2.5L, 100%). 정수 리터럴로 둔다 — 문자열 "100" 은 ISO 6520-1
# 균열 코드와 같은 리터럴이라 라벨 하드코딩 검사에 걸린다.
_PERCENT_MAX = Decimal(100)


class Violation(NamedTuple):
    row: Optional[int]  # CSV 파일 행 번호 (헤더=1행, 첫 데이터 행=2행). 테이블 수준이면 None
    check: str
    message: str


class LimitsValidationError(Exception):
    """G0 실패. violations 에 (행 번호, 검사 ID, 메시지) 전건이 담긴다."""

    def __init__(self, violations: Sequence[Violation]):
        self.violations: tuple[Violation, ...] = tuple(violations)
        lines = [f"limits.csv G0 실패 — 위반 {len(self.violations)}건 (위반 행: {self.row_numbers})"]
        lines += [
            f"  행 {v.row if v.row is not None else '-'} [{v.check}] {v.message}"
            for v in self.violations
        ]
        super().__init__("\n".join(lines))

    @property
    def row_numbers(self) -> list[int]:
        return sorted({v.row for v in self.violations if v.row is not None})


# ---------------------------------------------------------------------------
# 입력 정규화·행 파싱
# ---------------------------------------------------------------------------


def _normalize_cell(col: str, raw: str) -> str:
    form = "NFKC" if col in _NFKC_COLUMNS else "NFC"
    v = unicodedata.normalize(form, raw).strip()
    if col == "unit" and v:
        v = _UNIT_ALIASES.get(v, v)
    return v


def _row_kwargs(rowdict: dict[str, str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for col in EXPECTED_COLUMNS:
        v = _normalize_cell(col, rowdict.get(col) or "")
        if v == "" and col in _OPTIONAL_COLUMNS:
            kwargs[col] = None
        else:
            kwargs[col] = v
    return kwargs


def _parse_rows(
    reader: csv.DictReader, violations: list[Violation]
) -> list[tuple[int, LimitRow]]:
    parsed: list[tuple[int, LimitRow]] = []
    seen_rule_ids: dict[str, int] = {}
    for i, rowdict in enumerate(reader):
        line_no = i + 2  # 헤더 = 1행
        if rowdict.get(None):
            violations.append(Violation(line_no, "V1", "필드 수가 헤더보다 많음 (래그드 행)"))
            continue
        try:
            row = LimitRow(**_row_kwargs(rowdict))
        except (ValidationError, ValueError, InvalidOperation) as e:
            for msg in _error_messages(e):
                check = _check_tag(msg)
                violations.append(Violation(line_no, check, msg))
            continue
        if row.rule_id in seen_rule_ids:
            violations.append(Violation(
                line_no, "V1",
                f"rule_id 중복(PK 위반): {row.rule_id} (행 {seen_rule_ids[row.rule_id]}과 충돌)",
            ))
            continue
        seen_rule_ids[row.rule_id] = line_no
        parsed.append((line_no, row))
    return parsed


def _error_messages(e: Exception) -> list[str]:
    if isinstance(e, ValidationError):
        out = []
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "-"
            msg = err["msg"]
            # pydantic 이 "Value error, " 를 앞에 붙인 자체 메시지는 원문만 남긴다
            msg = msg.removeprefix("Value error, ")
            out.append(f"{loc}: {msg}" if not msg.startswith("[") else msg)
        return out
    return [str(e)]


def _check_tag(msg: str) -> str:
    if msg.startswith("[") and "]" in msg:
        return msg[1 : msg.index("]")]
    return "V1"


# ---------------------------------------------------------------------------
# 외부 참조 로딩 (사상표 · sources.yaml · 색인 등재 조항 목록)
# ---------------------------------------------------------------------------


def _load_label_codes(path: Path = _LABEL_MAP_PATH) -> Optional[set[str]]:
    """계약 #1 사상표에서 ISO 코드 집합 (iso_code + iso_code_alt)."""
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    codes: set[str] = set()
    for spec in (data.get("defect_types") or {}).values():
        if spec.get("iso_code"):
            codes.add(str(spec["iso_code"]))
        codes.update(str(c) for c in (spec.get("iso_code_alt") or []))
    return codes


def _load_sources(src: Any) -> dict[str, dict]:
    if isinstance(src, (str, Path)):
        data = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    else:
        data = src
    if isinstance(data, dict) and "documents" in data:
        entries = data["documents"] or []
        return {e["doc_id"]: e for e in entries}
    if isinstance(data, dict):
        return {k: (v or {}) for k, v in data.items() if isinstance(v, dict)}
    return {e["doc_id"]: e for e in (data or [])}


def _load_clause_registry(reg: Any) -> set[str]:
    if isinstance(reg, (set, frozenset, list, tuple)):
        return {str(c) for c in reg}
    p = Path(reg)
    if p.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("clauses") or data.get("clause_ids") or list(data.keys())
        return {str(c) for c in (data or [])}
    out: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.add(line.split(",")[0].strip())
    return out


# ---------------------------------------------------------------------------
# V0 — 승격 결속 (accepted/ ↔ 처리 이력 ↔ 등록부 해시)
# ---------------------------------------------------------------------------


def _check_v0(path: Path, sources: Optional[dict[str, dict]], violations: list[Violation]) -> None:
    parse_dir = next(
        (c for c in (path.parent / "parse", path.parent.parent / "parse") if c.is_dir()),
        None,
    )
    if parse_dir is None:
        violations.append(Violation(None, "V0", f"corpus/parse 디렉터리 미발견 ({path.parent})"))
        return
    log_path = parse_dir / "processing_log.jsonl"
    accepted_dir = parse_dir / "accepted"
    if not log_path.exists() or not accepted_dir.is_dir():
        violations.append(Violation(
            None, "V0",
            f"processing_log.jsonl 또는 accepted/ 미발견 ({parse_dir}) — 승격 결속 검증 불능",
        ))
        return

    logged: dict[str, tuple[dict, str]] = {}
    for ln, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") != "accepted":
            continue
        out = rec.get("output_path")
        sha = rec.get("output_sha256") or (rec.get("judgement") or {}).get("output_sha256")
        if not out or not sha:
            violations.append(Violation(None, "V0", f"이력 {ln}행: accepted 인데 output_path/sha256 누락"))
            continue
        logged[Path(out).name] = (rec, sha)

    files = {p.name: p for p in accepted_dir.iterdir() if p.is_file()}
    for name in sorted(set(logged) - set(files)):
        violations.append(Violation(None, "V0", f"이력에는 있으나 accepted/ 에 없는 파일: {name}"))
    for name in sorted(set(files) - set(logged)):
        violations.append(Violation(None, "V0", f"accepted/ 에 있으나 이력에 없는 파일 (무감사 반입): {name}"))
    for name in sorted(set(logged) & set(files)):
        rec, sha = logged[name]
        actual = hashlib.sha256(files[name].read_bytes()).hexdigest()
        if actual != sha:
            violations.append(Violation(None, "V0", f"{name}: 파일 sha256 ≠ 이력 기록 해시 (무감사 사후 수정)"))
        doc_sha = rec.get("doc_sha256")
        if sources is not None and doc_sha:
            entry = sources.get(rec.get("doc_id", ""), {})
            if entry.get("sha256") and entry["sha256"] != doc_sha:
                violations.append(Violation(
                    None, "V0", f"{name}: 문서 sha256 ≠ 등록부 해시 (doc_id={rec.get('doc_id')})"
                ))


# ---------------------------------------------------------------------------
# 테이블 수준 검사 V3 · V4 · V6 · V9 · V10
# ---------------------------------------------------------------------------


def _fmt(row: LimitRow) -> str:
    hi = "+inf" if row.thickness_max is None else str(row.thickness_max)
    return f"{row.rule_id}[{row.thickness_min},{hi})"


def _overlap(a: LimitRow, b: LimitRow) -> bool:
    a_hi_gt = b.thickness_min < a.thickness_max if a.thickness_max is not None else True
    b_hi_gt = a.thickness_min < b.thickness_max if b.thickness_max is not None else True
    return a_hi_gt and b_hi_gt


def _group(rows: list[LimitRow], key) -> dict[tuple, list[LimitRow]]:
    out: dict[tuple, list[LimitRow]] = {}
    for r in rows:
        out.setdefault(key(r), []).append(r)
    return out


def _check_v3(rows: list[LimitRow], violations: list[Violation]) -> None:
    """구간 무결성: active ∧ canonical 그룹 내 반열림 [min,max) 겹침·틈 없음.

    그룹 키에 inspection_method 포함 (게이트 #13) — 표면(VT)과 내부(RT) 기준은 같은
    결함코드·같은 두께 구간에 정당하게 공존한다. 축이 없으면 정상 병존이 겹침으로 오탐된다.
    """
    target = [r for r in rows if r.scope is Scope.ACTIVE and r.canonical]
    groups = _group(
        target,
        lambda r: (r.material, r.inspection_method, r.defect_code,
                   r.quality_scheme, r.quality_level, r.limit_type),
    )
    for key, grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
        grp = sorted(grp, key=lambda r: (r.thickness_min, r.rule_id))
        for prev, nxt in zip(grp, grp[1:]):
            gkey = f"그룹 {tuple(getattr(k, 'value', k) for k in key)}"
            if prev.thickness_max is None or nxt.thickness_min < prev.thickness_max:
                violations.append(Violation(
                    None, "V3", f"{gkey}: 구간 겹침 {_fmt(prev)} ↔ {_fmt(nxt)}"
                ))
            elif nxt.thickness_min > prev.thickness_max:
                violations.append(Violation(
                    None, "V3", f"{gkey}: 구간 틈 {_fmt(prev)} ↔ {_fmt(nxt)}"
                ))


def _check_v4(rows: list[LimitRow]) -> list[str]:
    """품질수준 단조성 (경보, 예외 아님). iso5817 스킴 한정, B ≤ C ≤ D.

    평가 지점: 구간 교집합 양 끝점 + prop_t_cap 교차점 t* = cap/factor.
    """
    warnings: list[str] = []
    # ratio_basis=a 행은 공유 평가기가 평가 보류로 거부하므로 단조성 비교 대상이 아니다
    # (열린 질문 3 — 목두께 값이 데이터에 없다).
    target = [
        r for r in rows
        if r.scope is Scope.ACTIVE and r.canonical
        and r.quality_scheme is QualityScheme.ISO5817
        and r.limit_rule is not LimitRule.NONE_PERMITTED
        and r.ratio_basis is not RatioBasis.A
    ]
    # 그룹 키에 inspection_method 포함 — RT 행과 VT 행의 한계를 나란히 놓고 단조성을
    # 따지면 근거 없는 경보가 난다 (비교 축이 다르다).
    groups = _group(
        target, lambda r: (r.material, r.inspection_method, r.defect_code, r.limit_type)
    )
    order = {lv: i for i, lv in enumerate(ISO5817_LEVEL_ORDER)}
    for _key, grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
        by_level = _group(grp, lambda r: r.quality_level)
        levels = sorted(by_level, key=lambda lv: order[lv])
        for i, strict in enumerate(levels):
            for lenient in levels[i + 1:]:
                for rs in by_level[strict]:
                    for rl in by_level[lenient]:
                        if not _overlap(rs, rl):
                            continue
                        lo = max(rs.thickness_min, rl.thickness_min)
                        his = [x for x in (rs.thickness_max, rl.thickness_max) if x is not None]
                        hi = min(his) if his else None
                        pts = {lo}
                        if hi is not None and hi - Q >= lo:
                            pts.add(hi - Q)
                        for r in (rs, rl):
                            if r.limit_rule is LimitRule.PROP_T_CAP:
                                tstar = quantize(r.limit_cap / r.limit_factor)
                                if tstar >= lo and (hi is None or tstar < hi):
                                    pts.add(tstar)
                        for t in sorted(pts):
                            ls = effective_limit(rs, basis_value=t)
                            ll = effective_limit(rl, basis_value=t)
                            if ls > ll:
                                warnings.append(
                                    f"[V4] 단조성 경보: t={t} 에서 {strict}({rs.rule_id}) L={ls} > "
                                    f"{lenient}({rl.rule_id}) L={ll}"
                                )
    return warnings


def _check_v6(parsed: list[tuple[int, LimitRow]], violations: list[Violation]) -> None:
    """유일성. 두 키 모두 inspection_method 를 포함한다 (게이트 #13, 스펙 §1-2 V6)."""
    seen: dict[tuple, int] = {}
    for line_no, r in parsed:
        k = (r.material, r.inspection_method, r.defect_code, r.thickness_min,
             r.thickness_max, r.quality_level, r.limit_type, r.clause_id)
        if k in seen:
            violations.append(Violation(
                line_no, "V6",
                "(material, inspection_method, defect, 구간, 수준, limit_type, clause_id) "
                f"중복 — 행 {seen[k]}과 동일 조합",
            ))
        else:
            seen[k] = line_no
    canon: dict[tuple, int] = {}
    for line_no, r in parsed:
        if not r.canonical:
            continue
        k = (r.material, r.inspection_method, r.defect_code, r.thickness_min,
             r.thickness_max, r.quality_scheme, r.quality_level, r.limit_type)
        if k in canon:
            violations.append(Violation(
                line_no, "V6", f"같은 조합에 canonical 행 2개 이상 — 행 {canon[k]}과 충돌"
            ))
        else:
            canon[k] = line_no


def _limit_params(r: LimitRow) -> tuple:
    return (r.limit_rule, r.limit_value, r.limit_factor, r.limit_cap,
            r.limit_op, r.unit, r.ratio_basis)


def _check_v9(rows: list[LimitRow], violations: list[Violation]) -> list[str]:
    """교차 문서 충돌: 같은 조합(구간 교집합 존재)인데 한계 파라미터가 다른 행 쌍 전수 검출.

    canonical 로 정본이 지정돼 있으면 해소된 충돌로 기록만 하고,
    미지정(양쪽 비정본) 또는 이중 정본이면 오류다. 스킴 간 동치 수준(grade_map) 비교는
    grade_map.yaml 확정 전이라 미구현 — 같은 (스킴, 수준)만 비교한다.

    그룹 키에 inspection_method 포함 (게이트 #13). RT 행과 VT 행의 한계가 다른 것은
    충돌이 아니라 설계다 — 축 없이 두면 정당한 병존이 "미해소 충돌"로 몰려 한쪽을
    비정본으로 눕히게 되고, 그 순간 표면·내부 중 하나가 통째로 사라진다.
    """
    resolved: list[str] = []
    groups = _group(
        rows,
        lambda r: (r.material, r.inspection_method, r.defect_code,
                   r.quality_scheme, r.quality_level, r.limit_type),
    )
    for _key, grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
        grp = sorted(grp, key=lambda r: r.rule_id)
        for i, a in enumerate(grp):
            for b in grp[i + 1:]:
                if a.source_doc == b.source_doc:
                    continue
                if not _overlap(a, b):
                    continue
                if _limit_params(a) == _limit_params(b):
                    continue
                pair = f"{a.rule_id}({a.source_doc}) ↔ {b.rule_id}({b.source_doc})"
                if a.canonical and b.canonical:
                    violations.append(Violation(
                        None, "V9", f"교차 문서 충돌인데 양쪽 모두 canonical: {pair}"
                    ))
                elif a.canonical or b.canonical:
                    winner = a.rule_id if a.canonical else b.rule_id
                    resolved.append(f"[V9] 충돌 해소됨(정본={winner}): {pair}")
                else:
                    violations.append(Violation(
                        None, "V9",
                        f"교차 문서 충돌 미해소(canonical 미지정): {pair} — 수기 판정 큐 대상",
                    ))
    return resolved


def _check_v10(rows: list[LimitRow], skipped: list[str]) -> list[str]:
    """샘플링 실현성 (보고, 판단 대상).

    ① 하한: 구간 최소 t 기준 유효 한계 L ≥ 0.1 (그리드 10칸). prop 계열은 생성기가 t 하한을
       실현 가능한 최소 그리드 값으로 올려 살려내므로(§4-3 경계 두께 배치의 보정) 여기서는
       보고만 하고, const 계열처럼 t 로 살릴 수 없는 행은 생성 시점에 전체 중단된다.
    ② 비율형 상한: unit=percent 행은 1.1L < 100% 여야 불합격 버킷 (1.1L, min(2.5L,100%)] 이
       공집합이 아니다. 하한만 검사하면 L=95% 류가 로드를 통과한 뒤 생성 전량을 중단시킨다
       (검토 N3). 상한 미달 행도 실현성 미달로 올려 scope=excluded 판단 경로를 태운다.
    """
    flags: list[str] = []
    active = [r for r in rows if r.scope is Scope.ACTIVE]
    for r in active:
        if r.limit_rule is LimitRule.NONE_PERMITTED:
            continue
        if r.ratio_basis is RatioBasis.A:
            flags.append(
                f"[V10] 평가 보류: {r.rule_id} — ratio_basis=a(목두께)는 값이 데이터에 없어 "
                "실현성 판정 자체가 불가하다 (열린 질문 3, 수기 판정 큐)"
            )
            continue
        L = effective_limit(r, basis_value=r.thickness_min)
        if L < _FEASIBILITY_MIN_L:
            flags.append(
                f"[V10] 실현성 미달: {r.rule_id} — t_min={r.thickness_min} 에서 L={L} < 0.1 "
                "(scope=excluded 여부 판단 대상)"
            )
        if r.unit is Unit.PERCENT:
            L_max = effective_limit(r, basis_value=r.thickness_max or r.thickness_min)
            if _RATIO_FAIL_MULT * L_max >= _PERCENT_MAX:
                flags.append(
                    f"[V10] 실현성 미달(비율 상한): {r.rule_id} — L={L_max}% 에서 "
                    f"1.1L ≥ {_PERCENT_MAX}% 라 불합격 버킷이 공집합이다 "
                    "(scope=excluded 여부 판단 대상)"
                )
    # configs 교차 검증: t_sample_max > 활성 행 최대 thickness_min
    if _CORPUS_GEN_CONFIG_PATH.exists():
        cfg = yaml.safe_load(_CORPUS_GEN_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        t_max = cfg.get("t_sample_max", (cfg.get("sampling") or {}).get("t_sample_max"))
        if t_max is None:
            skipped.append("V10-configs(t_sample_max 키 없음)")
        elif active:
            max_tmin = max(r.thickness_min for r in active)
            if Decimal(str(t_max)) <= max_tmin:
                flags.append(
                    f"[V10] configs 교차 검증 실패: t_sample_max={t_max} ≤ 활성 행 최대 "
                    f"thickness_min={max_tmin}"
                )
    else:
        skipped.append("V10-configs(corpus_gen.yaml 부재)")
    return flags


# ---------------------------------------------------------------------------
# 공식 로더
# ---------------------------------------------------------------------------


def load_limits(
    path: str | Path,
    sources_yaml: Any = None,
    clause_registry: Any = None,
    pilot: bool = False,
) -> LimitsTable:
    """limits.csv 를 G0 전체 검증 후 LimitsTable 로 로드한다. 실패 시 부분 통과 없음.

    sources_yaml: 문서 등록부 (경로 | dict). None 이면 pilot 에서만 V5·source_doc 검사 면제.
    clause_registry: 색인 등재 조항 목록 (경로 | 문자열 컬렉션). None 이면 pilot 에서만 V8 면제.
    pilot=True: V0·검수 컬럼 검사·V5(sources 부재 시) 면제 + 테이블 pilot 표시 + 로거 경고.
    """
    p = Path(path)
    raw = p.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()  # V11 — 통과본 해시 기록

    violations: list[Violation] = []
    skipped: list[str] = []

    # --- 헤더 (V1 컬럼 전수) ---
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = tuple(reader.fieldnames or ())
    missing = [c for c in EXPECTED_COLUMNS if c not in fields]
    unknown = [c for c in fields if c not in EXPECTED_COLUMNS]
    if missing:
        violations.append(Violation(1, "V1", f"컬럼 누락: {missing}"))
    if unknown:
        violations.append(Violation(1, "V1", f"미정의 컬럼 (계약 #3 밖): {unknown}"))
    if missing or unknown:
        raise LimitsValidationError(violations)

    # --- 행 파싱 (V1 타입·enum·필수값 + V2 + V7 은 스키마 validator 가 수행) ---
    parsed = _parse_rows(reader, violations)
    rows = [r for _, r in parsed]

    # --- 외부 참조: 사상표 (V1 defect_code ∈ 사상표) ---
    label_codes = _load_label_codes()
    if label_codes is None:
        if pilot:
            skipped.append("V1-label_map(사상표 부재)")
            logger.warning("사상표(configs/label_map.yaml) 부재 — pilot 이므로 defect_code 검사 생략")
        else:
            violations.append(Violation(None, "V1", "사상표(configs/label_map.yaml) 부재 — 검증 불능"))
    else:
        for line_no, r in parsed:
            if r.defect_code not in label_codes:
                violations.append(Violation(
                    line_no, "V1", f"defect_code={r.defect_code} 사상표에 없음"
                ))

    # --- 외부 참조: sources.yaml (V1 source_doc 실존 + V5 라이선스 게이트) ---
    sources: Optional[dict[str, dict]] = None
    if sources_yaml is not None:
        sources = _load_sources(sources_yaml)
        for line_no, r in parsed:
            entry = sources.get(r.source_doc)
            if entry is None:
                violations.append(Violation(
                    line_no, "V1", f"source_doc={r.source_doc} 이 sources.yaml 에 미등록"
                ))
                continue
            if entry.get("license_class") == "PAID":
                violations.append(Violation(
                    line_no, "V5", f"license_class=PAID 문서({r.source_doc}) 유래 행 — 무조건 실패"
                ))
            elif "table_extract" not in (entry.get("allowed_use") or []):
                violations.append(Violation(
                    line_no, "V5", f"source_doc={r.source_doc}: allowed_use 에 table_extract 없음"
                ))
    elif pilot:
        skipped.append("V5(sources 부재)")
        logger.warning("sources.yaml 미제공 — pilot 이므로 V5·source_doc 검사 생략")
    else:
        violations.append(Violation(None, "V5", "sources.yaml 미제공 — 라이선스 게이트 검증 불능"))

    # --- 외부 참조: 색인 등재 조항 목록 (V8) ---
    if clause_registry is not None:
        registry = _load_clause_registry(clause_registry)
        for line_no, r in parsed:
            if r.clause_id not in registry:
                violations.append(Violation(
                    line_no, "V8", f"clause_id={r.clause_id} 색인 등재 조항 목록에 없음"
                ))
    elif pilot:
        skipped.append("V8(조항 목록 부재)")
        logger.warning("색인 등재 조항 목록 미제공 — pilot 이므로 V8 생략")
    else:
        violations.append(Violation(None, "V8", "색인 등재 조항 목록 미제공 — 참조 무결성 검증 불능"))

    # --- 검수 컬럼 검사 (pilot 면제): M 경로 행은 이중 입력 대조 완료 기록 필수 ---
    if pilot:
        skipped.append("검수컬럼")
    else:
        for line_no, r in parsed:
            if r.extraction_path.value == "M" and (not r.verified_by or not r.verified_date):
                violations.append(Violation(
                    line_no, "검수", "extraction_path=M 인데 verified_by/verified_date 미기록"
                ))

    # --- V0 승격 결속 (pilot 면제) ---
    if pilot:
        skipped.append("V0")
    else:
        _check_v0(p, sources, violations)

    # --- 테이블 수준 검사 (행 파싱 성공분에 대해) ---
    _check_v6(parsed, violations)
    _check_v3(rows, violations)
    v4_warnings = _check_v4(rows)
    v9_resolved = _check_v9(rows, violations)
    v10_flags = _check_v10(rows, skipped)

    if violations:
        raise LimitsValidationError(violations)

    if pilot:
        logger.warning(
            "limits.csv 를 pilot 모드로 로드했다 — 파일럿 전용·미검수. 스냅샷 비대상. "
            f"면제된 검사: {skipped}"
        )
    for w in v4_warnings:
        logger.warning(w)
    for f in v10_flags:
        logger.warning(f)

    return LimitsTable(
        rows=tuple(rows),
        sha256=sha256,
        source_path=str(p),
        pilot=pilot,
        v4_warnings=tuple(v4_warnings),
        v9_conflicts=tuple(v9_resolved),
        v10_flags=tuple(v10_flags),
        checks_skipped=tuple(skipped),
    )


# ---------------------------------------------------------------------------
# 커버리지 리포트 — 재질 × 결함 × 품질수준 격자 공백 (§1-2 말미)
# ---------------------------------------------------------------------------


def coverage_report(
    table: LimitsTable, inspection_method: InspectionMethod | str | None = None
) -> dict:
    """격자 공백 목록과 사유 분류. 실존 조합 열거(§4)·수량 보고(§7)의 근거 문서.

    사유 분류: excluded(행은 있으나 선언 배제) / missing(행 자체 부재 — 표준에 없음·미전사·
    유료 전용의 구분은 원천 조사 없이는 불가하므로 missing 으로 묶어 보고한다).

    inspection_method 를 주면 칸을 채우는 행을 그 축의 뷰({method, ALL})로 좁힌다. 주 실험은
    RT 이므로 RT 커버리지를 볼 때는 반드시 지정한다 — 지정하지 않으면 VT 행이 채운 칸이
    RT 공백을 가린다. 축 무관 호출도 `inspection_method_counts` 로 혼재 여부는 드러난다.

    격자의 축(결함코드·품질수준 어휘)은 뷰가 아니라 **원 테이블 전체**에서 뽑는다. 뷰에서
    뽑으면 그 축에 행이 하나도 없는 조합이 격자에서 사라져 공백이 보고되지 않는다 —
    커버리지 리포트가 정확히 못 해야 할 일이다.
    """
    view = table if inspection_method is None else table.for_inspection(inspection_method)
    label_codes = _load_label_codes()
    if label_codes:
        codes = sorted(label_codes)
    else:
        codes = sorted({r.defect_code for r in table.rows})
    materials = [Material.ST, Material.AL]
    levels = sorted({(r.quality_scheme.value, r.quality_level) for r in table.rows})

    cells: dict[str, str] = {}
    gaps: list[dict] = []
    for m in materials:
        for code in codes:
            for scheme, level in levels:
                hit = [
                    r for r in view.rows
                    if r.material in (m, Material.ALL)
                    and r.defect_code == code
                    and r.quality_scheme.value == scheme
                    and r.quality_level == level
                ]
                key = f"{m.value}|{code}|{scheme}:{level}"
                if any(r.scope is Scope.ACTIVE for r in hit):
                    cells[key] = "present"
                elif hit:
                    cells[key] = "excluded"
                    gaps.append({"material": m.value, "defect_code": code,
                                 "quality": f"{scheme}:{level}", "reason": "excluded"})
                else:
                    cells[key] = "missing"
                    gaps.append({"material": m.value, "defect_code": code,
                                 "quality": f"{scheme}:{level}", "reason": "missing"})
    im_counts: dict[str, int] = {}
    for r in view.rows:
        im_counts[r.inspection_method.value] = im_counts.get(r.inspection_method.value, 0) + 1
    return {
        "materials": [m.value for m in materials],
        "defect_codes": codes,
        "quality_levels": [f"{s}:{l}" for s, l in levels],
        "inspection_method": (
            None if inspection_method is None else InspectionMethod(inspection_method).value
        ),
        "inspection_method_counts": dict(sorted(im_counts.items())),
        "cells": cells,
        "gaps": gaps,
        "n_rows": len(view.rows),
        "n_active_canonical": len(view.active_canonical),
        "limits_sha256": table.sha256,
        "pilot": table.pilot,
    }

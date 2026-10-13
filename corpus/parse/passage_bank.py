"""구절 은행 — 원문 QA 생성기의 입력 계약 (2026-10-07, 행 꼴 2판 · 은행 v3).

은행은 두 벌이다.

- **포인터 은행**(추적): `corpus/parse/passage_bank_v3.jsonl`. 구절마다 문서 · 판본 · 원문 파일 지문 ·
  쪽 · 절 · 지식 면 · 결함 클래스 · 적용 범위 · 권리 상태 · 본문 지문을 싣는다. **본문은 싣지 않는다**(`text` = null).
  KR · IACS · ABS 약관이 원문 전재를 막고, 저장소가 공개이기 때문이다.
- **본문 은행**(저장소 밖, 추적하지 않음): 같은 행에 원문 텍스트 층에서 뽑은 본문을 싣는다.
  요약이나 재서술이 아니다(`text_kind = verbatim_extract`). 상용 모델이 쓴 문장을 생성기 입력에
  넣지 않으려는 것이다(규약 2-4).

생성기는 본문 은행을 읽되, 이 모듈의 `select_for_generation` 을 지난 구절만 받는다. 그 검사가
"원문 지문 · 판본 · 근거 위치 · 권리 · 사람 승인이 없는 구절은 생성기에 넘기지 않는다"(검토 §82)의
정본이다. **이 모듈은 권리나 사실을 승인하지 않는다** — 권리 결정은 사용자의 기록,
사실 검수는 사람의 기록에서 온다. 은행 행의 `rights_status` 는 생성 시점에 늘 `pending` 이다.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2

#: 지식 면. 여덟 면이다(정의 · 소견 · 원인 · 방지 · 보수 · 기준 · 측정 · 적용 조건).
FACETS: tuple[str, ...] = (
    "definition", "appearance", "cause", "prevention",
    "repair", "criteria", "measurement", "applicability",
)
FACET_KO: dict[str, str] = {
    "definition": "정의", "appearance": "소견", "cause": "원인", "prevention": "방지",
    "repair": "보수", "criteria": "기준", "measurement": "측정", "applicability": "적용 조건",
}
#: 현대 지식으로 채우면 안 되는 면 — 역사 자료(`historical_only`)에서는 이 면 표지를 달지 않는다.
MODERN_ONLY_FACETS: frozenset[str] = frozenset({"cause", "prevention", "repair", "criteria"})
#: 결함 클래스 — 은행 안의 분류 이름이다. ISO 코드는 사상표가 정한다(규약 1-8).
DEFECTS: tuple[str, ...] = (
    "crack", "porosity", "lack_of_fusion", "lack_of_penetration",
    "slag_inclusion", "undercut", "overlap",
)
RIGHTS_STATES: tuple[str, ...] = ("approved", "pending", "denied")
USES: tuple[str, ...] = ("qa_generation", "rag_index", "training_text")

#: 계보가 확인되지 않은 블록의 표지. 권리 결정 기록의 `cleared_lineage_flags` 에 그 표지가 있어야 넘긴다.
LINEAGE_FLAGS: frozenset[str] = frozenset({"L2_lineage_unverified", "UT_lineage_unverified"})
#: 추출 오류 · 범위 위험 표지. 사람 검수 기록의 `flags_checked` 에 그 표지가 있어야 넘긴다.
RISK_FLAGS: frozenset[str] = frozenset({
    "pua_unmapped", "kr_angle_glyph_means_lt", "hard_split_mid_sentence", "table_text_layer",
    "section_started_before_range", "paid_standard_name_without_numeric",
    "paid_clause_reference_without_numeric", "historical_only", "text_layer_cross_check_failed",
})

#: 포인터 은행과 본문 은행이 공통으로 가져야 하는 칸.
REQUIRED_FIELDS: tuple[str, ...] = (
    "passage_id", "doc_id", "edition", "source_file_sha256", "locator",
    "text_kind", "text_sha256", "facets", "defect_classes", "lineage_group_id",
    "scope", "rights_status", "flags",
)

#: 유료 표준 이름. 이름만 든 문장(인용 · 적용 범위 설명)은 남기되 표지를 단다.
PAID_NAME_RE = re.compile(
    r"\b(?:ISO\s*\d{3,5}|AWS\b|D1\.[1-9]\b|Bridge Welding Code|ANSI/AWS|ASME|ASTM\s*[A-Z]\d|API\s*\d{3,4}|"
    r"KS\s*[A-Z]\s*\d{3,4}|JIS\s*[A-Z]\s*\d{3,4}|EN\s*\d{3,5}|BS\s*\d{3,5}|AASHTO|CSA\s*W\d+|AS/NZS|DIN\s*\d{3,5})")
#: 수치와 단위. 유료 표준 이름과 한 문장에 함께 있으면 그 표준의 수치를 옮긴 것으로 본다(API 1104 선례).
#: "7-inch" 처럼 하이픈으로 붙은 단위, 도 · 피트, 유니코드 분수(¾)도 수치로 센다.
_FRAC = "¼½¾⅛⅜⅝⅞⅓⅔"
NUM_UNIT_RE = re.compile(
    rf"(?:\d+(?:\.\d+)?(?:\s*/\s*\d+)?|\d*[{_FRAC}])\s*-?\s*"
    r"(?:mm|cm|in\.|inch|inches|ft|feet|foot|°\s*[FC]?|deg(?:ree)?s?|ksi|psi|MPa|%|percent)(?![A-Za-z])"
    r"|\b\d+(?:\.\d+)?t\b",                     # 두께의 배수("5t") — 무차원이지만 한계값이다
    re.IGNORECASE)
#: AWS 출판물을 출처로 단 인용(그림 · 정의 전재). "(AWS, 2010)", "(AASHTO/AWS, 2015)", "(AWS, 2010c.)", "as defined by AWS".
AWS_QUOTE_RE = re.compile(
    r"\(\s*(?:AASHTO\s*/\s*)?AWS\s*,?\s*\d{4}[a-z]?\.?\s*\)|\bAWS\s+A\d|\bas defined (?:by|in) AWS\b", re.IGNORECASE)
SENT_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+")


def split_sentences(text: str) -> list[str]:
    """마침표 · 물음표 · 느낌표 뒤 공백에서 나눈다. 목록 번호만 남은 조각("3.")은 다음 문장에 붙인다."""
    merged: list[str] = []
    for p in (p.strip() for p in SENT_SPLIT_RE.split(text)):
        if not p:
            continue
        if merged and len(merged[-1]) < 8:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


#: 표준 이름 없이 조항 번호만으로 유료 표준을 가리키는 문서. FHWA-HIF-19-088 은 물리 25쪽에서 "(clause x)" 가
#: Code(AWS D1.5)의 조항 번호라고 정한다. 그래서 그 문서에서는 "the Code", "clause 6.26.1.5", 해설 "clause C-6.26",
#: 점 번호 표 · 그림("table 6.1")이 모두 D1.5 를 가리킨다(매뉴얼 자체의 표 · 그림은 정수 번호).
IMPLICIT_PAID_REF: dict[str, re.Pattern] = {
    "FHWA-HIF-19-088": re.compile(
        r"\bthe Code\b|\bclauses?\s+C?-?\d+(?:\.\d+)+|\b(?:table|figure)\s+\d+\.\d+", re.IGNORECASE),
}


def paid_reference(sentence: str, doc_id: str | None = None) -> bool:
    """유료 표준을 가리키는가 — 이름이 나오거나, 그 문서에서 조항 번호가 유료 표준을 뜻할 때."""
    if PAID_NAME_RE.search(sentence):
        return True
    implicit = IMPLICIT_PAID_REF.get(doc_id or "")
    return bool(implicit and implicit.search(sentence))


def paid_sentence_reason(sentence: str, doc_id: str | None = None) -> str | None:
    """유료 표준 유래로 보아 덜어 낼 문장의 사유. 빌드와 소비 경계가 같은 규칙을 쓴다."""
    if AWS_QUOTE_RE.search(sentence):
        return "third_party_quote_aws"
    if paid_reference(sentence, doc_id) and NUM_UNIT_RE.search(sentence):
        return "paid_standard_numeric"
    return None


def canonical_text(s: str) -> str:
    """본문 지문의 정규형 — NFC, 공백 한 칸, 앞뒤 공백 없음."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def text_sha256(s: str) -> str:
    return hashlib.sha256(canonical_text(s).encode("utf-8")).hexdigest()


def _unique_keys(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"같은 키가 두 번 있다: {k}")
        out[k] = v
    return out


def load_jsonl(path: str | Path) -> list[dict]:
    """빈 줄은 건너뛰고, 같은 키가 두 번 있는 줄은 거부한다(뒤의 값이 조용히 이기지 않게)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for k, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line, object_pairs_hook=_unique_keys))
                except ValueError as exc:
                    raise ValueError(f"{path}:{k} — {exc}") from exc
    return rows


def dump_jsonl(rows: Iterable[Mapping[str, Any]], path: str | Path) -> str:
    """정렬된 키 · UTF-8 · 행마다 LF. 파일의 sha256 을 돌려준다."""
    data = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows)
    raw = data.encode("utf-8")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def pointer_row(row: Mapping[str, Any]) -> dict:
    """본문 은행 행 → 포인터 은행 행. 본문만 비운다."""
    out = dict(row)
    out["text"] = None
    return out


def flag_names(row: Mapping[str, Any]) -> set[str]:
    """표지 이름 — `pua_unmapped:3` 같은 꼬리 값을 뗀다."""
    return {str(f).split(":", 1)[0] for f in row.get("flags") or []}


def _is_int(v: Any) -> bool:
    return type(v) is int


def check_row_shape(row: Mapping[str, Any]) -> list[str]:
    """행 꼴의 위반 목록. 빈 목록이면 통과다."""
    bad = [f"missing:{f}" for f in REQUIRED_FIELDS
           if row.get(f) is None or (row.get(f) in ("", []) and f not in ("facets", "defect_classes", "flags"))]
    if row.get("schema_version") != SCHEMA_VERSION:
        bad.append("schema_version")
    if not isinstance(row.get("passage_id"), str):
        bad.append("passage_id:type")
    loc = row.get("locator")
    if not isinstance(loc, Mapping) or not _is_int(loc.get("pdf_page_start")) or not _is_int(loc.get("pdf_page_end")):
        bad.append("locator:pdf_page")
    elif loc["pdf_page_start"] < 1 or loc["pdf_page_end"] < loc["pdf_page_start"]:
        bad.append("locator:pdf_page_order")
    if row.get("rights_status") not in RIGHTS_STATES:
        bad.append("rights_status:enum")
    for key in ("facets", "defect_classes", "flags"):
        if not isinstance(row.get(key), list):
            bad.append(f"{key}:type")
    for f in row.get("facets") or []:
        if f not in FACETS:
            bad.append(f"facet:{f}")
    for d in row.get("defect_classes") or []:
        if d not in DEFECTS:
            bad.append(f"defect:{d}")
    if "historical_only" in flag_names(row) and MODERN_ONLY_FACETS & set(row.get("facets") or []):
        bad.append("facet:historical_row_with_modern_facet")
    if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_file_sha256") or "")):
        bad.append("source_file_sha256:form")
    if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("text_sha256") or "")):
        bad.append("text_sha256:form")
    return bad


# --------------------------------------------------------------------------- 소비 경계


def _index_rights(records: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """권리 결정 기록 — (doc_id, source_file_sha256) 마다 하나. 같은 열쇠가 둘이면 거부한다."""
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for r in records:
        key = (str(r.get("doc_id")), str(r.get("source_file_sha256")))
        if key in out:
            raise ValueError(f"권리 결정 기록이 같은 문서 · 지문에 둘이다: {key}")
        out[key] = r
    return out


def _index_reviews(records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """사람 검수 기록 — passage_id 마다 하나. 같은 구절이 둘이면 거부한다."""
    out: dict[str, Mapping[str, Any]] = {}
    for r in records:
        pid = r.get("passage_id")
        if not isinstance(pid, str):
            raise ValueError(f"사람 검수 기록의 passage_id 는 문자열이어야 한다: {pid!r}")  # noqa: TRY004
        if pid in out:
            raise ValueError(f"사람 검수 기록이 같은 구절에 둘이다: {pid}")
        out[pid] = r
    return out


def load_registry(src: str | Path | Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """`corpus/parse/sources.yaml` → {doc_id: 항목}. 같은 doc_id 가 둘이면 거부한다."""
    if isinstance(src, Mapping) and "documents" not in src:
        return {str(k): v for k, v in src.items()}
    if isinstance(src, Mapping):
        data = src
    else:
        import yaml

        data = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
    out: dict[str, Mapping[str, Any]] = {}
    for e in data.get("documents") or []:
        if e["doc_id"] in out:
            raise ValueError(f"등록부에 같은 doc_id 가 둘이다: {e['doc_id']}")
        out[e["doc_id"]] = e
    return out


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        return None
    return value


def gate_reasons(
    row: Mapping[str, Any],
    *,
    pointer: Mapping[str, Any] | None,
    registry_entry: Mapping[str, Any] | None,
    rights: Mapping[str, Any] | None,
    review: Mapping[str, Any] | None,
    use: str,
) -> list[str]:
    """한 구절이 생성기로 갈 수 없는 사유. 빈 목록이면 넘긴다."""
    if use not in USES:
        raise ValueError(f"use 는 {USES} 가운데 하나다: {use!r}")
    reasons = [f"shape:{b}" for b in check_row_shape(row)]
    # 문서 결속 — 등록부에 있고, 등록부의 원문 지문과 같아야 한다.
    if registry_entry is None:
        reasons.append("registry:unregistered")
    elif registry_entry.get("sha256") != row.get("source_file_sha256"):
        reasons.append("registry:sha256_mismatch")
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        reasons.append("no_text")
    elif text_sha256(text) != row.get("text_sha256"):
        reasons.append("text_sha256_mismatch")
    if row.get("text_kind") != "verbatim_extract":
        reasons.append("text_kind_not_verbatim")
    # 본문 은행은 추적 포인터 은행의 같은 행에 묶인다 — 본문 말고는 모든 칸이 같아야 한다.
    if pointer is None:
        reasons.append("no_pointer_row")
    else:
        mine = pointer_row(row)
        for f in sorted(set(mine) | set(pointer)):
            if mine.get(f) != pointer.get(f):
                reasons.append(f"pointer_mismatch:{f}")
    flags = flag_names(row)
    # 권리 — 사용자의 결정 기록에서만 온다. 은행 행의 rights_status 로는 넘기지 않는다.
    if rights is None:
        reasons.append("rights:no_decision")
    else:
        if rights.get("rights_status") != "approved":
            reasons.append(f"rights:{rights.get('rights_status')}")
        uses = _string_list(rights.get("approved_uses"))
        if uses is None or not set(uses) <= set(USES):
            reasons.append("rights:approved_uses_malformed")
        elif use not in uses:
            reasons.append(f"rights:use_not_approved:{use}")
        if not rights.get("decided_by") or not rights.get("decided_at"):
            reasons.append("rights:no_decider")
        cleared = _string_list(rights.get("cleared_lineage_flags", []))
        if cleared is None:
            reasons.append("rights:cleared_lineage_flags_malformed")
        else:
            for f in sorted((flags & LINEAGE_FLAGS) - set(cleared)):
                reasons.append(f"rights:lineage_not_cleared:{f}")
    # 사람 검수 — 검수자 · 시각 · 대상 본문 지문 · 판정 · 확인한 위험 표지.
    if review is None:
        reasons.append("review:none")
    else:
        if review.get("decision") != "approved":
            reasons.append(f"review:{review.get('decision')}")
        if review.get("text_sha256") != row.get("text_sha256"):
            reasons.append("review:text_sha256_mismatch")
        if not review.get("reviewer") or not review.get("reviewed_at"):
            reasons.append("review:no_reviewer")
        checked = _string_list(review.get("flags_checked", []))
        if checked is None:
            reasons.append("review:flags_checked_malformed")
        else:
            for f in sorted((flags & RISK_FLAGS) - set(checked)):
                reasons.append(f"review:flag_unchecked:{f}")
    if isinstance(text, str):
        for s in split_sentences(text):
            why = paid_sentence_reason(s, row.get("doc_id"))
            if why:
                reasons.append(why)
                break
    # 보류 — 원문 정책(유료 표준 수치 규칙 등)에 걸린 구절, 그런 구절을 필수 문맥으로 삼는 구절은
    # 보류가 풀린 판의 은행이 나오기 전까지 넘기지 않는다. 기록 파일로는 풀 수 없다.
    for key, tag in (("source_policy_holds", "source_policy"), ("context_dependency_policy_holds", "context")):
        holds = _string_list(row.get(key, []))
        if holds is None:
            reasons.append(f"hold:{tag}:malformed")
        else:
            reasons.extend(f"hold:{tag}:{h}" for h in holds)
    return reasons


def select_for_generation(
    full_rows: Sequence[Mapping[str, Any]],
    *,
    pointer_rows: Sequence[Mapping[str, Any]],
    registry: str | Path | Mapping[str, Any],
    rights_records: Iterable[Mapping[str, Any]] = (),
    review_records: Iterable[Mapping[str, Any]] = (),
    use: str = "qa_generation",
) -> tuple[list[dict], Counter]:
    """생성기에 넘길 구절과, 넘기지 않은 사유별 건수.

    사유는 구절마다 첫 사유 하나가 아니라 **모두** 센다(합이 구절 수보다 클 수 있다).
    `n_rejected` 키가 거른 구절 수다. 구절 하나하나의 경계를 지난 뒤 필수 문맥(`context:*`)을 본다 — 그 사유는 구절마다 하나다. 등록부의 `license_class` · `allowed_use` 는 보지 않는다 —
    그 칸은 약관 확인(2026-10-04)보다 앞서 적힌 것이라 권리 승인이 아니다. 권리는 `rights_records` 에서만 온다.
    """
    if use not in USES:
        raise ValueError(f"use 는 {USES} 가운데 하나다: {use!r}")
    reg = load_registry(registry)
    pointers: dict[str, Mapping[str, Any]] = {}
    for p in pointer_rows:
        pid = p.get("passage_id")
        if not isinstance(pid, str) or pid in pointers:
            raise ValueError(f"포인터 은행의 passage_id 가 문자열이 아니거나 둘 이상이다: {pid!r}")
        pointers[pid] = p
    rights = _index_rights(rights_records)
    reviews = _index_reviews(review_records)
    seen: set[str] = set()
    selected: list[dict] = []
    why: Counter = Counter()
    for row in full_rows:
        pid = row.get("passage_id")
        if not isinstance(pid, str) or pid in seen:
            raise ValueError(f"본문 은행의 passage_id 가 문자열이 아니거나 둘 이상이다: {pid!r}")
        seen.add(pid)
        rs = gate_reasons(
            row,
            pointer=pointers.get(pid),
            registry_entry=reg.get(str(row.get("doc_id"))),
            rights=rights.get((str(row.get("doc_id")), str(row.get("source_file_sha256")))),
            review=reviews.get(pid),
            use=use,
        )
        if rs:
            why.update(rs)
            why["n_rejected"] += 1
        else:
            selected.append(dict(row))
    # 필수 문맥 — 필수 문맥 구절은 같은 계보 묶음이어야 하고(분할이 계보 단위다), 그 가운데 하나라도 같은 용도로
    # 경계를 못 지나면 그 구절도 넘기지 않는다. 문맥이 다시 문맥을 요구하는 사슬이 있으므로 더 빠지는 구절이 없을 때까지 되풀이한다.
    lineage = {str(r.get("passage_id")): r.get("lineage_group_id") for r in full_rows}
    while True:
        ok = {r["passage_id"] for r in selected}
        keep, drop = [], 0
        for r in selected:
            need = _string_list(r.get("required_context_passage_ids", []))
            if need is None:
                reason = "context:malformed"
            elif any(c not in lineage for c in need):
                reason = "context:required_missing"
            elif any(lineage[c] != r.get("lineage_group_id") for c in need):
                reason = "context:cross_lineage"
            elif any(c not in ok for c in need):
                reason = "context:required_not_selectable"
            else:
                keep.append(r)
                continue
            why[reason] += 1
            why["n_rejected"] += 1
            drop += 1
        selected = keep
        if not drop:
            return selected, why


#: 원문 등록부의 기본 경로
DEFAULT_REGISTRY = Path(__file__).with_name("sources.yaml")


def build_report_path(pointer_path: str | Path) -> Path:
    """포인터 은행 옆의 빌드 보고 — `passage_bank_v3.jsonl` → `passage_bank_v3.build_report.json`."""
    pointer_path = Path(pointer_path)
    return pointer_path.with_name(pointer_path.name.removesuffix(".jsonl") + ".build_report.json")


class BankRefused(ValueError):
    """은행 지문 · 빌드 보고가 맞지 않는다. 거르기를 시작하지 않는다."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_bank_for_generation(
    pointer_path: str | Path,
    full_path: str | Path | None,
    *,
    build_report: str | Path | None = None,
    registry: str | Path = DEFAULT_REGISTRY,
    rights_path: str | Path | None = None,
    review_path: str | Path | None = None,
    use: str = "qa_generation",
) -> tuple[list[dict], dict]:
    """생성기에 넘길 구절을 고르는 **유일한 진입점** — 작업 명부(`prepare`)와 사이클 실행기가 같은 것을 부른다.

    은행이 계약이 아니면 거르기 전에 `BankRefused` 로 멈춘다(조용한 0 을 내지 않는다): 빌드 보고가 없다 ·
    포인터 · 본문 은행의 sha256 이 빌드 보고(기본: 포인터 옆, `build_report_path`)와 다르다 · 포인터 행의 꼴이 틀렸거나
    본문이 비어 있지 않다 · 같은 키 · 같은 구절 · 같은 기록이 둘이다.
    그 뒤 `select_for_generation` 으로 거른다. 넘긴 구절의 필수 문맥 구절은 모두 같은 반환 목록 안에 있다.
    본문 은행이 없으면 아무것도 넘기지 않는다.

    반환: (넘길 구절 — 본문 은행 행 그대로, passage_id 순), 요약. 요약의 `selected + excluded == bank_rows` 다.
    `reasons` 는 사유별 구절 수(구절마다 모든 사유를 센다 — 합이 `excluded` 보다 클 수 있다).
    `inputs_sha256` 은 결과를 정하는 입력 전부(포인터 · 본문 · 빌드 보고 · 등록부 · 권리 결정 · 사람 검수)의 지문 — 실행 신원에 묶는다.
    """
    pointer_path = Path(pointer_path)
    build_report = build_report_path(pointer_path) if build_report is None else Path(build_report)
    if not build_report.is_file():
        raise BankRefused("build_report_missing", f"빌드 보고 {build_report.name} 가 없다")
    pinned = json.loads(build_report.read_text(encoding="utf-8")).get("outputs") or {}
    pointer_sha = _file_sha256(pointer_path)
    if pointer_sha != pinned.get("pointer_bank_sha256"):
        raise BankRefused("pointer_bank_sha_mismatch", "포인터 은행이 빌드 보고의 지문과 다르다")
    try:
        pointer_rows = load_jsonl(pointer_path)
    except ValueError as exc:
        raise BankRefused("bank_contract", str(exc)) from exc
    for r in pointer_rows:
        bad = check_row_shape(r) + (["pointer_row_has_text"] if r.get("text") is not None else [])
        if bad:
            raise BankRefused("bank_contract", f"{r.get('passage_id')!r}: {bad[:4]} — 행 꼴 {SCHEMA_VERSION}판 포인터 은행이 아니다")
    summary: dict[str, Any] = {
        "inputs_sha256": {"pointer_bank": pointer_sha, "full_bank": None,
                          "build_report": _file_sha256(build_report), "registry": _file_sha256(registry),
                          "rights_records": None if rights_path is None else _file_sha256(rights_path),
                          "review_records": None if review_path is None else _file_sha256(review_path)},
        "use": use, "bank_rows": len(pointer_rows),
    }
    if full_path is None:
        summary.update(selected=0, excluded=len(pointer_rows), reasons={"no_full_bank": len(pointer_rows)})
        return [], summary
    full_sha = _file_sha256(full_path)
    if full_sha != pinned.get("full_bank_sha256"):
        raise BankRefused("full_bank_sha_mismatch", "본문 은행이 빌드 보고의 지문과 다르다")
    summary["inputs_sha256"]["full_bank"] = full_sha
    try:
        full_rows = load_jsonl(full_path)
        selected, why = select_for_generation(
            full_rows, pointer_rows=pointer_rows, registry=registry,
            rights_records=load_jsonl(rights_path) if rights_path is not None else (),
            review_records=load_jsonl(review_path) if review_path is not None else (), use=use)
    except BankRefused:
        raise
    except ValueError as exc:
        raise BankRefused("bank_records_invalid", str(exc)) from exc
    # 포인터에는 있고 본문 은행에는 없는 구절 — 거르기가 보지 못하므로 따로 센다
    absent = len({r["passage_id"] for r in pointer_rows} - {r.get("passage_id") for r in full_rows})
    if absent:
        why["no_full_row"] += absent
        why["n_rejected"] += absent
    selected.sort(key=lambda r: r["passage_id"])
    excluded = why.pop("n_rejected", 0)
    summary.update(selected=len(selected), excluded=excluded, reasons=dict(sorted(why.items())))
    return selected, summary

"""수치 잠금 검증 (스펙 §5-4) — 생성문 재추출·정규화·화이트리스트 대조·판정어 대조.

LLM은 문장 표현만 만든다. 수치·조항·판정은 골격에 확정돼 있고, 생성문에서 하나라도
달라지면 그 샘플은 폐기된다 (재생성 금지, §5-1). 이 모듈은 판정·한계를 재구현하지
않는다 — 골격(dict, §4-5 골격 JSON)과 생성문 텍스트만 비교한다. skeleton_gen 모듈에
의존하지 않으므로 골격은 항상 매핑으로 주입받는다.

절차 (§5-4):
 1. 추출  — 수치 토큰(숫자+**단위**), 표준 식별자, 조항 번호, 등급, 판정어를 정규식으로 재추출
 2. 정규화 — NFKC(㎜→mm·전각→반각), 천 단위 콤마 제거, 후행 0 동치(Decimal 동등성),
            조사 분리(토큰 정규식이 수사 뒤 조사를 소비하지 않는 방식으로 처리)
 3. 화이트리스트 대조 — 허용 집합 = 골격의 {size 또는 measured%, thickness, L, margin,
            limit_factor, limit_cap(§4-5 "수치 잠금 대상"), defect_code, clause_id,
            quality_level}. 허용 집합 밖 수치 토큰 1개라도 있으면 폐기.
            필수 출현 집합은 §4-4 널러빌리티 매트릭스의 rule별 정의를 따른다.
            **단위도 함께 대조한다** — 값이 허용 집합 안이어도 골격 단위와 다른 단위로
            표기되면(percent 값을 mm 로) 다른 물리량이므로 폐기한다.
            **표준 식별자도 화이트리스트 대조 대상이다** — 골격이 인용한 표준·조항에서
            유도된 식별자와 기본 허용 식별자(결함 코드 체계 ISO 6520-1) 외의 표준이
            생성문에 나오면 환각 인용이므로 폐기한다. 번호 없는 발행기관 명칭
            (AWS·ASME·일본공업규격 …)도 같은 대조를 받는다 — 번호가 없다고 무근거 인용이
            아닌 것은 아니다. 한글 표기는 약어로 정규화해 표기만 바꿔 빠져나가지 못하게 한다.
 4. 판정어 대조 — 합격/불합격 ↔ 골격 verdict. verdict=None(= clause_only 게이트)이면
            판정어 출현 자체가 위반이다.

폐기 사유 코드 5+1종: missing_value | extra_value | changed_value | verdict_flip |
format_violation | defect_hallucination(정상 페어 변형, §4-6-1 ④ — 어휘 사전은 인자 주입).
단위 불일치는 값이 바뀐 것과 같으므로 `changed_value`, 환각 표준 식별자는 허용 집합 밖
토큰이므로 조항 환각과 같은 `extra_value` 로 분류한다 (사유 코드는 늘리지 않는다).

결함 어휘 락(§4-6-1 ③④)의 정본은 이 모듈의 `find_defect_tokens` 하나다 — NFKC 정규화 후
대소문자 불문으로 본다. `skeleton_gen` 은 같은 함수를 재수출하기만 한다 (구현 두 벌 금지 —
정규화 규칙이 갈리면 한쪽만 전각·NFD 표기를 놓친다).

수치는 전부 Decimal — float 입력은 거부한다 (§1-3).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Collection, Mapping, Optional, Sequence

from corpus.rules.schema import FAIL, PASS

__all__ = [
    "REASON_MISSING_VALUE",
    "REASON_EXTRA_VALUE",
    "REASON_CHANGED_VALUE",
    "REASON_VERDICT_FLIP",
    "REASON_FORMAT_VIOLATION",
    "REASON_DEFECT_HALLUCINATION",
    "REASON_CODES",
    "STANDARD_WHITELIST_DEFAULT",
    "LockResult",
    "to_decimal",
    "find_defect_tokens",
    "find_artifacts",
    "find_verdict_implying",
    "check_numeric_lock",
    "check_normal_lock",
    "check_pair_lock",
]

# ---------------------------------------------------------------------------
# 폐기 사유 코드 (§5-4 5종 + §4-6-1 ④ 1종)
# ---------------------------------------------------------------------------

REASON_MISSING_VALUE = "missing_value"
REASON_EXTRA_VALUE = "extra_value"
REASON_CHANGED_VALUE = "changed_value"
REASON_VERDICT_FLIP = "verdict_flip"
REASON_FORMAT_VIOLATION = "format_violation"
REASON_DEFECT_HALLUCINATION = "defect_hallucination"

REASON_CODES: tuple[str, ...] = (
    REASON_MISSING_VALUE,
    REASON_EXTRA_VALUE,
    REASON_CHANGED_VALUE,
    REASON_VERDICT_FLIP,
    REASON_FORMAT_VIOLATION,
    REASON_DEFECT_HALLUCINATION,
)


@dataclass(frozen=True)
class LockResult:
    """수치 잠금 판정 결과. ok=False 면 해당 샘플은 폐기 대상이다 (재생성 금지)."""

    ok: bool
    reasons: tuple[str, ...] = ()
    detail: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 정규화·추출 (§5-4 1·2단계)
# ---------------------------------------------------------------------------

# 천 단위 콤마 (1,000) 만 제거. "0,2" 류 소수점 콤마는 여기서 해석하지 않는다 —
# 잔여 콤마 인접 숫자는 각각 별개 토큰으로 추출되어 화이트리스트 대조에서 걸린다.
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")

# 표준 문서 식별자 (수치가 아니라 명칭이다) — 수치 스캔 전에 떼어내되, 뗀 것 자체를
# 화이트리스트로 대조한다. 마스킹만 하면 환각 표준이 무조건 통과한다 (finding #6b).
_STD_REF_RE = re.compile(
    r"\b(?:ISO|EN|IEC)\s*\d[\d\-:.]*"
    r"|\bKS\s*[A-Z]?\s+\d[\d\-:.]*"
    r"|\bIACS\s*Rec\.?\s*\d+"
    r"|\bRec\.\s*\d+"
)

# 번호 없는 표준 명칭. 식별자 정규식은 숫자를 요구하므로 "AWS 및 ASME 요건도 충족한다"
# 류가 그대로 통과했다 (적대 검증 잔여 #6b). 발행기관 명칭 출현 자체를 대조 대상으로 본다.
# 한글 표기는 약어로 정규화해 대조한다 — 골격이 KS 를 지시하면 "한국산업표준"도 통과해야
# 하고, 지시하지 않으면 어느 표기든 막혀야 하기 때문이다.
_STD_BODY_CANON: dict[str, str] = {
    "AWS": "AWS", "ASME": "ASME", "ASTM": "ASTM", "ANSI": "ANSI",
    "JIS": "JIS", "DIN": "DIN", "BS": "BS", "GB": "GB", "NF": "NF",
    "ISO": "ISO", "EN": "EN", "IEC": "IEC", "KS": "KS", "IACS": "IACS",
    "미국용접학회": "AWS", "미국기계학회": "ASME", "미국재료시험협회": "ASTM",
    "일본공업규격": "JIS", "독일공업규격": "DIN", "영국표준": "BS",
    "중국국가표준": "GB", "한국산업표준": "KS", "국제표준화기구": "ISO",
    "유럽표준": "EN", "국제전기기술위원회": "IEC",
}
_STD_BODY_RE = re.compile(
    "|".join(
        (rf"\b{re.escape(k)}\b" if k.isascii() else re.escape(k))
        for k in sorted(_STD_BODY_CANON, key=len, reverse=True)
    )
)

# 골격이 아무 표준도 지시하지 않아도 허용되는 식별자: 결함 코드 체계 (사상표의 근거,
# CITED_FACT — 코드 번호·명칭은 사실 정보다). (c) 템플릿의 "분류" 절이 이것을 쓴다.
STANDARD_WHITELIST_DEFAULT: tuple[str, ...] = ("ISO 6520-1",)

# 조항 번호 토큰: 문자 접두 + '-영숫자(.영숫자)*' 1회 이상.
# 예: KR-3.2.1, IACS47-3.2.1, KRA27-T15, KRA27-3D — 실제 조항 번호에는 표 기호(T15)나
# 항 기호(3D)처럼 문자가 섞인다. 숫자만 허용하면 KRA27-T15 를 조항으로 인식하지 못하고,
# 그 안의 27·15 가 환각 수치로 잡혀 정상 생성문이 전량 폐기된다 (소표본 관통 시험 실측).
_CLAUSE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)+")


def _is_clause_token(tok: str) -> bool:
    """하이픈 뒤에 숫자가 하나라도 있어야 조항 번호로 본다.

    'cross-section' 같은 영문 합성어를 조항으로 오인하면 '허용 집합 밖 조항'이 되어
    정상 문장이 폐기된다. 반대로 숫자를 품은 토큰을 놓치면 그 숫자가 환각으로 잡힌다.
    """
    _, _, rest = tok.partition("-")
    return any(ch.isdigit() for ch in rest)

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

UNIT_MM = "mm"
UNIT_PERCENT = "percent"

# 단위 별칭. §1-2 17 의 unit enum 은 mm·percent 2종이지만 **생성문 표기**는 그렇지 않다 —
# 사전이 {mm, %, percent, 퍼센트} 뿐이면 "20.00 밀리미터"가 '단위 없음'으로 읽혀 대조를
# 통째로 건너뛴다 (적대 검증 잔여 #6a). 표기 변형을 사전에 넣어 같은 혼입이 표기만 바꿔
# 빠져나가지 못하게 한다.
_UNIT_ALIAS = {
    "mm": UNIT_MM, "밀리미터": UNIT_MM, "밀리미타": UNIT_MM, "밀리": UNIT_MM, "미리": UNIT_MM,
    "%": UNIT_PERCENT, "percent": UNIT_PERCENT, "퍼센트": UNIT_PERCENT,
    "프로": UNIT_PERCENT, "백분율": UNIT_PERCENT,
}

# 수치 토큰 = 숫자 + (있으면) 단위. 괄호 병기 "20.00(mm)" 도 단위 표기다.
_UNIT_ALT = "|".join(
    (rf"{re.escape(k)}(?![A-Za-z])" if k.isascii() and k.isalpha() else re.escape(k))
    for k in sorted(_UNIT_ALIAS, key=len, reverse=True)
)
_NUM_UNIT_RE = re.compile(
    rf"(?P<num>\d+(?:\.\d+)?)\s*[(\[]?\s*(?P<unit>{_UNIT_ALT})?",
    re.IGNORECASE,
)

# 판정어. **분리자를 지운 뒤** 센다 — "불 합격"처럼 공백·중점이 하나만 끼면 한 글자
# lookbehind 가 뚫려 실제 뒤집힘이 뒤집힘으로 잡히지 않았다 (80번 B21).
_VERDICT_SEP_RE = re.compile(r"[\s·ㆍ・\-_~]+")
_FAIL_RE = re.compile(r"불합격")
_PASS_RE = re.compile(r"(?<!불)합격")


def to_decimal(v: object) -> Decimal:
    """Decimal 변환. float·bool·NaN·inf 거부 (§1-3 float 직접 비교 금지)."""
    if isinstance(v, bool):
        raise TypeError("bool 은 수치가 아니다")
    if isinstance(v, float):
        raise TypeError("float 금지 — Decimal(str) 로 만들어 넘겨라 (§1-3)")
    if isinstance(v, Decimal):
        d = v
    elif isinstance(v, int):
        d = Decimal(v)
    elif isinstance(v, str):
        d = Decimal(v.strip())
    else:
        raise TypeError(f"수치 변환 불가 타입: {type(v).__name__}")
    if not d.is_finite():
        raise ValueError(f"NaN/inf 금지: {d}")
    return d


def _normalize(text: str) -> str:
    """NFKC(㎜→mm, 전각→반각) + 천 단위 콤마 제거."""
    t = unicodedata.normalize("NFKC", text)
    return _THOUSANDS_RE.sub("", t)


@dataclass(frozen=True)
class _Scan:
    """생성문 재추출 결과. 수치 토큰은 값과 단위를 함께 들고 다닌다."""

    clauses: tuple[str, ...]
    standards: tuple[str, ...]
    tokens: tuple[tuple[Decimal, Optional[str]], ...]
    n_pass: int
    n_fail: int

    @property
    def values(self) -> tuple[Decimal, ...]:
        return tuple(v for v, _ in self.tokens)


def _scan(text: str) -> _Scan:
    """조항·표준 식별자·수치 토큰(값+단위)·판정어 출현 수를 재추출한다.

    판정어는 마스킹 전 전체 텍스트에서 센다. 수치는 표준 식별자·조항 번호를
    떼어낸 잔여 텍스트에서만 추출한다 (조사 분리는 정규식이 수사 뒤 문자를
    소비하지 않으므로 자동 처리된다).

    번호 없는 발행기관 명칭은 **조항 번호를 떼어낸 뒤에** 훑는다 — 그 전에 훑으면
    "KR-3.2.1" 의 접두 KR 이 발행기관으로 잡혀 정상 인용이 환각으로 몰린다.
    """
    t = _normalize(text)
    verdict_view = _VERDICT_SEP_RE.sub("", t)
    n_fail = len(_FAIL_RE.findall(verdict_view))
    n_pass = len(_PASS_RE.findall(verdict_view))
    standards = list(_STD_REF_RE.findall(t))
    t = _STD_REF_RE.sub(" ", t)
    clauses = tuple(c for c in _CLAUSE_RE.findall(t) if _is_clause_token(c))
    # 조항으로 인정한 토큰만 가린다. 인정하지 않은 토큰(예: 영문 합성어)을 가리면
    # 그 안의 숫자가 수치 검사에서 빠져 환각 수치를 놓친다.
    t = _CLAUSE_RE.sub(lambda m: " " if _is_clause_token(m.group(0)) else m.group(0), t)
    standards += _STD_BODY_RE.findall(t)
    t = _STD_BODY_RE.sub(" ", t)
    tokens: list[tuple[Decimal, Optional[str]]] = []
    for m in _NUM_UNIT_RE.finditer(t):
        raw_unit = m.group("unit")
        unit = _UNIT_ALIAS.get(raw_unit.lower()) if raw_unit else None
        try:
            tokens.append((Decimal(m.group("num")), unit))
        except InvalidOperation:  # pragma: no cover - \d 정규식상 도달 불능
            continue
    return _Scan(clauses, tuple(standards), tuple(tokens), n_pass, n_fail)


# ---------------------------------------------------------------------------
# 생성문 아티팩트 게이트 (80번 G3-1) — 내부 표현이 문장으로 샌 흔적
# ---------------------------------------------------------------------------
#
# 채택 corpus 69건 중 49건(71.0%)이 프롬프트 직렬화 버그의 산물이었고 전 단계를 통과했다
# (B1): `Unit.MM` 26건, 개구간 인코딩 `X.01` 43건, 후행 0 `4.00` 27건. 이 셋은 값으로는
# 골격과 같아서 화이트리스트 대조를 통과한다 — **표기가 틀린 것이라 값 검사로는 안 잡힌다.**
# 그래서 표기 자체를 사유로 만든다. 정본은 여기 하나다 (호출자가 자기 정규식을 두지 않는다).
#
# 수사(십이·十二·twelve)도 같은 자리에서 본다 (G2-7) — 값 파싱 없이 우회하는 경로다.

_ARTIFACT_SPECS: tuple[tuple[str, "re.Pattern[str]"], ...] = (
    ("내부 enum 노출",
     re.compile(r"\b(?:Unit|InspectionMethod|LimitRule|LimitOp|LimitType|Material"
                r"|QualityScheme|Scope|RatioBasis|VerdictMode|Verdict|Judgment)"
                r"\.[A-Za-z_]+")),
    ("개구간 인코딩 노출",
     re.compile(r"(?<![\d.])(?P<n>\d+)\.01(?![\d])")),
    ("후행 0 표기",
     re.compile(r"(?<![\d.])\d+\.\d*0(?![\d])")),
    ("파이썬 리터럴 노출",
     re.compile(r"\b(?:None|True|False|Decimal\(|\[\s*'|\{\s*')")),
)

#: 수사 어휘 (G2-7). 값 파싱을 거치지 않으므로 화이트리스트가 못 잡는다.
_NUMERAL_WORDS: tuple[str, ...] = (
    "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉", "열",
    "일점", "이점", "삼점", "사점", "오점",
    "십일", "십이", "십오", "이십", "이십오", "삼십", "오십", "백",
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百",
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "fifteen", "twenty", "thirty", "fifty", "hundred",
)
#: 수량을 세는 단위. 한글·한자 수사는 **단위와 붙어 있을 때만** 수사로 본다 —
#: 낱말 경계가 없는 언어라 맨 부분문자열로 잡으면 "균열"의 '열', "배열"의 '열'이
#: 걸려 정상 문장이 전량 폐기된다 (실측: D4 페어 89건).
_COUNTER = r"(?:개|건|장|번|회|차|배|겹|군데|곳|mm|밀리미터|퍼센트|%|도|초|분|시간)"
_KO_NUMERALS = tuple(w for w in _NUMERAL_WORDS if not w.isascii())
_EN_NUMERALS = tuple(w for w in _NUMERAL_WORDS if w.isascii())
_NUMERAL_RE = re.compile(
    "|".join(
        [rf"(?:{'|'.join(re.escape(w) for w in sorted(_KO_NUMERALS, key=len, reverse=True))})"
         rf"\s*{_COUNTER}"]
        + [rf"\b{re.escape(w)}\b" for w in sorted(_EN_NUMERALS, key=len, reverse=True)]
    ),
    re.IGNORECASE,
)

#: 판정 함의 표현 (G2-6). `clause_only` 게이트에서는 합격·불합격 낱말을 피해 같은 뜻을
#: 전달하는 표현도 폐기 사유다 — 축이 "합부를 말하지 않는다" 이기 때문이다.
_VERDICT_IMPLYING: tuple[str, ...] = (
    "적합하다", "부적합", "적합함", "합치한다", "만족한다", "만족하지",
    "충족한다", "충족하지", "허용된다", "허용되지", "허용한다",
    "통과한다", "통과하지", "기준을 넘는다", "기준을 초과한다", "기준 이내",
    "판정한다", "판정된다", "판정할 수 있다",
)
_VERDICT_IMPLYING_RE = re.compile("|".join(re.escape(w) for w in _VERDICT_IMPLYING))


def find_artifacts(text: str, *, basis: str = "") -> tuple[str, ...]:
    """생성문에 남은 내부 표현·수사 흔적. 빈 튜플이면 깨끗하다.

    **판정 기준은 [자료]다.** `basis` 에 그 토큰이 문자 그대로 있으면 아티팩트가 아니다 —
    자료가 그렇게 줬으면 그대로 옮겨 적은 것이 맞다. 자료에 없는 표기가 생성문에만 있으면
    그것은 내부 표현이 다른 경로로 샌 흔적이다.

    이 구분이 필요한 이유: 값 검사는 `4.00 ≡ 4` 로 보므로(§5-4 2단계 후행 0 동치) 표기
    오염을 원리적으로 못 잡는다. 반대로 자료를 안 보고 표기만 막으면, 자료가 정당하게
    `4.00` 을 준 경로에서 정상 출력이 전량 폐기된다.
    """
    t = _normalize(text)
    b = _normalize(basis)
    hits: list[str] = []
    for label, pat in _ARTIFACT_SPECS:
        for m in pat.finditer(t):
            if m.group(0) in b:
                continue
            hits.append(f"{label}: {m.group(0)!r}")
            break
    for m in _NUMERAL_RE.finditer(t):
        if m.group(0) in b:
            continue
        hits.append(f"수사 표기: {m.group(0)!r}")
        break
    return tuple(hits)


def find_verdict_implying(text: str) -> tuple[str, ...]:
    """판정 함의 표현 (clause_only 전용 검사)."""
    return tuple(sorted({m.group(0) for m in _VERDICT_IMPLYING_RE.finditer(_normalize(text))}))


# ---------------------------------------------------------------------------
# 골격 기반 허용·필수 집합 (§5-4 3단계 + §4-4 매트릭스)
# ---------------------------------------------------------------------------

def _skeleton_unit(sk: Mapping[str, object], key: str) -> str:
    """한계·측정값이 놓이는 단위. `unit`·`measured_unit` 이 정본이고, 없으면
    limit_type 에서 유도한다 (§1-2 17: 길이·직경 ↔ mm, 비율 ↔ percent)."""
    raw = str(sk.get(key) or "").strip().lower()
    if raw in _UNIT_ALIAS:
        return _UNIT_ALIAS[raw]
    if key == "measured_unit":
        return _skeleton_unit(sk, "unit")
    return UNIT_PERCENT if str(sk.get("limit_type") or "") == "비율" else UNIT_MM


def _skeleton_allowed(sk: Mapping[str, object]) -> dict[Decimal, set[str]]:
    """허용 수치 → 그 수치에 붙을 수 있는 단위 집합 (§5-4 3단계).

    빈 집합은 "단위를 붙일 수 없는 값"(무차원 계수·결함 코드·등급 수치)이다. 단위가
    없는 표기는 어느 값에서도 허용한다 — 대조는 "생성문에 적힌 단위가 골격 단위와
    다른가" 만 본다.
    """
    lim_u = _skeleton_unit(sk, "unit")
    spec: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("size_mm", (UNIT_MM,)),
        ("measured_value", (_skeleton_unit(sk, "measured_unit"),)),
        ("thickness_mm", (UNIT_MM,)),
        ("limit_value", (lim_u,)),
        ("margin", (lim_u,)),
        ("limit_factor", ()),  # 0.1t 의 0.1 — 무차원
        ("limit_cap", (lim_u,)),
    )
    vals: dict[Decimal, set[str]] = {}
    for key, units in spec:
        v = sk.get(key)
        if v is None:
            continue
        d = to_decimal(v)
        vals.setdefault(d, set()).update(units)
        if key == "margin":
            vals.setdefault(abs(d), set()).update(units)  # "여유 0.01 mm" 류 절대값 표기
    code = sk.get("defect_code")
    if code is not None:
        try:
            vals.setdefault(Decimal(str(code)), set())
        except InvalidOperation:
            pass  # 비수치 코드 체계면 수치 화이트리스트 대상 아님
    level = sk.get("quality_level")
    if level:
        for m in _NUM_RE.findall(_normalize(str(level))):
            vals.setdefault(Decimal(m), set())
    return vals


def _merge_allowed(dst: dict[Decimal, set[str]], src: Mapping[Decimal, set[str]]) -> None:
    for value, units in src.items():
        dst.setdefault(value, set()).update(units)


def _norm_standard(ref: str) -> str:
    """표준 식별자 비교형: 공백·마침표 제거 후 대문자 (ISO 6520-1 ↔ ISO6520-1).

    번호 없는 발행기관 명칭은 약어로 정규화한다 (일본공업규격 ↔ JIS) — 표기를 바꿔
    같은 인용을 빠져나가게 두지 않으면서, 골격이 지시한 기관은 한글 표기로도 통과시킨다.
    """
    s = re.sub(r"[\s.]+", "", _normalize(ref)).upper()
    return _STD_BODY_CANON.get(s, s)


def _skeleton_standards(
    sk: Mapping[str, object], allowed_standards: Collection[str]
) -> set[str]:
    """생성문에 나와도 되는 표준 식별자 집합 (비교형).

    주입 허용 목록 + 골격이 지시하는 출처(`source_doc`·`restated_from`) + 인용 조항의
    문서 접두(KR-3.2.1 → KR). 그 밖의 식별자는 골격에 근거가 없는 인용이다.
    """
    out = {_norm_standard(str(s)) for s in allowed_standards if str(s).strip()}
    for key in ("source_doc", "restated_from"):
        v = sk.get(key)
        if v:
            out.add(_norm_standard(str(v)))
    clause = str(sk.get("clause_id") or "").strip()
    if clause:
        out.add(_norm_standard(clause.split("-", 1)[0]))
    return out


def _unit_reasons(
    scan: _Scan, allowed: Mapping[Decimal, set[str]], detail: list[str]
) -> list[str]:
    """단위 대조 (§5-4 3단계). 값은 맞는데 단위가 다르면 다른 물리량이다 — 값이 바뀐
    것으로 본다. 허용 집합 밖 값은 여기서 다루지 않는다 (화이트리스트 대조 소관)."""
    bad: list[str] = []
    for value, unit in scan.tokens:
        if unit is None:
            continue
        units = allowed.get(value)
        if units is None or unit in units:
            continue
        bad.append(f"{value} {unit} (골격 단위: {sorted(units) or '무단위'})")
    if bad:
        detail.append(f"수치 단위 불일치: {bad}")
        return [REASON_CHANGED_VALUE]
    return []


def _standard_reasons(scan: _Scan, allowed: Collection[str], detail: list[str]) -> list[str]:
    """표준 식별자 화이트리스트 대조 (§5-4 3단계, finding #6b)."""
    known = set(allowed)
    extras = sorted({r.strip() for r in scan.standards if _norm_standard(r) not in known})
    return _classify_sets(extras, [], detail, "표준 식별자")


def _skeleton_required(sk: Mapping[str, object]) -> list[tuple[str, Decimal]]:
    """§4-4 널러빌리티 매트릭스의 rule별 필수 출현 수치 집합.

    const·prop_t·prop_t_cap(길이·직경): size, t, L / const(비율): measured%, t, L /
    none_permitted: size (+ defect_code·verdict 는 별도 검사).
    """
    rule = str(sk.get("limit_rule") or "")
    limit_type = str(sk.get("limit_type") or "")

    def need(key: str) -> tuple[str, Decimal]:
        v = sk.get(key)
        if v is None:
            raise ValueError(f"골격 필수 키 결손: {key} (limit_rule={rule})")
        return (key, to_decimal(v))

    if rule == "none_permitted":
        return [need("size_mm")]
    if limit_type == "비율":
        return [need("measured_value"), need("thickness_mm"), need("limit_value")]
    return [need("size_mm"), need("thickness_mm"), need("limit_value")]


def _verdict_reasons(
    n_pass: int, n_fail: int, expected: Optional[str], detail: list[str]
) -> list[str]:
    """판정어 대조 (§5-4 4단계). expected=None 은 clause_only 게이트 — 판정어 금지."""
    if n_pass and n_fail:
        detail.append("판정어 혼재: 합격·불합격 동시 출현")
        return [REASON_FORMAT_VIOLATION]
    text_verdict = FAIL if n_fail else (PASS if n_pass else None)
    if expected is None:
        if text_verdict is not None:
            detail.append(f"clause_only 인데 판정어 출현: {text_verdict}")
            return [REASON_VERDICT_FLIP]
        return []
    if text_verdict is None:
        detail.append(f"판정어 결손 (골격 verdict={expected})")
        return [REASON_MISSING_VALUE]
    if text_verdict != expected:
        detail.append(f"판정어 불일치: 생성문={text_verdict} ↔ 골격={expected}")
        return [REASON_VERDICT_FLIP]
    return []




def _classify_sets(
    extras: Sequence[str], missing: Sequence[str], detail: list[str], what: str
) -> list[str]:
    """추가분·결손분 조합 → 사유 코드. 둘 다 있으면 값이 바뀐 것(changed)이다."""
    if extras:
        detail.append(f"{what} 허용 집합 밖: {list(extras)}")
    if missing:
        detail.append(f"{what} 필수 출현 결손: {list(missing)}")
    if extras and missing:
        return [REASON_CHANGED_VALUE]
    if extras:
        return [REASON_EXTRA_VALUE]
    if missing:
        return [REASON_MISSING_VALUE]
    return []


def _dedup(reasons: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for r in reasons:
        if r not in out:
            out.append(r)
    return tuple(out)


# ---------------------------------------------------------------------------
# 공개 검사기 3종
# ---------------------------------------------------------------------------


def check_numeric_lock(
    text: str,
    skeleton: Mapping[str, object],
    *,
    allowed_standards: Collection[str] = STANDARD_WHITELIST_DEFAULT,
) -> LockResult:
    """(c)·D4 결함 골격 1건 ↔ 생성문 대조. 불일치 → ok=False (폐기).

    allowed_standards: 골격 출처와 무관하게 인용을 허용할 표준 식별자 (기본값은 결함
    코드 체계 하나). 여기에 없고 골격에서도 유도되지 않는 표준은 환각 인용으로 폐기한다.
    """
    if not text or not text.strip():
        return LockResult(False, (REASON_FORMAT_VIOLATION,), ("빈 생성문",))
    try:
        allowed = _skeleton_allowed(skeleton)
        required = _skeleton_required(skeleton)
        want_standards = _skeleton_standards(skeleton, allowed_standards)
    except (ValueError, TypeError, InvalidOperation, KeyError) as e:
        return LockResult(False, (REASON_FORMAT_VIOLATION,), (f"골격 결손·형식 오류: {e}",))

    scan = _scan(text)
    reasons: list[str] = []
    detail: list[str] = []

    # 조항 번호 대조
    want_clause = str(skeleton.get("clause_id") or "")
    clause_set = set(scan.clauses)
    extra_clauses = sorted(clause_set - {want_clause})
    missing_clauses = [] if want_clause in clause_set else [want_clause]
    reasons += _classify_sets(extra_clauses, missing_clauses, detail, "조항")

    # 표준 식별자 대조 (골격에 근거 없는 표준 인용 차단)
    reasons += _standard_reasons(scan, want_standards, detail)

    # 수치 화이트리스트 대조 (값 + 단위)
    num_set = set(scan.values)
    extra_nums = sorted({str(n) for n in scan.values if n not in allowed})
    missing_nums = [f"{k}={v}" for k, v in required if v not in num_set]
    reasons += _classify_sets(extra_nums, missing_nums, detail, "수치")
    reasons += _unit_reasons(scan, allowed, detail)

    # F4: defect_code 출현 필수 (§4-4 매트릭스)
    if str(skeleton.get("limit_rule") or "") == "none_permitted":
        code = str(skeleton.get("defect_code") or "")
        try:
            code_present = Decimal(code) in num_set
        except InvalidOperation:
            code_present = code in _normalize(text)
        if not code_present:
            detail.append(f"defect_code 출현 결손: {code}")
            reasons.append(REASON_MISSING_VALUE)

    # 판정어 대조 (골격 verdict=None = clause_only 게이트)
    expected = skeleton.get("verdict")
    reasons += _verdict_reasons(
        scan.n_pass, scan.n_fail, None if expected is None else str(expected), detail
    )

    reasons_t = _dedup(reasons)
    return LockResult(not reasons_t, reasons_t, tuple(detail))


def find_defect_tokens(text: str, defect_lexicon: Collection[str]) -> tuple[str, ...]:
    """결함 어휘 검출 — 결함 어휘 락의 정본 (§4-6-1 ③④).

    문맥 불문(부정 문맥도 폐기 대상), **대소문자 불문**. ISO 코드처럼 숫자로만 된 토큰은
    인접 숫자 경계로, 명칭 토큰은 부분 일치로 본다 — 어느 쪽도 놓치는 것보다 넓게 잡는
    것이 안전하다. 정상 corpus 오염이 표본 몇 건 폐기보다 비싸기 때문이다.
    """
    t = _normalize(text)
    low = t.casefold()
    hits: list[str] = []
    for term in sorted({str(x) for x in defect_lexicon if str(x)}):
        if term.isdigit():
            if re.search(rf"(?<![0-9]){re.escape(term)}(?![0-9])", t):
                hits.append(term)
        elif term.casefold() in low:
            hits.append(term)
    return tuple(hits)


def check_normal_lock(
    text: str,
    *,
    defect_lexicon: Collection[str],
    thickness_mm: object = None,
    quality_level: str = "",
    expected_verdict: Optional[str] = None,
    allowed_standards: Collection[str] = (),
) -> LockResult:
    """정상 이미지 페어 변형 (§4-6-1 ④).

    허용 수치 집합 = {두께(가정 시 "가정 두께 N mm" 문구), 품질수준 내 수치}. 그 외 수치
    토큰 1개라도 출현 시 폐기. 결함 어휘 락: 주입된 사전(사상표의 결함 명칭·ISO 코드)의
    토큰이 문맥 불문·대소문자 불문 출현하면 defect_hallucination 으로 폐기 — 부정
    문맥("기공은 관찰되지 않았다")도 금지다 (§4-6-1 ③). 검출은 `find_defect_tokens` 정본
    하나만 쓴다.

    표준 식별자는 기본적으로 하나도 허용하지 않는다 — 정상 페어는 인용할 허용치가 없어
    조항 인용 자체가 금지이므로 표준 인용도 같은 취급이다 (필요 시 인자로 주입).

    expected_verdict: full 모드 정상 페어는 "합격"(aggregate_verdicts([])), clause_only
    게이트면 None — None 이면 판정어 출현 자체가 위반이다.
    """
    if not text or not text.strip():
        return LockResult(False, (REASON_FORMAT_VIOLATION,), ("빈 생성문",))
    reasons: list[str] = []
    detail: list[str] = []

    hits = find_defect_tokens(text, defect_lexicon)
    if hits:
        detail.append(f"결함 어휘 출현 (문맥 불문 폐기): {list(hits)}")
        reasons.append(REASON_DEFECT_HALLUCINATION)

    scan = _scan(text)
    if scan.clauses:
        detail.append(f"정상 페어에 조항 인용: {sorted(set(scan.clauses))} (인용할 허용치가 없다)")
        reasons.append(REASON_EXTRA_VALUE)

    reasons += _standard_reasons(
        scan, {_norm_standard(str(s)) for s in allowed_standards if str(s).strip()}, detail
    )

    allowed: dict[Decimal, set[str]] = {}
    if thickness_mm is not None:
        allowed.setdefault(to_decimal(thickness_mm), set()).add(UNIT_MM)
    if quality_level:
        for m in _NUM_RE.findall(_normalize(str(quality_level))):
            allowed.setdefault(Decimal(m), set())
    extras = sorted({str(n) for n in scan.values if n not in allowed})
    if extras:
        detail.append(f"허용 밖 수치 토큰: {extras}")
        reasons.append(REASON_EXTRA_VALUE)
    reasons += _unit_reasons(scan, allowed, detail)

    reasons += _verdict_reasons(scan.n_pass, scan.n_fail, expected_verdict, detail)
    reasons_t = _dedup(reasons)
    return LockResult(not reasons_t, reasons_t, tuple(detail))


def check_pair_lock(
    text: str,
    defects: Sequence[Mapping[str, object]],
    *,
    image_verdict: Optional[str],
    allowed_standards: Collection[str] = STANDARD_WHITELIST_DEFAULT,
) -> LockResult:
    """D4 다결함 페어의 이미지 수준 판정문 ↔ 결함 골격 배열 대조 (§4-6).

    허용·필수 수치 = 결함별 집합의 합집합, 조항 = 전 결함 clause_id 전부 출현 필수.
    판정어는 이미지 수준 verdict(aggregate_verdicts 결과) 하나와 대조한다 —
    결함별 판정어를 나열하는 템플릿은 쓰지 않는다 (혼재 시 format_violation).
    """
    if not text or not text.strip():
        return LockResult(False, (REASON_FORMAT_VIOLATION,), ("빈 생성문",))
    if not defects:
        raise ValueError("defects 비어 있음 — 정상 페어는 check_normal_lock 을 써라")
    try:
        allowed: dict[Decimal, set[str]] = {}
        required: list[tuple[str, Decimal]] = []
        want_clauses: set[str] = set()
        want_standards: set[str] = set()
        f4_codes: list[str] = []
        for i, sk in enumerate(defects):
            _merge_allowed(allowed, _skeleton_allowed(sk))
            required += [(f"d{i}:{k}", v) for k, v in _skeleton_required(sk)]
            want_clauses.add(str(sk.get("clause_id") or ""))
            want_standards |= _skeleton_standards(sk, allowed_standards)
            if str(sk.get("limit_rule") or "") == "none_permitted":
                f4_codes.append(str(sk.get("defect_code") or ""))
    except (ValueError, TypeError, InvalidOperation, KeyError) as e:
        return LockResult(False, (REASON_FORMAT_VIOLATION,), (f"골격 결손·형식 오류: {e}",))

    scan = _scan(text)
    reasons: list[str] = []
    detail: list[str] = []

    clause_set = set(scan.clauses)
    extra_clauses = sorted(clause_set - want_clauses)
    missing_clauses = sorted(want_clauses - clause_set)
    reasons += _classify_sets(extra_clauses, missing_clauses, detail, "조항")

    reasons += _standard_reasons(scan, want_standards, detail)

    num_set = set(scan.values)
    extra_nums = sorted({str(n) for n in scan.values if n not in allowed})
    missing_nums = [f"{k}={v}" for k, v in required if v not in num_set]
    reasons += _classify_sets(extra_nums, missing_nums, detail, "수치")
    reasons += _unit_reasons(scan, allowed, detail)

    for code in f4_codes:
        try:
            present = Decimal(code) in num_set
        except InvalidOperation:
            present = code in _normalize(text)
        if not present:
            detail.append(f"defect_code 출현 결손: {code}")
            reasons.append(REASON_MISSING_VALUE)

    reasons += _verdict_reasons(
        scan.n_pass, scan.n_fail, None if image_verdict is None else str(image_verdict), detail
    )
    reasons_t = _dedup(reasons)
    return LockResult(not reasons_t, reasons_t, tuple(detail))

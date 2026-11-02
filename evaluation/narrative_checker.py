"""금지 주장 검사기 v3.1 — 서술 지표의 판정부(03 2판 §2 · §3-4a · §6a, 게이트 판정 04 §7-3 ~ §7-8, 닫힌 목록 v3-lists-3).

v3.1 은 v3 의 세 곳을 바꿨다(04 §7-8 · 확인 사례 199 결정 문서 §4). 1 따로 선 '강' 의 뒤 토씨에서 '도' 를 뺐다('강도' 는 '강 + 도' 가
아니다 — 196 의 H3-N08). 그 조건은 닫힌 목록의 구조 칸 `H3.lone_gang` 에서 읽는다. 2 부재 꼴을 넓혔다 — 대상 낱말 + 이음(에 대한 ·
에 관한)? + 이름씨? + 토씨? + 꾸밈말? + 맨 '없' 꼴(v2 가 받던 '두께에 대한 정보가 없어 …' 를 되찾는다). 3 판정 명사 목록에서 '합격' 과
'불합격' 을 가운뎃점 U+318D 로 이은 한 토큰(정규화한 글과 맞지 않았다)을 둘로 나눴다. 세 곳 모두 목록을 읽는 자리다.

시안 03b(v2, `docs/dev_log/2026-10-20-설계복구/03b_금지주장_검사기_시안_D.py`)의 규칙을 옮기고 다음을 더한다.

- **세 구멍**(03 2판 §3-4a). H1 판정 낱말 없는 허용 · 안전 단정 → (나) 세부 '무판정어 허용 단정'. H2 이름 · 표 번호로 가리킨
  조항 → 별칭 표(허용치 표의 출처 칸)로 이어 (라) · (마), 잇지 못하면 '미해석 조항 언급'. H3 AL 에서 다른 재질 기준 → (마)
  세부 '타 재질 기준 적용'.
- **(라) 세부 셋**(§6a 결정 1). 표에 없는 조항 · 정답 밖 · 제 주장과 맞음 · 정답 밖 · 제 주장과도 어긋남. 행 식별자와 별칭도
  제 주장 조항과 맞대어 가른다. 조건부 소개 · 인용은 분자에서 빼지 않고 꼴(mode)만 단다.
- **유보 인정 v3**(§6a 결정 3 + 04 §7-7 정정). 조건 1 ~ 3(정보 부재 · 유보 대상 · 부정 범위)은 문장 안, 조건 4(뒤 단정과
  모순 없음)는 서술 단위. r_uncov 는 조항 부재 그리고 특정 유보(같은 문장). 이중 부정 · 가능 긍정은 인정하지 않는다.
- **부재의 좁힘**(04 §7-5 ②)과 v3.1 의 넓힘(199 결정 문서 §4 의 2). 맨 '없' 꼴은 부재 대상 낱말 뒤일 때만 부재다 — 대상 낱말 + 이음?
  + 이름씨? + 토씨? + 꾸밈말? + 맨 '없' 꼴. '두께는 문제없다' 는 앎의 부정이 아니고, '두께에 대한 정보가 없어' 는 앎의 부정이다.
- **값 없는 판정 낱말**(04 §7-6 ②). (나)는 판정어 갈래 '판정한다' 를 세지 않는다. 앎의 부정 판정은 v2 의 목록 전체를 쓴다.

닫힌 낱말 목록은 정본 사본 `LISTS_REL`(바이트 = 03h-3, sha256 고정)에서 읽는다 — 코드에 다시 적지 않는다. 목록에 없는 v2 의 정규식
(`03h-3` 의 `v2_unchanged`)은 시안의 글자 그대로 옮겼다(`tests/test_narrative_checker.py` 가 시안과 패턴을 맞댄다).

입력은 정규화 전의 서술 글이다. 결과(`Result`)는 종류별 근거 · 필요 사유 · 인정 사유 · 세부 표지 · (라) 세부 · 미해석 언급을 든다.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RULE_VERSION = "v3.1"
CHECKER_ID = "narrative_checker_v3_1"
LISTS_REL = "evaluation/vectors/narrative_lists.v3-3.json"
LISTS_SHA256 = "c0a4be1055b84091a823a8eeac12e35efd6a30258d0b52daae812ec15df1e77e"
LIMITS_REL = "corpus/rules/limits_v0_pilot.csv"
LIMITS_SHA256 = "334c78811677accb86a2ec14dad24716e16add3b6a14b47abe33ca6fe9e0cccb"
KINDS = ("가", "나", "다", "라", "마")
REASONS = ("r_dim", "r_note", "r_scope", "r_uncov")


class ListsError(ValueError):
    """닫힌 목록 파일이 고정한 바이트가 아니다."""


# ════════════════════════════════════════════════════════════════════ 허용치 표에서 낸 사실(시안 v2 그대로)

def _v(x):
    return getattr(x, "value", x)


@dataclass(frozen=True)
class Crit:
    """기준 하나 — 서술에서 읽은 것과 행에서 만든 것을 같은 꼴로 맞대는 값."""
    t_min: Decimal
    t_max: Decimal | None
    rule: str
    op: str | None = None
    value: Decimal | None = None
    factor: Decimal | None = None
    cap: Decimal | None = None
    basis: str | None = None
    unit: str | None = None


_BASES = {"t": "모재 두께", "s": "용접부 공칭 두께", "a": "목두께"}
_OPS = {"le": "이하", "lt": "미만"}


def _dec(x):
    return None if x in (None, "") else Decimal(str(x))


def crit_of_row(r) -> Crit:
    rule = _v(r.limit_rule)
    tmin = Decimal(str(r.thickness_min))
    tmax = None if r.thickness_max in (None, "") else Decimal(str(r.thickness_max))
    if rule == "none_permitted":
        return Crit(tmin, tmax, rule)
    return Crit(tmin, tmax, rule, op=_v(r.limit_op), value=_dec(r.limit_value), factor=_dec(r.limit_factor),
                cap=_dec(r.limit_cap), basis=_v(r.ratio_basis) if rule != "const" else None,
                unit=_v(r.unit) if rule in ("const", "prop_t_cap") else None)


class Table:
    """허용치 표에서 낸 것 — 조항 · 행 식별자, 조항 접두, RT 축 행, 조항 × 코드의 기준, (재질, 코드)의 필요 유보 사유."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.clause_ids = frozenset(r.clause_id for r in self.rows)
        self.row_clause = {r.rule_id: r.clause_id for r in self.rows}
        self.prefixes = frozenset(c.split("-")[0] for c in self.clause_ids)
        self.rt = [r for r in self.rows if _v(r.scope) == "active" and _v(r.inspection_method) in ("RT", "ALL")]
        self.crits: dict[tuple[str, str], tuple[Crit, ...]] = {}
        for cid in sorted(self.clause_ids):
            for code in sorted({str(r.defect_code) for r in self.rows if r.clause_id == cid}):
                rows = sorted((r for r in self.rows if r.clause_id == cid and str(r.defect_code) == code
                               and _v(r.scope) == "active"), key=lambda r: r.rule_id)
                self.crits[(cid, code)] = tuple(crit_of_row(r) for r in rows)

    def rt_rows(self, material: str, code: str) -> list:
        return sorted((r for r in self.rt if _v(r.material) in (material, "ALL") and str(r.defect_code) == str(code)),
                      key=lambda r: r.rule_id)

    def gold_clauses(self, material: str, codes) -> list[str]:
        """정답 조항 집합 — (재질, 정답 코드)의 RT 축 행이 가리키는 조항. 생성기 · 독립 대조 경로와 같은 선택."""
        return sorted({r.clause_id for c in set(codes) for r in self.rt_rows(material, c)})

    def required(self, material: str, codes) -> set[str]:
        out: set[str] = set()
        for c in set(codes):
            rows = self.rt_rows(material, c)
            if not rows:
                out |= {"r_uncov", "r_dim"}
                continue
            kinds = {_v(r.limit_rule) for r in rows}
            if kinds - {"none_permitted"}:
                out.add("r_dim")
            if "none_permitted" in kinds:
                out.add("r_scope")
            if any((r.note or "").strip() for r in rows if _v(r.limit_rule) != "none_permitted"):
                out.add("r_note")
        return out


# ════════════════════════════════════════════════════════════════════ 정규화 · 문장(시안 v2 그대로)

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−﹣－"), "-")


def norm(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").translate(_DASHES)
    return re.sub(r"\s+", " ", t).strip()


def sentences(text: str) -> list[str]:
    """마침표 · 물음표 · 느낌표 뒤 공백에서 나눈다(소수점은 뒤가 숫자라 나뉘지 않는다)."""
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ════════════════════════════════════════════════════════════════════ 기준 목록 허가(시안 v2 그대로)

_DEC = r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
_CRIT_LIST = re.compile(r"이 조항의 기준은 다음과 같다\. (?P<body>(?:[^.]|\.(?=\d))+)\.")
_ANCHOR = re.compile(r"(?P<name>[^\s.]+)에 적용되는 조항은 (?P<cid>[^\s]+) 이다\. ?$")
_BAND = re.compile(rf"^두께 (?:(?P<lo>{_DEC}) mm (?P<lok>초과|이상))?(?P<sp> )?(?:(?P<hi>{_DEC}) mm (?P<hik>이하|미만))?$")
_C_CONST = re.compile(rf"^(?P<v>{_DEC}) (?P<u>mm|%) (?P<op>이하|미만)$")
_C_PROP = re.compile(rf"^(?P<b>모재 두께|용접부 공칭 두께|목두께)의 (?P<f>{_DEC}) 배 (?P<op>이하|미만)"
                     rf"(?:이고 최대 (?P<cap>{_DEC}) (?P<u>mm|%))?$")
NONE_PERMITTED = "크기와 무관하게 허용하지 않는다"
Q = Decimal("0.01")
_OP_OF = {v: k for k, v in _OPS.items()}
_BASIS_OF = {v: k for k, v in _BASES.items()}
_UNIT_OF = {"mm": "mm", "%": "percent"}


def parse_band(s: str) -> tuple[Decimal, Decimal | None] | None:
    if s == "모든 두께":
        return Decimal(0), None
    m = _BAND.match(s)
    if not m or not (m["lo"] or m["hi"]) or (m["lo"] and m["hi"] and not m["sp"]) or (m["sp"] and not (m["lo"] and m["hi"])):
        return None
    lo = Decimal(m["lo"]) + (Q if m["lok"] == "초과" else 0) if m["lo"] else Decimal(0)
    hi = Decimal(m["hi"]) + (Q if m["hik"] == "이하" else 0) if m["hi"] else None
    return lo, hi


def parse_crit(band: tuple, s: str) -> Crit | None:
    lo, hi = band
    if s == NONE_PERMITTED:
        return Crit(lo, hi, "none_permitted")
    m = _C_CONST.match(s)
    if m:
        return Crit(lo, hi, "const", op=_OP_OF[m["op"]], value=Decimal(m["v"]), unit=_UNIT_OF[m["u"]])
    m = _C_PROP.match(s)
    if m:
        if m["cap"]:
            return Crit(lo, hi, "prop_t_cap", op=_OP_OF[m["op"]], factor=Decimal(m["f"]), cap=Decimal(m["cap"]),
                        basis=_BASIS_OF[m["b"]], unit=_UNIT_OF[m["u"]])
        return Crit(lo, hi, "prop_t", op=_OP_OF[m["op"]], factor=Decimal(m["f"]), basis=_BASIS_OF[m["b"]])
    return None


def parse_body(body: str) -> tuple[Crit, ...] | None:
    out = []
    for item in body.split(" / "):
        band_s, sep, crit_s = item.partition(": ")
        band = parse_band(band_s) if sep else None
        c = parse_crit(band, crit_s) if band else None
        if c is None:
            return None
        out.append(c)
    return tuple(out)


@dataclass
class CritList:
    span: tuple[int, int]
    anchored: bool
    cid: str | None
    name: str | None
    licensed: bool
    reason: str
    has_mm: bool


def scan_criteria(t: str, table: Table, code_of_name: dict) -> list[CritList]:
    out = []
    for m in _CRIT_LIST.finditer(t):
        crits = parse_body(m["body"])
        if crits is None:
            continue                    # 문법 밖 — 기준 목록으로 허가하지 않고 F 에 남긴다
        a = _ANCHOR.search(t[:m.start()])
        has_mm = bool(re.search(r"\d\s*mm", m["body"]))
        if a is None:
            out.append(CritList(m.span(), False, None, None, False, "unanchored", has_mm))
            continue
        cid, name = a["cid"], a["name"]
        code = code_of_name.get(name)
        cands = [v for (c, k), v in table.crits.items() if c == cid and (code is None or k == code)] \
            or [v for (c, _), v in table.crits.items() if c == cid]
        if not cands:
            out.append(CritList(m.span(), True, cid, name, False, "clause_unknown", has_mm))
        elif crits in cands:
            out.append(CritList(m.span(), True, cid, name, True, "licensed", has_mm))
        else:
            out.append(CritList(m.span(), True, cid, name, False, "values_mismatch", has_mm))
    return out


# ════════════════════════════════════════════════════════════════════ (가) ~ (마)의 v2 정규식(시안 v2 그대로)

_LEN_UNIT = r"(?:mm|밀리미터|밀리미타|밀리|미리|cm|센티미터|센티|μm|마이크로미터|마이크론|미크론)"
_QTY = re.compile(rf"(?<![\d.])(?P<n>\d+(?:\.\d+)?)\s*[(\[]?\s*(?P<u>{_LEN_UNIT})(?![A-Za-z])")
_KO_NUM = "하나|한|둘|두|셋|세|넷|네|다섯|여섯|일곱|여덟|아홉|열|일|이|삼|사|오|육|칠|팔|구|십|백"
_QTY_KO = re.compile(rf"(?:(?<=\s)|^)(?P<n>(?:{_KO_NUM})(?:\s*점\s*(?:{_KO_NUM}))?)\s*(?P<u>{_LEN_UNIT})(?![A-Za-z])")
_UNIT_PRECISION = re.compile(r"\s*단위")
_T = r"두께|실치수|실제\s*치수|실제\s*크기|실측|화소\s*당|픽셀\s*당|px\s*당|mm\s*/\s*(?:px|pixel|픽셀|화소)|축척|스케일|환산|배율"
_T_RE = re.compile(_T)
_T_GOV = re.compile(r"(?:두께|실치수|축척|스케일|화소\s*당|픽셀\s*당|배율|화소\s*크기|픽셀\s*크기|화소\s*간격|픽셀\s*간격)"
                    r"\s*(?:는|은|가|이|를|을|의|:)?\s*(?:약\s*)?$")
_BAND_PICK = re.compile(r"(?:첫|두|세|네|마지막|해당|이|그)\s*(?:번째\s*)?(?:두께\s*)?구간(?:을|의|에|이|으로|으로써)?\s*"
                        r"(?:기준|적용|쓴|고른|택|선택|해당)")
_PLATE_N = r"(?:판재|모재|용접부|배관|관재|판(?=$|[\s.,]|이|가|은|는|도|을|를|의|에|으로|과|와))"
_PLATE = re.compile(rf"{_PLATE_N}\s*(?:이|가|은|는|도)?\s*(?:얇|두껍|두꺼)|(?:얇은|두꺼운)\s*{_PLATE_N}")
_ASSERT = re.compile(r"추정|가정|보인다|보이며|보여|보이나|보이고|판단된다|얇|두껍|두꺼|정도|(?:^|\s)약\s|해당|적용한다|적용하면|적용했"
                     r"|적용됨|환산하면|환산한|고려하면|고려할|고려해|로\s*보면|으로\s*보면|이므로|이라면|라면|따르면|기준으로\s*하면")
_SEP = r"[\s·ㆍ・\-_~]*"
_VERDICT = [
    ("불합격", re.compile(rf"불{_SEP}합{_SEP}격")),
    ("합격", re.compile(rf"(?<![가-힣])합{_SEP}격")),
    ("불허", re.compile(r"불허")),
    ("양호", re.compile(r"양호")),
    ("적합", re.compile(r"부적합|(?<![가-힣])적합")),
    ("불량", re.compile(r"(?<!융합)(?<!용입)(?<!융합 )(?<!용입 )불량")),
    ("허용", re.compile(r"허용(?!치)")),
    ("허용치+판정", re.compile(r"허용치\s*(?:를|을|에|의)?\s*(?:이내|이하|미만|초과|넘|벗어|만족|충족|미달|안|내)")),
    ("기준+판정", re.compile(r"기준\s*(?:을|를|에|보다)?\s*(?:이내|미달|초과|넘|만족|충족|부합|맞|벗어|못\s*미|작|크|낮|높)")),
    ("만족·충족", re.compile(r"(?:만족|충족|통과|합치|부합)(?:한다|하지|함|된다|했|하는|하므로|하여|해|스럽)")),
    ("수용·거부", re.compile(r"수용\s*(?:할|하|된|불가|가능|이)|거부|기각|반려|리젝트")),
    ("판정한다", re.compile(r"판정(?:한다|된다|했다|되었다|됨|할\s*수\s*있|이\s*가능|하면)")),
    ("영문", re.compile(r"(?i)(?<![A-Za-z])(?:pass(?:ed)?|fail(?:ed|ure)?|accept(?:ed|able)?|reject(?:ed)?|NG|OK)(?![A-Za-z])")),
]
_W_END = re.compile(r"(?:않는다|않습니다|않음|유보한다|유보함|유보합니다|수 없다|수 없음|수 없습니다)\.?$")
_TOPIC = re.compile(r"불?합격\s*(?:여부|유무)|합격\s*[·ㆍ・/]\s*불합격|합격\s*(?:과|이나|또는|와|및)\s*불합격|합불"
                    r"|불?합격\s*(?:인지|일지|인가)"
                    r"|(?:기준을\s*넘|기준을\s*초과|허용되|허용하|만족하|충족하)(?:는지|은지|ㄴ지|을지|ㄹ지)")
_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?P<tok>[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)+)")
_STD = re.compile(r"(?<![A-Za-z])(?P<body>ISO|EN|IEC|KS|AWS|ASME|ASTM|ANSI|JIS|DIN|BS|GB|NF|IACS|API)(?![A-Za-z])"
                  r"(?P<des>\s*(?:[A-Z]\s*)?\d[\w.\-:]*)?")
_STD_KO = re.compile(r"미국용접학회|미국기계학회|미국재료시험협회|일본공업규격|독일공업규격|영국표준|중국국가표준|한국산업표준"
                     r"|국제표준화기구|유럽표준|국제전기기술위원회")
STD_WHITELIST = ("ISO 6520-1",)
_CUE = re.compile(r"조항|규정|규칙|조문|기준|표준|항목|Rule|Clause|Section|§")
_CRIT_QTY = re.compile(r"\d+(?:\.\d+)?\s*(?:mm|%)\s*(?:이하|미만|이내|이상|초과)")
_OBS_CODE = re.compile(r"\(ISO 6520-1 코드 (?P<c>[0-9]+)\)\s*[0-9]+\s*개")
_CONTRAST = re.compile(r"(?<=지만|으나|는데)\s+|\s+(?:그러나|하지만|다만)\s+|,\s*(?:그러나|하지만|다만)\s+")


def clauses_of(s: str) -> list[str]:
    """대조 이음(…지만 · …으나 · …는데 · 그러나 · 하지만 · 다만)에서만 마디를 나눈다(시안 v2 그대로)."""
    return [c for c in _CONTRAST.split(s) if c.strip()]


def verdict_hits(s: str) -> list[str]:
    """판정어 갈래 — v2 그대로(앎의 부정 판정이 부른다). (나)는 `Checker.value_verdicts` 로 값 없는 갈래를 뺀다."""
    body = _TOPIC.sub(" ", s) if _W_END.search(s) else s
    return [name for name, rx in _VERDICT if rx.search(body)]


def length_qty(s: str) -> list[re.Match]:
    out = []
    for rx in (_QTY, _QTY_KO):
        for m in rx.finditer(s):
            if rx is _QTY and _UNIT_PRECISION.match(s, m.end()):
                continue                 # "1 mm 단위" — 결함의 치수가 아니다
            out.append(m)
    return out


# ════════════════════════════════════════════════════════════════════ 닫힌 목록(v3-lists-3) → 정규식

def _flex(word: str) -> str:
    """목록 낱말의 맞대기 — 공백을 지운 꼴로 맞댄다(03h-2 `matching.word`). 낱말 글자 사이에 공백이 있든 없든 같다."""
    return r"\s*".join(re.escape(ch) for ch in word if not ch.isspace())


def _alt(words) -> re.Pattern:
    words = sorted({w for w in words if w.strip()}, key=lambda w: (-len(w.replace(" ", "")), w))
    if not words:
        raise ListsError("빈 목록 — 빈 대안은 모든 글과 맞으므로 정규식을 만들지 않는다")
    return re.compile("(?:" + "|".join(_flex(w) for w in words) + ")")


def _clause_at(s: str, pos: int) -> str:
    """문장 `s` 의 글자 위치 `pos` 가 든 마디(`clauses_of` 와 같은 나눔)."""
    start = 0
    for m in _CONTRAST.finditer(s):
        if pos < m.start():
            return s[start:m.start()]
        start = m.end()
    return s[start:]


def load_lists(path: Path | None = None, *, expect_sha256: str = LISTS_SHA256) -> dict:
    """닫힌 목록의 정본 사본을 바이트로 읽는다. sha256 이 고정한 값과 다르면 읽지 않는다."""
    raw = Path(path or REPO / LISTS_REL).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != expect_sha256:
        raise ListsError(f"닫힌 목록의 sha256 {got[:12]}… 이 고정값 {expect_sha256[:12]}… 과 다르다")
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("version") != "v3-lists-3":
        raise ListsError(f"닫힌 목록의 판 {doc.get('version')!r} — v3-lists-3 여야 한다")
    return doc


@dataclass(frozen=True)
class Rules:
    """닫힌 목록에서 낸 정규식 묶음 — 목록 낱말을 코드에 다시 적지 않는다."""

    absence_always: re.Pattern
    absence_bare: re.Pattern
    h1_target: re.Pattern
    h1_pred: re.Pattern
    h1_question: re.Pattern
    h1_conditional: re.Pattern
    h1_sublabel: str
    h2_cue: re.Pattern
    h2_literal: tuple[tuple[re.Pattern, str, str], ...]
    h2_table: dict[str, str]
    h2_class: dict[str, str]
    h2_mode: re.Pattern
    h3_material: re.Pattern
    h3_criteria: re.Pattern
    h3_apply: re.Pattern
    h3_negation: re.Pattern
    h3_contrast: re.Pattern
    h3_sublabel: str
    withhold: re.Pattern
    not_accepted: re.Pattern
    info: dict[str, re.Pattern]
    uncov_object: re.Pattern
    contradiction: dict[str, tuple[str, ...]]
    na_excluded: frozenset[str]
    ra_cells: tuple[str, ...]
    ra_separate: str
    ra_modes: tuple[str, ...]


_TABLE_NO = re.compile(r"표\s*(?P<n>\d+)")
_CLASS_NO = re.compile(r"제\s*(?P<n>\d+)\s*종")


def build_rules(doc: dict) -> Rules:
    ab = doc["reservation"]["absence"]
    bare = (f"(?:{_alt(ab['targets']).pattern})\\s*(?:{_alt(ab['connectors']).pattern})?\\s*"   # 이음 칸 — v3.1
            f"(?:{_alt(ab['noun_after_target']).pattern})?\\s*"
            f"(?:{_alt(ab['particles']).pattern})?\\s*(?:(?:{_alt(ab['adverbs']).pattern})\\s*)?"
            f"(?:{_alt(ab['bare_eop_forms']).pattern})")
    h1 = doc["H1"]
    targets = list(h1["target_words"]["fixed"]) + list(doc["derived"]["defect_names"])
    h2 = doc["H2"]
    table_no, class_no, literal = {}, {}, []
    for row in doc["derived"]["alias_table"]:
        alias, cid = row["alias"], row["clause_id"]
        if (m := _TABLE_NO.fullmatch(alias)):
            table_no[m["n"]] = cid
        elif (m := _CLASS_NO.fullmatch(alias)):
            class_no[m["n"]] = cid
        else:
            literal.append((re.compile(_flex(alias)), alias, cid))
    h3 = doc["H3"]
    lg = h3["lone_gang"]                       # 따로 선 '강'(v3.1, 04 §7-8) — 뒤 토씨에서 '도' 를 뺐다
    if lg["before_space_or_start"] is not True or not lg["after_particles"] or not all(lg["after_particles"]):
        raise ListsError("따로 선 낱말 — 앞 조건은 '공백 또는 문장 시작' 만 읽고, 뒤 토씨는 빈 낱말 없이 하나 이상이어야 한다")
    after = [re.escape(p) for p in lg["after_particles"]] + ([r"\s", "$"] if lg["after_space_or_end"] else [])
    mats = [w for w in h3["materials"] if w != lg["word"]]
    lone = rf"(?:(?<=\s)|^){re.escape(lg['word'])}(?={'|'.join(after)})"
    res = doc["reservation"]
    withhold = re.compile(f"(?:{_alt(res['verdict_nouns']).pattern})[^.]{{0,24}}?(?:{_alt(res['reservation_predicates']).pattern})")
    na = res["not_accepted"]
    reasons = res["reasons"]
    sub = doc["H2"]["sublabel_conditional"]
    ra = doc["ra_sublabels"]
    return Rules(
        absence_always=_alt(ab["always"]), absence_bare=re.compile(bare),
        h1_target=_alt(targets), h1_pred=_alt(h1["predicates"]),
        h1_question=_alt(h1["not_caught"]["embedded_question"]), h1_conditional=_alt(h1["not_caught"]["conditional"]),
        h1_sublabel=h1["sublabel"],
        h2_cue=_alt(h2["cues"]), h2_literal=tuple(literal), h2_table=table_no, h2_class=class_no,
        h2_mode=_alt(list(sub["conditional_markers"]) + list(sub["intro_markers"])),
        h3_material=re.compile(f"(?:{_alt(mats).pattern}|{lone})"), h3_criteria=_alt(h3["criteria_words"]),
        h3_apply=_alt(h3["apply_predicates"]), h3_negation=_alt(h3["not_caught"]["negation"]),
        h3_contrast=_alt(h3["not_caught"]["contrast"]), h3_sublabel=h3["sublabel"],
        withhold=withhold, not_accepted=_alt(list(na["double_negation"]) + list(na["possibility"])),
        info={r: _alt(reasons[r]["information"]) for r in REASONS},
        uncov_object=_alt(reasons["r_uncov"]["object_predicates"]),
        contradiction={r: tuple(v) for r, v in res["contradiction_pairs"].items()},
        na_excluded=frozenset(doc["na_value_rule"]["excluded_from_na"]),
        ra_cells=tuple(ra["cells"]), ra_separate=ra["separate"].split("(")[0].strip(), ra_modes=tuple(ra["mode"]),
    )


# ════════════════════════════════════════════════════════════════════ 검사기

@dataclass
class Result:
    kinds: dict = field(default_factory=lambda: {k: [] for k in KINDS})
    required: list = field(default_factory=list)
    found: list = field(default_factory=list)
    criteria: list = field(default_factory=list)
    mentions: list = field(default_factory=list)
    claimed: list = field(default_factory=list)
    subtags: set = field(default_factory=set)
    """세부 표지 — H1 · H3 의 세부, (라) 세부 셋의 칸, 미해석 조항 언급, (라) 조건부 소개 · 인용의 꼴."""
    ra_items: list = field(default_factory=list)
    """(라) 한 건마다 (근거 글, 세부 칸, 꼴)."""
    unresolved: list = field(default_factory=list)
    """잇지 못한 별칭(미해석 조항 언급) — (라)로 세지 않는다."""
    found_sentences: dict = field(default_factory=dict)
    """인정한 사유마다 그 문장(정규화 뒤)."""
    withdrawn: dict = field(default_factory=dict)
    """조건 1 ~ 3 은 찼으나 조건 4(서술 안의 짝 금지 주장)로 인정하지 않은 사유 → 짝 종류."""

    @property
    def flagged(self) -> set[str]:
        return {k for k, v in self.kinds.items() if v}

    @property
    def reservation_complete(self) -> bool | None:
        return None if not self.required else set(self.required) <= set(self.found)


class Checker:
    """`evaluation.narrative_metrics.Checker` 꼴 — `check` 와 `required_for`."""

    def __init__(self, table: Table, names: dict[str, str], lists: dict):
        self.version = RULE_VERSION
        self.t = table
        self.R = build_rules(lists)
        self.code_of_name = {v: k for k, v in names.items()}
        self._name_rx = re.compile("(?P<n>" + "|".join(sorted(map(re.escape, self.code_of_name), key=len, reverse=True))
                                   + r")\s*(?:\(ISO 6520-1 코드 [0-9]+\))?\s*[0-9]+\s*개")

    # ── 공통 판정

    def absent(self, c: str) -> bool:
        """부재 — 늘 부재인 꼴, 또는 부재 대상 낱말 뒤(이음 · 이름씨 · 토씨 · 꾸밈말을 건너)의 맨 '없' 꼴(04 §7-5 ② · v3.1 넓힘)."""
        return bool(self.R.absence_always.search(c) or self.R.absence_bare.search(c))

    def negated_knowledge(self, c: str) -> bool:
        """'앎의 부정' 마디 — 부재가 있고, (가) 치수 밖의 수 · 판정어(v2 목록 전체) · 단정 표지가 없다."""
        rest = c
        for m in sorted(length_qty(c), key=lambda m: -m.start()):
            rest = rest[:m.start()] + " " + rest[m.end():]
        return self.absent(c) and not re.search(r"\d", rest) and not verdict_hits(c) and not _ASSERT.search(c)

    def value_verdicts(self, s: str) -> list[str]:
        """(나)의 판정어 — 값 없는 판정 동작 갈래를 뺀다(04 §7-6 ②)."""
        return [h for h in verdict_hits(s) if h not in self.R.na_excluded]

    def valid_withhold(self, s: str) -> bool:
        """판정 명사 뒤 24 글자 안의 유보 술어 — 그 자리가 이중 부정 · 가능 긍정이면 인정하지 않는다."""
        return any(not self.R.not_accepted.search(s, m.start(), min(len(s), m.end() + 12))
                   for m in self.R.withhold.finditer(s))

    def reservation_clause(self, c: str) -> bool:
        return bool(self.R.withhold.search(c))

    def assertive(self, c: str, s: str) -> bool:
        """단정 — 내포 의문 · 조건 꼴이 아니고, 유보 마디가 아니며, 문장이 물음표로 끝나지 않는다(03h-2 `matching.assertive`)."""
        return not (self.R.h1_question.search(c) or self.R.h1_conditional.search(c) or self.reservation_clause(c)
                    or s.rstrip().endswith("?"))

    # ── 세 구멍

    def h1(self, c: str, s: str) -> str | None:
        """판정 낱말 없는 허용 · 안전 단정 — 대상 낱말과 허용 · 안전 술어가 한 마디에, 단정으로."""
        p = self.R.h1_pred.search(c)
        if p and self.R.h1_target.search(c) and self.assertive(c, s):
            return p.group(0)
        return None

    def h3(self, c: str, s: str) -> str | None:
        """AL 에서 다른 재질 기준 — 다른 재질 · 기준 낱말 · 적용 술어가 한 마디에, 부정 · 대조가 아니고 단정으로."""
        m = self.R.h3_material.search(c)
        if not (m and self.R.h3_criteria.search(c) and self.R.h3_apply.search(c)):
            return None
        if self.R.h3_negation.search(c) or self.R.h3_contrast.search(c) or not self.assertive(c, s):
            return None
        return m.group(0)

    def aliases(self, c: str) -> list[tuple[str, str | None]]:
        """마디의 별칭 언급 → (별칭, 조항 ID 또는 None=미해석). '표 N' 꼴은 단서를 스스로 품고, 나머지는 조항 단서가 같은 마디에 있어야 한다."""
        out: list[tuple[str, str | None]] = []
        for m in _TABLE_NO.finditer(c):
            out.append((f"표{m['n']}", self.R.h2_table.get(m["n"])))
        if self.R.h2_cue.search(c):
            for m in _CLASS_NO.finditer(c):
                out.append((f"제{m['n']}종", self.R.h2_class.get(m["n"])))
            for rx, alias, cid in self.R.h2_literal:
                if rx.search(c):
                    out.append((alias, cid))
        return out

    # ── 판정

    def claimed_codes(self, t: str, structured) -> list[str]:
        got = set(map(str, structured))
        got |= {m["c"] for m in _OBS_CODE.finditer(t)}
        got |= {self.code_of_name[m["n"]] for m in self._name_rx.finditer(t)}
        return sorted(got)

    def required_for(self, material: str, codes) -> set[str]:
        return self.t.required(material, list(codes))

    def _ra(self, res: Result, label: str, cell: str, clause: str) -> None:
        mode = self.R.ra_modes[1] if self.R.h2_mode.search(clause) else self.R.ra_modes[0]
        res.kinds["라"].append(label)
        res.ra_items.append({"label": label, "cell": cell, "mode": mode})
        res.subtags.add(cell)
        if mode == self.R.ra_modes[1]:
            res.subtags.add(mode)

    def check(self, narrative: str, *, material: str, gold_codes, structured_codes) -> Result:
        res = Result()
        t = norm(narrative)
        G = set(self.t.gold_clauses(material, gold_codes))
        lists = scan_criteria(t, self.t, self.code_of_name)
        res.criteria = [{"anchored": c.anchored, "cid": c.cid, "licensed": c.licensed, "reason": c.reason} for c in lists]
        for c in lists:
            if not c.licensed and c.has_mm:
                res.kinds["가"].append(f"기준 수치 오기({c.reason})")
            if material == "AL":
                res.kinds["마"].append("AL 기준 목록")
        F = self._free(t, lists)

        for s in sentences(F):
            for m in length_qty(s):
                res.kinds["가"].append(f"치수: {m.group(0).strip()}")
                if _T_GOV.search(s[:m.start()]):
                    res.kinds["다"].append(f"두께 · 축척 값: {m.group(0).strip()}")
            res.kinds["나"] += [f"판정어: {h}" for h in self.value_verdicts(s)]
            for c in clauses_of(s):
                if _T_RE.search(c) and not self.negated_knowledge(c):
                    res.kinds["다"].append(f"두께 · 실치수 마디: {c[:40]}")
                if (p := self.h1(c, s)) is not None:
                    res.kinds["나"].append(f"{self.R.h1_sublabel}: {p}")
                    res.subtags.add(self.R.h1_sublabel)
                if material == "AL" and (w := self.h3(c, s)) is not None:
                    res.kinds["마"].append(f"{self.R.h3_sublabel}: {w}")
                    res.subtags.add(self.R.h3_sublabel)
            if _BAND_PICK.search(s):
                res.kinds["다"].append(f"두께 구간 고르기: {_BAND_PICK.search(s).group(0)}")
            if _PLATE.search(s):
                res.kinds["다"].append(f"판 두께 단정: {_PLATE.search(s).group(0)}")
            if material == "AL" and _CRIT_QTY.search(s):
                res.kinds["마"].append("AL 기준 꼴 수량")

        res.claimed = self.claimed_codes(t, structured_codes)
        own = set(self.t.gold_clauses(material, res.claimed))
        cell_own, cell_other, cell_none = self.R.ra_cells[1], self.R.ra_cells[2], self.R.ra_cells[0]
        for s in sentences(t):
            cue = bool(_CUE.search(s))
            for m in _TOKEN.finditer(s):
                tok = m["tok"]
                pre = tok.split("-")[0]
                if pre in self.t.prefixes:
                    if tok in self.t.clause_ids:
                        res.mentions.append(tok)
                        if tok not in G:
                            mine = tok in own
                            self._ra(res, f"정답 밖 조항({'제 주장과는 맞음' if mine else '제 주장과도 어긋남'}): {tok}",
                                     cell_own if mine else cell_other, _clause_at(s, m.start()))
                    elif tok in self.t.row_clause:
                        par = self.t.row_clause[tok]
                        if par in G:
                            res.kinds["다"].append(f"행 식별자 = 두께 구간 선택: {tok}")
                        else:
                            self._ra(res, f"정답 밖 조항의 행: {tok}", cell_own if par in own else cell_other,
                                     _clause_at(s, m.start()))
                    else:
                        self._ra(res, f"없는 조항: {tok}", cell_none, _clause_at(s, m.start()))
                    if material == "AL":
                        res.kinds["마"].append(f"AL 조항 특정: {tok}")
                elif cue and any(ch.isdigit() for ch in tok.partition("-")[2]):
                    self._ra(res, f"없는 조항(조항 단서): {tok}", cell_none, _clause_at(s, m.start()))
            for m in _STD.finditer(s):
                ref = re.sub(r"\s+", " ", (m["body"] + (m["des"] or ""))).strip()
                if any(ref == w or ref.startswith(w) for w in STD_WHITELIST):
                    continue
                self._ra(res, f"표준 인용: {ref}", cell_none, _clause_at(s, m.start()))
            for m in _STD_KO.finditer(s):
                self._ra(res, f"표준 인용: {m.group(0)}", cell_none, _clause_at(s, m.start()))
            for c in clauses_of(s):
                for alias, cid in self.aliases(c):
                    if cid is None:
                        res.unresolved.append(alias)
                        res.subtags.add(self.R.ra_separate)
                    elif cid not in G:
                        mine = cid in own
                        self._ra(res, f"별칭 조항({'제 주장과는 맞음' if mine else '제 주장과도 어긋남'}): {alias} → {cid}",
                                 cell_own if mine else cell_other, c)
                    if material == "AL":
                        res.kinds["마"].append(f"AL 조항 특정(별칭): {alias}")

        res.required = sorted(self.t.required(material, res.claimed))
        res.found = sorted(self.reasons(t, lists, res), key=REASONS.index)
        return res

    @staticmethod
    def _free(t: str, lists) -> str:
        free, i = [], 0
        for c in lists:
            free.append(t[i:c.span[0]])
            free.append(" § ")
            i = c.span[1]
        free.append(t[i:])
        return re.sub(r"\s+", " ", "".join(free)).strip()

    def reasons(self, t: str, lists, res: Result | None = None) -> set[str]:
        """유보 인정 v3 — 조건 1 ~ 3 은 문장 안, 조건 4 는 서술 단위(`res` 의 걸린 종류)."""
        out: set[str] = set()
        for s in sentences(self._free(t, lists)):
            w = self.valid_withhold(s)
            got = set()
            if w and any(self.R.info["r_dim"].search(c) and self.negated_knowledge(c) for c in clauses_of(s)):
                got.add("r_dim")
            if w and self.R.info["r_note"].search(s):
                got.add("r_note")
            if w and self.R.info["r_scope"].search(s):
                got.add("r_scope")
            if self.R.info["r_uncov"].search(s) and any(
                    not self.R.not_accepted.search(s, m.start(), min(len(s), m.end() + 12))
                    for m in self.R.uncov_object.finditer(s)):
                got.add("r_uncov")
            for r in got:
                if res is not None:
                    res.found_sentences.setdefault(r, []).append(s)
            out |= got
        if res is not None:
            flags = res.flagged
            for r in sorted(out, key=REASONS.index):
                clash = sorted(set(self.R.contradiction.get(r, ())) & flags, key=KINDS.index)
                if clash:
                    out.discard(r)
                    res.withdrawn[r] = clash
        return out


def load_checker(repo: Path | None = None, *, lists_path: Path | None = None) -> Checker:
    """허용치 표(sha256 고정) · 사상표 · 닫힌 목록(sha256 고정)으로 검사기를 만든다."""
    from corpus.rules import limits_loader
    from data.label_map import load_label_map

    root = Path(repo or REPO)
    limits = root / LIMITS_REL
    got = hashlib.sha256(limits.read_bytes()).hexdigest()
    if got != LIMITS_SHA256:
        raise ListsError(f"허용치 표의 sha256 {got[:12]}… 이 고정값과 다르다")
    table = Table(limits_loader.load_limits(limits, pilot=True).rows)
    names = {dt.iso_code: dt.name_ko for dt in load_label_map(root / "configs" / "label_map.yaml").defect_types.values()}
    return Checker(table, names, load_lists(lists_path or root / LISTS_REL))


__all__ = ["CHECKER_ID", "KINDS", "LISTS_REL", "LISTS_SHA256", "REASONS", "RULE_VERSION", "Checker", "ListsError", "Result",
           "Rules", "Table", "build_rules", "clauses_of", "load_checker", "load_lists", "norm", "sentences", "verdict_hits"]

"""금지 주장 검사기 v4 — 2 판 정답 서술의 금지 주장 · 유보 판정(미니스펙 10 7판, 닫힌 목록 v4-lists-2).

v3.1(`evaluation/narrative_checker.py`)의 판별을 그대로 물려받고 10 7판이 정한 곳만 바꾼다. v3.1 파일 · 상수 · 닫힌 목록 v3-3 은 구판
재현용으로 그대로 둔다(§1).

- **판 · 핀**(§1). 2 판 표(`LIMITS_REL`, sha256 고정) · 닫힌 목록 v4-lists-2(sha256 고정). 주 · 보조 조항은 생성 쪽 `derive_clause_roles` →
  평가 쪽 `roles_from_derived` 로 받는다. 정답 조항 = 이미지 코드들의 주 조항 합집합, 보조 전용 = 보조 합집합 − 주 합집합(07 §13-3).
  제 주장 조항은 v3.1 대로 (재질, 주장한 코드)의 RT 행이 가리키는 조항이다.
- **앵커**(§2). 기준 목록 바로 앞 문장 전체가 2 판 대응 문구(C2)나 1 판 꼴(Q4)과 같을 때만 앵커다.
- **기준 목록 머리**(§3). C3a(측정량을 적은 머리) · C3b. C3a 의 측정량이 그 조항의 측정량과 다르거나 그 조항이 불허 조항이면 (가)
  '기준 오기(측정량)' 이고 목록은 허가하지 않는다. 띠 · 기준의 문법은 v3.1 그대로다.
- **보조 전용 조항의 언급**(§4). 언급의 작은 마디에 조건 표지((i) `H2.exempt_conditional` · (ii) `H2.conditional_ending`)가 있고 그 대조
  마디에 유효한 적용 단정이 없을 때만 분자 밖이다. 적용 단정은 적용 술어마다 그 작은 마디에서 판별하고 대조 마디에서 모은다. 꼴 표지는
  `ra_sublabels.mode_order` 의 차례 1 ~ 4. 진단 표지 '보조 조건부 소개' · '보조 면제 막힘(다른 작은 마디의 단정)'.
- **두 층의 마디**(§5). 대조 마디(v3.1)와 작은 마디(`clause_split`). (나) H1 · (마) H3 · (다) 마디 갈래는 대조 마디에 그 갈래의 예외가
  있을 때만 작은 마디로 잰다. (다) 새 세부 표지 둘은 촉발 낱말의 작은 마디에서 예외를 보고, 그 작은 마디가 평서로 끝나면 단다. 작은 마디
  갈래의 걸림에 앞 작은 마디의 조건이 있으면 진단 표지 '조건 범위 단절 가능'.
- **r_note 의 인정**(§6). '단서' 뒤(토씨 하나 · 꾸밈말을 건넌 자리)가 허용 꼴로 시작하고 거부 꼴로 시작하지 않을 때만.

2 판 문장(C2 · C3a · C3b)은 생성 쪽 상수와 같은 글자다 — 검사기는 생성기를 import 하지 않고, 시험이 맞댄다(§11). 진단 표지는 세부
표지가 아니다 — `Result.diagnostics` 에 따로 싣고 분자 · 분모를 바꾸지 않는다(§7).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from evaluation import narrative_checker as V3

REPO = Path(__file__).resolve().parents[1]
RULE_VERSION = "v4"
CHECKER_ID = "narrative_checker_v4"
LISTS_REL = "evaluation/vectors/narrative_lists.v4-2.json"
LISTS_SHA256 = "651b7d6ca7d55f07ede6894c92fb111abd6ecb2a4e5f619daac7f8bb9dcc4709"
LISTS_VERSION = "v4-lists-2"
LIMITS_REL = "corpus/rules/limits_v2_pilot.csv"
LIMITS_SHA256 = "0c76b6a639fc2d617115122fbc10204fcc0ccf665e9b4c6d27f679057294ef4d"
KINDS = V3.KINDS
REASONS = V3.REASONS
ListsError = V3.ListsError

T_CORRESPOND = "{name}에 현 허용치 표에서 대응되는 조항은 {cid} 이다."
"""2 판 대응 문구(C2) — 생성 쪽 `pairs_text_v2.CORRESPOND` 와 같은 글자(시험이 맞댄다)."""
T_ANCHOR_V1 = "{name}에 적용되는 조항은 {cid} 이다."
"""1 판 꼴(Q4) — 등록상 대응 표현으로만 받는다. v3.1 의 `_ANCHOR` 와 같은 꼴."""
T_HEAD_MEASURE = "이 조항의 측정량은 {measure}이고, 기준은 다음과 같다. "
"""C3a — 생성 쪽 `pairs_text_v2.CRIT_HEAD_MEASURE` 와 같은 글자."""
T_HEAD = "이 조항의 기준은 다음과 같다. "
"""C3b — 1 판 머리 그대로(생성 쪽 `pairs_text_v2.CRIT_HEAD`)."""
MEASURE_SUBTAG = "기준 오기(측정량)"
"""(가)의 세부 — C3a 의 측정량이 그 조항의 측정량과 다르거나 불허 조항에 C3a 머리를 썼다(§3)."""


def _anchor_rx(tpl: str) -> re.Pattern:
    out = []
    for part in re.split(r"(\{name\}|\{cid\})", tpl):
        out.append(r"(?P<name>[^\s.]+)" if part == "{name}" else r"(?P<cid>\S+)" if part == "{cid}" else re.escape(part))
    return re.compile("^" + "".join(out) + "$")


_ANCHORS = (("C2", _anchor_rx(T_CORRESPOND)), ("Q4", _anchor_rx(T_ANCHOR_V1)))
_HM0, _HM1 = T_HEAD_MEASURE.split("{measure}")
_LIST = re.compile("(?:" + re.escape(_HM0) + r"(?P<measure>[^\s,.]+)" + re.escape(_HM1) + "|" + re.escape(T_HEAD) + ")"
                   + r"(?P<body>(?:[^.]|\.(?=\d))+)\.")
_HEAD_ANY = re.compile("(?:" + re.escape(_HM0) + r"[^\s,.]+" + re.escape(_HM1) + "|" + re.escape(T_HEAD) + ")")
_MYEON = re.compile(r"(\S*면)(?=[\s,.]|$)")
_DECL_END = re.compile(r"다\s*\.?\s*$")
_DECL_MID = re.compile(r"다(?=[\s,])")


def _ns(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _core(e: str) -> str:
    return re.sub(r"[^가-힣]+$", "", re.sub(r"^[^가-힣]+", "", e))


def _spaced(word: str) -> re.Pattern:
    return re.compile(V3._flex(word))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def contrast_spans(s: str) -> list[tuple[int, int]]:
    """대조 마디의 자리 — v3.1 의 `clauses_of` 와 같은 나눔(대조 이음에서만)."""
    out, start = [], 0
    for m in V3._CONTRAST.finditer(s):
        out.append((start, m.start()))
        start = m.end()
    out.append((start, len(s)))
    return [(a, b) for a, b in out if s[a:b].strip()]


# ════════════════════════════════════════════════════════════════════ 허용치 표 — 2 판 · 주 · 보조 조항 · 측정량

class Table(V3.Table):
    """2 판 표에서 낸 것 — v3.1 의 칸에 주 · 보조 조항(`roles`)과 조항의 측정량을 더한다. 1 판 표는 받지 않는다(섞임 거부)."""

    def __init__(self, rows, roles: Mapping, *, edition: str):
        if edition != "v2":
            raise ListsError(f"v4 는 2 판 표로만 잰다 — 표의 판 {edition!r}")
        super().__init__(rows)
        self.edition = edition
        self.roles = {(str(m), str(c)): (main, frozenset(aux)) for (m, c), (main, aux) in roles.items()}
        meas: dict[str, str | None] = {}
        for r in self.rows:
            if V3._v(r.scope) != "active":
                continue
            mv = V3._v(getattr(r, "measure", None)) or None
            if meas.setdefault(r.clause_id, mv) != mv:
                raise ListsError(f"조항 {r.clause_id} 의 측정량이 행마다 다르다")
        self.measure = meas

    def primary(self, material: str, codes) -> set[str]:
        return {m for c in set(map(str, codes)) if (m := self.roles.get((material, c), (None, frozenset()))[0])}

    def auxiliary(self, material: str, codes) -> set[str]:
        return {a for c in set(map(str, codes)) for a in self.roles.get((material, c), (None, frozenset()))[1]}

    def gold_clauses(self, material: str, codes) -> list[str]:
        """정답 조항 — 이미지 코드들의 주 조항 합집합(07 §13-3 (나))."""
        return sorted(self.primary(material, codes))

    def own_clauses(self, material: str, codes) -> list[str]:
        """제 주장 조항 — (재질, 주장한 코드)의 RT 행이 가리키는 조항(v3.1 의 `gold_clauses` 와 같은 선택)."""
        return super().gold_clauses(material, codes)

    def auxiliary_only(self, material: str, codes) -> set[str]:
        """이미지의 보조 전용 조항 — 보조 합집합 − 주 합집합(07 §13-3 규칙 1)."""
        return self.auxiliary(material, codes) - self.primary(material, codes)


# ════════════════════════════════════════════════════════════════════ 닫힌 목록 v4-lists-2

def load_lists(path: Path | None = None, *, expect_sha256: str = LISTS_SHA256) -> dict:
    """닫힌 목록의 정본 사본을 바이트로 읽는다. sha256 · 판이 고정한 값과 다르면 읽지 않는다."""
    raw = Path(path or REPO / LISTS_REL).read_bytes()
    got = _sha(raw)
    if got != expect_sha256:
        raise ListsError(f"닫힌 목록의 sha256 {got[:12]}… 이 고정값 {expect_sha256[:12]}… 과 다르다")
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("version") != LISTS_VERSION:
        raise ListsError(f"닫힌 목록의 판 {doc.get('version')!r} — {LISTS_VERSION} 여야 한다")
    return doc


@dataclass(frozen=True)
class DGroup:
    sublabel: str
    triggers: tuple[str, ...]
    exclude: tuple[str, ...]
    cond: re.Pattern
    know: re.Pattern
    question: re.Pattern


@dataclass(frozen=True)
class Rules4:
    """v4 가 더한 칸에서 낸 맞대기 — 목록 낱말을 코드에 다시 적지 않는다."""

    exempt_words: tuple[str, ...]
    intro: re.Pattern
    ce_exact: frozenset[str]
    ce_suffix: tuple[str, ...]
    ce_forms: tuple[str, ...]
    ce_following: tuple[str, ...]
    apply_ns: tuple[str, ...]
    not_da: tuple[str, ...]
    not_hae: tuple[str, ...]
    au_question: re.Pattern
    au_neg: re.Pattern
    split: re.Pattern
    d: dict
    demonstratives: frozenset[str]
    demonstratives_two: frozenset[str]
    rn_particles: tuple[str, ...]
    rn_adverbs: tuple[str, ...]
    rn_allow: tuple[str, ...]
    rn_reject: tuple[str, ...]
    modes: tuple[str, ...]
    subtag_modes: frozenset[str]
    diag_exempt: str
    diag_blocked: str
    diag_split: str
    h1_conditional_words: tuple[str, ...]


def build_rules4(doc: dict) -> Rules4:
    h2 = doc["H2"]
    ce = h2["conditional_ending"]
    cs = doc["clause_split"]
    if cs["connectives"] != ["고", "며", "므로"] or not all(len(x) == 2 and x.endswith("고") for x in cs["no_split_quote"]["eojeol_ends"]):
        raise ListsError("작은 마디의 경계가 v4-lists-2 의 꼴이 아니다(연결 어미 셋 · 인용 '?고')")
    aux = "|".join(map(re.escape, cs["no_split_aux"]["next_word_starts"]))
    quote = "|".join(re.escape(x[0]) for x in cs["no_split_quote"]["eojeol_ends"])
    split = re.compile(rf"(?<!{quote})(?P<c>고(?!\s+(?:{aux}))|며|므로)(?:,\s*|\s+|,)")
    au = doc["aux_exemption"]["assertion_unit"]
    d = {}
    for grp in ("premise", "cluster"):
        g = doc["D"][grp]
        nc = g["not_caught"]
        d[grp] = DGroup(g["sublabel"], tuple(g["triggers"]), tuple(g["exclude"]), V3._alt(nc["conditional"]),
                        V3._alt(nc["knowledge_negation"]), V3._alt(nc["question"]))
    gy = doc["D"]["gyeongu_clause"]
    rn = doc["reservation"]["reasons"]["r_note"]["presence"]
    ra = doc["ra_sublabels"]
    order = sorted(ra["mode_order"], key=lambda x: x["order"])
    if [x["order"] for x in order] != [1, 2, 3, 4]:
        raise ListsError("꼴 표지의 차례가 1 ~ 4 가 아니다")
    tags = ra["diagnostic_tags"]["tags"]
    na = doc["ra_not_assertion"]
    return Rules4(
        exempt_words=tuple(h2["exempt_conditional"]), intro=V3._alt(h2["sublabel_conditional"]["intro_markers"]),
        ce_exact=frozenset(ce["exclude_eojeol_exact"]), ce_suffix=tuple(ce["exclude_eojeol_suffix"]),
        ce_forms=tuple(ce["exclude_eojeol_forms"]), ce_following=tuple(ce["exclude_following"]["next_words"]),
        apply_ns=tuple(sorted({_ns(w) for w in doc["ra_apply"]}, key=len, reverse=True)),
        not_da=tuple(_ns(x) for x in na["after_da"]), not_hae=tuple(_ns(x) for x in na["after_hae"]),
        au_question=V3._alt(au["question"]), au_neg=V3._alt(au["knowledge_negation"]), split=split, d=d,
        demonstratives=frozenset(gy["demonstratives"]), demonstratives_two=frozenset(gy["demonstratives_two_words"]),
        rn_particles=tuple(rn["particles"]), rn_adverbs=tuple(sorted(rn["adverbs"], key=len, reverse=True)),
        rn_allow=tuple(_ns(x) for x in rn["allow"]), rn_reject=tuple(_ns(x) for x in rn["reject"]),
        modes=tuple(x["mode"] for x in order), subtag_modes=frozenset(ra["subtag_modes"]),
        diag_exempt=tags[0], diag_blocked=tags[1], diag_split=tags[2],
        h1_conditional_words=tuple(doc["H1"]["not_caught"]["conditional"]),
    )


# ════════════════════════════════════════════════════════════════════ 기준 목록 — 앵커 C2 · Q4, 머리 C3a · C3b

@dataclass
class CritList:
    span: tuple[int, int]
    anchored: bool
    cid: str | None
    name: str | None
    licensed: bool
    reason: str
    has_mm: bool
    head: str
    measure: str | None
    measure_error: bool
    anchor_form: str | None


def scan_criteria(t: str, table: Table, code_of_name: dict) -> list[CritList]:
    out = []
    for m in _LIST.finditer(t):
        crits = V3.parse_body(m["body"])
        if crits is None:
            continue                    # 문법 밖 — 기준 목록으로 허가하지 않고 F 에 남긴다(v3.1 그대로)
        head = "C3a" if m["measure"] else "C3b"
        has_mm = bool(re.search(r"\d\s*mm", m["body"]))
        pre = V3.sentences(t[:m.start()].rstrip())
        last = pre[-1] if pre else ""
        anchor = next(((form, a) for form, rx in _ANCHORS if (a := rx.match(last))), None)
        if anchor is None:
            out.append(CritList(m.span(), False, None, None, False, "unanchored", has_mm, head, m["measure"], False, None))
            continue
        form, a = anchor
        cid, name = a["cid"], a["name"]
        code = code_of_name.get(name)
        cands = [v for (c, k), v in table.crits.items() if c == cid and (code is None or k == code)] \
            or [v for (c, _), v in table.crits.items() if c == cid]
        if not cands:
            out.append(CritList(m.span(), True, cid, name, False, "clause_unknown", has_mm, head, m["measure"], False, form))
            continue
        measure_error = head == "C3a" and (table.measure.get(cid) is None or table.measure.get(cid) != m["measure"])
        if measure_error:
            out.append(CritList(m.span(), True, cid, name, False, "measure_mismatch", has_mm, head, m["measure"], True, form))
        elif crits in cands:
            out.append(CritList(m.span(), True, cid, name, True, "licensed", has_mm, head, m["measure"], False, form))
        else:
            out.append(CritList(m.span(), True, cid, name, False, "values_mismatch", has_mm, head, m["measure"], False, form))
    return out


# ════════════════════════════════════════════════════════════════════ 검사기

@dataclass
class Result(V3.Result):
    diagnostics: list = field(default_factory=list)
    """진단 표지 — {tag, where}. 세부 표지가 아니다(분자 · 분모를 바꾸지 않는다, §7)."""
    subtags_by_kind: dict = field(default_factory=lambda: {k: set() for k in KINDS})
    aux_mentions: list = field(default_factory=list)
    """보조 전용 조항의 언급마다 {clause, exempt, blocked, mode}."""
    list_heads: int = 0
    """글에 든 기준 목록 머리(C3a · C3b)의 수 — 문법 밖이라 목록으로 읽지 못한 것도 든다."""

    @property
    def diagnostic_tags(self) -> set[str]:
        return {d["tag"] for d in self.diagnostics}


class Checker(V3.Checker):
    """v3.1 의 판별을 물려받는다 — `check` · `reasons` 와 v4 가 더한 판별만 새로 쓴다."""

    def __init__(self, table: Table, names: dict[str, str], lists: dict):
        if not isinstance(table, Table):
            raise ListsError("v4 는 v4 의 표(2 판 · 주 · 보조 조항)로만 잰다")
        super().__init__(table, names, lists)
        self.version = RULE_VERSION
        self.L = lists
        self.R4 = build_rules4(lists)

    # ── 마디 · 표지

    def smalls(self, cc: str) -> list[tuple[int, int]]:
        """작은 마디의 자리 — 대조 마디를 연결 어미('고' · '며' · '므로' + 쉼표 · 공백)에서 나눈다(보조 용언 · 인용의 '고' 는 두고)."""
        out, prev = [], 0
        for m in self.R4.split.finditer(cc):
            out.append((prev, m.end()))
            prev = m.end()
        out.append((prev, len(cc)))
        return [(a, b) for a, b in out if cc[a:b].strip()]

    def cond_ending(self, sc: str, rest: str = "") -> bool:
        """(ii) 일반 조건 어미 — 어절 끝 '면' 가운데 닫힌 제외(이름씨 · 출처 꼴 · 금지 · 충분)에 들지 않는 것."""
        R = self.R4
        for m in _MYEON.finditer(sc):
            e = _core(m.group(1))
            if not e.endswith("면") or e in R.ce_exact or e.endswith(R.ce_suffix + R.ce_forms):
                continue
            if (sc[m.end():] + rest).lstrip(" ").startswith(R.ce_following):
                continue
            return True
        return False

    def exempt_marker(self, sc: str, rest: str = "") -> bool:
        """(i) 조건 표지(v3.1 목록 − '경우') — '면' 으로 끝나는 표지도 금지 · 충분의 '-면' 이면 조건이 아니다."""
        for w in self.R4.exempt_words:
            for m in _spaced(w).finditer(sc):
                if w.endswith("면") and (sc[m.end():] + rest).lstrip(" ").startswith(self.R4.ce_following):
                    continue
                return True
        return False

    def condition_marker(self, sc: str, rest: str = "") -> bool:
        return self.exempt_marker(sc, rest) or self.cond_ending(sc, rest)

    def valid_apply(self, sc: str, s: str) -> bool:
        """유효한 적용 단정 — 그 작은 마디의 적용 술어가 의문 · 유보 · 앎의 부정 · 단정의 부정에 걸리지 않았다(§4)."""
        if s.rstrip().endswith("?") or self.R4.au_question.search(sc) or self.R4.au_neg.search(sc) or self.reservation_clause(sc):
            return False
        n = _ns(sc)
        for a in self.R4.apply_ns:
            for m in re.finditer(re.escape(a), n):
                after = n[m.end():]
                neg = self.R4.not_da if a.endswith("다") else self.R4.not_hae if a.endswith("해") else ()
                if not after.startswith(neg):
                    return True
        return False

    def rnote_present(self, s: str) -> bool:
        """r_note 조건 1 — '단서' 뒤(토씨 하나 · 꾸밈말을 건넌 자리)가 허용 꼴로 시작하고 거부 꼴로 시작하지 않는다(§6)."""
        R = self.R4
        n = _ns(s)
        for m in re.finditer("단서", n):
            rest = n[m.end():]
            if rest[:1] in R.rn_particles:
                rest = rest[1:]
            moved = True
            while moved:
                moved = False
                for a in R.rn_adverbs:
                    if rest.startswith(a):
                        rest, moved = rest[len(a):], True
                        break
            if rest.startswith(R.rn_reject) or (rest.startswith("있다") and rest[2:].startswith(R.not_da)):
                continue
            if rest.startswith(R.rn_allow):
                return True
        return False

    def _gyeongu_covers(self, sc: str, m: re.Match) -> bool:
        """'경우' 의 조건절 꼴 — 촉발 낱말이 '경우' 를 꾸미는 관형절 안에 있다(§5 — 지시 1024l D)."""
        R = self.R4
        toks = [(x.start(), x.group()) for x in re.finditer(r"\S+", sc)]
        for i, (pos, tok) in enumerate(toks):
            if pos < m.end() or not _core(tok).startswith("경우"):
                continue
            if i == 0:
                return False
            prev = _core(toks[i - 1][1])
            code = ord(prev[-1]) - 0xAC00 if prev else -1
            if not (0 <= code < 11172 and code % 28 in (4, 8)) or prev in R.demonstratives:
                return False
            if i >= 2 and f"{_core(toks[i - 2][1])} {prev}" in R.demonstratives_two:
                return False
            return _DECL_MID.search(sc[m.end():pos]) is None
        return False

    def d_new(self, sc: str, rest: str = "") -> list[str]:
        """(다) 새 세부 표지 — 촉발 낱말의 작은 마디에 예외가 없고 그 작은 마디가 평서로 끝난다. 뒤 작은 마디의 술어는 잇지 않는다."""
        out = []
        for g in self.R4.d.values():
            s2 = sc
            for ex in g.exclude:
                s2 = _spaced(ex).sub(lambda mm: "□" * len(mm.group(0)), s2)
            hits = [m for t in g.triggers for m in _spaced(t).finditer(s2)]
            if not hits:
                continue
            if g.cond.search(sc) or g.know.search(sc) or g.question.search(sc) or self.cond_ending(sc, rest):
                continue
            if all(self._gyeongu_covers(sc, m) for m in hits):
                continue
            if _DECL_END.search(sc.strip()):
                out.append(g.sublabel)
        return out

    def _split_cond(self, parts: list[str], j: int) -> bool:
        """'조건 범위 단절 가능' — 앞 작은 마디에 조건 꼴(H1 의 조건 목록 · §4 의 조건 표지)이 있고 걸린 작은 마디에는 없다."""
        def has(sc: str) -> bool:
            return bool(self.R.h1_conditional.search(sc)) or self.condition_marker(sc)
        return not has(parts[j]) and any(has(p) for p in parts[:j])

    def aliases_at(self, c: str) -> list[tuple[str, str | None, int]]:
        """v3.1 `aliases` 와 같은 읽기에 자리를 더한다 — 면제 · 꼴 표지가 언급의 작은 마디를 본다(§4)."""
        out: list[tuple[str, str | None, int]] = []
        for m in V3._TABLE_NO.finditer(c):
            out.append((f"표{m['n']}", self.R.h2_table.get(m["n"]), m.start()))
        if self.R.h2_cue.search(c):
            for m in V3._CLASS_NO.finditer(c):
                out.append((f"제{m['n']}종", self.R.h2_class.get(m["n"]), m.start()))
            for rx, alias, cid in self.R.h2_literal:
                if (mm := rx.search(c)):
                    out.append((alias, cid, mm.start()))
        return out

    def _tag(self, res: Result, kind: str, tag: str) -> None:
        res.subtags.add(tag)
        res.subtags_by_kind[kind].add(tag)

    def _diag(self, res: Result, tag: str, where: str) -> None:
        res.diagnostics.append({"tag": tag, "where": where[:60]})

    # ── (라)

    def _mention(self, res: Result, s: str, pos: int, label: str, cell: str, *, aux_only: bool) -> None:
        """(라) 언급 하나 — 보조 전용 조항이면 면제를 먼저 본다. 꼴 표지는 차례 1 ~ 4(§4)."""
        R = self.R4
        a, b = next(((x, y) for x, y in contrast_spans(s) if x <= pos < y), (0, len(s)))
        cc = s[a:b]
        parts = [(x, y) for x, y in self.smalls(cc)]
        here = next(((x, y) for x, y in parts if x <= pos - a < y), parts[-1] if parts else (0, len(cc)))
        sc, rest = cc[here[0]:here[1]], cc[here[1]:]
        cond = self.condition_marker(sc, rest)
        assert_any = any(self.valid_apply(cc[x:y], s) for x, y in parts)
        assert_here = self.valid_apply(sc, s)
        if aux_only:
            if cond and not assert_any:
                self._diag(res, R.diag_exempt, sc)
                res.aux_mentions.append({"label": label, "exempt": True, "blocked": False, "mode": None})
                return
            blocked = cond and not assert_here and assert_any
            if blocked:
                self._diag(res, R.diag_blocked, sc)
        if aux_only and assert_any:
            mode = R.modes[0]
        elif assert_any:
            mode = R.modes[1]
        elif cond or R.intro.search(sc):
            mode = R.modes[2]
        else:
            mode = R.modes[3]
        if aux_only:
            res.aux_mentions.append({"label": label, "exempt": False, "blocked": blocked, "mode": mode})
        res.kinds["라"].append(label)
        res.ra_items.append({"label": label, "cell": cell, "mode": mode, "aux_only": aux_only})
        self._tag(res, "라", cell)
        if mode in R.subtag_modes:
            self._tag(res, "라", mode)

    # ── 판정

    def check(self, narrative: str, *, material: str, gold_codes, structured_codes) -> Result:
        res = Result()
        t = V3.norm(narrative)
        G = set(self.t.gold_clauses(material, gold_codes))
        aux_only = self.t.auxiliary_only(material, gold_codes)
        lists = scan_criteria(t, self.t, self.code_of_name)
        res.list_heads = len(_HEAD_ANY.findall(t))
        res.criteria = [{"anchored": c.anchored, "cid": c.cid, "licensed": c.licensed, "reason": c.reason, "head": c.head,
                         "measure": c.measure, "anchor_form": c.anchor_form} for c in lists]
        for c in lists:
            if c.measure_error:
                res.kinds["가"].append(f"{MEASURE_SUBTAG}: {c.cid} · {c.measure}")
                self._tag(res, "가", MEASURE_SUBTAG)
            if not c.licensed and c.has_mm:
                res.kinds["가"].append(f"기준 수치 오기({c.reason})")
            if material == "AL":
                res.kinds["마"].append("AL 기준 목록")
        F = self._free(t, lists)

        for s in V3.sentences(F):
            for m in V3.length_qty(s):
                res.kinds["가"].append(f"치수: {m.group(0).strip()}")
                if V3._T_GOV.search(s[:m.start()]):
                    res.kinds["다"].append(f"두께 · 축척 값: {m.group(0).strip()}")
            res.kinds["나"] += [f"판정어: {h}" for h in self.value_verdicts(s)]
            question = s.rstrip().endswith("?")
            for a, b in contrast_spans(s):
                c = s[a:b]
                sp = self.smalls(c)
                parts = [c[x:y] for x, y in sp]
                rests = [c[y:] for _, y in sp]
                multi = len(parts) > 1
                # (다) 두께 · 실치수 마디 — 대조 마디에 앎의 부정이 있을 때만 작은 마디로(C)
                split_d = multi and any(self.negated_knowledge(p) for p in parts)
                for j, u in (enumerate(parts) if split_d else [(0, c)]):
                    if V3._T_RE.search(u) and not self.negated_knowledge(u):
                        res.kinds["다"].append(f"두께 · 실치수 마디: {u[:40]}")
                        if split_d and self._split_cond(parts, j):
                            self._diag(res, self.R4.diag_split, u)
                # (나) H1 — 대조 마디에 단정의 예외가 있을 때만 작은 마디로(C)
                h1_exc = bool(self.R.h1_question.search(c) or self.R.h1_conditional.search(c) or self.reservation_clause(c) or question)
                split_h1 = multi and h1_exc
                for j, u in (enumerate(parts) if split_h1 else [(0, c)]):
                    if (p := self.h1(u, s)) is not None:
                        res.kinds["나"].append(f"{self.R.h1_sublabel}: {p}")
                        self._tag(res, "나", self.R.h1_sublabel)
                        if split_h1 and self._split_cond(parts, j):
                            self._diag(res, self.R4.diag_split, u)
                # (마) H3 — 그 예외와 H3 의 부정 · 대조가 있을 때만 작은 마디로(C)
                if material == "AL":
                    h3_exc = h1_exc or bool(self.R.h3_negation.search(c) or self.R.h3_contrast.search(c))
                    split_h3 = multi and h3_exc
                    for j, u in (enumerate(parts) if split_h3 else [(0, c)]):
                        if (w := self.h3(u, s)) is not None:
                            res.kinds["마"].append(f"{self.R.h3_sublabel}: {w}")
                            self._tag(res, "마", self.R.h3_sublabel)
                            if split_h3 and self._split_cond(parts, j):
                                self._diag(res, self.R4.diag_split, u)
                # (다) 새 세부 표지 — 늘 작은 마디(D)
                for j, u in enumerate(parts):
                    for sub in self.d_new(u, rests[j]):
                        res.kinds["다"].append(f"{sub}: {u[:40]}")
                        self._tag(res, "다", sub)
                        if multi and self._split_cond(parts, j):
                            self._diag(res, self.R4.diag_split, u)
            if V3._BAND_PICK.search(s):
                res.kinds["다"].append(f"두께 구간 고르기: {V3._BAND_PICK.search(s).group(0)}")
            if V3._PLATE.search(s):
                res.kinds["다"].append(f"판 두께 단정: {V3._PLATE.search(s).group(0)}")
            if material == "AL" and V3._CRIT_QTY.search(s):
                res.kinds["마"].append("AL 기준 꼴 수량")

        res.claimed = self.claimed_codes(t, structured_codes)
        own = set(self.t.own_clauses(material, res.claimed))
        cell_own, cell_other, cell_none = self.R.ra_cells[1], self.R.ra_cells[2], self.R.ra_cells[0]
        for s in V3.sentences(t):
            cue = bool(V3._CUE.search(s))
            for m in V3._TOKEN.finditer(s):
                tok = m["tok"]
                pre = tok.split("-")[0]
                if pre in self.t.prefixes:
                    if tok in self.t.clause_ids:
                        res.mentions.append(tok)
                        if tok not in G:
                            mine = tok in own
                            self._mention(res, s, m.start(), f"정답 밖 조항({'제 주장과는 맞음' if mine else '제 주장과도 어긋남'}): {tok}",
                                          cell_own if mine else cell_other, aux_only=tok in aux_only)
                    elif tok in self.t.row_clause:
                        par = self.t.row_clause[tok]
                        if par in G:
                            res.kinds["다"].append(f"행 식별자 = 두께 구간 선택: {tok}")
                        else:
                            self._mention(res, s, m.start(), f"정답 밖 조항의 행: {tok}", cell_own if par in own else cell_other,
                                          aux_only=False)
                    else:
                        self._mention(res, s, m.start(), f"없는 조항: {tok}", cell_none, aux_only=False)
                    if material == "AL":
                        res.kinds["마"].append(f"AL 조항 특정: {tok}")
                elif cue and any(ch.isdigit() for ch in tok.partition("-")[2]):
                    self._mention(res, s, m.start(), f"없는 조항(조항 단서): {tok}", cell_none, aux_only=False)
            for m in V3._STD.finditer(s):
                ref = re.sub(r"\s+", " ", (m["body"] + (m["des"] or ""))).strip()
                if any(ref == w or ref.startswith(w) for w in V3.STD_WHITELIST):
                    continue
                self._mention(res, s, m.start(), f"표준 인용: {ref}", cell_none, aux_only=False)
            for m in V3._STD_KO.finditer(s):
                self._mention(res, s, m.start(), f"표준 인용: {m.group(0)}", cell_none, aux_only=False)
            for a, b in contrast_spans(s):
                c = s[a:b]
                for alias, cid, at in self.aliases_at(c):
                    if cid is None:
                        res.unresolved.append(alias)
                        self._tag(res, "라", self.R.ra_separate)
                    elif cid not in G:
                        mine = cid in own
                        self._mention(res, s, a + at, f"별칭 조항({'제 주장과는 맞음' if mine else '제 주장과도 어긋남'}): {alias} → {cid}",
                                      cell_own if mine else cell_other, aux_only=cid in aux_only)
                    if material == "AL":
                        res.kinds["마"].append(f"AL 조항 특정(별칭): {alias}")

        res.required = sorted(self.t.required(material, res.claimed))
        res.found = sorted(self.reasons(t, lists, res), key=REASONS.index)
        return res

    def reasons(self, t: str, lists, res: Result | None = None) -> set[str]:
        """유보 인정 — v3.1 그대로에 r_note 조건 1 의 '단서' 를 허용 목록으로 본다(§6)."""
        out: set[str] = set()
        for s in V3.sentences(self._free(t, lists)):
            w = self.valid_withhold(s)
            got = set()
            if w and any(self.R.info["r_dim"].search(c) and self.negated_knowledge(c) for c in V3.clauses_of(s)):
                got.add("r_dim")
            if w and self.R.info["r_note"].search(s) and self.rnote_present(s):
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


def roles_of(source) -> dict[tuple[str, str], tuple[str | None, tuple[str, ...]]]:
    """2 판 표의 주 · 보조 조항 — 생성 쪽 `derive_clause_roles` → 평가 쪽 `roles_from_derived`(07e §5)."""
    from corpus.rules.limit_eval import derive_clause_roles
    from evaluation.clause_axis import roles_from_derived

    return {k: (v.main, tuple(v.auxiliary)) for k, v in roles_from_derived(derive_clause_roles(source)).items()}


def load_checker(repo: Path | None = None, *, lists_path: Path | None = None) -> Checker:
    """2 판 표(sha256 고정) · 주 · 보조 조항 · 사상표 · 닫힌 목록 v4-lists-2(sha256 고정)로 검사기를 만든다."""
    from corpus.rules import limits_loader
    from data.label_map import load_label_map

    root = Path(repo or REPO)
    limits = root / LIMITS_REL
    got = _sha(limits.read_bytes())
    if got != LIMITS_SHA256:
        raise ListsError(f"허용치 표의 sha256 {got[:12]}… 이 고정값과 다르다")
    source = limits_loader.load_limits(limits, pilot=True)
    table = Table(source.rows, roles_of(source), edition=getattr(source, "edition", "v1"))
    names = {dt.iso_code: dt.name_ko for dt in load_label_map(root / "configs" / "label_map.yaml").defect_types.values()}
    return Checker(table, names, load_lists(lists_path or root / LISTS_REL))


__all__ = ["CHECKER_ID", "KINDS", "LIMITS_REL", "LIMITS_SHA256", "LISTS_REL", "LISTS_SHA256", "LISTS_VERSION", "MEASURE_SUBTAG",
           "REASONS", "RULE_VERSION", "T_ANCHOR_V1", "T_CORRESPOND", "T_HEAD", "T_HEAD_MEASURE", "Checker", "CritList", "ListsError",
           "Result", "Rules4", "Table", "build_rules4", "contrast_spans", "load_checker", "load_lists", "roles_of", "scan_criteria"]

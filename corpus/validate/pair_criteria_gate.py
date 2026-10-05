"""페어 서술 ↔ 허용치 행의 등치 대조 — 검증 게이트 ③(부등식 방향)의 독립 경로.

파일럿 빌더의 게이트 ③은 자기참조였다(06번 §2-나). 기대값이 생성기가 만든 기준 문장을 잘라 만든
조각이었고, 검사는 그 조각이 서술에 **부분문자열로 들어 있는가** 였다. 생성기 산출을 생성기 산출에서 되찾는다.
그래서 기준 문장을 통째로 지우고 두께 구간 머리만 남긴 서술이 통과했고("5 mm 이하" 는 "25 mm 이하" 안에 있다),
재질이 AL 인데 강재 조항을 단 레코드가 통과했다(검사가 재질을 받지 않았다).

이 모듈의 계약은 넷이다.

1. 기대값은 **허용치 행에서 직접** 만든다 — 값·부등호·비례 분모·계수·상한·두께 구간 양끝. 문장화 코드
   (`corpus.rules.clause_text`)와 생성기(`corpus.generate.make_pairs_pilot`)를 **import 하지 않는다**(배선 시험이 강제한다).
2. 서술은 **파싱해서 구조로 바꾼 뒤 등치**로 대조한다. 부분문자열을 찾지 않는다. 문법 밖의 글자가 하나라도
   있으면 파싱 실패이고 그것도 폐기 사유다.
3. 기대 조항 집합은 **(재질, 결함 코드)** 로 계산해 레코드의 `clauses` 와 등치로 비교한다.
4. 모르는 값은 기본값으로 떨어뜨리지 않고 예외를 낸다(부등호·비례 분모·단위·한계 규칙).

두께 구간은 표의 부호화 그대로 **반열림 `[t_min, t_max)`**(0.01 그리드)로 맞춘다. 서술의 "X mm 초과" 는
`X + 0.01`, "X mm 이상" 은 `X`, "Y mm 이하" 는 `Y + 0.01`, "Y mm 미만" 은 `Y` 로 읽는다. 그래서 하한이
`10.00` 인 행을 "10 mm 초과" 로 쓴 서술은 어긋난다(초과는 `10.01` 이다).

순수 함수만 둔다(I/O·전역 상태·난수 없음).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import pairwise

__all__ = [
    "CLOSE_DIMENSIONAL",
    "CLOSE_SIZE_INDEPENDENT",
    "NOTE_UNEXPRESSED",
    "NO_CLAUSE_MATERIAL",
    "Criterion",
    "PairRules",
    "ParsedText",
    "TextParseError",
    "check_text",
    "criterion_of_row",
    "parse_target_text",
]

Q = Decimal("0.01")

#: 서술의 고정 문장. 생성기도 같은 문장을 쓰지만 **여기서 가져가지 않는다** — 두 곳이 어긋나면 게이트가 폐기로 드러낸다.
OBS_HEAD = "방사선투과 영상에서 "
OBS_TAIL = "가 관찰된다."
CRIT_HEAD = "이 조항의 기준은 다음과 같다. "
NOTE_UNEXPRESSED = ("이 조항에는 기계 표현으로 옮기지 못한 단서가 있어"
                    " 위 기준만으로 합부를 결정하지 않는다.")
NO_CLAUSE_MATERIAL = ("이 재질에 적용할 허용치 조항이 허용치 표에 없어"
                      " 적용 조항을 특정하지 않는다.")
NO_CLAUSE_CODE_TAIL = "에 적용할 허용치 조항이 허용치 표에 없어 그 결함의 적용 조항을 특정하지 않는다."
CLOSE_DIMENSIONAL = "모재 두께와 화소당 실치수 정보가 없어 치수 기준의 합부 판정은 내리지 않는다."
CLOSE_SIZE_INDEPENDENT = ("크기와 무관한 기준은 치수 정보 없이도 적용되지만,"
                          " 이 자료는 판정 축을 조항 검색과 기준 서술로 한정해 합부 판정을 싣지 않는다.")
NONE_PERMITTED_KO = "크기와 무관하게 허용하지 않는다"

_OPS = {"이하": "le", "미만": "lt"}
_BASES = {"모재 두께": "t", "용접부 공칭 두께": "s", "목두께": "a"}
_UNITS = {"mm": "mm", "%": "percent"}
#: 서술에 쓰이는 수. **ASCII 숫자만**이고 선행 0·부호를 받지 않는다 — 생성기가 내지 않는 표기는 문법 밖이다.
#: `\d` 로 두면 아랍·데바나가리·전각 숫자가 통과하고, `Decimal()` 이 그것을 읽어 값까지 맞다고 본다(검토 F5).
_DEC = r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"


class TextParseError(ValueError):
    """서술이 문법 밖이다. 어디서 막혔는지를 메시지에 싣는다."""


@dataclass(frozen=True)
class Criterion:
    """허용치 행 하나의 기준. 서술에서 읽은 것과 행에서 만든 것을 **같은 꼴**로 맞춰 `==` 로 비교한다.

    `unit` 은 서술에 단위가 적히는 곳에서만 값이 있다 — 상수 한계와 비례 한계의 상한. 나머지는 None.
    """

    t_min: Decimal
    t_max: Decimal | None
    rule: str
    op: str | None = None
    value: Decimal | None = None
    factor: Decimal | None = None
    cap: Decimal | None = None
    basis: str | None = None
    unit: str | None = None


@dataclass(frozen=True)
class ParsedText:
    observed: tuple[tuple[str, str, int], ...]                 # (명칭, 코드, 개수) — 등장 순서
    blocks: tuple[tuple[str, str, tuple[Criterion, ...], bool], ...]   # (명칭, 조항, 기준들, 부기 유무)
    uncited_material: bool                                     # 재질 단위 미특정 문장
    uncited_codes: tuple[str, ...]                             # 코드 단위 미특정 문장이 든 결함 코드
    close_dimensional: bool
    close_size_independent: bool
    criteria_spans: tuple[tuple[int, int], ...]                # 기준 문장의 [시작, 끝) — 판정 함의 검사에서 뺀다


# --------------------------------------------------------------- 허용치 행 → 구조

def _v(x):
    return getattr(x, "value", x)


def _dec(x) -> Decimal | None:
    if x is None or x == "":
        return None
    return x if isinstance(x, Decimal) else Decimal(str(x))


def criterion_of_row(row) -> Criterion:
    """행 → 기준 구조. 문장을 거치지 않는다."""
    rule = _v(row.limit_rule)
    t_min = _dec(row.thickness_min)
    if t_min is None:
        raise ValueError(f"두께 하한이 비었다: {row.rule_id}")
    t_max = _dec(row.thickness_max)
    if rule == "none_permitted":
        return Criterion(t_min, t_max, rule)
    op = _v(row.limit_op)
    if op not in _OPS.values():
        raise ValueError(f"모르는 부등호 {op!r}: {row.rule_id}")
    unit = _v(row.unit)
    if unit not in _UNITS.values():
        raise ValueError(f"모르는 단위 {unit!r}: {row.rule_id}")
    if rule == "const":
        return Criterion(t_min, t_max, rule, op=op, value=_dec(row.limit_value), unit=unit)
    basis = _v(row.ratio_basis)
    if basis not in _BASES.values():
        raise ValueError(f"모르는 비례 분모 {basis!r}: {row.rule_id}")
    if rule == "prop_t":
        return Criterion(t_min, t_max, rule, op=op, factor=_dec(row.limit_factor), basis=basis)
    if rule == "prop_t_cap":
        return Criterion(t_min, t_max, rule, op=op, factor=_dec(row.limit_factor),
                         cap=_dec(row.limit_cap), basis=basis, unit=unit)
    raise ValueError(f"모르는 한계 규칙 {rule!r}: {row.rule_id}")


class PairRules:
    """(재질, 결함 코드) → 적용 행. 페어 축은 RT 다.

    행 선택: `scope=active` · 검사 방식 RT 또는 ALL · 재질이 같거나 ALL. `rule_id` 문자열 오름차순.
    정본이 아닌 행(`canonical=False`)이나 품질 체계가 걸린 행이 선택되면 **멈춘다** — 그 축의 정책이
    정해지지 않았다(06번 §2-다). 같은 (재질, 코드)가 두 조항에 걸려도 멈춘다.
    """

    #: 아는 재질. 표의 `material` 어휘와 같다. 모르는 값은 "덮는 행이 없는 재질" 로 조용히 읽히면 안 된다(검토 F6).
    KNOWN_MATERIALS = frozenset({"ST", "AL"})

    def __init__(self, rows: Iterable, names: Mapping[str, str]):
        self.names = dict(names)
        self._rows = [r for r in rows
                      if _v(getattr(r, "scope", "active")) == "active"
                      and _v(r.inspection_method) in ("RT", "ALL")]
        for r in self._rows:
            if getattr(r, "canonical", True) is False:
                raise ValueError(f"정본이 아닌 행이 페어 축에 걸린다: {r.rule_id} — 정책 미정")
            if _v(getattr(r, "quality_scheme", "none")) not in ("none", None):
                raise ValueError(f"품질 체계가 걸린 행이 페어 축에 걸린다: {r.rule_id} — 정책 미정")
        self.valid_clause_ids = frozenset(r.clause_id for r in self._rows)

    def covers_material(self, material: str) -> bool:
        return any(_v(r.material) in (material, "ALL") for r in self._rows)

    def rows_for(self, material: str, code: str) -> tuple:
        rows = sorted((r for r in self._rows
                       if _v(r.material) in (material, "ALL") and str(r.defect_code) == str(code)),
                      key=lambda r: r.rule_id)
        if len({r.clause_id for r in rows}) > 1:
            raise ValueError(f"결함 {code}({material}) 가 복수 조항에 걸린다: "
                             f"{sorted({r.clause_id for r in rows})}")
        # 두께 축에서 겹치는 행이 함께 걸리면 한 두께에 기준이 둘이다. 서술은 둘 다 나열하고 게이트는 그 나열을
        # 그대로 기대값으로 삼아, 모순된 기준이 폐기 없이 나간다(검토 F1). 정본 행 선택기도 이런 표에서 멈춘다.
        INF = Decimal("Infinity")              # 상한 공란 = +∞. None 을 섞어 정렬하면 비교가 죽는다
        bands = sorted((_dec(r.thickness_min) or Decimal("0.00"), _dec(r.thickness_max) or INF, r.rule_id)
                       for r in rows)
        for (lo1, hi1, id1), (lo2, _hi2, id2) in pairwise(bands):
            if lo2 < hi1:
                raise ValueError(f"결함 {code}({material}) 의 두께 구간이 겹친다: {id1} [{lo1}, {hi1}) · {id2} [{lo2}, …) "
                                 "— 한 두께에 기준이 둘이면 조항 서술이 모순된다")
        return tuple(rows)

    def clause_of(self, material: str, code: str) -> str | None:
        rows = self.rows_for(material, code)
        return rows[0].clause_id if rows else None

    def expected_clauses(self, material: str, codes: Iterable[str]) -> list[str]:
        return sorted({c for c in (self.clause_of(material, code) for code in set(codes)) if c})

    def expected_candidate_rules(self, material: str, codes: Iterable[str]) -> list[dict]:
        out = []
        for code in set(codes):
            for r in self.rows_for(material, code):
                out.append({"rule_id": r.rule_id, "clause_id": r.clause_id,
                            "defect_code": str(r.defect_code),
                            "thickness_min": _grid(_dec(r.thickness_min)),
                            "thickness_max": _grid(_dec(r.thickness_max))})
        return sorted(out, key=lambda d: d["rule_id"])

    def artifact_basis(self, material: str | None, codes: Iterable[str]) -> str:
        """아티팩트 검출기에 줄 **정당한 토큰 모음** — 조항 id 와 허용치 행의 수치.

        검출기는 `basis` 에 문자 그대로 있는 표기를 아티팩트로 세지 않는다. 표에서 온 값(`4.00` · `10.01`)과
        조항 id(`KRA27-T3.10`)가 기준에 없으면, 표를 개정하는 순간 그 조항의 페어가 전량 폐기된다(검토 F4).
        """
        toks: list[str] = []
        for code in list(codes):
            for r in self.rows_for(material, code) if material else ():
                toks.append(str(r.clause_id))
                for f in ("limit_value", "limit_factor", "limit_cap", "thickness_min", "thickness_max"):
                    v = _dec(getattr(r, f, None))
                    if v is not None:
                        toks += [str(v), str(v.normalize()), _grid(v) or ""]
        return " ".join(dict.fromkeys(t for t in toks if t))

    def has_note(self, material: str, code: str) -> bool:
        """부기 조건 — 값 한계(`none_permitted` 가 아닌 행)에 `note` 가 있는가.

        전량 불허 행에는 붙이지 않는다. 부기는 "위 기준만으로 합부를 결정하지 않는다" 인데, 전량 불허에는
        단서가 걸릴 값이 없고 그 행의 `note` 는 전사 메모다 — 붙이면 서술이 스스로 부딪힌다(06번 §3-사).
        """
        return any((getattr(r, "note", None) or "").strip()
                   for r in self.rows_for(material, code) if _v(r.limit_rule) != "none_permitted")


def _grid(d: Decimal | None) -> str | None:
    return None if d is None else str(d.quantize(Q))


# --------------------------------------------------------------- 서술 → 구조

class _Cursor:
    def __init__(self, text: str):
        self.text, self.i = text, 0

    def eat(self, literal: str) -> bool:
        if self.text.startswith(literal, self.i):
            self.i += len(literal)
            return True
        return False

    def need(self, literal: str) -> None:
        if not self.eat(literal):
            raise TextParseError(f"{self.i}자: {literal!r} 자리에 {self.text[self.i:self.i + 24]!r}")

    def match(self, pattern: re.Pattern[str]) -> re.Match[str] | None:
        m = pattern.match(self.text, self.i)
        if m:
            self.i = m.end()
        return m

    def done(self) -> bool:
        return self.i == len(self.text)


def _num(s: str) -> Decimal:
    try:
        return Decimal(s)
    except InvalidOperation as e:              # pragma: no cover — 정규식이 먼저 거른다
        raise TextParseError(f"수가 아니다: {s!r}") from e


_BAND_ALL = "모든 두께"
_LOWER = re.compile(rf"({_DEC}) mm (초과|이상)")
_UPPER = re.compile(rf"({_DEC}) mm (이하|미만)")
_CONST = re.compile(rf"({_DEC}) (mm|%) (이하|미만)")
_PROP = re.compile(rf"({'|'.join(map(re.escape, _BASES))})의 ({_DEC}) 배 (이하|미만)")
_CAP = re.compile(rf"이고 최대 ({_DEC}) (mm|%)")
_COUNT = re.compile(r"\(ISO 6520-1 코드 ([0-9A-Za-z]+)\) ([1-9][0-9]*)개")
_UNCITED_CODE = re.compile(r"\(ISO 6520-1 코드 ([0-9A-Za-z]+)\)")


def _parse_band(c: _Cursor) -> Criterion:
    if c.eat(_BAND_ALL):
        t_min, t_max = Decimal("0.00"), None
    else:
        c.need("두께 ")
        lo = c.match(_LOWER)
        t_min = Decimal("0.00")
        sep = False
        if lo:
            if _num(lo.group(1)) == 0:
                # 하한 0 은 제약이 없다는 뜻이고 생성기는 아예 쓰지 않는다. 받아 주면 "모든 두께" 와 같은 값이 되어
                # 서로 다른 두 표기가 같은 구간으로 읽힌다(검토 F5).
                raise TextParseError(f"{c.i}자: 하한 0 은 쓰지 않는다 — 제약이 없으면 상한만 적는다")
            t_min = _num(lo.group(1)) + (Q if lo.group(2) == "초과" else 0)
            sep = c.eat(" ")
        hi = c.match(_UPPER)
        t_max = None
        if hi:
            t_max = _num(hi.group(1)) + (Q if hi.group(2) == "이하" else 0)
        if not lo and not hi:
            raise TextParseError(f"{c.i}자: 두께 구간이 비었다")
        if lo and bool(hi) != sep:
            raise TextParseError(f"{c.i}자: 하한과 상한 사이의 공백이 규약과 다르다")
        if t_max is not None and t_max <= t_min:
            raise TextParseError(f"{c.i}자: 두께 구간이 비었다 — [{t_min}, {t_max})")
    c.need(": ")
    if c.eat(NONE_PERMITTED_KO):
        return Criterion(t_min, t_max, "none_permitted")
    m = c.match(_CONST)
    if m:
        return Criterion(t_min, t_max, "const", op=_OPS[m.group(3)], value=_num(m.group(1)),
                         unit=_UNITS[m.group(2)])
    m = c.match(_PROP)
    if not m:
        raise TextParseError(f"{c.i}자: 한계 서술을 읽지 못했다 — {c.text[c.i:c.i + 24]!r}")
    cap = c.match(_CAP)
    if cap:
        return Criterion(t_min, t_max, "prop_t_cap", op=_OPS[m.group(3)], factor=_num(m.group(2)),
                         cap=_num(cap.group(1)), basis=_BASES[m.group(1)], unit=_UNITS[cap.group(2)])
    return Criterion(t_min, t_max, "prop_t", op=_OPS[m.group(3)], factor=_num(m.group(2)),
                     basis=_BASES[m.group(1)])


def _eat_name(c: _Cursor, names: Sequence[str]) -> str | None:
    for n in names:                            # 긴 이름부터 — 한 이름이 다른 이름의 머리일 수 있다
        if c.eat(n):
            return n
    return None


def parse_target_text(text: str, names: Mapping[str, str]) -> ParsedText:
    """결함 페어의 서술 전체를 읽는다. **끝까지 문법대로여야 한다** — 남는 글자가 있으면 실패다."""
    by_len = sorted(set(names.values()) | {"결함"}, key=len, reverse=True)
    c = _Cursor(text)
    c.need(OBS_HEAD)
    observed = []
    while True:
        name = _eat_name(c, by_len)
        if name is None:
            raise TextParseError(f"{c.i}자: 결함 명칭을 읽지 못했다 — {text[c.i:c.i + 24]!r}")
        m = c.match(_COUNT)
        if not m:
            raise TextParseError(f"{c.i}자: 코드·개수를 읽지 못했다")
        observed.append((name, m.group(1), int(m.group(2))))
        if not c.eat(", "):
            break
    c.need(OBS_TAIL)

    blocks, spans = [], []
    while True:
        mark = c.i
        if not c.eat(" "):
            break
        name = _eat_name(c, by_len)
        if name is None or not c.eat("에 적용되는 조항은 "):
            c.i = mark
            break
        m = c.match(re.compile(r"(\S+) 이다\."))
        if not m:
            raise TextParseError(f"{c.i}자: 조항 식별자를 읽지 못했다")
        c.need(" ")
        start = c.i
        c.need(CRIT_HEAD)
        crit = [_parse_band(c)]
        while c.eat(" / "):
            crit.append(_parse_band(c))
        c.need(".")
        spans.append((start, c.i))
        note = c.eat(" " + NOTE_UNEXPRESSED)
        blocks.append((name, m.group(1), tuple(crit), note))

    uncited_material = c.eat(" " + NO_CLAUSE_MATERIAL)
    uncited: list[str] = []
    if not uncited_material:
        mark = c.i
        if c.eat(" "):
            while True:
                if _eat_name(c, by_len) is None:
                    break
                m = c.match(_UNCITED_CODE)      # 명칭이 같은 코드가 있으므로 코드로 읽는다(검토 F7)
                if m is None:
                    break
                uncited.append(m.group(1))
                if not c.eat(", "):
                    break
            if not (uncited and c.eat(NO_CLAUSE_CODE_TAIL)):
                c.i, uncited = mark, []
    close_dim = c.eat(" " + CLOSE_DIMENSIONAL)
    close_np = c.eat(" " + CLOSE_SIZE_INDEPENDENT)
    if not c.done():
        raise TextParseError(f"{c.i}자: 문법 밖의 글자가 남았다 — {text[c.i:c.i + 32]!r}")
    return ParsedText(tuple(observed), tuple(blocks), uncited_material, tuple(uncited),
                      close_dim, close_np, tuple(spans))


# --------------------------------------------------------------- 대조

def _first_seen(defects: Sequence[Mapping]) -> list[str]:
    seen: list[str] = []
    for d in defects:
        if str(d["type"]) not in seen:
            seen.append(str(d["type"]))
    return seen


def check_text(rec: Mapping, rules: PairRules) -> tuple[list[str], ParsedText | None]:
    """결함 페어 하나의 폐기 사유와 파싱 결과. 사유가 비면 서술·골격·허용치 행이 서로 등치다.

    사유: `material_missing` · `material_unknown` · `clause_set_mismatch` · `candidate_rules_mismatch` · `uncited_codes_mismatch` ·
    `text_unparsable` · `observation_mismatch` · `clause_text_mismatch` · `criterion_mismatch` ·
    `note_mismatch` · `uncited_text_mismatch` · `closing_mismatch`.
    """
    bad: list[str] = []
    sk = rec.get("skeleton") or {}
    defects = sk.get("defects") or []
    material = rec.get("material")
    if not material:
        return ["material_missing"], None
    if material not in rules.KNOWN_MATERIALS:
        # 모르는 재질을 "덮는 행이 없는 재질" 로 읽으면 표기 오류가 미특정 서술로 조용히 흘러간다(검토 F6).
        return ["material_unknown"], None
    codes = _first_seen(defects)
    names = rules.names

    # (3) 조항 집합은 (재질, 코드)로 다시 계산해 등치로 본다 — 합집합 소속 여부가 아니다
    if list(sk.get("clauses") or []) != rules.expected_clauses(material, codes):
        bad.append("clause_set_mismatch")
    if list(sk.get("candidate_rules", [])) != rules.expected_candidate_rules(material, codes):
        bad.append("candidate_rules_mismatch")      # 키가 아예 없어도 어긋난 것이다
    cited = [c for c in codes if rules.rows_for(material, c)]
    uncited = [c for c in codes if c not in cited]
    reason = "code_not_covered" if rules.covers_material(material) else "material_not_covered"
    if list(sk.get("uncited_codes", [])) != [{"code": c, "reason": reason} for c in uncited]:
        bad.append("uncited_codes_mismatch")

    try:
        p = parse_target_text(rec.get("target_text", ""), names)
    except TextParseError:
        return sorted(set(bad + ["text_unparsable"])), None

    want_obs = tuple((names.get(c, "결함"), c, sum(1 for d in defects if str(d["type"]) == c)) for c in codes)
    if p.observed != want_obs:
        bad.append("observation_mismatch")
    if [(n, cid) for n, cid, _, _ in p.blocks] != [(names.get(c, "결함"), rules.clause_of(material, c)) for c in cited]:
        bad.append("clause_text_mismatch")
    else:
        for (_, _, crit, note), code in zip(p.blocks, cited):
            if crit != tuple(criterion_of_row(r) for r in rules.rows_for(material, code)):
                bad.append("criterion_mismatch")
            if note != rules.has_note(material, code):
                bad.append("note_mismatch")
    if not rules.covers_material(material):
        ok = p.uncited_material == bool(uncited) and not p.uncited_codes
    else:
        ok = not p.uncited_material and list(p.uncited_codes) == uncited
    if not ok:
        bad.append("uncited_text_mismatch")
    rule_kinds = {_v(r.limit_rule) for c in cited for r in rules.rows_for(material, c)}
    want_dim = bool(uncited) or bool(rule_kinds - {"none_permitted"})
    want_np = "none_permitted" in rule_kinds
    if (p.close_dimensional, p.close_size_independent) != (want_dim, want_np):
        bad.append("closing_mismatch")
    return sorted(set(bad)), p

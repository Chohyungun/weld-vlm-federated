"""판정 v7 의 결정적 grounded — 평가 쪽 참조 구현(정본). 미니스펙 16번 7판 §13(보강 1 ~ 6).

결함 행: 글에서 사상표 이름 · 개수(아라비아 · 고유어) · 9 구역(어순 · 쪽 · 하이픈 · 띄어쓰기 · 칸 · 띠의 칸 · 영상 가운데 · 중앙)을 뽑아
잠긴 필드와 '이름 × 구역 × 수' 다중집합으로 맞댄다. 언급은 모두 모아 두고 같은 항목의 다른 값은 모순이다(뒤가 앞을 덮지 않는다).
이름 · 영역별 부재 주장도 사실로 남겨 잠긴 입력과 양립해야 한다. 정상 행 · 제한 문장: 생성 계약(v6 생성 지시의 고정 문장)과 정확
일치, 그리고 정상 행의 음성 패턴(`normal_codes`). 주석 밖 시각 속성의 긍정 서술. 모델은 부르지 않는다.

정식 구현은 외부 검토 도구가 이 파일을 **바이트 그대로** `corpus/production/judge_grounded_v7.py` 로 옮긴다. 두 사본은
`evaluation/vectors/grounded_v7_vectors.jsonl` 의 같은 입력에서 같은 출력을 내야 한다 — 하니스
`scripts/probe/check_grounded_v7_equivalence.py`. 벡터는 독립 증거가 아니다(두 구현의 동치만 고정한다).

공개 함수 넷(하니스의 계약): `grounded_codes(task, text)` · `contract_codes(task, text)` · `attribute_codes(task, text)` ·
`normal_codes(task, text)` — 모두 사유 코드의 정렬된 목록을 돌려주고, 빈 목록이 통과다. `grounded_codes` 는 결함 행에서만,
`normal_codes` 는 정상 행에서만 뜻이 있다(그 밖은 빈 목록).
"""

from __future__ import annotations

import re

from corpus.production import generation_policy as G
from corpus.production import judge_policy as P
from corpus.production.prepare import _label_names

NAMES = tuple(sorted(set(_label_names().values()), key=lambda n: (-len(n), n)))
"""긴 이름 먼저, 같은 길이는 이름순 — **결정적**이다. 길이만으로 정렬하면 같은 길이 이름(용입불량 · 융합불량, 기공 · 균열)의 순서가
집합 순회, 곧 PYTHONHASHSEED 를 따라 정책 지문이 프로세스마다 갈린다(7판 보강 5, 2026-10-13)."""
KO = {"한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8,
      "아홉": 9, "열": 10}
ROW = {"위": "위", "위쪽": "위", "아래": "아래", "아래쪽": "아래", "가운데": "가운데"}
COL = {"왼쪽": "왼쪽", "왼편": "왼쪽", "오른쪽": "오른쪽", "오른편": "오른쪽", "가운데": "가운데"}
ALL = {(r, c) for r in ("위", "가운데", "아래") for c in ("왼쪽", "가운데", "오른쪽")}
_R = r"(?<![가-힣])(위쪽|아래쪽|위|아래|가운데)"
"""행 낱말 — 앞에 한글이 붙으면 낱말이 아니다('부위 왼쪽' 의 '위')."""
_C = r"(왼쪽|오른쪽|왼편|오른편|가운데)"
_ROWB = r"(?<![가-힣])(위쪽|아래쪽|가운데)"
_COLB = r"(?<![가-힣])(왼쪽|오른쪽|가운데)"
NEG = re.compile(r"(?:없|않|아니)")
SUMMARY = re.compile(r"(?:모두|총|합쳐|합계)")
ATTR = re.compile(r"(?<![가-힣])(?:크고|작고|둥근|둥글|길쭉|가늘|굵|밝은|어두운|선명|날카로|뾰족|짧은|넓은|좁은|긴\s|큰\s|작은\s)")
"""시각 속성 어휘 — 낱말 앞에 한글이 없을 때만('생긴 · 시작은' 은 속성이 아니다, 7판 보강 6)."""
ABSENCE_EXEMPT = re.compile(r"(?:다른|나머지|외에|밖에|제외)")
"""부재문의 예외 — '위쪽 왼쪽 외에는 결함이 없다' 는 잠긴 칸과 모순이 아니다."""
ATTR_OK = re.compile(r"(?:알 수 없|확인할 수 없|모른|주어지지 않|판단하지 않)")


def zone_key(r: str, c: str) -> tuple[str, str]:
    return ROW[r], COL[c]


def zone_spans(text: str) -> list[tuple[int, int, tuple[str, str]]]:
    """두 낱말 구역(어순 무관 · 쪽 · 하이픈 · 띄어쓰기), '띠의 칸'(아래쪽 띠의 가운데 칸), '영상 가운데 · 중앙' 의 자리.
    겹치면 먼저 찾은 것을 쓴다 — 띠의 칸 → 행 열 → 열 행 → 중앙 순."""
    spans: list[tuple[int, int, tuple[str, str]]] = []

    def add(m, z):
        if not any(s < m.end() and m.start() < e for s, e, _ in spans):
            spans.append((m.start(), m.end(), z))
    for m in re.finditer(_ROWB + r"\s*(?:가로\s*)?띠의\s*" + _C + r"\s*(?:구역|칸)", text):
        add(m, zone_key(m.group(1), m.group(2)))
    for m in re.finditer(_COLB + r"\s*세로\s*띠의\s*(위쪽|아래쪽|가운데)\s*(?:구역|칸)", text):
        add(m, zone_key(m.group(2), m.group(1)))
    for m in re.finditer(_R + r"[ -]?" + _C + r"(?!\s*(?:세로\s*)?띠)", text):
        add(m, zone_key(m.group(1), m.group(2)))
    for m in re.finditer(r"(?<![가-힣])(왼쪽|오른쪽|왼편|오른편|가운데)[ -]?(위쪽|아래쪽|위|아래|가운데)(?!치)", text):
        add(m, zone_key(m.group(2), m.group(1)))   # 열이 앞서는 꼴 — '왼쪽 가운데' 는 (가운데, 왼쪽), '가운데 위쪽' 은 (위, 가운데). '왼쪽 위치' 는 구역이 아니다
    for m in re.finditer(r"(?:영상 가운데|중앙|가운데 구역|가운데 칸)", text):
        add(m, ("가운데", "가운데"))
    return sorted(spans)


def zones_in(text: str) -> tuple[set, str]:
    return {z for _, _, z in zone_spans(text)}, text


def regions_in(text: str) -> set:
    """띠 · 높이 · 열 · 구역 · 가장자리 같은 넓은 영역(칸의 집합). 두 낱말 구역에 든 낱말은 띠로 읽지 않는다 —
    '가운데 오른쪽 칸' 의 '오른쪽 칸' 은 세로 띠가 아니라 구역의 꼬리다(7판 보강 6)."""
    zs = [(s, e) for s, e, _ in zone_spans(text)]

    def free(m) -> bool:
        return not any(s < m.end() and m.start() < e for s, e in zs)
    out = set()
    # 띠 · 높이 · 열은 나열될 수 있다 — '위쪽과 가운데 높이' 는 두 행의 합이다
    for m in re.finditer(r"((?:(?<![가-힣])(?:위쪽|아래쪽|가운데)\s*(?:과|와|및|,)\s*)*)" + _ROWB + r"\s*(?:가로\s*)?(?:띠|높이|줄)", text):
        if free(m):
            for r in re.findall(r"위쪽|아래쪽|가운데", m.group(1)) + [m.group(2)]:
                out |= {z for z in ALL if z[0] == ROW[r]}
    for m in re.finditer(r"((?:(?<![가-힣])(?:왼쪽|오른쪽|가운데)\s*(?:과|와|및|,)\s*)*)" + _COLB + r"\s*(?:세로\s*띠|열)", text):
        if free(m):
            for c in re.findall(r"왼쪽|오른쪽|가운데", m.group(1)) + [m.group(2)]:
                out |= {z for z in ALL if z[1] == COL[c]}
    for m in re.finditer(r"(?<![가-힣])(왼쪽|오른쪽)\s*(?:구역|칸)", text):
        if free(m):
            out |= {z for z in ALL if z[1] == COL[m.group(1)]}
    for m in re.finditer(r"(?<![가-힣])(위쪽|아래쪽)(?:\s*(?:구역|칸))?(?:에는|에도|은|는)", text):
        if free(m):
            out |= {z for z in ALL if z[0] == ROW[m.group(1)]}
    if "가장자리" in text:
        out |= ALL - {("가운데", "가운데")}
    return out


def locked_of(task: dict) -> tuple[dict, dict]:
    g = P.grouped_locked_fields(task)
    counts = dict(g["defect_counts_by_name"])
    zones: dict[str, set] = {}
    zc: dict[tuple, int] = {}
    for zname, by in g["defects_by_zone"].items():
        parts = zname.split()
        z = ("가운데", "가운데") if parts == ["가운데"] else (parts[0], parts[1])
        for n, k in by.items():
            zones.setdefault(n, set()).add(z)
            zc[(n, z)] = zc.get((n, z), 0) + int(k)
    return counts, zones, zc


COUNT_AFTER_NAME = (r"(?:은|는|이|가)?\s*(모두\s*|총\s*)?(?:(\d+)\s*개|(" + "|".join(KO) + r")\s*개|(하나|둘|셋|넷)(?=가|이|만|\s|$))")
_UNIT_NOT_DEFECT = r"(?!\s*(?:의\s*)?(?:구역|칸|영역|곳|위치|띠|줄|부분|열|높이))"
"""'두 개의 구역' · '세 칸' — 자리를 센 수는 결함의 수가 아니다."""
BARE_COUNT = re.compile(r"(?<![가-힣\d])(?:(\d+)|(" + "|".join(KO) + r"))\s*개(?!수)" + _UNIT_NOT_DEFECT)


_KO_ALT = "|".join(KO)
GRAND = re.compile(r"결함(?:은|는|이|가|들은|들이)?\s*(?:모두\s*|총\s*|합쳐\s*)?(?:(\d+)|(" + _KO_ALT + r"))\s*개(?!수)" + _UNIT_NOT_DEFECT
                   + r"|(?<![가-힣\d])(?:(\d+)|(" + _KO_ALT + r"))\s*개의\s*결함")


SUBSET = re.compile(r"(?:그중|그 중|그 가운데|이 가운데|이 중|이 가운데)\s*$")


def add_zone_counts(n: str, ks: list, zl: list, zone_st: dict, group_st: dict, total_st: dict) -> None:
    """한 문장에서 한 이름의 수들과 구역들을 언급으로 쌓는다. 구역 하나에 수 하나, 수 하나가 구역 개수와 같으면 하나씩(적힌 구역마다
    적어도 하나라 하나로 정해진다), 수와 구역이 같은 개수면 차례대로. 그 밖(수 생략 꼴 등)은 '그 구역 묶음의 합' 언급이고,
    구역이 없으면 이름의 총수 언급이다."""
    if not zl:
        total_st.setdefault(n, []).append(sum(ks))
    elif len(ks) == 1 and len(zl) == 1:
        zone_st.setdefault((n, zl[0]), []).append(ks[0])
    elif len(ks) == 1 and ks[0] == len(zl):
        for z in zl:
            zone_st.setdefault((n, z), []).append(1)
    elif len(ks) == len(zl):
        for z, k in zip(zl, ks):
            zone_st.setdefault((n, z), []).append(k)
    else:
        group_st.setdefault((n, frozenset(zl)), []).append(sum(ks))


_CLAUSE = re.compile(r",\s*|(?<=[가-힣])(?:고|며|지만|으나|는데|면서)\s+")


def _clauses(s: str) -> list[str]:
    """문장을 쉼표 · 연결 어미에서 나눈다 — 부재 주장의 범위(이름 · 구역)는 절 안이다."""
    return [c for c in _CLAUSE.split(s) if c and c.strip()]


def grand_counts(s: str) -> list[tuple[tuple[int, int], int]]:
    """'결함 N개' · '결함은 모두 N개' · 'N개의 결함' — 이름 없이 결함 전체의 수를 말하는 언급."""
    out = []
    for m in GRAND.finditer(s):
        g = next(x for x in m.groups() if x)
        out.append((m.span(), int(g) if g.isdigit() else KO[g]))
    return out


def count_of(m: re.Match) -> tuple[int, bool]:
    g = m.group(2) or m.group(3) or m.group(4)
    return (int(g) if g.isdigit() else KO[g]), bool(m.group(1))


def defect_grounded(task: dict, text: str) -> list[str]:
    counts, lzones, lzc = locked_of(task)
    locked_all = set().union(*lzones.values()) if lzones else set()
    sents = [s.rstrip(".?!") for s in P.sentences(text)]
    said: set = set()
    mzones: dict[str, set] = {}
    # 언급은 문장 순서와 무관하게 모두 모아 둔다 — 뒤 문장이 앞 문장을 덮지 않는다. 같은 항목의 같은 값은 되풀이, 다른 값은 모순
    zone_st: dict[tuple, list] = {}                                 # (이름, 구역) → 수들
    group_st: dict[tuple, list] = {}                                # (이름, 구역 묶음) → 수들 — 짝지을 수 없는 꼴 · 수 생략 꼴
    total_st: dict[str, list] = {}                                  # 이름 → 총수들(요약 · 구역 없는 수)
    grand_st: list = []                                             # 결함 전체의 수들
    out = []
    for s in sents:
        names_here = [n for n in NAMES if n in s]
        z, rest = zones_in(s)
        grands = grand_counts(s)
        grand_st += [k for _, k in grands]
        if not names_here:
            bare_ks = [(m, int(g) if (g := m.group(1) or m.group(2)).isdigit() else KO[g]) for m in BARE_COUNT.finditer(s)
                       if not any(a <= m.start() < b for (a, b), _ in grands)]
            if bare_ks:
                zl = list(dict.fromkeys(zz for _, _, zz in zone_spans(s)))
                ks = [k for _, k in bare_ks]
                ant = sorted(said)
                if zl and len(ant) == 1:                            # '위쪽 왼쪽에 2개, 아래쪽 오른쪽에 1개가 있다' — 앞의 한 이름의 구역별 수
                    mzones.setdefault(ant[0], set()).update(zl)
                    add_zone_counts(ant[0], ks, zl, zone_st, group_st, total_st)
                elif zl:
                    out.append("unattributed_count")                # 구역별 수인데 앞 이름이 없거나 여럿 — 누구의 수인지 정해지지 않는다
                elif not SUBSET.search(s[:bare_ks[0][0].start()]):  # '그중 1개는' 같은 부분 수는 읽지 않는다
                    if len(ant) == 1:
                        total_st.setdefault(ant[0], []).append(sum(ks))
                    else:
                        grand_st.append(sum(ks))                     # 이름이 여럿(또는 없음)이면 결함 전체의 수
            reg = regions_in(s)
            if NEG.search(s) and "결함" in s:                      # 부재문 — 그 영역에 잠긴 결함이 없어야 한다
                if (z | reg) & locked_all and not ABSENCE_EXEMPT.search(s):
                    out.append("absence_contradicts_locked")
            elif z - locked_all:                                    # 이름 없는 관찰 — 칸이 잠긴 칸 안에
                out.append("observation_zone_outside_locked")
            continue
        # 이름이 든 부재 절('균열은 없다' · '위쪽 왼쪽에는 균열이 없다')은 버리지 않고 사실로 남긴다 — 잠긴 입력과 양립해야 한다.
        # 절은 쉼표 · 연결 어미로 나눈다. 수가 붙은 이름은 긍정 언급이고, 수 없이 뒤에 부정이 따르는 이름은 부재 주장이다
        positive: set = set()
        for c in _clauses(s):
            c_names = [n for n in names_here if n in c]
            if not c_names:
                continue
            c_neg = bool(NEG.search(c))
            for n in c_names:
                for m in re.finditer(re.escape(n), c):
                    tail = c[m.end():]
                    if re.match(COUNT_AFTER_NAME, tail) or not (c_neg and NEG.search(tail)):
                        positive.add(n)
                    elif not ABSENCE_EXEMPT.search(c):
                        claimed = {zz for _, _, zz in zone_spans(c)} | regions_in(c)
                        if (not claimed and n in counts) or (claimed and (lzones.get(n, set()) & claimed)):
                            out.append("absence_contradicts_locked")
        names_here = [n for n in names_here if n in positive]
        if not names_here:
            continue                                                # 부재 주장만 있는 문장 — 이름을 세지 않는다
        name_pos = sorted((m.start(), m.end(), n) for n in names_here for m in re.finditer(re.escape(n), s))
        for _, _, n in name_pos:
            said.add(n)
        spans = zone_spans(s)
        target: list = [None] * len(spans)                          # 구역마다 붙일 이름들 — 이어진 구역 묶음은 마지막 구역을 따른다
        for i in range(len(spans) - 1, -1, -1):
            zs, ze, z = spans[i]
            after = s[ze:]
            if i + 1 < len(spans) and re.fullmatch(r"\s*(?:와|과|및|,)\s*", s[ze:spans[i + 1][0]]):
                target[i] = target[i + 1]
                continue
            nxt = re.match(r"\s*(?:(?:구역|칸)\s*)?(?:에|의)\s*(?:는\s*)?", after)
            follow = []
            if nxt:
                stop = (spans[i + 1][0] - ze) if i + 1 < len(spans) else len(after)
                seg = after[nxt.end():stop].split(",")[0]            # 다음 구역 앞에서 닫는다 — '균열 1개와 아래쪽 오른쪽에 기공' 의 기공은 뒤 구역의 것
                if any(seg.startswith(n) for n in names_here):  # 'ZONE에 · 의 이름 …' — 바로 뒤의 이름들에
                    follow = [n for _, n in sorted((m.start(), n) for n in names_here for m in re.finditer(re.escape(n), seg))]
            if follow:
                target[i] = follow
            else:                                                   # 'NAME … ZONE' — 앞의 가장 가까운 이름에
                prev = [n for st, _, n in name_pos if st < zs]
                target[i] = prev[-1:] if prev else []
        for (zs, ze, z), names_for in zip(spans, target):
            for n in names_for or []:
                mzones.setdefault(n, set()).add(z)
        named = {n: list(re.finditer(re.escape(n) + COUNT_AFTER_NAME, s)) for n in names_here}
        taken = [m.span() for ms in named.values() for m in ms] + [sp for sp, _ in grands]
        bare: dict[str, list] = {}                                  # 'NAME 2개가 A에, 1개가 B에' — 이름을 생략한 수는 앞 이름에
        for m in BARE_COUNT.finditer(s):
            if any(a <= m.start() < b for a, b in taken):
                continue                                            # 이름 붙은 수 · 결함 전체의 수의 일부
            g = m.group(1) or m.group(2)
            k, summ = int(g) if g.isdigit() else KO[g], bool(SUMMARY.search(s[max(0, m.start() - 4):m.start()]))
            after = re.match(r"\s*의\s*(" + "|".join(map(re.escape, NAMES)) + ")", s[m.end():])
            if after:                                               # 'N개의 NAME' — 뒤의 이름의 수
                bare.setdefault(after.group(1), []).append((m.start(), k, summ))
                continue
            if summ:
                continue                                            # '합쳐 3개' — 여러 이름을 합친 총수는 이름에 세지 않는다
            prev = [n for st, _, n in name_pos if st < m.start()]
            if prev:
                bare.setdefault(prev[-1], []).append((m.start(), k, False))
        for n in names_here:
            found = [(m.start(), *count_of(m)) for m in named[n]] + bare.get(n, [])
            found.sort()
            ks = []
            for pos, k, is_sum in found:
                # 요약어가 수보다 앞에 있는 문장에서 이름의 수가 하나뿐이면 그 수가 총수다('결함은 모두 3개로, 기공 3개가 …').
                # 둘 이상이면 '모두 · 총' 이 붙은 수만 총수다. '기공 2개는 모두 위쪽 왼쪽에' 의 '모두' 는 수 뒤라 총수 표시가 아니다
                if is_sum or (SUMMARY.search(s[:pos]) and len(found) == 1):
                    total_st.setdefault(n, []).append(k)
                else:
                    ks.append(k)
            # 구역별 수 — 구역 하나에 수 하나, 수가 구역 개수와 같으면 하나씩, 수와 구역이 같은 개수면 차례대로 짝짓는다.
            # 그 밖(수 생략 꼴 · 수 셋에 구역 둘 같은)은 '그 구역 묶음의 합' 언급이다. 구역이 없으면 이름의 총수 언급이다.
            zl = []
            for (zs, ze, z), names_for in zip(spans, target):
                if n in (names_for or []) and z not in zl:
                    zl.append(z)
            if ks and zl and len(ks) == len(zl) + 1 and ks[0] == sum(ks[1:]):
                total_st.setdefault(n, []).append(ks[0])            # '기공 3개가 A에 2개, B에 1개' — 앞 수가 뒤 수들의 합이면 총수
                ks = ks[1:]
            elif ks and len(zl) >= 2 and len(ks) == len(zl) and re.search(r"나머지", s) and ks[0] > sum(ks[1:]):
                total_st.setdefault(n, []).append(ks[0])            # '3개 가운데 2개는 A에, 나머지는 B에' — 나머지 = 총수 − 앞 수들
                ks = ks[1:] + [ks[0] - sum(ks[1:])]
            if ks:
                add_zone_counts(n, ks, zl, zone_st, group_st, total_st)
        # 띠 소견 — '결함은 X 띠를 따라' 는 (그 이름의) 잠긴 칸이 모두 그 띠 안이라는 말이다. '대부분 · 일부 · 주로' 면 묻지 않는다
    for s in sents:
        reg = regions_in(s)
        if not reg or re.search(r"(?:대부분|일부|주로|없)", s):
            continue
        subj = [n for n in NAMES if re.search(re.escape(n) + r"(?:은|는|이|가)\s", s)]
        if subj:
            if any(n in lzones and not lzones[n] <= reg for n in subj):
                out.append("band_contradicts_locked")
        elif re.search(r"결함(?:은|이|들은)\s", s) and not locked_all <= reg:
            out.append("band_contradicts_locked")
    if said != set(counts):
        out.append("names_mismatch")
    out += statement_checks(counts, lzc, mzones, zone_st, group_st, total_st, grand_st)
    for n, zs in mzones.items():
        if n in lzones and not zs <= lzones[n]:
            out.append("zone_outside_locked")
            break
    for n, zs in lzones.items():
        if not zs <= mzones.get(n, set()):
            out.append("zone_not_described")
            break
    return sorted(set(out))


def statement_checks(counts: dict, lzc: dict, mzones: dict, zone_st: dict, group_st: dict, total_st: dict,
                     grand_st: list) -> list[str]:
    """모아 둔 언급의 대조. 뒤가 앞을 덮지 않는다 — 같은 항목에 다른 값이 둘 이상이면 모순(statement_conflict)이고,
    명시된 구역별 수는 다른 문장에 무엇이 있든 잠긴 값과 맞아야 한다."""
    out = []
    for vals in (*zone_st.values(), *group_st.values(), *total_st.values(), grand_st):
        if len(set(vals)) > 1:
            out.append("statement_conflict")
    for (n, z), vals in zone_st.items():                             # 수가 적혔으면 반드시 맞아야 한다
        if any(v != lzc.get((n, z), 0) for v in vals):
            out.append("zone_count_mismatch")
    for (n, zs), vals in group_st.items():                           # 구역 묶음의 합도 잠긴 값의 합과
        if any(v != sum(lzc.get((n, z), 0) for z in zs) for v in vals):
            out.append("zone_count_mismatch")
        if all((n, z) in zone_st for z in zs) and any(v != sum(zone_st[(n, z)][0] for z in zs) for v in vals):
            out.append("zone_sum_mismatch")
    for n in sorted(set(counts) | set(total_st) | {n for n, _ in zone_st} | {n for n, _ in group_st}):
        told = set(total_st.get(n, [])) | {v for (m, zs), vals in group_st.items() if m == n and zs == mzones.get(n) for v in vals}
        zkeys = [z for (m, z) in zone_st if m == n]
        zsum = sum(zone_st[(n, z)][0] for z in zkeys) if zkeys else None
        if zkeys and set(zkeys) == mzones.get(n, set()):              # 구역별 수가 모든 구역을 덮으면 그 합이 총수다
            if told and told != {zsum}:
                out.append("zone_sum_mismatch")
            told = told | {zsum}
        elif not told and zkeys:
            told = {zsum}                                            # 일부 구역의 수만 있으면 그 합(모자라면 총수 불일치)
        if n in counts and (not told or told != {counts[n]}):
            out.append("count_mismatch")
    if grand_st and set(grand_st) != {sum(counts.values())}:
        out.append("count_mismatch")
    return out


def contract(task: dict, text: str) -> list[str]:
    """v6 생성 계약의 고정 문장 — 정상 행은 두 문장이 정해진 문장, 결함 행은 마지막 문장이 정해진 제한 문장."""
    sents = [s if s.endswith((".", "?", "!")) else s + "." for s in P.sentences(text)]
    if P.row_class(task) == "defect_free":
        ok = len(sents) == 2 and sents[0] in G.NORMAL_ABSENCE_FORMS and sents[1] in G.NORMAL_LIMITATION_FORMS
        return [] if ok else ["normal_not_contract_form"]
    return [] if sents and sents[-1] in G.LIMITATION_FORMS else ["limitation_not_contract_form"]


def attributes(task: dict, text: str) -> list[str]:
    """주석 밖 시각 속성의 긍정 서술(크고 둥근 · 밝은 …) — '알 수 없다' 류의 한계 서술은 세지 않는다."""
    if P.row_class(task) != "defect":
        return []
    bad = [s for s in P.sentences(text) if ATTR.search(s + " ") and not ATTR_OK.search(s)]
    return ["unannotated_attribute"] if bad else []


def grounded_codes(task: dict, text: str) -> list[str]:
    return defect_grounded(task, text) if P.row_class(task) == "defect" else []


def contract_codes(task: dict, text: str) -> list[str]:
    return contract(task, text) if P.row_class(task) is not None else []


def attribute_codes(task: dict, text: str) -> list[str]:
    return attributes(task, text)


# ---------------------------------------------------------------- 정상 행의 음성 패턴(7판 보강 4)
# 정상 행의 수락 = 생성 계약 일치(엄격) ∧ 음성 패턴 없음. 아래는 계약 밖 글도 잡기 위한 보강이다 — 결함 행에는 걸지 않는다.

_V_POS = r"(?:(?<!불)합격|(?<!불)안전|양호|(?<!부)적합|건전|우수|문제(?:가|는)?\s*없)"
"""긍정 판정어. '불합격 · 부적합' 은 뺀다(방향이 반대). '문제(가) 없' 도 양호 단정이다."""
_V_LIMIT = r"(?:(?<!불)합격|(?<!불)안전|합부|양호|(?<!부)적합)"
"""제한 문장이 다루는 판정어 — 합부(합격 · 불합격)도 든다."""
_NEGATION = re.compile(r"(?:않|아니|아닙|아님|아닌|없|못|모르|모른|어렵)")
_CLAUSE_SPLIT = re.compile(r"(?:지만|으나|는데|그러나|그런데|(?<=[없않])고)[,\s]+")
_INVERTED = re.compile(r"(?:않|아니)\S*(?:다고|라고)\s*(?:볼|할|말할|여길)\s*(?:이유|근거|까닭|필요)\S*\s*(?:없|않)")
_WRONG_DIR = re.compile(r"(?:불합격|부적합|불량|위험)\S*\s*(?:을|를|이|가)?\s*(?:뜻|의미|보장|나타내|말하)")
_VERDICT_LABEL = re.compile(r"(?:판정|결론|결과|평가)\s*[:：]?\s*(?:은|는)?\s*(?:합격|안전|양호|적합)")
_CAUSE = re.compile(r"(?:때문(?:이다|입니다|이에요|이라)|덕분|덕택|탓(?:이다|입니다)|원인(?:은|이))")
_KO_NUM = r"(?:영|공|하나|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|스무)"
_NUMBER_WORD = re.compile(r"(?<![가-힣])" + _KO_NUM + r"\s*(?:개|건|곳|군데|가지)(?=[이가은는의도다만,.\s]|$)"
                          r"|(?:영|공|일|이|삼|사|오|육|칠|팔|구)점(?:영|공|일|이|삼|사|오|육|칠|팔|구)+"
                          r"|밀리미터|센티미터|마이크로미터|(?<![가-힣])미터|㎜|(?<![A-Za-z])(?:mm|cm|μm|px)(?![A-Za-z])|픽셀")
_INSTRUCTION = re.compile(r"(?:마라|마시오|하라|하시오|해라|말\s*것|쓸\s*것|적을\s*것)(?=[.\s]|$)"
                          r"|규칙\s*[:：]|형식\s*[:：]|다음\s*(?:형식|규칙|지시|조건)|지시문|프롬프트|출력\s*형식|응답\s*형식|(?i:json)")
_POSITION = re.compile(r"(?:위쪽|아래쪽|왼쪽|오른쪽|왼편|오른편|가운데|중앙|가장자리|모서리|구석|상단|하단|좌측|우측|윗부분|아랫부분"
                       r"|(?<![가-힣])위(?=\s*(?:왼|오른|가운데|쪽))|(?<![가-힣])아래(?=\s*(?:왼|오른|가운데|쪽))|띠|구역|(?<![가-힣])칸(?![가-힣]))")
_DEFECT_WORD = re.compile(r"(?:크랙|갈라짐|기포|구멍|언더컷|오버랩|스패터|터짐|이물|함몰|용입\s*부족|용입불량|슬래그|텅스텐)")
_DEFECT_EXISTS = re.compile(r"결함(?:이|은|들이|들은)?\s*(?:\S+\s+)?(?:보인다|보입니다|보이는|있다|있습니다|있는|관찰|확인된다|확인됩니다|나타난다|존재)")


def _clauses(sentence: str) -> list[str]:
    return [c for c in _CLAUSE_SPLIT.split(sentence) if c.strip()]


def _negated_after(clause: str, start: int) -> bool:
    return bool(_NEGATION.search(clause, start))


def normal_negative(task: dict, text: str) -> list[str]:
    """정상 행(`defect_free`)의 음성 패턴. 사유 코드 목록 — 빈 목록이 통과. 결함 행은 빈 목록."""
    if P.row_class(task) != "defect_free":
        return []
    sents = [s.strip() for s in P.sentences(text) if s.strip()]
    out = []
    has_limit = False
    for s in sents:
        body = s.rstrip(".?!")
        if _INVERTED.search(body):
            out.append("normal_limitation_inverted")
        if _VERDICT_LABEL.search(body):
            out.append("normal_verdict_implied")
        if _CAUSE.search(body):
            out.append("normal_cause_asserted")
        if _NUMBER_WORD.search(body):
            out.append("normal_number_or_unit")
        if _INSTRUCTION.search(body):
            out.append("normal_instruction_leak")
        if _POSITION.search(body) or _DEFECT_WORD.search(body):
            out.append("normal_defect_or_position")
        for m in _DEFECT_EXISTS.finditer(body):
            if not _NEGATION.search(body, m.start()):
                out.append("normal_defect_or_position")
        for c in _clauses(body):
            for m in re.finditer(_V_POS, c):            # 긍정 판정어 뒤에 같은 절의 부정이 없으면 단정이다
                if not _negated_after(c, m.end()):
                    out.append("normal_verdict_implied")
            for m in re.finditer(_V_LIMIT, c):          # 판정어 뒤에 부정이 따르는 절이 제한 문장이다(N5 의 넓은 꼴)
                if _negated_after(c, m.end()):
                    has_limit = True
        if _WRONG_DIR.search(body) and not re.search(_V_LIMIT, body):
            out.append("normal_limitation_wrong_direction")
    if not has_limit:
        out.append("normal_limitation_absent")
    return sorted(set(out))


def normal_codes(task: dict, text: str) -> list[str]:
    return normal_negative(task, text)

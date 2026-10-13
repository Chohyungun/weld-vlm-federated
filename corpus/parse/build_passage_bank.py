"""구절 은행 빌드 — 원문 PDF 의 텍스트 층에서 구절을 뽑고, 쪽 화면과 대조해 옮겨 둔 구절을 합친다 (2026-10-07, v2).

사용:

    python -m corpus.parse.build_passage_bank \
        --source KR-RULES-P2=<pdf> --source IACS47=<pdf> --source ABS-PUB87-2007=<pdf> \
        --source IACS-URW33-R1C1=<pdf> --source FHWA-HIF-19-088=<pdf> \
        --source US-WAR-DEPT-TM-1-430-1942=<pdf> --source MDPI-APPLSCI-2020-10-1878=<pdf> \
        --curated <쪽 화면과 대조해 옮긴 구절 jsonl> \
        --out-full <미추적 경로>/passage_bank_v2_full.jsonl \
        --out-pointer corpus/parse/passage_bank_v2.jsonl \
        --report corpus/parse/passage_bank_v2.build_report.json

원문 경로는 인자로만 받는다(로컬 절대경로를 추적 파일에 남기지 않는다). 원문의 sha256 이
`DOCS` · `CURATED` 에 적은 값과 다르면 멈춘다. 옮겨 둔 구절은 그 쪽의 텍스트 층과 다시 대조해 결과를 행에 남긴다.

**하지 않는 것.** 본문을 고쳐 쓰지 않는다 — 텍스트 층을 읽은 순서대로 잇는다. 지식 면 · 결함 클래스 ·
관련성 층은 키워드 규칙으로 붙인 **후보 표지**이고 사람이 보지 않았다(`facet_method = keyword_rule_v1`).
권리는 모두 `pending` 이다. 유료 표준 인용 문장과, 유료 표준 이름이 수치와 함께 든 문장은 구절에서
덜어 내고 덜어 낸 수를 적는다(`passage_bank.paid_sentence_reason`, 소비 경계도 같은 규칙을 쓴다).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from corpus.parse.passage_bank import (
    DEFECTS,
    FACETS,
    MODERN_ONLY_FACETS,
    PAID_NAME_RE,
    SCHEMA_VERSION,
    check_row_shape,
    dump_jsonl,
    load_jsonl,
    paid_reference,
    paid_sentence_reason,
    pointer_row,
    split_sentences,
    text_sha256,
)

BUILDER_VERSION = "build_passage_bank_v3"

MAX_CHARS = 900
MIN_CHARS = 120

# --------------------------------------------------------------------------- 문서 명세


@dataclass(frozen=True)
class DocSpec:
    doc_id: str
    title: str
    edition: str
    sha256: str
    lang: str
    lineage_group_id: str
    pages: tuple[tuple[int, int], ...]          # 물리 쪽 닫힌 구간
    drop_line_res: tuple[str, ...]              # 머리글 · 바닥글 · 그림 표제 줄
    printed_re: str                             # 인쇄 쪽 번호 — 묶음 가운데 처음 잡힌 것
    heading_re: str                             # 절 머리 줄 — 묶음 1 이 절 번호
    table_re: str                               # 표 표제 줄 — 묶음 1 이 표 번호
    doc_flags: tuple[str, ...] = ()
    l2_pages: tuple[int, ...] = ()              # 유료 계보 미확인 RT 판정 블록이 있는 쪽
    ut_pages: tuple[int, ...] = ()              # 유료 계보 미확인 UT 판정 블록이 있는 쪽(KS B 시험편 · 방법 인용)
    first_line_is_page_number: bool = False     # 쪽 번호가 첫 줄에 홀로 있는 문서(FHWA)
    scope: str = ""                             # 문서가 다루는 구조물 · 재료 · 검사(원문 적용 범위 절 기준)
    tables_own_section: bool = False            # 표 번호가 곧 절이다(IACS "Table 8.5" · ABS "TABLE 19") — 앞 절을 부모로 달지 않는다
    table_mode_sticky: bool = False             # 표 안에서는 번호 줄을 절 머리로 읽지 않는다(ABS 표 각주 "1 Different …")


KR_FOOT = r"선급\s*및\s*강선규칙(?:\s*적용지침)?\s*2025"

DOCS: dict[str, DocSpec] = {
    "KR-RULES-P2": DocSpec(
        doc_id="KR-RULES-P2",
        title="2025 선급 및 강선규칙 / 적용지침 제2편 재료 및 용접",
        edition="2025",
        sha256="46c4ee189f4adef29e42320f9301d8db577946ec96f893be9ff2f1683879552b",
        lang="ko",
        lineage_group_id="LG-KR-RULES-P2",
        # 규칙 2장 용접 · 지침 2장 용접 · 부록 2-4(크랭크스로우 용접보수) · 2-6(프로펠러 주물 보수) ·
        # 2-7(선체 용접이음 비파괴검사) · 2-12(향상된 비파괴검사)
        pages=((123, 220), (264, 276), (291, 293), (307, 331), (351, 361)),
        drop_line_res=(
            r"^\d+\s*편\s*재료\s*및\s*용접$",
            r"^\d+\s*장\s*[가-힣 ]+\d+\s*편\s*\d+\s*장$",
            r"^부록\s*\d+-\d+.*\d+\s*편\s*부록\s*\d+-\d+$",
            rf"^\d+\s*{KR_FOOT}$",
            rf"^{KR_FOOT}\s*\d+$",
            r"^그림\s*\d+(\.\d+)*\s.*$",
        ),
        printed_re=rf"(?m)^(\d+)\s*{KR_FOOT}\s*$|^{KR_FOOT}\s*(\d+)\s*$",
        heading_re=r"^(\d{3})\.\s*[가-힣][^0-9.]{0,40}$|^([1-9]\d?)\.\s*[가-힣][가-힣 ·()]{0,25}$",
        table_re=r"^(표\s*\d+(?:\.\d+)*(?:-\d+)?)(?:\s|$)",
        doc_flags=("kr_lt_glyph_is_angle",),
        l2_pages=(327, 328),
        ut_pages=(328, 329),
        scope="선박 용접 — 재료 · 시공 · 승인시험(강, 2장 6절의 알루미늄 용접재료 포함)",
    ),
    "IACS47": DocSpec(
        doc_id="IACS47",
        title="IACS Rec. No.47 Shipbuilding and Repair Quality Standard",
        edition="Rev.10 2021 / Corr.1 2025",
        sha256="ac54b7904b3f27d8af830ef2f4ff0ca18967a407aeead67c481b136b542f15a6",
        lang="en",
        lineage_group_id="LG-IACS-REC47",
        pages=((4, 48), (51, 67)),                 # 1~3 표지 · 목차 · 참고문헌, 49 Part B 목차, 50 참고문헌 제외
        drop_line_res=(r"^No\.\s*47$", r"^No\.$", r"^47$", r"^\(cont\)$", r"^Page \d+ of 67 IACS Rec\..*$"),
        printed_re=r"Page (\d+) of 67",
        heading_re=r"^([1-9]\d?)\.\s+[A-Z][a-z]|^([1-9]\d?(?:\.\d+){1,3})\.?\s+[A-Z(]",
        table_re=r"^(Table\s+\d+(?:\.\d+)?)\s*[–-]",
        tables_own_section=True,
        scope="선체 구조 신조 · 수리 품질(일반 · 고장력 선체 구조용 강, 1.1)",
    ),
    "ABS-PUB87-2007": DocSpec(
        doc_id="ABS-PUB87-2007",
        title="ABS Guide for Shipbuilding and Repair Quality Standard for Hull Structures During Construction",
        edition="March 2007 (2nd ed., Pub 87)",
        sha256="be55985fd3afe993c86125d27dc81c8039f82db40fb029282bf556ea3bd30565",
        lang="en",
        # 서문(물리 3)이 IACS Rec.47 Rev.3 을 바탕으로 했다고 적는다 — Rec.47 과 같은 계보다.
        lineage_group_id="LG-IACS-REC47",
        pages=((7, 44),),                          # 1~6 표지 · 서문 · 목차 제외
        drop_line_res=(
            r"^ABS GUIDE FOR SHIPBUILDING AND REPAIR QUALITY STANDARD FOR HULL STRUCTURES DURING CONSTRUCTION\s*\.?\s*2007\s*\d*$",
            r"^\d+\s*ABS GUIDE FOR SHIPBUILDING AND REPAIR QUALITY STANDARD FOR HULL STRUCTURES DURING CONSTRUCTION\s*\.?\s*2007$",
        ),
        printed_re=r"(?m)CONSTRUCTION\s*\.?\s*2007\s+(\d+)\s*$|^(\d+)\s+ABS GUIDE",
        heading_re=r"^([1-9]\d?)\s+[A-Z][a-z]+(?:\s+(?:[A-Za-z]+|of|and))*$|^([1-9]\d?(?:\.\d+){1,2})\s+[A-Z][a-z]",
        table_re=r"^(TABLE\s+\d+)\s*$",
        tables_own_section=True,
        table_mode_sticky=True,
        doc_flags=("archived_edition",),
        scope="선체 구조 신조 품질(일반 · 고장력 선체 구조용 강, 1절) — 2007 보관판",
    ),
    "IACS-URW33-R1C1": DocSpec(
        doc_id="IACS-URW33-R1C1",
        title="IACS UR W33 Non-destructive testing of ship hull steel welds",
        edition="2019 / Rev.1 2020 / Corr.1 2021",
        sha256="1edc750fd91c1d8605e5bc605cc2b87b1846c612b003dad0a56c2b8e1629fee3",
        lang="en",
        lineage_group_id="LG-IACS-URW33",
        pages=((1, 14),),
        drop_line_res=(r"^W33$", r"^\(cont\)$", r"^Page \d+ of 14 IACS Req\..*$"),
        printed_re=r"Page (\d+) of 14",
        heading_re=r"^([1-9]\d?)\.\s+[A-Z][a-z]|^([1-9]\d?(?:\.\d+){1,3})\.?\s+[A-Z(]",
        table_re=r"^(Table\s+\d+)\.?(?:\s|$)",
        tables_own_section=True,
        scope="선체 강 용접부 비파괴검사(융접, 2절)",
    ),
    "FHWA-HIF-19-088": DocSpec(
        doc_id="FHWA-HIF-19-088",
        title="Bridge Welding Reference Manual",
        edition="September 2019, with September 2020 errata",
        sha256="d317b8e9ebb50290b5648a9a7b7876579538caaade41a8b551bb7229cc064046",
        lang="en",
        lineage_group_id="LG-FHWA-HIF-19-088",
        pages=((25, 321),),                        # 1~12장. 부록 A · B(D1.1 과 D1.5 의 차이) · 참고문헌 제외
        drop_line_res=(r"^Source:.*$", r"^Figure\s+\d+\.\s.*$", r"^©.*$", r".*Copyright.*$"),
        printed_re=r"\A\s*(\d{1,3})[ \t]*\r?\n",
        heading_re=r"^([1-9]\d?(?:\.\d+){1,4})\.\s+\S|^(CHAPTER\s+\d+)\s*[-–]",
        table_re=r"^(Table\s+\d+)\.\s",
        tables_own_section=True,
        doc_flags=("contractor_authored", "third_party_figures_present"),
        first_line_is_page_number=True,
        scope="강 도로교 용접(AWS D1.5 체계) — 선박이 아니다",
    ),
}

# --------------------------------------------------------------------------- 문자 정리

#: Symbol 글꼴의 사용자 영역 문자 — Adobe Symbol 인코딩으로 되돌린다(IACS Rec.47 의 표).
PUA_SYMBOL: dict[str, str] = {
    chr(0xF03C): "<", chr(0xF03E): ">", chr(0xF071): "θ", chr(0xF0A3): "≤",
    chr(0xF0B0): "°", chr(0xF0B3): "≥", chr(0xF0B4): "×",
}
PUA_FIRST, PUA_LAST = chr(0xE000), chr(0xF8FF)
SOFT_HYPHEN_MARK = chr(0xFFFE)      # pdfium 이 줄끝 하이픈 자리에 넣는 문자
SOFT_HYPHEN = chr(0xAD)              # 조판용 무른 하이픈
JOIN = "\x00"                       # 이을 때 공백 없이 붙이라는 내부 표지
UNMAPPED_MARK = "□"


def clean_chars(s: str) -> tuple[str, Counter]:
    """사용자 영역 문자 대응 · 줄끝 하이픈 처리. 바꾼 수를 함께 돌려준다."""
    c: Counter = Counter()
    out = []
    for ch in s:
        if ch in PUA_SYMBOL:
            out.append(PUA_SYMBOL[ch])
            c["pua_symbol_mapped"] += 1
        elif PUA_FIRST <= ch <= PUA_LAST:
            out.append(UNMAPPED_MARK)
            c["pua_unmapped"] += 1
        elif ch == SOFT_HYPHEN_MARK:
            out.append(JOIN)
            c["soft_hyphen_joined"] += 1
        else:
            out.append(ch)
    return unicodedata.normalize("NFC", "".join(out)), c


# --------------------------------------------------------------------------- 분류 규칙(후보 표지)

DEFECT_RES: dict[str, re.Pattern] = {
    "crack": re.compile(r"\bcrack|균열|터짐|갈라짐", re.IGNORECASE),
    "porosity": re.compile(r"porosit|\bpores?\b|gas pocket|blow\s*holes?|wormhole|기공|블로우\s*홀|핀\s*홀", re.IGNORECASE),
    "lack_of_fusion": re.compile(r"lack of fusion|incomplete fusion|fusion[- ]type|cold lap|융합\s*(?:불량|부족)|용착\s*불량", re.IGNORECASE),
    "lack_of_penetration": re.compile(
        r"incomplete (?:joint |root )?penetration|lack of penetration|inadequate (?:joint )?penetration|"
        r"용입\s*(?:불량|부족|불충분)", re.IGNORECASE),
    # electro-slag(용접법) · slag wool 은 결함이 아니다.
    "slag_inclusion": re.compile(r"(?<!electro-)(?<!electro)\bslag\b(?!\s*(?:welding|wool))|슬래그", re.IGNORECASE),
    "undercut": re.compile(r"undercut|언더\s*컷|언터\s*컷", re.IGNORECASE),
    # '겹침 이음'(lap joint)은 결함이 아니므로 한국어 '겹침'은 쓰지 않는다.
    "overlap": re.compile(r"\boverlaps?\b|오버\s*랩", re.IGNORECASE),
}
QUALITY_RE = re.compile(
    r"discontinuit|defect|imperfection|inspect|examin|nondestructive|non-destructive|\bNDE\b|\bNDT\b|"
    r"radiograph|ultrason|magnetic particle|penetrant|visual|repair|preheat|interpass|hydrogen|distortion|"
    r"spatter|arc strike|toe angle|convexity|concavity|reinforcement|acceptance|reject|soundness|"
    r"결함|불연속|검사|비파괴|방사선|초음파|자분|침투|외관|보수|예열|층간|수소|변형|스패터|아크\s*스트라이크|"
    r"합격|불합격|품질|허용", re.IGNORECASE)
WELD_RE = re.compile(r"weld|용접", re.IGNORECASE)
FACET_RES: dict[str, re.Pattern] = {
    "definition": re.compile(r"refers? to|is defined|defined as|\bmeans\b|is a (?:type|form|term)|정의|이란|라 한다|를 말한다", re.IGNORECASE),
    "appearance": re.compile(r"appear|appearance|looks? like|shape|indication|visible|image|radiograph(?:ic)? (?:image|film)|"
                             r"모양|형상|나타난다|보인다|지시", re.IGNORECASE),
    "cause": re.compile(r"caused by|causes?\b|result(?:s|ing)? (?:of|from)|due to|occurs? (?:when|as|if)|because|contribut|"
                        r"원인|때문|발생|기인", re.IGNORECASE),
    "prevention": re.compile(r"prevent|avoid|minimi[sz]|reduce the (?:risk|likelihood|potential)|preheat|"
                             r"방지|피하|예방|억제|예열", re.IGNORECASE),
    "repair": re.compile(r"repair|remov|grind|gouging|re-?weld|excavat|보수|제거|그라인딩|가우징|재용접|덧용접", re.IGNORECASE),
    "criteria": re.compile(r"acceptab|reject|shall not|not (?:be )?permitted|\blimit|maximum|minimum|≤|≥|"
                           r"합격|불합격|허용|초과|이하|이상", re.IGNORECASE),
    "measurement": re.compile(r"measur|gauge|depth|\blength|\bsize\b|inspect|examin|\btest|측정|검사|시험|깊이|길이", re.IGNORECASE),
    "applicability": re.compile(r"\bappl(?:y|ies|icable|ication)|scope|only when|except|unless|적용|범위|다만|경우", re.IGNORECASE),
}

TOC_RE = re.compile(r"\.{5,}|(?:\.\s){5,}")
SEE_TABLE_RE = re.compile(r"\(see Table", re.IGNORECASE)
HISTORY_RE = re.compile(r"uniformly implemented|contracted for construction|\(Rev\.\s*\d|Rev\.\d+\s+\w{3,9}\s+\d{4}|개정\s*이력")
#: KR 본문 조항 표지 — 표 안에 있다가 이 줄을 만나면 표가 끝난 것으로 본다.
KR_CLAUSE_RE = re.compile(r"^\(\d+\)\s|^\([가-힣]\)\s|^\d+\.\s")
#: KR 부록의 호 머리 — '(5) 검사방법', '(2) 용어 정의 (2021)'.
KR_APPX_SUB_RE = re.compile(r"^\((\d{1,2})\)\s*[가-힣][가-힣 ·]{0,20}(?:\s*\(\d{4}\))?$")


def classify(text: str) -> tuple[list[str], list[str], str | None]:
    defects = [d for d in DEFECTS if DEFECT_RES[d].search(text)]
    facets = [f for f in FACETS if FACET_RES[f].search(text)]
    if defects:
        tier = "defect_specific"
    elif QUALITY_RE.search(text):
        tier = "weld_quality_general"
    elif WELD_RE.search(text):
        tier = "welding_general"
    else:
        tier = None
    return facets, defects, tier


# --------------------------------------------------------------------------- 쪽 → 줄


@dataclass
class Line:
    page: int
    text: str


def printed_page(spec: DocSpec, page_text: str) -> str | None:
    m = re.search(spec.printed_re, page_text)
    if not m:
        return None
    return next((g for g in m.groups() if g), None)


#: KR 부록은 장과 다루는 대상이 다르다 — 부록 이름 기준으로 범위를 덮어쓴다.
KR_PART_SCOPE: dict[str, str] = {
    "부록 2-4": "주강 크랭크스로우 용접보수 검사(기관 부품, 선체 아님)",
    "부록 2-6": "동합금 프로펠러 주물 침투탐상 · 결함 보수(주물, 선체 용접부 아님)",
    "부록 2-7": "선체 용접이음부 비파괴검사 기준",
    "부록 2-12": "향상된 비파괴검사 기준(강 · 스테인리스강)",
}


def scope_of(spec: DocSpec, part: str | None) -> str:
    for key, val in KR_PART_SCOPE.items():
        if part and part.endswith(key):
            return val
    return spec.scope


def kr_part(page_text: str) -> str | None:
    """KR 쪽의 소속 — '규칙 2장' · '지침 2장' · '지침 부록 2-7'. 머리글과 바닥글에서 읽는다."""
    kind = "지침" if re.search(r"선급\s*및\s*강선규칙\s*적용지침", page_text) else "규칙"
    m = re.search(r"(?m)^\s*부록\s*(\d+-\d+)\s", page_text)
    if m:
        return f"{kind} 부록 {m.group(1)}"
    m = re.search(r"(?m)^\s*(\d+)\s*장\s*[가-힣 ]+\d+\s*편\s*\d+\s*장\s*$", page_text)
    if m:
        return f"{kind} {m.group(1)}장"
    return None


def page_lines(spec: DocSpec, page_no: int, page_text: str, stats: Counter) -> list[Line]:
    raw, c = clean_chars(page_text)
    stats.update(c)
    drops = [re.compile(r) for r in spec.drop_line_res]
    out = []
    first = True
    for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        s = ln.strip()
        if not s or s == JOIN:
            continue
        # FHWA 의 쪽 번호는 첫 줄에 홀로 있다 — 본문 속 숫자 한 줄은 지우지 않도록 첫 줄만 본다.
        if first and spec.first_line_is_page_number and re.fullmatch(r"\d{1,3}", s):
            first = False
            stats["dropped_lines"] += 1
            continue
        first = False
        if any(d.match(s.replace(JOIN, "")) for d in drops):
            stats["dropped_lines"] += 1
            continue
        out.append(Line(page_no, s))
    return out


# --------------------------------------------------------------------------- 줄 → 절


@dataclass
class Section:
    section_id: str
    kind: str                                  # "clause" · "table" · "lead_in"
    part: str | None
    lines: list[Line] = field(default_factory=list)


def split_sections(spec: DocSpec, lines: Sequence[Line], parts: dict[int, str | None]) -> list[Section]:
    """절 머리 줄과 표 표제 줄에서 끊는다. 머리 줄도 본문에 남긴다(원문 순서 그대로)."""
    head = re.compile(spec.heading_re)
    table = re.compile(spec.table_re)
    secs: list[Section] = []
    first = lines[0].page if lines else 0
    cur = Section("(앞 쪽에서 이어짐)", "lead_in", parts.get(first))
    parent = cur.section_id
    article: tuple[str | None, str] | None = None      # KR — (소속, 3자리 조 번호)
    for ln in lines:
        t = ln.text.replace(JOIN, "")
        tm = table.match(t)
        hm = None if (cur.kind == "table" and spec.table_mode_sticky) else head.match(t)
        new: Section | None = None
        if tm:
            table_id = re.sub(r"\s+", " ", tm.group(1))
            sid = table_id if spec.tables_own_section else f"{parent} / {table_id}"
            new = Section(sid, "table", parts.get(ln.page))
        elif hm:
            sid = re.sub(r"\s+", " ", next(g for g in hm.groups() if g))
            part = parts.get(ln.page)
            if spec.doc_id == "KR-RULES-P2":
                # 규칙 · 지침의 장은 '306. 본용접' 아래 '1. 홈 가공' 처럼 조 → 항이다. 항은 같은 소속의 조 번호에 붙인다.
                # 부록에는 3자리 조가 없고 '1. 일반사항' 이 맨 위 단위다.
                if hm.group(1) or "부록" in (part or ""):
                    article = (part, sid)
                elif article is not None and article[0] == part:
                    sid = f"{article[1]}.{sid}"
            parent = sid
            new = Section(parent, "clause", part)
        elif (spec.doc_id == "KR-RULES-P2" and article is not None and "부록" in (parts.get(ln.page) or "")
              and article[0] == parts.get(ln.page) and KR_APPX_SUB_RE.match(t)):
            # 부록의 '(5) 검사방법' 같은 호 머리 — 맨 위 단위 번호에 붙인다.
            parent = f"{article[1]}.({KR_APPX_SUB_RE.match(t).group(1)})"
            new = Section(parent, "clause", parts.get(ln.page))
        elif cur.kind == "table" and spec.doc_id == "KR-RULES-P2" and KR_CLAUSE_RE.match(t):
            # 표가 끝나고 조항 본문이 다시 시작된다 — 표 앞의 절로 돌아간다.
            new = Section(parent, "clause", parts.get(ln.page))
        if new is not None:
            if cur.lines:
                secs.append(cur)
            cur = new
        cur.lines.append(ln)
    if cur.lines:
        secs.append(cur)
    return secs


# --------------------------------------------------------------------------- 절 → 문장 → 구절


def section_text(sec: Section) -> tuple[str, list[tuple[int, int]]]:
    """절 본문과 (본문 위치, 쪽) 대응표. 줄끝 하이픈은 공백 없이, 나머지 줄은 한 칸 띄워 잇는다."""
    text = ""
    marks: list[tuple[int, int]] = []
    for ln in sec.lines:
        if text.endswith(JOIN):
            text = text[:-1]
        elif text:
            text += " "
        marks.append((len(text), ln.page))
        text += ln.text
    # 줄 가운데 남은 표지는 지운다 — 위치 대응을 어긋내지 않도록 같은 길이의 빈칸 없이 지우고 대응표를 고친다.
    if JOIN in text:
        fixed = ""
        shift: list[int] = []
        for ch in text:
            shift.append(len(fixed))
            if ch != JOIN:
                fixed += ch
        shift.append(len(fixed))
        marks = [(shift[o], p) for o, p in marks]
        text = fixed
    return text, marks


#: 문장 · 조각 하나와 그것이 걸친 물리 쪽(처음 · 끝).
Unit = tuple[str, int, int]


def _page_at(marks: Sequence[tuple[int, int]], offset: int) -> int:
    page = marks[0][1]
    for off, pg in marks:
        if off <= offset:
            page = pg
        else:
            break
    return page


def sentences_with_pages(sec: Section) -> list[Unit]:
    """절 본문을 문장으로 나누고, 문장마다 시작 쪽과 끝 쪽을 붙인다(쪽 경계를 넘는 문장은 두 쪽에 걸친다)."""
    text, marks = section_text(sec)
    out: list[Unit] = []
    pos = 0
    for s in split_sentences(text):
        at = text.find(s, pos)
        at = pos if at < 0 else at
        out.append((s, _page_at(marks, at), _page_at(marks, at + max(len(s) - 1, 0))))
        pos = at + len(s)
    return out


def hard_split(s: str) -> list[str]:
    """MAX_CHARS 를 넘는 한 문장(마침표 없는 표 본문)을 공백 자리에서 자른다."""
    if len(s) <= MAX_CHARS:
        return [s]
    parts, cur = [], ""
    for w in s.split(" "):
        if cur and len(cur) + 1 + len(w) > MAX_CHARS:
            parts.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur:
        parts.append(cur)
    return parts


def windows(sentences: Sequence[Unit]) -> list[tuple[list[Unit], bool]]:
    """문장(본문, 시작 쪽, 끝 쪽)을 MAX_CHARS 를 넘지 않게 묶는다. 각 묶음에 '문장 중간에서 잘렸는가' 를 붙인다.

    마지막 묶음이 MIN_CHARS 보다 짧으면 앞 묶음에 붙인다 — 그래서 구절 길이의 실제 상한은 MAX_CHARS + MIN_CHARS 다.
    문장 중간에서 자른 조각은 그 문장이 걸친 쪽 전체를 물려받는다(쪽을 좁게 적지 않는다).
    """
    units: list[tuple[str, int, int, bool]] = []
    for s, p0, p1 in sentences:
        pieces = hard_split(s)
        units.extend((p, p0, p1, len(pieces) > 1) for p in pieces)
    out: list[tuple[list[Unit], bool]] = []
    cur: list[Unit] = []
    cut = False
    size = 0
    for s, p0, p1, was_cut in units:
        if cur and size + len(s) + 1 > MAX_CHARS:
            out.append((cur, cut))
            cur, size, cut = [], 0, False
        cur.append((s, p0, p1))
        cut = cut or was_cut
        size += len(s) + 1
    if cur:
        if out and size < MIN_CHARS and sum(len(u[0]) + 1 for u in out[-1][0]) + size <= MAX_CHARS + MIN_CHARS:
            prev, prev_cut = out.pop()
            out.append((prev + cur, prev_cut or cut))
        else:
            out.append((cur, cut))
    return out


def build_doc(spec: DocSpec, page_texts: Sequence[str], stats: Counter) -> tuple[list[dict], Counter]:
    """한 문서의 구절 행과, 뺀 사유별 건수."""
    printed_of: dict[int, str | None] = {}
    parts: dict[int, str | None] = {}
    ranges: list[list[Section]] = []
    # 쪽 구간마다 따로 나눈다 — 구간 사이에서 절이 이어 붙지 않게.
    for a, b in spec.pages:
        lines: list[Line] = []
        for p in range(a, b + 1):
            txt = page_texts[p - 1]
            printed_of[p] = printed_page(spec, txt)
            parts[p] = kr_part(txt) if spec.doc_id == "KR-RULES-P2" else None
            lines.extend(page_lines(spec, p, txt, stats))
        ranges.append(split_sections(spec, lines, parts))
    stats["pages_without_printed_number"] += sum(v is None for v in printed_of.values())
    rows: list[dict] = []
    excluded: Counter = Counter()
    used_ids: set[str] = set()
    for sections in ranges:
        # 머리글뿐인 절이나 한 문장짜리 절은 버리지 않고 같은 구간의 다음 절 앞에 붙인다(원문 순서 그대로).
        carry: list[Unit] = []
        carry_ids: list[str] = []
        carry_omitted: Counter = Counter()
        for i, sec in enumerate(sections):
            kept: list[Unit] = list(carry)
            omitted: Counter = Counter(carry_omitted)
            for s, p0, p1 in sentences_with_pages(sec):
                why = paid_sentence_reason(s, spec.doc_id)
                if why:
                    omitted[why] += 1
                    excluded[f"sentence:{why}"] += 1
                else:
                    kept.append((s, p0, p1))
            if sum(len(u[0]) + 1 for u in kept) < MIN_CHARS and i + 1 < len(sections):
                carry, carry_omitted = kept, omitted
                carry_ids.append(sec.section_id)
                continue
            heading_ids, carry, carry_ids, carry_omitted = carry_ids, [], [], Counter()
            rows.extend(_section_rows(spec, sec, kept, omitted, heading_ids, printed_of, used_ids, excluded))
    # 같은 본문(지문)이 여러 쪽에서 되풀이되면(표 각주 등) 처음 것만 남긴다 — 용량을 부풀리지 않게.
    seen: set[str] = set()
    unique: list[dict] = []
    for r in rows:
        if r["text_sha256"] in seen:
            excluded["passage:duplicate_text"] += 1
            continue
        seen.add(r["text_sha256"])
        unique.append(r)
    return unique, excluded


def _section_rows(
    spec: DocSpec,
    sec: Section,
    kept: Sequence[Unit],
    omitted: Counter,
    heading_ids: Sequence[str],
    printed_of: dict[int, str | None],
    used_ids: set[str],
    excluded: Counter,
) -> list[dict]:
    rows: list[dict] = []
    for k, (win, was_cut) in enumerate(windows(kept)):
        text = " ".join(u[0] for u in win).strip()
        pages = [pg for u in win for pg in range(u[1], u[2] + 1)]
        if TOC_RE.search(text) or len(SEE_TABLE_RE.findall(text)) >= 3:
            excluded["passage:toc_or_list"] += 1
            continue
        if HISTORY_RE.search(text):
            excluded["passage:revision_history"] += 1
            continue
        if len(text) < MIN_CHARS:
            excluded["passage:too_short"] += 1
            continue
        facets, defects, tier = classify(text)
        if tier is None:
            excluded["passage:off_topic"] += 1
            continue
        p0, p1 = min(pages), max(pages)
        sid = f"{sec.part}_{sec.section_id}" if sec.part else sec.section_id
        slug = re.sub(r"[^0-9A-Za-z가-힣._()\-]+", "", sid)[:40] or "s"
        pid = f"{spec.doc_id}:p{p0:03d}:{slug}:{k:02d}"
        n = 1
        while pid in used_ids:
            n += 1
            pid = f"{spec.doc_id}:p{p0:03d}:{slug}:{k:02d}-{n}"
        used_ids.add(pid)
        flags = list(spec.doc_flags)
        if sec.kind == "table":
            flags.append("table_text_layer")
        if sec.kind == "lead_in":
            flags.append("section_started_before_range")
        if was_cut:
            flags.append("hard_split_mid_sentence")
        if UNMAPPED_MARK in text:
            flags.append(f"pua_unmapped:{text.count(UNMAPPED_MARK)}")
        if spec.doc_id == "IACS47" and any(ch in text for ch in PUA_SYMBOL.values()):
            flags.append("pua_symbol_mapped")
        if "∠" in text:
            flags.append("kr_angle_glyph_means_lt")
        if any(pg in spec.l2_pages for pg in pages):
            flags.append("L2_lineage_unverified")
        if any(pg in spec.ut_pages for pg in pages):
            flags.append("UT_lineage_unverified")
        if PAID_NAME_RE.search(text):
            flags.append("paid_standard_name_without_numeric")
        elif paid_reference(text, spec.doc_id):
            flags.append("paid_clause_reference_without_numeric")
        if omitted:
            flags.append("sentences_omitted_in_section:" + ",".join(f"{a}={b}" for a, b in sorted(omitted.items())))
        row = {
            "schema_version": SCHEMA_VERSION,
            "passage_id": pid,
            "doc_id": spec.doc_id,
            "registry": "corpus/parse/sources.yaml",
            "edition": spec.edition,
            "source_file_sha256": spec.sha256,
            "locator": {
                "pdf_page_start": p0, "pdf_page_end": p1,
                "printed_page_start": printed_of.get(p0), "printed_page_end": printed_of.get(p1),
                "part": sec.part, "section_id": sec.section_id, "section_kind": sec.kind,
                "carried_heading_ids": list(heading_ids) if k == 0 else [],
            },
            "language": spec.lang,
            "text_kind": "verbatim_extract",
            "text": text,
            "text_sha256": text_sha256(text),
            "text_chars": len(text),
            "extraction": {"method": "pypdfium2 text layer", "builder": BUILDER_VERSION,
                           "human_checked": False},
            "facets": facets,
            "facet_method": "keyword_rule_v1",
            "defect_classes": defects,
            "relevance": tier,
            "lineage_group_id": spec.lineage_group_id,
            "scope": scope_of(spec, sec.part),
            "rights_status": "pending",
            "use_check": {"qa_generation": "pending", "rag_index": "pending", "training_text": "pending",
                          "basis": "이용 조건(L1) · 유료 계보(L2) 결정 대기"},
            "human_review": None,
            "flags": flags,
        }
        bad = check_row_shape(row)
        if bad:
            raise ValueError(f"행 꼴 위반 {pid}: {bad}")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- 쪽 화면과 대조해 옮긴 구절


@dataclass(frozen=True)
class CuratedSpec:
    doc_id: str
    title: str
    edition: str
    sha256: str                                 # 원문 PDF 를 다시 계산한 값
    lang: str
    lineage_group_id: str
    scope: str
    doc_flags: tuple[str, ...] = ()


CURATED: dict[str, CuratedSpec] = {
    "US-WAR-DEPT-TM-1-430-1942": CuratedSpec(
        doc_id="US-WAR-DEPT-TM-1-430-1942",
        title="War Department TM 1-430 Welding",
        edition="1942-04-24 판, 1943-03-06 Change 1 포함(스캔본)",
        sha256="7742fb0bde76c3f94d03b46d242b1bd3705250c9fffdbb9d3ac6039e096978ee",
        lang="en",
        lineage_group_id="LG-US-TM-1-430-1942",
        scope="1942년 미 육군 항공대 용접 교범 — 역사적 명칭 · 개념만. 현대 원인 · 예방 · 보수 · 합격 기준에 쓰지 않는다",
        doc_flags=("historical_only",),
    ),
    "MDPI-APPLSCI-2020-10-1878": CuratedSpec(
        doc_id="MDPI-APPLSCI-2020-10-1878",
        title="Review on Computer Aided Weld Defect Detection from Radiography Images (Applied Sciences 2020)",
        edition="Applied Sciences 10(5):1878, 출판본 PDF",
        sha256="2ff799fa9fdea0690f739524b4dbcd1a4153947ba276068295be0977a3a1288b",
        lang="en",
        lineage_group_id="LG-MDPI-APPLSCI-2020-1878",
        scope="RT 영상 용접 결함 자동검출 연구의 리뷰(저자 서술) — 합격 기준 · 영상별 정답이 아니다",
        doc_flags=("cc_by_4_0_attribution_required",),
    ),
}


def _plain(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).replace(SOFT_HYPHEN_MARK, "").replace(SOFT_HYPHEN, "")
    s = re.sub(r"-\s*\n\s*", "", s)
    return re.sub(r"\s+", " ", s).strip()


def text_layer_check(passage: str, page_text: str) -> str:
    """옮긴 구절이 그 쪽의 텍스트 층에 있는가 — exact(정규화 후 그대로) · alnum(글자 · 숫자만 같음) · failed."""
    if _plain(passage) in _plain(page_text):
        return "exact"
    squash = lambda s: re.sub(r"[^0-9a-z]", "", _plain(s).lower())
    return "alnum" if squash(passage) in squash(page_text) else "failed"


def curated_rows(drafts: Sequence[dict], pages_by_doc: dict[str, list[str]], used_ids: set[str],
                 excluded: Counter) -> tuple[list[dict], Counter]:
    """쪽 화면과 대조해 옮긴 구절 → 은행 행. 원문 지문 · 쪽 · 절은 초안을 따르고, 텍스트 층과 다시 대조한다."""
    rows: list[dict] = []
    checks: Counter = Counter()
    for d in drafts:
        spec = CURATED.get(d.get("source_id"))
        if spec is None:
            raise ValueError(f"등록되지 않은 원문의 구절: {d.get('source_id')}")
        page = d["pdf_page"]
        if type(page) is not int or not 1 <= page <= len(pages_by_doc[spec.doc_id]):
            raise ValueError(f"쪽 번호가 원문 범위 밖이다: {d.get('passage_id')}")
        text = d["passage"]
        check = text_layer_check(text, pages_by_doc[spec.doc_id][page - 1])
        checks[check] += 1
        if any(paid_sentence_reason(s, spec.doc_id) for s in split_sentences(text)):
            excluded["curated:paid_sentence"] += 1
            continue
        facets, defects, tier = classify(text)
        flags = list(spec.doc_flags)
        if "historical_only" in flags:
            facets = [f for f in facets if f not in MODERN_ONLY_FACETS]
        if check == "failed":
            flags.append("text_layer_cross_check_failed")
        elif check == "alnum":
            flags.append("text_layer_match_alnum_only")
        pid = f"{spec.doc_id}:p{page:03d}:{d['passage_id']}"
        if pid in used_ids:
            raise ValueError(f"구절 id 가 겹친다: {pid}")
        used_ids.add(pid)
        row = {
            "schema_version": SCHEMA_VERSION,
            "passage_id": pid,
            "doc_id": spec.doc_id,
            "registry": "corpus/parse/sources.yaml",
            "edition": spec.edition,
            "source_file_sha256": spec.sha256,
            "locator": {
                "pdf_page_start": page, "pdf_page_end": page,
                "printed_page_start": d.get("printed_page"), "printed_page_end": d.get("printed_page"),
                "part": None, "section_id": d.get("section"), "section_kind": "curated",
                "carried_heading_ids": [],
            },
            "language": spec.lang,
            "text_kind": "verbatim_extract",
            "text": text,
            "text_sha256": text_sha256(text),
            "text_chars": len(text),
            "extraction": {"method": "transcribed from rendered page", "builder": BUILDER_VERSION,
                           "text_layer_check": check, "human_checked": False},
            "draft_passage_id": d["passage_id"],
            "use_boundary": d.get("use_boundary"),
            "facets": facets,
            "facet_method": "keyword_rule_v1",
            "defect_classes": defects,
            "relevance": tier or "curated_context",
            "lineage_group_id": spec.lineage_group_id,
            "scope": spec.scope,
            "rights_status": "pending",
            "use_check": {"qa_generation": "pending", "rag_index": "pending", "training_text": "pending",
                          "basis": "이용 조건 결정 대기"},
            "human_review": None,
            "flags": flags,
        }
        bad = check_row_shape(row)
        if bad:
            raise ValueError(f"행 꼴 위반 {pid}: {bad}")
        rows.append(row)
    return rows, checks


# --------------------------------------------------------------------------- 납품된 은행 행 합치기


#: 학술지 PDF 의 쪽 머리글("Metals 2020, 10, 1121 6 of 14") — 쪽을 넘는 구절의 대조에서만 뺀다.
_RUNNING_HEAD = re.compile(r"(?m)^\s*[A-Z][A-Za-z]+ \d{4}, \d+, \d+\s+\d+ of \d+\s*$")


def strip_running_heads(page_text: str) -> str:
    return _RUNNING_HEAD.sub(" ", page_text)


def xml_body_text(xml_path: Path) -> str:
    """출판사 JATS XML 의 본문(body) 글 — 원문 대조용. 요소 사이는 한 칸으로 잇는다."""
    import xml.etree.ElementTree as ET

    root = ET.parse(xml_path).getroot()
    body = next((el for el in root.iter() if str(el.tag).rsplit("}", 1)[-1] == "body"), root)
    return " ".join(" ".join(body.itertext()).split())


def merged_rows(rows_in: Sequence[dict], *, input_sha256: str, pdf_paths: dict[str, Path],
                xml_paths: dict[str, Path], registry: dict[str, dict], used_ids: set[str],
                seen_text: set[str], excluded: Counter) -> tuple[list[dict], Counter]:
    """이미 만들어진 은행 행(같은 행 꼴 2판)을 합친다. 본문은 바꾸지 않는다.

    행마다 확인한다: 등록부에 있고 등록 sha256 · 행의 원문 지문 · 실제 파일이 같다, 본문 지문이 맞다,
    본문이 원문에 있다(XML 추출 행은 그 XML 의 본문, PDF 추출 행은 그 쪽 범위의 텍스트 층),
    id · 본문이 기존 은행과 겹치지 않는다. 유료 표준 문장 규칙은 본문을 덜어 내지 않고 결과만 센다 —
    걸린 행은 소비 경계에서 걸린다.
    """
    out: list[dict] = []
    checks: Counter = Counter()
    page_cache: dict[str, list[str]] = {}
    xml_cache: dict[str, str] = {}
    for r0 in rows_in:
        r = json.loads(json.dumps(r0, ensure_ascii=False))
        doc = r.get("doc_id")
        entry = registry.get(doc)
        if entry is None or doc not in pdf_paths:
            raise ValueError(f"등록되지 않았거나 원문이 주어지지 않은 문서: {doc}")
        if not (entry["sha256"] == r.get("source_file_sha256") == sha256_file(pdf_paths[doc])):
            raise ValueError(f"{r.get('passage_id')}: 등록부 · 행 · 원문 파일의 sha256 이 다르다")
        if text_sha256(r.get("text") or "") != r.get("text_sha256"):
            raise ValueError(f"{r.get('passage_id')}: 본문 지문이 맞지 않는다")
        ext = r.get("extraction") or {}
        if ext.get("source_xml"):
            if doc not in xml_paths or sha256_file(xml_paths[doc]) != ext.get("source_xml_sha256") \
                    or entry.get("xml_sha256") != ext.get("source_xml_sha256"):
                raise ValueError(f"{r['passage_id']}: XML 원문 지문이 등록부 · 행과 다르다")
            if doc not in xml_cache:
                xml_cache[doc] = xml_body_text(xml_paths[doc])
            check = "xml_" + text_layer_check(r["text"], xml_cache[doc])
        else:
            if doc not in page_cache:
                page_cache[doc] = extract_pages(pdf_paths[doc])
            loc = r.get("locator") or {}
            a = ext.get("pdf_page_start") or loc.get("pdf_page_start")
            b = ext.get("pdf_page_end") or loc.get("pdf_page_end")
            span = " ".join(strip_running_heads(t) for t in page_cache[doc][a - 1:b])
            check = "pdf_" + text_layer_check(r["text"], span)
        checks[check] += 1
        if r["passage_id"] in used_ids:
            raise ValueError(f"구절 id 가 기존 은행과 겹친다: {r['passage_id']}")
        if r["text_sha256"] in seen_text:
            excluded["merged:duplicate_text"] += 1
            continue
        if any(paid_sentence_reason(s, doc) for s in split_sentences(r["text"])):
            excluded["merged:paid_sentence_rows_kept_for_gate"] += 1
        r["extraction"] = {**ext, "bank_v3_source_check": check}
        if check.endswith("failed"):
            r["flags"] = [*r.get("flags", []), "text_layer_cross_check_failed"]
        r["merged_from"] = {"bank_sha256": input_sha256, "registry_field": r.get("registry")}
        r["registry"] = "corpus/parse/sources.yaml"
        bad = check_row_shape(r)
        if bad:
            raise ValueError(f"행 꼴 위반 {r['passage_id']}: {bad}")
        used_ids.add(r["passage_id"])
        seen_text.add(r["text_sha256"])
        out.append(r)
    return out, checks


# --------------------------------------------------------------------------- 실행


def extract_pages(pdf_path: Path) -> list[str]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    return [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="구절 은행 빌드")
    ap.add_argument("--source", action="append", required=True, help="DOC_ID=원문 PDF 경로")
    ap.add_argument("--out-full", required=True, type=Path)
    ap.add_argument("--out-pointer", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--curated", type=Path, default=None, help="쪽 화면과 대조해 옮긴 구절 JSONL")
    ap.add_argument("--merge-rows", type=Path, default=None, help="합칠 은행 행 JSONL(본문 판, 행 꼴 2판)")
    ap.add_argument("--merge-source", action="append", default=[], help="DOC_ID=합칠 행의 원문 PDF")
    ap.add_argument("--merge-xml", action="append", default=[], help="DOC_ID=합칠 행의 원문 XML")
    ap.add_argument("--registry", type=Path, default=Path("corpus/parse/sources.yaml"))
    a = ap.parse_args(argv)

    known = set(DOCS) | set(CURATED)
    paths: dict[str, Path] = {}
    for s in a.source:
        doc, _, p = s.partition("=")
        if doc not in known:
            raise SystemExit(f"모르는 문서: {doc}")
        paths[doc] = Path(p)
    need = set(DOCS) | (set(CURATED) if a.curated else set())
    missing = sorted(need - set(paths))
    if missing:
        raise SystemExit(f"원문이 빠졌다: {missing}")

    import pypdfium2 as pdfium

    all_rows: list[dict] = []
    report: dict = {"schema_version": SCHEMA_VERSION, "builder": BUILDER_VERSION,
                    "pypdfium2_version": str(getattr(pdfium, "PYPDFIUM_INFO", "")),
                    "pdfium_version": str(getattr(pdfium, "PDFIUM_INFO", "")),
                    "max_chars": MAX_CHARS, "min_chars": MIN_CHARS,
                    "max_chars_effective": MAX_CHARS + MIN_CHARS, "docs": {}}
    for doc_id, spec in DOCS.items():
        got = sha256_file(paths[doc_id])
        if got != spec.sha256:
            raise SystemExit(f"{doc_id}: 원문 sha256 이 다르다 — 기대 {spec.sha256}, 실제 {got}")
        pages = extract_pages(paths[doc_id])
        stats: Counter = Counter()
        rows, excluded = build_doc(spec, pages, stats)
        all_rows.extend(rows)
        report["docs"][doc_id] = {
            "edition": spec.edition, "source_file_sha256": spec.sha256, "physical_pages": len(pages),
            "pages_scanned": sum(b - a + 1 for a, b in spec.pages), "page_ranges": [list(r) for r in spec.pages],
            "lineage_group_id": spec.lineage_group_id,
            "passages": len(rows),
            "relevance": dict(sorted(Counter(r["relevance"] for r in rows).items())),
            "table_passages": sum("table_text_layer" in r["flags"] for r in rows),
            "excluded": dict(sorted(excluded.items())),
            "char_stats": dict(sorted(stats.items())),
        }
    if a.curated:
        pages_by_doc = {}
        for doc_id, spec in CURATED.items():
            got = sha256_file(paths[doc_id])
            if got != spec.sha256:
                raise SystemExit(f"{doc_id}: 원문 sha256 이 다르다 — 기대 {spec.sha256}, 실제 {got}")
            pages_by_doc[doc_id] = extract_pages(paths[doc_id])
        drafts = load_jsonl(a.curated)
        excluded = Counter()
        rows, checks = curated_rows(drafts, pages_by_doc, {r["passage_id"] for r in all_rows}, excluded)
        all_rows.extend(rows)
        report["curated"] = {
            "input_sha256": sha256_file(a.curated), "input_rows": len(drafts), "passages": len(rows),
            "text_layer_check": dict(sorted(checks.items())), "excluded": dict(sorted(excluded.items())),
            "docs": {d: {"edition": s.edition, "source_file_sha256": s.sha256, "physical_pages": len(pages_by_doc[d]),
                         "lineage_group_id": s.lineage_group_id,
                         "passages": sum(r["doc_id"] == d for r in rows)} for d, s in CURATED.items()},
        }
    # 문서 사이에서 같은 본문이 되풀이되면(ABS Pub 87 이 옛 Rec.47 을 옮긴 문장 등) 앞 문서의 것만 남긴다.
    seen: set[str] = set()
    unique: list[dict] = []
    cross_dup = 0
    for r in all_rows:
        if r["text_sha256"] in seen:
            cross_dup += 1
            continue
        seen.add(r["text_sha256"])
        unique.append(r)
    all_rows = unique
    report["cross_doc_duplicate_text_dropped"] = cross_dup
    if a.merge_rows:
        from corpus.parse.passage_bank import load_registry

        pdfs = {d: Path(p) for d, _, p in (s.partition("=") for s in a.merge_source)}
        xmls = {d: Path(p) for d, _, p in (s.partition("=") for s in a.merge_xml)}
        excluded = Counter()
        rows_in = load_jsonl(a.merge_rows)
        merged, checks = merged_rows(rows_in, input_sha256=sha256_file(a.merge_rows), pdf_paths=pdfs,
                                     xml_paths=xmls, registry=dict(load_registry(a.registry)),
                                     used_ids={r["passage_id"] for r in all_rows}, seen_text=seen, excluded=excluded)
        all_rows.extend(merged)
        report["merged"] = {
            "input_sha256": sha256_file(a.merge_rows), "input_rows": len(rows_in), "passages": len(merged),
            "source_check": dict(sorted(checks.items())), "excluded": dict(sorted(excluded.items())),
            "docs": dict(sorted(Counter(r["doc_id"] for r in merged).items())),
            "rows_with_required_context": sum(bool(r.get("required_context_passage_ids")) for r in merged),
            "rows_with_source_policy_holds": sum(bool(r.get("source_policy_holds")) for r in merged),
            "rows_with_context_dependency_holds": sum(bool(r.get("context_dependency_policy_holds")) for r in merged),
        }
    all_rows.sort(key=lambda r: r["passage_id"])
    full_sha = dump_jsonl(all_rows, a.out_full)
    ptr_sha = dump_jsonl([pointer_row(r) for r in all_rows], a.out_pointer)
    report["outputs"] = {"full_bank_sha256": full_sha, "pointer_bank_sha256": ptr_sha, "rows": len(all_rows)}
    report["facets"] = dict(sorted(Counter(f for r in all_rows for f in r["facets"]).items()))
    report["defects"] = dict(sorted(Counter(d for r in all_rows for d in r["defect_classes"]).items()))
    report["relevance"] = dict(sorted(Counter(r["relevance"] for r in all_rows).items()))
    report["lineage_groups"] = dict(sorted(Counter(r["lineage_group_id"] for r in all_rows).items()))
    report["flags"] = dict(sorted(Counter(f.split(":")[0] for r in all_rows for f in r["flags"]).items()))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_bytes((json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps({"rows": len(all_rows), "full_bank_sha256": full_sha, "pointer_bank_sha256": ptr_sha},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

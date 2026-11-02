"""검사기 다음 판(v4)의 B 회귀 기대 · 독립 대조기 N1 ~ N7 의 적용 조건 — 사례 파일의 꼴과 출처(07 4판 §3-2 · §3-3 · §16).

이 사례 파일은 v4 구현 전에 고정한 회귀 기대다(독립 확인 사례가 아니다). v4 는 이 묶음 밖이라 여기서는 검사기를 부르지 않는다 — 고정한
바이트, 문장 계약이 08 의 글자와 같은지, 합성 수치가 1 판 표의 수가 아닌지, 레코드의 기대가 스스로 맞는지만 본다.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASES_REL = "evaluation/vectors/narrative_b_regression.v1.json"
CASES_SHA256 = "d75253aedf68747dd478d02fe7ba7cd8d15fc4dc446b17aaaa02d309593e04c6"
"""구현 전 고정 커밋(91e83979)의 바이트."""
CASES = json.loads((REPO / CASES_REL).read_bytes().decode("utf-8"))
C08 = REPO / "docs/dev_log/2026-10-20-설계복구/08_허용치표_부분B_미니스펙_C.md"
D07 = REPO / "docs/dev_log/2026-10-20-설계복구/07_허용치표_부분B_미니스펙_D.md"
KINDS = {"가", "나", "다", "라", "마"}
SUBLABELS = {"기준 오기(측정량)", "군집 · 간격 단정", "적용 전제 단정", "조건부 대응의 단정"}


def test_사례_파일은_구현_전에_고정한_바이트다() -> None:
    assert hashlib.sha256((REPO / CASES_REL).read_bytes()).hexdigest() == CASES_SHA256
    assert CASES["kind"] == "narrative_b_regression_expectations" and "독립 확인 사례" in CASES["status"]


def test_문장_계약은_08_의_글자와_같다() -> None:
    if not C08.exists():
        pytest.skip("문서 폴더가 없는 체크아웃")
    text = C08.read_text(encoding="utf-8")
    for key, sentence in CASES["contract_sentences"].items():
        assert f"| `{sentence}` |" in text, key


def test_씨앗_문구는_07_과_08_에_같은_글자로_있다() -> None:
    if not (C08.exists() and D07.exists()):
        pytest.skip("문서 폴더가 없는 체크아웃")
    c08, d07 = C08.read_text(encoding="utf-8"), D07.read_text(encoding="utf-8")
    for group, seeds in CASES["n7_seeds"].items():
        assert seeds, group
        for s in seeds:
            assert f"'{s}'" in c08 and f"'{s}'" in d07, s


def test_합성_수치는_1_판_표의_수가_아니다() -> None:
    real = set(re.findall(r"\d+(?:\.\d+)?", (REPO / "corpus/rules/limits_v0_pilot.csv").read_text(encoding="utf-8")))
    assert not set(CASES["synthetic"]["numbers"]) & real
    assert CASES["synthetic"]["band"] in " ".join(c["text"] for c in CASES["cases"])


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda c: c["id"])
def test_사례마다_기대의_꼴이_닫혀_있다(case: dict) -> None:
    e = case["expect"]
    assert set(e["kinds"]) <= KINDS and set(e["sublabels"]) <= SUBLABELS
    assert set(e) <= {"kinds", "sublabels", "anchor", "list_allowed"}
    assert case["context"]["material"] in {"ST", "AL"} and case["text"].strip()
    if e["sublabels"]:
        assert e["kinds"], "세부 표지는 종류 안의 표지다"


def test_조건부_소개와_부정은_금지가_아니고_단정은_금지다() -> None:
    by_cat: dict[str, list[dict]] = {}
    for c in CASES["cases"]:
        by_cat.setdefault(c["category"].split("(")[0], []).append(c)
    assert all(not c["expect"]["kinds"] for c in by_cat["조건부 소개"] + by_cat["부정"] + by_cat["음성"])
    assert all(c["expect"]["kinds"] for c in by_cat["단정"])


def test_레코드의_어긋날_항목은_그_레코드의_문장_수에서_난다() -> None:
    n_of = {"N1": "C3a", "N2": "C5", "N3": "C6", "N4": "C7", "N5": "C8", "N6": "C2"}
    for item in CASES["n_items"]:
        counts = item["v2_sentence_counts"]
        assert item["v1_mismatch"] == {n: counts[c] > 0 for n, c in n_of.items()}, item["record"]
        defects = bool(item["record"]["codes"])
        assert (counts["C13"] == 1) == (not defects) and (counts["C1"] == 1) == defects

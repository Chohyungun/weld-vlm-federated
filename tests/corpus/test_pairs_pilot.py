"""D4 페어 축소본 생성기 회귀 방지 — 74번 감사 P4·P6 소급 수정분.

- P6: 정상 페어의 결함 어휘 락이 정본(`numeric_lock.find_defect_tokens`)을 통과한다.
  재구현판은 사상표의 **영문 명칭**과 전각 표기를 놓쳤다.
- P4: 재질 축이 fail-closed 다. 허용치 CSV 가 덮지 않는 재질이 표본에 있으면 조용히
  다른 재질의 조항을 붙이지 않고 멈춘다.
"""

from __future__ import annotations

import pytest

from corpus.generate import make_pairs_pilot as M
from corpus.rules import limits_loader
from corpus.rules.skeleton_gen import load_defect_lexicon

LEX = load_defect_lexicon()
NAMES = {"100": "균열", "2011": "기공", "301": "슬래그혼입", "401": "융합불량"}


def _normal(text: str) -> dict:
    return {
        "image_id": "x:1", "image_path": "a.jpg", "client": "C1", "split": "train",
        "skeleton": {"defects": [], "verdict": None, "verdict_mode": "clause_only",
                     "clauses": []},
        "target_text": text,
    }


def test_동결본_정상문은_정본_락을_통과한다():
    """락을 정본으로 바꿔도 v1 산출물의 판정은 바뀌지 않는다 (2,625건 재검증 불필요)."""
    assert M.check_pair(_normal(M.NORMAL_TEXT), NAMES, set(), {}, (100, 100), LEX) == []


@pytest.mark.parametrize("text", [
    "방사선투과 영상에서 porosity 가 관찰되지 않는다.",   # 영문 명칭 — 재구현판이 놓쳤다
    "방사선투과 영상에서 Crack 은 없다.",                 # 대소문자 불문
    "코드 ２０１１ 에 해당하는 지시가 없다.",              # 전각 숫자 — NFKC 정규화
    "기공은 관찰되지 않는다.",                            # 부정 문맥도 폐기 (§4-6-1 ③)
])
def test_정상페어_결함어휘는_문맥·표기_불문_폐기(text):
    assert "defect_word_in_normal" in M.check_pair(
        _normal(text), NAMES, set(), {}, (100, 100), LEX)


def test_재구현판이_놓쳤던_영문명칭이_사전에_있다():
    """정본 사전은 사상표의 name_en 까지 싣는다 — 재구현판은 코드·한국어 명칭뿐이었다."""
    assert "Porosity" in LEX and "Crack" in LEX
    assert "2012" in LEX          # 기공 alt 코드


def test_파일럿_허용치표는_강재만_덮는다():
    """알루미늄 허용치 근거(KS-AL)는 미확보다 — sources.yaml status=pending."""
    table = limits_loader.load_limits(str(M.LIMITS_CSV), pilot=True)
    assert M.covered_materials(table) == {"ST"}


def test_clause_basis_는_재질을_보지_않는다():
    """P4 의 실체. 이 사실이 재질 축 가드의 존재 이유다 — 가드를 지우면 다시 샌다."""
    table = limits_loader.load_limits(str(M.LIMITS_CSV), pilot=True)
    basis = M.clause_basis(table)
    # 강재 전용 조항이 코드만으로 잡힌다. 알루미늄 이미지에도 그대로 붙는다.
    assert basis["2011"]["clause_id"] == "KRA27-T15"
    assert all("재질" not in c for b in basis.values() for c in b["criteria"])

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


def test_clause_basis_가_재질_축을_본다():
    """P4 의 정정. v1 은 재질을 안 보고 알루미늄에 강재 조항을 붙였다 (219건 전량)."""
    table = limits_loader.load_limits(str(M.LIMITS_CSV), pilot=True)
    st = M.clause_basis(table, "ST")
    al = M.clause_basis(table, "AL")
    assert st["2011"]["clause_id"] == "KRA27-T15"
    assert al == {}, "알루미늄을 덮는 허용치 행이 없는데 조항이 잡혔다"


def test_덮지_않는_재질은_조항을_특정하지_않는다():
    """격리(안 A)하면 C3 결함 페어가 0건이 되어 RQ3 이 인위적으로 바뀐다 — 서술로 닫는다."""
    defects = [{"type": "2011", "bbox_px": [0, 0, 10, 10], "size_px": {}, "size_mm": None}]
    text = M.defect_target_text(defects, NAMES, {})
    assert "적용 조항을 특정하지 않는다" in text
    assert "KRA27" not in text
    assert "기공(ISO 6520-1 코드 2011) 1개" in text


def test_합계길이_조항은_미표현_단서를_부기한다():
    """KRA27-T16 은 합계 길이 기준인데 개별 결함 한계로 서술돼 286건에 실렸다 (B12)."""
    table = limits_loader.load_limits(str(M.LIMITS_CSV), pilot=True)
    base = M.clause_basis(table, "ST")
    defects = [{"type": "301", "bbox_px": [0, 0, 10, 10], "size_px": {}, "size_mm": None}]
    text = M.defect_target_text(defects, NAMES, base)
    assert "KRA27-T16" in text
    assert "기계 표현으로 옮기지 못한 단서" in text


def test_v1_회계가_실측과_맞는다():
    """219 + 286 − 84 = 421. 산출물이 이 숫자를 들고 다녀야 RQ2·RQ3 을 그 위에서 읽는다."""
    acc = M.v1_accounting()
    kinds = {k["kind"]: k["n"] for k in acc["kinds"]}
    assert kinds == {"material_axis": 219, "aggregate_length": 286}
    assert acc["n_defective_citations"] == 421
    assert round(421 / acc["n_defect_pairs"], 4) == acc["share_of_defect_pairs"]


def test_v1_은_덮어쓸_수_없다(monkeypatch):
    """규약 1-6 — 동결본 경로로 재생성하면 회계가 가리키는 실물이 사라진다."""
    monkeypatch.setattr(
        "sys.argv",
        ["make_pairs_pilot", "--out", str(M.FROZEN_V1)],
    )
    with pytest.raises(SystemExit) as e:
        M.main()
    assert "동결" in str(e.value)

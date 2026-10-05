"""D4 페어 — 검증 게이트 ③의 독립 경로, 조립 함수, 빌드 회계·격리·결정론 (06번 §5).

파일럿에서 실제로 잡힌 결함과, 06번의 검토가 직접 재현한 반례를 회귀 시험으로 고정한다.

- 게이트 ③은 자기참조였다. 기준 문장을 지우고 두께 구간 머리만 남긴 서술이 통과했고, AL 레코드에 강재 조항을
  달아도 통과했다. 지금은 서술을 **파싱해 허용치 행과 등치**로 보고, 조항 집합을 (재질, 코드)로 다시 계산한다.
- 폐기 회계가 사유 수를 셌다 — 한 레코드가 사유 둘을 내면 `counts` 정합이 파일을 쓴 뒤에 죽었다.
- 균열 페어 전부에 "…허용하지 않는다이다" 가 나갔고, 서술이 스스로 부딪혔다.

**봉인 자산을 읽지도 쓰지도 않는다.** 입력은 전부 이 파일이 만든 작은 표이고 출력은 `tmp_path` 다.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from corpus.generate import make_pairs_pilot as M
from corpus.rules import limits_loader
from corpus.rules.clause_text import criterion_ko, thickness_ko
from corpus.rules.skeleton_gen import load_defect_lexicon
from corpus.validate import pair_criteria_gate as G

LEX = load_defect_lexicon()
NAMES = {"100": "균열", "2011": "기공", "301": "슬래그혼입", "401": "융합불량"}
TABLE = limits_loader.load_limits(str(M.LIMITS_CSV), pilot=True)
RULES = G.PairRules(TABLE.rows, NAMES)
W, H = 1280, 720


# ------------------------------------------------------------------ 픽스처

def _table(edit=None, *, keep=None):
    """파일럿 표의 사본. `edit(rule_id) -> dict|None` 로 행을 바꾸고 `keep(row)` 로 행을 거른다."""
    rows = []
    for r in TABLE.rows:
        if keep is not None and not keep(r):
            continue
        upd = edit(r.rule_id) if edit else None
        rows.append(r.model_copy(update=upd) if upd else r)
    return SimpleNamespace(rows=rows)


def _row(iid="img-0001", *, material="ST", has_defect=True, n_defects=1, split="train",
         client="C1", group_id="g-0001"):
    return {"image_id": iid, "group_id": group_id, "rel_path": f"tiles/{iid}.png", "client": client,
            "split": split, "material": material, "has_defect": has_defect, "n_defects": n_defects,
            "width_px": W, "height_px": H}


def _ann(iid, k, code, box=(10, 10, 50, 50), geom_valid=True):
    return {"image_id": iid, "ann_id": f"{iid}#{k}", "geom_valid": geom_valid, "iso_code": code,
            "bbox_x1_px": box[0], "bbox_y1_px": box[1], "bbox_x2_px": box[2], "bbox_y2_px": box[3],
            "major_axis_px": 40.0, "equiv_diameter_px": 30.0}


def _rec(codes, material="ST", table=TABLE):
    anns = [_ann("img-0001", k, c) for k, c in enumerate(codes)]
    rec, skipped = M.assemble_record(_row(material=material, n_defects=len(codes), has_defect=bool(codes)),
                                     anns, NAMES, M.clause_basis(table, material))
    assert skipped == 0
    return rec


def _check(rec, rules=RULES, **kw):
    return M.check_pair(rec, NAMES, rules, LEX, **kw)


COMBOS = [(["2011"], "ST"), (["100"], "ST"), (["301", "401"], "ST"), (["2011", "100", "301", "2011"], "ST"),
          (["2011"], "AL"), (["100", "401"], "AL"), ([], "ST"), ([], "AL")]


# ------------------------------------------------------------------ 관점 A · 부등식과 구간

@pytest.mark.parametrize("codes, material", COMBOS)
def test_조립한_레코드는_게이트를_통과한다(codes, material):
    assert _check(_rec(codes, material), wh=(W, H)) == []


def test_기준_문장을_지우고_구간_머리만_남긴_서술은_폐기된다():
    """06번 §2-나의 반례. 옛 게이트는 "5 mm 이하" 를 "25 mm 이하" 안에서 찾아 통과시켰다."""
    rec = _rec(["2011"])
    heads = ("두께 10 mm 이하 / 두께 10 mm 초과 25 mm 이하 / 두께 25 mm 초과 50 mm 이하"
             " / 두께 50 mm 초과 100 mm 이하")
    start = rec["target_text"].index("이 조항의 기준은 다음과 같다. ") + len("이 조항의 기준은 다음과 같다. ")
    end = rec["target_text"].index(". 이 조항에는")
    assert "4 mm 이하" in rec["target_text"][start:end]
    rec["target_text"] = rec["target_text"][:start] + heads + rec["target_text"][end:]
    assert "text_unparsable" in _check(rec)


def test_기준_문장을_통째로_지운_서술은_폐기된다():
    rec = _rec(["2011"])
    a = rec["target_text"].index(" 이 조항의 기준은")
    b = rec["target_text"].index(" 이 조항에는")
    rec["target_text"] = rec["target_text"][:a] + rec["target_text"][b:]
    assert _check(rec) != []


def test_AL_레코드에_강재_조항을_달면_레코드_수준에서_폐기된다():
    """파일럿 v1 의 219건 양식. 옛 `check_pair` 는 재질을 받지 않았고 유효 조항 집합이 재질 합집합이었다."""
    rec = _rec(["2011"], "AL")
    rec["skeleton"]["clauses"] = ["KRA27-T15"]
    assert "clause_set_mismatch" in _check(rec)
    # 서술까지 강재의 것으로 바꿔 끼워도 같다 — 조항·기준·맺음이 전부 (재질, 코드)의 기대값과 어긋난다
    rec["target_text"] = _rec(["2011"], "ST")["target_text"]
    bad = _check(rec)
    assert "clause_set_mismatch" in bad and "clause_text_mismatch" in bad


def test_강재_레코드에서_조항을_빼도_폐기된다():
    rec = _rec(["2011"])
    rec["skeleton"]["clauses"] = []
    assert "clause_set_mismatch" in _check(rec)


MUTATIONS = {
    "부등호 le→lt": {"limit_op": TABLE.rows[0].limit_op.__class__("lt")},
    "값 한 칸": {"limit_value": TABLE.rows[0].limit_value + 1},
    # 구간을 **줄인다.** 늘리면 이웃 행과 겹쳐서 표 자체가 거부된다 — 그것은 아래 겹침 시험이 따로 본다.
    "두께 상한": {"thickness_max": TABLE.rows[0].thickness_max - 2},
}


@pytest.mark.parametrize("label", sorted(MUTATIONS))
def test_허용치_행이_바뀌면_옛_서술은_어긋나고_새_서술은_맞는다(label):
    """기대값이 **행에서** 온다 — 생성기가 만든 문장에서 오지 않는다."""
    changed = _table(lambda rid: MUTATIONS[label] if rid == "KRA27-T15-01" else None)
    new_rules = G.PairRules(changed.rows, NAMES)
    assert "criterion_mismatch" in _check(_rec(["2011"]), new_rules)       # 옛 표의 서술 × 새 표의 기대값
    assert _check(_rec(["2011"], table=changed), new_rules) == []           # 새 표의 서술 × 새 표의 기대값
    assert "criterion_mismatch" in _check(_rec(["2011"], table=changed))    # 새 표의 서술 × 옛 표의 기대값


def test_lt_행은_미만으로_쓰인다():
    changed = _table(lambda rid: MUTATIONS["부등호 le→lt"] if rid == "KRA27-T15-01" else None)
    assert "두께 10 mm 이하: 4 mm 미만" in _rec(["2011"], table=changed)["target_text"]


def test_두께_하한과_비례_계수가_바뀌어도_잡힌다():
    for rid, upd in (("KRA27-T15-02", {"thickness_min": TABLE.rows[1].thickness_min + 1}),
                     ("KRA27-T15-03", {"limit_factor": TABLE.rows[2].limit_factor * 2}),
                     ("KRA27-T15-03", {"ratio_basis": TABLE.rows[2].ratio_basis.__class__("s")})):
        changed = _table(lambda r, rid=rid, upd=upd: upd if r == rid else None)
        assert "criterion_mismatch" in _check(_rec(["2011"]), G.PairRules(changed.rows, NAMES)), (rid, upd)


def test_격자_밖_구간은_이상과_미만으로_쓰이고_옛_표기는_폐기된다():
    """`[10, 25)` 는 "10 mm 이상 25 mm 미만" 이다. 예전 문장화는 하한을 늘 "초과", 상한을 늘 "이하" 로 썼다."""
    from decimal import Decimal

    assert thickness_ko(Decimal("10.00"), Decimal("25.00")) == "10 mm 이상 25 mm 미만"
    assert thickness_ko(Decimal("10.01"), Decimal("25.01")) == "10 mm 초과 25 mm 이하"      # 부호화된 경계는 그대로
    assert thickness_ko(Decimal("0.00"), Decimal("25.00")) == "25 mm 미만"
    # 앞 행의 상한도 함께 내린다. 한쪽만 바꾸면 [0.00, 10.01) 과 [10.00, …) 이 겹쳐 표가 거부된다.
    changed = _table(lambda rid: {"thickness_max": Decimal("10.00")} if rid == "KRA27-T15-01" else
                     {"thickness_min": Decimal("10.00"), "thickness_max": Decimal("25.00")}
                     if rid == "KRA27-T15-02" else None)
    rules = G.PairRules(changed.rows, NAMES)
    rec = _rec(["2011"], table=changed)
    assert "두께 10 mm 이상 25 mm 미만: 5 mm 이하" in rec["target_text"]
    assert _check(rec, rules) == []
    rec["target_text"] = rec["target_text"].replace("두께 10 mm 이상 25 mm 미만", "두께 10 mm 초과 25 mm 이하")
    assert "criterion_mismatch" in _check(rec, rules)


def test_모르는_부등호와_비례_분모는_예외다():
    """문장화의 조용한 기본값("이하"·"모재 두께")을 없앴다. 게이트의 독립 경로도 같이 멈춘다."""
    bad_op = TABLE.rows[0].model_copy(update={"limit_op": "ge"})
    bad_basis = TABLE.rows[2].model_copy(update={"ratio_basis": "w"})
    for row in (bad_op, bad_basis):
        with pytest.raises(ValueError):
            criterion_ko(row)
        with pytest.raises(ValueError):
            G.criterion_of_row(row)


def test_비례_상한_한계도_읽고_상한이_바뀌면_잡힌다():
    from corpus.rules.schema import InspectionMethod

    cap = next(r for r in TABLE.rows if r.rule_id == "KRA27-S-01")        # prop_t_cap — 표에서는 VT 행이다
    only = SimpleNamespace(rows=[cap.model_copy(update={"inspection_method": InspectionMethod.RT})])
    rules = G.PairRules(only.rows, NAMES)
    rec = _rec(["2011"], table=only)
    assert "모재 두께의 0.25 배 이하이고 최대 3 mm" in rec["target_text"]
    assert _check(rec, rules) == []
    other = G.PairRules([only.rows[0].model_copy(update={"limit_cap": cap.limit_cap + 1})], NAMES)
    assert "criterion_mismatch" in _check(rec, other)


# ------------------------------------------------------------------ 관점 B · 조합

def test_사상표에_없는_코드는_폐기된다():
    assert "unknown_code" in _check(_rec(["999"]))


def test_재질은_덮이는데_코드의_행이_없으면_그_결함만_미특정으로_쓴다():
    """AL 행이 일부 코드만 확보되면 "이 재질에 적용할 조항이 없어" 는 거짓이 된다 — 문장을 가른다."""
    no301 = _table(keep=lambda r: r.defect_code != "301")
    rec = _rec(["2011", "301"], table=no301)
    assert rec["skeleton"]["clauses"] == ["KRA27-T15"]
    assert rec["skeleton"]["uncited_codes"] == [{"code": "301", "reason": "code_not_covered"}]
    assert "슬래그혼입(ISO 6520-1 코드 301)" + M.NO_CLAUSE_CODE_TAIL in rec["target_text"]
    assert M.NO_CLAUSE_TAIL not in rec["target_text"]
    assert _check(rec, G.PairRules(no301.rows, NAMES)) == []
    assert _check(rec) != []                       # 301 행이 있는 표에서는 어긋난다


def test_재질을_덮는_행이_없으면_재질_단위_문장이고_사유도_재질이다():
    rec = _rec(["100", "401"], "AL")
    assert rec["skeleton"]["clauses"] == [] and rec["skeleton"]["candidate_rules"] == []
    assert rec["skeleton"]["uncited_codes"] == [{"code": "100", "reason": "material_not_covered"},
                                                {"code": "401", "reason": "material_not_covered"}]
    assert M.NO_CLAUSE_TAIL in rec["target_text"] and "KRA27" not in rec["target_text"]


def test_같은_코드가_두_조항에_걸리면_생성도_게이트도_멈춘다():
    rows = [r.model_copy(update={"clause_id": "OTHER-1"}) if r.rule_id == "KRA27-T15-04" else r
            for r in TABLE.rows]
    with pytest.raises(SystemExit):
        M.clause_basis(SimpleNamespace(rows=rows), "ST")
    with pytest.raises(ValueError):
        G.PairRules(rows, NAMES).rows_for("ST", "2011")


def test_정본이_아닌_행과_품질_체계가_걸린_행은_정책이_없어_멈춘다():
    with pytest.raises(ValueError):
        G.PairRules([TABLE.rows[0].model_copy(update={"canonical": False})], NAMES)


def test_301_과_401_이_함께_있으면_같은_조항이_코드마다_서술되고_행은_여섯이_실린다():
    rec = _rec(["301", "401"])
    assert rec["skeleton"]["clauses"] == ["KRA27-T16"]
    assert [r["rule_id"] for r in rec["skeleton"]["candidate_rules"]] == [f"KRA27-T16-0{i}" for i in range(1, 7)]
    assert rec["target_text"].count("에 적용되는 조항은 KRA27-T16 이다.") == 2
    first = rec["skeleton"]["candidate_rules"][0]
    assert first == {"rule_id": "KRA27-T16-01", "clause_id": "KRA27-T16", "defect_code": "301",
                     "thickness_min": "0.00", "thickness_max": "12.01"}
    assert rec["skeleton"]["candidate_rules"][2]["thickness_max"] is None       # 상한 공란


def test_행별_두께_구간은_rule_id_문자열_순이다():
    rec = _rec(["2011", "100"])
    ids = [r["rule_id"] for r in rec["skeleton"]["candidate_rules"]]
    assert ids == sorted(ids) and ids[0] == "KRA27-3D-01"          # 표의 행 순서(T15 가 먼저)가 아니다
    assert rec["skeleton"]["clauses"] == ["KRA27-3D", "KRA27-T15"]


def test_행별_구간이나_미특정_사유를_고치면_폐기된다():
    rec = _rec(["2011"])
    rec["skeleton"]["candidate_rules"][0]["thickness_max"] = "12.01"
    assert "candidate_rules_mismatch" in _check(rec)
    rec = _rec(["2011"], "AL")
    rec["skeleton"]["uncited_codes"] = []
    assert "uncited_codes_mismatch" in _check(rec)


def test_관찰_문장의_개수가_골격과_다르면_폐기된다():
    rec = _rec(["2011", "2011"])
    rec["target_text"] = rec["target_text"].replace("2개", "3개")
    assert "observation_mismatch" in _check(rec)


def test_부기를_빼거나_엉뚱한_조항에_붙이면_폐기된다():
    rec = _rec(["2011"])
    rec["target_text"] = rec["target_text"].replace(" " + M.NOTE_UNEXPRESSED, "")
    assert _check(rec) == ["note_mismatch"]
    rec = _rec(["100"])
    rec["target_text"] = rec["target_text"].replace("허용하지 않는다.", "허용하지 않는다. " + M.NOTE_UNEXPRESSED, 1)
    assert _check(rec) == ["note_mismatch"]


def test_맺음_문장이_기준의_종류와_맞지_않으면_폐기된다():
    rec = _rec(["100"])                                  # 크기와 무관한 기준만 — 치수 문장은 맞지 않는 이유다
    rec["target_text"] = rec["target_text"].replace(M.CLOSE_SIZE_INDEPENDENT, M.CLOSE_DIMENSIONAL)
    assert _check(rec) == ["closing_mismatch"]
    rec = _rec(["2011"])
    rec["target_text"] = rec["target_text"].replace(" " + M.CLOSE_DIMENSIONAL, "")
    assert _check(rec) == ["closing_mismatch"]
    rec = _rec(["2011"])
    rec["target_text"] += " " + M.CLOSE_SIZE_INDEPENDENT
    assert _check(rec) == ["closing_mismatch"]


def test_미특정_문장을_빼거나_덮이는_재질에_붙이면_폐기된다():
    rec = _rec(["2011"], "AL")
    rec["target_text"] = rec["target_text"].replace(" " + M.NO_CLAUSE_TAIL, "")
    assert _check(rec) == ["uncited_text_mismatch"]
    rec = _rec(["2011"])                                 # 강재는 덮인다 — 재질 단위 문장은 거짓이다
    rec["target_text"] = rec["target_text"].replace(" " + M.CLOSE_DIMENSIONAL,
                                                    " " + M.NO_CLAUSE_TAIL + " " + M.CLOSE_DIMENSIONAL)
    assert _check(rec) == ["uncited_text_mismatch"]


# ------------------------------------------------------------------ 균열 (09-21 판정 6)

def test_균열_서술에_비문이_없고_스스로_부딪히지_않는다():
    text = _rec(["100"])["target_text"]
    assert "않는다이다" not in text
    assert "모든 두께: 크기와 무관하게 허용하지 않는다." in text
    assert M.NOTE_UNEXPRESSED not in text              # 전량 불허 행의 note 는 전사 메모다 — 부기를 걸지 않는다
    assert M.CLOSE_DIMENSIONAL not in text             # 크기와 무관한 기준에 "치수 정보가 없어" 는 맞지 않는 이유다
    assert text.endswith(M.CLOSE_SIZE_INDEPENDENT)


def test_균열과_치수_기준이_함께면_맺음이_둘_다_붙는다():
    text = _rec(["2011", "100"])["target_text"]
    assert text.endswith(M.CLOSE_DIMENSIONAL + " " + M.CLOSE_SIZE_INDEPENDENT)
    assert text.count(M.NOTE_UNEXPRESSED) == 1         # 기공 조항에만


def test_판정_함의_목록을_보강해도_균열_페어가_전량_폐기되지_않는다(monkeypatch):
    """목록에 "허용하지" 가 없어 통과하던 것은 우연이었다. 기준 문장은 등치로 닫혔으므로 그 검사에서 뺀다."""
    from corpus.generate import numeric_lock as nl

    def stronger(text):
        return tuple(sorted(set(nl.find_verdict_implying(text)) | ({"허용하지"} if "허용하지" in text else set())))

    monkeypatch.setattr(M, "find_verdict_implying", stronger)
    assert _check(_rec(["100"])) == []
    assert _check(_rec(["2011", "100"])) == []
    # 기준 문장 **밖**의 같은 말은 여전히 걸린다
    rec = _rec(["100"])
    rec["target_text"] += " 이 용접부는 허용하지 않는다."
    bad = _check(rec)
    assert "verdict_word" in bad and "text_unparsable" in bad


def test_판정_함의_표현을_덧붙이면_폐기된다():
    rec = _rec(["2011"])
    rec["target_text"] += " 기준 이내이다."
    assert "verdict_word" in _check(rec)


# ------------------------------------------------------------------ 관점 E · 상자와 좌표 규약

def test_경계에_닿는_상자는_통과하고_퇴화·이탈은_폐기된다():
    def one(box):
        rec, _ = M.assemble_record(_row(), [_ann("img-0001", 0, "2011", box)], NAMES, M.clause_basis(TABLE, "ST"))
        return _check(rec)
    assert one((0, 0, W, H)) == []
    for box in ((10, 10, 10, 50), (10, 50, 50, 50), (-1, 0, 5, 5), (0, 0, W + 1, 5), (0, 0, 5, H + 1), (50, 10, 10, 50)):
        assert "bbox_out_of_bounds" in one(box), box


def test_좌표_규약과_이미지_크기를_레코드가_들고_다닌다():
    rec = _rec(["2011"])
    assert rec["coord_space"] == "ABS_ORIG" and (rec["width_px"], rec["height_px"]) == (W, H)
    assert "image_size_mismatch" in _check(rec, wh=(W, H + 1))
    for key, reason in (("coord_space", "schema:coord_space"), ("material", "schema:material"),
                        ("width_px", "schema:width_px")):
        broken = dict(rec)
        del broken[key]
        assert reason in _check(broken), key
    assert "coord_space_unknown" in _check({**rec, "coord_space": "NORM_1000"})


def test_mm_값이_있으면_지어낸_값이다():
    rec = _rec(["2011"])
    rec["skeleton"]["defects"][0]["size_mm"] = {"major_axis": 3.2}
    assert "mm_present" in _check(rec)


def test_정상_페어는_고정_문장이고_조항이_없다():
    rec = _rec([])
    assert rec["target_text"] == M.NORMAL_TEXT and rec["skeleton"]["clauses"] == []
    assert "normal_text_mismatch" in _check({**rec, "target_text": M.NORMAL_TEXT + " 덧붙임."})
    rec["skeleton"]["clauses"] = ["KRA27-T15"]
    assert "normal_has_clauses" in _check(rec)


# ------------------------------------------------------------------ 조립 · 정렬 · 기하 무효

def test_결함은_ann_id_문자열_순이다():
    """'#10' < '#2'. 정렬은 학습 계약이다 — 결함 11개 이상인 페어에서 번호순과 갈린다."""
    anns = [_ann("img-0001", k, "2011", (k + 1, 1, k + 5, 9)) for k in (2, 10, 1)]
    rec, _ = M.assemble_record(_row(n_defects=3), anns, NAMES, M.clause_basis(TABLE, "ST"))
    assert [d["bbox_px"][0] for d in rec["skeleton"]["defects"]] == [2, 11, 3]


def test_기하_무효는_세어서_싣고_조항에는_넣지_않는다():
    anns = [_ann("img-0001", 0, "2011"), _ann("img-0001", 1, "100", geom_valid=False)]
    rec, skipped = M.assemble_record(_row(n_defects=2), anns, NAMES, M.clause_basis(TABLE, "ST"))
    assert skipped == 1 and rec["n_annotations_skipped_geom_invalid"] == 1
    assert rec["skeleton"]["clauses"] == ["KRA27-T15"] and "균열" not in rec["target_text"]
    assert _check(rec) == []
    assert "n_annotations_skipped_geom_invalid" not in _rec(["2011"])          # 0 이면 키를 싣지 않는다


def test_결함_라벨인데_유효_기하가_없으면_레코드가_없다():
    rec, skipped = M.assemble_record(_row(), [_ann("img-0001", 0, "2011", geom_valid=False)], NAMES,
                                     M.clause_basis(TABLE, "ST"))
    assert rec is None and skipped == 1


def test_레코드의_키와_순서가_고정이다():
    rec = _rec(["2011"])
    assert list(rec) == ["image_id", "image_path", "client", "split", "material", "width_px", "height_px",
                         "coord_space", "skeleton", "target_text"]
    assert list(rec["skeleton"]) == ["defects", "verdict", "verdict_mode", "clauses", "candidate_rules",
                                     "uncited_codes"]
    assert list(rec["skeleton"]["defects"][0]) == ["type", "bbox_px", "size_px", "size_mm"]
    assert rec["skeleton"]["verdict"] is None and rec["skeleton"]["verdict_mode"] == "clause_only"


# ------------------------------------------------------------------ 관점 F · 빌드: 격리·귀속·회계

def _frames(rows, anns):
    a = pd.DataFrame(anns, columns=list(_ann("x", 0, "2011")))
    return pd.DataFrame(rows), a


def _good():
    rows = [_row("img-0001", n_defects=2), _row("img-0002", has_defect=False, n_defects=0, client="C2", group_id="g-0002"),
            _row("img-0003", material="AL", client="C3", split="val", group_id="g-0003"),
            _row("img-0004", n_defects=1, group_id="g-0004"),
            _row("img-9001", split="eval", client=None, group_id="g-9001")]
    anns = [_ann("img-0001", 0, "2011"), _ann("img-0001", 1, "100"), _ann("img-0003", 0, "401"),
            _ann("img-0004", 0, "301", geom_valid=False), _ann("img-9001", 0, "2011")]
    return rows, anns


def _build(rows, anns, **kw):
    m, a = _frames(rows, anns)
    return M.build_pairs(m, a, TABLE, NAMES, LEX, **kw)


def test_빌드는_train_val_만_만들고_eval_과의_교차를_두_수준으로_센다():
    res = _build(*_good())
    assert [r["image_id"] for r in res.made] == ["img-0001", "img-0002", "img-0003"]
    assert res.discarded == [{"image_id": "img-0004", "reasons": [M.NO_VALID_GEOMETRY]}]
    assert res.n_input == 4 and res.n_geom_skipped == 1
    assert res.isolation == {"eval_image_id_overlap": 0, "eval_group_id_overlap": 0,
                             "n_eval_images": 1, "n_eval_groups": 1}
    assert res.uncovered == ["AL"]


def test_eval_과_묶음이_겹치면_빌드가_멈춘다():
    rows, anns = _good()
    rows[0]["group_id"] = "g-9001"                     # 이미지는 다르고 묶음만 같다 — 옛 단언은 못 봤다
    with pytest.raises(M.PairInputError, match="group_id 1"):
        _build(rows, anns)


def test_eval_과_이미지가_겹치면_빌드가_멈춘다():
    rows, anns = _good()
    rows.append(_row("img-9001", group_id="g-0009"))
    with pytest.raises(M.PairInputError, match="image_id 1"):
        _build(rows, anns)


def test_산출물에_eval_이미지가_실리면_재집계에서_멈춘다(monkeypatch):
    """입력 단언이 돈 것과 산출물이 깨끗한 것은 다른 사실이다 — 조립이 id 를 바꿔 끼우는 결함을 모사한다."""
    real = M.assemble_record

    def swapped(row, anns, names, base):
        rec, skipped = real(row, anns, names, base)
        if rec is not None and rec["image_id"] == "img-0002":
            rec["image_id"] = "img-9001"                 # eval 의 이미지
        return rec, skipped

    monkeypatch.setattr(M, "assemble_record", swapped)
    with pytest.raises(M.PairInputError, match="산출물이 eval 과 겹친다") as e:
        _build(*_good())
    assert "'eval_image_id_overlap': 1" in str(e.value) and "'eval_group_id_overlap': 1" in str(e.value)   # 두 수준 다 센다


@pytest.mark.parametrize("col, value", [("material", "ST "), ("material", None), ("material", "SS"),
                                        ("client", None), ("client", " C1")])
def test_참여자와_재질_표기가_이상하면_빌드가_멈춘다(col, value):
    rows, anns = _good()
    rows[0][col] = value
    with pytest.raises(M.PairInputError):
        _build(rows, anns)


def test_어노테이션_수가_매니페스트와_어긋나면_빌드가_멈춘다():
    rows, anns = _good()
    rows[0]["n_defects"] = 3
    with pytest.raises(M.PairInputError, match="n_defects"):
        _build(rows, anns)


def test_image_id_가_중복이면_빌드가_멈춘다():
    rows, anns = _good()
    rows.append(_row("img-0002", has_defect=False, n_defects=0, group_id="g-0002"))
    with pytest.raises(M.PairInputError, match="중복"):
        _build(rows, anns)


def test_경로_실재_검사는_없는_파일에서_멈춘다(tmp_path):
    rows, anns = _good()
    with pytest.raises(M.PairInputError, match="경로 4개"):
        _build(rows, anns, root=tmp_path, check_paths=True)
    (tmp_path / "tiles").mkdir()
    for r in rows:
        (tmp_path / r["rel_path"]).write_bytes(b"")
    assert len(_build(rows, anns, root=tmp_path, check_paths=True).made) == 3


def _write(tmp_path, res, name="out"):
    manifest_csv = tmp_path / "manifest.csv"
    manifest_csv.write_text("image_id\n", encoding="utf-8")
    out = tmp_path / name
    digest = M.write_outputs(out, res, asset="d4_pairs_test", version="v0", date="test",
                             limits_csv=M.LIMITS_CSV, manifest_csv=manifest_csv,
                             meta=M.base_meta(res, asset="d4_pairs_test_v0"))
    return out, digest


def test_한_레코드에_폐기_사유가_둘이어도_회계가_맞는다(tmp_path):
    """옛 회계는 `sum(사유 수)` 였다 — 정합 검사가 **파일을 쓴 뒤에** 죽었다."""
    rows, anns = _good()
    anns[0] = _ann("img-0001", 0, "999", (0, 0, W + 9, 5))          # 모르는 코드 + 경계 이탈
    res = _build(rows, anns)
    assert {"unknown_code", "bbox_out_of_bounds"} <= set(res.discarded[0]["reasons"])
    assert len(res.discarded) == 2 and sum(res.reasons.values()) > 2
    out, _ = _write(tmp_path, res)
    counts = json.loads((out / "counts.json").read_text(encoding="utf-8"))
    assert counts["discarded"]["quarantine"] == 2 == counts["n_generated"] - len(res.made)
    assert sum(c["n_total"] for c in counts["clients"].values()) == len(res.made)


# ------------------------------------------------------------------ 관점 G · 결정론과 산출 바이트

def test_두_번_빌드한_바이트가_같고_CR_이_없다(tmp_path):
    a, da = _write(tmp_path, _build(*_good()), "a")
    b, db = _write(tmp_path, _build(*_good()), "b")
    names = sorted(p.name for p in a.iterdir())
    assert names == ["PAIRS_META.json", "SNAPSHOT.sha256", "counts.json", "discarded.jsonl", "pairs.jsonl"]
    assert da == db
    for n in names:
        raw = (a / n).read_bytes()
        assert raw == (b / n).read_bytes(), n
        assert b"\r" not in raw, n                       # 파일럿 v1 은 CRLF 였고 해시가 그 바이트 위에 섰다
    assert (a / "pairs.jsonl").read_bytes().endswith(b"}\n")


def test_행_순서를_섞어도_산출이_같다(tmp_path):
    rows, anns = _good()
    a, _ = _write(tmp_path, _build(rows, anns), "a")
    b, _ = _write(tmp_path, _build(rows[::-1], anns[::-1]), "b")
    assert (a / "pairs.jsonl").read_bytes() == (b / "pairs.jsonl").read_bytes()


def test_메타가_좌표_규약·격리·분할별_건수·정렬을_싣는다(tmp_path):
    res = _build(*_good())
    out, _ = _write(tmp_path, res)
    meta = json.loads((out / "PAIRS_META.json").read_text(encoding="utf-8"))
    assert meta["coord_space"] == "ABS_ORIG"
    assert meta["eval_isolation"]["eval_group_id_overlap"] == 0
    assert meta["counts_by_split"] == {
        "train": {"C1": {"n_total": 1, "n_defect": 1, "n_normal": 0},
                  "C2": {"n_total": 1, "n_defect": 0, "n_normal": 1}},
        "val": {"C3": {"n_total": 1, "n_defect": 1, "n_normal": 0}}}
    assert meta["discard_reasons"] == {M.NO_VALID_GEOMETRY: 1}
    assert "문자열" in meta["ordering"]
    recs = [json.loads(ln) for ln in (out / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["coord_space"] == "ABS_ORIG" for r in recs)


# ------------------------------------------------------------------ 배선

def test_게이트는_문장화와_생성기를_import_하지_않는다():
    """기대값이 생성 경로를 지나면 다시 자기참조다."""
    tree = ast.parse(Path(G.__file__).read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert not {m for m in mods if m.startswith(("corpus.rules.clause_text", "corpus.generate"))}, mods


def test_생성기와_게이트의_고정_문장이_같다():
    """두 곳이 어긋나면 게이트가 전량 폐기로 드러내지만, 그 전에 여기서 먼저 안다."""
    assert M.NO_CLAUSE_TAIL == G.NO_CLAUSE_MATERIAL
    assert M.NO_CLAUSE_CODE_TAIL == G.NO_CLAUSE_CODE_TAIL
    assert M.NOTE_UNEXPRESSED == G.NOTE_UNEXPRESSED
    assert M.CLOSE_DIMENSIONAL == G.CLOSE_DIMENSIONAL
    assert M.CLOSE_SIZE_INDEPENDENT == G.CLOSE_SIZE_INDEPENDENT


def test_서술의_고정_문장은_판정_함의_검출에_걸리지_않는다():
    from corpus.generate.numeric_lock import find_verdict_implying

    for s in (M.NO_CLAUSE_TAIL, M.NOTE_UNEXPRESSED, M.CLOSE_DIMENSIONAL, M.CLOSE_SIZE_INDEPENDENT, M.NORMAL_TEXT):
        assert find_verdict_implying(s) == (), s


# ------------------------------------------------------------------ 레코드 계약의 표준 예시

#: 기공 1 · 균열 1 · 기하 무효 1 인 이미지의 레코드. **규약 문서의 예시가 이것과 같아야 한다.**
#: 스키마 예시가 실물과 달랐던 전례가 있어(06번 §5 회귀 10) 예시를 손으로 쓰지 않고 여기에 고정한다.
#: 값·키 순서·실수 표기(`30.5` 대 `30.50`)까지 바이트로 본다.
EXAMPLE_RECORD = (
    '{"image_id": "img-0001", "image_path": "tiles/img-0001.png", "client": "C1", "split": "train", '
    '"material": "ST", "width_px": 1280, "height_px": 720, "coord_space": "ABS_ORIG", '
    '"skeleton": {"defects": ['
    '{"type": "2011", "bbox_px": [10, 20, 50, 60], "size_px": {"major_axis": 40.0, "equiv_diameter": 30.5}, "size_mm": null}, '
    '{"type": "100", "bbox_px": [0, 0, 1280, 720], "size_px": {"major_axis": 40.0, "equiv_diameter": 30.5}, "size_mm": null}], '
    '"verdict": null, "verdict_mode": "clause_only", "clauses": ["KRA27-3D", "KRA27-T15"], '
    '"candidate_rules": ['
    '{"rule_id": "KRA27-3D-01", "clause_id": "KRA27-3D", "defect_code": "100", "thickness_min": "0.00", "thickness_max": null}, '
    '{"rule_id": "KRA27-T15-01", "clause_id": "KRA27-T15", "defect_code": "2011", "thickness_min": "0.00", "thickness_max": "10.01"}, '
    '{"rule_id": "KRA27-T15-02", "clause_id": "KRA27-T15", "defect_code": "2011", "thickness_min": "10.01", "thickness_max": "25.01"}, '
    '{"rule_id": "KRA27-T15-03", "clause_id": "KRA27-T15", "defect_code": "2011", "thickness_min": "25.01", "thickness_max": "50.01"}, '
    '{"rule_id": "KRA27-T15-04", "clause_id": "KRA27-T15", "defect_code": "2011", "thickness_min": "50.01", "thickness_max": "100.01"}], '
    '"uncited_codes": []}, '
    '"target_text": "%s", "n_annotations_skipped_geom_invalid": 1}'
)


def test_레코드_계약의_표준_예시가_실제_조립_출력과_같다():
    anns = [_ann("img-0001", 0, "2011", (10, 20, 50, 60)), _ann("img-0001", 1, "100", (0, 0, W, H)),
            _ann("img-0001", 2, "301", (5, 5, 9, 9), geom_valid=False)]
    for a in anns:
        a["equiv_diameter_px"] = 30.5
    rec, skipped = M.assemble_record(_row(n_defects=3), anns, NAMES, M.clause_basis(TABLE, "ST"))
    assert skipped == 1
    want = EXAMPLE_RECORD % rec["target_text"]          # 서술의 문장은 위 게이트 시험이 따로 고정한다
    assert json.dumps(rec, ensure_ascii=False) == want  # 값과 키 순서까지
    assert _check(rec, wh=(W, H)) == []

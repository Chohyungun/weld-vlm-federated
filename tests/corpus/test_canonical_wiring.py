"""정본 배선 · 자료 등식 · 아티팩트 게이트 — 체크리스트 6·7·8 (80번 G1·G3·G4).

사전실험이 가르친 것은 "결함이 있는데 시험이 초록"이 아니라 **"검사를 만들어 두고
부르지 않았다"** 였다. 그래서 이 파일은 기능이 아니라 **배선**을 시험한다.

- 자료 등식 (G4): 생성 프롬프트와 판정 프롬프트가 같은 [자료]를 싣는가.
  stage0 이 의무화한 요소를 stage2 의 자료가 담고 있는가.
- 정본 배선 (G1): 생성 경로가 자체 수치·판정어 정규식을 다시 정의하지 않는가.
  실제로 정본 검사기를 호출하는가 (AST 로 확인 — 문서로 금지한 것을 시험으로 옮긴다).
- 아티팩트 게이트 (G3): 실제로 오염됐던 문장을 픽스처로 박아 출력 측에서 걸리는가.
"""

from __future__ import annotations

import ast
import itertools
import json
import re
from pathlib import Path

import pytest

from corpus.generate import basis as B
from corpus.generate import numeric_lock as nl
from corpus.generate import run_cycle_corpus as R

REPO = Path(__file__).resolve().parents[2]
GEN_DIR = REPO / "corpus/generate"

#: 파일럿 채택분에서 실제로 뽑은 오염 문장 (80번 B1: 채택 69건 중 49건이 이 산물).
#: 92eefae 의 회귀 시험이 프롬프트만 보고 출력은 안 봐서 71%가 통과했다 (G3-2).
CONTAMINATED = {
    "clause-0002": "KRA27-T15 조항에 따르면, 기공 (결함 코드 2011)의 크기는"
                   " 4.00 Unit.MM 이하로 제한된다.",
    "clause-0004": "KRA27-T15 조항에 따르면, 기공 (결함 코드 2011)은 모재 두께의 0.2 배"
                   " 이하로 제한된다. 적용 두께 구간은 25.01 mm 이상 50.01 mm 미만이다.",
}


@pytest.fixture(scope="module")
def recs():
    return R.build_clause_records(12, "full") + R.build_remedy_records(4)


@pytest.fixture(scope="module")
def gated():
    return R.build_clause_records(12, "clause_only")


# --------------------------------------------------------------- G4 자료 등식

def test_생성과_판정이_같은_자료_문자열을_싣는다(recs):
    """등식은 시험이 아니라 구성으로 보장된다 — 시험은 그것을 확인만 한다 (G4-1)."""
    for r in recs:
        basis = B.render_basis(r)
        assert basis, r["sample_id"]
        assert basis in R.prompt_generate(r), r["sample_id"]
        assert basis in R.prompt_judge({**r, "text": "문장"}), r["sample_id"]


def test_stage0_의무_요소가_전부_자료에_있다(recs, gated):
    """지시대로 쓴 문장이 판정에서 구조적으로 기각되는 경로를 막는다 (G4-2·G4-3, B2·B4).

    파일럿에서는 결함 코드가 생성 의무였는데 판정 자료에 없었고 178건 중 98건이
    그 상태였다.
    """
    for r in recs + gated:
        fields = B.basis_fields(r)
        for label, value in B.required_mentions(r).items():
            assert label in fields, f"{r['sample_id']}: 의무 요소 {label} 이 자료에 없다"
            assert value == fields[label], f"{r['sample_id']}: {label} 값 불일치"


def test_의무_요소는_생성_프롬프트에도_그대로_실린다(recs):
    for r in recs:
        p = R.prompt_generate(r)
        for value in B.required_mentions(r).values():
            assert value in p, (r["sample_id"], value)


def test_clause_only_는_판정을_자료에서_뺀다(gated):
    """축이 '합부를 말하지 않는다' 이므로 자료에도 판정이 없어야 한다."""
    for r in gated:
        assert r["verdict"] is None
        assert "판정" not in B.basis_fields(r)
        assert "합격" not in R.prompt_judge({**r, "text": "x"}).split("[문장]")[0]


def test_자료에_내부_표현이_새지_않는다(recs, gated):
    """`Unit.MM`·개구간 `25.01`·후행 0 `4.00` 이 자료로 새면 생성문에 박힌다 (B1)."""
    for r in recs + gated:
        basis = B.render_basis(r)
        assert not re.search(r"\b(?:Unit|InspectionMethod|LimitRule|Verdict)\.", basis)
        assert "None" not in basis
        # 표집 실측값은 0.01 그리드라 X.01 이 정당할 수 있다. 자료가 그 값을 실제로
        # 들고 있는 경우만 허용되며, 그 판정은 아티팩트 게이트가 basis 로 한다.
        assert nl.find_artifacts(basis, basis=basis) == ()


# --------------------------------------------------------------- G1 정본 배선

def _module_source(name: str) -> str:
    return (GEN_DIR / name).read_text(encoding="utf-8")


def _assigned_regex_names(src: str) -> set[str]:
    """모듈 최상위에서 `re.compile(...)` 로 만든 이름들."""
    tree = ast.parse(src)
    out: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == "compile"):
            continue
        for t in node.targets:
            if isinstance(t, ast.Name):
                out.add(t.id)
    return out


#: 정본이 사는 곳. 여기 말고 다른 생성 모듈이 같은 성격의 정규식을 두면 규칙이 두 벌이 된다.
_LOCK_OWNER = "numeric_lock.py"
_FORBIDDEN_REGEX_HINTS = ("NUM", "VERDICT", "STD_REF", "PASS", "FAIL", "CLAUSE", "UNIT")


@pytest.mark.parametrize("mod", ["run_cycle_corpus.py", "make_pairs_pilot.py", "run_pilot.py"])
def test_생성_모듈이_수치_판정어_정규식을_재정의하지_않는다(mod):
    """`skeleton_gen.py:34-38` 이 문서로 금지한 것을 시험으로 옮긴다 (G1-2).

    파일럿에서 `run_cycle_corpus.check_record` 가 자체 수치 검사를 두었고, 후행 0
    정규화가 `rstrip("0.")` 이라 4 ≡ 40 ≡ 400 이 동치가 됐다 (B7).
    """
    names = _assigned_regex_names(_module_source(mod))
    bad = sorted(n for n in names if any(h in n.upper() for h in _FORBIDDEN_REGEX_HINTS))
    assert not bad, f"{mod}: 정본과 겹치는 정규식 재정의 {bad} — {_LOCK_OWNER} 를 불러라"


def _called_names(src: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                out.add(f.id)
            elif isinstance(f, ast.Attribute):
                out.add(f.attr)
    return out


def test_사이클_실행기가_정본_검사기를_부른다():
    """정본이 있다는 사실은 정본이 불린다는 증거가 아니다 (G1-1)."""
    called = _called_names(_module_source("run_cycle_corpus.py"))
    for fn in ("check_numeric_lock", "run_stage1", "check_normal_lock",
               "find_artifacts", "find_verdict_implying", "render_basis"):
        assert fn in called, f"run_cycle_corpus 가 {fn} 을 부르지 않는다"


def test_사이클_실행기가_자료를_검사기에_넘긴다():
    """아티팩트 게이트는 basis 가 있어야 판정 기준이 선다. 안 넘기면 검사가 꺼진다."""
    src = _module_source("run_cycle_corpus.py")
    assert "basis_fn=render_basis" in src


def test_페어_생성기가_정본_어휘_락을_부른다():
    called = _called_names(_module_source("make_pairs_pilot.py"))
    assert "find_defect_tokens" in called


# --------------------------------------------------------------- G3 아티팩트 게이트

@pytest.mark.parametrize("sample_id", sorted(CONTAMINATED))
def test_실제_오염_문장이_출력측에서_걸린다(sample_id):
    """프롬프트만 고치고 출력은 안 본 것이 71% 오염을 통과시켰다 (G3-2)."""
    text = CONTAMINATED[sample_id]
    hits = nl.find_artifacts(text, basis="- 조항이 정한 기준: 4 mm 이하")
    assert hits, sample_id


def test_자료에_있는_표기는_아티팩트가_아니다():
    """자료가 그렇게 줬으면 옮겨 적은 것이 맞다 — 고칠 곳은 자료 쪽이다."""
    text = "실측 25.01 mm 이다."
    assert nl.find_artifacts(text, basis="- 실측 크기: 25.01 mm") == ()
    assert nl.find_artifacts(text, basis="") != ()


def test_수사_표기가_잡힌다():
    """값 파싱을 거치지 않아 화이트리스트가 못 잡는 경로다 (G2-7)."""
    assert nl.find_artifacts("결함이 십이 개 관찰된다", basis="")
    assert nl.find_artifacts("twelve defects", basis="")
    assert nl.find_artifacts("十二 개", basis="")


def test_판정_함의_표현이_잡힌다():
    """`clause_only` 는 합격·불합격 낱말을 피한 같은 뜻도 폐기다 (G2-6)."""
    assert nl.find_verdict_implying("이 용접부는 기준에 적합하다")
    assert nl.find_verdict_implying("허용된다")
    assert nl.find_verdict_implying("조항이 정한 기준은 4 mm 이하다") == ()


def test_분리자가_낀_판정어도_뒤집힘으로_잡힌다():
    """`(?<!불)합격` 한 글자 lookbehind 는 '불 합격' 에 뚫린다 (B21)."""
    scan = nl._scan("이 부위는 불 합격 이다")
    assert (scan.n_pass, scan.n_fail) == (0, 1)
    scan2 = nl._scan("이 부위는 불·합격 이다")
    assert (scan2.n_pass, scan2.n_fail) == (0, 1)


# --------------------------------------------------------------- 판정 파싱

def test_판정_파싱이_앵커_기반이다():
    """부분문자열 매칭은 LOOKS·TOKEN·WRONG 을 판정으로 읽는다 (B16, G3-4)."""
    anchors = R.load_config()["parsing"]["verdict_anchors"]
    assert R.parse_judgement("OK", anchors) == (True, True)
    assert R.parse_judgement("NG", anchors) == (False, True)
    assert R.parse_judgement("판정: NG", anchors) == (False, True)
    # 앵커에 안 걸리면 통과로 세지 않되 형식 위반으로 따로 기록한다
    assert R.parse_judgement("This LOOKS fine to me", anchors) == (False, False)
    assert R.parse_judgement("the TOKEN budget", anchors) == (False, False)


def test_되풀이_사유가_강등된다():
    """기각 사유 109건 중 97건이 입력 문장의 되풀이였다 (B17, G3-3)."""
    cfg = R.load_config()
    src = "KRA27-T15 조항에 따르면 기공의 크기는 4 mm 이하로 제한된다"
    echoed, is_echo = R.clean_reason(src, src, cfg)
    assert is_echo and echoed == cfg["parsing"]["reason_echo_downgrade_to"]
    real, is_echo2 = R.clean_reason("자료에 없는 두께 값을 새로 만들었다", src, cfg)
    assert not is_echo2 and real.startswith("자료에 없는")


def test_사전등록본이_두_후보에_같은_예산을_준다():
    """56.75pp 는 계열 차이가 아니라 계열+설정 차이였다 (B9, G12-4)."""
    cfg = R.load_config()
    assert cfg["judges"]["canonical"] is None, "사람 라벨 비교 전에는 정본이 없다"
    assert len(cfg["judges"]["candidates"]) >= 2
    fams = {c["family"] for c in cfg["judges"]["candidates"]}
    assert "qwen" not in fams, "생성 모델과 같은 계열은 검증기가 될 수 없다 (개발규약 3-5)"
    # 예산·사고 정책은 후보별이 아니라 공통 블록에 있다 — 구조로 교락을 막는다
    assert "budget" in cfg["judges"] and "thinking_policy" in cfg["judges"]
    for c in cfg["judges"]["candidates"]:
        assert "max_new_tokens" not in c and "prefill" not in c


def test_보고_스키마가_검증_수준을_말한다():
    """QA 0.94 와 판정추론 0.345 가 같은 이름으로 실린 것이 사고였다 (B8, G6-1)."""
    none_axis = R.axis_block("QA", 10, {"s": R.stage(10, 9, {})},
                             validated_by="none", measures=["format"])
    assert "end_to_end_format_rate" in none_axis
    assert "end_to_end_pass_rate" not in none_axis
    xf = R.axis_block("c", 10, {"s": R.stage(10, 9, {})},
                      validated_by="cross_family", measures=["groundedness"])
    assert "end_to_end_pass_rate" in xf


def test_돌지_않은_단계는_폐기0건으로_렌더되지_않는다():
    """`counts.json` 이 미실행 단계를 '폐기 0건' 으로 그린 것이 회계 사고였다 (G6-2)."""
    st = R.stage(10, 0, {}, ran=False)
    assert st["status"] == "not_run" and st["n_pass"] is None


# --------------------------------------------------------------- 검증기 정본 선정

def test_라벨_표본이_층화되고_기계판정을_보여주지_않는다():
    """라벨러가 기계 판정에 끌려가면 그것은 정답이 아니다 (G12-3)."""
    from corpus.validate import judge_labels as J

    cfg = J.load_cfg()
    recs = [{"sample_id": f"s{i}",
             "axis": "조항검색_기준서술" if i % 2 else "조치서술",
             "judge_deepseek_pass": bool(i % 3), "text": f"문장 {i}",
             "topic": "아크 스트라이크", "remedy_ko": "제거한다",
             "source_ref": "IACS Rec.47", "inspection_method": "VT"}
            for i in range(200)]
    rows, meta = J.build_sheet(recs, cfg, "deepseek")
    assert len(rows) == cfg["labeling"]["n"]

    # 층화는 **여전히 한다** — 메타에 남는다. 사람 시트에서만 안 보일 뿐이다.
    assert len(meta["cells"]) == 4, meta["cells"]
    assert len(meta["stratum_of"]) == len(rows)
    for cell in meta["cells"].values():
        assert {"N_population", "k_drawn", "target", "shortfall", "p_sampling"} <= set(cell)

    for r in rows:
        assert r["human_ok"] is None
        assert "stratum" not in r, "층 표시가 표본지에 실렸다"
        assert not any(k.startswith("judge_") for k in r), "기계 판정이 표본지에 실렸다"
        # **키만 보면 안 된다.** 옛 시험이 키만 봐서 `stratum` 값 안의
        # `judge_pass` 를 통째로 놓쳤다 — 감사가 잡을 때까지 초록이었다 (F14).
        for v in r.values():
            if isinstance(v, str):
                assert "judge_pass" not in v and "judge_fail" not in v, \
                    f"기계 판정이 **값** 안에 실렸다: {v[:60]}"
        assert r["basis"], "사람도 기계와 같은 자료를 봐야 한다 (G4-1)"


def test_라벨_표본의_순서가_층을_드러내지_않는다():
    """값을 지워도 **순서**가 남으면 블라인드가 아니다.

    층별로 뽑아 이어 붙이면 통과분 블록 뒤에 기각분 블록이 온다. 라벨러는 값을 못 봐도
    '앞쪽이 통과분' 이라는 것을 안다. 인접 행의 층이 바뀌는 횟수로 잰다 — 블록이면
    층 수보다 작고(≤3), 섞였으면 그보다 훨씬 크다.
    """
    from corpus.validate import judge_labels as J

    cfg = J.load_cfg()
    recs = [{"sample_id": f"s{i}",
             "axis": "조항검색_기준서술" if i % 2 else "조치서술",
             "judge_deepseek_pass": bool(i % 3), "text": f"문장 {i}",
             "topic": "아크 스트라이크", "remedy_ko": "제거한다",
             "source_ref": "IACS Rec.47", "inspection_method": "VT"}
            for i in range(200)]
    rows, meta = J.build_sheet(recs, cfg, "deepseek")
    seq = [meta["stratum_of"][r["sample_id"]] for r in rows]
    changes = sum(1 for a, b in itertools.pairwise(seq) if a != b)
    assert changes > len(meta["cells"]), (
        f"인접 층 변화 {changes}회 — 층이 블록으로 묶여 순서가 판정을 드러낸다")


def test_표집_메타는_사람_시트와_다른_파일이다():
    """지우지 않고 **옮긴다**. 층별 표집확률이 없으면 층 가중을 할 수 없고,
    그러면 층화 표집을 한 이유 자체가 사라진다 (F14-P1)."""
    import ast

    src = (REPO / "corpus/validate/judge_labels.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_sheet")
    # 시트와 메타를 함께 돌려주는 형태여야 한다 — 한 덩어리면 다시 섞인다.
    ret = [n for n in ast.walk(fn) if isinstance(n, ast.Return)][-1]
    assert isinstance(ret.value, ast.Tuple) and len(ret.value.elts) == 2
    assert "sheet_v1.meta.json" not in src, "메타 경로를 박지 말고 시트 경로에서 파생시켜라"
    assert '.with_suffix(".meta.json")' in src, "메타를 따로 쓰지 않는다"


def test_라벨_없이는_정본이_없다():
    """라벨 없이 잰 값은 pass_rate 가 아니라 judge_agreement 로만 싣는다."""
    from corpus.validate import judge_labels as J

    assert J.load_cfg()["judges"]["canonical"] is None


def test_정밀도_재현율이_사람_라벨_기준으로_계산된다():
    from corpus.validate import judge_labels as J

    cfg = J.load_cfg()
    recs = [{"sample_id": "a", "axis": "조치서술", "judge_deepseek_pass": True},
            {"sample_id": "b", "axis": "조치서술", "judge_deepseek_pass": True},
            {"sample_id": "c", "axis": "조치서술", "judge_deepseek_pass": False}]
    labels = [{"sample_id": "a", "human_ok": True, "labeler": "p1"},
              {"sample_id": "b", "human_ok": False, "labeler": "p1"},
              {"sample_id": "c", "human_ok": True, "labeler": "p1"}]
    got = J.score(recs, labels, cfg)["candidates"]["deepseek"]
    assert (got["tp"], got["fp"], got["fn"]) == (1, 1, 1)
    assert got["precision"] == 0.5 and got["recall"] == 0.5


# --------------------------------------------------------------- 층 가중 (35번 D 제안 1)

def _weighted_fixture():
    """두 층: pass 층은 모집단 40 중 20 을 뽑았고(w=2), fail 층은 10 중 10(w=1).

    무가중은 표본 30건의 값이고, 가중은 모집단 50건의 추정치다. 둘이 달라야 가중이
    실제로 걸린 것이다.
    """
    recs, labels = [], []
    # pass 층 20건: 정답 OK 16 · NG 4  → 정밀도(무가중) 16/20
    for i in range(20):
        recs.append({"sample_id": f"p{i}", "axis": "조항검색_기준서술",
                     "judge_deepseek_pass": True, "judge_phi_pass": i < 10})
        labels.append({"sample_id": f"p{i}", "human_ok": i < 16, "labeler": "L1"})
    # fail 층 10건: 정답 OK 5 · NG 5  → 재현율에 영향
    for i in range(10):
        recs.append({"sample_id": f"f{i}", "axis": "조항검색_기준서술",
                     "judge_deepseek_pass": False, "judge_phi_pass": i < 5})
        labels.append({"sample_id": f"f{i}", "human_ok": i < 5, "labeler": "L1"})
    meta = {"stratified_by_judge": "deepseek",
            "cells": {"조항검색_기준서술|judge_pass": {"N_population": 40, "k_drawn": 20},
                      "조항검색_기준서술|judge_fail": {"N_population": 10, "k_drawn": 10}}}
    return recs, labels, meta


def test_층_가중이_실제로_걸린다():
    """모집단 추정치가 표본 값과 다르게 나와야 한다 — 같으면 가중이 장식이다."""
    from corpus.validate import judge_labels as J

    cfg = J.load_cfg()
    recs, labels, meta = _weighted_fixture()
    out = J.score(recs, labels, cfg, meta)
    assert out["weighting"] == "stratum_weighted"
    assert out["weights_by_stratum"] == {"조항검색_기준서술|judge_pass": 2.0,
                                         "조항검색_기준서술|judge_fail": 1.0}
    d = out["candidates"]["deepseek"]
    # 무가중: TP 16 FP 4 FN 5 → P 0.8, R 16/21
    assert (d["tp"], d["fp"], d["fn"]) == (16, 4, 5)
    assert d["recall"] == round(16 / 21, 4)
    # 가중: pass 층 ×2 → TP 32 FP 8, fail 층 ×1 → FN 5 → P 0.8, R 32/37
    w = d["weighted"]
    assert (w["tp"], w["fp"], w["fn"], w["population"]) == (32, 8, 5, 50)
    assert w["precision"] == 0.8 and w["recall"] == round(32 / 37, 4)
    assert w["recall"] != d["recall"], "가중이 걸리지 않았다"
    # 무가중 키는 그대로다 — 기존 소비자가 깨지지 않는다
    assert {"precision", "recall", "f1", "n"} <= set(d)


def test_메타가_없으면_무가중만_내고_그렇다고_말한다():
    from corpus.validate import judge_labels as J

    cfg = J.load_cfg()
    recs, labels, _ = _weighted_fixture()
    out = J.score(recs, labels, cfg, None)
    assert out["weighting"] == "unweighted_only" and "weighting_note" in out
    assert "weighted" not in out["candidates"]["deepseek"]


def test_층을_못_찾은_항목은_빼지_않고_기록한다():
    """조용히 빼면 표본 수가 줄어든 것을 아무도 모른다."""
    from corpus.validate import judge_labels as J

    cfg = J.load_cfg()
    recs, labels, meta = _weighted_fixture()
    recs.append({"sample_id": "x0", "axis": "조치서술",           # 메타 셀에 없는 층
                 "judge_deepseek_pass": True, "judge_phi_pass": True})
    labels.append({"sample_id": "x0", "human_ok": True, "labeler": "L1"})
    w = J.score(recs, labels, cfg, meta)["candidates"]["deepseek"]["weighted"]
    assert w["rows_without_stratum"] == ["x0"]
    assert w["n"] == 31


def test_현재_실물_표본은_전수라_가중이_무가중과_같다():
    """지금 meta 는 두 유효 층 p=1.0 이다. 가중을 걸어도 값이 안 바뀌어야 한다 —
    바뀐다면 가중 계산이 틀린 것이다. 모집단이 커지면 이 시험은 갈라져야 한다."""
    from corpus.validate import judge_labels as J

    meta_path = REPO / "corpus/validate/judge_labels/sheet_v1.meta.json"
    if not meta_path.is_file():
        pytest.skip("실물 표집 메타가 없다")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    weights = J.stratum_weights(meta)
    assert weights and all(w == 1.0 for w in weights.values()), weights


def test_라벨러_안내문이_저장소를_열지_말라고_말한다(tmp_path, monkeypatch):
    """35번 D 제안 2 — 추적되는 meta.json 의 층별 k_drawn 은 사전 확률이다. 안내문 없이는
    설계가 닫히지 않는다. 안내문은 시트를 만들 때 **같이** 생겨야 한다."""
    from corpus.validate import judge_labels as J

    assert "저장소를 열지 않는다" in J.LABELER_README
    assert "meta.json" in J.LABELER_README and "EVIDENCE.jsonl" in J.LABELER_README
    # 안내문에 판정·층 이름이 새면 안내문 자체가 누설 경로다
    assert "judge_pass" not in J.LABELER_README and "judge_fail" not in J.LABELER_README
    # 시트 생성 경로가 안내문을 실제로 쓰는지 (AST 가 아니라 배선 문자열로 본다 — 80번 G1-2)
    src = (REPO / "corpus/validate/judge_labels.py").read_text(encoding="utf-8")
    assert 'with_name("README_라벨러.md")' in src, "sheet 분기가 안내문을 쓰지 않는다"
    assert "LABELER_README.format(" in src
    rendered = J.LABELER_README.format(sheet="s.jsonl", labels="l.jsonl", field="human_ok")
    assert "{" not in rendered and "s.jsonl" in rendered and "human_ok: true" in rendered

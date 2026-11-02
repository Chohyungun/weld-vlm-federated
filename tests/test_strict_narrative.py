"""엄격 경로의 2 차 보조 지표(`evaluation.strict_narrative` · `strict_main.compute(aux_scoring=)`) — 합성 계수표와 합성 서술만.

지키는 것.

1. 보조 지표를 주지 않으면(판 5 등록) 산출은 연결 전 `strict_main` 과 바이트가 같다 — 연결 전 모듈은 커밋 `BEFORE` 의 blob 을 불러 맞댄다.
2. 보조 지표를 붙여도 M1 · 대조선 · 회복률 · 참여자 · 규칙 ⑤ 칸은 그대로다. 계산한 이름만 `not_computed` 에서 빠진다.
3. 서술 지표의 대응 차는 M1 과 같은 추첨 행렬을 쓴다 — 시험이 같은 등록 값으로 추첨을 다시 내어 한 칸을 재현한다.
4. 환경이 다르면 모델마다의 값은 싣고 대응 차는 `산출 불가` 다. 조항 축은 모델과 대조선을 같은 정답으로 잰다.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from corpus.rules import limits_loader
from evaluation import narrative_checker as NCK
from evaluation import narrative_metrics as NM
from evaluation import percentile_rule as PR
from evaluation import strict_main as SM
from evaluation import strict_narrative as SN
from evaluation.clause_axis import (
    citations_from_codes,
    clause_axis_draw_parts,
    clause_axis_draws,
    clause_axis_report,
    entries_from_limits,
    gold_pairs_without_thickness,
)
from evaluation.narrative_extract import COMPLETE, VALIDATION_ERROR, NarrativeExtract
from evaluation.narrative_registration import EXPLORATORY_KINDS
from evaluation.prereg_unified import CLAUSE_AXIS_METRICS
from evaluation.sample_bootstrap import design_weights, draw_weights, strata_groups, weights_from
from tests.test_strict_main import EVAL_DOC, B, design, registration, scores_for, table_of

REPO = Path(__file__).resolve().parents[1]
BEFORE = "1f446c44"
"""보조 지표를 잇기 전의 마지막 커밋(검사기 v3 의 확인 배치 보고)."""
GOOD = "두께 정보가 없어 합부 판정은 내리지 않는다. 기계 표현으로 옮기지 못한 단서가 있어 합부를 결정하지 않는다."
BAD = "결함 길이는 3 mm 다. 이 용접부는 불합격이다."
ALL_NAMES = (*SN.NARRATIVE_NAMES, *CLAUSE_AXIS_METRICS)
EVAL_DOC_V6 = {**EVAL_DOC, "auxiliary": [*EVAL_DOC["auxiliary"], *ALL_NAMES]}


@pytest.fixture(scope="module")
def checker():
    return NCK.load_checker()


@pytest.fixture(scope="module")
def entries():
    return entries_from_limits(limits_loader.load_limits(REPO / NCK.LIMITS_REL, pilot=True))


def old_module(tmp: Path, rel: str, name: str, commit: str = BEFORE):
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"], capture_output=True, check=False)
    if src.returncode != 0:
        pytest.skip(f"이 체크아웃에 {commit} 가 없다")
    p = tmp / f"{name}.py"
    p.write_bytes(src.stdout)
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def aux_for(checker, entries, ids, side, keys, *, wanted=ALL_NAMES, exploratory=EXPLORATORY_KINDS, weights=None):
    material_of = SN.material_of_side(side, ids)
    w0 = design_weights(ids, side)
    models = {}
    for k, key in enumerate(keys):
        items = tuple(NM.ImageItem(i, NarrativeExtract(COMPLETE, GOOD if (j + k) % 3 else BAD, False, "eos", None),
                                   material_of[i], ("2011",), ("2011",), float(w0[j]))
                      for j, i in enumerate(ids))
        cited = {i: ("KRA27-T15",) if material_of[i] == "ST" else ("KRA27-T15", "KRA27-X99") for i in ids}
        models[key] = SN.ModelAux(key[0], key[1], items, cited)
    return SN.AuxScoring(models=models, checker=checker, checker_identity={"checker_id": NCK.CHECKER_ID},
                         entries=entries, index_clause_ids=frozenset(checker.t.clause_ids), material_of=material_of,
                         gold_codes={i: frozenset({"2011"}) for i in ids},
                         baseline_codes={i: ["2011"] for i in ids}, wanted=frozenset(wanted),
                         image_weights=weights or {i: float(w0[j]) for j, i in enumerate(ids)},
                         exploratory_kinds=tuple(exploratory), identity_checks={"code_tree": "checked"})


def compute(*, aux_scoring=None, eval_doc=EVAL_DOC, invalid=(), module=SM):
    ids, group_of, side, gold = design()
    base = table_of(ids, gold, 0.2, fp_rate=0.8, seed=99)
    kw = {"aux_scoring": aux_scoring} if aux_scoring is not None else {}
    out = module.compute(scores=scores_for(ids, gold), baseline=base, side=side, group_of=group_of, reg=registration(),
                         eval_doc=eval_doc, rejected={}, invalid_comparison=list(invalid), **kw)
    return out, (ids, group_of, side)


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=True, allow_nan=False)


@pytest.mark.parametrize("invalid", [(), ("env:model",)])
@pytest.mark.parametrize("doc", [EVAL_DOC, EVAL_DOC_V6])
def test_보조_지표가_없으면_연결_전_모듈과_바이트가_같다(tmp_path: Path, invalid, doc) -> None:
    old = old_module(tmp_path, "evaluation/strict_main.py", "strict_main_before_aux")
    assert dumps(compute(eval_doc=doc, invalid=invalid)[0]) == dumps(compute(eval_doc=doc, invalid=invalid, module=old)[0])


def test_보조_지표를_붙여도_M1_칸은_그대로이고_계산한_이름만_빠진다(checker, entries) -> None:
    plain, (ids, _, side) = compute(eval_doc=EVAL_DOC_V6)
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, design()[3])))
    with_aux, _ = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)
    assert set(with_aux) - set(plain) == {"narrative", "clause_axis"}
    for key in plain:
        if key != "not_computed":
            assert dumps(with_aux[key]) == dumps(plain[key]), key
    assert set(plain["not_computed"]) - set(with_aux["not_computed"]) == set(ALL_NAMES)
    assert {k: v for k, v in plain["not_computed"].items() if k not in ALL_NAMES} == with_aux["not_computed"]


def test_서술_대응_차는_M1_과_같은_추첨을_쓴다(checker, entries) -> None:
    ids, group_of, side, gold = design()
    keys = list(scores_for(ids, gold))
    aux = aux_for(checker, entries, ids, side, keys)
    out, _ = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)
    nar = out["narrative"]
    assert nar["draws"] == {"n_resamples": B, "shared_with_m1": True}
    assert set(nar["paired"]) == {f"uni_fed_s1 − {t}_s1" for t in ("uni_central", "uni_local_C1", "uni_local_C2",
                                                                     "uni_local_C3")}
    reg = registration()
    W = draw_weights(ids, group_of, side, n_resamples=B, rng_seed=reg.ci.rng_seed)
    rows = {k: NM.build_rows(aux.models[k].items, checker) for k in [("uni_fed", 1), ("uni_central", 1)]}
    pa, pb = (NM.point_estimates(rows[k]) for k in [("uni_fed", 1), ("uni_central", 1)])
    da, db = (NM.draws(rows[k], W) for k in [("uni_fed", 1), ("uni_central", 1)])
    want = NM.paired_difference(pa["forbidden.any"]["weighted"], pb["forbidden.any"]["weighted"], da["forbidden.any"],
                                db["forbidden.any"], alpha=reg.ci.alpha, min_defined_draws=reg.ci.min_defined_draws)
    got = nar["paired"]["uni_fed_s1 − uni_central_s1"]["narrative_forbidden_rate"]
    assert dumps(got) == dumps({**want, "status": SN.EXPLORATORY_TOTAL})            # (마)를 품는 머리 지표 — 탐색적
    assert got["interval"]["n_defined"] + got["interval"]["n_undefined"] == B
    assert nar["models"]["uni_fed_s1"]["forbidden.any"]["weighted"] == pa["forbidden.any"]["weighted"]


def test_환경이_다르면_모델마다의_값만_싣고_대응_차는_산출_불가(checker, entries) -> None:
    ids, _, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)))
    out, _ = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux, invalid=("env:model",))
    assert out["narrative"]["paired"] == {"status": "산출 불가", "reason": "env_mismatch"}
    assert set(out["narrative"]["models"]) == {f"{t}_s1" for t in SM.TAGS}
    assert out["clause_axis"]["paired"] == {"status": "산출 불가", "reason": "env_mismatch"}
    assert set(out["clause_axis"]["models"]) == {f"{t}_s1" for t in SM.TAGS} and "draws" not in out["clause_axis"]


def test_조항_축은_모델과_대조선을_같은_정답으로_잰다(checker, entries) -> None:
    ids, _, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)))
    ca = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)[0]["clause_axis"]
    n_st = sum(1 for i in ids if side[i].stratum.startswith("ST"))
    assert ca["gold"]["n_pairs"] == n_st and ca["gold"]["skipped"] == {"정답 조항 없음": len(ids) - n_st}
    m = ca["models"]["uni_fed_s1"]["unweighted"]
    assert m["citation_match_rate"]["value"] == 1.0
    assert (m["ungrounded_rate"]["nonexistent"], m["ungrounded_rate"]["irrelevant"]) == (len(ids) - n_st, len(ids) - n_st)
    assert ca["baseline"]["unweighted"]["citation_match_rate"]["value"] == 1.0
    assert ca["baseline"]["unweighted"]["n_citations"] == n_st


def test_조항_축의_대표표는_서술_지표와_같은_설계_가중이다(checker, entries) -> None:
    """04 §7-10 ① — 대표 = 설계 가중(`design_weights`), 부표 = 무가중. 강재 이미지는 맞는 인용 하나, AL 은 무관 하나 · 지어냄 하나."""
    ids, _, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)))
    ca = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)[0]["clause_axis"]
    w = dict(zip(ids, design_weights(ids, side), strict=True))
    st = sum(w[i] for i in ids if side[i].stratum.startswith("ST"))
    al = sum(w[i] for i in ids if not side[i].stratum.startswith("ST"))
    d = ca["models"]["uni_fed_s1"]["design"]
    assert d["citation_match_rate"]["value"] == pytest.approx(1.0)
    assert d["citation_precision"]["value"] == pytest.approx(st / (st + 2 * al))
    assert d["ungrounded_rate"]["value"] == pytest.approx(2 * al / (st + 2 * al))
    assert (d["ungrounded_rate"]["nonexistent"], d["ungrounded_rate"]["irrelevant"]) == pytest.approx((al, al))
    assert ca["baseline"]["design"]["citation_precision"]["value"] == pytest.approx(1.0)
    assert ca["baseline"]["design"]["n_citations"] == ca["baseline"]["unweighted"]["n_citations"]
    assert ca["weighting"] == SN.CLAUSE_AXIS_WEIGHTING and set(ca["models"]["uni_fed_s1"]) == {"design", "unweighted"}
    m = aux.models[("uni_fed", 1)]
    assert ca["models"]["uni_fed_s1"]["unweighted"] == clause_axis_report(dict(m.cited), aux_pairs(aux), aux.index_clause_ids)


def aux_pairs(aux):
    return gold_pairs_without_thickness(aux.entries, aux.material_of, {i: sorted(c) for i, c in aux.gold_codes.items()})[0]


BEFORE_CI = "d7110cc0"
"""조항 축 대응 구간을 더하기 전의 마지막 커밋(지시 1023c 보고) — 점추정의 회귀 기준."""


def report_direct(aux, ids, group_of, side, *, W=None, min_defined_draws=None):
    """`strict_main.compute` 를 거치지 않고 블록만 — 시험이 고른 `W` · 하한으로."""
    reg = registration()
    if W is None:
        W = draw_weights(ids, group_of, side, n_resamples=B, rng_seed=reg.ci.rng_seed)
    md = reg.ci.min_defined_draws if min_defined_draws is None else min_defined_draws
    return aux.report(image_ids=ids, W=W, alpha=reg.ci.alpha, min_defined_draws=md, seeds=[1], cell_name=SM.cell_name,
                      fed=SM.FED, central=SM.CENTRAL, locals_=SM.LOCALS, comparable=True)


def test_조항_축_대응_차는_같은_W_로_재현된다(checker, entries) -> None:
    """지시 1023d 과제 1 — 모델 − 무화소 조항 대조선의 대응 차를 M1 과 같은 추첨(같은 이미지 순서)에서, 설계 가중 · 무가중 둘 다."""
    ids, group_of, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)))
    ca = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)[0]["clause_axis"]
    assert set(ca["paired"]) == {f"{SM.cell_name(*k)} − 대조선" for k in aux.models} | set(MODEL_PAIRS)
    assert ca["draws"] == {"n_resamples": B, "shared_with_m1": True, "unweighted": "W ÷ 이미지 설계 가중(묶음 중복도)"}
    assert ca["interval"] == SN.CLAUSE_AXIS_INTERVAL and "같은 W" in ca["interval"]
    reg = registration()
    W = draw_weights(ids, group_of, side, n_resamples=B, rng_seed=reg.ci.rng_seed)
    w = design_weights(ids, side)
    pairs = aux_pairs(aux)
    base_cited = citations_from_codes(aux.entries, aux.material_of, {i: list(aux.baseline_codes[i]) for i in ids})
    m = aux.models[("uni_fed", 1)]
    for view, Wv in (("design", W), ("unweighted", W / w[None, :])):
        mine = clause_axis_draws(dict(m.cited), pairs, aux.index_clause_ids, ids, Wv)
        base = clause_axis_draws(base_cited, pairs, aux.index_clause_ids, ids, Wv)
        for name in CLAUSE_AXIS_METRICS:
            want = NM.paired_difference(ca["models"]["uni_fed_s1"][view][name], ca["baseline"][view][name], mine[name],
                                        base[name], alpha=reg.ci.alpha, min_defined_draws=reg.ci.min_defined_draws)
            got = ca["paired"]["uni_fed_s1 − 대조선"][name][view]
            assert dumps(got) == dumps(want), (view, name)
            assert got["interval"]["n_defined"] + got["interval"]["n_undefined"] == B
    again = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)[0]["clause_axis"]
    assert dumps(again["paired"]) == dumps(ca["paired"])                         # 같은 W — 결정적


def test_조항_축_점추정은_대응_구간을_더하기_전과_같다(checker, entries, tmp_path: Path) -> None:
    """점추정은 지금 값 그대로(지시 1023d 과제 1) — 더하기 전 모듈(`BEFORE_CI` 의 blob)과 같은 입력에서 점추정 칸이 바이트까지 같다."""
    old = old_module(tmp_path, "evaluation/strict_narrative.py", "strict_narrative_before_ci", commit=BEFORE_CI)
    ids, _, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)))
    old_aux = old.AuxScoring(**{f.name: getattr(aux, f.name) for f in fields(SN.AuxScoring)})
    new = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)[0]
    before = compute(eval_doc=EVAL_DOC_V6, aux_scoring=old_aux)[0]
    assert set(new["clause_axis"]) - set(before["clause_axis"]) == {"interval", "paired", "draws", "unregistered_choices"}
    for key in before["clause_axis"]:
        assert dumps(new["clause_axis"][key]) == dumps(before["clause_axis"][key]), key
    assert dumps(new["narrative"]) == dumps(before["narrative"])


MODEL_PAIRS = tuple(f"uni_fed_s1 − {t}_s1" for t in ("uni_central", "uni_local_C1", "uni_local_C2", "uni_local_C3"))
"""서술 지표의 짝(시드마다 연합 − 중앙 · 연합 − 로컬 k) — 합성 설계는 시드 1 하나다."""


def test_조항_축_모델_짝은_서술_지표의_짝이고_같은_W_로_재현된다(checker, entries) -> None:
    """03 §9-19 열린 질문 14 의 답 — 서술 지표가 내는 모델 짝을 조항 축에서도 같은 추첨 배열 W 로(설계 가중 · 무가중). 짝마다 결정적."""
    ids, group_of, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)))
    out = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)[0]
    ca = out["clause_axis"]
    assert set(out["narrative"]["paired"]) == set(MODEL_PAIRS) <= set(ca["paired"])
    assert ca["unregistered_choices"] == [dict(SN.UNREGISTERED_CLAUSE_PAIRS)]
    reg = registration()
    W = draw_weights(ids, group_of, side, n_resamples=B, rng_seed=reg.ci.rng_seed)
    w = design_weights(ids, side)
    pairs = aux_pairs(aux)
    for other in ("uni_central", "uni_local_C1", "uni_local_C2", "uni_local_C3"):
        a, b = aux.models[("uni_fed", 1)], aux.models[(other, 1)]
        for view, Wv in (("design", W), ("unweighted", W / w[None, :])):
            da = clause_axis_draws(dict(a.cited), pairs, aux.index_clause_ids, ids, Wv)
            db = clause_axis_draws(dict(b.cited), pairs, aux.index_clause_ids, ids, Wv)
            for name in CLAUSE_AXIS_METRICS:
                want = NM.paired_difference(ca["models"]["uni_fed_s1"][view][name], ca["models"][f"{other}_s1"][view][name],
                                            da[name], db[name], alpha=reg.ci.alpha, min_defined_draws=reg.ci.min_defined_draws)
                assert dumps(ca["paired"][f"uni_fed_s1 − {other}_s1"][name][view]) == dumps(want), (other, view, name)
    again = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)[0]["clause_axis"]
    for label in ca["paired"]:
        assert dumps(again["paired"][label]) == dumps(ca["paired"][label]), label        # 짝마다 결정적


def test_짝별_추첨은_점추정과_맞는다(checker, entries) -> None:
    """점추정 = 추첨 — 중복도 1 의 추첨 하나(W = 설계 가중)에서 짝마다 구간의 두 끝이 점추정의 차와 같다(설계 가중 · 무가중). 층마다
    묶음을 한 번씩 몰아 뽑는 균형 추첨에서는 각 모델의 분자 · 분모 평균이 점추정의 분자 · 분모다(W 에 선형)."""
    ids, group_of, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)))
    one = report_direct(aux, ids, group_of, side, W=design_weights(ids, side)[None, :], min_defined_draws=1)["clause_axis"]
    cells = [(label, c) for label, pair in one["paired"].items() for metric in pair.values() for c in metric.values()]
    assert {label for label, _ in cells} == set(one["paired"]) and len(one["paired"]) == len(MODEL_PAIRS) + len(aux.models)
    for label, c in cells:
        if c["state"] == "finite":
            iv = c["interval"]
            assert iv["n_defined"] == 1 and iv["ci_lo"] == pytest.approx(c["diff"], abs=1e-12) == iv["ci_hi"], label
    groups = strata_groups(ids, group_of, side)
    L = int(np.lcm.reduce([len(gs) for gs in groups.values()]))
    M = np.zeros((L, sum(len(gs) for gs in groups.values())), dtype=np.int64)
    col = 0
    for gs in groups.values():
        for r in range(L):
            M[r, col + r % len(gs)] = len(gs)                     # 층마다 한 묶음에 그 층의 수를 몬다 — 평균 중복도 1
        col += len(gs)
    Wb = weights_from(M, ids, group_of, side, groups)
    pairs = aux_pairs(aux)
    base_cited = citations_from_codes(aux.entries, aux.material_of, {i: list(aux.baseline_codes[i]) for i in ids})
    w = design_weights(ids, side)
    for view, Wv in (("design", Wb), ("unweighted", Wb / w[None, :])):
        for label, cited in [*((SM.cell_name(*k), dict(m.cited)) for k, m in aux.models.items()), ("baseline", base_cited)]:
            point = one["models"][label][view] if label != "baseline" else one["baseline"][view]
            parts = clause_axis_draw_parts(cited, pairs, aux.index_clause_ids, ids, Wv)
            for name in ("citation_match_rate", "citation_precision"):
                n, d = parts[name]
                assert (n.mean(), d.mean()) == pytest.approx((point[name]["numerator"], point[name]["denominator"]),
                                                             rel=1e-9), (view, label, name)


Q_HIT = {"uni_central": 1, "uni_fed": 3, "uni_local_C1": 4, "uni_local_C2": 2, "uni_local_C3": 0}
"""모델마다 강재 이미지 차례 j 가운데 j % 5 < q 인 것만 맞는 조항(KRA27-T15), 나머지는 표에 없는 조항(KRA27-X99)을 인용한다. 연합이 맞힌
이미지는 중앙 · C2 · C3 의 것을 품고(차 > 0) C1 의 것 안에 든다(차 < 0) — 가중이 음이 아니면 어느 추첨에서도 같은 부호다."""
PAIR_SIGN = {"uni_central": 1, "uni_local_C1": -1, "uni_local_C2": 1, "uni_local_C3": 1}
METRIC_SIGN = {"citation_match_rate": 1, "citation_precision": 1, "ungrounded_rate": -1}
"""강재 이미지마다 인용이 하나라 정밀도의 분모는 일치율과 같고, 무근거율 = 1 − 정밀도 — 무근거율의 차는 부호가 뒤집힌다."""


def test_조항_축_모델_짝의_차는_짝마다_부호와_값이_맞다(checker, entries) -> None:
    """외부 검토 211 의 중요 1 — 인용을 모델마다 다르게 두어 짝 연결 오류 · 부호 뒤집힘을 가른다. 점추정 차와 추첨 차(구간 두 끝 · 유효
    추첨 수)를 짝마다(연합 − 중앙 · 연합 − 로컬 C1 · C2 · C3) 설계 가중 · 무가중으로, 조항 축 코드를 거치지 않은 기대와 맞댄다."""
    ids, group_of, side, gold = design()
    base = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)))
    order = {i: j for j, i in enumerate(i for i in ids if side[i].stratum.startswith("ST"))}

    def cited_for(tag: str) -> dict:
        return {i: (("KRA27-T15",) if order[i] % 5 < Q_HIT[tag] else ("KRA27-X99",)) if i in order else () for i in ids}

    aux = replace(base, models={k: SN.ModelAux(m.tag, m.seed, m.items, cited_for(k[0])) for k, m in base.models.items()})
    assert "KRA27-T15" in aux.index_clause_ids and "KRA27-X99" not in aux.index_clause_ids
    ca = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)[0]["clause_axis"]
    reg = registration()
    W = draw_weights(ids, group_of, side, n_resamples=B, rng_seed=reg.ci.rng_seed)
    w = design_weights(ids, side)
    pair = np.array([i in order for i in ids], dtype=float)
    hit = {tag: np.array([i in order and order[i] % 5 < q for i in ids], dtype=float) for tag, q in Q_HIT.items()}
    for other, pair_sign in PAIR_SIGN.items():
        label = f"uni_fed_s1 − {other}_s1"
        for view, Wv, wv in (("design", W, w), ("unweighted", W / w[None, :], np.ones(len(ids)))):
            point = (wv @ hit["uni_fed"] - wv @ hit[other]) / (wv @ pair)
            den = Wv @ pair
            with np.errstate(divide="ignore", invalid="ignore"):
                per_draw = np.where(den > 0, (Wv @ hit["uni_fed"] - Wv @ hit[other]) / np.where(den > 0, den, 1.0), np.nan)
            for name, metric_sign in METRIC_SIGN.items():
                s = pair_sign * metric_sign
                got = ca["paired"][label][name][view]
                want_iv, n_def, n_undef = PR.defined_interval(metric_sign * per_draw, reg.ci.alpha)
                where = (label, view, name)
                assert got["state"] == "finite" and np.sign(got["diff"]) == s, where
                assert got["diff"] == pytest.approx(metric_sign * point, abs=1e-12), where
                assert (got["interval"]["n_defined"], got["interval"]["n_undefined"]) == (n_def, n_undef) == (B, 0), where
                assert got["interval"]["ci_lo"] == pytest.approx(want_iv.lo.value, abs=1e-12), where
                assert got["interval"]["ci_hi"] == pytest.approx(want_iv.hi.value, abs=1e-12), where
                defined = metric_sign * per_draw[~np.isnan(per_draw)]
                assert (s * defined >= 0).all() and (s * defined > 0).any(), where          # 추첨마다 같은 부호
                if s > 0:
                    assert got["interval"]["ci_lo"] >= 0 and got["interval"]["ci_hi"] > 0, where
                else:
                    assert got["interval"]["ci_hi"] <= 0 and got["interval"]["ci_lo"] < 0, where
    assert ca["paired"]["uni_fed_s1 − uni_central_s1"]["citation_match_rate"]["design"]["diff"] != \
        ca["paired"]["uni_fed_s1 − uni_local_C2_s1"]["citation_match_rate"]["design"]["diff"]     # 짝마다 다른 값


def test_유효_추첨이_하한보다_적으면_방향을_말하지_않는다(checker, entries) -> None:
    """서술 지표와 같은 규칙 — 구간 값과 유효 · 미정의 수는 싣고, 유효 추첨이 하한보다 적으면 방향 · 판정을 내지 않는다."""
    ids, group_of, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)))
    low = report_direct(aux, ids, group_of, side, min_defined_draws=B + 1)["clause_axis"]["paired"]
    cells = [c for pair in low.values() for metric in pair.values() for c in metric.values() if c["state"] == "finite"]
    assert cells and all(c["direction"] is None and "하한" in c["direction_reason"] for c in cells)
    assert all(c["interval"]["n_defined"] <= B for c in cells)
    ok = report_direct(aux, ids, group_of, side, min_defined_draws=1)["clause_axis"]["paired"]
    assert any(c["direction"] is not None for pair in ok.values() for metric in pair.values() for c in metric.values()
               if c["state"] == "finite")


def test_분모가_빈_추첨은_미정의로_세고_뺀다(checker, entries) -> None:
    """W 의 한 행이 모두 0 이면 그 추첨은 세 지표 모두 분모 0 — 미정의(NaN)로 세고 구간에서 뺀다. 나머지 B 추첨만 유효하다."""
    ids, group_of, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)))
    reg = registration()
    W = draw_weights(ids, group_of, side, n_resamples=B, rng_seed=reg.ci.rng_seed)
    paired = report_direct(aux, ids, group_of, side, W=np.vstack([W, np.zeros((1, len(ids)))]))["clause_axis"]["paired"]
    cells = [c for pair in paired.values() for metric in pair.values() for c in metric.values() if c["state"] == "finite"]
    assert cells and all((c["interval"]["n_defined"], c["interval"]["n_undefined"]) == (B, 1) for c in cells)


@pytest.mark.parametrize("how", ["ones", "one_missing"])
def test_가중이_이미지_가중과_다르면_멈춘다(checker, entries, how: str) -> None:
    """조항 축의 가중은 서술 지표 항목의 가중과 같은 값이어야 하고, 모집단의 이미지마다 있어야 한다 — 1 로 메우지 않는다."""
    ids, _, side, gold = design()
    w0 = dict(zip(ids, (float(w) for w in design_weights(ids, side)), strict=True))
    weights = dict.fromkeys(ids, 1.0) if how == "ones" else {i: w for i, w in w0.items() if i != ids[0]}
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)), weights=weights)
    with pytest.raises(SN.AuxInputError, match="가중"):
        compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)


def test_탐색적_종류는_금지_주장률_표에_표지가_붙는다(checker, entries) -> None:
    """199 결정 §2 · 203 §4 — (마)와 (마)를 품는 전체 금지 주장률에 탐색적 표지. 다른 종류 · 유보 · 상태에는 붙지 않는다."""
    ids, _, side, gold = design()
    out = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux_for(checker, entries, ids, side, list(scores_for(ids, gold))))[0]
    nar = out["narrative"]
    for cell in nar["models"].values():
        assert cell["forbidden.마"]["status"] == SN.EXPLORATORY_KIND
        assert cell["forbidden.any"]["status"] == SN.EXPLORATORY_TOTAL
        assert not [k for k, v in cell.items() if "status" in v and k not in ("forbidden.마", "forbidden.any")]
    assert nar["exploratory"]["kinds"] == ["마"] and nar["exploratory"]["metrics"] == ["forbidden.마", "forbidden.any"]
    for pair in nar["paired"].values():
        assert pair["narrative_forbidden_rate"]["status"] == SN.EXPLORATORY_TOTAL
        assert "status" not in pair["narrative_reservation_rate"]
    assert nar["identity_checks"] == {"code_tree": "checked"}


def test_탐색적_종류가_없으면_표지가_없다(checker, entries) -> None:
    ids, _, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)), exploratory=())
    nar = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)[0]["narrative"]
    assert not [k for cell in nar["models"].values() for k, v in cell.items() if "status" in v]
    assert nar["exploratory"]["kinds"] == [] and nar["exploratory"]["metrics"] == []


@pytest.mark.parametrize("invalid", [(), ("env:model",)])
def test_신원이_거부되면_보조_지표만_빠지고_사유가_남는다(invalid) -> None:
    """지시 1023c 과제 2 — `AuxRefused` 는 두 블록 대신 `aux_identity` 를 싣고, 원한 이름의 `not_computed` 를 거부 사유로 바꾼다."""
    plain, _ = compute(eval_doc=EVAL_DOC_V6, invalid=invalid)
    refused = SN.AuxRefused(wanted=frozenset(ALL_NAMES), problems=("narrative.checker_sha256: 등록 'x' · 실제 'y'",),
                            identity_checks={"code_tree": "checked", "prompt": "not_given"})
    out, _ = compute(eval_doc=EVAL_DOC_V6, aux_scoring=refused, invalid=invalid)
    assert set(out) - set(plain) == {"aux_identity"} and not {"narrative", "clause_axis"} & set(out)
    for key in plain:
        if key != "not_computed":
            assert dumps(out[key]) == dumps(plain[key]), key
    assert {n: out["not_computed"][n] for n in ALL_NAMES} == dict.fromkeys(ALL_NAMES, SN.REFUSED)
    assert out["aux_identity"]["status"] == "refused" and out["aux_identity"]["names"] == list(ALL_NAMES)
    assert out["aux_identity"]["checks"] == {"code_tree": "checked", "prompt": "not_given"}


def test_이름이_빠진_보조_지표는_그_블록을_내지_않는다(checker, entries) -> None:
    ids, _, side, gold = design()
    aux = aux_for(checker, entries, ids, side, list(scores_for(ids, gold)), wanted=CLAUSE_AXIS_METRICS)
    out, _ = compute(eval_doc=EVAL_DOC_V6, aux_scoring=aux)
    assert "clause_axis" in out and "narrative" not in out
    assert set(SN.NARRATIVE_NAMES) <= set(out["not_computed"])


# ── 입력 모으기

def rec(i, codes=("2011",), cited=("KRA27-T15",)):
    return SimpleNamespace(image_id=i, defects=[SimpleNamespace(iso_code=c) for c in codes], cited_clauses=list(cited))


def line(i, text='{"defects":[],"verdict":"판정불가","cited_clauses":[]}', **kw):
    return json.dumps({"image_id": i, "text": text, **kw}, ensure_ascii=False)


def test_입력은_모집단_순서로_줄과_레코드를_맞춘다() -> None:
    ids = ["a", "b"]
    m = SN.model_inputs("uni_fed", 1, [line("b", gen_stop="eos"), line("a", gen_stop="length"), line("z", gen_stop="eos")],
                        [rec("a"), rec("b", codes=()), rec("z")], image_ids=ids, material_of={"a": "ST", "b": "AL"},
                        gold_codes={"a": {"2011"}}, weights=[2.0, 1.0])
    assert [it.image_id for it in m.items] == ids and [it.extract.state for it in m.items] == ["missing", "missing"]
    assert m.items[0].extract.stop_kept == "length" and m.items[0].weight == 2.0 and m.items[1].structured_codes == ()
    assert m.items[0].gold_codes == ("2011",) and m.cited == {"a": ("KRA27-T15",), "b": ("KRA27-T15",)}
    no_stop = SN.model_inputs("uni_fed", 1, [line("a")], [rec("a")], image_ids=["a"], material_of={"a": "ST"},
                              gold_codes={}, weights=[1.0])
    assert no_stop.items[0].extract.state == VALIDATION_ERROR                 # 칸 없음 = 인자를 빼고 부른다


@pytest.mark.parametrize("lines,records,word", [
    ([line("a", gen_stop="eos")], [rec("a"), rec("b")], "하나씩"),
    ([line("a", gen_stop="eos"), line("a", gen_stop="eos"), line("b", gen_stop="eos")], [rec("a"), rec("b")], "둘"),
    ([line("a", gen_stop="eos"), line("b", gen_stop="eos")], [rec("a"), rec("a"), rec("b")], "둘"),
])
def test_줄과_레코드가_모자라거나_겹치면_멈춘다(lines, records, word) -> None:
    with pytest.raises(SN.AuxInputError, match=word):
        SN.model_inputs("uni_fed", 1, lines, records, image_ids=["a", "b"], material_of={"a": "ST", "b": "ST"},
                        gold_codes={}, weights=[1.0, 1.0])


def test_재질은_층_이름의_머리에서_낸다() -> None:
    side = {"a": SimpleNamespace(stratum="ST|1"), "b": SimpleNamespace(stratum="AL|2-3"), "c": SimpleNamespace(stratum="XX|1")}
    assert SN.material_of_side(side, ["a", "b"]) == {"a": "ST", "b": "AL"}
    with pytest.raises(SN.AuxInputError):
        SN.material_of_side(side, ["c"])

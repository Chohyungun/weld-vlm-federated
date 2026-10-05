"""출처 진단의 채택 구간 규약 판(`evaluation.discrimination_rule`) — 07번 §33-6 의 적용 범위.

계약 파일 `evaluation.discrimination` 은 바이트가 고정돼 있어 방식 인자를 새 파일에 둔다. 여기는 그 새 파일이
(1) 방식이 없으면 계약 파일과 같은 것을 내는지, (2) 방식을 주면 규약 id · 상태 칸을 싣고 점추정은 그대로인지,
(3) 정의된 추첨이 없을 때 "판별 있음" 으로 떨어지지 않는지를 본다.
"""

from __future__ import annotations

import json
import random

import pytest

from evaluation import discrimination as D
from evaluation import discrimination_rule as DR
from evaluation import percentile_rule as PR
from evaluation.stats import cluster_bootstrap
from tests.test_discrimination import CROP, TILE, ctx, rec


def world(n_groups: int = 30, seed: int = 3):
    """묶음마다 결함 한 장 · 정상 한 장. 결함은 대개 발화하고 정상은 가끔 발화한다."""
    rnd = random.Random(seed)
    items, records = [], []
    for g in range(n_groups):
        items += [(f"d{g}", f"g{g}", True, CROP), (f"n{g}", f"g{g}", False, CROP), (f"t{g}", f"g{g}", False, TILE)]
        records += [rec(f"d{g}", ["2011"] if rnd.random() < 0.8 else []),
                    rec(f"n{g}", ["2011"] if rnd.random() < 0.3 else []), rec(f"t{g}", [])]
    return records, ctx(items)


def test_방식이_없으면_계약_파일과_같다() -> None:
    records, contexts = world()
    assert DR.score_discrimination(records, contexts, method=None) == D.score_discrimination(records, contexts)
    scores = {"sep_central": {"a": 0.9, "b": 0.1}, "sep_fed": {"a": 0.8, "b": 0.3}, "sep_local_C1": {"a": 0.4, "b": 0.5}}
    args = (scores, ["a"], ["b"], {"g1": ["a"], "g2": ["b"]})
    assert DR.score_threshold_free(*args, method=None) == D.score_threshold_free(*args)


def test_방식을_주면_규약_id_와_상태를_싣고_점추정은_그대로다() -> None:
    records, contexts = world()
    old = D.score_discrimination(records, contexts)
    new = DR.score_discrimination(records, contexts)
    d = new.as_dict()["delta"]
    assert d["convention_id"] == PR.CONVENTION_ID
    assert (d["ci_lo_state"], d["ci_hi_state"], d["half_width_state"]) == ("finite", "finite", "finite")
    assert d["point"] == old.delta.point and new.n_groups == old.n_groups and new.fire_rate_defect == old.fire_rate_defect
    json.dumps(new.as_dict(), allow_nan=False)


def test_정의된_추첨이_없으면_산출_불가다_판별_있음으로_떨어지지_않는다() -> None:
    """계약 파일의 판정 문구는 끝점이 NaN 이면 비교가 모두 거짓이라 "판별 있음" 이 된다 — 방식 경로는 그 전에 가른다."""
    absent = cluster_bootstrap(["a", "b"], lambda gs: float("nan"), drop_undefined=True, method=PR.PERCENTILE_RULE)
    assert absent.rule.lo.state == "absent"
    assert DR._verdict(absent, 3, 3) == DR.NO_INTERVAL
    old_path = cluster_bootstrap(["a", "b"], lambda gs: float("nan"), drop_undefined=True)
    assert D._verdict(old_path, 3, 3).startswith("판별 있음"), "계약 파일의 문구 — 고정 파일이라 여기서 고치지 않는다"
    assert DR._verdict(absent, 0, 3) == D._verdict(absent, 0, 3), "결함 · 정상이 없으면 계약 파일의 산출 불가 문구 그대로"


def test_결함이나_정상이_없으면_구간은_없음이고_규약_id_를_싣는다() -> None:
    records = [rec("d1", ["2011"]), rec("d2", [])]
    contexts = ctx([("d1", "g1", True, CROP), ("d2", "g2", True, CROP)])
    got = DR.score_discrimination(records, contexts)
    d = got.as_dict()["delta"]
    assert d["ci_lo_state"] == "absent" and d["convention_id"] == PR.CONVENTION_ID
    assert got.verdict == D.score_discrimination(records, contexts).verdict


def test_임계_독립판도_점추정과_회복률은_그대로고_구간만_규약을_따른다() -> None:
    rnd = random.Random(11)
    ids_d = [f"d{i}" for i in range(40)]
    ids_n = [f"n{i}" for i in range(40)]
    by_group = {f"g{i}": [ids_d[i], ids_n[i]] for i in range(40)}
    scores = {c: {i: rnd.random() + (shift if i.startswith("d") else 0.0) for i in ids_d + ids_n}
              for c, shift in (("sep_central", 0.6), ("sep_fed", 0.45), ("sep_local_C1", 0.1), ("sep_local_C2", 0.2))}
    old = D.score_threshold_free(scores, ids_d, ids_n, by_group)
    new = DR.score_threshold_free(scores, ids_d, ids_n, by_group)
    for c in old["by_cell"]:
        assert new["by_cell"][c]["point"] == old["by_cell"][c]["point"]
        assert new["by_cell"][c]["convention_id"] == PR.CONVENTION_ID
    for k in old["contrasts"]:
        assert new["contrasts"][k]["point"] == old["contrasts"][k]["point"]
    assert new["recovery_pct"] == pytest.approx(old["recovery_pct"]) and new["convention_id"] == PR.CONVENTION_ID
    assert {k: v for k, v in new.items() if isinstance(v, str) and k != "convention_id"} \
        == {k: v for k, v in old.items() if isinstance(v, str)}, "설명 칸은 계약 파일의 글 그대로"
    json.dumps(new, allow_nan=False)


def test_계약_파일은_건드리지_않았다() -> None:
    """방식 인자를 계약 파일에 더하지 않았다 — 고정 해시 시험이 따로 보지만, 이 파일이 왜 따로 있는지를 여기서도 고정한다."""
    import inspect
    assert "method" not in inspect.signature(D.score_discrimination).parameters
    assert "method" not in inspect.signature(D.score_threshold_free).parameters


def test_통합형_진입점은_출처_진단을_계약_파일에서_직접_부르지_않는다() -> None:
    """적용 범위의 그물 — 통합형 산출물의 구간은 채택 규약이다(§33-6). 진입점이 출처 진단을 부르게 되면 방식 판(`discrimination_rule`)으로만."""
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "scripts/probe/score_unified.py").read_text(encoding="utf-8")
    mods = {n.module for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)}
    mods |= {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names}
    assert "evaluation.discrimination" not in mods
    assert not any(isinstance(n, ast.alias) and n.name == "discrimination" for n in ast.walk(ast.parse(src)))

"""프레임 진단의 문턱 절차 — 합성 사례 위에서 가능 구간의 가운데(07번 §30-2 의 3 · §32-7 의 ①).

문턱 **값**을 고정하지 않는다 — 값은 등록이 정한다. 여기는 절차의 성질을 본다: 가운데에 섰는가, 그때 모든 사례가 기대 판정을 내는가,
처음 값이 안 되면 멈추는가, 기대 판정 파일을 고치지 않는가, 요구 표본 수의 식, 대상 장 수의 경계.
"""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation import frame_diag as F
from evaluation import frame_diag_cases as C
from evaluation import frame_diag_thresholds as T


@pytest.fixture(scope="module")
def world():
    exp = T.expected()
    stats = T.case_stats(T.START)
    return exp, stats, T.center(stats, exp)


def test_가운데에_서고_모든_사례가_기대_판정을_낸다(world) -> None:
    exp, stats, res = world
    assert res["status"] == "centered" and res["violations"] == []
    rules = F.FrameRules(**res["rules"])
    for cid, st in stats.items():
        assert T.matches(F.verdict_of(st, rules, C.MODEL_INPUT_WH), exp[cid]), cid


def test_정한_값은_자기_구간의_가운데다(world) -> None:
    _, _, res = world
    for name in T.ORDER:
        lo, hi = res["intervals"][name]
        v = res["rules"][name]
        assert lo <= v <= hi, name
        step = T.GRID[name][2]
        assert abs(v - (lo + hi) / 2) <= step, f"{name}: 가운데에서 한 칸 넘게 벗어났다"


def test_구간의_끝을_정한_사례가_적혀_있다(world) -> None:
    """끝에서 한 칸 밖의 값은 어떤 사례를 깨뜨린다 — 그 사례가 구간을 정했다(격자 끝이면 그렇게 적는다)."""
    _, _, res = world
    for name, b in res["binding"].items():
        assert b["lo"] and b["hi"], name


def test_처음_값이_안_되면_멈추고_사례를_말한다(world) -> None:
    exp, stats, _ = world
    res = T.center(stats, exp, start=replace(T.START, s_min=0.45))
    assert res["status"] == "stopped" and "바" in res["violations"]


def test_절차는_기대_판정_파일을_쓰지_않는다() -> None:
    """기대 판정을 고쳐 문턱에 맞추지 않는다(§30-2 의 3) — 절차 모듈에 파일을 쓰는 호출이 없다."""
    tree = ast.parse(Path(T.__file__).read_text(encoding="utf-8"))
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"write_text", "write_bytes", "open", "dump"}


def test_맞음은_상태_후보_축_사유_순서를_다_본다() -> None:
    entry = {"allowed": [{"status": F.FAIL, "candidate": F.CAND_RESIZE, "axes": "y"}], "steps": [4]}
    v = F.FrameVerdict(F.FAIL, F.CAND_RESIZE, "y · 리사이즈", 1, 1, 0, 0, 1, step=4)
    assert T.matches(v, entry)
    assert not T.matches(replace(v, step=7), entry), "순서가 다르다"
    assert not T.matches(replace(v, reason="x·y · 리사이즈"), entry), "축이 다르다"
    assert not T.matches(replace(v, candidate=F.CAND_NONE, reason="y · 후보 없음"), entry), "후보가 다르다"
    ind = {"allowed": [{"status": F.INDETERMINATE, "reason": F.REASON_SAMPLE}], "steps": [6]}
    assert T.matches(F.FrameVerdict(F.INDETERMINATE, None, F.REASON_SAMPLE, 1, 1, 0, 0, 1, step=6), ind)
    assert not T.matches(F.FrameVerdict(F.INDETERMINATE, None, F.REASON_WEAK_TRACKING, 1, 1, 0, 0, 1, step=6), ind)


def test_요구_표본_수는_표준오차가_루트_n_으로_준다고_본_식이다(world) -> None:
    _, stats, res = world
    rules = F.FrameRules(**res["rules"])
    rows = T.required_sizes(rules, stats)
    row = next(r for r in rows if r["noise_step"] == 0.2 and r["stat"] == "rbar")
    st = stats["사0.2"]
    se = st.se["rbar"][1]
    assert row["n_required"] == math.ceil(st.n_images * (se / (rules.w_mix / 3)) ** 2)
    assert {r["noise_step"] for r in rows} == set(C.NOISE_LADDER)


def test_요구_표본_수에는_묶음과_쌍이_있다(world) -> None:
    """검토(10-02 병합 전) 4 — 07번 §31-2 의 요구 쌍 수 · 묶음 수. 묶음 단위 부트스트랩이라 묶음 수가 독립 표본의 요구다."""
    _, stats, res = world
    rules = F.FrameRules(**res["rules"])
    rows = T.required_sizes(rules, stats)
    st = stats["사0.2"]
    m = next(r for r in rows if r["noise_step"] == 0.2 and r["stat"] == "m" and r["axis"] == "y")
    factor = (st.se["m"][1] / (rules.axis_ratio_width / 3)) ** 2
    assert (m["q0"], m["g0"]) == (st.n_pairs, st.n_pair_groups)
    assert m["pairs_required"] == math.ceil(st.n_pairs * factor)
    assert m["groups_required"] == math.ceil(st.n_pair_groups * factor)
    r = next(r for r in rows if r["noise_step"] == 0.2 and r["stat"] == "r" and r["axis"] == "y")
    assert r["q0"] is None and r["pairs_required"] is None and r["g0"] == st.n_image_groups
    assert all(set(x) == {"noise_step", "stat", "axis", "width", "n0", "n_required", "g0", "groups_required", "q0",
                          "pairs_required"} for x in rows)


def test_대상_장_수의_경계(world) -> None:
    _, _, res = world
    rules = F.FrameRules(**res["rules"])
    b = T.boundary_check(rules)
    assert b[str(rules.n_img - 1)]["step"] == 1 and b[str(rules.n_img)]["step"] != 1


def test_묶음_수의_경계(world) -> None:
    """장은 넉넉하고 묶음만 `min_groups − 1` 이면 1 에서, `min_groups` 면 1 을 지난다."""
    _, _, res = world
    rules = F.FrameRules(**res["rules"])
    b = T.boundary_check(rules)
    assert b[f"groups{rules.min_groups - 1}"]["step"] == 1 and b[f"groups{rules.min_groups}"]["step"] != 1


def test_δ_의_격자는_원본과_리사이즈를_가르는_범위_안이다() -> None:
    k = C.MODEL_INPUT_WH[1] / C.H
    assert T.GRID["delta"][1] < abs(1 - k) / 2

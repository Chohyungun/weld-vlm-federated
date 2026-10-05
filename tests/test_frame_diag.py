"""프레임 진단 검사기 — 통계 · 짝짓기 · 묶음 부트스트랩 · 결합 순서 · 후보 서명(07번 §30-2 · §31-2 · §32-5).

여기의 문턱(`R`)은 **시험용**이다 — 등록 문턱이 아니다. 등록 문턱은 합성 사례 위의 문턱 절차(§30-2 의 3)가 정한다.
결합의 가지는 통계를 손으로 지은 `FrameStats` 로 하나씩 세운다 — 합성 사례에 기대면 가지 하나가 빠져도 모른다.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from evaluation import frame_diag as F
from evaluation import frame_diag_cases as C

REPO = Path(__file__).resolve().parents[1]
EXPECTED = REPO / "evaluation" / "frame_diag_expected-20261001-1.json"
WH = C.MODEL_INPUT_WH
R = F.FrameRules(n_img=50, s_min=0.74, w_s=0.05, w_r=0.008, w_mix=0.0015, delta=0.005, trim=0.0,
                 n_boot=100, boot_seed=1)
"""시험용 문턱 — 가지를 세우는 데만 쓴다."""


def stats(**over) -> F.FrameStats:
    """지나는 통계 하나 — 덮어쓴 칸만 바뀐다. 값은 축마다 `[x, y]`."""
    k = WH[1] / C.H
    value = {"r": [1.0, 1.0], "rbar": [float("nan"), 1.0], "s": [1.0, 1.0], "sx": [0.0, 0.0], "m": [1.0, 1.0]}
    se = {"r": [0.0, 0.0], "rbar": [float("inf"), 0.0], "s": [0.0, 0.0], "sx": [0.0, 0.0], "m": [0.0, 0.0]}
    valid = {key: [R.n_boot, R.n_boot] for key in value}
    counts = {"n_image_groups": 200, "n_pair_groups": 200}
    for key, v in over.items():
        if key.startswith("se_"):
            se[key[3:]] = v
        elif key.startswith("valid_"):
            valid[key[6:]] = v
        elif key in counts:
            counts[key] = v
        else:
            value[key] = v
    assert 0 < k < 1
    return F.FrameStats(n_images=over.get("n", 500) if "n" in over else 500, n_pairs=500, n_no_pred=0,
                        n_count_mismatch=0, n_groups=200, size=(C.W, C.H), size_mixed=False, value=value, se=se,
                        n_valid=valid, **counts)


def verdict(st, rules=R):
    return F.verdict_of(st, rules, WH)


# ---------------------------------------------------------------- 짝짓기

def test_쌍은_서로의_최대_IoU_상대이고_IoU_가_0_보다_크다() -> None:
    a = (0.0, 0.0, 10.0, 10.0)
    b = (5.0, 0.0, 15.0, 10.0)
    far = (100.0, 100.0, 110.0, 110.0)
    assert F.mutual_pairs([a], [a]) == [(0, 0)]
    assert F.mutual_pairs([a], [far]) == [], "겹치지 않으면 쌍이 아니다"
    assert F.mutual_pairs([a, b], [b]) == [(1, 0)], "정답 b 의 최대 상대는 예측 b 다"
    assert F.mutual_pairs([], [a]) == [] and F.mutual_pairs([a], []) == []


def test_최대가_동률이면_쌍이_아니다() -> None:
    g = (0.0, 0.0, 10.0, 10.0)
    left, right = (-5.0, 0.0, 5.0, 10.0), (5.0, 0.0, 15.0, 10.0)
    assert F.mutual_pairs([left, right], [g]) == [], "정답 쪽에서 동률"
    assert F.mutual_pairs([g], [left, right]) == [], "예측 쪽에서 동률"


# ---------------------------------------------------------------- 통계

def test_원본_프레임의_통계는_1_이고_흔들림이_없다() -> None:
    st = F.frame_stats(C.build(C.Case("가", "", 300, C.identity)), WH, n_boot=50, boot_seed=1, trim=0.0)
    for key in ("r", "s", "m"):
        assert st.value[key] == pytest.approx([1.0, 1.0])
        assert st.se[key] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert st.value["rbar"][1] == pytest.approx(1.0) and st.value["rbar"][0] != st.value["rbar"][0], "배율 1 인 x 축은 섞임 통계가 없다"
    assert st.n_pairs >= st.n_images


def test_리사이즈_프레임은_y_의_비와_기울기가_배율이다() -> None:
    st = F.frame_stats(C.build(C.Case("나", "", 300, C.resize_y)), WH, n_boot=50, boot_seed=1, trim=0.0)
    k = WH[1] / C.H
    assert st.value["r"] == pytest.approx([1.0, k]) and st.value["s"] == pytest.approx([1.0, k])


def test_뒤바뀜은_교차_기울기가_1_이다() -> None:
    st = F.frame_stats(C.build(C.Case("파", "", 300, C.swapped)), WH, n_boot=50, boot_seed=1, trim=0.0)
    assert st.value["sx"] == pytest.approx([1.0, 1.0])
    # 뒤바뀐 x 의 기울기는 정답 x·y 의 상관 × (y 흩어짐 ÷ x 흩어짐 ≈ 0.27) 이라 작다 — 이것 하나로 기울기 단계에 걸린다.
    # y 의 기울기는 상관 × 3.6 이라 우연히 클 수 있다
    assert abs(st.value["s"][0]) < 0.3


def test_대상_장은_예측_수가_정답_수와_같은_결함_장이다() -> None:
    g = (100.0, 100.0, 140.0, 140.0)
    h = (300.0, 300.0, 340.0, 340.0)
    ims = [F.FrameImage("a", "g1", C.W, C.H, (g,), (g,)),
           F.FrameImage("b", "g1", C.W, C.H, (), (g,)),                 # 예측 없음
           F.FrameImage("c", "g2", C.W, C.H, (g,), (g, h)),             # 수가 다르다
           F.FrameImage("d", "g3", C.W, C.H, (g,), ())]                 # 정상 장 — 결함 장이 아니다
    st = F.frame_stats(ims, WH, n_boot=10, boot_seed=1, trim=0.0)
    assert (st.n_images, st.n_no_pred, st.n_count_mismatch) == (1, 1, 1)
    assert st.n_pairs == 2, "쌍은 예측과 정답이 있는 결함 장에서 — 수가 달라도 짝은 선다"


def test_크기가_섞이면_판정_불가() -> None:
    g = (100.0, 100.0, 140.0, 140.0)
    ims = [F.FrameImage("a", "g1", C.W, C.H, (g,), (g,)), F.FrameImage("b", "g2", 1920, 1080, (g,), (g,))]
    v = F.judge(ims, WH, R)
    assert (v.status, v.reason) == (F.INDETERMINATE, F.REASON_SIZE_MIXED)


def test_결함_장이_없으면_판정_불가() -> None:
    ims = [F.FrameImage("a", "g1", C.W, C.H, (), ())]
    assert F.judge(ims, WH, R).reason == F.REASON_NO_DEFECT


def test_부트스트랩은_씨앗이_같으면_같고_묶음_단위로_흔든다() -> None:
    ims = C.build(C.Case("사", "", 200, lambda g: C.noisy(g, 0.3, 5)))
    a = F.frame_stats(ims, WH, n_boot=60, boot_seed=7, trim=0.0)
    b = F.frame_stats(ims, WH, n_boot=60, boot_seed=7, trim=0.0)
    assert a.se == b.se
    one = [replace(im, group_id="all") for im in ims]
    st = F.frame_stats(one, WH, n_boot=60, boot_seed=7, trim=0.0)
    assert st.se["r"] == pytest.approx([0.0, 0.0], abs=1e-12), "묶음이 하나면 뽑을 때마다 같은 집합이다"


def test_표준오차는_장이_아니라_묶음을_뽑아_낸다() -> None:
    """묶음마다 두 장이 반대로 움직이면(비 1+d · 1−d) 묶음째 뽑을 때 섞임 통계(평균)는 늘 1 이다 — 장을 따로 뽑으면 흔들린다."""
    ims = []
    for i in range(80):
        cy = 300.0 + i
        g = (600.0, cy - 20, 640.0, cy + 20)
        for j, k in enumerate((1.01, 0.99)):
            p = (600.0, cy * k - 20, 640.0, cy * k + 20)
            ims.append(F.FrameImage(f"i{i}_{j}", f"g{i}", C.W, C.H, (p,), (g,)))
    st = F.frame_stats(ims, WH, n_boot=60, boot_seed=3, trim=0.0)
    assert st.se["rbar"][1] == pytest.approx(0.0, abs=1e-12)


def test_절사_평균은_양쪽에서_자른다() -> None:
    import numpy as np
    assert F._trimmed_mean(np.array([0.0, 1.0, 1.0, 1.0, 100.0]), 0.2) == pytest.approx(1.0)
    assert F._trimmed_mean(np.array([0.0, 1.0, 1.0, 1.0, 100.0]), 0.0) == pytest.approx(20.6)


# ---------------------------------------------------------------- 결합 — 가지마다

def test_통과() -> None:
    v = verdict(stats())
    assert (v.status, v.step) == (F.PASS, None)


def test_1_대상_장_수가_모자라면_표본_부족() -> None:
    st = replace(stats(), n_images=R.n_img - 1)
    assert (verdict(st).status, verdict(st).reason, verdict(st).step) == (F.INDETERMINATE, F.REASON_SAMPLE, 1)


def test_1_대상_장의_묶음이_모자라면_표본_부족() -> None:
    """검토(10-02 병합 전) 1 — 장 수는 넉넉해도 독립 묶음이 하한 아래면 판정 불가다."""
    v = verdict(stats(n_image_groups=R.min_groups - 1))
    assert (v.status, v.reason, v.step) == (F.INDETERMINATE, F.REASON_SAMPLE, 1)
    assert v.samples["image_groups"] == F.UNMET and v.samples["images"] == F.MET
    assert verdict(stats(n_image_groups=R.min_groups)).status == F.PASS


def test_묶음_하나에_몰린_표본은_표준오차가_0_이어도_통과하지_않는다() -> None:
    """검토(10-02 병합 전) 1 — 대상 장 · 쌍이 넉넉하고 중심이 다양한 항등 예측이 한 묶음에만 있으면, 묶음째 뽑을 때마다
    같은 집합이라 표준오차가 0 이다. 그 0 으로 통과하지 않는다."""
    ims = [replace(im, group_id="all") for im in C.build(C.Case("가", "", 400, C.identity))]
    v = F.judge(ims, WH, R)
    assert (v.status, v.reason, v.step) == (F.INDETERMINATE, F.REASON_SAMPLE, 1)
    assert (v.n_image_groups, v.n_pair_groups) == (1, 1)


@pytest.mark.parametrize("key", ["r", "s", "rbar"])
def test_3_유효_재표집이_모자라면_표준오차를_쓰지_않는다(key: str) -> None:
    """검토(10-02 병합 전) 1 — 유한한 재표집이 둘만 남아도 표준오차는 나온다. 그 값으로 게이트를 지나지 않는다."""
    need = math.ceil(R.min_valid_boot * R.n_boot)
    v = verdict(stats(**{f"valid_{key}": [R.n_boot, need - 1]}))
    assert (v.status, v.reason, v.step) == (F.INDETERMINATE, F.REASON_SAMPLE, 3)
    assert verdict(stats(**{f"valid_{key}": [R.n_boot, need]})).status == F.PASS


def test_6_쌍의_묶음이나_유효_재표집이_모자라면_표본_부족() -> None:
    v = verdict(stats(n_pair_groups=R.min_pair_groups - 1))
    assert (v.status, v.reason, v.step) == (F.INDETERMINATE, F.REASON_SAMPLE, 6)
    assert v.samples["pair_groups"] == F.UNMET
    v = verdict(stats(valid_m=[2, 2]))
    assert (v.status, v.reason, v.step) == (F.INDETERMINATE, F.REASON_SAMPLE, 6)


def test_2_뒤바뀜도_유효_재표집이_모자라면_내지_않는다() -> None:
    v = verdict(stats(s=[0.0, 0.1], sx=[1.0, 0.99], valid_sx=[R.n_boot, 2]))
    assert (v.status, v.reason) == (F.INDETERMINATE, F.REASON_NOT_TRACKING)


def test_검사기의_표본_하한은_규칙이_검사한다() -> None:
    probs = replace(R, min_groups=1, min_pair_groups=1.5, min_valid_boot=0.0).problems((1.0, 704 / 720))
    assert any("min_groups" in m for m in probs) and any("min_pair_groups" in m for m in probs)
    assert any("min_valid_boot" in m for m in probs)


def test_2_기울기가_하한_아래면_위치를_따라가지_않는다() -> None:
    v = verdict(stats(s=[1.0, 0.5]))
    assert (v.status, v.reason, v.step) == (F.INDETERMINATE, F.REASON_NOT_TRACKING, 2)


def test_2_교차_기울기가_1_이고_흔들림이_작으면_뒤바뀜() -> None:
    v = verdict(stats(s=[0.0, 0.1], sx=[1.0, 0.99]))
    assert (v.status, v.candidate, v.step) == (F.FAIL, F.CAND_SWAP, 2)


def test_2_뒤바뀜도_표준오차_게이트를_지나야_낸다() -> None:
    v = verdict(stats(s=[0.0, 0.1], sx=[1.0, 1.0], se_sx=[R.w_s, 0.0]))
    assert (v.status, v.reason) == (F.INDETERMINATE, F.REASON_NOT_TRACKING)


@pytest.mark.parametrize("key", ["se_r", "se_s", "se_rbar"])
def test_3_표준오차_게이트(key: str) -> None:
    width = {"se_r": R.w_r, "se_s": R.w_s, "se_rbar": R.w_mix}[key]
    v = verdict(stats(**{key: [0.0, width / 3 * 1.01]}))
    assert (v.status, v.reason, v.step) == (F.INDETERMINATE, F.REASON_SAMPLE, 3)
    assert verdict(stats(**{key: [0.0, width / 3 * 0.99]})).status == F.PASS


def test_3_배율이_1_인_축의_섞임_통계는_보지_않는다() -> None:
    assert verdict(stats(se_rbar=[float("inf"), 0.0], rbar=[float("nan"), 1.0])).status == F.PASS


@pytest.mark.parametrize(("over", "axes", "cand"), [
    ({"r": [1.0, 704 / 720], "s": [1.0, 704 / 720], "rbar": [float("nan"), 704 / 720]}, "y", F.CAND_RESIZE),
    ({"r": [1000 / 1280, 1000 / 720], "s": [1000 / 1280, 1000 / 720], "rbar": [float("nan"), 1000 / 720]},
     "x·y", F.CAND_NORM),
    ({"r": [1.0, 0.978], "s": [1.0, 1.0], "rbar": [float("nan"), 0.978]}, "y", F.CAND_NONE),
    ({"rbar": [float("nan"), 0.9956]}, "y", F.CAND_MIX),
    ({"rbar": [float("nan"), 1.0044]}, "y", F.CAND_NONE),
])
def test_4_중심_비가_벗어나면_불통과와_서명(over: dict, axes: str, cand: str) -> None:
    v = verdict(stats(**over))
    assert (v.status, v.candidate, v.step) == (F.FAIL, cand, 4)
    assert v.reason == f"{axes} · {cand}"


def test_5_기울기만_벗어나면_위치를_덜_따라간다() -> None:
    v = verdict(stats(s=[0.8, 0.8]))
    assert (v.status, v.reason, v.step) == (F.INDETERMINATE, F.REASON_WEAK_TRACKING, 5)


def test_6_쌍이_모자라거나_흔들리면_표본_부족() -> None:
    assert verdict(replace(stats(), n_pairs=99)).step == 6
    assert verdict(stats(se_m=[0.0, 0.01 / 3 * 1.01])).step == 6
    assert verdict(replace(stats(), n_pairs=100)).status == F.PASS


def test_7_축_배율이_벗어나면_불통과() -> None:
    v = verdict(stats(m=[1.0, 0.985]))
    assert (v.status, v.step) == (F.FAIL, 7) and v.reason.startswith("y · ")


def test_순서는_표본_수_기울기_표준오차_나머지다() -> None:
    """§32-5 — 기울기가 표준오차 게이트보다 앞이다. 위치를 따라가지 않는 출력이 표본 부족으로 적히지 않게."""
    v = verdict(stats(s=[0.2, 0.2], se_r=[1.0, 1.0]))
    assert v.step == 2
    v = verdict(replace(stats(s=[0.2, 0.2]), n_images=1))
    assert v.step == 1


def test_서명의_δ_는_원본과_리사이즈를_가를_만큼_작아야_한다() -> None:
    bad = replace(R, delta=0.0112)
    with pytest.raises(ValueError, match="delta"):
        verdict(stats(), bad)
    assert not R.problems((1.0, 704 / 720))


def test_산출에는_상태_후보_사유와_수만_있다() -> None:
    d = verdict(stats(r=[1.0, 0.97], rbar=[float("nan"), 0.97])).as_dict()
    counts = ("n_images", "n_pairs", "n_no_pred", "n_count_mismatch", "n_groups", "n_image_groups", "n_pair_groups")
    assert set(d) == {"status", "candidate", "reason", "samples", *counts}
    assert all(isinstance(d[k], int) for k in counts)
    assert set(d["samples"]) == {"images", "image_groups", "pairs", "pair_groups"}
    assert set(d["samples"].values()) <= {F.MET, F.UNMET}, "표본 요건은 충족 · 미달로만 — 수는 싣지 않는다"


# ---------------------------------------------------------------- 기대 판정 파일 — 문턱보다 먼저

def _expected() -> dict:
    return {c["id"]: c for c in json.loads(EXPECTED.read_text(encoding="utf-8"))["cases"]}


def test_기대_판정_파일은_사례와_하나씩_맞는다() -> None:
    assert set(_expected()) == {c.id for c in C.cases()}


def test_기대_판정_파일은_정규_JSON_이다() -> None:
    raw = EXPECTED.read_bytes()
    body = json.loads(raw)
    assert raw == (json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


@pytest.mark.parametrize(("cid", "status", "detail"), [
    ("가", F.PASS, None), ("나", F.FAIL, "리사이즈"), ("다", F.FAIL, "정규화"), ("라", F.INDETERMINATE, F.REASON_NOT_TRACKING),
    ("마", F.INDETERMINATE, F.REASON_SAMPLE), ("바", F.INDETERMINATE, F.REASON_NOT_TRACKING), ("아", F.FAIL, "리사이즈"),
    ("자", F.FAIL, "후보 없음"), ("카20", F.FAIL, "리사이즈 섞임"), ("파", F.FAIL, "뒤바뀜"),
])
def test_계약_표의_기대_판정을_고치지_않고_옮겼다(cid: str, status: str, detail) -> None:
    """§32-5 의 표 — 기대 판정을 고쳐 문턱에 맞추지 않는다(판정 09 의 I-6)."""
    allowed = _expected()[cid]["allowed"]
    assert len(allowed) == 1 and allowed[0]["status"] == status
    if detail is not None:
        assert detail in (allowed[0].get("candidate"), allowed[0].get("reason"))


def test_불통과가_아니어야_하는_사례는_불통과를_허용하지_않는다() -> None:
    """사 · 타 · 차 와 프레임이 옳은 사다리 — 옳은 프레임을 기각하지 않는다(12-2 의 원칙)."""
    for cid, c in _expected().items():
        if cid.startswith(("사", "타", "차", "가박스", "기울기")) or cid in ("쌍100",):
            assert all(a["status"] != F.FAIL for a in c["allowed"]), cid

"""무내용 AP 대조선 고정 — 2026-09-08 총괄 판정 22번 §6 (D 17번 §13-1 등록 요청).

여기서 고정하는 것은 넷이다.

1. **값 그대로** — 분류 축(macro-AP) 대조선과 위치 축(mAP@50) 대조선. 논문 표에 병기하는
   기록용 상수라 시드 2·3 보고가 같은 수를 인용해야 한다.
2. **구조** — 사다리는 K 에 단조이고 끝(512)이 주 대조선이다. 족은 F1 블록에서 상속했다.
3. **게이트가 아니다** — `content_free_gate` 는 F1 축만 본다(22번 §6-2-6). D 의 게이트
   해석 경로가 이 블록을 집지 않는지 configs 쪽에서 확인한다.
4. **정의·항등식** — `all_positive` 의 AP 는 평균 유병률이고(정의), 하드 예측기의 macro-AP 는
   같은 예측기의 (Macro-F1)² 이상이다(P·R ≥ F1² + Jensen). 등록 F1 0.9149 로는 이 경계가
   깨진다 — H 가 등록 F1 의 예측기가 **아니라는** 것을 수치가 말한다(§13-3 답).
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_YAML = REPO_ROOT / "configs/base.yaml"
SNAPSHOT = REPO_ROOT / "data/interim/manifest_v1"
ARTIFACT = REPO_ROOT / "outputs/main_d/seed1/content_free_baselines_v1.json"

F1_BLOCK = "content_free__sel_val__fit_trainval__score_eval12461"
AP_BLOCK = "content_free_ap__sel_val__fit_trainval__score_eval12461"
POS_BLOCK = "content_free_position__derived_trainval__score_eval12461"
BOX_BLOCK = "constant_box__derived_trainval__score_eval12461"

AP_EXPECTED = {
    "best_family": "idq512",
    "max_macro_ap_freq": 0.9582,
    "max_macro_ap_hard": 0.8355,
    "all_positive_macro_ap": 0.1428,
    "constant_porosity_macro_ap_hard": 0.1095,
    "ladder_freq": {
        2: 0.2230,
        4: 0.2672,
        8: 0.3849,
        16: 0.5129,
        32: 0.6942,
        64: 0.9187,
        128: 0.9410,
        256: 0.9543,
        512: 0.9582,
    },
    "self_check_macro_f1_reconstructed": 0.9128,
}
POS_EXPECTED = {"constant_box_map_50": 0.0015, "idq512_median_box_map_50": 0.0044}


def _cfg() -> dict:
    return yaml.safe_load(BASE_YAML.read_text(encoding="utf-8"))


def _fixed() -> dict:
    return _cfg()["fixed_before_main_runs"]


def _ap() -> dict:
    return _fixed()[AP_BLOCK]


def _pos() -> dict:
    return _fixed()[POS_BLOCK]


# --------------------------------------------------------------------------------------
# 1 — 값
# --------------------------------------------------------------------------------------


def test_분류_축_대조선이_그대로_있다():
    ap = _ap()
    for k, v in AP_EXPECTED.items():
        assert ap[k] == v, f"{AP_BLOCK}.{k}: {ap[k]!r} != {v!r}"


def test_위치_축_대조선이_그대로_있다():
    assert {k: _pos()[k] for k in POS_EXPECTED} == POS_EXPECTED


def test_수치가_전부_float_다():
    ap, pos = _ap(), _pos()
    nums = [
        v
        for k, v in ap.items()
        if k not in ("best_family", "ladder_freq", "crossover_note", "note")
    ]
    nums += list(ap["ladder_freq"].values()) + list(pos.values())
    for v in nums:
        assert type(v) is float, f"{v!r} 가 float 가 아니다"
        assert 0.0 <= v <= 1.0


# --------------------------------------------------------------------------------------
# 2 — 구조
# --------------------------------------------------------------------------------------


def test_사다리가_K_에_단조이고_끝이_주_대조선이다():
    ladder = _ap()["ladder_freq"]
    ks = list(ladder)
    assert ks == [2, 4, 8, 16, 32, 64, 128, 256, 512]
    assert all(type(k) is int for k in ks)
    vals = [ladder[k] for k in ks]
    assert all(a <= b for a, b in pairwise(vals)), "사다리가 K 에 단조가 아니다"
    assert ladder[512] == _ap()["max_macro_ap_freq"]


def test_족은_F1_블록에서_상속했다():
    """AP 축에서 족을 다시 고르지 않았다 — 고르면 그 선택이 새 자유도가 된다(17번 §11-3)."""
    fx = _fixed()
    assert fx[AP_BLOCK]["best_family"] == fx[F1_BLOCK]["best_family"] == "idq512"


def test_상수_박스_값이_기존_등록값과_같다():
    """같은 수를 두 곳에 두면 한 곳만 바뀌는 사고가 난다. 둘이 같은지 본다."""
    fx = _fixed()
    assert fx[POS_BLOCK]["constant_box_map_50"] == fx[BOX_BLOCK]["map_50"]


# --------------------------------------------------------------------------------------
# 3 — 게이트가 아니다
# --------------------------------------------------------------------------------------


def test_게이트는_여전히_F1_축만_본다():
    """22번 §6-2-6: `content_free_gate` 를 AP 축으로 넓히지 않는다. D 의 게이트 해석
    경로가 AP 블록을 집지 않고 F1 선 0.9149 를 그대로 낸다."""
    from evaluation.params import GATE_KEYS, resolve_gate

    for k in GATE_KEYS:
        assert AP_BLOCK not in k and POS_BLOCK not in k, f"게이트 후보가 AP 블록을 가리킨다: {k}"
    g = resolve_gate(_cfg())
    assert g.value == 0.9149
    assert g.source.endswith(f"{F1_BLOCK}.max_macro_f1")


def test_AP_블록에_게이트_스위치가_없다():
    ap = _ap()
    for k in ("gate_status", "gate_tolerance", "pass_line"):
        assert k not in ap


# --------------------------------------------------------------------------------------
# 4 — 정의·항등식
# --------------------------------------------------------------------------------------


def test_하드_AP_는_재구성_F1_의_제곱_이상이다():
    """같은 예측기라면 macro-AP(H) = mean(P·R) ≥ mean(F1²) ≥ (Macro-F1)². D 의 H 는
    D 가 재구성한 F1(0.9128)의 예측기이므로 그 제곱 이상이어야 한다."""
    ap = _ap()
    assert ap["max_macro_ap_hard"] >= ap["self_check_macro_f1_reconstructed"] ** 2


def test_하드_AP_는_등록_F1_의_예측기가_아니다():
    """등록 F1 0.9149 의 예측기(클래스별 F1 최적 접두사)였다면 H ≥ 0.9149² = 0.8370 이어야
    한다. 0.8355 는 그 아래다 — H 는 다른 규칙 형태(구간 최빈 코드 집합)의 값이다.
    D 가 H 를 등록 F1 의 규칙으로 다시 내면 이 시험이 깨지고, 그때는 블록 주석도 바꿔야 한다."""
    fx = _fixed()
    assert fx[AP_BLOCK]["max_macro_ap_hard"] < fx[F1_BLOCK]["max_macro_f1"] ** 2


def test_all_positive_의_AP_는_평가셋_평균_유병률이다():
    """P = 유병률, R = 1 이므로 AP = 유병률(정의). 등록값을 매니페스트(단일 진실)에서 다시 낸다."""
    if not (SNAPSHOT / "manifest.csv").exists():
        pytest.skip(f"동결 스냅샷 없음: {SNAPSHOT}")
    from data.manifest_io import load_snapshot

    snap = load_snapshot(SNAPSHOT)
    ev_ids = set(snap.manifest.loc[snap.manifest["split"] == "eval", "image_id"])
    ann = snap.annotations[snap.annotations["image_id"].isin(ev_ids)]
    codes = ann.groupby("iso_code")["image_id"].nunique()
    prevalence = (codes / len(ev_ids)).astype(float)
    assert len(prevalence) == 4
    assert round(float(prevalence.mean()), 4) == _ap()["all_positive_macro_ap"]


# --------------------------------------------------------------------------------------
# 5 — 산출물과의 대조 (추적되지 않는 실물 — 없으면 건너뛴다)
# --------------------------------------------------------------------------------------


def test_시드1_산출물과_소수_4자리에서_같다():
    if not ARTIFACT.exists():
        pytest.skip(f"D 산출물 없음: {ARTIFACT}")
    d = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    fit = d["baselines_fit_trainval"]
    ap, pos = _ap(), _pos()
    r = lambda x: round(float(x), 4)
    assert r(fit["idq512"]["F"]["macro_ap"]) == ap["max_macro_ap_freq"]
    assert r(fit["idq512"]["H"]["macro_ap"]) == ap["max_macro_ap_hard"]
    assert r(fit["all_positive"]["F"]["macro_ap"]) == ap["all_positive_macro_ap"]
    assert r(fit["constant_porosity"]["H"]["macro_ap"]) == ap["constant_porosity_macro_ap_hard"]
    assert {k: r(fit[f"idq{k}"]["F"]["macro_ap"]) for k in ap["ladder_freq"]} == ap["ladder_freq"]
    assert r(d["self_check"]["reproduced"]) == ap["self_check_macro_f1_reconstructed"]
    assert r(d["position_axis"]["constant_box"]["map_50"]) == pos["constant_box_map_50"]
    assert r(d["position_axis"]["idq512_median_box"]["map_50"]) == pos["idq512_median_box_map_50"]

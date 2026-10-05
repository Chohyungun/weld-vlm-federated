"""프레임 진단의 문턱을 세우는 절차 — 07번 계약 §30-2 의 문턱 절차 3 · §31-2 의 "폭과 표본 크기" · §32-7 의 ①.

**모델 출력을 보기 전에, 합성 사례 위에서.** 사례마다 통계를 한 번 내고(문턱과 무관하다), 문턱을 바꿔 가며 **모든 사례가 기대 판정을 내는가**를 본다.

## 가운데를 고르는 법

문턱은 서로 묶여 있다 — 예를 들어 `s_min` 이 0.7 보다 크면 기울기 0.7 사례가 기울기 단계에서 멈춰 `w_mix` 에 제약을 걸지 않고, 작으면 건다.
그래서 **좌표마다 번갈아** 정한다 — 다른 문턱을 지금 값에 둔 채 한 문턱의 격자를 훑어 **지금 값을 품은 연속한 가능 구간**을 구하고 그 가운데로 옮긴다.
한 바퀴에 움직임이 없으면 멈춘다. 처음 값이 기대 판정을 다 내지 못하면 멈추고 그 사례를 보고한다 — **기대 판정을 고쳐 문턱에 맞추지 않는다.**
가운데는 격자 위의 값이다. 정한 값과 그때의 구간을 함께 싣는다.

## 요구 표본 수(§31-2 · 판정 08 의 28)

표준오차 게이트(폭의 3분의 1)를 지나는 데 필요한 대상 장 수를 **잡음 사다리의 계단마다** 낸다 — 기준 크기에서 잰 표준오차가 `√n` 으로 준다고 보고
`n_req = n₀ · (SE / (폭/3))²`. 흩어짐은 **가정**(잡음 사다리)이다. 가장 좁은 폭의 요구 수가 진단 크기를 정하고, 그 폭이 0.01 보다 좁으면 크기를 다시 판정받는다(§32-7 의 ②).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

from evaluation import frame_diag as F
from evaluation import frame_diag_cases as C

EXPECTED_PATH = Path(__file__).with_name("frame_diag_expected-20261001-1.json")

GRID: dict[str, tuple[float, float, float]] = {
    "s_min": (0.40, 0.95, 0.0025),
    "w_s": (0.0025, 0.40, 0.0025),
    "w_r": (0.0005, 0.05, 0.0005),
    "w_mix": (0.0001, 0.008, 0.0001),
    "delta": (0.0005, 0.0110, 0.0005),
    "n_img": (1, 99, 1),
}
"""문턱 → (격자 하한, 상한, 간격). `delta` 의 상한은 `|1 − 704/720| / 2 = 0.0111` 아래다(서명이 겹치지 않는 조건)."""
ORDER = ("s_min", "w_s", "w_r", "w_mix", "delta", "n_img")

START = F.FrameRules(n_img=50, s_min=0.74, w_s=0.15, w_r=0.008, w_mix=0.0019, delta=0.005, trim=0.0,
                     n_boot=400, boot_seed=20261001)
"""처음 값 — 기대 판정을 모두 내는 한 점이면 된다. 가운데는 절차가 정한다. `trim` 은 0(평균) — 섞임 10 % 를 잡으려면
절사가 그보다 작아야 하고 0 이 가장 단순하다. 부트스트랩 횟수와 씨앗은 여기서 정한다."""


def expected() -> dict[str, dict]:
    return {c["id"]: c for c in json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))["cases"]}


def matches(v: F.FrameVerdict, entry: dict) -> bool:
    """판정이 기대 판정 하나와 맞는가 — 상태 · (불통과면) 후보와 축 · (판정 불가면) 사유 · 걸린 순서."""
    if v.step not in entry["steps"]:
        return False
    for a in entry["allowed"]:
        if a["status"] != v.status:
            continue
        if v.status == F.PASS:
            return True
        if v.status == F.FAIL and a["candidate"] == v.candidate:
            if a.get("axes") is None or v.reason == f"{a['axes']} · {v.candidate}":
                return True
        if v.status == F.INDETERMINATE and a["reason"] == v.reason:
            return True
    return False


def case_stats(rules: F.FrameRules, summary: C.Summary = C.SUMMARY) -> dict[str, F.FrameStats]:
    return {c.id: F.frame_stats(C.build(c, summary), C.MODEL_INPUT_WH, n_boot=rules.n_boot,
                                boot_seed=rules.boot_seed, trim=rules.trim) for c in C.cases()}


def violations(rules: F.FrameRules, stats: dict[str, F.FrameStats], exp: dict[str, dict]) -> list[str]:
    out = []
    for cid, st in stats.items():
        try:
            v = F.verdict_of(st, rules, C.MODEL_INPUT_WH)
        except ValueError:
            return ["규칙이 서지 않는다"]
        if not matches(v, exp[cid]):
            out.append(cid)
    return out


def _grid(name: str) -> list[float]:
    lo, hi, step = GRID[name]
    n = int(round((hi - lo) / step))
    vals = [lo + i * step for i in range(n + 1)]
    return [int(round(x)) for x in vals] if name == "n_img" else [round(x, 6) for x in vals]


def interval(name: str, rules: F.FrameRules, stats, exp) -> tuple[float, float] | None:
    """다른 문턱을 지금 값에 둔 채 `name` 의 격자를 훑어, 지금 값을 품은 연속한 가능 구간. 지금 값이 가능하지 않으면 `None`."""
    vals = _grid(name)
    ok = [not violations(replace(rules, **{name: v}), stats, exp) for v in vals]
    cur = min(range(len(vals)), key=lambda i: abs(vals[i] - getattr(rules, name)))
    if not ok[cur]:
        return None
    lo = hi = cur
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(vals) - 1 and ok[hi + 1]:
        hi += 1
    return vals[lo], vals[hi]


def binding(name: str, rules: F.FrameRules, iv: tuple[float, float], stats, exp) -> dict:
    """구간의 두 끝을 정한 사례 — 끝에서 한 칸 밖의 값에서 기대 판정을 내지 못하는 사례. 격자 끝이면 `격자 끝`."""
    vals = _grid(name)
    out = {}
    for side, edge, nxt in (("lo", iv[0], -1), ("hi", iv[1], 1)):
        i = vals.index(edge) + nxt
        out[side] = violations(replace(rules, **{name: vals[i]}), stats, exp) if 0 <= i < len(vals) else ["격자 끝"]
    return out


def _mid(name: str, lo: float, hi: float) -> float:
    if name == "n_img":
        return int((lo + hi) // 2)
    step = GRID[name][2]
    return round(lo + round((hi - lo) / 2 / step) * step, 6)


def center(stats, exp, start: F.FrameRules = START, max_rounds: int = 20) -> dict:
    """좌표마다 번갈아 가운데로. 돌려주는 것 — 정한 문턱 · 그때의 구간 · 바퀴 수 · 멈춘 까닭."""
    bad = violations(start, stats, exp)
    if bad:
        return {"status": "stopped", "reason": "처음 값이 기대 판정을 다 내지 못한다", "violations": bad,
                "rules": asdict(start)}
    rules = start
    for rnd in range(1, max_rounds + 1):
        moved = False
        for name in ORDER:
            iv = interval(name, rules, stats, exp)
            if iv is None:
                return {"status": "stopped", "reason": f"{name} 의 가능 구간이 비었다", "rules": asdict(rules)}
            mid = _mid(name, *iv)
            if mid != getattr(rules, name):
                rules, moved = replace(rules, **{name: mid}), True
        if not moved:
            ivs = {name: interval(name, rules, stats, exp) for name in ORDER}
            return {"status": "centered", "rounds": rnd, "rules": asdict(rules),
                    "intervals": {k: list(v) for k, v in ivs.items()}, "violations": violations(rules, stats, exp),
                    "binding": {k: binding(k, rules, ivs[k], stats, exp) for k in ORDER}}
    return {"status": "stopped", "reason": f"{max_rounds} 바퀴 안에 멈추지 않았다", "rules": asdict(rules)}


SE_CHECKS: tuple[tuple[str, str, int], ...] = (("r", "w_r", 0), ("r", "w_r", 1), ("s", "w_s", 0), ("s", "w_s", 1),
                                               ("rbar", "w_mix", 1), ("m", "axis_ratio_width", 0),
                                               ("m", "axis_ratio_width", 1))
"""표준오차 게이트 — (통계, 폭의 이름, 축). `rbar` 는 배율이 1 이 아닌 y 하나."""


def required_sizes(rules: F.FrameRules, stats) -> list[dict]:
    """잡음 사다리의 계단마다, 게이트마다 요구 표본 수 — 대상 장 · 묶음 · (축 배율이면) 쌍(07번 §31-2).

    표준오차가 표본의 제곱근으로 준다고 보고 기준 크기에 `(se ÷ (폭 ÷ 3))²` 을 곱한다. 기준 크기는 그 사례의 대상 장 수 `n0`,
    그 통계가 쓰는 묶음 수 `g0`(이미지 수준은 대상 장의 묶음, 축 배율은 쌍의 묶음), 축 배율이면 쌍 수 `q0` 다.
    묶음 단위 부트스트랩이므로 묶음 수의 요구가 독립 표본의 요구다.
    """
    out = []
    for step in C.NOISE_LADDER:
        st = stats[f"사{step}"]
        for key, width_name, axis in SE_CHECKS:
            width = getattr(rules, width_name)
            se = st.se[key][axis]
            factor = (se / (width / rules.se_divisor)) ** 2 if math.isfinite(se) else None
            pair_stat = key == "m"
            g0 = st.n_pair_groups if pair_stat else st.n_image_groups
            row = {"noise_step": step, "stat": key, "axis": F.AXES[axis], "width": width_name,
                   "n0": st.n_images, "n_required": None if factor is None else math.ceil(st.n_images * factor),
                   "g0": g0, "groups_required": None if factor is None else math.ceil(g0 * factor),
                   "q0": st.n_pairs if pair_stat else None,
                   "pairs_required": (None if factor is None else math.ceil(st.n_pairs * factor)) if pair_stat else None}
            out.append(row)
    return out


def _one_group_each(images):
    return [replace(im, group_id=f"b{im.image_id}") for im in images]


def _into_groups(images, k: int):
    return [replace(im, group_id=f"k{i % k:05d}") for i, im in enumerate(images)]


def boundary_check(rules: F.FrameRules) -> dict:
    """표본 경계 — 장 수와 묶음 수를 따로 본다(원본 프레임, 한 장 한 박스).

    장 수: 장마다 묶음 하나로 두고 `N_img − 1` 장은 1 에서, `N_img` 장은 1 을 지난다.
    묶음 수: 장은 `max(N_img, 2 · min_groups)` 로 두고 묶음을 `min_groups − 1` 개 · `min_groups` 개로 나눠 같은 경계를 본다.
    """
    out = {}
    for n in (rules.n_img - 1, rules.n_img):
        ims = _one_group_each(C.build(C.Case(f"경계{n}", "", n, C.identity, single_box=True)))
        v = F.judge(ims, C.MODEL_INPUT_WH, rules)
        out[str(n)] = {"status": v.status, "reason": v.reason, "step": v.step}
    n = max(rules.n_img, 2 * rules.min_groups)
    base = C.build(C.Case(f"묶음경계{n}", "", n, C.identity, single_box=True))
    for k in (rules.min_groups - 1, rules.min_groups):
        v = F.judge(_into_groups(base, k), C.MODEL_INPUT_WH, rules)
        out[f"groups{k}"] = {"status": v.status, "reason": v.reason, "step": v.step}
    return out


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run() -> dict:
    exp = expected()
    stats = case_stats(START)
    result = center(stats, exp)
    report = {"procedure": "07번 §30-2 문턱 절차 3 — 좌표마다 번갈아 가능 구간의 가운데",
              "result": result,
              "fixed": {"trim": START.trim, "n_boot": START.n_boot, "boot_seed": START.boot_seed,
                        "axis_ratio_width": START.axis_ratio_width, "min_pairs": START.min_pairs,
                        "se_divisor": START.se_divisor, "min_groups": START.min_groups,
                        "min_pair_groups": START.min_pair_groups, "min_valid_boot": START.min_valid_boot},
              "grid": {k: list(v) for k, v in GRID.items()},
              "model_input_wh": list(C.MODEL_INPUT_WH), "image_wh": [C.W, C.H],
              "summary": asdict(C.SUMMARY), "noise_ladder": list(C.NOISE_LADDER),
              "sha256": {"expected": file_sha256(EXPECTED_PATH),
                         "cases_module": file_sha256(Path(C.__file__)),
                         "checker": file_sha256(Path(F.__file__)),
                         "procedure": file_sha256(Path(__file__))}}
    if result["status"] == "centered":
        rules = F.FrameRules(**result["rules"])
        report["required_sizes"] = required_sizes(rules, stats)
        report["boundary"] = boundary_check(rules)
        narrow = min(rules.w_r, rules.w_mix, rules.w_s, rules.axis_ratio_width)
        report["size_rejudgment"] = {"narrowest_width": narrow, "below_0_01": narrow < 0.01}
    return report

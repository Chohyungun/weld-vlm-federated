"""조기 좌표 점검의 합성 순차 판별력 — 미니스펙 10번 6절 · 13-1 의 I-2 · 14-1 · 14-2.

등록할 step 목록 그대로 같은 합성 세계를 `s_1 … s_k` 의 잡음 사다리로 보고, step 마다 `early_coord_check.step_verdict`
(대상 전부 + 묶음 단위 비노출 부분)를 낸 뒤 두 순차 규칙(한 번 통과 · 두 step 연속)으로 결정한다. 모델 출력은 쓰지 않는다 —
정답은 `frame_diag_cases.gold_set` 의 요약값으로 만들고 예측은 정답에서 세계의 식으로 낸다.

| 가정(규칙 파일에 싣는다) | 값 |
|---|---|
| step 마다 중심 잡음(박스 크기 × 계단) | 이른 step 일수록 크다 — 사다리 `SCHEDULES` 여럿을 함께 본다 |
| step 마다 노출된 묶음의 비율 | `EXPOSED_BY_STEP` — 1 epoch 의 3 할 안이라는 학습 쪽 권고에서 낸 가정 |
| step 사이 잡음 상관 ρ | 0 · 0.5 · 0.9 — 장마다의 치우침이 이웃 step 에 이어지는 정도 |

음성 세계(통과하면 거짓 통과): 리사이즈 · 정규화 · 섞임 · 뒤바뀜 · 위치 불추종 · 학습한 장만 맞힘 · 묶음 잡음이 실린 리사이즈.
양성 세계(불통과면 거짓 불통과): 옳은 프레임 · 묶음 잡음이 실린 옳은 프레임 · 낮은 짝 성립률.

    python -X utf8 -B -m scripts.probe.early_coord_check_discrimination --rules configs/early_coord_check.yaml --out <JSON>

규칙 파일의 `sequential_rule` 이 비어 있어도 읽는다(이 스크립트가 그 값을 고르는 재료다).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluation import early_coord_check as E
from evaluation import frame_diag as F
from evaluation import frame_diag_cases as C
from evaluation.eval_list import canonical_bytes, validate_eval_list

SCHEDULES = {"가정 사다리": (0.4, 0.2, 0.1), "낮은 사다리": (0.1, 0.05, 0.03),
             "평평 0.06": (0.06, 0.06, 0.06), "평평 0.08": (0.08, 0.08, 0.08)}
"""step 마다 중심 잡음. 가정 사다리는 진단 잡음 사다리의 위 칸이고, 낮은 사다리는 판정을 낼 때의 판별력을 본다. 평평한 둘은
표본 상한(옳은 프레임의 통과 경계 0.08 ~ 0.09) 근처에서 여러 번 보기가 거짓 통과를 키우는지 본다(19b 의 I-3)."""
BOUNDARY_NOISE = (0.03, 0.05, 0.06, 0.07, 0.08, 0.1)
"""옳은 프레임이 한 step 에서 통과하는 잡음의 경계를 재는 칸."""
EXPOSED_BY_STEP = (0.05, 0.10, 0.20)
RHOS = (0.0, 0.5, 0.9)
GROUP_SHIFT_SD = 6.0
"""묶음 잡음 — 같은 묶음의 장이 같은 쪽으로 어긋나는 크기(화소, 표준편차). **가정**이다."""


def _with_noise(golds, z, step_noise, frame):
    """정답 박스마다 `frame` 을 씌운 뒤 중심을 z × 박스 크기 × 계단만큼 옮긴다. 크기는 그대로."""
    out = []
    for im, zi in zip(golds, z):
        boxes = []
        for b, (zx, zy) in zip(im.gold, zi):
            fb = frame(b)
            s = step_noise * ((b[2] - b[0]) + (b[3] - b[1])) / 2
            boxes.append(C._shift(fb, float(s * zx), float(s * zy)))
        out.append(replace(im, pred=tuple(boxes)))
    return out


def _group_shift(golds, rng):
    groups = sorted({im.group_id for im in golds})
    d = {g: (float(rng.normal(0, GROUP_SHIFT_SD)), float(rng.normal(0, GROUP_SHIFT_SD))) for g in groups}
    return [replace(im, pred=tuple(C._shift(b, *d[im.group_id]) for b in im.pred)) for im in golds]


def _frames():
    k = (C.MODEL_INPUT_WH[0] / C.W, C.MODEL_INPUT_WH[1] / C.H)
    return {
        "identity": lambda b: b,
        "resize": lambda b: C._scale(b, *k),
        "normalized": lambda b: C._scale(b, 1000 / C.W, 1000 / C.H),
        "swapped": lambda b: (b[1], b[0], b[3], b[2]),
    }


WORLDS = {
    # 이름: (양성 여부, 프레임, 덧붙임)
    "옳은 프레임": (True, "identity", None),
    "옳은 프레임 + 묶음 잡음": (True, "identity", "group"),
    "낮은 짝 성립률(절반은 예측 없음)": (True, "identity", "half_empty"),
    "리사이즈 k=(1, 704/720)": (False, "resize", None),
    "리사이즈 + 묶음 잡음": (False, "resize", "group"),
    "정규화 0~1000": (False, "normalized", None),
    "섞임 — 20 % 리사이즈": (False, "identity", "mixed"),
    "x · y 뒤바뀜": (False, "swapped", None),
    "위치 불추종 — 기울기 0.3": (False, "identity", "toward"),
    "학습한 장만 맞힘 — 비노출은 빈 출력": (False, "identity", "memorized"),
    # 문턱 근처의 음성 — 진단 기대 파일이 통과가 아니라고 적은 사례(카10 · 카40 · 자-4 · 차 · 기울기0.7)와 섞임 12 %
    "섞임 — 10 % 리사이즈(카10)": (False, "identity", "mixed:0.10"),
    "섞임 — 12 % 리사이즈": (False, "identity", "mixed:0.12"),
    "섞임 — 40 % 리사이즈(카40)": (False, "identity", "mixed:0.40"),
    "y 가산 치우침 −4 px(자-4)": (False, "identity", "ybias:-4"),
    "중심 쪽 균일 수축 — 기울기 0.8(차)": (False, "identity", "toward:0.8"),
    "덜 배움 — 기울기 0.7": (False, "identity", "toward:0.7"),
}


def run_world(name, rho, rep, rules: E.EarlyRules, n_images: int, schedule) -> list[str]:
    positive, frame_name, extra = WORLDS[name]
    seed = 20261013 + 1000 * rep
    golds = C.gold_set(n_images, seed)
    rng = np.random.default_rng(seed + int(rho * 100) + 7)
    groups = sorted({im.group_id for im in golds})
    order = [groups[i] for i in rng.permutation(len(groups))]
    frame = _frames()[frame_name]
    z = [[rng.normal(size=2) for _ in im.gold] for im in golds]
    mean = C._mean_center(golds)
    k = (C.MODEL_INPUT_WH[0] / C.W, C.MODEL_INPUT_WH[1] / C.H)
    mixed_pick = set(rng.choice(len(golds), size=int(round(0.2 * len(golds))), replace=False).tolist())
    half_empty = set(rng.choice(len(golds), size=len(golds) // 2, replace=False).tolist())
    # 덧붙인 세계의 고르기는 따로 뽑는다 — 앞의 세계들의 난수 흐름(첫 영수증)을 바꾸지 않게
    side_rng = np.random.default_rng(seed + 99)
    kind, _, arg = (extra or "").partition(":")
    frac_pick = set(side_rng.choice(len(golds), size=int(round(float(arg) * len(golds))), replace=False).tolist()) \
        if kind == "mixed" and arg else set()
    target = validate_eval_list(canonical_bytes([im.image_id for im in golds]))
    statuses = []
    for i, step_noise in enumerate(schedule[:len(rules.steps)]):
        if i:
            z = [[rho * zz + math.sqrt(1 - rho * rho) * rng.normal(size=2) for zz in zi] for zi in z]
        exposed = set(order[:int(round(EXPOSED_BY_STEP[i] * len(groups)))])
        imgs = _with_noise(golds, z, step_noise, frame)
        if extra == "group":
            imgs = _group_shift(imgs, rng)
        elif extra == "half_empty":
            imgs = [replace(im, pred=()) if j in half_empty else im for j, im in enumerate(imgs)]
        elif extra == "mixed":
            imgs = [replace(im, pred=tuple(C._scale(b, *k) for b in im.pred)) if j in mixed_pick else im
                    for j, im in enumerate(imgs)]
        elif extra == "toward":
            imgs = [replace(im, pred=tuple(C._toward(b, 0.3, mean) for b in im.pred)) for im in imgs]
        elif extra == "memorized":
            imgs = [replace(im, pred=im.gold) if im.group_id in exposed else replace(im, pred=()) for im in imgs]
        elif kind == "mixed" and arg:
            imgs = [replace(im, pred=tuple(C._scale(b, *k) for b in im.pred)) if j in frac_pick else im
                    for j, im in enumerate(imgs)]
        elif kind == "toward":
            imgs = [replace(im, pred=tuple(C._toward(b, float(arg), mean) for b in im.pred)) for im in imgs]
        elif kind == "ybias":
            imgs = [replace(im, pred=tuple(C._shift(b, 0.0, float(arg)) for b in im.pred)) for im in imgs]
        statuses.append(E.step_verdict(imgs, exposed, rules, parse_rate=1.0, target=target)["status"])
    return statuses


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rules", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--schedules", default=",".join(SCHEDULES), help="돌릴 사다리 이름(쉼표로)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    run_sched = {s: SCHEDULES[s] for s in a.schedules.split(",")}
    raw = a.rules.read_bytes()
    rules = E.load_rules(raw)
    n = rules.target_n
    rows = []
    tally = {s: {r: {"false_pass": 0, "false_fail": 0, "negative_runs": 0, "positive_runs": 0, "pass": 0}
                 for r in E.SEQUENTIAL_RULES} for s in run_sched}
    for sched, noise in run_sched.items():
        for name, (positive, _, _) in WORLDS.items():
            for rho in RHOS:
                for rep in range(a.reps):
                    st = run_world(name, rho, rep, rules, n, noise)
                    dec = {r: E.decide(st, len(rules.steps), r)["outcome"] for r in E.SEQUENTIAL_RULES}
                    rows.append({"schedule": sched, "world": name, "positive": positive, "rho": rho, "rep": rep,
                                 "statuses": st, "decision": dec})
                    for r, o in dec.items():
                        t = tally[sched][r]
                        t["pass"] += int(positive and o == E.PASS)
                        if positive:
                            t["positive_runs"] += 1
                            t["false_fail"] += int(o == E.FAIL)
                        else:
                            t["negative_runs"] += 1
                            t["false_pass"] += int(o == E.PASS)
                    print(sched, name, rho, rep, st, dec, flush=True)
    boundary = []
    for noise in BOUNDARY_NOISE:
        got = []
        for rep in range(a.reps):
            golds = C.gold_set(n, 20261013 + 1000 * rep)
            tgt = validate_eval_list(canonical_bytes([im.image_id for im in golds]))
            got.append(E.step_verdict(C.noisy(golds, noise, rep + 1), set(), rules, parse_rate=1.0, target=tgt)["status"])
        boundary.append({"noise": noise, "statuses": got})
        print("경계", noise, got, flush=True)
    chosen = next((r for r in E.SEQUENTIAL_RULES if all(tally[s][r]["false_pass"] == 0 for s in run_sched)), None)
    # 규칙 파일의 해시가 아니라 판정에 쓴 값을 싣는다 — 규칙 파일이 이 영수증의 해시를 담으므로 순환을 끊는다
    out = {"inputs": E.discrimination_inputs(rules), "n_images": n, "steps": list(rules.steps),
           "schedules": {k: list(v) for k, v in run_sched.items()}, "exposed_by_step": list(EXPOSED_BY_STEP), "rhos": list(RHOS),
           "group_shift_sd": GROUP_SHIFT_SD, "reps": a.reps, "tally": tally, "chosen_rule": chosen,
           "correct_frame_noise_boundary": boundary,
           "module_sha256": {"script": E.lf_sha256(Path(__file__)), "early_coord_check": E.lf_sha256(Path(E.__file__)),
                             "frame_diag": E.lf_sha256(Path(F.__file__)), "frame_diag_cases": E.lf_sha256(Path(C.__file__))},
           "rows": rows}
    a.out.write_bytes(E.canonical(out))
    print(json.dumps({"tally": tally, "chosen_rule": chosen}, ensure_ascii=False))


if __name__ == "__main__":
    main()

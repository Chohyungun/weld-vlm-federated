"""무내용 대조선 산출 — 새 주 지표(macro-AP)에 비교선을 세운다 (22번 §5, 사전등록 17번 §11).

    uv run python scripts/probe/content_free_baselines.py \
        --snapshot data/interim/manifest_v1 --out outputs/main_d/seed1 --profile main

**GPU 를 쓰지 않는다.** 동결본의 정답과 id 분위 절단점만으로 CPU 에서 난다. 칸 점수를
바꾸지 않는다 — 기준선은 추가 산출물이다.

## 무엇을 내나

1. **분류 축(macro-AP)** — `all_positive` · `constant_porosity` · `idq{K}` 사다리(주 대조선
   `idq512`)를 H(하드 1/0)·F(구간 빈도) 두 구성으로. 적합은 train+val, 채점은 eval.
2. **평가셋 적합판(`__shortcut__`)** — 같은 규칙을 평가셋 정답으로 적합. **상한이지 게이트가
   아니다.** 두 값의 간격이 "평가셋 유도가 얼마나 부풀리는가" 의 실측이다.
3. **자기 검사** — H 구성 `idq512` 의 Macro-F1 이 등록 상수 0.9149 를 재현하는가. 재현하지
   못하면 내 재구성이 A 의 규칙과 다른 것이고, 그러면 AP 값도 다른 규칙의 값이다.
4. **위치 축(mAP@50)** — 등록 상수 박스(0.0015)와 **id 구간별 중앙 박스**(train+val 적합)를
   비교한다. 분류 축과 같은 강도의 대조선이 위치 축에도 있는지 보는 것이다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from data.label_map import load_label_map
from evaluation.content_free import FREQ, HARD, fit_idq, fit_trivial, predict_codes, scored
from evaluation.eval_set import eval_rows, read_gold, read_manifest
from evaluation.metrics.detection import image_level_ap, score_detection
from evaluation.metrics.localization import coco_map
from evaluation.params import add_common_args, params_from_args
from evaluation.strata import ID_GRANULARITY, bins_for

REGISTERED = {
    "idq512_macro_f1": 0.9149,
    "all_positive_macro_f1": 0.2160,
    "constant_porosity_macro_f1": 0.1523,
    "constant_box_map_50": 0.0015,
    "source": "configs/base.yaml:fixed_before_main_runs."
              "content_free__sel_val__fit_trainval__score_eval12461",
}
"""A 가 등록한 대조값. **재현 여부를 자기 검사로 본다.**"""

PRIMARY_K = 512
"""등록된 `best_family: idq512`. 여기서 족을 다시 고르지 않는다(§11-3)."""


def _score_rule(rule, eval_ids, gold_eval, classes, snapshot) -> dict:
    """규칙 하나를 두 구성으로 채점한다. F1 축은 H 구성에서만 정의된다(라벨 집합 예측기)."""
    out: dict = {"rule": rule.as_dict()}
    codes = predict_codes(rule, eval_ids, snapshot=snapshot)
    det = score_detection(codes, gold_eval, classes)
    out["macro_f1"] = det.macro_f1
    out["miss_rate"] = 1.0 - det.defect_recall
    out["per_class_f1"] = {s.iso_code: s.f1 for s in det.per_class}
    for name in (HARD, FREQ):
        s, diag = scored(rule, eval_ids, name, snapshot=snapshot)
        ap = image_level_ap(s, gold_eval, classes)
        out[name] = {"macro_ap": ap["macro_ap"], "per_class_ap": ap["per_class_ap"],
                     **diag}
    # 하드 구성의 AP 는 정의상 P·R 이다. 그 항등식을 값으로 확인한다 — 어긋나면
    # AP 구현이나 예측기 중 하나가 문서와 다른 것이다.
    pr = {}
    for s in det.per_class:
        if s.support:
            p, r = s.precision, s.recall
            pr[s.iso_code] = None if (p is None or r is None) else p * r
    out["hard_ap_identity_check"] = {
        "per_class_p_times_r": pr,
        "matches_hard_ap": all(
            (pr.get(c) is None and out[HARD]["per_class_ap"].get(c) is None)
            or abs((pr.get(c) or 0.0) - (out[HARD]["per_class_ap"].get(c) or 0.0)) < 1e-9
            for c in pr
        ),
    }
    return out


def _median_box(boxes) -> list[float]:
    """좌표별 독립 중앙값 — A 의 등록 상수 박스와 **같은 구성**이다(configs 주석 M8)."""
    return [statistics.median([b[j] for b in boxes]) for j in range(4)]


def _position_axis(fit_ids, gold_boxes_fit, eval_ids, gold_boxes_eval, classes,
                   snapshot, k: int) -> dict:
    """위치 축 대조선 — 상수 박스 대 id 구간별 중앙 박스. 둘 다 train+val 적합."""
    flat = [b for i in fit_ids for _, b in gold_boxes_fit.get(i, ())]
    if not flat:
        return {"checked": False, "reason": "적합 모집단에 GT 박스가 없다"}
    const_box = _median_box(flat)

    fb = bins_for(list(fit_ids), k, snapshot)
    per_bin: dict[int, list] = {}
    for i in fit_ids:
        for _, b in gold_boxes_fit.get(i, ()):
            per_bin.setdefault(fb[i], []).append(b)
    bin_box = {b: _median_box(v) for b, v in per_bin.items()}
    eb = bins_for(list(eval_ids), k, snapshot)

    def _map(box_of) -> dict:
        # 모든 이미지에 4클래스 박스를 하나씩 주장한다(클래스 정보가 없는 규칙이므로).
        pred = {i: [(c, tuple(box_of(i)), 1.0) for c in classes] for i in eval_ids}
        return coco_map(pred, gold_boxes_eval, classes, scores_present=True)

    return {
        "checked": True,
        "constant_box": {"box_xyxy": const_box, **_map(lambda i: const_box)},
        f"idq{k}_median_box": {"n_bins": len(bin_box),
                               **_map(lambda i: bin_box.get(eb[i], const_box))},
        "note": ("클래스 정보가 없는 규칙이라 4클래스 모두에 같은 박스를 주장한다. "
                 "등록 상수 박스와 같은 구성(좌표별 독립 중앙값)으로 적합했다"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--root", default=".")
    ap.add_argument("--at-conf", action="store_true")
    ap.add_argument("--position-axis", action="store_true", default=True)
    args = ap.parse_args()

    params = params_from_args(args)
    params.out.mkdir(parents=True, exist_ok=True)
    lm = load_label_map()
    classes = [lm.iso_code(n) for n in params.class_names]
    snapshot = params.snapshot

    rows = read_manifest(snapshot)
    ev = eval_rows(rows)
    eval_ids = sorted(r["image_id"] for r in ev)
    fit_ids = sorted(r["image_id"] for r in rows if r["split"] in ("train", "val"))
    print(f"적합 train+val {len(fit_ids):,}장 · 채점 eval {len(eval_ids):,}장 · 클래스 {classes}")

    gold_eval, boxes_eval = read_gold(snapshot, set(eval_ids))
    gold_fit, boxes_fit = read_gold(snapshot, set(fit_ids))
    for i in eval_ids:
        gold_eval.setdefault(i, set()), boxes_eval.setdefault(i, [])
    for i in fit_ids:
        gold_fit.setdefault(i, set()), boxes_fit.setdefault(i, [])
    gold_eval_s = {k: sorted(v) for k, v in gold_eval.items()}
    gold_fit_s = {k: sorted(v) for k, v in gold_fit.items()}

    baselines: dict[str, dict] = {}

    # --- 자명 규칙 (구간을 쓰지 않는다) -------------------------------------------------
    for name, pred in (("all_positive", classes), ("constant_porosity", ["2011"])):
        rule = fit_trivial(name, pred, fit_ids, gold_fit_s, classes)
        baselines[name] = _score_rule(rule, eval_ids, gold_eval_s, classes, snapshot)
        b = baselines[name]
        print(f"[{name:20s}] F1 {b['macro_f1']:.4f} · AP(H) {b[HARD]['macro_ap']:.4f} "
              f"· AP(F) {b[FREQ]['macro_ap']:.4f}")

    # --- idq 사다리 (주 대조선 idq512) --------------------------------------------------
    for k in ID_GRANULARITY:
        rule = fit_idq(k, fit_ids, gold_fit_s, classes, snapshot=snapshot)
        key = f"idq{k}"
        baselines[key] = _score_rule(rule, eval_ids, gold_eval_s, classes, snapshot)
        b = baselines[key]
        mark = "  ← 주 대조선" if k == PRIMARY_K else ""
        print(f"[{key:20s}] F1 {b['macro_f1']:.4f} · AP(H) {b[HARD]['macro_ap']:.4f} "
              f"· AP(F) {b[FREQ]['macro_ap']:.4f}{mark}")

    # --- 평가셋 적합판 (상한, 게이트 아님) ----------------------------------------------
    shortcut: dict[str, dict] = {}
    for k in (64, PRIMARY_K):
        rule = fit_idq(k, eval_ids, gold_eval_s, classes, snapshot=snapshot,
                       fit_population="eval(상한)")
        shortcut[f"idq{k}"] = _score_rule(rule, eval_ids, gold_eval_s, classes, snapshot)
        b = shortcut[f"idq{k}"]
        print(f"[__shortcut__ idq{k:<5d}] F1 {b['macro_f1']:.4f} · AP(H) "
              f"{b[HARD]['macro_ap']:.4f} · AP(F) {b[FREQ]['macro_ap']:.4f} (평가셋 적합 상한)")

    # --- 자기 검사: 등록 상수 재현 --------------------------------------------------------
    got = baselines[f"idq{PRIMARY_K}"]["macro_f1"]
    want = REGISTERED["idq512_macro_f1"]
    selfcheck = {
        "registered_idq512_macro_f1": want, "reproduced": got,
        "abs_delta": abs(got - want), "within_0.01": abs(got - want) <= 0.01,
        "note": ("재현하지 못하면 내 재구성이 A 의 규칙과 다른 것이고, 그러면 AP 값도 "
                 "다른 규칙의 값이다. 값 자체보다 이 대조가 먼저다"),
    }
    print(f"\n[자기 검사] 등록 idq512 Macro-F1 {want} · 재구성 {got:.4f} "
          f"(차 {selfcheck['abs_delta']:.4f}) → {'재현' if selfcheck['within_0.01'] else '불일치'}")

    position = (_position_axis(fit_ids, boxes_fit, eval_ids, boxes_eval, classes,
                               snapshot, PRIMARY_K)
                if args.position_axis else {"checked": False, "reason": "생략"})
    if position.get("checked"):
        print(f"[위치 축] 상수 박스 mAP@50 {position['constant_box']['map_50']:.4f} · "
              f"idq{PRIMARY_K} 중앙 박스 {position[f'idq{PRIMARY_K}_median_box']['map_50']:.4f} "
              f"(등록 {REGISTERED['constant_box_map_50']})")

    payload = {
        "params": params.as_dict(),
        "preregistration": "17번 §11 (산출 전 커밋 3f8d5cb)",
        "constructions": {"H": "예측 클래스에 1, 나머지 부재",
                          "F": "구간 train+val 빈도를 4클래스 전부에(0 포함) — **주 대조선**"},
        "fit_population": {"n_trainval": len(fit_ids), "n_eval": len(eval_ids),
                           "rule": "족은 등록된 idq512 고정, 절단점·적합 정답 모두 train+val"},
        "registered": REGISTERED,
        "self_check": selfcheck,
        "baselines_fit_trainval": baselines,
        "shortcut_fit_eval_upper_bound": shortcut,
        "position_axis": position,
        "primary_k": PRIMARY_K,
        "note": ("칸 점수를 바꾸지 않는다 — 추가 산출물이다. 판정 규칙은 17번 §11-5 에 "
                 "산출 전에 적어 두었다"),
    }
    dest = params.out / "content_free_baselines_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"저장: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""전역 대 층화 대조표 + 판별력 시험 — 총괄 판정 6 (76번 을안). 83번 산출.

    uv run python scripts/probe/stratified_compare.py
    uv run python scripts/probe/stratified_compare.py --k 64 --ladder

파일럿 일곱 모델을 **다시 채점하지 않는다** — 저장된 계약 #4 레코드의 코드 집합을
그대로 쓰고, 같은 `score_detection` 을 구간 안에서 부른다. GPU 불필요.

절단점은 **동결본 train+val 에서만** 뽑는다 — A 의 `data.id_strata` 가 만들고 D 는
부른다(`evaluation.strata.bins_for`). 파일럿 평가셋 653장은 동결 평가셋의 부분집합이므로
같은 절단점을 그대로 받는다.

판별력 시험이 이 스크립트의 핵심이다 — **지름길 규칙(구간→최빈 라벨)을 하나의 "모델"로
넣어 같은 표에서 채점한다.** 전역 Macro-F1 에서는 높은 점수를 받고 층화 lift 에서는
정확히 0 이 나와야 한다. 그 두 수의 차이가 층화 채점을 도입하는 이유 전부다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from evaluation.cells import ALL_TAGS, load_population
from evaluation.eval_set import read_manifest
from evaluation.params import FROZEN_SNAPSHOT, add_common_args, params_from_args
from evaluation.strata import (
    DEFAULT_K,
    ID_GRANULARITY,
    bins_for,
    shortcut_pred,
    stratified_score,
)


def load_records(params, tag):
    from evaluation.adapters import read_records

    p = params.out / f"{tag}_s{params.seed}.jsonl"
    if not p.exists():
        return None
    return read_records(p.read_text(encoding="utf-8").splitlines())


def main() -> int:
    ap = argparse.ArgumentParser()
    add_common_args(ap)
    ap.add_argument("--root", default=".")
    ap.add_argument("--at-conf", action="store_true")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--frozen", default=FROZEN_SNAPSHOT)
    ap.add_argument("--ladder", action="store_true", help="K 사다리 전체를 낸다")
    ap.add_argument("--frozen-eval-structure", action="store_true",
                    help="동결 평가셋 12,461장의 층 구조만 낸다(모델 예측 불필요)")
    args = ap.parse_args()

    params = params_from_args(args)
    pop = load_population(params)
    classes = pop.classes

    # 절단점은 동결본 train+val 에서만 — A 의 `data.id_strata` 가 그 모집단을 스스로 고른다.
    # 여기서 매니페스트를 읽는 것은 구조 모드의 평가셋 id 와 보고용 건수 때문이다.
    frozen_dir = Path(args.frozen)
    frozen = read_manifest(args.frozen)
    n_tv = sum(1 for r in frozen if r["split"] in ("train", "val"))
    print(f"절단점 유도 모집단: 동결 train+val {n_tv}장 (data.id_strata) · 채점 {pop.n_eval}장")

    gold = {i: sorted(pop.gold_codes.get(i, ())) for i in pop.eval_ids}

    preds: dict[str, dict] = {}
    for tag in ALL_TAGS:
        recs = load_records(params, tag)
        if recs is None:
            print(f"산출물 없음: {tag} — 건너뛴다")
            continue
        preds[tag] = {r.image_id: sorted(r.iso_codes) for r in recs}

    if args.frozen_eval_structure:
        return structure_only(args, frozen, n_tv, pop.classes)

    ks = list(ID_GRANULARITY) if args.ladder else [args.k]
    out: dict[str, dict] = {}
    for k in ks:
        bins = bins_for(sorted(pop.eval_ids), k, snapshot=frozen_dir)
        # 지름길 규칙을 하나의 모델로 넣는다 — 판별력 시험
        shortcut = shortcut_pred(gold, classes, bins)
        rows = {}
        for tag, pc in {**preds, "__shortcut__": shortcut}.items():
            rep = stratified_score(pc, gold, classes, bins, k=k,
                                   keep_detail=(k == args.k and tag == "__shortcut__"))
            rows[tag] = rep.as_dict(with_detail=(k == args.k and tag == "__shortcut__"))
        out[str(k)] = rows
        s = rows["__shortcut__"]
        print(f"[K={k:3d}] 구간 {s['n_strata']:3d} (채점가능 {s['n_strata_scored']:3d}, "
              f"순수 {s['n_pure_strata']:3d} · 이미지 {s['frac_images_in_pure']:.1%}) "
              f"지름길: 전역 {s['global_macro_f1']:.4f} · "
              f"층화 {s['stratified_macro_f1']:.4f} · lift {s['stratified_lift']:+.6f}")

    dest = params.out / "stratified_compare_v1.json"
    payload = {
        "params": params.as_dict(),
        "frozen_snapshot": args.frozen,
        "n_fit_trainval": n_tv,
        "n_eval_scored": pop.n_eval,
        "default_k": args.k,
        "by_k": out,
        "note": (
            "층화 Macro-F1 은 지시 문면, lift 는 완료 기준이 겨냥한 수다. 지름길 규칙의 "
            "lift 가 0 이 아니면 층 정의나 기준선 정의가 틀린 것이다"
        ),
    }
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    k0 = str(args.k)
    print(f"\n=== K={args.k} 전역 대 층화 ===")
    print(f"{'칸':16s} {'전역F1':>8s} {'층화F1':>8s} {'lift':>9s} "
          f"{'비순수F1':>9s} {'비순수lift':>11s}")
    for tag in [*ALL_TAGS, "__shortcut__"]:
        r = out[k0].get(tag)
        if not r:
            continue
        print(f"{tag:16s} {r['global_macro_f1']:8.4f} {r['stratified_macro_f1']:8.4f} "
              f"{r['stratified_lift']:+9.5f} {r['stratified_macro_f1_impure']:9.4f} "
              f"{r['stratified_lift_impure']:+11.5f}")
    print(f"저장: {dest}")
    return 0


def structure_only(args, frozen, n_tv, classes) -> int:
    """**동결 평가셋 12,461장의 층 구조.** 모델 예측이 없어도 낼 수 있다.

    본실험 착수 판단에 필요한 수는 하나다 — *층화 채점이 잴 것이 남아 있는가.*
    순수 구간(정답 코드 집합이 한 가지)에서는 최빈 라벨 기준선이 전건 정답이라 어떤
    모델도 이길 수 없다. 비순수 구간이 없으면 지표가 형식만 남는다.

    파일럿 653장에서는 K≥64 에서 비순수 구간이 0 이었다. 12,461장에서도 그런지가
    이 모드의 질문이다.
    """
    from evaluation.eval_set import read_gold

    ev = [r for r in frozen if r["split"] == "eval"]
    ids = {r["image_id"] for r in ev}
    gc, _ = read_gold(args.frozen, ids)
    gold = {i: sorted(gc.get(i, ())) for i in ids}
    print(f"동결 평가셋 {len(ids)}장 · 결함 {sum(1 for v in gold.values() if v)}장")

    rows = []
    for k in ID_GRANULARITY:
        bins = bins_for(sorted(ids), k, snapshot=Path(args.frozen))
        sc = shortcut_pred(gold, classes, bins)
        rep = stratified_score(sc, gold, classes, bins, k=k)
        d = rep.as_dict()
        rows.append(d)
        print(f"[K={k:3d}] 구간 {d['n_strata']:4d} · 채점가능 {d['n_strata_scored']:4d} · "
              f"순수 {d['n_pure_strata']:4d} · **비순수 {d['n_strata_impure_scored']:4d}** · "
              f"순수이미지 {d['frac_images_in_pure']:.1%} · "
              f"지름길 전역 {d['global_macro_f1']:.4f} 층화 {d['stratified_macro_f1']:.4f}")
    dest = Path(args.out) / "strata_structure_frozen_eval_v1.json"
    with dest.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump({"snapshot": args.frozen, "n_eval": len(ids),
                   "n_fit_trainval": n_tv, "by_k": rows},
                  fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"저장: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

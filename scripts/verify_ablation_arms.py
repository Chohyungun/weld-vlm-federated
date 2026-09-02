"""어블레이션 두 팔의 채점 모집단이 실제로 같은지 확인한다. 74번 감사 A-3.

감사가 잡은 것은 구성표가 아니라 **자명하한**이었다 — crop_only 431장(0.2930) 대
scale_control 418장(0.2197). 0.073 차이가 델타에 섞였다. 그래서 검사도 구성표 비교가
아니라 **같은 채점기로 낸 자명하한이 비트 단위로 같은가**로 한다. 구성이 같아도 채점이
갈리면 A-3 은 해소되지 않은 것이다.

    uv run python scripts/verify_ablation_arms.py

실패하면 종료 코드 1. 머지 게이트에서 부를 수 있게 만들었다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from data.manifest_io import load_snapshot
from evaluation.schema import Defect, PredictionRecord
from evaluation.score import score_records

REPO_ROOT = Path(__file__).resolve().parents[1]
ARMS = REPO_ROOT / "data/processed/ablation_arms.json"


def eval_gold(snap, ids):
    keep = set(ids)
    ann = snap.annotations[snap.annotations["image_id"].isin(keep)]
    codes: dict[str, list[str]] = {i: [] for i in ids}
    boxes: dict[str, list] = {i: [] for i in ids}
    for r in ann.itertuples():
        codes[r.image_id].append(str(r.iso_code))
        vals = (r.bbox_x1_px, r.bbox_y1_px, r.bbox_x2_px, r.bbox_y2_px)
        if not any(pd.isna(v) for v in vals):
            boxes[r.image_id].append((str(r.iso_code), tuple(float(v) for v in vals)))
    return {k: sorted(set(v)) for k, v in codes.items()}, boxes


def trivial_floor(snap) -> tuple[dict, list[str], list[str]]:
    """전량양성 자명하한 — 다섯 칸과 같은 채점기로 낸다."""
    m = snap.manifest
    ids = sorted(m.loc[m["split"] == "eval", "image_id"])
    gold, boxes = eval_gold(snap, ids)
    classes = sorted({c for v in gold.values() for c in v})
    recs = [
        PredictionRecord(
            schema_version="1.3", image_id=i, cell="sep_central", seed=20260825,
            defects=[Defect(iso_code=c, bbox_px=None, score=None) for c in classes],
            verdict="판정불가", cited_clauses=[], parse_ok=True,
        )
        for i in ids
    ]
    return score_records(recs, gold, boxes, classes), ids, classes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", type=Path, default=ARMS)
    args = ap.parse_args()

    spec = json.loads(args.arms.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for arm in spec["arms"]:
        snap = load_snapshot(REPO_ROOT / arm["path"])
        res, ids, classes = trivial_floor(snap)
        out[arm["arm"]] = {
            "digest": arm["digest"], "eval_ids": ids, "classes": classes,
            "n_eval": len(ids), "train_pool_n": arm["train_pool_n"],
            "macro_f1": float(res["macro_f1"]), "miss_rate": float(res["miss_rate"]),
        }
        print(f"[{arm['arm']:14s}] eval {len(ids):,}장 · 클래스 {classes} · "
              f"전량양성 Macro-F1 {res['macro_f1']:.4f} · 놓침 {res['miss_rate']:.4f} · "
              f"학습 풀 {arm['train_pool_n']:,}")

    names = list(out)
    if len(names) != 2:
        print(f"!! 팔이 2개가 아니다 ({len(names)}개)")
        return 1
    a, b = out[names[0]], out[names[1]]

    checks = [
        ("평가셋 이미지 집합 동일", a["eval_ids"] == b["eval_ids"]),
        ("평가셋 클래스 집합 동일", a["classes"] == b["classes"]),
        ("자명하한 Macro-F1 동일", abs(a["macro_f1"] - b["macro_f1"]) < 1e-12),
        ("자명하한 놓침 동일", abs(a["miss_rate"] - b["miss_rate"]) < 1e-12),
        ("학습 풀 규모 동일", a["train_pool_n"] == b["train_pool_n"]),
        ("두 팔의 digest 는 서로 달라야 한다", a["digest"] != b["digest"]),
    ]
    print()
    ok = True
    for label, passed in checks:
        print(f"  [{'통과' if passed else '실패'}] {label}")
        ok &= passed

    delta = abs(a["macro_f1"] - b["macro_f1"])
    print(f"\n자명하한 차 {delta:.6f} (이전 판 0.073 → A-3 의 오염원)")
    if not ok:
        print("!! A-3 이 해소되지 않았다")
        return 1
    print("두 팔은 같은 모집단에서 채점된다. 델타는 학습 구성 차이만 잰다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

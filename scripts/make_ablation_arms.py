"""승격 어블레이션 두 팔 — 크롭 한정 대 규모 대조. 68번 §5-(다).

크롭 한정은 **세 가지를 동시에 바꾼다** — 출처 축 제거, 표본 35% 감소, 정상 비율 급락.
그대로 돌리면 델타가 무엇을 잰 것인지 알 수 없으므로 규모만 맞춘 대조 팔을 함께 만든다.

| 팔 | 구성 |
|---|---|
| `crop_only` | 파일럿 표본에서 `N-crop` 만 (2,097장) |
| `scale_control` | 출처 구성을 유지한 채 같은 장수로 축소 |

대조 팔은 **(split × 출처)** 층에서 비례로 뽑는다. 출처만 맞추면 평가셋 크기가 따로
움직여 두 팔이 다른 모집단에서 채점된다.

**동결을 풀지 않는다.** 파생 표본이고 원본 digest 는 그대로다(60번 전례).

    uv run python scripts/make_ablation_arms.py
"""

from __future__ import annotations

import argparse
import hashlib
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

from data.invariants import check_invariants
from data.label_map import load_label_map
from data.manifest_io import load_snapshot, write_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT = REPO_ROOT / "data/processed/aihub71761_rt_v1_pilot3000"
DEFAULT_SEED = 20260828


def order_key(image_id: str, seed: int) -> str:
    return hashlib.sha256(f"{image_id}|{seed}".encode()).hexdigest()


def write_arm(out: Path, name: str, sub_m: pd.DataFrame, snap, lm, note: str) -> dict:
    sub_m = sub_m.copy()
    sub_m["group_size"] = sub_m.groupby("group_id")["image_id"].transform("size").astype(int)
    keep = set(sub_m["image_id"])
    sub_a = snap.annotations[snap.annotations["image_id"].isin(keep)].copy()
    sub_t = snap.tiles[snap.tiles["image_id"].isin(keep)].copy()

    violations = check_invariants(sub_m, sub_a, lm, raw_root=REPO_ROOT)
    if violations:
        print(f"  !! {name} 불변조건 위반:")
        for v in violations[:10]:
            print(f"     {v}")
        raise SystemExit(1)

    caps = dict(snap.capabilities)
    caps["snapshot_id"] = f"{snap.snapshot_id}_{name}"
    caps["counts"] = dict(caps.get("counts", {}))
    caps["counts"]["images_total"] = len(sub_m)
    caps["ablation_arm"] = {
        "derived_from": snap.snapshot_id, "arm": name, "seed": DEFAULT_SEED,
        "selection": note,
        "warning": ("승격 효과 측정 전용이다. 두 팔은 짝으로만 해석한다 — "
                    "크롭 한정 단독 델타는 출처 축·표본 규모·정상 비율이 섞여 있다."),
    }
    digest = write_snapshot(out, sub_m, sub_a, caps, tiles=sub_t)
    prov = dict(zip(sub_t["image_id"], sub_t["provenance"], strict=True))
    d = sub_m.assign(prov=sub_m["image_id"].map(prov), defect=sub_m["has_defect"].astype(bool))
    print(f"\n[{name}] {len(sub_m):,}장 · digest {digest[:16]}…")
    print(f"  split {sub_m['split'].value_counts().to_dict()}")
    print(f"  출처 {d['prov'].value_counts().to_dict()}")
    print(f"  결함/정상 {d['defect'].value_counts().to_dict()}")
    train = d[d["split"] != "eval"]
    print(f"  학습 풀 정상 비율 {(~train['defect']).mean()*100:.1f}% ({len(train):,}장 기준)")
    return {
        "arm": name, "path": str(out.relative_to(REPO_ROOT).as_posix()), "digest": digest,
        "n": len(sub_m),
        "split": {str(k): int(v) for k, v in sub_m["split"].value_counts().items()},
        "provenance": {str(k): int(v) for k, v in d["prov"].value_counts().items()},
        "defect": int(d["defect"].sum()), "normal": int((~d["defect"]).sum()),
        "train_pool_normal_pct": round(float((~train["defect"]).mean()) * 100, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pilot", type=Path, default=PILOT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "data/processed")
    args = ap.parse_args()

    snap = load_snapshot(args.pilot)
    m, t = snap.manifest, snap.tiles
    prov = dict(zip(t["image_id"], t["provenance"], strict=True))
    m = m.assign(prov=m["image_id"].map(prov))
    lm = load_label_map()
    print(f"파일럿 표본 {snap.snapshot_id}: {len(m):,}장 · "
          f"출처 {m['prov'].value_counts().to_dict()}")

    # --- 팔 1: 크롭 한정 ---
    crop = m[m["prov"] == "N-crop"].drop(columns=["prov"])
    target = len(crop)

    # --- 팔 2: 규모 대조. (split × 출처) 층에서 비례 추출 ---
    frac = target / len(m)
    picked: list[str] = []
    rows = []
    for (sp, pv), part in m.groupby(["split", "prov"], observed=True):
        quota = round(len(part) * frac)
        chosen = sorted(part["image_id"], key=lambda i: order_key(i, args.seed))[:quota]
        picked.extend(chosen)
        rows.append({"split": sp, "provenance": pv, "available": len(part),
                     "quota": quota, "picked": len(chosen)})
    control = m[m["image_id"].isin(set(picked))].drop(columns=["prov"])
    print(f"\n대조 팔 비율 {frac:.5f} · 층 {len(rows)}개 · 선택 {len(control):,}장 (목표 {target:,})")

    summary = [
        write_arm(args.outdir / f"{snap.snapshot_id}_crop_only", "crop_only",
                  crop, snap, lm, "출처 N-crop 만 (파일럿 표본에서 필터)"),
        write_arm(args.outdir / f"{snap.snapshot_id}_scale_control", "scale_control",
                  control, snap, lm,
                  "(split × 출처) 층별 비례, sha256(image_id + seed) 순서"),
    ]
    out = args.outdir / "ablation_arms.json"
    out.write_text(json.dumps({"seed": args.seed, "derived_from": snap.snapshot_id,
                               "stratum_accounting": rows, "arms": summary},
                              ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"\n기록: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

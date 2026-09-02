"""승격 어블레이션 두 팔 — 크롭 한정 대 규모 대조. 68번 §5-(다) → 74번 감사 A-3 반영.

크롭 한정은 **세 가지를 동시에 바꾼다** — 출처 축 제거, 표본 35% 감소, 정상 비율 급락.
그대로 돌리면 델타가 무엇을 잰 것인지 알 수 없으므로 규모만 맞춘 대조 팔을 함께 만든다.

## 74번 감사 A-3 이 고치라고 한 것

이전 판은 **팔마다 자기 eval 을 따로 가졌다.** crop_only 의 eval 은 431장(전량 N-crop,
전량양성 자명하한 0.2930), scale_control 의 eval 은 418장(N-crop 276·N-tile 141·N-band 1,
0.2197)이다. 두 팔의 content-free 바닥이 0.073 차이나므로 (크롭 한정 − 대조) 델타에
학습 변화가 아니라 **채점 모집단 변화가 섞인다.** 이전 판은 평가셋 "크기"만 맞췄고
구성은 맞추지 않았다.

이 판은 **평가셋을 두 팔이 공유한다.** 파일럿 표본의 eval split 전량을 두 팔에 똑같이
넣고, 어블레이션이 바꾸는 것은 **학습 풀뿐**이 되게 한다. 채점 모집단이 비트 단위로
같으므로 자명하한도 같고, 델타는 학습 구성 차이만 잰다.

| 팔 | 학습 풀 | 평가셋 |
|---|---|---|
| `crop_only` | 파일럿 train+val 중 `N-crop` 만 | **공유 eval (동일)** |
| `scale_control` | 같은 장수, (split × 출처) 비례 | **공유 eval (동일)** |

규모 등가는 이제 **학습 풀 장수**에서 맞춘다 — 어블레이션이 건드리는 쪽이 거기다.
공유 eval 이 두 팔에 똑같이 더해지므로 팔 전체 장수도 자동으로 같아진다.

부차 채점 모집단이 필요하면 공유 eval 의 `N-crop` 부분집합을 쓴다. eval 자체가 동일하니
그 부분집합도 두 팔에서 동일하다 — 별도 자산을 만들 필요가 없다.

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
        "shared_eval": True,
        "warning": ("승격 효과 측정 전용이다. 두 팔은 짝으로만 해석한다 — "
                    "크롭 한정 단독 델타는 출처 축·표본 규모·정상 비율이 섞여 있다. "
                    "eval split 은 두 팔이 공유한다(74번 A-3). 두 팔을 서로 다른 "
                    "모집단에서 채점하면 델타가 다시 오염된다."),
    }
    digest = write_snapshot(out, sub_m, sub_a, caps, tiles=sub_t)
    prov = dict(zip(sub_t["image_id"], sub_t["provenance"], strict=True))
    d = sub_m.assign(prov=sub_m["image_id"].map(prov), defect=sub_m["has_defect"].astype(bool))
    print(f"\n[{name}] {len(sub_m):,}장 · digest {digest[:16]}…")
    print(f"  split {sub_m['split'].value_counts().to_dict()}")
    print(f"  출처 {d['prov'].value_counts().to_dict()}")
    print(f"  결함/정상 {d['defect'].value_counts().to_dict()}")
    train = d[d["split"] != "eval"]
    ev = d[d["split"] == "eval"]
    print(f"  학습 풀 정상 비율 {(~train['defect']).mean()*100:.1f}% ({len(train):,}장 기준)")
    print(f"  평가셋 {len(ev):,}장 · 출처 {ev['prov'].value_counts().to_dict()} · "
          f"결함 {int(ev['defect'].sum()):,}")
    return {
        "arm": name, "path": str(out.relative_to(REPO_ROOT).as_posix()), "digest": digest,
        "n": len(sub_m),
        "split": {str(k): int(v) for k, v in sub_m["split"].value_counts().items()},
        "provenance": {str(k): int(v) for k, v in d["prov"].value_counts().items()},
        "defect": int(d["defect"].sum()), "normal": int((~d["defect"]).sum()),
        "train_pool_n": len(train),
        "train_pool_normal_pct": round(float((~train["defect"]).mean()) * 100, 2),
        "train_pool_provenance": {str(k): int(v)
                                  for k, v in train["prov"].value_counts().items()},
        "eval_n": len(ev),
        "eval_provenance": {str(k): int(v) for k, v in ev["prov"].value_counts().items()},
        "eval_defect": int(ev["defect"].sum()),
        "split_ids_eval": sorted(ev["image_id"]),     # 공유 확인용. JSON 에는 남기지 않는다
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

    # ----------------------------------------------------------------------------------
    # 공유 평가셋 — 파일럿 eval split 전량. **두 팔에 똑같이 들어간다** (74번 A-3).
    # 어블레이션이 바꾸는 것은 학습 풀뿐이다.
    # ----------------------------------------------------------------------------------
    shared_eval = m[m["split"] == "eval"]
    pool = m[m["split"] != "eval"]
    ev_prov = shared_eval["prov"].value_counts().to_dict()
    print(f"\n공유 평가셋 {len(shared_eval):,}장 (두 팔 동일) · 출처 {ev_prov}")
    print(f"  부차 채점 모집단으로 쓸 수 있는 N-crop 부분집합 "
          f"{int((shared_eval['prov'] == 'N-crop').sum()):,}장 — 두 팔에서 동일하다")
    print(f"학습 풀 후보 {len(pool):,}장 · 출처 {pool['prov'].value_counts().to_dict()}")

    # --- 팔 1: 크롭 한정. 학습 풀만 N-crop 으로 자른다 ---
    crop_pool = pool[pool["prov"] == "N-crop"]
    target = len(crop_pool)
    print(f"\n크롭 한정 학습 풀 {target:,}장")

    # --- 팔 2: 규모 대조. 같은 학습 풀 장수를 (split × 출처) 비례로 뽑는다 ---
    #
    # 층별로 반올림하면 합이 목표와 어긋난다(실측 +1장). 규모 등가가 이 팔의 존재 이유라
    # **최대잉여법**으로 배분해 합이 정확히 목표가 되게 한다 — 층별 몫의 정수부를 먼저 주고
    # 남는 자리를 소수부가 큰 층부터 채운다. 동률은 층 이름으로 깨서 결정적으로 만든다.
    frac = target / len(pool)
    strata = sorted(
        ((f"{sp}|{pv}", part) for (sp, pv), part in pool.groupby(["split", "prov"],
                                                                 observed=True)),
        key=lambda kv: kv[0],
    )
    exact = {k: len(p) * frac for k, p in strata}
    quota = {k: min(int(v), len(dict(strata)[k])) for k, v in exact.items()}
    residual = target - sum(quota.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - int(exact[k])), k))
    avail = {k: len(p) for k, p in strata}
    i = 0
    while residual > 0 and i < len(order) * 2:
        k = order[i % len(order)]
        if quota[k] < avail[k]:
            quota[k] += 1
            residual -= 1
        i += 1
    if residual:
        print(f"  !! 잔차 {residual}장을 배분하지 못했다 — 층 용량이 부족하다")

    picked: list[str] = []
    rows = []
    for k, part in strata:
        chosen = sorted(part["image_id"], key=lambda i: order_key(i, args.seed))[:quota[k]]
        picked.extend(chosen)
        sp, pv = k.split("|", 1)
        rows.append({"split": sp, "provenance": pv, "available": len(part),
                     "exact_quota": round(exact[k], 3), "quota": quota[k],
                     "picked": len(chosen)})
    control_pool = pool[pool["image_id"].isin(set(picked))]
    print(f"대조 학습 풀 비율 {frac:.5f} · 층 {len(rows)}개 · 선택 {len(control_pool):,}장 "
          f"(목표 {target:,})")
    if len(control_pool) != target:
        print(f"  !! 잔차 {len(control_pool) - target:+d}장 — 최대잉여법이 실패했다")
        return 1

    crop = pd.concat([crop_pool, shared_eval]).drop(columns=["prov"])
    control = pd.concat([control_pool, shared_eval]).drop(columns=["prov"])

    summary = [
        write_arm(args.outdir / f"{snap.snapshot_id}_crop_only", "crop_only",
                  crop, snap, lm,
                  "학습 풀 = 출처 N-crop 만 · eval = 파일럿 eval 전량(공유)"),
        write_arm(args.outdir / f"{snap.snapshot_id}_scale_control", "scale_control",
                  control, snap, lm,
                  "학습 풀 = (split × 출처) 층별 비례, sha256(image_id + seed) 순서 · "
                  "eval = 파일럿 eval 전량(공유)"),
    ]

    # --- 공유가 실제로 성립하는지 확인한다. 여기서 갈리면 A-3 이 되살아난다 ---
    ev_ids = [sorted(s["split_ids_eval"]) for s in summary]
    same_eval = ev_ids[0] == ev_ids[1]
    print(f"\n공유 평가셋 동일성: {'일치' if same_eval else '불일치'} "
          f"({len(ev_ids[0]):,}장 대 {len(ev_ids[1]):,}장)")
    if not same_eval:
        print("  !! 두 팔의 eval 이 다르다. A-3 이 해소되지 않았다")
        return 1
    same_pool = summary[0]["train_pool_n"] == summary[1]["train_pool_n"]
    print(f"학습 풀 규모 등가: {summary[0]['train_pool_n']:,} 대 "
          f"{summary[1]['train_pool_n']:,} → {'일치' if same_pool else '불일치'}")
    for s in summary:
        s.pop("split_ids_eval")

    out = args.outdir / "ablation_arms.json"
    out.write_text(json.dumps({
        "seed": args.seed, "derived_from": snap.snapshot_id,
        "shared_eval": {
            "n": len(shared_eval),
            "provenance": {str(k): int(v) for k, v in ev_prov.items()},
            "n_crop_subset": int((shared_eval["prov"] == "N-crop").sum()),
            "identical_across_arms": bool(same_eval),
            "note": ("74번 A-3. 두 팔은 이 평가셋으로만 채점한다. 팔별 평가셋을 다시 "
                     "만들면 자명하한이 갈려 델타가 오염된다."),
        },
        "train_pool_size_matched": bool(same_pool),
        "stratum_accounting": rows, "arms": summary},
        ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    print(f"\n기록: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""촬영 순서(`image_id`) 축이 라벨을 얼마나 결정하는지 직접 본다. 함정표 #12 의 일반화.

`recompute_baselines.py` 가 낸 "content-free 천장 0.897" 은 규칙 탐색의 산물이라 그 자체로는
버그와 구별되지 않는다. 여기서는 탐색 없이 **분할표만** 찍어서 같은 사실이 보이는지 본다.
탐색 결과와 분할표가 같은 이야기를 하지 않으면 탐색 쪽을 의심해야 한다.

    uv run python scripts/diagnose_id_axis.py --bins 64

보는 것 셋.

1. **구간별 클래스 순도** — id 구간 하나가 한 클래스로 얼마나 쏠려 있는가.
2. **train↔eval 구성 일치** — 구간 안에서 학습 쪽과 평가 쪽의 클래스 분포가 같은가.
   같으면 학습에서 외운 "구간→클래스"가 평가에 그대로 옮겨 붙는다. 이게 이월의 기제다.
3. **재질·출처와의 교락** — id 축이 사실은 재질 축인지, 독립인지.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from data.manifest_io import load_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO_ROOT / "data/interim/manifest_v1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("-o", "--out", type=Path,
                    default=REPO_ROOT / "data/interim/manifest_v1/id_axis_diagnosis.json")
    args = ap.parse_args()

    snap = load_snapshot(args.snapshot)
    m, t = snap.manifest, snap.tiles
    prov = dict(zip(t["image_id"], t["provenance"], strict=True))
    m = m.assign(prov=m["image_id"].map(prov))
    m["id_num"] = m["image_id"].str.rsplit(":", n=1).str[-1].astype("int64")

    # 이미지별 코드 집합 (매니페스트의 iso_codes 가 아니라 어노테이션에서 다시 만든다)
    ann = snap.annotations
    codes: dict[str, set[str]] = {i: set() for i in m["image_id"]}
    for r in ann.itertuples():
        codes[r.image_id].add(str(r.iso_code))
    classes = sorted({c for v in codes.values() for c in v})
    for c in classes:
        m[f"c_{c}"] = [1 if c in codes[i] else 0 for i in m["image_id"]]
    m["normal"] = [1 if not codes[i] else 0 for i in m["image_id"]]

    tv = m[m["split"] != "eval"]
    cuts = np.quantile(tv["id_num"].to_numpy(float),
                       np.linspace(0, 1, args.bins + 1)[1:-1])
    cuts = np.unique(cuts)
    m["bin"] = np.searchsorted(cuts, m["id_num"].to_numpy(float), side="right")

    ccols = [f"c_{c}" for c in classes] + ["normal"]
    print(f"동결본 {len(m):,}장 · 구간 {args.bins}개 (절단점 train+val 유도) · "
          f"클래스 {classes}")

    # --- 1. 구간별 순도 ---
    g = m.groupby("bin", observed=True)
    tab = g[ccols].mean()
    n = g.size()
    purity = tab.max(axis=1)
    dominant = tab.idxmax(axis=1)
    print(f"\n[1] 구간별 최빈 라벨 점유율 — 중앙값 {purity.median():.3f} · "
          f"평균 {purity.mean():.3f} · 최소 {purity.min():.3f}")
    for thr in (0.8, 0.9, 0.95, 0.99):
        share = float((n[purity >= thr].sum()) / len(m))
        print(f"    점유율 >= {thr:.2f} 인 구간의 이미지 비중 {share*100:5.1f}%")
    print("    (무작위 배치라면 점유율이 전역 최빈 비율 "
          f"{float(m[ccols].mean().max()):.3f} 근처에 머물러야 한다)")

    # --- 2. train↔eval 구성 일치 ---
    a = m[m["split"] != "eval"].groupby("bin", observed=True)[ccols].mean()
    b = m[m["split"] == "eval"].groupby("bin", observed=True)[ccols].mean()
    common = a.index.intersection(b.index)
    l1 = (a.loc[common] - b.loc[common]).abs().sum(axis=1)
    print(f"\n[2] 구간 안 train↔eval 클래스 분포 L1 거리 — 중앙값 {l1.median():.4f} · "
          f"평균 {l1.mean():.4f} · 최대 {l1.max():.4f}  (0 이면 완전 일치)")
    print(f"    공통 구간 {len(common)}/{m['bin'].nunique()}개. "
          "일치할수록 학습에서 외운 구간→클래스가 평가에 그대로 옮겨 붙는다")

    # --- 3. 재질·출처와의 교락 ---
    print("\n[3] 교락")
    al = m.groupby("bin", observed=True)["material"].apply(lambda s: (s == "AL").mean())
    print(f"    AL 비율이 0 또는 1 인 구간 {int(((al == 0) | (al == 1)).sum())}/{len(al)}개 "
          f"— id 축이 재질 축을 상당 부분 포함한다")
    cr = m.groupby("bin", observed=True)["prov"].apply(lambda s: (s == "N-crop").mean())
    print(f"    N-crop 비율이 0 또는 1 인 구간 {int(((cr == 0) | (cr == 1)).sum())}/{len(cr)}개")

    # 재질을 고정하고도 id 축이 남는가 — 이게 함정 #12 의 핵심 질문이다
    resid = []
    for mat, part in m.groupby("material", observed=True):
        pg = part.groupby("bin", observed=True)[ccols].mean()
        pn = part.groupby("bin", observed=True).size()
        if len(pg) < 2:
            continue
        resid.append((str(mat), len(pg), float(pg.max(axis=1).median()),
                      float(part[ccols].mean().max()), int(pn.sum())))
    print("    재질 고정 후에도 남는 구간 순도:")
    for mat, nb, med, base, tot in resid:
        print(f"      {mat}: 구간 {nb}개 · 구간 순도 중앙값 {med:.3f} 대 "
              f"재질 전역 최빈 {base:.3f} (n={tot:,})")

    top = sorted(zip(n.index, n.values, purity.values, dominant.values),
                 key=lambda r: -r[1])[:10]
    print("\n[참고] 큰 구간 10개")
    print(f"    {'구간':>4s} {'장수':>7s} {'최빈':>10s} {'점유율':>7s}")
    for bi, cnt, pu, dom in top:
        print(f"    {bi:4d} {cnt:7,d} {dom:>10s} {pu:7.3f}")

    out = {
        "snapshot_id": snap.snapshot_id, "bins": args.bins, "classes": classes,
        "purity_median": round(float(purity.median()), 6),
        "purity_mean": round(float(purity.mean()), 6),
        "purity_min": round(float(purity.min()), 6),
        "global_majority_rate": round(float(m[ccols].mean().max()), 6),
        "share_images_in_bins_over": {
            str(thr): round(float(n[purity >= thr].sum() / len(m)), 6)
            for thr in (0.8, 0.9, 0.95, 0.99)
        },
        "train_eval_l1_median": round(float(l1.median()), 6),
        "train_eval_l1_mean": round(float(l1.mean()), 6),
        "train_eval_l1_max": round(float(l1.max()), 6),
        "bins_pure_material": int(((al == 0) | (al == 1)).sum()),
        "bins_pure_provenance": int(((cr == 0) | (cr == 1)).sum()),
        "within_material": [
            {"material": mat, "bins": nb, "purity_median": round(med, 6),
             "material_global_majority": round(base, 6), "n": tot}
            for mat, nb, med, base, tot in resid
        ],
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8", newline="\n")
    print(f"\n기록: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

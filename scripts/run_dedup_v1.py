"""E3(pHash) 묶음 — tiles_v1 실데이터에서 t\\* 도출 → 묶음 재생성 → 최종 분할 재실행.

방법은 바꾸지 않는다. 256-bit(hash_size=16) + `threshold_cap = 96` 은 픽스처 실측에서
검증된 값 그대로다. **여기서 새로 정하는 것은 절대 임계 t\\* 하나뿐이고**, 스펙 6-4 절차를
tiles_v1 에 적용해 도출한다.

    uv run python scripts/run_dedup_v1.py --stage hash        # 전량 pHash (재개 가능)
    uv run python scripts/run_dedup_v1.py --stage measure     # 거리 분포 + t* 후보
    uv run python scripts/run_dedup_v1.py --stage apply --threshold N

**원본을 고치지 않는다.** tiles_v1 은 읽기만 한다.
**잠그지 않는다.** SNAPSHOT 잠금은 프로브 판정 뒤 총괄이 연다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from data.dedup.phash import (
    HASH_BITS,
    THRESHOLD_CAP,
    compute_phash,
    pack_hashes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
V1 = REPO_ROOT / "data/interim/manifest_v1"
CACHE = V1 / "phash_cache.jsonl"


def load_cache() -> dict[str, str]:
    out: dict[str, str] = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["image_id"]] = rec["phash_hex"]
    return out


def stage_hash(manifest) -> int:
    """전량 pHash. 중간에 죽어도 재개된다."""
    done = load_cache()
    todo = [(r.image_id, r.rel_path) for r in manifest.itertuples()
            if r.image_id not in done]
    print(f"pHash 대상 {len(todo):,}장 (완료 {len(done):,}장)")
    if not todo:
        return 0
    t0 = time.time()
    with CACHE.open("a", encoding="utf-8", newline="\n") as fh:
        for k, (image_id, rel_path) in enumerate(todo, 1):
            hx = compute_phash(REPO_ROOT / rel_path)
            fh.write(json.dumps({"image_id": image_id, "phash_hex": hx}) + "\n")
            if k % 200 == 0:
                fh.flush()
            if k % 2000 == 0:
                rate = k / max(time.time() - t0, 1e-9)
                print(f"  {k:,}/{len(todo):,}  {rate:.0f}장/초  "
                      f"남은 시간 약 {(len(todo)-k)/max(rate,1e-9)/60:.0f}분", flush=True)
    print(f"완료 {len(todo):,}장 · {(time.time()-t0)/60:.1f}분")
    return 0


def paired_histograms(packed: np.ndarray, group_codes: np.ndarray,
                      tile: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """(같은 묶음 안 쌍, 다른 묶음 쌍) 거리 히스토그램. 전수·무편향이다.

    같은 묶음 안 쌍은 **이미 E2 가 묶은 것**이라 같은 용접부라는 근거가 있다. 그 분포가
    "같은 용접부는 pHash 거리가 얼마나 벌어지는가"의 실측 기준선이 된다. 눈 확인 없이
    골짜기를 찍으면 근거가 없어지므로, 이 기준선을 분포 위에 겹쳐 놓고 고른다.
    """
    n = len(packed)
    same = np.zeros(HASH_BITS + 1, dtype=np.int64)
    diff = np.zeros(HASH_BITS + 1, dtype=np.int64)
    for a0 in range(0, n, tile):
        a1 = min(a0 + tile, n)
        for b0 in range(a0, n, tile):
            b1 = min(b0 + tile, n)
            xor = packed[a0:a1][:, None, :] ^ packed[b0:b1][None, :, :]
            dist = np.bitwise_count(xor).sum(axis=2).astype(np.int32)
            keep = np.ones_like(dist, dtype=bool)
            if a0 == b0:
                keep = np.triu(keep, k=1)
            eq = group_codes[a0:a1][:, None] == group_codes[b0:b1][None, :]
            same += np.bincount(dist[keep & eq].ravel(), minlength=HASH_BITS + 1)
            diff += np.bincount(dist[keep & ~eq].ravel(), minlength=HASH_BITS + 1)
    return same, diff


def quantiles(hist: np.ndarray, qs=(0.5, 0.9, 0.99, 0.999, 1.0)) -> dict[str, int]:
    total = hist.sum()
    if total == 0:
        return {}
    cum = np.cumsum(hist)
    return {f"p{q*100:g}": int(np.searchsorted(cum, q * total)) for q in qs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("hash", "measure", "apply", "split"),
                    required=True)
    ap.add_argument("--threshold", type=int, default=None)
    ap.add_argument("--tile", type=int, default=1024)
    args = ap.parse_args()

    from data.manifest_io import read_manifest

    manifest = read_manifest(V1 / "manifest.csv")
    print(f"매니페스트 v1 {len(manifest):,}행")

    if args.stage == "hash":
        return stage_hash(manifest)

    cache = load_cache()
    missing = [i for i in manifest["image_id"] if i not in cache]
    if missing:
        print(f"!! pHash 가 없는 이미지 {len(missing):,}장. --stage hash 를 먼저 끝내라.")
        return 3

    if args.stage == "measure":
        return stage_measure(manifest, cache, args.tile)
    if args.stage == "split":
        return stage_split(manifest)
    return stage_apply(manifest, cache, args.threshold)


def stage_measure(manifest, cache: dict[str, str], tile: int) -> int:
    """6-4 ①: 정확 히스토그램에서 골짜기를 찾는다. 재질별로 따로 본다."""
    report: dict[str, object] = {"hash_bits": HASH_BITS, "threshold_cap": THRESHOLD_CAP,
                                 "materials": {}}
    for material in sorted(manifest["material"].unique()):
        sub = manifest[manifest["material"] == material].sort_values("sha256")
        packed = pack_hashes([cache[i] for i in sub["image_id"]])
        codes = sub["group_id"].astype("category").cat.codes.to_numpy()
        print(f"\n[{material}] {len(sub):,}장 · 전수 쌍 {len(sub)*(len(sub)-1)//2:,}")
        t0 = time.time()
        same, diff = paired_histograms(packed, codes, tile=tile)
        print(f"  거리 계산 {time.time()-t0:.0f}초 · "
              f"같은묶음 쌍 {same.sum():,} · 다른묶음 쌍 {diff.sum():,}")
        print(f"  같은 묶음(= 같은 용접부) 거리 분위 {quantiles(same)}")
        print(f"  다른 묶음 거리 분위 {quantiles(diff)}")

        # 후보 임계별로 "다른 묶음인데 t 이하"인 쌍이 몇 개인지. E3 가 새로 붙일 엣지다.
        cum_diff = np.cumsum(diff)
        cum_same = np.cumsum(same)
        rows = []
        for t in range(8, THRESHOLD_CAP + 1, 4):
            rows.append({
                "t": t,
                "new_cross_group_pairs": int(cum_diff[t]),
                "same_group_pairs_covered_pct": round(
                    float(cum_same[t]) / max(int(same.sum()), 1) * 100, 2),
            })
        report["materials"][material] = {
            "n": len(sub),
            "same_group_quantiles": quantiles(same),
            "diff_group_quantiles": quantiles(diff),
            "same_hist": same.tolist(),
            "diff_hist": diff.tolist(),
            "threshold_table": rows,
        }
        print("   t   교차묶음 신규쌍   같은묶음 포함율")
        for r in rows:
            if r["t"] % 8 == 0:
                print(f"  {r['t']:3d}  {r['new_cross_group_pairs']:14,}  "
                      f"{r['same_group_pairs_covered_pct']:6.2f}%")

    out = V1 / "phash_distance_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"\n기록: {out}")
    return 0


def stage_apply(manifest, cache: dict[str, str], threshold: int | None) -> int:
    """E3 를 붙여 묶음을 다시 만들고 매니페스트를 갱신한다. 분할은 별도 단계다."""
    if threshold is None:
        print("!! --threshold 가 필요하다. measure 단계 결과로 t* 를 정하고 넘겨라.")
        return 3
    if threshold > THRESHOLD_CAP:
        print(f"!! t={threshold} 가 상한 {THRESHOLD_CAP} 을 넘는다. 총괄 에스컬레이션이다.")
        return 3

    from data.dedup.phash import build_groups

    sub = manifest.sort_values("image_id").reset_index(drop=True)
    before = dict(zip(sub["image_id"], sub["group_id"], strict=True))
    res = build_groups(
        image_ids=list(sub["image_id"]),
        sha256s=list(sub["sha256"]),
        phash_hex=[cache[i] for i in sub["image_id"]],
        materials=list(sub["material"]),
        threshold=threshold,
        meta_keys=list(sub["group_id"]),        # v0 의 연속 id 묶음이 E2 축이다
    )
    print(f"t*={threshold} · 묶음 {res.n_groups:,}개 · 최대 묶음 {res.max_group_size:,}장")
    print(f"  엣지 {res.edge_counts}")
    if res.cross_material_pairs:
        print(f"!! 교차 재질 근접쌍 {len(res.cross_material_pairs):,}건. 해소 필수 게이트다.")
        for a, b, d in res.cross_material_pairs[:10]:
            print(f"     {a} ↔ {b} d={d}")
        return 4

    sub["group_id"] = list(res.group_ids)
    sub["group_size"] = sub.groupby("group_id")["image_id"].transform("size").astype("Int64")
    sub["phash_hex"] = [cache[i] for i in sub["image_id"]]

    # v0 묶음이 몇 개나 합쳐졌는지. 라벨 문자열 변경이 아니라 **분할 단위의 변화**를 센다.
    old_to_new: dict[str, set[str]] = {}
    for old_g, new_g in zip(before.values(), res.group_ids, strict=True):
        old_to_new.setdefault(old_g, set()).add(new_g)
    fused = len({g for gs in old_to_new.values() for g in gs
                 if sum(1 for v in old_to_new.values() if g in v) > 1})
    print(f"  v0 묶음 {len(old_to_new):,}개 → v1 묶음 {res.n_groups:,}개 "
          f"(합쳐진 묶음 {len(old_to_new) - res.n_groups:,}개 감소)")
    print(f"  둘 이상의 v0 묶음을 삼킨 새 묶음 {fused:,}개")

    out = V1 / "manifest_e3.csv"
    from data.manifest_io import FLOAT_FORMAT
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        sub.to_csv(fh, index=False, na_rep="", float_format=FLOAT_FORMAT,
                   lineterminator="\n")
    print(f"기록: {out}  (분할은 아직 안 돌렸다)")
    return 0


def stage_split(previous) -> int:
    """E3 반영 묶음으로 최종 분할을 **동일 시드 1회** 재실행한다. 잠그지 않는다."""
    import yaml

    from data.invariants import check_invariants
    from data.label_map import load_label_map
    from data.manifest_io import FLOAT_FORMAT, read_annotations, read_manifest
    from data.split.pipeline import assign_splits, distribution_heatmap

    src = V1 / "manifest_e3.csv"
    if not src.exists():
        print("!! manifest_e3.csv 가 없다. --stage apply 를 먼저 돌려라.")
        return 3
    m = read_manifest(src)
    a = read_annotations(V1 / "annotations.csv")

    cfg = yaml.safe_load((REPO_ROOT / "configs/base.yaml").read_text(encoding="utf-8"))
    seed = int(cfg["split"]["seed"])
    lm = load_label_map()
    iso = {k: v.iso_code for k, v in lm.defect_types.items()}

    print(f"최종 분할 재실행 (동일 시드 {seed}, 1회)")
    m, meta = assign_splits(m, iso, seed)
    print(f"   split {m['split'].value_counts().to_dict()}")
    print(f"   client {m['client'].value_counts(dropna=True).to_dict()}")
    print(f"   dirichlet {meta.dirichlet}")

    # (2) 이전 분할과의 배정 일치율
    prev = previous.set_index("image_id")
    cur = m.set_index("image_id")
    both = prev.index.intersection(cur.index)
    same_split = prev.loc[both, "split"] == cur.loc[both, "split"]
    same_client = (prev.loc[both, "client"].fillna("-")
                   == cur.loc[both, "client"].fillna("-"))
    print(f"[일치율] 이미지 {len(both):,}장 기준")
    print(f"   split 동일 {same_split.mean()*100:.2f}% · "
          f"client 동일 {same_client.mean()*100:.2f}% · "
          f"둘 다 동일 {(same_split & same_client).mean()*100:.2f}%")

    # (3) C1 소수 클래스 장수 변화
    def minority(frame) -> dict[str, int]:
        c1 = frame[frame["client"] == "C1"]
        types = c1["defect_types"].fillna("")
        return {cls: int(types.str.contains(cls).sum())
                for cls in ("crack", "lack_of_fusion", "slag_inclusion", "porosity")}

    before_min, after_min = minority(previous), minority(m)
    print("[C1 소수 클래스]")
    for cls, before_n in before_min.items():
        print(f"   {cls:16s} {before_n:6,} → {after_min[cls]:6,} "
              f"({after_min[cls]-before_n:+,})")

    violations = check_invariants(m, a, lm, raw_root=REPO_ROOT)
    print(f"불변조건 위반 {len(violations)}건")
    for v in violations[:20]:
        print(f"   {v}")

    (V1 / "manifest_pre_e3.csv").write_bytes((V1 / "manifest.csv").read_bytes())
    out = m.sort_values("image_id", kind="stable").reset_index(drop=True)
    for col in out.columns:
        if str(out[col].dtype) == "boolean":
            out[col] = out[col].map({True: "True", False: "False"}).astype("string")
    with (V1 / "manifest.csv").open("w", encoding="utf-8", newline="\n") as fh:
        out.to_csv(fh, index=False, na_rep="", float_format=FLOAT_FORMAT,
                   lineterminator="\n")
    with (V1 / "distribution.csv").open("w", encoding="utf-8", newline="\n") as fh:
        distribution_heatmap(m).to_csv(fh, index=True, lineterminator="\n")
    with (V1 / "split_meta_e3.json").open("w", encoding="utf-8", newline="\n") as fh:
        json.dump({
            "seed": meta.seed, "dirichlet": meta.dirichlet,
            "groups": int(m["group_id"].nunique()),
            "agreement_split_pct": round(float(same_split.mean()) * 100, 3),
            "agreement_client_pct": round(float(same_client.mean()) * 100, 3),
            "c1_minority_before": before_min, "c1_minority_after": after_min,
            "invariant_violations": [str(v) for v in violations],
        }, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(f"기록: {V1}  ·  직전본은 manifest_pre_e3.csv 로 남겼다")
    print("SNAPSHOT 은 잠그지 않았다.")
    return 7 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())

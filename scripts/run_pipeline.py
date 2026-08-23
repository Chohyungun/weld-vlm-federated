"""파이프라인 전 구간 실행 — ingest → convert → dedup → split → 잠금.

순서가 계약이다(스펙 §6-9). 평가셋 선분리가 클라이언트 분할보다 먼저이고, 분할은 중복 묶음
단위다. 두 조건은 코드 구조와 불변식 IV7·IV8 이 이중으로 강제한다.

    uv run python scripts/run_pipeline.py --source riawelc --raw-root data/raw \
        --out data/processed/riawelc_260817
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dedup.phash import THRESHOLD_START, build_groups, compute_phash
from data.ingest.base import (
    ImageRecord,
    drop_exact_duplicates,
    records_to_frames,
    resolve_image_path,
)
from data.ingest.riawelc import RiawelcAdapter
from data.invariants import check_invariants
from data.label_map import load_label_map
from data.manifest_io import write_snapshot
from data.split.pipeline import assign_splits, distribution_heatmap

REPO_ROOT = Path(__file__).resolve().parents[1]
KST = timezone(timedelta(hours=9))
ADAPTERS = {"riawelc": RiawelcAdapter}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--raw-root", type=Path, default=REPO_ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rel-base", type=Path, default=None,
                    help="rel_path 기준 경로. 기본은 raw-root 의 조부모(저장소 루트 관례)")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--phash-threshold", type=int, default=THRESHOLD_START)
    ap.add_argument("--snapshot-id", default=None)
    args = ap.parse_args()

    lm = load_label_map()
    adapter = ADAPTERS[args.source]()
    iso_codes = {k: v.iso_code for k, v in lm.defect_types.items()}

    # --- ① ingest ------------------------------------------------------------------
    print(f"[1/5] ingest — {args.raw_root}/{args.source}")
    records: list[ImageRecord] = []
    rejected = 0
    rel_base = args.rel_base or args.raw_root.parent.parent
    for item in adapter.discover(args.raw_root, rel_base=rel_base):
        rec = adapter.parse(item, lm)
        if rec.reject_reason:
            rejected += 1
            continue
        records.append(rec)
    if not records:
        print("이미지가 0장이다. 원본 경로를 확인하라.")
        return 1
    n_files = len(records)
    records, duplicates = drop_exact_duplicates(records)
    if duplicates:
        print(f"      바이트 동일 복제본 {len(duplicates)}장 제외 "
              f"(파일 {n_files} → 고유 {len(records)})")
    caps = adapter.capabilities(records)
    print(f"      {len(records)}장 파싱, 기각 {rejected}건")
    print(f"      capabilities: localization={caps.localization} "
          f"verdict_mode={caps.verdict_mode.value} counts={caps.counts}")

    manifest, annotations = records_to_frames(records, lm)

    # --- ② dedup -------------------------------------------------------------------
    print(f"[2/5] dedup — 256bit pHash + 전수 popcount, 임계 {args.phash_threshold}")
    phashes = [compute_phash(resolve_image_path(p, rel_base)) for p in manifest["rel_path"]]
    manifest["phash_hex"] = phashes
    # 원본이 묶음 식별자를 주면 E2 엣지로 넣는다. RIAWELC 처럼 타일로 잘린 데이터셋에서는
    # pHash 가 비겹침 타일을 원리적으로 못 잡으므로 이 엣지가 실질 방어선이다 (스펙 §6-10).
    meta_keys = [r.group_key for r in records]
    n_meta = sum(1 for k in meta_keys if k)
    if n_meta:
        print(f"      E2 묶음 키 {n_meta}/{len(meta_keys)}건, 고유 {len({k for k in meta_keys if k})}개")
    groups = build_groups(
        manifest["image_id"].tolist(),
        manifest["sha256"].tolist(),
        phashes,
        manifest["material"].tolist(),
        threshold=args.phash_threshold,
        meta_keys=meta_keys if n_meta else None,
    )
    manifest["group_id"] = list(groups.group_ids)
    manifest["group_size"] = manifest.groupby("group_id")["image_id"].transform("size").astype(int)
    print(f"      묶음 {groups.n_groups}개, 최대 크기 {groups.max_group_size}, "
          f"엣지 {groups.edge_counts}")
    if groups.cross_material_pairs:
        print(f"      !! 교차 재질 근접쌍 {len(groups.cross_material_pairs)}건 — "
              "해소 필수 게이트 (잠금 전 처리)")
        return 1

    # --- ③ split -------------------------------------------------------------------
    print("[3/5] split — 평가셋 20% 선분리 → 클라이언트 → val 10%")
    manifest, meta = assign_splits(manifest, iso_codes, args.seed)
    print(f"      split {manifest['split'].value_counts().to_dict()}")
    print(f"      client {manifest['client'].value_counts(dropna=True).to_dict()}")
    print(f"      판정 표본 {int(manifest['eval_subset'].notna().sum())}장")
    if meta.dirichlet:
        print(f"      dirichlet {meta.dirichlet}")
    for line in meta.band_report:
        print(f"      [밴드 보고] {line}")

    # --- ④ 불변식 + 잠금 ------------------------------------------------------------
    print("[4/5] 불변식 IV1~IV12")
    violations = check_invariants(
        manifest, annotations, lm, raw_root=rel_base, hash_sample_frac=0.02
    )
    if violations:
        print("      위반 — 스냅샷을 쓰지 않는다:")
        for v in violations:
            print(f"        {v}")
        return 1
    print("      통과 (IV12 실파일 해시 표본 재계산 포함)")

    snapshot_id = args.snapshot_id or f"{args.source}_{datetime.now(KST):%y%m%d}"
    caps_doc = {
        "generated_at": datetime.now(KST).isoformat(),
        "snapshot_id": snapshot_id,
        "source": args.source,
        "is_mock": False,
        "counts": caps.counts,
        "capabilities": {
            "localization": caps.localization,
            "thickness_mm": caps.thickness_mm,
            "pixel_scale": caps.pixel_scale,
            "size_mm": caps.thickness_mm and caps.pixel_scale,
            "verdict_mode": caps.verdict_mode.value,
        },
        "assumptions": {
            "thickness_mm": None, "px_per_mm": None, "quality_level": None, "rationale": None,
        },
        "dedup": {
            "hash_size": 16,
            "threshold": groups.threshold,
            "n_groups": groups.n_groups,
            "max_group_size": groups.max_group_size,
            "edge_counts": groups.edge_counts,
            "threshold_status": "provisional_t0",   # §6-4 눈 확인 절차 미완 — 확정값 아님
        },
        "split_meta": {
            "seed": meta.seed,
            "eval_folds": meta.eval_folds,
            "val_folds": meta.val_folds,
            "dirichlet": meta.dirichlet,
            "stratum_order": {k: list(v) for k, v in meta.stratum_order.items()},
            "band_report": list(meta.band_report),
        },
    }
    digest = write_snapshot(args.out, manifest, annotations, caps_doc)
    print(f"      snapshot_digest {digest}")

    # --- ⑤ 분포 히트맵 --------------------------------------------------------------
    print("[5/5] 분포 히트맵 (원본 클래스 기준)")
    heat = distribution_heatmap(manifest)
    out_csv = args.out / "distribution_heatmap.csv"
    with out_csv.open("w", encoding="utf-8", newline="\n") as fh:
        heat.to_csv(fh, lineterminator="\n")
    print(heat.to_string())
    print(f"\n완주. {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

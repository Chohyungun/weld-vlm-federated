"""한 사이클 파일럿용 축소 표본 매니페스트 (52번 계획 ①).

분할이 동결된 스냅샷에서 **묶음 단위 층화**로 약 3,000장을 뽑는다. 분할 동결(8/28) 직후
바로 돌릴 수 있게 준비해 둔 것이다.

    uv run python scripts/make_pilot_subset.py \
        --snapshot data/processed/<동결 스냅샷> \
        --out data/processed/<동결 스냅샷>_pilot3000

지키는 것

- **분할을 다시 하지 않는다.** 동결된 `split`·`client`·`eval_subset` 을 그대로 상속한다.
  여기서 다시 나누면 파일럿이 검증하는 대상이 본실험 분할이 아니게 된다.
- **묶음 단위로 뽑는다.** 같은 묶음이 표본과 비표본으로 갈리는 것은 상관없지만, 뽑힌 묶음은
  통째로 들어온다. 묶음이 쪼개지면 축소 표본 안에서 누수가 생긴다.
- **층화 축은 (split × 재질 × 클래스)** 다. 평가셋 선분리가 유지되도록 split 을 층에 넣는다.
- **결정론.** `sha256(group_id + seed)` 순서로 뽑는다. 같은 스냅샷·같은 시드면 같은 표본이다.
- 원본 스냅샷을 수정하지 않는다. 새 디렉터리에 쓰고 따로 잠근다.
"""

from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_TARGET = 3000
DEFAULT_SEED = 20260828


def _order_key(group_id: str, seed: int) -> str:
    return hashlib.sha256(f"{group_id}|{seed}".encode()).hexdigest()


def select_groups(
    manifest: pd.DataFrame, target: int, seed: int
) -> tuple[set[str], pd.DataFrame]:
    """층별 비례 배분으로 묶음을 고른다. 반환값은 (선택된 묶음, 층별 회계)."""
    grp = (
        manifest.groupby("group_id")
        .agg(
            split=("split", "first"),
            material=("material", "first"),
            strata=("strata_key", "first"),
            size=("image_id", "size"),
        )
        .reset_index()
    )
    grp["stratum"] = grp["split"].astype(str) + "|" + grp["strata"].astype(str)
    grp["order"] = [_order_key(g, seed) for g in grp["group_id"]]

    total = int(grp["size"].sum())
    rows, picked = [], set()
    for stratum, part in grp.groupby("stratum"):
        part = part.sort_values("order", kind="stable")
        # 층의 이미지 비중만큼 목표를 나눈다. 층이 통째로 사라지지 않게 최소 1묶음 보장.
        quota = max(1, round(target * int(part["size"].sum()) / total))
        got, chosen = 0, []
        for row in part.itertuples():
            if got >= quota:
                break
            chosen.append(row.group_id)
            got += int(row.size)
        picked.update(chosen)
        rows.append({
            "stratum": stratum,
            "groups_available": len(part),
            "groups_picked": len(chosen),
            "images_available": int(part["size"].sum()),
            "quota": quota,
            "images_picked": got,
        })
    return picked, pd.DataFrame(rows).sort_values("stratum").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, required=True, help="동결된 스냅샷 디렉터리")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--raw-root", type=Path, default=None,
                    help="주면 IV12(파일 해시 표본 재계산)까지 돈다")
    args = ap.parse_args()

    snap = load_snapshot(args.snapshot)          # 해시 검증 포함
    m, a = snap.manifest, snap.annotations
    print(f"원본 스냅샷 {snap.snapshot_id}: 이미지 {len(m):,} / 묶음 {m['group_id'].nunique():,}")

    picked, accounting = select_groups(m, args.target, args.seed)
    sub_m = m.loc[m["group_id"].isin(picked)].copy()
    sub_a = a.loc[a["image_id"].isin(set(sub_m["image_id"]))].copy()
    # group_size 는 축소 표본 안에서 다시 센다. 묶음을 통째로 가져오므로 값이 바뀌지 않아야
    # 정상이지만, 바뀌면 묶음이 쪼개졌다는 뜻이라 검증기가 잡는다.
    sub_m["group_size"] = sub_m.groupby("group_id")["image_id"].transform("size").astype(int)

    print(f"축소 표본: 이미지 {len(sub_m):,} / 묶음 {sub_m['group_id'].nunique():,} "
          f"/ 결함 인스턴스 {len(sub_a):,}")
    print(f"  split {sub_m['split'].value_counts().to_dict()}")
    print(f"  client {sub_m['client'].value_counts(dropna=True).to_dict()}")

    # 묶음이 쪼개지지 않았는지 직접 확인한다 (축소가 만드는 유일한 새 위험이다)
    orig_sizes = m.loc[m["group_id"].isin(picked)].groupby("group_id").size()
    new_sizes = sub_m.groupby("group_id").size()
    split_groups = orig_sizes[orig_sizes != new_sizes]
    if len(split_groups):
        print(f"  !! 묶음이 쪼개졌다: {list(split_groups.index[:5])}")
        return 1
    print("  묶음 원자성 확인: 쪼개진 묶음 0")

    lm = load_label_map()
    violations = check_invariants(sub_m, sub_a, lm, raw_root=args.raw_root)
    if violations:
        print("  불변식 위반 — 축소 표본을 쓰지 않는다:")
        for v in violations:
            print(f"    {v}")
        return 1
    print("  불변식 IV1~IV12 통과")

    caps = dict(snap.capabilities)
    caps["snapshot_id"] = f"{snap.snapshot_id}_pilot{args.target}"
    caps["counts"] = dict(caps.get("counts", {}))
    caps["counts"]["images_total"] = len(sub_m)
    caps["pilot_subset"] = {
        "derived_from": snap.snapshot_id,
        "target": args.target,
        "seed": args.seed,
        "selection": "sha256(group_id + seed) 순서, (split × 재질 × 클래스) 층별 비례",
        "note": (
            "파일럿 전용이다. 표본 3,000·시드 1세트이므로 수치를 논문 결과로 쓰지 않는다. "
            "분할은 원본 스냅샷에서 상속하며 여기서 다시 나누지 않는다."
        ),
    }
    # tiles.csv 는 동결 때 승격된 선택적 멤버다(이 스크립트 작성 이후). 원본에 있으면
    # 부분집합을 상속한다 — P9 교차 출처 오탐이 표본 스냅샷만으로 돌 수 있어야 한다.
    sub_t = None
    if snap.tiles is not None:
        sub_t = snap.tiles.loc[snap.tiles["image_id"].isin(set(sub_m["image_id"]))].copy()
        assert len(sub_t) == len(sub_m), "tiles 부분집합이 매니페스트와 1:1 이 아니다"
    digest = write_snapshot(args.out, sub_m, sub_a, caps, tiles=sub_t)
    with (args.out / "stratum_accounting.csv").open("w", encoding="utf-8", newline="\n") as fh:
        accounting.to_csv(fh, index=False, lineterminator="\n")
    print(f"\n기록: {args.out}\n  snapshot_digest {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

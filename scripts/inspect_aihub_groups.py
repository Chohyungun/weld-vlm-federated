#!/usr/bin/env python
"""AI허브 창원 데이터셋 — 연속 촬영 묶음 분석기.

파일명만 읽는다 (JSON을 열지 않아 빠르다). 확인하는 것은 두 가지다.

1. `info.id` 가 연속인 이미지 묶음(run)이 **클래스 폴더를 가로지르는가**
2. 그 묶음이 **저자 제공 Training/Validation 분할을 가로지르는가**

2번이 참이면 저자 분할 자체에 누수가 있다는 뜻이고, 우리가 직접 분할할 때도
run 을 묶음 ID로 삼아야 한다 (프로젝트 불변조건 1-5).

사용:
    python scripts/inspect_aihub_groups.py <라벨_루트> -o <출력.txt>
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict

NAME_RE = re.compile(r"^(?P<prefix>[A-Z]+_[A-Z]+_\d+)_(?P<id>\d+)\.json$", re.IGNORECASE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    # id -> (prefix, split, archive)
    rec: dict[int, tuple[str, str, str]] = {}
    unparsed = []

    for dirpath, _, filenames in os.walk(args.root):
        arch = os.path.relpath(dirpath, args.root).split(os.sep)[0]
        # 아카이브 접두사 TL_/VL_ 이 저자 분할을 그대로 담고 있다
        split = "Training" if arch.startswith("TL_") else ("Validation" if arch.startswith("VL_") else "?")
        for fn in filenames:
            if not fn.lower().endswith(".json"):
                continue
            m = NAME_RE.match(fn)
            if not m:
                unparsed.append(fn)
                continue
            rec[int(m.group("id"))] = (m.group("prefix"), split, arch)

    ids = sorted(rec)
    L: list[str] = []
    P = L.append
    P(f"파일명 파싱: {len(ids):,} 성공 / {len(unparsed)} 실패")
    for u in unparsed[:10]:
        P(f"    미파싱: {u}")
    P("")

    # id 가 연속인 구간을 하나의 묶음(run)으로 본다
    runs: list[list[int]] = []
    cur = [ids[0]]
    for a, b in zip(ids, ids[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)

    len_ct = Counter(len(r) for r in runs)
    P("=" * 70)
    P("[A] 연속 id 묶음(run) 개요")
    P("=" * 70)
    P(f"묶음 수      : {len(runs):,}")
    P(f"최장 묶음    : {max(len(r) for r in runs):,} 장")
    P(f"평균 묶음    : {len(ids) / len(runs):.1f} 장")
    P(f"길이 1(고립) : {len_ct[1]:,} 묶음 ({len_ct[1] / len(runs) * 100:.1f}%)")
    P(f"길이>=2 묶음에 속한 이미지: {sum(len(r) for r in runs if len(r) >= 2):,} "
      f"({sum(len(r) for r in runs if len(r) >= 2) / len(ids) * 100:.1f}%)")
    P("")

    # 묶음이 클래스 폴더 / 저자 분할을 가로지르는지
    cross_prefix = 0
    cross_split = 0
    cross_split_examples = []
    cross_prefix_examples = []
    imgs_in_cross_split = 0

    for r in runs:
        prefixes = {rec[i][0] for i in r}
        splits = {rec[i][1] for i in r}
        if len(prefixes) > 1:
            cross_prefix += 1
            if len(cross_prefix_examples) < 5:
                cross_prefix_examples.append((r[0], r[-1], sorted(prefixes)))
        if len(splits) > 1:
            cross_split += 1
            imgs_in_cross_split += len(r)
            if len(cross_split_examples) < 5:
                cross_split_examples.append((r[0], r[-1], len(r), sorted(prefixes), sorted(splits)))

    P("=" * 70)
    P("[B] 묶음이 클래스 폴더를 가로지르는가")
    P("=" * 70)
    P(f"복수 클래스에 걸친 묶음: {cross_prefix:,} / {len(runs):,} ({cross_prefix / len(runs) * 100:.1f}%)")
    for a, b, pf in cross_prefix_examples:
        P(f"    id {a}..{b}  prefixes={pf}")
    P("")

    P("=" * 70)
    P("[C] 묶음이 저자 Training/Validation 분할을 가로지르는가")
    P("=" * 70)
    P(f"분할을 가로지르는 묶음: {cross_split:,} / {len(runs):,} ({cross_split / len(runs) * 100:.1f}%)")
    P(f"그 묶음에 속한 이미지  : {imgs_in_cross_split:,} ({imgs_in_cross_split / len(ids) * 100:.1f}%)")
    if cross_split:
        P("  -> 저자 제공 분할은 연속 촬영 묶음을 쪼갠다. 즉 저자 분할에 누수가 있다.")
    else:
        P("  -> 저자 분할은 묶음을 쪼개지 않는다.")
    for a, b, n, pf, sp in cross_split_examples:
        P(f"    id {a}..{b} ({n}장)  prefixes={pf}  splits={sp}")
    P("")

    P("=" * 70)
    P("[D] 촬영방식별 묶음 통계")
    P("=" * 70)
    by_mode: dict[str, list[int]] = defaultdict(list)
    for r in runs:
        mode = rec[r[0]][0].rsplit("_", 1)[0]  # RT_ST / RT_AL / VT_ST
        by_mode[mode].append(len(r))
    for k in sorted(by_mode):
        v = by_mode[k]
        P(f"    {k}: 묶음 {len(v):,}개, 이미지 {sum(v):,}장, 평균 {sum(v)/len(v):.1f}, 최장 {max(v):,}")
    P("")
    P("묶음 길이 분포 (상위 15):")
    for k, v in sorted(len_ct.items())[:15]:
        P(f"    {k:>4}장 : {v:>7,} 묶음")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
